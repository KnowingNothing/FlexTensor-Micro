import tvm


def intrinsic_gemv_uint8_int8_compute(M, K):
    int32_lanes = M  # 16 int32 lanes in AVX512
    num_int8_elements = K  # 4 int8 elements in int32
    data = tvm.placeholder((num_int8_elements,), dtype="uint8", name="data")
    kernel = tvm.placeholder(
        (int32_lanes, num_int8_elements), dtype="int8", name="kernel")
    k = tvm.reduce_axis((0, num_int8_elements), name="k")
    C = tvm.compute(
        (int32_lanes,),
        lambda i: tvm.sum(data[k].astype("int32") *
                          kernel[i, k].astype("int32"), axis=k),
        name="C",
    )
    return C, [data, kernel, C]


def intrinsic_gemv_uint8_int8(M, K):
    _, (data, kernel, C) = intrinsic_gemv_uint8_int8_compute(M, K)

    a_buffer = tvm.decl_buffer(data.shape, dtype='uint8', name="a_buffer",
                               offset_factor=1,
                               strides=[1])
    b_buffer = tvm.decl_buffer(kernel.shape, dtype='int8', name="b_buffer",
                               offset_factor=1,
                               strides=[tvm.var('ldw'), 1])

    def _intrin_func(ins, outs):
        def _instr(index):
            ib = tvm.ir_builder.create()
            if index == 1:
                ib.emit(outs[0].vstore(0, tvm.const(0, "int32x16")))
                return ib.get()

            a_int8 = ins[0].vload([0], "uint8x4")
            re_int32 = tvm.call_intrin(
                "int32", "reinterpret", a_int8)
            vec_ai32 = re_int32.astype("int32x16")
            vec_a = tvm.call_intrin(
                "int8x64", "reinterpret", vec_ai32)
            vec_b = ins[1].vload([0, 0], "int8x64")
            vec_one = tvm.const(1, "int16x32")
            pair_reduction = tvm.call_llvm_intrin(
                "int16x32",
                "llvm.x86.avx512.pmaddubs.w.512",
                tvm.const(0, "uint32"),
                vec_a,
                vec_b,
            )
            quad_reduction = tvm.call_llvm_intrin(
                "int32x16",
                "llvm.x86.avx512.pmaddw.d.512",
                tvm.const(0, "uint32"),
                pair_reduction,
                vec_one,
            )
            if index == 0:
                ib.emit(outs[0].vstore([0], quad_reduction))
            else:
                ib.emit(outs[0].vstore([0], quad_reduction +
                        outs[0].vload([0], "int32x16")))
            return ib.get()

        # body, reset, update
        return _instr(0), _instr(1), _instr(2)

    with tvm.build_config(offset_factor=1, partition_const_loop=True):
        return tvm.decl_tensor_intrin(C.op, _intrin_func, binds={data: a_buffer, kernel: b_buffer})
    
    
def gemv_uint8_int8(i, j, k, dtype="int32"):
    a = tvm.placeholder((i, k, 16, 4), name='a', dtype="int8")
    b = tvm.placeholder((k, 4), name='b', dtype="uint8")
    kk = tvm.reduce_axis((0, k), name='k')
    kki = tvm.reduce_axis((0, 4), name="ki")
    c = tvm.compute((i, 16), lambda ii, iii:
                    tvm.sum(b[kk, kki].astype(dtype) * a[ii, kk, iii, kki].astype(dtype), axis=[kk, kki]), name='c')
    return [c.op], [b, a, c]


def schedule_gemv():
    outs, ins = gemv_uint8_int8(32, 32, 32)
    (b, a, c) = ins
    
    def tile_axes(s, op, axis, factors):
        ret = []
        for f in reversed(factors[:-1]):
            axis, inner = s[op].split(axis, factor=f)
            ret.append(inner)
        ret.append(axis)
        return list(reversed(ret))
    
    sch = tvm.create_schedule(c.op)
    i, ii = sch[c].op.axis
    rk, rki = sch[c].op.reduce_axis
    i0, i1 = tile_axes(sch, c, i, [16, 2])
    rk0, rk1 = tile_axes(sch, c, rk, [16, 2])
    sch[c].reorder(i0, rk0, i1, rk1, ii, rki)
    sch[c].parallel(i0)
    
    intrin = intrinsic_gemv_uint8_int8(16, 4)
    sch[c].tensorize(ii, intrin)
    print(tvm.lower(sch, [b, a, c], simple_mode=True))
    
    func = tvm.build(sch, [b, a, c], target="llvm -mcpu=skylake-avx512", target_host="llvm")
    
    
if __name__ == "__main__":
    schedule_gemv()