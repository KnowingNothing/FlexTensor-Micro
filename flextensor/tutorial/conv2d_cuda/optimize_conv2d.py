import os
import sys
import argparse
import time
import json
import tvm 
import numpy as np
from tvm import rpc
from flextensor.utils import Config, RpcInfo
from flextensor.task import Task, TASK_TABLE
from flextensor.scheduler import schedule, schedule_with_config
from flextensor.measure import _evaluate
from flextensor.utils import to_tuple
from flextensor.configs.conv2d_config import *


shape_dict = {
    "yolo": yolo_shapes,
    "google": google_shapes,
    "squeeze": squeeze_shapes,
    "res": res_shapes,
    "vgg-16": vgg_16_shapes,
    "vgg-19": vgg_19_shapes,
    "yolo_b8": yolo_shapes_b8,
    "mobile_v2": mobilev2_shapes,
}


LOCAL_RPC = False
LIB_DIR = "."


def evaluate(name, s, bufs, target, dev_id, number, rpc_info):
    if rpc_info is not None:
        host = rpc_info.host
        port = rpc_info.port
    else:
        # local
        host = "0.0.0.0"
        port = 9090     # default port
    if host == "0.0.0.0":
        if LOCAL_RPC:
            use_rpc = True
        else:
            use_rpc = False
    else:
        use_rpc = True
    if use_rpc:
        remote = rpc.connect(host, port)
        ctx = remote.context(target, dev_id)
    else:
        ctx = tvm.context(target, dev_id)
    tvm_arys = []
    for buf in bufs:
        shape = to_tuple(buf.shape)
        tmp = np.random.uniform(-10, 10, size=shape).astype(buf.dtype)
        tmp = tvm.nd.array(tmp, ctx)
        tvm_arys.append(tmp)
    try:
        func_file = "{}.tar".format(name)
        if rpc_info is not None and rpc_info.target_host is not None:
            func = tvm.build(s, bufs, target=target, target_host=rpc_info.target_host)
        else:
            func = tvm.build(s, bufs, target=target)
        if use_rpc:
            func.export_library(os.path.join(LIB_DIR, func_file))
            remote.upload(os.path.join(LIB_DIR, func_file))
            func = remote.load_module(func_file)
        evaluator = func.time_evaluator(func.entry_name, ctx, number=number)
        time_cost = evaluator(*tvm_arys).mean * 1e3
    except Exception as e:
        print(e)
        time_cost = float("inf")
    finally:
        while len(tvm_arys) > 0:
            del tvm_arys[-1]
        if os.path.exists(os.path.join(LIB_DIR, func_file)):
            try:
                os.remove(os.path.join(LIB_DIR, func_file))
            except Exception as e:
                print(e)
        elif os.path.exists(os.path.join(LIB_DIR, func_file + ".so")):
            try:
                os.remove(os.path.join(LIB_DIR, func_file))
            except Exception as e:
                print(e)
    return time_cost


def optimize(prefix, from_, shapes, target="llvm", dev_id=0, trials=100, timeout=4.0,
    parallel=1, method="searching", use_model=False, rpc_info=None, force_inline=False, logfile=sys.stdout):
    ret = dict()
    for i, shape in enumerate(shapes):
        print("Optimize {} convolution layer {} shape {}".format(prefix, i + 1 + from_, shape), flush=True)
        batch, in_channel, height, width, out_channel, _, k_h, k_w, _, stride, padding, dilation, groups = shape
        # create an empty task but has the correct key we want
        task = Task(
            "conv2d",
            prefix + str(i + from_), 
            None, 
            (batch, in_channel, height, width, out_channel, k_h, stride, padding, dilation, groups), 
            target, 
            dev_id
            )
        beg = time.time()
        s, bufs, configs = schedule(
            task.key, 
            op_trial=trials, 
            timeout=timeout, 
            op_stop=30, 
            parallel=parallel, 
            method=method,
            use_model=use_model,
            trials=[trials//10, trials],
            force_inline=force_inline,
            rpc_info=rpc_info,
            )
        end = time.time()
        # print(tvm.lower(s, bufs, simple_mode=True))
        print("######################################")
        print("op schedules:")
        for config in configs.op_config_lst:
            print("----------------------------------")
            for name, value in config.items():
                if value:
                    print(name, value)
        print("graph schedules:")
        for name, value in configs.graph_config.items():
            if value:
                print(name, value)
        ret[task.key] = configs
        string = json.dumps(configs)
        line = task.key + ":" + string
        print(line, file=logfile, flush=True)
        s, bufs = schedule_with_config(task.key, configs)
        time_cost = evaluate(task.key, s, bufs, target, task.dev_id, 10, rpc_info)
        print("Use", time_cost, "ms")
        print("Cost", end - beg, "s")
        print()
    return ret


def test(task_key, configs, dev_id=None, rpc_info=None, dump_ir=False, dump_code=False):
    task = TASK_TABLE[task_key]
    s, bufs = schedule_with_config(task_key, configs)
    if dump_ir:
        print(tvm.lower(s, bufs, simple_mode=True))
    if dump_code:
        func = tvm.build(s, bufs, "cuda")
        print(func.imported_modules[0].get_source())
    dev_id = dev_id if dev_id is not None else task.dev_id
    time_cost = evaluate(task_key, s, bufs, task.target, dev_id, 100, rpc_info)
    # print(task_key, "use", time_cost, "ms")
    # print()
    return time_cost


# @tvm.register_func
# def tvm_callback_cuda_compile(code):
#     """use nvcc to generate ptx code for better optimization"""
#     ptx = tvm.contrib.nvcc.compile_cuda(code, target="ptx", arch="sm_53")
#     return ptx


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--shapes", help="Use which shapes [yolo, google, res, squeeze, vgg-16, vgg-19]", type=str, default="")
    parser.add_argument("-f", "--from_", help="From which shape", type=int, default=0)
    parser.add_argument("-t", "--to", help="To which shape", type=int, default=-1)
    parser.add_argument("-l", "--log", help="Log file name", type=str, default="")
    parser.add_argument("--test", help="test file name", type=str, default="")
    parser.add_argument("--trials", help="number of trials for op", type=int, default=100)
    parser.add_argument("--target", help="target device type", type=str, default="llvm")
    parser.add_argument("--device", help="target device number", type=int, default=0)
    parser.add_argument("--timeout", help="timeout", type=float, default=4.0)
    parser.add_argument("--parallel", help="parallel", type=int, default=1)
    parser.add_argument("--use_model", help="use performance model", action="store_true")
    parser.add_argument("--method", help="how to schedule", type=str, default="searching")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--target_host", type=str, default="llvm")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--force_inline", action="store_true")
    parser.add_argument("--use_rpc", action="store_true")
    parser.add_argument("--dump_ir", action="store_true")
    parser.add_argument("--dump_code", action="store_true")
    # parser.add_argument("--op_hint", type=str, default="split_fuse")
    args = parser.parse_args()
    if args.use_rpc:
        rpc_info = RpcInfo(args.host, args.port, target_host=args.target_host)
    else:
        rpc_info = None
    if args.shapes != "":
        shapes = shape_dict[args.shapes]
        if args.to < 0:
            end = len(shapes)
        else:
            end = args.to
        
        if args.log != "":
            with open(args.log, "a") as flog:
                ret = optimize(
                    args.shapes, args.from_, 
                    shapes[args.from_:end], 
                    target=args.target, 
                    dev_id=args.device, 
                    timeout=args.timeout, 
                    trials=args.trials, 
                    parallel=args.parallel,
                    method=args.method,
                    use_model=args.use_model,
                    rpc_info=rpc_info,
                    force_inline=args.force_inline,
                    logfile=flog,
                    )
        else:
            ret = optimize(
                args.shapes, 
                args.from_, 
                shapes[args.from_:end], 
                target=args.target, 
                dev_id=args.device, 
                timeout=args.timeout, 
                trials=args.trials, 
                parallel=args.parallel,
                method=args.method,
                use_model=args.use_model,
                rpc_info=rpc_info,
                force_inline=args.force_inline,
                logfile=sys.stdout,
                )
    if args.test != "":
        print("Case,execution time(ms)")
        with open(args.test, "r") as fin:
            for i, line in enumerate(fin):
                name, string = line.split(":", 1)
                obj = json.loads(string)
                configs = Config(obj[0], obj[1])
                time_cost = test(name, configs, dev_id=args.device, rpc_info=rpc_info, dump_ir=args.dump_ir, dump_code=args.dump_code)
                print(f"{int(i+1)},{time_cost}")