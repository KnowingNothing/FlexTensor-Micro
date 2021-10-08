# run flextensor (cascadelake)
python optimize_gemm.py --target "llvm -mcpu=cascadelake" --target_host "llvm -mcpu=cascadelake" --parallel 8 --timeout 20 --log gemm_config.log --dtype int32

# run flextensor (skylake)
python optimize_gemm.py --target "llvm -mcpu=skylake-avx512" --target_host "llvm -mcpu=skylake-avx512" --parallel 8 --timeout 20 --log gemm_config.log

# run test
python optimize_gemm.py --test gemm_optimize_log.txt

# run baseline
python gemm_baseline.py --type numpy --number 100

# run plot
python plot.py