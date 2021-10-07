# run flextensor
python optimize_conv2d.py --shapes res --target llvm --parallel 8 --timeout 20 --log resnet_config.log

# run test
python optimize_conv2d.py --test resnet_optimize_log.txt

# run baseline
python conv2d_baseline.py --type tvm_generic --shapes res --number 100

# run plot
python plot.py