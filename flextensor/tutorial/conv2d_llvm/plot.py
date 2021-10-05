import subprocess
from matplotlib import pyplot as plt
import numpy as np

flex_res = subprocess.Popen(
    "python optimize_conv2d.py --test resnet_optimize_log.txt", shell=True, stdout=subprocess.PIPE)
flex_perf = flex_res.stdout.read().decode()
flex_res.stdout.close()

lines = flex_perf.split("\n")[1:]
flex_data = []
for line in lines:
    if len(line):
        case, cost = line.split(",")
        flex_data.append(float(cost))
        print(line)
        
base_res = subprocess.Popen(
    "python conv2d_baseline.py --type tvm_generic --target llvm --shapes res --number 100", shell=True, stdout=subprocess.PIPE)
base_perf = base_res.stdout.read().decode()
base_res.stdout.close()

lines = base_perf.split("\n")[1:]
base_data = []
for line in lines:
    if len(line) and "," in line:
        case, cost = line.split(",")
        base_data.append(float(cost))
        print(line)

assert len(flex_data) == len(base_data)
        
base_bars = [1.0 for i in range(len(base_data))]
flex_bars = [base_data[i] / flex_data[i] for i in range(len(base_data))]

acc = 1
count = 0
for v in flex_bars:
    acc = acc * v
    count += 1
print("Geomean:", np.power(acc, 1/count))

names = ["" + str(i) for i in range(len(base_data))]
x = [i * 1 for i in range(len(base_data))]

plt.bar(x, base_bars, width=0.4, label="autotvm", fc="g")
x = [i + 0.4 for i in x]
plt.bar(x, flex_bars, width=0.4, label="flextensor", tick_label=names, fc="r")
plt.legend()
plt.show()
plt.savefig("resnet_llvm_performance_compare_tmp.png")