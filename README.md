# FlexTensor MICRO Tutorial

## Introductions

Tensor computation plays a paramount role in a broad range
of domains, including machine learning, data analytics, and
scientific computing. The wide adoption of tensor computation
and its huge computation cost has led to high demand
for flexible, portable, and high-performance library implementation
on heterogeneous hardware accelerators such
as GPUs and FPGAs. However, the current tensor library
implementation mainly requires programmers to manually
design low-level implementation and optimize from the algorithm,
architecture, and compilation perspectives. Such
a manual development process often takes months or even
years, which falls far behind the rapid evolution of the application
algorithms.

We introduce FlexTensor, which is a schedule
exploration and optimization framework for tensor computation
on heterogeneous systems. FlexTensor can optimize
tensor computation programs without human interference,
allowing programmers to only work on high-level programming
abstraction without considering the hardware platform
details. FlexTensor systematically explores the optimization
design spaces that are composed of many different schedules
for different hardware. Then, FlexTensor combines different
exploration techniques, including heuristic method and 
machine learning method to find the optimized schedule
configuration. Finally, based on the results of exploration,
customized schedules are automatically generated for different
hardware. In the experiments, we test 12 different kinds
of tensor computations with totally hundreds of test cases
and FlexTensor achieves average 1.83x performance speedup
on NVIDIA V100 GPU compared to cuDNN; 1.72x performance
speedup on Intel Xeon CPU compared to MKL-DNN
for 2D convolution; 1.5x performance speedup on Xilinx
VU9P FPGA compared to OpenCL baselines; 2.21x speedup
on NVIDIA V100 GPU compared to the state-of-the-art.


## Installation

Requires: `Python 3.5+`, `Numpy`, `TVM <= v0.6`

1. Install our [modified TVM](https://github.com/KireinaHoro/tvm.git), follow the [instructions](https://docs.tvm.ai/install/from_source.html).
2. Clone this repo:
   ```sh
   git clone https://github.com/KnowingNothing/FlexTensor-Micro.git
   ```
3. Set the environments:
   `export AUTO_HOME=path/to/FlexTensor-Micro`
   `export PYTHONPATH=$AUTO_HOME:$PYTHONPATH`

To run the baselines, `PyTorch` is required.



## Tutorial Files

flextensor/tutorials/*