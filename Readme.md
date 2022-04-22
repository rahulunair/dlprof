## Profiling deep learning frameworks

Examples on how to profile deep learning workloads with Linux tools. In this scenario we are checking if Intel vector instructions are available on the platform you are running the workload, if oneDNN, MKL is being used and also we use the Pytorch bottlneck profiler to profile a workload.

### Pytorch

<!-- Pull a docker image with Pytorch

- Stacks Pytorch image

```bash
docker pull sysstacks/dlrs-pytorch-clearlinux
```

#### Run benchmark inside the container

Instantiate the stacks image, and mount the current directory to the image.

```bash
docker run -v`pwd`:/workspace/bmark/ -it sysstacks/dlrs-pytorch-clearlinux
```
-->

### Install pytorch and utilities

```bash
conda create --name pytorch-intel python==3.9
conda activate pytorch-intel
conda install -c intel pytorch
conda install -c intel torchvision
conda install -c intel intel-extension-for-pytorch
git clone https://github.com/rahulunair/dlprof
cd dlprof
```

1. First let's check if the platforms supports AVX512:

```bash
cd scripts
chmod +x check_platform.sh
./check_platform.sh
```

If the platform has AVX-512 extensions, you will see an output like:

```bash
=====================================================================================================
Wed 23 Jun 2021 03:41:42 AM UTC -- [Done]: Success, the platform supports AVX-512 (fp32) instructions
=====================================================================================================
=====================================================================================================================
Wed 23 Jun 2021 03:41:42 AM UTC -- [Error] : Intel® AVX-512 VNNI (int8) extensions are not available :: (avx512_vnni)
=====================================================================================================================
====================================================================================================================
Wed 23 Jun 2021 03:41:42 AM UTC -- [Error] : Intel® AMX (bf16, int8) extensions are not available :: (amx_tile)
====================================================================================================================
cd ..
```

2. Let's check if [OneDNN](https://oneapi-src.github.io/oneDNN/dev_guide_verbose.html) accelerated kernels are being used while running Pytorch on the platform you are running:

```bash
export ONEDNN_VERBOSE=2   # enable verbose mode for oneDNN, same as MKLDNN_VERBOSE=2 
cd benchmarks
python cnn_benchmarks.py
```

If OneDNN kernels are bieng used, the output should look something like below, it shows the oneDNN version, instructions available on the hardware and also the optimized kernels being called while running the code.

```python
dnnl_verbose,info,oneDNN v1.7.0 (commit 7aed236906b1f7a05c0917e5257a1af05e9ff683)
dnnl_verbose,info,cpu,runtime:OpenMP
dnnl_verbose,info,cpu,isa:Intel AVX-512 with AVX512BW, AVX512VL, and AVX512DQ extensions
dnnl_verbose,info,gpu,runtime:none
dnnl_verbose,create:cache_miss,cpu,reorder,jit:uni,undef,src_f32::blocked:abcd:f0 dst_f32::blocked:Acdb16a:f0,,,64x3x7x7,0.214111
dnnl_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:abcd:f0 dst_f32::blocked:Acdb16a:f0,,,64x3x7x7,0.1521
dnnl_verbose,create:cache_miss,cpu,convolution,jit:avx512_common,forward_training,src_f32::blocked:abcd:f0 wei_f32::blocked:Acdb16a:f0 bia_undef::undef::f0 dst_f32::blocked:aBcd16b:f0,scratchpad_mode:user;,alg:convolution_direct,mb32_ic3oc64_ih224oh112kh7sh2dh0ph3_iw224ow112kw7sw2dw0pw3,0.25
dnnl_verbose,exec,cpu,convolution,jit:avx512_common,forward_training,src_f32::blocked:abcd:f0 wei_f32::blocked:Acdb16a:f0 bia_undef::undef::f0 dst_f32::blocked:aBcd16b:f0,scratchpad_mode:user;,alg:convolution_direct,mb32_ic3oc64_ih224oh112kh7sh2dh0ph3_iw224ow112kw7sw2dw0pw3,28.7791
dnnl_verbose,create:cache_miss,cpu,reorder,jit:blk,undef,src_f32::blocked:aBcd16b:f0 dst_f32::blocked:abcd:f0,,,32x64x112x112,0.0529785
dnnl_verbose,exec,cpu,reorder,jit:blk,undef,src_f32::blocked:aBcd16b:f0 dst_f32::blocked:abcd:f0,,,32x64x112x112,10.345
dnnl_verbose,create:cache_miss,cpu,reorder,jit:blk,undef,src_f32::blocked:abcd:f0 dst_f32::blocked:aBcd16b:f0,,,32x64x56x56,0.0471191
dnnl_verbose,exec,cpu,reorder,jit:blk,undef,src_f32::blocked:abcd:f0 dst_f32::blocked:aBcd16b:f0,,,32x64x56x56,2.69409
dnnl_verbose,create:cache_miss,cpu,reorder,jit:uni,undef,src_f32::blocked:abcd:f0 dst_f32::blocked:ABcd16b16a:f0,,,64x64x3x3,0.0710449
dnnl_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:abcd:f0 dst_f32::blocked:ABcd16b16a:f0,,,64x64x3x3,0.0290527
dnnl_verbose,create:cache_miss,cpu,convolution,jit:avx512_common,forward_training,src_f32::blocked:aBcd16b:f0 wei_f32::blocked:ABcd16b16a:f0
```

3. Now that we have figured out the platform supports vector instructions(AVX512) and oneDNN kernels are being called, let's run the code to benchmark with the process tuner script. The `./scripts/tune_shim.sh` script sets number of threads openMP can use and also set the block time of the threads. The script goes through three different combinations

- single thread
- single socket
- All cores

```bash
cd /workspace/bmark/
./scripts/tune_shim.sh ./benchmark/cnn_benchmarks.py
```

A sample output while running a workload (`cnn_benchmarks.py`) with the `tune_shim.sh` script is shown below:

```bash
-------------------------------  ---------------  ---------------  ---------------  ---------------  --------------- 
Name                             Self CPU total %  Self CPU total   CPU total %      CPU total        CPU time avg   
-------------------------------  ---------------  ---------------  ---------------  ---------------  --------------- 
uniform_                         16.09%           176.567ms        16.09%           176.567ms        176.567ms       
uniform_                         6.44%            70.649ms         6.44%            70.649ms         70.649ms        
MkldnnConvolutionBackward        6.14%            67.374ms         6.14%            67.374ms         67.374ms        
MkldnnConvolutionBackward        5.99%            65.736ms         5.99%            65.736ms         65.736ms        
mkldnn_convolution_backward      5.99%            65.718ms         5.99%            65.718ms         65.718ms        
MkldnnConvolutionBackward        5.98%            65.589ms         5.98%            65.589ms         65.589ms        
mkldnn_convolution_backward      5.97%            65.571ms         5.97%            65.571ms         65.571ms        
MkldnnConvolutionBackward        5.93%            65.135ms         5.93%            65.135ms         65.135ms        
mkldnn_convolution_backward      5.93%            65.118ms         5.93%            65.118ms         65.118ms        
MkldnnConvolutionBackward        5.93%            65.083ms         5.93%            65.083ms         65.083ms        
MkldnnConvolutionBackward        5.93%            65.068ms         5.93%            65.068ms         65.068ms        
mkldnn_convolution_backward      5.93%            65.067ms         5.93%            65.067ms         65.067ms        
mkldnn_convolution_backward      5.93%            65.050ms         5.93%            65.050ms         65.050ms        
MkldnnConvolutionBackward        5.92%            64.977ms         5.92%            64.977ms         64.977ms        
mkldnn_convolution_backward      5.92%            64.957ms         5.92%            64.957ms         64.957ms        
-------------------------------  ---------------  ---------------  ---------------  ---------------  --------------- 
Self CPU time total: 1.098s
```
This uses Pytorch's [bottleneck](https://pytorch.org/docs/stable/bottleneck.html) profiler and gives us information on how much time it takes for each autograd flow, this can be used to figure out which layer is taking the most time and which layers we need to optimize.

4. [Perf](http://www.brendangregg.com/perf.html)

Finally let's use the linuxtools perf profiler on the host to see the top processes while running the benchmark

```bash
perf top
```

An example output of running `perf top` is given below:

![image](images/perf.png)

As we can see the most amount of time is spent on the omp library, and also we can see, avx512 mkl blas is being used as well.

This was a 101 on how to use basic linux tools and Pytorch inbuilt ones to profile a workload to identify potential bottlenecks.

## Further Analysis

perf record can be used to measure the CPU stack traces and perf report can be used the view the file generated by perf record.
