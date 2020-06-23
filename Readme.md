## profiling deep learning frameworks.

Examples on how to profile deep learning workloads with Linux tools.

### Pytorch

Pull a docker image with Pytorch

- Stacks Pytorch image

```bash
docker pull sysstacks/dlrs-pytorch-clearlinux
```

#### Run benchmark inside the container

Instantiate the stacks image, and mount the current directory to the image.

```bash
docker run -v`pwd`:/workspace/bmark/ -it sysstacks/dlrs-pytorch-clearlinux
```

1. First let's check if the platforms supports AVX512 instructions:

```bash
cd /workpsace/bmark/scripts
./check_platform.sh
```

If the platform has AVX-512 extensions, you will see an output like:

```bash
==============================================================================================
Tue 23 Jun 2020 03:41:42 AM UTC -- [Done]: Success, the platform supports AVX-512 instructions
==============================================================================================
```

2. Let's check if MKL-DNN accelerated kernels are being used while running Pytorch on the platform you are running:

```bash
MKLDNN_VERBOSE=1
cd /workspace/bmark/benchmarks
python cnn_benchmarks.py
```

If MKL-DNN kernels are bieng used, the output should look something like below, it shows the MKL-DNN version, 
Instructions available on the hardware and also the optimized kernels being called while running the code.

```python
mkldnn_verbose,info,Intel MKL-DNN v0.21.1 (commit 7d2fd500bc78936d1d648ca713b901012f470dbc)
mkldnn_verbose,info,Detected ISA is Intel AVX-512 with AVX512BW, AVX512VL, and AVX512DQ extensions
mkldnn_verbose,create,convolution,jit:avx512_common,forward_training,fsrc:nchw fwei:Ohwi16o fbia:x fdst:nChw16c,alg:convolution_direct,mb64_ic3oc64_ih224oh55kh11sh4dh0ph2_iw224ow55kw11sw4dw0pw2,0.24707
mkldnn_verbose,create,reorder,jit:uni,undef,in:f32_oihw out:f32_Ohwi16o,num:1,64x3x11x11,0.0249023
mkldnn_verbose,exec,reorder,jit:uni,undef,in:f32_oihw out:f32_Ohwi16o,num:1,64x3x11x11,0.432129
mkldnn_verbose,exec,convolution,jit:avx512_common,forward_training,fsrc:nchw fwei:Ohwi16o fbia:x fdst:nChw16c,alg:convolution_direct,mb64_ic3oc64_ih224oh55kh11sh4dh0ph2_iw224ow55kw11sw4dw0pw2,15.6321
mkldnn_verbose,create,reorder,jit:uni,undef,in:f32_nChw16c out:f32_nchw,num:1,64x64x55x55,0.0539551
mkldnn_verbose,exec,reorder,jit:uni,undef,in:f32_nChw16c out:f32_nchw,num:1,64x64x55x55,4.82983
mkldnn_verbose,create,convolution,jit:avx512_common,forward_training,fsrc:nChw16c fwei:OIhw16i16o fbia:x fdst:nChw16c,alg:convolution_direct,mb64_ic64oc192_ih27oh27kh5sh1dh0ph2_iw27ow27kw5sw1dw0pw2,0.419189
```

3. Now that we have figured out the platform supports AVX-512 vector instructions and MKL-DNN kernels are being called, let's run the code to benchmark with the process tuner script. The `tuner_shim.sh` script sets number of threads openMP can use and also set the block time of the threads. The script goes through three different combinations

- single thread
- single socket
- All cores

```bash
cd /workspace/bmark/scripts
./tune_shim.sh ../benchmark/cnn_benchmarks.py
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
CUDA time total: 0.000us
```
This uses Pytorch's [bottleneck](https://pytorch.org/docs/stable/bottleneck.html) profiler and gives us information on how much time it takes for each autograd flow, this can be used to figure out which layer is taking the most time and which layers we need to optimize.

4. Perf

Finally let's use the linuxtools perf profiler on the host to see the top processes while running the benchmark

```bash
perf top
```

An example output of running `perf top` is given below:

![image](images/perf.png)

As we can see the most amount of time is used by the omp library, and also we can see, avx512 mkl blas is being used as well.

This was a 101 on how to use basic linux tools and Pytorch inbuilt ones to profile a workload to identify potential bottlenecks.

