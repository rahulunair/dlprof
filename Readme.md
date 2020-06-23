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

Run the code to benchmark  with the tuner, that sets threading paramaters and attempts different values.

```bash
cd /workspace/bmark/scripts
./tune_shim.sh ../benchmark/cnn_benchmarks.py1
```

First let's check if the ISA supports AVX512 instructions:

```bash
lscpu | grep avx512
```

```python
Flags:  fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi
mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs
bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx
est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer
aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb cat_l3 cdp_l3 invpcid_single pti
ssbd mba ibrs ibpb stibp tpr_shadow vnmi flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 hle avx2
smep bmi2 erms invpcid rtm cqm mpx rdt_a avx512f avx512dq rdseed adx smap clflushopt clwb intel_pt
avx512cd avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc
cqm_mbm_local dtherm ida arat pln pts hwp hwp_act_window hwp_epp hwp_pkg_req md_clear flush_l1d
```

Let's check if MKL-DNN accelerated kernels are being used while running Pytorch on the platform you are running:

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

Output example

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

## Perf

Using perf profiler, let's see the top processes while running the benchmark

```bash
perf top
```
![image](images/perf.png)


