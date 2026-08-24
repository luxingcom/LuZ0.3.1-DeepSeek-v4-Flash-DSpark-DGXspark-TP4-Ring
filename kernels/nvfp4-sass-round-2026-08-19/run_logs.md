# NVFP4 双算子 SASS 轮次 — 真机复测运行日志（2026-08-19, DGX Spark）

环境：node01 / vllm-tp4-rank0 容器 / torch 2.11.0+cu130 / triton 3.6.0 / CC=(12,1)=sm_121a
交付包：nvfp4-kernels-delivery(2).zip → 容器内 /vllm-workspace/nvfp4-delivery-v13/

> 注：verify 脚本中 time 列打印值含 ×1000 单位 bug（print 用 t*1e3），但 TFLOPS / GB/s 速率数值正确（do_bench 返回毫秒，速率公式以秒换算）。

## 1) kernel① v13 正确性 + 全算子性能（verify_v13.py）

KERNEL1 v13 CORRECTNESS  (vs torch ref, rtol/atol = 5e-2)
[PASS] M=  256 K= 4096 N= 4096 bias=False max_abs_err=0.0000
[PASS] M=  256 K= 4096 N= 4096 bias=True  max_abs_err=0.0000
[PASS] M=  512 K= 2048 N= 4096 bias=False max_abs_err=0.0000
[PASS] M=  512 K= 2048 N= 4096 bias=True  max_abs_err=0.0000
[PASS] M= 1024 K= 4096 N= 2048 bias=False max_abs_err=0.0000
[PASS] M= 1024 K= 4096 N= 2048 bias=True  max_abs_err=0.0000
[PASS] M=  128 K= 4096 N= 4096 bias=False max_abs_err=0.0000
[PASS] M=  128 K= 4096 N= 4096 bias=True  max_abs_err=0.0000
CORRECTNESS: 8/8 passed

KERNEL1 v13 PERF  (全算子：含 A 量化 kernel + 主机侧 W 重打包，W 缓存)
     M      K      N |   TFLOPS
   256   4096   4096 |      7.1
   512   4096   4096 |      7.8
  1024   4096   4096 |      8.0
   256   8192   8192 |     12.8
   512   8192   8192 |     13.7
  1024   8192   4096 |      8.2
   256   4096  16384 |     19.6

## 2) kernel① v13 GEMM-ONLY 性能（对齐 round6 口径：A 预量化、W 已缓存，仅测 _nvfp4_gemm_kernel）

KERNEL1 v13 GEMM-ONLY PERF  (A pre-quant, W cached; 对比 round6 基线 20.5~45.8)
     M      K      N |   TFLOPS
   256   4096   4096 |     29.6
   512   4096   4096 |     38.8
  1024   4096   4096 |     32.7
   256   8192   8192 |     38.1
   512   8192   8192 |     44.3
  1024   8192   4096 |     46.9
   256   4096  16384 |     37.6

## 3) kernel② v12.1 正确性 + 带宽（verify_v121.py）

KERNEL2 v12.1 CORRECTNESS  (byte-exact vs torch ref)
[PASS] T=    1 shape=(1, 584) dtype=torch.uint8
[PASS] T=    4 shape=(4, 584) dtype=torch.uint8
[PASS] T=   16 shape=(16, 584) dtype=torch.uint8
[PASS] T=   64 shape=(64, 584) dtype=torch.uint8
[PASS] T=  256 shape=(256, 584) dtype=torch.uint8
[PASS] T= 1024 shape=(1024, 584) dtype=torch.uint8
[PASS] T= 4096 shape=(4096, 584) dtype=torch.uint8
CORRECTNESS: 7/7 passed

KERNEL2 v12.1 PERF  (GB/s, bytes/token=4680)
     T |     GB/s
     1 |      0.4
     4 |      1.5
    16 |      5.2
    64 |     12.2
   256 |     16.6
  1024 |     18.3
  4096 |     18.7
 16384 |     18.8

## 4) kernel② paged v11 pytest（维持版本）

$ python -m pytest test_nvfp4_ds_mla_kv_linear_paged.py -q
.....                                                                    [100%]
5 passed in 217.26s (0:03:37)
