# v15 轮执行日志（2026-08-19 16:30–17:10 GMT+8）

## 输入
- `nvfp4-kernels-delivery(3).zip` → 解压 `nvfp4-kernels-delivery-3/`
- 上传：scp → node01:/tmp/nvfp4-delivery-v15 → docker cp → `vllm-tp4-rank0:/vllm-workspace/nvfp4-delivery-v15/`

## 关键事实核查
- shipped test/benchmark 仍 import **v11 回退版**（`_triton.py`），非 v15 → 自建 harness 指向 v15 模块
- 生产转换器格式：`W_packed [K, N//2]`（N 打包，lo=第1元素）；v15 期望 `[K//2, N]`（K 打包）→ **布局不兼容（D1）**

## kernel① v15
1. `verify_v15_k1.py`：repack 适配后 **8/8 PASS，max_abs_err=0.000000**；as-shipped 直喂复现崩溃
   `RuntimeError: shape '[4096, 2048]' is invalid for input of size 16777216`
2. `gemm_only_bench_v15.py`：GEMM-only **26.7~81.4 TFLOPS**（最佳 1024,4096,4096=81.4）；全算子 7.2~14.9
3. `sass_v15.py`（清缓存+nvdisasm 直出）：**HMMA.16816.F32.BF16 ×64；TCGEN05=0；e2m1=0；FFMA=0**

## CUTLASS mmaf_sm120
- 工具链：nvcc 13.0 ✓ / cmake 4.4.2（pip 装）✓ / 无 git
- CUTLASS 获取：GitHub 直连限速（~1MB/10s，容器+本地同）→ 放弃 3.9 源码下载；捆绑 4.3.4=裁剪子集（缺 `arch/mma_sm120.h`、无 CMake config）；PyPI cutlass 0.9.0=Python 绑定无 C++ 头
- torch C++ 头：`torch/all.h` 不存在 → torch extension 编译断（B1）
- 静态验证：捆绑 4.3.4 头实证 blockscaled `Arguments` 必填 `layout_SFA/SFB`（B3，对应 .cu TODO）
- 结论：骨架不可构建；建议 PR #42209 复用

## kernel② v15
1. `verify_v15_k2.py`：**7/7 逐字节 PASS**；带宽 T=65536 → **1.3 GB/s(584B 口径)**
2. `probe_k2_bw.py`（.fn 固定配置）：全配置 10.0~10.6 GB/s(4680 口径) → 设计问题非配置问题；**v11=53.4 GB/s 对照**；**v14 编译即挂**（`_GROUP` 非 constexpr）
3. paged v11 全量重跑：**5/5 PASS**（首跑 1 FAIL 为并发清 Triton 缓存的文件竞争假阳性）

## 环境清洁
- 全程仅 rank0 容器内跑测试 kernel；4 rank healthy（rank0@01/rank1@02/rank3@03/rank2@04）、GPU 0%、无残留进程、未恢复生产

## 已落盘
- 容器：`/vllm-workspace/nvfp4-delivery-v15/evidence/`（SASS 转储+摘要）
- 本地：`deliverables/engineering-assurance/nvfp4-v15-round-2026-08-19/`（报告+日志+脚本+证据+zip）
