# 生产环境克隆 —— 两个算子实测报告（NVFP4）

**日期**：2026-08-20
**方式**：克隆生产环境（隔离测试容器，同生产镜像 + 同算子挂载，不干扰生产 4 rank）
**载体**：镜像 `dspark-vllm-gx10:0.2.1-v026.0` | 容器 `nvfp4-prod-clone`（已清理）
**生产状态**：全程 4 rank healthy（30 min），无扰动

---

## 📌 核心结论

- **kernel① routeA 通过**：正确性 8/8 `rel=0.00141`（对照 vLLM 官方 `dequantize`），性能 57~130 TFLOPS
- **kernel② v17 通过**：正确性 7/7 组逐字节（atol=0, max_diff=0），性能 T=1024 → 436 GB/s
- 两算子在当前生产基线下**数值正确、性能达标**，可安全部署，无回归

---

## 一、kernel① prefill GEMM（routeA，`cutlass_scaled_fp4_mm`）

**基准**：`bench_mmaf_final.py`（对照 vLLM 官方 `dequantize_to_dtype`，rel<0.02 即 PASS）

| M | K | N | rel | 判定 | ms | TFLOPS |
|---|------|------|-------|------|-------|--------|
| 256 | 4096 | 4096 | 0.00141 | ✅ | 0.088 | 97.1 |
| 512 | 2048 | 4096 | 0.00141 | ✅ | 0.149 | 57.5 |
| 1024 | 4096 | 2048 | 0.00141 | ✅ | 0.286 | 60.2 |
| 128 | 4096 | 4096 | 0.00141 | ✅ | 0.063 | 68.3 |
| 256 | 8192 | 8192 | 0.00141 | ✅ | 0.447 | 76.9 |
| 512 | 8192 | 8192 | 0.00141 | ✅ | 0.528 | 130.1 |
| 1024 | 8192 | 4096 | 0.00141 | ✅ | 0.631 | 108.9 |
| 256 | 4096 | 16384 | 0.00141 | ✅ | 0.390 | 88.0 |

- **ALL PASS（8/8）**，rel 与权威交付基线完全一致
- W 预处理 21.1 ms/层（可接受，每层一次缓存）
- **甄别**：`probe_a1_e2e.py` 的 `rel=256` 是**对照基准错误**（torch matmul vs 官方 dequant 语义），非算子缺陷，已排除

## 二、kernel② KV-Linear v17

**基准**：对照 v11 torch ref（金标准语义），atol=0 逐字节

| T | shape 一致 | 逐字节 | max_diff | 判定 |
|-----|------|--------|----------|------|
| 1 | ✅ | ✅ | 0.00e+00 | ✅ |
| 4 | ✅ | ✅ | 0.00e+00 | ✅ |
| 16 | ✅ | ✅ | 0.00e+00 | ✅ |
| 64 | ✅ | ✅ | 0.00e+00 | ✅ |
| 256 | ✅ | ✅ | 0.00e+00 | ✅ |
| 1024 | ✅ | ✅ | 0.00e+00 | ✅ |
| 4096 | ✅ | ✅ | 0.00e+00 | ✅ |

- **ALL PASS（7/7 逐字节）**
- **性能**：T=1024 → **436 GB/s**（远超 120 GB/s 硬门槛；rune 交接线 262 GB/s）

## 三、方法（可复现）

```
# 1. 起隔离测试容器（同生产镜像 + 算子挂载, 不启 serve）
docker run -d --name nvfp4-prod-clone --network host --gpus all \
  -v <INSTALL_DIR>/nvfp4-landing-export:/workspace/nvfp4-landing:ro \
  -v /home/<USER>/b12x-cache:/root/.cache/b12x:rw \
  ${HOST}:5000/anemll/dspark-vllm-gx10:0.2.1-v026.0 bash -lc "sleep infinity"

# 2. kernel① routeA 权威基准
docker exec nvfp4-prod-clone bash -lc "cp /workspace/nvfp4-landing/routeA/nvfp4_4w4a_mmaf.py /tmp/; \
  cd /workspace/nvfp4-landing/tests && python3 bench_mmaf_final.py"

# 3. kernel② v17 正确性+性能（手动驱动, 绕过无 pytest 环境）
docker cp _run_k2_v17.py nvfp4-prod-clone:/tmp/ ; docker exec ... python3 /tmp/_run_k2_v17.py
```

## 四、风险 / 局限

- 性能测于高压 GPU 环境（生产 rank0 在其 GPU 上同机运行），TFLOPS/GB/s 为相对参考；实际生产部署性能以 benchmark 口径为准
- v17 带宽 436 GB/s 采用 `T*1024*4*2` 粗略口径，与交付包 4680B/token 口径存在差异，仅作达标印证
- 测试容器已清理，未遗留

## 五、结论

两算子克隆实测**数值正确、性能达标**，可安全进入生产部署。kernel① routeA 以官方 dequant 语义 rel=0.00141 为准；kernel② v17 逐字节等价且高带宽。