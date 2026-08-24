# FP8 质量门工具链（_fp8_qg_toolchain）

零 GPU 设计 + 脚本落地；噪声底实测/参考集采集排入 GPU 窗口。设计文档见
`../fp8-quality-gate-toolchain-2026-08-24.md`。

## 组成

| 文件 | 作用 | GPU 需要 |
|---|---|---|
| `reference_set.json` | 参考集 31 条（Tier A 4 / B 24 / C 3） | 否 |
| `run_common.py` | 共享 API/logprobs/KL/PPL/漂移 | 否 |
| `reference_set_builder.py` | validate / build（64K 长上下文）/ stats | 否 |
| `reference_set_collector.py` | 在线采集（greedy / dist / temp） | 是 |
| `quality_gate_noise_floor.py` | BF16 噪声底分析 + 门限推荐 | 否 |
| `kl_gate.py` | KL 门 + 困惑度门（含重标定） | 否 |
| `greedy_baseline.py` | greedy 4/4 golden 资产固化 + 比对 | capture 是 / compare 否 |
| `temp_top_p_gate.py` | 温度/top-p 抽验 | collect 是 / analyze 否 |
| `selftest.py` | 零 GPU 逻辑自检 | 否 |
| `_generated/` | build 产出的 64K prompt | 否 |

## 快速开始

```bash
# 0) 零 GPU 自检（无需 API key）
python3 selftest.py

# 1) GPU 窗口：BF16 噪声底采集（背靠背两遍）
VLLM_API_KEY=... python3 reference_set_collector.py collect \
    --quant bf16 --config dist --tag run1 --out runs/run_bf16_dist_run1.json
VLLM_API_KEY=... python3 reference_set_collector.py collect \
    --quant bf16 --config dist --tag run2 --out runs/run_bf16_dist_run2.json

# 2) 零 GPU：噪声底 + 门限
python3 quality_gate_noise_floor.py analyze runs/run_bf16_dist_run1.json \
    runs/run_bf16_dist_run2.json --out assets/noise_floor.json

# 3) GPU 窗口：FP8 采集（FP8 镜像就位后）
VLLM_API_KEY=... python3 reference_set_collector.py collect \
    --quant fp8 --config dist --tag cand --out runs/run_fp8_dist.json

# 4) 零 GPU：KL + 困惑度门判定
python3 kl_gate.py gate --baseline runs/run_bf16_dist_run1.json \
    --candidate runs/run_fp8_dist.json --noise-floor assets/noise_floor.json

# 5) greedy 基线（import 既有 quality_gate 快照免采集）
python3 greedy_baseline.py import-snapshot \
    --from <INSTALL_DIR>/backup/quality-gate/reference-latest.json
python3 greedy_baseline.py compare --candidate <fp8 run 或 reference json>

# 6) 温度/top-p 抽验
VLLM_API_KEY=... python3 temp_top_p_gate.py collect --quant bf16 --temperature 0.6 --top-p 1.0 --reps 5
VLLM_API_KEY=... python3 temp_top_p_gate.py collect --quant fp8  --temperature 0.6 --top-p 1.0 --reps 5
python3 temp_top_p_gate.py analyze --baseline runs/run_bf16_t0.6_p1.0.json \
    --candidate runs/run_fp8_t0.6_p1.0.json
```

退出码约定：0=PASS，1=FAIL，2=用法/环境错误（与 quality_gate.py 一致）。
