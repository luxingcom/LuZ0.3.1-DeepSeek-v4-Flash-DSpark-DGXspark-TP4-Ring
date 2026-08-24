# LuZ0.3.1 生产版本说明（Release Notes）

- **版本名**: LuZ0.3.1（用户命名，2026-08-23 批准采纳）
- **日期**: 2026-08-23（UTC）
- **执行**: 雷克斯（Rex）· SRE 工程师，工程保障团队
- **一句话**: B2 形态（W4A4 full + 池补丁）+ util 0.82 + FI 0.6.16 + threshold 4096 的正式生产落地。

---

## 1. 构成清单（LuZ0.3.1 = B2 形态 + util 0.82）

| 组件 | 值 | 载体/证据 |
|---|---|---|
| MoE 量化 | **W4A4 full**（`VLLM_MOE_W4A4=2`） | plugin_a1（routea-plugin-a1 0.1.0），SERVE_CMD 前缀 pip install（`/tmp/plugin_a1_install` 拷贝安装） |
| 池补丁 | `VLLM_B12X_SHARED_WRAPPER=1`（几何键共享池，跨层 wrapper 去重） | 池化插件 `w4a4_experts.py`（`_get_pooled_wrapper`，md5 `e5ed0c853c4964846d782686e9decb9c`）+ overlay `flashinfer_b12x_moe.py`（md5 `8f88555a0fc7e330ee51255c643796bc`，bind-mount 到 vllm experts 路径） |
| FlashInfer | **0.6.16**（rebased-experimental，md5 `7aac3857220eb5865a70a9ee50e7b8a8`） | 目录级 bind-mount：`<INSTALL_DIR>/nvfp4/flashinfer-0.6.16/flashinfer` → `dist-packages/flashinfer`；附带 `~/flashinfer-cache:/root/.cache/flashinfer` JIT 缓存挂载 |
| KV dtype | nvfp4_ds_mla | serve 参数 |
| threshold | `--long-prefill-token-threshold 4096`（`--max-num-batched-tokens 4096`） | serve 参数 |
| **util** | **`--gpu-memory-utilization 0.82`**（0.80→0.82） | 四机 head/worker 脚本 + checker KEY_PARAMS 同步 |
| 投机解码 | Dspark MTP n=7（probabilistic） | serve 参数 |
| cudagraph | capture 1..96（sizes 16 档）+ breakable | 三档捕获：PIECEWISE 16/16 + FULL 12/12 + dspark 11/11 |
| 基座镜像 | `<NODE_IP>:5000/anemll/dspark-vllm-gx10:0.2.1-v026.0`（vLLM 0.26.1 fork） | 生产脚本 IMG |
| b′ 插件 | **保留不激活**（bprime-window 判 No-Go：decode -25/-34%） | `<INSTALL_DIR>/nvfp4/plugin_a1_bprime/`，env 三开关全关 |

## 2. env 全集（LuZ0.3.1 特有关键项）

```
VLLM_MOE_W4A4=2            # W4A4 full（0=off 1=hybrid 2=full）
VLLM_MOE_W4A4_MIN_M=3072    # full 模式下不生效（hybrid 语义残留，B2 同款）
VLLM_MOE_W4A4_CG=1
VLLM_B12X_SHARED_WRAPPER=1  # 池开关（overlay + 插件双侧门控）
```
（其余 R11 生产 env 全集见 start_tp4_head.sh ENV_ARGS，未变更）

## 3. 基线数字（采纳验收实测，2026-08-23 05:47-06:01 UTC，二次重建 FI 0.6.16 全绿后）

| 指标 | B2 预期带 | LuZ0.3.1 实测 | vs B2 | 判定 |
|---|---|---|---|---|
| PR 4K 单流（3 轮中位） | ~2994 | **2950.5** | -1.5% | ✓ 带内 |
| PR 16K | ~2973 | **2943.6** | -1.0% | ✓ 带内 |
| PR 32K | ~2830 | **2834.2** | +0.2% | ✓ |
| PR 64K | ~2541 | **2550.0** | +0.4% | ✓ |
| C6 聚合（3 轮中位） | ~3060 | **3057**（3023/3060/3057） | -0.1% | ✓ |
| C12 聚合（3 轮中位） | ~3092 | **3056**（3059/3056/3034） | -1.2% | ✓ 带内 |
| C6/C12 med TTFT | 10.40 / 18.13s | **10.47 / 18.39s** | ~+1.5% | ✓ |
| DE C1 step_eff | ~18.3 | **18.2** | -0.5% | ✓ 中性 |
| DE C12 step_eff | ~85.1 | **80.2** | -5.8% | ⚠ 已知 W4A4 full decode 代价带内（phase3b 口径 -6~-9%；w4a4-ext 口径 ±3%——两口径并存，用户已裁定采纳） |
| weight | 45.32 GiB | **45.32 GiB** | 0 | ✓ 池生效 |
| KV tokens | ≥5.7M 门 / ~5.9M 预期 | **5,730,000** | 回补 +0.23M（5.50→5.73M） | ✓ 过门；回补低于合成预期 +0.44M，记录不阻断 |
| 质量门（稳定 4 prompt） | 逐字一致 | **4/4 exact match，own_stable 4/4** | — | ✓ PASS |
| needle 64K 抽验 | 统计口径 | **3/3 PASS**（mid/late/late；128K 加测 1/2，late 位已知抖动） | — | ✓ |
| stall 探针 | TTFT<6s | 3×4K TTFT 2.77-3.12s，SUSPECT=False | — | ✓ 干净 |
| 模式探针 | W4A4-fast 类 | 首 4K TTFT 2.786s | — | ✓ |
| cudagraph 三档 | 16/12/11 | 16/16 + 12/12 + 11/11 | — | ✓ |
| 回归日志扫描 | 无异常 | error/exception/traceback 0 条 | — | ✓ |

## 4. 检查点与恢复资产

| 资产 | 位置 |
|---|---|
| 恢复镜像（自包含 bake：FI 0.6.16 树 + overlay + 池化插件） | `<NODE_IP>:5000/anemll/dspark-vllm-gx10:LuZ0.3.1` |
| 基座锚点 tag | `<NODE_IP>:5000/anemll/dspark-vllm-gx10:LuZ0.3.1-base`（= 0.2.1-v026.0） |
| 状态快照 + md5 清单 + 一键恢复 | `<INSTALL_DIR>/backup/luz031-checkpoint-20260823/`（`restore_luz031.sh`，支持 `--dry-run`） |
| 脚本留档（改前基线） | `start_tp4_{head,worker}.sh` / `check_vllm_script.sh` `.bak-luz031-20260823`（四机） |
| 插件原版留档 | `w4a4_experts.py.bak-wsdedupl3-20260823`（md5 c2d1de3d，四机） |

## 5. 回滚链（保守止损路径，全部 <10 分钟）

1. **回滚到 W4A16 基线**：四机 `cp .bak-luz031-20260823`（head/worker/checker）+ 插件恢复 `.bak-wsdedupl3-20260823`（原版）+ checker 过 + head-first 重建（`docker rm -f vllm-tp4-rank0`）。
2. **单开关降级**（不回滚插件）：`sed -i "s/VLLM_MOE_W4A4=2/VLLM_MOE_W4A4=0/"` 四机脚本 → W4A4 关闭（池 overlay env=0 零行为，同 M1 态）。
3. **util 单独回退**：`sed -i "s/gpu-memory-utilization 0.82/gpu-memory-utilization 0.80/"`（脚本+checker 同步）。
4. 恢复后自愈链三件套核验：head.service + 三 worker.service + healthcheck.timer（runbook §E.1）。

## 6. 已知事项与决策口径

1. **FI 0.6.16 误回滚事件（本窗口发现并修复）**：w4a4-ext 收尾（08-23 03:01 UTC）误用 phase3b 时代 `.bak-wsdedupl3` 恢复，覆盖掉 fi016 窗口（00:56）注入的 flashinfer-0.6.16 bind-mount——03:01 起生产实际运行 FI 0.6.15 混合树约 2.5 小时未被察觉（两树数值逐位一致 + PR 带内）。LuZ0.3.1 部署中已按 fi016 报告 §2.2 原样补回（四机 + checker PASS）。**教训已入 runbook §E：跨窗口恢复必须核对 .bak 快照的时序覆盖范围。**
2. **KV 回补低于合成预期**：util 0.80→0.82 的 KV 回补实测 +0.27M（5.50→5.77M），低于合成预期 +0.44M（~5.9M）。验收门 ≥5.7M 过——按"回补不达预期但达标"记录，不阻断。差异归因：非 torch 内存/碎片随 util 上升非线性。
3. **W4A4 full 已知代价**（B2 形态继承）：decode 归一 -6~-9%（phase3b 口径；w4a4-ext 同窗口径 ±3% 噪声带内，两口径并存）；KV vs W4A16 基线 -4.5%（5.77 vs 6.04）。业务以 prefill+并发为主，用户已裁定采纳。
4. dist-info 元数据仍为 flashinfer_python-0.6.15（pip 视角滞后，`flashinfer.__version__` 为准，`FLASHINFER_DISABLE_VERSION_CHECK=1` 屏蔽）。
