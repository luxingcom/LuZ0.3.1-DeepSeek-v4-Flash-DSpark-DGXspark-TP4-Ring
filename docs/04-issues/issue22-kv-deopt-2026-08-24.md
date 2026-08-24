# issue22 nvfp4_ds_mla 长上下文 dispatch 实证 + P1-2 密钥日志脱敏

- **执行人**：雷克斯（Rex）· SRE 工程师（sre-engineer-2）
- **日期**：2026-08-24
- **任务**：P1-1 issue22 dispatch 缺陷实证（可能推翻"131K 物理极限"结论）；P1-2 密钥日志脱敏
- **纪律遵守**：未动生产容器（vllm-tp4-* 全程运行，head /health=200）；无 OOM（未起额外 GPU 任务）；key 不回传（本报告全部以 `<REDACTED>`/前缀 `<KEY_PREFIX_OLD>` 指代）

---

## 0. TL;DR（一页结论）

1. **issue22（architect 的 P1-1 主假设）大概率不成立**：取证证实本集群 DeepSeek V4 Flash 模型实际活跃后端是 `FLASHINFER_MLA_SPARSE_DSV4`（`DeepseekV4FlashInferSM120Attention` → `flashinfer_trtllm_batch_decode_sparse_mla_dsv4`），**不经过 `flashmla_sparse.py`**（该文件未被 `vllm/models/deepseek_v4/` import）。L861 的 `== "fp8_ds_mla"` 检查对 DSV4 是**死代码**；`nvfp4_ds_mla` 主缓存内容实为 FP8 e4m3（584B 信封），与 `fp8_ds_mla` 走同一 dtype 无关内核。
2. **"131K ~8 t/s 物理极限"结论大概率维持**：无 dispatch 慢路径可退 → 长上下文 decode 瓶颈仍指向 GB10 UMA 访存延迟（既有归因），issue22 补丁即使注入也预期无运行态效果。
3. **待 A/B 实证闭环**：取证为强证据但非运行态对比；A/B 工具/补丁/runbook 已全部就绪（本窗口内因"勿动生产"未执行——需 GPU 窗口：停生产→克隆 A/B→恢复生产）。
4. **P1-2 完成**：①脚本层 echo SERVE_CMD 掩码已落地四机（`.bak` + checker PASS）；②vLLM `non-default args` 明文 api_key 已确认，脱敏 overlay 已制作+容器内功能验证+四机部署就位（待下个重启窗口接线）；③当前 key 已暴露于 ≥6 处日志流，**建议轮换**（与生产重启窗口绑定）。

---

## 1. 执行 1：issue22 dispatch 实证

### 1.1 基线确认（issue22 bug 态存在）

生产容器（LuZ0.3.1, digest 85f2149f/d55d355a）实测：

```
/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/flashmla_sparse.py:861
    use_fp8_cache = self.kv_cache_dtype == "fp8_ds_mla"      # bug 态: nvfp4_ds_mla 判 False
```

生产启动参数：`--kv-cache-dtype nvfp4_ds_mla`、`--max-model-len 600000`、`--long-prefill-token-threshold 4096`、util 0.82（`/proc/1/cmdline` 实测）。**上游 issue22 形态与本集群 bug 态一致**（architect 对照正确）。

### 1.2 关键取证：活跃路径不是 flashmla_sparse.py

| # | 证据 | 出处（生产容器只读） |
|---|---|---|
| F1 | 启动日志：`[flashinfer_sparse_mla_warmup.py:124] Autotuning FlashInfer SM120 sparse MLA **DSv4** decode` | `docker logs vllm-tp4-rank0` |
| F2 | warmup/autotune 缓存命中 `sparse_mla_sm120_decode_dsv4 (runner=SparseMlaDecodeV3Runner)` | 同 F1 |
| F3 | `_select_dsv4_attn_cls`：SM12x 默认 → `DeepseekV4FlashInferSM120Attention`；通用 `FLASHINFER_MLA_SPARSE(_SM120)` 对 DSV4 显式 raise | `vllm/models/deepseek_v4/nvidia/model.py:760-790` |
| F4 | `DeepseekV4DecoderLayer.__init__` **直接实例化**注意力类（L808），不经 supports_combination/校验 | `nvidia/model.py:808` |
| F5 | `flashmla_sparse` 在 `vllm/models/deepseek_v4/` 全树 **0 处 import** | `grep -rn "flashmla_sparse" .../deepseek_v4/` = 空 |
| F6 | DSV4 decode forward 直接调 `flashinfer_trtllm_batch_decode_sparse_mla_dsv4(..., kv_layout="NHD")`，KV 按 uint8 不透明字节处理，**无 dtype 分派** | `nvidia/flashinfer_sparse.py:818` |
| F7 | `supports_combination` 在 `backends/` 无调用方（仅定义）→ nvfp4 不被"声明拒绝"阻断 | `grep -rn "supports_combination" .../backends/` |
| F8 | 主 MLA 缓存对 `fp8_ds_mla` 与 `nvfp4_ds_mla` 都返回 584B/token；nvfp4 主缓存数据实为 FP8 e4m3（head=512 禁 fp4 写入） | 既有 P-ISSUE22B F11-F14 + `sparse_mla.py` |

**综合结论**：生产 DSV4 模型经 flashinfer DSV4 内核 decode，**不存在**"nvfp4 → `_forward_bf16_kv` 慢路径"这一 dispatch 缺陷的运行态路径。L861 补丁（上游 Issue#22）在**我们 fork 的 DSV4 模型上无运行态效果**。此结论与 2026-08-13 P-ISSUE22B 取证一致，与 2026-08-24 architect 报告"最严重 P1"假设相左。

### 1.3 A/B 实证资产（已就绪，待 GPU 窗口执行）

- **判据**：B 臂 decode t/s 显著提升 → issue22 坐实、物理极限推翻；无显著差异 → 维持既有结论（dispatch 缺陷非主因）。
- **工具**（`deliverables/engineering-assurance/_luz031_official_bench/`）：
  - `bench_longctx_decode.py` — 长上下文 decode-only 测量（131K/256K/600K，自校准 prompt 长度，decode-only 口径 = `(ct-1)/(t_last-t_first)`，TTFT/prompt_tokens 附采）；
  - `patch_issue22_l861.py` — B1 臂单行 L861 注入/回滚（对齐上游 `in ("fp8_ds_mla","nvfp4_ds_mla")`，幂等）；
  - `patch_issue22_both.py` — B2 臂双门控（L271+L861）注入/回滚（若通用路径活跃时防 fp8_extra_metadata 缺失崩溃）；
  - `issue22_ab_runbook.sh` — 编排：预检→起克隆→校准→A臂→注入+重启→B臂→回滚+重启→A复核→停克隆→汇总（门禁 APPROVED=launch）。
- **预判**：依据 §1.2 取证，B 臂预期**无显著差异**（L861 为死代码）；若 B 臂意外显著提升，则需重新审视活跃后端选择逻辑。

> **为何本窗口未执行**：A/B 需停生产起克隆（单节点 1 GPU + 121GB UMA 已被生产占满 ~99GiB，无余量并发）。任务纪律"勿动生产容器、生产在跑勿动"，故未代停生产。**执行前置**：督导批准 + GPU 窗口（停 vllm-tp4-* → `APPROVED=launch bash issue22_ab_runbook.sh` → 恢复生产基线 + healthcheck timer）。

### 1.4 结论（issue22 / 物理极限）

- 取证层面：**issue22 dispatch 缺陷在本集群运行态不成立**（活跃 DSV4 路径无 dtype 分派）。
- 既有"131K decode ~8 t/s 为 GB10 物理极限"结论**暂维持**；A/B 为最终闭环（数据表在窗口执行后回填）。

---

## 2. 执行 2：P1-2 密钥日志脱敏

### 2.1 泄漏面确认

| 泄漏点 | 现状 | 证据 |
|---|---|---|
| ① 启动脚本 echo SERVE_CMD | head `start_tp4_head.sh:77` / worker `start_tp4_worker.sh:76` 明文打印含 `--api-key` 的完整命令 | 四机实测 |
| ② vLLM `docker logs` | `non-default args: {'api_key': ['<明文>'], ...}`（`api_utils.py:273 log_non_default_args`） | `docker logs vllm-tp4-rank0` |
| ③ 历史 bench 日志 | `/tmp/_bench_luz031/logs/clone_{head,rank1,rank2,rank3}.log` + `prod_head.log` 各含明文 key 1 处 | grep 前缀 `<KEY_PREFIX_OLD>` 命中 |

### 2.2 处置①：脚本层掩码（已落地）

- **改动**：`echo "[i] serve 命令: $SERVE_CMD"` → 先构造 `MASKED_SERVE_CMD`（`sed -E 's/(--api-key[= ]+)[^ ]+/\1********/g'`）再 echo；`SERVE_CMD` 本体不变（docker run 仍用完整含 key 命令）。
- **范围**：node01 `start_tp4_head.sh`；node01/04/03 `start_tp4_worker.sh`。
- **核验**：四机 `check_vllm_script.sh` ✅ 全部通过；`.bak-ma<API_KEY>` 留档；运行时验证 `--api-key ********`（无明文泄漏，SERVE_CMD 保留 key）。
- 补丁脚本：`deliverables/engineering-assurance/_p12_mask_20260824/mask_serve_cmd.py`（幂等，含 --check 演练）。

### 2.3 处置②：vLLM 自身日志（已确认 + 评估 + overlay 就位）

- **确认**：vLLM 启动日志 `non-default args` 明文打印 api_key。
- **评估**：环境级掩码推荐**只读 bind-mount overlay**（与现有 flashinfer_b12x_moe.py/tilelang.py overlay 同模式）：
  ```
  -v <INSTALL_DIR>/overlay-mask/api_utils.py:/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/serve/utils/api_utils.py:ro
  ```
  改 `log_non_default_args` 将 `api_key/hf_token/dspark_api_keys` 替换为 `<redacted>`（对齐上游 PR#89）。
- **已做**：masked `api_utils.py` 在容器内**编译 + 功能验证 PASS**（真 vLLM parser：masked 输出 `api_key: '<redacted>'`；原版对照确认泄漏）。overlay 文件已四机部署 `<INSTALL_DIR>/overlay-mask/api_utils.py`（md5 一致）。
- **待做**：在**下个重启窗口**往 head/worker 脚本 BINDS 加上述一行（叠加到 2.2 已改脚本，checker 复验）→ 生效于下次 vLLM 启动。

### 2.4 处置③：key 轮换评估与建议

- **现状**：当前 key（`<KEY_PREFIX_OLD>…`）为 2026-08-13 轮换引入，已在 **≥6 处日志流**明文暴露（2.1 表）→ 视为已泄露，**建议轮换**。
- **消费面**（四机实测）：
  - `<INSTALL_DIR>/secrets/vllm.env`（600 root:root）四机在位；systemd `vllm-tp4-head.service`/`vllm-tp4-worker.service`/`vllm-cluster.service` 均 `EnvironmentFile` 引用；
  - vLLM head 8001 是唯一 API 监听（`--api-key` 校验）；worker 容器 SERVE_CMD 也带 `--api-key`（同 env 注入）；
  - gateway `aicad-v18-server` 无 `VLLM_API_KEY` env（自带 MODEL_CONFIG_ENC_KEY 等），下游如何调用 8001 需在窗口前与 gateway 侧核对（模型配置存储/加密配置）。
- **轮换步骤**（与生产重启窗口绑定，预估 ~15-20 min）：
  1. 生成新 64-hex key；
  2. 四机更新 `<INSTALL_DIR>/secrets/vllm.env`（保持 600 root:root，md5 一致）；
  3. 若 gateway/客户端直连 8001 携带旧 key → 同步更新其配置；
  4. 重启 vLLM head + workers（沿用 08-13 顺序：worker 03→04→02→head，head-first 拉起）；
  5. 验证矩阵：新 key /health + /v1/models 200、**旧 key 401**、冒烟、四机 healthy、日志无 error；
  6. 追加核对：下次启动日志 `non-default args` 不再出现明文 key（若已接线 overlay 则直接满足）。
- **影响**：需短暂中断（重启窗口）；因 key 已泄露，风险等级 P1，建议**尽快排窗口**，可与 P1-1 A/B 窗口合并（同一次停/起）。

---

## 3. 资产索引

| 资产 | 路径 |
|---|---|
| P1-2 补丁脚本（幂等，含演练） | `deliverables/engineering-assurance/_p12_mask_20260824/mask_serve_cmd.py` |
| vLLM 脱敏 overlay（masked api_utils.py + 原始 + 容器测试） | `deliverables/engineering-assurance/_p12_mask_20260824/overlay-mask/` |
| 长上下文测量工具 | `deliverables/engineering-assurance/_luz031_official_bench/bench_longctx_decode.py` |
| issue22 注入/回滚（B1 单行） | `deliverables/engineering-assurance/_luz031_official_bench/patch_issue22_l861.py` |
| issue22 注入/回滚（B2 双门控） | `deliverables/engineering-assurance/_luz031_official_bench/patch_issue22_both.py` |
| A/B 编排 runbook（门禁） | `deliverables/engineering-assurance/_luz031_official_bench/issue22_ab_runbook.sh` |
| 四机已改启动脚本 `.bak-ma<API_KEY>` | `<INSTALL_DIR>/scripts/start_tp4_{head,worker}.sh.bak-ma<API_KEY>` |

## 4. 诚实声明

1. **A/B 未执行**：受"勿动生产"纪律 + 无 GPU 余量限制，本窗口未停生产起克隆；取证为只读证据，A/B 数据表待 GPU 窗口执行后回填（runbook 已就绪）。
2. **issue22 结论基于取证（强证据）而非 A/B（运行态对比）**：若 A/B 意外显示 B 臂显著提升，需重新审视后端选择/克隆环境差异。
3. **P1-2 脚本掩码已生效于脚本层**；vLLM 日志 overlay 已就位但**接线到下个重启窗口**；轮换需窗口执行。
4. 本报告全程未落明文 key；所有 SSH 操作只读 + 脚本层可控修改（生产容器未重启/未触碰）。

---

*本报告由工程保障团队 SRE（sre-engineer-2）生成；A/B 窗口与 key 轮换窗口需工程负责人/督导批准后执行。*
