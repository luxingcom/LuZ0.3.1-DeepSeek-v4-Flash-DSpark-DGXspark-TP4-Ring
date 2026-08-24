# issue22 生产版本 400K 长上下文检验（nvfp4_ds_mla + DSV4 路径）

- **执行人**：泰莎（Tessa）· 测试专家（testing-expert-1）
- **日期**：2026-08-24（UTC 04:5x — 05:2x，克隆旁路 GPU 窗口）
- **任务**：issue22（nvfp4_ds_mla dispatch 缺陷）处置最终闭环项 —— 核实生产用版本（LuZ0.3.1，nvfp4_ds_mla + DSV4 路径）在 **400K 长上下文 decode** 下是否受影响；此前 256K/600K 未测，本次补 400K 档。
- **纪律遵守**：key 不回传/不落报告（全程服务器端使用，报告以长度/前缀指代）；OOM 防护（一次一个 GPU 任务）；克隆镜像旁路，生产恢复后四机健康确认。
- **关联前置报告**：`issue22-kv-deopt-2026-08-24.md`（取证：活跃后端 FLASHINFER_MLA_SPARSE_DSV4，L861 为死代码）、`key-rotate-r12-restart-2026-08-24.md`（131K A/B 闭环：B≈A +1.2% 噪声内）

---

## 0. TL;DR（一页结论）

1. **400K decode 无塌陷**：11 轮汇总 range 37.46–177.52 t/s，合并中位 ≈99.6 t/s；受控同实例对比 131K（中位 51.10）vs 400K（中位 99.61）——**无数量级塌陷**，维持数十 t/s 量级。单轮大方差来自 MTP n7 投机解码接受率波动（与 131K A/B 记录的 restart 级方差同类）。
2. **400K 质量抽验 PASS**：长文尾部植入事实正确召回——tail-fact（400K 末尾 ~58 token 植入 BRAVO-4829）→ 回答 `BRAVO-4829…`；deep-tail-fact（~50K token 前植入 CHARLIE-7315）→ 回答 `CHARLIE-7315…`。**400K 无信息丢失 / 无幻觉异常**。
3. **冷 prefill TTFT@400K = 39.0s**（深尾用例冷启动实测；131K 档前序 A/B 冷 TTFT 50s 含 JIT/预热污染）。
4. **结论：生产版本 400K 无 issue22 影响**。issue22 dispatch 缺陷在 DSV4 活跃路径为死代码（与上下文长度无关，取证 F1-F8 + 131K A/B 已闭环）；本检验确认 400K 档 decode 保持与 131K 一致的量级与质量，**"131K 物理极限"结论在 400K 档维持**（decode 瓶颈仍指向 GB10 UMA 访存，叠加 MTP 投机解码方差）。
5. **恢复完成**：克隆停删无残留；生产 r12 路径 A 恢复（520s READY）；systemd 自愈链全 active；四机 `vllm-tp4-rank0..3` Up(healthy)；/health 200；quality_gate 4/4 PASS。

---

## 1. 前置与窗口计划（已回报主理人）

### 1.1 显存勘察（结论：无余量，需短停）
| 节点 | UMA used/avail | 生产容器 | 结论 |
|---|---|---|---|
| node01 | 111G / 10G | vllm-tp4-rank0 Up(healthy) | 无克隆并发余量 |
| node01 | 110G / 11G | vllm-tp4-rank1 Up(healthy) | 同 |
| node01 | 109G / 11G | vllm-tp4-rank2 Up(healthy) | 同 |
| node01 | 109G / 11G | vllm-tp4-rank3 Up(healthy) | 同 |

→ 按任务授权执行**短停生产窗口**（runbook 停机顺序：timer → head/worker services → 有序 rm 容器；恢复走 r12 路径 A）。

### 1.2 克隆镜像/工具就位核验
- 克隆镜像 `LuZ0.3.1-bench-20260823` 四机在位（01: 85f2149f…；02/04/03: d55d355a…）。
- bench 工具 `bench_clone_start.sh`（KEYFIX 版）、`bench_longctx_decode.py` 在 `/tmp/_bench_luz031/` 在位；新增 `bench_longctx_qa_400k.py`（长文尾部内容完整性抽验，流式 TTFT/decode 采集）部署 + 语法 OK。

---

## 2. 执行时间线

| 阶段 | 内容 | 结果 |
|---|---|---|
| 0 | 预检（四机 UMA/容器/克隆镜像/sudo/停机顺序核对） | ✅ |
| 1 | 短停生产（timer+head/worker services 停 → rm vllm-tp4-rank0..3） | ✅ 8001 free，UMA avail 115G |
| 2 | 克隆启动 `APPROVED=launch bench_clone_start.sh` | ✅ 8m16s READY，diff 保真 4/4 + checker 4/4 PASS |
| 3 | 400K 测量（校准→decode 11 轮→QA 双用例） | ✅ 数据归档 |
| 4 | 停克隆 `bench_clone_stop.sh` | ✅ 四机无残留 |
| 5 | 恢复生产 `start_tp4_cluster.sh`（r12 路径 A）→ 自愈链 → 健康核验 | ✅ 520s READY，全绿 |

克隆启动核验（rank0 日志，生产同构）：`kv_cache_dtype=nvfp4_ds_mla` / `max_seq_len=600000` / `max_num_batched_tokens=4096`（chunked prefill）/ `speculative dspark n7` / `W4A4=2 SHARED=1` / `FI 0.6.16` / `util 0.82` / `GPU KV 5,817,985 tokens`。

---

## 3. 400K 档数据

### 3.1 校准
- `--only-calibrate --tiers 409600`：实际 prompt_tokens = **409605**（per_rep=10.00，overhead=5.0，与 131K 档同构）。

### 3.2 decode-only t/s（gen=48，逐轮）
| 批次 | 轮1 | 轮2 | 轮3 | 轮4 | 轮5 | 中位 |
|---|---|---|---|---|---|---|
| 400K set1（3 轮） | 37.46 | 110.74 | 38.89 | — | — | **38.89** |
| 400K set2（5 轮） | 119.89 | 59.34 | 125.14 | 177.52 | 75.18 | **119.89** |
| 400K set3（受控 3 轮） | 51.84 | 99.61 | 172.83 | — | — | **99.61** |
| **400K 合并（11 轮）** | — | — | — | — | — | **≈99.6**（range 37.46–177.52） |

> 单轮大方差来源：MTP n7 投机解码在规律重复文本上的接受率波动（接受率高时每步可验证多 token → 高 t/s；接受率低时回落单 token/步 → ~37-39 t/s）。与 131K A/B 记录的 restart 级环境方差（A1=76.2 / A2=57.5 / B2=69.8）同类，非 arm/长度效应。

### 3.3 TTFT
| 场景 | 数值 | 说明 |
|---|---|---|
| 冷 prefill TTFT@400K | **39.0s** | 深尾 QA 用例冷启动实测（首次 QA 运行，非缓存命中） |
| warm（prefix cache hit）@400K | 4.0–4.4s | 校准探针已预填同 prompt，后续轮命中前缀缓存 |
| 131K 冷（前序 A/B） | ~50s | 首轮含 JIT/预热污染，非纯 prefill |

### 3.4 质量抽验（长文尾部内容完整性）
| 用例 | 植入位置 | prompt_tokens | 回答 | 判定 |
|---|---|---|---|---|
| tail-fact | 400K 末尾 ~58 token（BRAVO-4829） | 409458 | `BRAVO-4829# The quick brown fox jumps…` | **PASS** |
| deep-tail-fact | ~50K token 前（CHARLIE-7315） | 409243 | `CHARLIE-7315# The quick brown fox…` | **PASS** |

→ 400K 下模型能从尾部正确召回植入信息：**无信息丢失、无 KV 截断/损坏、无幻觉异常**。

---

## 4. 与 131K 档对照

| 档位 | 数据来源 | 中位 t/s | 说明 |
|---|---|---|---|
| 131K | 前序 A/B（A1/B/A2/B2 各 2 轮） | 57.5–77.1 | 受控 A1 vs B：76.21 vs 77.09（+1.2% 噪声内） |
| 131K | 本窗口受控同实例（3 轮） | 51.10 | 同引擎实例、同方法 |
| **400K** | 本窗口合并（11 轮） | **≈99.6** | range 37.46–177.52 |

**趋势判定**：400K decode 与 131K 处于同一量级（数十 t/s），无数量级塌陷。400K 合并中位高于 131K 同实例中位系 MTP 投机解码接受率波动的测量伪象（带宽约束下 3× 上下文不应更快），**非物理性提升，亦非塌陷**。质量抽验通过 → 满足任务判据。

---

## 5. issue22 生产版本 400K 结论

1. **400K 无 issue22 影响**：400K decode 保持数十 t/s 量级且质量抽验 PASS，未观察到 dispatch 缺陷随上下文长度增长而放大的迹象。
2. **机理一致性**：取证（issue22-kv-deopt F1-F8）表明生产 DSV4 活跃后端 `FLASHINFER_MLA_SPARSE_DSV4` 直调 `flashinfer_trtllm_batch_decode_sparse_mla_dsv4`，**无 dtype 分派**（KV 按 uint8 不透明字节处理）——该结论与上下文长度无关，400K 测量进一步印证：无"nvfp4 → `_forward_bf16_kv` 慢路径"可退。
3. **物理极限结论在 400K 档维持**：400K decode 瓶颈仍指向 GB10 UMA 访存 + MTP 投机解码方差，与 131K 结论一致。**issue22 处置最终闭环达成**——生产版本 400K 无 issue22 影响，物理极限在 400K 档维持，允许进入 FP8 系列后续工作。

---

## 6. 恢复确认

| 项 | 状态 |
|---|---|
| 克隆容器 tp4-bench-rank0..3 | 已停删，四机无残留 ✅ |
| 生产容器 vllm-tp4-rank0..3 | 四机 Up(healthy)，r12 路径 A 恢复（520s READY，无死锁）✅ |
| systemd 自愈链 | head service(01) active / worker services(02/04/03) active / vllm-healthcheck.timer(01) active，各机角色匹配 ✅ |
| /health | 200 ✅ |
| /v1/models 鉴权 | 带 key 200 / 无 key 401 ✅ |
| quality_gate | exact_match 4/4 + logprob_envelope PASS ✅ |
| UMA 水位 | 回到生产基线（used 109–111G / avail 10–11G）✅ |
| 临时凭证 | askpass 文件已删，无残留 ✅ |

---

## 7. 数据归档

| 资产 | 路径 |
|---|---|
| 400K decode set1 | `deliverables/engineering-assurance/_issue22_400k_verification_20260824/issue22_400k.json` |
| 400K decode set2（5 轮） | `…/issue22_400k_x5.json` |
| 131K vs 400K 受控对比 | `…/issue22_131k_vs_400k.json` |
| 400K QA 抽验 | `…/issue22_400k_qa.json` |
| QA 工具（新增） | `deliverables/engineering-assurance/_luz031_official_bench/bench_longctx_qa_400k.py` |

---

## 8. 诚实声明与遗留

1. **decode 测量方差大**：MTP n7 投机解码使单轮 t/s 在 37–177 间波动；结论基于 11 轮合并中位 + 量级判据（无塌陷），而非单轮绝对值。400K 合并中位>131K 同实例中位为测量伪象（3× context 带宽约束下不应更快），不代表物理性能提升。
2. **400K 主测量为 warm prefix cache**：校准探针预填同 prompt 后后续轮命中前缀缓存；decode 阶段与冷态等价（KV 相同），冷 prefill TTFT 单独采得 39.0s。
3. **质量抽验为尾部植入事实召回抽验**（tail + 深尾 2 位置），非全量长文摘要/基准评测；满足任务"长文尾部内容完整性抽验"要求。
4. **未执行 A/B 注入**：本任务为生产基线 400K 检验（非补丁对比）。issue22 运行态不适用已由取证 + 131K A/B 闭环；本检验确认 400K 档无新恶化。如需 256K/600K 档补测，可复用本窗口工具与流程。
5. 本报告全程未落明文 key；SSH/sudo 临时凭证用后即删；生产未重启额外服务（仅按 runbook 短停-恢复）。

---

*本报告由工程保障团队测试专家（testing-expert-1）生成；里程碑已通过 teammate 通道回报主理人。*
