# API Key 手册 + ABCDEF 六方案基准对比综合报告

**日期**：2026-08-05
**工作流**：工作流 4（部署验证）+ 工作流 1（测试对比）组合
**参与成员**：Docu（技术文档师）/ Tessa（测试专家）/ Zhen（主理人，汇编）

---

## 📌 TL;DR（执行摘要）

- **API Key 手册已落盘**：LiteLLM 4000 + 8003 双轨全部 key 的使用说明（6 节 + §7 待确认项）。
- **ABCDEF 六方案对比完成**：**seqs 是吞吐最大杠杆**（A/C seqs=128 → c5 100-111 t/s vs F/E 6 → 77-91 vs D 1 → ~34）；**DSpark 投机带来 ~1.5× decode 加速**（A vs B 无 spec：-38%）；**F vs E 稳态持平**；**D 无法并发**（seqs=1）。
- **F 生产已恢复**（双机 healthy，8001/8003/4000 全通，复测无漂移）。
- **严重度分布**：🟠高 0 / 🟡中 2（A 接受率 45.8% 异常待复核、C 32k 档回退待查）/ 🟢低若干
- **阻塞 / 非阻塞**：非阻塞。

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟢 六方案对比完成，F 保持生产 |
| 阻塞项数量 | 0 |
| 关键行动项 | 4 条 |
| 建议下一步 | 复核 A 接受率异常 + C 32k 回退；F 生产维持 |

---

## 🔑 一、API Key 使用手册（Docu）

**文件**：`deliverables/engineering-assurance/litellm-api-key-manual-2026-08-05.md`

| Key | 值 | 用途 | 网关 |
|-----|-----|------|------|
| Master | sk-litellm-master-b9158f0b67dec7d9e395d54cb462afe2 | 管理（勿外泄） | 4000 |
| Chat | sk-U_cIbL63-5c27rayJO3S6w | local-v4-flash / deepseek-v4-flash（rpm300/tpm50k） | 4000 |
| Embedding | <API_KEY> | local-embedding（rpm300/tpm100k） | 4000 |
| 8003 客户端 | <API_KEY>-64b0374c6f2840fe | chat/responses/embeddings/models | 8003 |
| 内部 | <API_KEY>-11282c642841cb21092911db1135e2528d34eb881abc9bfa | vLLM 8001（勿外泄） | 内部 |

**§7 待确认项**：8003 实测模型名 / 8003 key 限流 / 8003 key 轮换步骤 / Master key 是否已改 env

---

## 📊 二、ABCDEF 六方案基准对比（Tessa）

### 方案定义与关键结果

| 方案 | 镜像 | ctx | seqs | DSpark | c1 decode | c5 agg | 接受率 |
|------|------|-----|------|--------|-----------|--------|--------|
| A | hybrid-1.6 | 393K | 128 | prob | **45.9** | **102.6** | 45.8% ⚠️ |
| B | hybrid-1.6 | 393K | 128 | 无 | 27.4 | 57.9 | N/A |
| C | hybrid fix | 393K | 128 | prob | 38.7 | 97.7 | 36.9% |
| D | hybrid-1.6 | 1M | 1 | prob | 39.4 | 34.7 | 36.7% |
| E | anemll 0.1.1 | 600K | 6 | prob | 40.4 | 77.3 | 34.1% |
| **F** | anemll 0.2.1 | 600K | 6 | **greedy** | 39.6 | 87.2 | 34.1% |

### 核心结论

1. **seqs 是吞吐最大杠杆**：A/C（seqs=128）c5 达 100-111 t/s；F/E（6）77-91；D（1）~34。并发场景 A/C 全面领先。
2. **DSpark 投机带来 ~1.5× decode 加速**：A（有 spec）42-45 t/s vs B（无 spec）27.4（-38%），c5 聚合 +64-95%。**spec 剥离不可取。**
3. **C(fix,0.85) vs A(0.8)**：中小 ctx 相当（±10%），但 C 在 32768 档明显落后（c5 81.3 vs A 106.2）——大 ctx 下 0.85 显存利用率疑似触发调度竞争，建议查 C 的 32k 回退。
4. **F vs E 稳态持平**（差距 ≤16%），接受率均 34.1%（随机前缀下 greedy/prob 无差异）；F 优势仍是 E 的冷启动 JIT 惩罚。
5. **D 无法并发**：c3/c5 恒定 ~34-35 t/s、TTFT 排队 4-15s；单流与 F 相当。
6. **冷启动普遍**：A/B/C/D/E 每档首请求 5-16s（JIT），仅生产热身态 F 无此惩罚。

### ⚠️ 需关注

- **A 接受率 45.8%** 明显高于 C/D 的 36.7-36.9%（同 hybrid 同 workload），差异 >8pt 超噪声——建议更大样本复核（可能含冷启动窗口偏差）
- 32k 档全部正常完成未降级

---

## ✅ 行动清单（按优先级排序）

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | 复核 A 方案接受率 45.8% 异常（更大样本） | Testing | P1 | 本周 |
| 2 | 排查 C 方案 32k 档回退（0.85 显存利用率调度竞争假设） | SRE+Archi | P1 | 本周 |
| 3 | API Key 手册 §7 待确认项回填（8003 模型名/限流/轮换/master env） | SRE | P1 | 本周 |
| 4 | 批量对比测试预热机制标准化（规避冷启动 JIT 污染） | Testing | P2 | 按需 |

---

## ⚠️ 待完善 / 已知局限

- 接受率绑定 workload：随机前缀文本下 greedy/prob 无差异（34.1% vs 34.1%），thinking 负载下 greedy 更高（此前 79.2% vs 74.4%）
- A 接受率异常与 C 32k 回退未定论（见行动项）
- 4000 的 client token 未在本次验证（测试前状态）

---

## 📚 数据来源 & 成员产出索引

- **Docu（技术文档师）**：`litellm-api-key-manual-2026-08-05.md`（6 节 + §7）
- **Tessa（测试专家）**：`_tessa_abcdef_raw_2026-08-05.txt` + `bench-compare-ABCDEF-2026-08-05.md` + `def_bench_data/raw_{A,B,C,D_abc,E_abc,F}.json`（make_tables.py 六方案可复现）
- **前置报告**：`dualtrack-def-bench-2026-08-05.md`、`litellm-gateway-verify-abcde-compare-2026-08-05.md`

---

> 本报告由工程保障团队 AI 协作生成，关键决策（F 生产维持、A/C 异常复核）请由人类工程负责人复核。
