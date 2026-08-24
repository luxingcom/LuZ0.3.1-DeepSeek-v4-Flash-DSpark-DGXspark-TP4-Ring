# TileLang A+D + probabilistic 切换 落地报告

**日期**：2026-08-05
**工作流**：工作流 1（代码审查）+ 工作流 4（部署落地）组合
**参与成员**：Cody（代码审查师）/ Rex（SRE 工程师）/ Zhen（主理人，汇编）

---

## 📌 TL;DR（执行摘要）

- **TileLang A+D 已部署**：`mhc/tilelang.py` 两档固定 patch（decode/prefill）——cache key 不再随 num_tokens 变，**运行时 JIT 归零，c1 TTFT 18s → ~1s**；预热脚本 + TILELANG_CACHE_DIR 持久卷就绪。
- **probabilistic 切换已落地**：采样改 probabilistic + 保留动态 K，**code c5 +48% / json c5 +13%**（vs greedy 基线 ~87），GSM8K 18/20 无退化；per-key 模板（prob temp0.7 / greedy temp0.1）上线。
- **性能影响**：正向（无额外采样开销，接受率 0.78-0.84）；**客户端需 temp>0.1 拿满加速**。
- **阻塞 / 非阻塞**：非阻塞。F 生产运行新基线（probabilistic + 动态K + tilelang patch）。

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟢 落地完成并验证通过 |
| 阻塞项数量 | 0 |
| 关键行动项 | 4 条 |
| 建议下一步 | 稳定后跑完整 matrix 固化基线 + 客户端 temp>0.1 约定固化 |

---

## 🔧 一、TileLang A+D（Cody）

### Patch 设计（3 处，仅 mhc/tilelang.py）
| 函数 | 改动 |
|------|------|
| mhc_fused_post_pre_tilelang | decode 档（≤16 tokens）→ tile_n=2/n_splits=8；prefill 档 → n_splits=4/tile_n=1 |
| mhc_pre_tilelang / mhc_pre_broadcast_tilelang | n_splits 固定=4（原随 num_tokens 变） |

### 正确性确认
- split-k 纯属 K 归约分片（T.serial 累加部分和）→ 数学结果不变
- tile_n 整除 hc_mult3 已确认（24/2 ✓）+ 奇数守卫
- 两档固定值 8/4 均为线上已验证组合（max-num-seqs=6 → decode 恒 <8）

### 实施
- ✅ 双机挂载（md5 三端一致 8779bea4）+ py_compile + 容器内 import 验证
- ✅ 预热脚本 `~/warmup_mhc.sh`（2 请求覆盖两档，幂等）
- ✅ 持久卷 `-v ~/tilelang-cache:/root/.tilelang/cache:rw`（已写入编译 kernel，重启免重编）

### 预期效果
- cache key 不再随 num_tokens 变 → 运行时 JIT 归零 → **c1 TTFT 18s → ~1s**

---

## 🎲 二、probabilistic 切换（Rex）

### 切换内容
| 项 | 变更 |
|----|------|
| 采样 | greedy → **probabilistic**（保留动态 K） |
| 持久卷 | TILELANG_CACHE_DIR 双机挂载 |
| 脚本备份 | .bak.greedy-* / .bak-tilelang-2tier-* |
| 一次切换 | worker 先 → head 后，READY 330s，双机确认 probabilistic+动态K+双挂载 |

### 验证（temp 0.7）
| 项 | 结果 |
|----|------|
| 接受率 | 0.784（prob 生效） |
| **code c5** | **129 t/s（+48% vs ~87 基线）** |
| **json c5** | **98.5 t/s（+13%）** |
| GSM8K | 18/20 无退化 |

### per-key 模板（LiteLLM）
| key | 值 | 模型 | temp |
|-----|-----|------|------|
| **Prob** | `<API_KEY>` | dspark-prob | 0.7 |
| **Greedy** | `<API_KEY>_C8xuN9rqHhg` | dspark-greedy | 0.1 |

- config.yaml 加两条目（保留原条目），proxy 重启加载，实测 200
- 文档：`litellm-api-key-manual-2026-08-05.md` §2.4 + `deploy-f-probabilistic-switch-2026-08-05.md`

---

## ⚠️ 待办 / 已知局限

| # | 项 | 说明 |
|---|-----|------|
| 1 | **key 级 aliases 不生效**（LiteLLM 1.83.7 实测） | 客户端须**显式 model=dspark-prob/greedy**（model=dspark→401） |
| 2 | chat_template_kwargs 透传未确认 | 思考链走 8003（自建网关） |
| 3 | 单窗口吞吐噪声大 | 稳定后跑完整 matrix 固化新基线 |
| 4 | **temp>0.1 靠请求端自觉** | LiteLLM 无 min_temperature，可后续加 guardrail |

---

## ✅ 行动清单（按优先级排序）

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | 稳定后跑完整 matrix（probabilistic+动态K+tilelang patch 三叠加）固化生产基线 | Testing | P1 | 本周 |
| 2 | 客户端 temp>0.1 约定固化（API 文档 + 可选 LiteLLM guardrail） | Docu+SRE | P1 | 本周 |
| 3 | 确认思考链客户端走 8003（per-key 模板覆盖 chat 路径） | SRE | P1 | 本周 |
| 4 | 观察期：新基线运行 1-2 天确认稳定（接受率/吞吐/质量） | SRE | P1 | 持续 |

---

## ⚠️ 待完善 / 已知局限

- key aliases 不生效 → 客户端需显式 model 名（文档已标注）
- chat_template_kwargs 透传未验证（思考链场景）
- 完整 matrix 待稳定后补跑

---

## 📚 数据来源 & 成员产出索引

- **Cody（代码审查师）**：`tilelang.two-tier.patch` / `tilelang.py.patched` / `warmup_mhc.sh`（本地 patch/audit/patch/）
- **Rex（SRE 工程师）**：`deploy-f-probabilistic-switch-2026-08-05.md` / `litellm-api-key-manual-2026-08-05.md` §2.4 / `prob_PROD_verify_matrix.json` / `prob_PROD_gsm8k.json`
- **前置报告**：`recheck-latency-jit-adaptive-2026-08-05.md`、`combo-ab-prob-eval-2026-08-05.md`

---

> 本报告由工程保障团队 AI 协作生成，关键决策（probabilistic 生产切换、tilelang patch 上线）请由人类工程负责人复核。
