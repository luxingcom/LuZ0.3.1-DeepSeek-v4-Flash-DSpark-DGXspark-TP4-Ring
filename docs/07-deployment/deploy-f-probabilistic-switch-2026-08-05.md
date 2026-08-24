# F 生产 probabilistic 切换部署记录

**日期**：2026-08-05
**执行人**：SRE 雷克斯（Rex）+ 协作：科迪（code-reviewer，tilelang.py 2tier 挂载）
**环境**：F 生产，anemll 0.2.1-v026.0，b12x + breakable + 动态K
**机器**：head=spark-05cd(<NODE_IP>) / worker=edgexpert-0c69(<NODE_IP>)

---

## 1. 变更总览

| # | 变更 | 生效方式 | 状态 |
|---|------|----------|------|
| ① | F 生产采样 greedy → **probabilistic**（保留动态K） | 容器切换 | ✅ |
| ② | 请求端 temp>0.1 强制（probabilistic 生效前提，源码确认 temp≈0 回退 greedy） | 客户端承担 + LiteLLM 默认值 | ✅（文档化） |
| ③ | LiteLLM per-key 模板（dspark-prob 0.7 / dspark-greedy 0.1） | LiteLLM config + 双 key | ✅ |
| ④ | TILELANG_CACHE_DIR 持久卷（~/tilelang-cache） | 容器切换（bind mount） | ✅ |

一次容器切换同时生效 ① + ④ + 科迪的 tilelang.py 2tier 挂载（Plan A+D 固定 tile_n/n_splits，消除 per-request JIT 重编译）。

---

## 2. TILELANG_CACHE_DIR 持久卷（Task 1）

- **确认路径**：镜像内 `tilelang.env` 默认 `TILELANG_CACHE_DIR=/root/.tilelang/cache`，TMP 同目录 `/root/.tilelang/cache/tmp`。
- **实施**：双机建 `~/tilelang-cache/`；启动脚本 BINDS 追加 `-v "$HOME/tilelang-cache:/root/.tilelang/cache:rw"`（head L114 / worker L113）。
- **验证**：双机 `/root/.tilelang/cache/0.1.9-aarch64/kernels/` 已写入编译 kernel（各 ~2.0M）。下次重启免 JIT 重编译。

---

## 3. probabilistic 切换（Task 2）

### 3.1 脚本变更
双机 `start_*_v026r.sh`：`draft_sample_method":"greedy"` → `"probabilistic"`，**保留** `num_speculative_tokens_per_batch_size:[[1,1,5],[2,4,4],[5,6,3]]`（动态K）。bash -n 通过。

### 3.2 容器切换
与科迪协调后**一次切换**（他已完成 tilelang.py 挂载合入脚本）：
1. worker(<MGMT_OCTET>) 先：`bash ~/start_worker_v026r.sh` → 容器 Up，probabilistic + 双 tilelang 挂载生效
2. head(<MGMT_OCTET>) 后：`bash ~/start_head_v026r.sh` → READY（330s 冷启动）

运行中容器确认：
```
--speculative-config '{"method":"dspark","num_speculative_tokens":5,
  "draft_sample_method":"probabilistic",
  "num_speculative_tokens_per_batch_size":[[1,1,5],[2,4,4],[5,6,3]]}'
```
双机挂载：`tilelang-cache:/root/.tilelang/cache:rw` + `patch-v026/.../tilelang.py` → `/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/mhc/tilelang.py:ro`（双机 md5 一致 8779bea4）。

### 3.3 快速验证（temp=0.7，post-switch）
| 负载 | conc | agg_tps | DSpark 接受率 | n_err |
|------|------|---------|---------------|-------|
| code | c1 | 58.3 | 0.656 | 0 |
| code | c5 | **129.1** | 0.808 | 0 |
| json | c1 | 65.2 | 0.820 | 0 |
| json | c5 | **98.5** | 0.839 | 0 |

- 全局 spec acceptance **0.7842**（与 probabilistic 评估范围一致，确认采样生效）
- greedy 基线 c5 ~87 → prob c5 code 129 / json 98（**code +48% / json +13%**，含 tilelang 2tier patch 协同收益；单窗口测量噪声较高，趋势明确）
- **GSM8K 20 题：18/20**（temp 0.7）——与 greedy 基线 18/20 持平，**无质量退化**

### 3.4 回滚
- 双机旧脚本备份：`start_*_v026r.sh.bak.greedy-*`（greedy+动态K+无tilelang卷）、`.bak.greedy-20260805_*`（greedy+动态K+tilelang卷）、`*.bak-tilelang-2tier-*`（科迪的汇聚点备份）
- 回滚=还原脚本后重跑 worker→head 切换；LiteLLM 侧无影响（per-key 模板独立）

---

## 4. LiteLLM per-key 模板（Task 3）

### 4.1 config.yaml（worker <MGMT_OCTET>，改前已备份 config.yaml.bak.20260805_150203）
新增 model_list 两条（保留原 local-v4-flash / deepseek-v4-flash / local-embedding）：
```yaml
  - model_name: dspark-prob
    litellm_params: { model: hosted_vllm/deepseek-v4-flash-0731,
                      api_base: http://<NODE_IP>:8001/v1,
                      api_key: os.environ/LITELLM_UPSTREAM_KEY, temperature: 0.7 }
  - model_name: dspark-greedy
    litellm_params: { model: hosted_vllm/deepseek-v4-flash-0731,
                      api_base: http://<NODE_IP>:8001/v1,
                      api_key: os.environ/LITELLM_UPSTREAM_KEY, temperature: 0.1 }
```
litellm-proxy 重启后 5 模型加载确认。

### 4.2 虚拟 Key（master key 生成）
| Key | 值 | models | aliases |
|-----|-----|--------|---------|
| Prob Key | `<API_KEY>` | `["dspark-prob"]` | `{"dspark":"dspark-prob"}` |
| Greedy Key | `<API_KEY>_C8xuN9rqHhg` | `["dspark-greedy"]` | `{"dspark":"dspark-greedy"}` |

**⚠️ 重要发现**：本版 LiteLLM（1.83.7）**key 级 aliases 不生效**（仅存储不应用；实测 `model=dspark` → 401 key_model_access_denied）。客户端**必须显式传 `model=dspark-prob` / `model=dspark-greedy`**。

### 4.3 验证
- prob key 调 `dspark-prob` → 200，输出有方差（temp 0.7 默认生效，可被请求覆盖——实测 temp=0 覆盖后确定性回归）
- greedy key 调 `dspark-greedy` → 200
- 结构化应用 → **Prob Key**（temp≥0.2 建议 0.7，否则 temp≈0 回退 greedy 失去增益）；散文/思考链 → **Greedy Key**
- `enable_thinking`：本版 LiteLLM 未确认 `chat_template_kwargs` 透传至 hosted_vllm，需思考链的客户端建议走 8003 直连或请求体 extra_body 处理（已文档化）

### 4.4 文档
`deliverables/engineering-assurance/litellm-api-key-manual-2026-08-05.md` 已追加 §2.4 per-key 采样模板 + Key 总览表 + 路由指引 + 故障排查。

---

## 5. 性能影响说明
- 切换后（temp 0.7, prob+动态K+tilelang patch）：code c5 agg_tps 129 / json c5 98.5，对比 greedy 基线 c5 ~87 提升约 +48% / +13%；GSM8K 无退化（18/20）
- **对性能影响**：probabilistic 采样本身无额外开销（acceptance 0.78~0.84），tilelang 2tier patch 消除 per-request JIT 重编译，TILELANG_CACHE_DIR 持久化免重启重编译 —— 均为正向或中性；客户端需确保 temp>0.1 才能拿到加速收益

## 6. 问题 / 待办
- [ ] LiteLLM key 级 aliases 不生效（上游行为），客户端统一用显式 model 名；如需 `model=dspark` 透传需升级 LiteLLM 或用 model_list 加 `dspark` 条目（默认 prob）
- [ ] `chat_template_kwargs` 透传未确认（enable_thinking），思考链客户端按 8003 直连处理
- [ ] 单窗口吞吐测量噪声较大，建议稳定后跑一轮完整 matrix（c1/c5/c10 × code/json/prose）固化基线
- [ ] 请求端 temp>0.1 强制靠客户端自觉，LiteLLM 无内置 min_temperature；若需硬强制可后续加 guardrail
