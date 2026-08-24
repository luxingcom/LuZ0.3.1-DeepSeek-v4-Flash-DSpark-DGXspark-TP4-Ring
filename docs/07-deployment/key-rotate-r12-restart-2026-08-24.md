# 合并短窗口：API key 轮换 + 脱敏 overlay 接线 + r12 生产重启实测 + issue22 A/B 归档

- **执行人**：雷克斯（Rex）· SRE 工程师（sre-engineer-3）
- **日期**：2026-08-24（UTC 02:1x — 03:5x）
- **批准**：主理人（工程负责人）批准的合并短窗口
- **纪律**：新 key 不回传/不落报告（本文以前缀 `<KEY_PREFIX_NEW>` 指代新 key、`<KEY_PREFIX_OLD>` 指代旧 key）；sudo 提权用后即删不留文件；复盘无责
- **关联前置报告**：`issue22-kv-deopt-2026-08-24.md`（sre-engineer-2 取证）、`b12x-gate-fix-2026-08-24.md`（sre-engineer-1 r12 修复）、`g1-production-restore-2026-08-24.md`（§3.3 死锁事故）

---

## 0. TL;DR（一页结论）

1. **key 轮换完成并验证**：四机 vLLM 已切换至新 key（前缀 `<KEY_PREFIX_NEW>`，64-hex）。受保护端点 `/v1/models` 验证：**新 key=200、旧 key=401**（`/health` 为 vLLM 开放端点不鉴权）。旧 key `<KEY_PREFIX_OLD>` 已留档于 `vllm.env.bak-keyrotate-20260824`。
2. **脱敏 overlay 生效**：生产 head 启动日志 `non-default args` 中 `api_key` 显示 `<redacted>`；四机 `docker logs` grep 旧/新 key 前缀 0 命中。脚本层掩码（`--api-key ********`）同样生效。
3. **r12 冷启动实测通过（两次）**：路径 A `start_tp4_cluster.sh`（v1.5-r12+keyfix）两次冷启动均无死锁，端到端 **560s / 520s** 全 healthy。错峰 `B12X_JIT_STAGGER=20s` 表现正常（未出现 JIT 瞬时失败，无需上调 30s）。**发现并修复 r12 初版回归**（worker ssh 启动丢失 `VLLM_API_KEY` 传递）。
4. **issue22 A/B 归档**：131K decode-only 四臂数据 B≈A（受控 A1 vs B：76.21 vs 77.09，+1.2% 噪声内）；**L861 注入无运行态效果**，与 sre-engineer-2 取证一致（L861 对 DSV4 为死代码），"131K 物理极限"结论维持。跨重启环境方差（A2=57.5 偏低）大于 arm 间差异。
5. **恢复完成**：systemd 自愈链已恢复（timer+head/worker services active），四机 `vllm-tp4-rank0..3` Up(healthy)，8001=200，quality_gate 4/4 PASS。

---

## 1. 窗口执行时间线

| 阶段 | 内容 | 结果 |
|---|---|---|
| 0 | 预检（容器/服务/资产/克隆镜像/脚本版本） | ✅ |
| 1 | 停机（timer→head/worker services→有序 rm 容器） | ✅ |
| 2 | key 轮换（备份→生成→四机写入→核验） | ✅ |
| 3 | overlay 接线（head/worker BINDS 加挂载）+ r12 部署 | ✅ |
| 4 | **生产重启 r12 路径 A**（首次） | ✅ 560s READY |
| 4a | **发现并修复 r12 keyfix 回归** | ✅ |
| 5 | issue22 A/B（停生产→克隆→四臂测量→停克隆） | ✅ 数据归档 |
| 6 | r12 路径 A 恢复生产（二次实测）→ 恢复 systemd 自愈链 | ✅ 520s READY |
| 7 | 安全核验（新 key 未泄漏 / 日志脱敏） | ✅ |

两次生产短停（符合 A/B 需 GPU 独占约束：生产 util 0.82 占满 UMA，克隆无法并发）。

---

## 2. API key 轮换确认（Phase 2）

### 2.1 操作
- 备份四机 `vllm.env` → `vllm.env.bak-keyrotate-20260824`（600 root:root，旧 key `<KEY_PREFIX_OLD>` 留档）。
- 生成新 64-hex key（`openssl rand -hex 32`），四机写入，保持 600 root:root。
- 新 key 全程通过 stdin 传递（不落命令行为/日志），仅以文件 `vllm.env` 为单一载体。

### 2.2 核验矩阵
| 项 | 结果 |
|---|---|
| 四机 vllm.env md5 一致 | `e376c96752c99347e0998cc9d1eada87`（四机相同）✅ |
| 权限 | `-rw------- root root` ✅ |
| 新 key 长度 | 64 hex ✅ |
| 新 key 前缀 | `<KEY_PREFIX_NEW>`（报告指代） |
| 旧 key 备份 | `vllm.env.bak-keyrotate-20260824` 内前缀 `<KEY_PREFIX_OLD>` ✅ |

### 2.3 运行时生效验证（重启后）
| 端点 | 新 key | 旧 key | 无 key |
|---|---|---|---|
| `/v1/models`（受保护） | **200** | **401** | 401 |
| `/health` | 200 | 200 | 200 |

> `/health` 为 vLLM 开放端点（不校验 key），属设计行为；受保护端点 `/v1/models` 完整验证了 401→200 切换。

### 2.4 systemd 同源确认
- `vllm-tp4-head.service` / `vllm-tp4-worker.service` 均 `EnvironmentFile=<INSTALL_DIR>/secrets/vllm.env`（单一来源，四机确认）。

---

## 3. 脱敏 overlay 接线确认（Phase 3）

### 3.1 接线
- 生产启动脚本 BINDS 新增一行（head `start_tp4_head.sh:178`；worker `start_tp4_worker.sh:213`）：
  ```
  -v <INSTALL_DIR>/overlay-mask/api_utils.py:/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/serve/utils/api_utils.py:ro
  ```
- 备份 `.bak-overlay-20260824`；四机 `bash -n` + `check_vllm_script.sh` ✅ 全 PASS。

### 3.2 生效确认（重启后）
- head 日志：`[api_utils.py:278] non-default args: {..., 'api_key': '<redacted>', ...}`（原泄漏点 L273 已脱敏）。
- 容器内 overlay md5 `d9c7aeb62458848c5547b02c43e4133a` = 源文件 md5（确认 bind-mount 生效）。
- 四机 `docker logs` grep `<KEY_PREFIX_OLD>|<KEY_PREFIX_NEW>` = **0 命中**（无明文 key）。
- 脚本层掩码：`echo "[i] serve 命令: $MASKED_SERVE_CMD"` 显示 `--api-key ********`（四机已落地，checker PASS）。

---

## 4. r12 生产重启实测（Phase 4 + 6）

### 4.1 r12 初版回归发现与修复（KEYFIX）
- **现象**：r12 首次路径 A 冷启动，step 2.5 新门禁正常（TCPStore 复核 0s 通过，无死锁），但 3 个 worker 全部未起 → step4 120s 超时中止；head 因无 worker 入域，300s 后 `Engine core initialization failed`（预期内，非死锁）。
- **根因**：r11 的 worker ssh 启动命令带 `VLLM_API_KEY=${VLLM_API_KEY}`；r12 初版改写 step 3 错峰时把该前缀遗漏 → 远端 worker 叶子脚本 `${VLLM_API_KEY:?}` 强依赖直接退出，容器从未创建。
- **处置**：恢复 `VLLM_API_KEY=${VLLM_API_KEY}` 传递（对齐 r11），加 `KEYFIX` 注释；本地+server 双份同步；`bash -n`+checker ✅；server 备份 `.bak-r12-keyfix-20260824`。

### 4.2 两次 r12 冷启动结果
| 项 | 第一次（Phase 4） | 第二次（Phase 6 恢复） |
|---|---|---|
| 脚本 | v1.5-r12+keyfix | v1.5-r12+keyfix |
| 门禁 step 2.5 | TCPStore 复核 0s 通过 ✅ | 同 ✅ |
| 错峰 | rank1@+20s, rank2@+40s, rank3@+60s | 同 |
| JIT 瞬时失败 | 无（无需上调 30s） | 无 |
| READY | 560s | 520s |
| 容器 | 四机 Up(healthy) | 四机 Up(healthy) |
| /health | 200 | 200 |
| quality_gate | 4/4 PASS | 4/4 PASS |
| 死锁 | 无 | 无 |

### 4.3 启动核验全项（第一次重启后采集，恢复后复验）
`W4A4=2 / SHARED=1 / max-num-batched-tokens=4096 / long-prefill-token-threshold=4096 / gpu-memory-utilization=0.82 / kv-cache-dtype=nvfp4_ds_mla / max-model-len=600000 / MTP n7(dspark, probabilistic) / FI 0.6.16 / CUMEM=0(kv_cache_usage 0.0, 无 OOM)`。

---

## 5. issue22 A/B 归档（Phase 5）

### 5.1 执行说明（重要）
- **交付的 `issue22_ab_runbook.sh` 端到端存在缺陷**（此前未在 GPU 窗口实测过）：
  1. 补丁脚本 `patch_issue22_l861.py` 以 `<container>` 参数设计（在宿主运行），但 runbook 以 `docker exec ... python3 /w/$PATCHER apply` 在容器内调用 → 参数不符；
  2. 克隆容器未挂载 `/w` → 补丁脚本"文件不存在"；
  3. 重启后就绪轮询被 `docker logs` 累积旧日志"假就绪"骗过 → B 臂在未就绪时测量（连接拒绝）。
- **处置**：弃用 runbook 自动编排，改**手动受控 A/B**：L861 单行注入以 `sed` 直接落地（锚点 `self.kv_cache_dtype == "fp8_ds_mla"` 四机各 1 处，替换为 `in ("fp8_ds_mla", "nvfp4_ds_mla")`），`py_compile` 验证；每臂重启后以"startup count 递增 + /health=200"双条件判真就绪。
- 另：`bench_clone_start.sh` 的 worker ssh 启动同样缺 `VLLM_API_KEY` 传递（与 r12 同类回归），已加 `KEYFIX` 修复（对齐 `launch_bench.sh` 惯例）。
- 数据归档：`deliverables/engineering-assurance/_issue22_ab_archive_20260824/issue22_{A1,B,A2,A2_extra,B2}_manual.json`。

### 5.2 数据（131K decode-only，t/s）
| 臂 | 轮1 | 轮2 | 中位 | 说明 |
|---|---|---|---|---|
| A1 baseline | 62.71 | 89.70 | **76.21** | 首轮含 JIT/预热（TTFT~50s） |
| B injected | 65.77 | 88.41 | **77.09** | L861 注入 |
| A2 baseline 复核 | 57.49 | 57.55 | **57.52** | 跨重启方差偏低 |
| A2-extra | 61.12 | — | 61.12 | 同一 A2 实例补测 |
| B2 injected 复核 | 74.11 | 65.46 | **69.79** | 跨重启复核 |

### 5.3 分析
- **受控对比 A1 vs B**：77.09 vs 76.21 = **+1.2%**（噪声内，无显著提升）。
- **跨重启环境方差**：A2 偏低（57.5），但 B2（晚于 A2）反而更高（69.8）→ A2 偏低非时间单调/热漂移，属 restart 级环境方差（flashinfer autotune 缓存命中差异 / cuda graph 捕获差异，见 `env-random-factors-tracking-2026-08-23.md`），**非 arm 效应**。
- 合并口径：baseline 各轮中位 ≈60，injected 各轮中位 ≈70；但差值主要由 A2 单臂拉低造成，A1↔B 严格受控对比无差异。

### 5.4 结论
- **L861 单行 dispatch 补丁对本集群 DSV4 无运行态效果**（B≈A，受控对比 +1.2% 噪声内）→ 与 sre-engineer-2 取证（活跃后端 `FLASHINFER_MLA_SPARSE_DSV4`，不经 `flashmla_sparse.py`）一致。
- **"131K ~物理极限"结论维持**：未观察到 issue22 dispatch 慢路径可退带来的 decode 提升。
- 归档状态：**issue22 证据闭环完成**（取证 + 运行态 A/B 双向支撑）。

---

## 6. 恢复清单（Phase 6）

| 项 | 状态 |
|---|---|
| 克隆容器 | `tp4-bench-rank0..3` 已停删（四机核验无残留）✅ |
| 生产容器（r12 路径 A 恢复） | 四机 `vllm-tp4-rank0..3` Up(healthy) ✅ |
| r12 恢复冷启动 | READY 520s，无死锁 ✅ |
| systemd 自愈链 | `vllm-healthcheck.timer`(head)+`vllm-tp4-head.service`(01)+`vllm-tp4-worker.service`(02/04/03) 全 active；monitor 已采纳运行中容器（docker wait 模式）✅ |
| 服务幂等 | head: active / worker: active / timer: active（各机角色匹配）✅ |
| 8001 | head 200 ✅ |
| quality_gate | 4/4 PASS ✅ |
| 新 key 生效 | `/v1/models` 新 key 200 / 旧 key 401 ✅ |
| 日志脱敏 | 四机 0 明文 key ✅ |

---

## 7. 安全处置（Phase 7）

| 项 | 说明 |
|---|---|
| 新 key 保管 | **未回传/未落任何报告或日志**；仅存在于 `<INSTALL_DIR>/secrets/vllm.env`（600 root）与 systemd EnvironmentFile 引用；本报告以前缀 `<KEY_PREFIX_NEW>` 指代 |
| 旧 key 泄漏面（处置前，据 sre-engineer-2 报告） | ① 启动脚本 echo SERVE_CMD（已掩码）② vLLM `docker logs` non-default args 明文（已 overlay 脱敏）③ 历史 bench 日志 `/tmp/_bench_luz031/logs/*.log` 含旧 key |
| 旧 key 处置 | 已轮换下线；备份 `.bak-keyrotate-20260824` 保留（回滚锚点，含旧 key，600 root） |
| 新 key 消费面 | vLLM head/worker 均经 vllm.env 注入；gateway `aicad-v18-server` 无 `VLLM_API_KEY` env（自带 MODEL_CONFIG_ENC_KEY），**建议窗口外与 gateway 侧核对**是否直连 8001 需同步新 key |

---

## 8. 备份与留档

| 资产 | 位置 |
|---|---|
| 旧 key 备份 | 四机 `<INSTALL_DIR>/secrets/vllm.env.bak-keyrotate-20260824` |
| overlay 修改前脚本 | 四机 `start_tp4_{head,worker}.sh.bak-overlay-20260824` |
| r12 部署前 r11 | `start_tp4_cluster.sh.bak-b12xgate-fix-20260824` |
| r12 keyfix 前 | `start_tp4_cluster.sh.bak-r12-keyfix-20260824` |
| A/B 数据 | `_issue22_ab_archive_20260824/*.json`（5 个臂） |
| r12 编排日志 | node01 `/tmp/r12_start_20260824.log`、`/tmp/r12_final_restore_20260824.log` |

---

## 9. 诚实声明与遗留

1. **issue22_ab_runbook.sh 交付态不可用**（参数/挂载/假就绪三处缺陷），本窗口以手动受控 A/B 完成；建议后续修订 runbook（补丁脚本调用方式、克隆容器挂 `/w`、重启就绪判定改"计数递增+health"）再复验一次以固化可复现流程。
2. **A/B 数据为 131K 单档、2 轮中位**：满足"低轮数证据闭环"；256K/600K 未测（时间窗口约束）。四臂观测到 restart 级环境方差，`env-random-factors-tracking` 已有记录。
3. **patch_issue22_l861.py 存在 shell 引号缺陷**（OLD 内含双引号与 `python3 -c "..."` 外层引号冲突 → 注入 AssertionError），本次改用 sed 直接落地；建议修订脚本的注入方式。
4. 新 key 与 gateway 下游消费面核对（§7）建议窗口外完成。
5. 本报告全程未落明文 key；SSH/sudo 操作用后即清，无遗留临时文件。

---

*本报告由工程保障团队 SRE（sre-engineer-3）生成；里程碑已通过 teammate 通道回报主理人。*
