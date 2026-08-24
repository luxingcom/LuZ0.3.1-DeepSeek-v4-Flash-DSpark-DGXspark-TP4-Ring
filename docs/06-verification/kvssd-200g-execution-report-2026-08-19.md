# KV 缓存 SSD 卸载统一 200G 磁盘缓存 — 性能优化落实执行报告

**日期**：2026-08-19
**工作流**：部署执行 + 性能优化落实（工作流 4 变体）
**参与成员**：Cody / Rex / Tessa / Docu（工程保障团队）
**执行督导**：甄宇航（Engineering Director）
**关联计划**：`delivery/kvssd-offload-2026-08-18/SSD统一200G执行计划-2026-08-19.md`

---

## 📌 TL;DR（执行摘要）

- **整体结论**：io 层薄壳补丁（幂等去重 + trim 裁剪 + zstd-3）已上线生效，磁盘落盘密度 **382KB/token → 70.7KB/token（5.4× 改善）**；同时修复了**配额未生效**的生产隐患（KV 原写根分区 → 现落入 200G loop 配额）。实测确认 ≤10KB/token 门禁在 io 薄壳方案下不可达成（详见 §3 关键发现），已按用户决策以实测值定档。
- **严重度分布**：🔴 严重 0 项（已闭环）/ 🟠 高 2 项（G-4 门禁不达、03/04 内存 4G 偏紧）/ 🟡 中 3 项 / 🟢 低 2 项
- **状态**：✅ 生产已上线（4 rank 健康、读回一致、monitor 自愈恢复）；⚠️ G-5 全量 benchmark 后台运行中（预计 3-4h 出结果）

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟡 有条件通过（核心目标调整后达成） |
| 阻塞项数量 | 0 |
| 关键行动项 | 5 条（见行动清单） |
| 落盘密度 | **70.7KB/token**（382KB 基线 → 5.4×；G-4 目标 ≤10KB 调整为实测定档） |
| 配额 | 四节点统一 **200G** loop 镜像，bind mount 修正后**真实生效** |
| 建议下一步 | G-5 benchmark 完成后对比基线定档；如需 ≤10KB 立项源码级（FileMapper）方案 |

---

## 一、执行结果（阶段 0-6）

### 阶段 0：准备 ✅
- 四节点启动脚本备份 `.bak-kvssd200g-20260819-100628`（01 head md5 `a6b82155`、02 head `9689b6d5`、02/03/04 worker `254a3c7c`，三方留档）
- zstd wheel `zstandard-0.25.0-cp312-manylinux2014_aarch64` 固化四节点 `<INSTALL_DIR>/envs/zstd/`（md5 `f235b7d6`）
- TP4 生产健康确认（HTTP 200）
- **阻塞项**：计划声称的 `/tmp/zstd_pkg/` wheel 实机不存在 → 已从 PyPI 重新下载并固化

### 阶段 1：io 补丁离线验证（G-1）✅
- Cody（code-reviewer）实现补丁：文件头 `KVZSTD01 + orig_len + payload_len + zstd payload`，4096 对齐；per-path lock + O_EXCL 确定性 tmp + flock 陈旧恢复；trim 到最后一个非零字节；legacy 兼容读；失败删文件 + raise（保留 `kv_load_failure_policy=fail` 语义）
- **容器内真实 KV 块验证（5 组全 PASS）**：byte 级往返一致 ✅ / 5 并发同 path 去重仅 1 份 ✅ / legacy 兼容读 ✅ / orig_len 篡改检测 raise ✅
- 压缩后 4.26MB → 614~807KB（有效段 81% 压缩率，NVFP4 高熵）
- Cody 审查：**有条件 Approve**；并指出原版 2 个 Critical bug（`exists()` TOCTOU 竞态 + 随机 tmp suffix 使 O_EXCL 失效）——补丁已修复

### 阶段 2：配额重建（G-2）✅
- 01/02 800G 镜像重建为 200G（备份 `kvssd.img.bak-800g-20260819-020906/020907` 保留）；03/04 保持 200G
- **关键修正（Rex 方案）**：实机发现 `/opt/aicad-kvssd` 是根分区普通目录、配额**从未生效**（19G KV 写根分区）→ 四节点 `mount --bind /mnt/kvssd-quota /opt/aicad-kvssd` + fstab 持久化（bind,nofail）→ **配额真实生效**（四节点 df 196G、inode=2、0700）

### 阶段 3：灰度挂载与 4 rank 重启（G-3）✅
- 四节点启动脚本补丁化：BINDS 追加 `io.py` 覆盖挂载（tilelang.py 同款先例）+ zstd wheel 挂载；SERVE_CMD 前缀注入 `pip3 install --no-index --no-deps`（实测 02/03/04 容器原本无 zstd，必须注入）
- **踩坑**：head 脚本依赖 `VLLM_API_KEY` 环境变量（无 source 逻辑）→ 手动 export 后启动；monitor（`vllm-tp4-head.service` + `vllm-healthcheck.timer`）在停机窗口自动拉起 rank0 → **先停 monitor 再操作，完成后恢复**
- throwaway 容器预验证：zstd 0.25.0 安装 + 补丁加载全 PASS
- **G-3 全绿**：4 rank 就绪 /health 200 / TieringOffloadingSpec 日志确认 / io.py 挂载路径确认 / zstd 0.25.0 四节点 / 内存增量 ≤3GB（01:+0、02:-2、03/04:-2） / dmesg OOM=0

### 阶段 4：存储效率复测（G-4）✅（按 Tessa 方案）
- 3×9K 随机前缀请求（前缀命中=0）：ΔS=2.97GB / 41977 tokens → **70.7KB/token**
- 落盘验证：魔数 `KVZSTD01` 生效、3536 文件 614~967KB（旧 4.26MB）、loop 200G 配额内 2.8G
- **读回验证 PASS**：同 prefix 二次请求 cached=8192 token（SSD 回载），TTFT 4.95s→0.71s（7×），输出一致，无 KV 损坏
- `kv_offload_store_bytes_total` 指标正常增长
- **门禁判定**：≤10KB 未达成 → 按用户决策以实测值定档（详见 §3）

### 阶段 5：全量 benchmark（G-5）🔄 运行中
- 已启动 54 组合（Tessa 校正：需 6 档 ctx 含 32768，5 档只有 45 组合），`--group KVSSD200G`，asyncio，rounds 3
- 命令：`cd <INSTALL_DIR> && python3 bench_prefill_decode_async.py --group KVSSD200G --endpoint http://<NODE_IP>:8001/v1 --key $VLLM_API_KEY --model deepseek-v4-flash-0731 --concurrency 1,3,5 --ctx 512,4096,16384,32768,65536,131072 --tasks coding,json,prose --rounds 3 --engine asyncio --out ./results_kvssd_200g`
- 基线：`_archive_scratch/bench_B/summary_B.json`（54 组合齐全，131072/c1: coding PR1767.75/TTFT65.48s 等）可作 B0
- 进度：5/54 时正常（短 ctx 每组 14-22s）；长 ctx 组合（131072）预计每组数分钟~十余分钟
- 门禁 G-5：131072 全 9 格跑通 + TTFT 劣化 ≤10% 或记录待定档

### 阶段 6：收尾 ✅（部分）
- monitor 自愈链路已恢复（service+timer active）
- 补丁/脚本已归档本地 `delivery/kvssd-offload-2026-08-18/kvpatch/`（io.py、apply_patch_to_scripts.py、verify_g1.py、g4_recheck.py、g4_readback.py、throwaway_validate.sh）
- Runbook 更新素材：`kvssd-200g-runbook-update-2026-08-19.md`（Docu 产出，待并入主 Runbook）
- 24h 观察建议：`kv_offload_*` 指标、DRAM 水位、SSD 增长（Grafana 面板待建）

---

## 二、关键发现（实机取证，推翻计划假设）

| # | 发现 | 证据 | 影响 |
|---|------|------|------|
| 1 | **配额从未生效** | `/opt/aicad-kvssd` inode 2369881 属根分区（/dev/nvme0n1p2），与 `/mnt/kvssd-quota`（loop inode 2）非同一文件系统；19G KV 写根分区 | 修复：bind mount 修正（本次已落地）；消除根分区写满隐患 |
| 2 | **跨 group 去重不可行** | g0~g4 内容 sha256 交集=0（150×150 抽样）；同 group 内 500 文件全唯一 | 计划"382→66.6KB/token（5×去重）"假设不成立；5 group 是不同数据（MLA latent/indexer/attn） |
| 3 | **zstd-3 压缩率远低于预期** | 容器内实测 comp/eff=81.1%（计划预期 60-65%）；NVFP4 打包数据高熵 | trim+zstd 后 382→70.7KB/token（5.4×），≤10KB 门禁不可达 |
| 4 | **仅 01 容器有 zstd** | 02/03/04 容器 `ModuleNotFoundError: zstandard`（同镜像不同容器层） | SERVE_CMD 注入 pip install wheel 是必需步骤（已落地） |
| 5 | **54 组合需 6 档 ctx** | 5 档 ctx×3 task×3 conc=45≠54；缺 32768（并发收益分界点） | Tessa 校正：benchmark 命令含 32768（已执行） |

---

## 三、门禁汇总

| 门禁 | 标准 | 结果 |
|------|------|------|
| G-1 | io 补丁离线 byte 一致 + ≤10KB/token | ✅ PASS（byte 一致；≤10KB 按 §2-3 定档调整） |
| G-2 | 四节点配额 200G（df ~196G）+ 0700 | ✅ PASS（重建 + bind 修正后真实生效） |
| G-3 | 4 rank 就绪 / 卸载框架加载 / 内存增量 ≤3GB | ✅ PASS |
| G-4 | 实测 bytes/token ≤10KB | ⚠️ 70.7KB/token（5.4×）；**用户决策接受实测值定档** |
| G-5 | 131072 组全跑通，TTFT 无回归 | 🔄 benchmark 运行中（基线齐备可对比） |

---

## ✅ 行动清单（按优先级排序）

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | G-5 benchmark 完成后：对比 B0 基线（summary_B.json）出 54 组合对比表，131072 TTFT 劣化 ≤10% 或记录待定档 | 主理人/Tessa | P0 | benchmark 完成后 |
| 2 | 24h 观察：`kv_offload_*` 指标族、DRAM 水位、SSD 增长曲线；Grafana 建面板（PromQL 见 Docu 素材） | Rex/运维 | P1 | 24h 内 |
| 3 | 如需 ≤10KB/token：立项 vLLM 源码级方案（FileMapper 布局改造 / 行内稀疏块索引），评估"每块 57 token 行内 46× 冗余"根因 | Archi | P2 | 另行立项 |
| 4 | Runbook 合并：Docu 产出（kvssd-200g-runbook-update-2026-08-19.md）并入主 Runbook，补充补丁运维/故障 FAQ | Docu | P2 | 1 周内 |
| 5 | 安全收尾：`kvssd.img.bak-800g-*`（01/02 各 800G）确认保留或归档；`/tmp/zstd_pkg` 等临时文件清理；`VLLM_API_KEY` 从命令行 history 清理 | 运维 | P3 | 3 天内 |

---

## ⚠️ 待完善 / 已知局限

1. **G-4 ≤10KB/token 门禁未达成**（70.7KB/token）。根因：5 group 数据不同（不可去重）+ NVFP4 高熵（zstd 81%）。io 薄壳补丁已达上限，进一步需源码级方案。
2. **03/04 内存 available 仅 4G**（重启后从 6G 降 2G，CPU 主层 2GiB 预分配）——大并发长序列场景有 OOM 风险，c5@131072 组合已放矩阵后段单独盯内存。
3. **写入放大仍存在**：每 KV 行 5 个 group 各写一份（内容不同，无法共享），文件数 ~5×行数。
4. **补丁未加自证日志行**（Rex 建议）——G-3 通过 import 路径核验判定；后续版本可在模块加载时打一行 `[kvpatch] io.py loaded`。
5. **monitor 停机窗口竞态**：本次 monitor 在 docker stop 后 30s 内自动拉起 rank0（vllm-tp4-head.service 的 docker wait）——停机操作必须先停 monitor，已纳入本报告运维要点，建议写入 Runbook。

---

## 📚 数据来源 & 成员产出索引

- Cody（代码审查师）：`deliverables/engineering-assurance/kvssd-io-patch-review-2026-08-19.md`（有条件 Approve；发现原版 Critical TOCTOU bug）
- Rex（SRE 工程师）：`deliverables/engineering-assurance/kvssd-deploy-checklist-2026-08-19.md`（CONDITIONAL-GO、P1-P8 前置、回滚分层 Tier-1/2、阶段2+3 合并窗口建议——本次全部采纳）
- Tessa（测试专家）：`deliverables/engineering-assurance/kvssd-perf-test-plan-2026-08-19.md`（G-4 复测方法、54 组合校正、读回验证——本次全部执行）
- Docu（技术文档师）：`deliverables/engineering-assurance/kvssd-200g-runbook-update-2026-08-19.md`（变更记录 + Runbook 增量 + Grafana 面板）
- 实机取证：01 节点 KV 文件分析（/tmp/inspect_kv*.py）、G-1 验证（/kvtest/verify_g1.py）、G-4 复测（/tmp/g4_recheck.py、g4_readback.py）
- 归档：`delivery/kvssd-offload-2026-08-18/kvpatch/`（io.py 补丁 + 验证脚本 + 变更脚本）
- 历史基线：`_archive_scratch/bench_B/summary_B.json`（B0）

---

> 本报告由工程保障团队 AI 协作生成，关键决策（门禁定档、停机窗口授权）已经用户确认，实测数据以生产节点取证为准。
