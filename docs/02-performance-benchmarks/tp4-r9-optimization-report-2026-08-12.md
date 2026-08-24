# TP4 优化落地轮总报告（r9：A/B 验证·参数固化·存储集中化·清理归档）

**日期**：2026-08-12
**参与成员**：Archi（副本 A/B）、Rex（固化/存储/清理）、Docu（归档）、Zhen（汇编）
**状态**：✅ 本轮全部落地，生产稳定；遗留 4 项（P2）

---

## 📌 TL;DR（执行摘要）

- **副本 A/B 验证三项全过**：①72 截断修复 decode TPOT conc=1 **7.4s→0.43s（17 倍）**（fix72/prefix_off 复现一致）②Prefix KV 命中率 **99.4%**、TTFT 冷→热 **-97.5%** ③graph 四组合无非法访问（#30044 排除）。
- **B3 参数固化生产生效**：capture-size 80 + capture-sizes 含 72/80、prefix-caching、long-prefill 1024、retention 4096——引擎实测 PIECEWISE 13/13、FULL 25/25、dspark 21/21。
- **B2 存储集中化完成**：03/04 权重改 NFS 双源挂载（01→03、02→04，RoCE），加载 182-212s（NFS +60-80%，已重新基线），24h 稳定后删本地备份（回收 ~312G）。
- **配置修订**：400k（原生支持，YaRN 1048576）、litellm 并发 12、util 0.60 确认为 400k×12 显存地板。
- **清理归档**：473G 回收 + 回滚锚点手册 + 测试归档。

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| A/B 验证 | fix72 17×（TPOT 7.4s→0.43s）；Prefix KV 99.4% 命中/-97.5% TTFT |
| 固化参数 | capture 80+72、prefix 全套——生产已生效 |
| 存储 | NFS 双源分挂上线，03/04 释放 156G/台（24h 后删备份） |
| 配置 | 400k 原生支持、并发 12、util 0.60=地板 |
| 清理 | 473G 回收 + 归档手册 |
| 遗留 | P2×4（24h 删备份、aot 评估、01 本地 worker 复查、131072 全矩阵） |

---

## 📊 A/B 验证（副本 8002，util 0.60，400k）

| 组合 | conc=1 TPOT | conc=6 | conc=12 | Prefix 命中 | graph 非法访问 |
|---|---|---|---|---|---|
| StepA（仅 max80） | 7405ms ⚠️污染 | 1418 | 2377 | - | 无 |
| **fix72**（capture 含 72） | **431ms** | 1551 | 2641 | - | 无 |
| prefix_off | 431ms | 1409 | 2506 | 0 | 无 |
| **prefix** | - | - | - | **99.4%** | 无 |

**三项通过**：
1. **72 截断修复**：显式 capture_sizes 含 72 → 稳态 batch=72 走 CUDA graph 快路径；fix72 vs prefix_off 431=431 复现（StepA 7405 受崩溃循环污染，已标注）
2. **Prefix KV**：35072/35275 tokens 命中（99.4%）；TTFT 冷 29.66s→热 0.75s（-97.5%）；retention 4096 生效；对 decode 无影响（纯 prefill 收益）
3. **graph 稳定**：四组合无 Illegal memory access / CUDA error

**排查经验**：capture-sizes nargs='+' 空格分隔；副本并发启动防 TCPStore 死锁；util 对齐生产（0.55 触发 KV 不足）；--log-level 不支持改 VLLM_LOGGING_LEVEL。

## 🔧 固化参数（B3，生产生效）

```bash
--max-cudagraph-capture-size 80
--cudagraph-capture-sizes 1 2 4 8 16 24 32 40 48 56 64 72 80
--enable-prefix-caching --enable-prompt-tokens-details
--long-prefill-token-threshold 1024
VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096
```
校验键 B1 无需改；四机脚本 md5=c815072c；引擎实测 capture 覆盖 PIECEWISE 13/13、FULL 25/25、dspark 21/21。

## 📦 存储集中化（B2）

| 项 | 状态 |
|---|---|
| NFS 双源 | 01 export→03 挂（<NODE_IP>）；02 export→04 挂（<NODE_IP>）；ro,hard,intr,noatime,timeo=50,vers=4.2 |
| 加载计时 | 03=181.9s、04=212.4s（NFS +60-80% vs 本地 111-117s）——已重新基线（维护窗口可容忍） |
| 完整性 | 01/02 源 75 文件一致（02 为有效备源）；本地备份 .local-backup 保留 |
| 待办 | **24h 稳定后 sha256 比对 → 删 03/04 本地备份（回收 ~312G）** |
| 回滚 | 卸载 NFS + 改名回本地 即恢复；备源 02 可切 |

## ✅ 其他落地

- **400k 配置**：原生支持（YaRN 1048576）；KV 1.2M tokens/机、满长并发≈3；**util 0.60 = 400k×12 显存地板**（0.55 即 KV 分配失败）
- **litellm 并发 12**：default_max_parallel_requests=12（--num_workers 2 → 上限 ~24）
- **清理 473G**：dspark 旧副本 312G + 构建残留 104G + backup-images + 旧 tag；**B4** 26 个旧 tag、**B5** worker 脚本统一
- **归档**：《回滚锚点与降级引用手册》rollback-anchors-2026-08-12.md + 测试索引 README + 临时文件清理
- **prefix 命中快速实测**：928-token 前缀第 2 请求 cached 768（82.7%）——生产链路正常

## ✅ 遗留（P2）

| # | 项 | 预期 |
|---|-----|------|
| 1 | 24h 稳定后删 03/04 本地权重备份（回收 ~312G） | 8/13 上午 |
| 2 | aot 组（BREAKABLE=0+AOT）单独立项评估 | 另行 |
| 3 | fix72 conc=6/12 略高于 StepA（+9%）核实（噪声 vs 图池开销） | 下轮 |
| 4 | 131072 全矩阵（首轮后固化配置复测） | 下轮 |
| 5 | 02 dspark 0.1.1 镜像 18.8G 删除确认 | 常规 |

---

## 📚 数据来源

- Archi：A/B 结果表 + 固化清单 + 排查经验（~/archi-test-runs/）
- Rex：固化落地（引擎实测）、NFS 切换（加载计时/回滚）、配置修订、清理 473G
- Docu：回滚锚点手册、测试归档

> 本报告由工程保障团队 AI 协作生成（2026-08-12），关键决策请由人类工程负责人复核签字。
