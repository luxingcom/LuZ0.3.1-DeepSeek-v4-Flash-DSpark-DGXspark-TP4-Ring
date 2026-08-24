# NVFP4 生产集群 —— 容器旧料清理清单 (cleanup-inventory)

> 由 testing 于 2026-08-20 现场盘点 /vllm-workspace 下所有 nvfp4-* 目录。
> **规则**：本清单仅【记录】，未经 team-lead 确认【不删除】任何容器目录。
> 保留中：`nvfp4-delivery-final`（routeA/B/sre 在用）、`nvfp4-landing`（工作区）、`nvfp4-testkit`。

## 待确认清理（历史轮归档，体积小）

| 目录 | 大小 | 修改时间 | 状态 | 说明 |
|------|------|----------|------|------|
| `/vllm-workspace/nvfp4-delivery` | 216K | 2026-08-19 11:03 | ⚠️ 待确认 | 早期交付根（v9/v10 时代），可能被引用，先保留 |
| `/vllm-workspace/nvfp4-delivery-v12` | 300K | 2026-08-19 12:59 | ⚠️ 待确认 | 历史轮归档 |
| `/vllm-workspace/nvfp4-delivery-v13` | 1.6M | 2026-08-19 14:08 | ⚠️ 待确认 | 历史轮归档 |
| `/vllm-workspace/nvfp4-delivery-v15` | 772K | 2026-08-19 16:08 | ⚠️ 待确认 | 历史轮归档 |
| `/vllm-workspace/nvfp4-delivery-v16` | 9.5M | 2026-08-19 16:48 | ⚠️ 待确认 | 历史轮归档（含 cubin/bc，体积最大） |
| `/vllm-workspace/nvfp4-delivery-v17` | 448K | 2026-08-19 16:53 | ⚠️ 待确认 | 历史轮归档（v17 逐字节测试脚本源） |

## 保留（明确在用，不清理）

| 目录 | 大小 | 修改时间 | 说明 |
|------|------|----------|------|
| `/vllm-workspace/nvfp4-delivery-final` | 500K | 2026-08-20 00:01 | 最终交付包（routeA/B/sre 在用） |
| `/vllm-workspace/nvfp4-landing` | — | 2026-08-20 00:56 | 统一落地工作区（routeA/routeB/docs） |
| `/vllm-workspace/nvfp4-testkit` | 192K | 2026-08-19 09:21 | 测试套件 |

## 说明
- 任务原始清单含 `nvfp4-delivery-v11`，但容器内**不存在** v11 目录（v11 时代交付在 `nvfp4-delivery` 根内）。
- 上述 6 个待确认目录合计约 ~12.8M，均为历史轮快照，移除不冲击 final/landing。
- 清理动作待 team-lead 逐项确认后由 testing 执行（先备份后删）。

---
生成环境：vllm-tp4-rank0 (dspark-vllm-gx10:0.2.1-v026.0)，node01(<MGMT_OCTET>)