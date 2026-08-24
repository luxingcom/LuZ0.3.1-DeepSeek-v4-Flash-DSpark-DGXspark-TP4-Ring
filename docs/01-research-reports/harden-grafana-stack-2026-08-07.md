# Grafana 监控栈持久化加固与残留清理报告

**日期**：2026-08-07
**工作流**：工作流 4（部署前检查/加固）——监控栈收尾
**参与成员**：主理人（实施）

---

## 📌 TL;DR（执行摘要）

- Grafana 面板已满足可用条件（v13：2s 刷新 + 预聚合低开销），完成**持久化 + 稳定性加固 + 残留清理**
- **持久化**：Prometheus `recording_rules.yml` 补入 compose 挂载（:ro），容器重建后规则从宿主加载（不再依赖 docker cp 可写层）；三容器数据卷 + restart=unless-stopped 全部确认
- **清理**：<MGMT_OCTET> `/tmp` 全部 13 个临时面板 JSON（8/1-8/7）+ 工作区 `_scratch*` 清零
- 预聚合指标重建后恢复（四机 4 series），监控栈三容器健康
- 无阻塞项

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟢 通过（持久化/加固/清理全部完成） |
| 阻塞项数量 | 0 |
| 关键行动项 | 3 条（见行动清单） |
| 建议下一步 | 观察 24h 监控栈稳定性；按需归档 <MGMT_OCTET> ~ 目录测试残留 |

---

## 1. 持久化（完成）

| 项 | 处理 |
|----|------|
| **recording_rules.yml 挂载** | ✅ compose（/opt/aicad/docker-compose.yml）prometheus 服务 volumes 补 `./monitoring/recording_rules.yml:/etc/prometheus/recording_rules.yml:ro`；容器重建（docker compose up -d）生效；rule_files 双文件确认 |
| Prometheus 数据 | ✅ aicad_promdata 卷（既有） |
| Grafana 数据（面板/用户） | ✅ aicad_grafanadata 卷 + provisioning + dashboard JSON 挂载（既有） |
| Alertmanager | ✅ 卷 + alertmanager.yml 挂载（既有） |
| compose 变更备份 | ✅ docker-compose.yml.bak-20260807 |
| Grafana 面板 v13 备份 | ✅ <MGMT_OCTET> `~/vllm-dashboard-v13-20260807.json`（15.7KB） |

## 2. 稳定性加固（确认）

| 项 | 状态 |
|----|------|
| restart 策略 | ✅ grafana/prometheus/alertmanager 全部 `unless-stopped` |
| 容器健康 | ✅ 三容器 running |
| 预聚合规则 | ✅ 重建后恢复（dcgx:cpu_util_percent / gpu_temp 各 4 series） |
| 日志轮转 | ✅ <MGMT_OCTET> daemon.json json-file 100m×3（既有） |
| 面板查询低开销 | ✅ 资源面板查 dcgx:* 预聚合（v13） |

## 3. 残留清理（完成）

| 位置 | 清理内容 |
|------|---------|
| <MGMT_OCTET> /tmp | 13 个临时面板 JSON（dash_v9~v13、dash_payload、dsq、dash_check/cluster/fixed/now/realtime 等 8/1-8/7 历史残留）+ p.yml |
| 工作区 | `_scratch*` 全部清零（0 残留） |
| <MGMT_OCTET> ~/backup-images 备份 tar | 保留（vllm 21G + embed 19G，容灾资产） |

## 4. 遗留（未处理，需用户决策）

- <MGMT_OCTET> `~` 目录历史测试残留（`_smoke4000.py`、`_tessa_A_recheck_2026-08-05.txt`、`D/E/F_progress.log`、`_gsm8k_dl`、`__pycache__` 等）——属于用户数据区，按安全规则不主动删除，建议归档或确认后清理

---

## ✅ 行动清单

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | 观察 24h：Prometheus 重建后稳定性、预聚合持续产出、面板 2s 刷新 | SRE | P2 | 24h 后 |
| 2 | 用户确认后归档/清理 <MGMT_OCTET> ~ 目录测试残留（列出清单） | SRE | P3 | 用户决策 |
| 3 | compose 变更已备份（.bak-20260807），下次完整 `docker compose up -d` 验证全栈 | SRE | P3 | 下次维护窗 |

---

## ⚠️ 待完善 / 已知局限

- recording_rules.yml 挂载依赖宿主文件存在（/opt/aicad/monitoring/recording_rules.yml ✅ 已确认）
- Prometheus 重建（compose up -d prometheus）仅影响 Prometheus 单容器，Grafana/Alertmanager 未动
- ~ 目录残留未清理（用户数据区，保守处理）

---

## 📚 数据来源

- docker inspect（挂载/restart/状态）
- compose config -q 校验 + 容器重建日志
- Prometheus API（rule_files 双文件、dcgx:* 四机数据）
- 文件系统核查（/tmp、工作区残留清单）

---

> 本报告由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。
