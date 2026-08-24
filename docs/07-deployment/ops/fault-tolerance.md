# 容错方案（Fault Tolerance）v1.5-R12

**日期**：2026-08-13（R12 修订）｜**维护**：Docu｜**目标**：单一故障点（SPOF）最小化 + 每层冗余有回退路径
**R11 变更**：模型数据层由 NFS 双源 → 本地 serving；shim 回滚锚点升级 v8 链。
**R12 变更**：NFS 集中化恢复完成（03→<NODE_IP> 主源 01 / 04→<NODE_IP> 备源 02，nfs4.2 ro hard，挂载点 `<MODELS_DIR>/deepseek-v4-flash-0731`）；fstab + systemd `RequiresMountsFor` 持久化；03/04 软链切回 NFS；`.local-backup` 保留 24h 观察后清理；01/02 nfs-server **保持 enabled 为备源**（勿删）；新增 `shim-deploy.sh` 四机 shim 部署/校验/回滚。

---

## 1. 容错分层总览

| 层 | 冗余机制 | 故障场景 | 恢复动作 |
|---|---|---|---|
| 模型数据 | **NFS 双源集中化（R12）**：01 主源→03、02 备源→04（ro hard nfs4.2）；01/02 本地目录为源，03/04 `.local-backup` 为 24h 观察期兜底 | 主源/NFS 断 | 备源冗余 + `.local-backup` 本地兜底（见 §2/§3） |
| 编排 | TP2 锚点脚本保留（start_v026r_cluster.sh 全程未动） | TP4 全面失败 | TP4→TP2 降级 |
| 补丁库 | `.bak-v2`(NCCL) / `.bak-v7`(shim) / 源码归档 tp4-20260812 | 补丁库异常 | 锚点覆盖回退 |
| 网络 | RoCE 环网（无交换机）+ 管理网双平面 | 单网段故障 | 控制面走管理网；环邻 2 路物理链路 |
| 容器 | `--restart no` + systemd 自愈（含互杀守卫） | 容器崩溃 | monitor 自动重建（见 self-recovery.md） |
| 镜像 | 注册表 02:5000 + 历史 tag | 镜像损坏 | 拉取历史 tag |
| 网关 | litellm 02:4000 并发 12；每日备份 cron | 网关故障 | 备份恢复 + 直连 8001（降级访问） |

## 2. 模型数据现状（R12 实测）与 NFS 恢复

- **当前**：软链 → 01/02 = 本地 `/home/<USER>/models/deepseek-v4-flash-0731`；03/04 = `<MODELS_DIR>/deepseek-v4-flash-0731`（**NFS 挂载点**）。NFS 双源挂载生效：03 `mount | grep nfs4` 见 <NODE_IP>、04 见 <NODE_IP>（均 nfs4.2 ro hard，`RequiresMountsFor` 随 systemd 持久化）。
- **回退路径（NFS 断）**：03/04 软链可切回 `<MODELS_DIR>/deepseek-v4-flash-0731.local-backup`（本地兜底，保留 24h 观察期；NFS 稳定后由 SRE 执行清理并删除软链回退项）。01/02 nfs-server 双源 enabled，互为备源。
- **维护检查**：`mount | grep nfs4` 双源在位 + `ls <MODELS_DIR>/deepseek-v4-flash-0731/config.json` + 软链指向 + NFS 加载计时（重启后实测 <270s）。

## 3. 备源切换演练（R12 已生效）

```bash
# 模拟主源故障：03 卸载 01 源
ssh node01 "sudo umount <MODELS_DIR>/deepseek-v4-flash-0731"
# 验证 04 侧备源正常
ssh node01 "ls <MODELS_DIR>/deepseek-v4-flash-0731/ && du -sh <MODELS_DIR>/deepseek-v4-flash-0731"
# 恢复主源
ssh node01 "sudo mount -a; mount | grep nfs4"
```
> 兜底：03/04 本地 `.local-backup` 副本在位（观察期）；NFS 断时 `ln -sfn <MODELS_DIR>/deepseek-v4-flash-0731.local-backup <INSTALL_DIR>/models/deepseek-v4-flash-0731` 即可切回本地 serving。

## 4. 回滚锚点矩阵（权威：rollback-anchors-2026-08-12.md）

| 锚点 | 位置 | 用途 |
|---|---|---|
| NCCL 补丁库 v2 | 四机 `/opt/nccl-ringonly/libnccl.so.2.30.7.bak-v2` | ring-only 回退 |
| shim 库 **v7**（v8 前） | 四机 `<INSTALL_DIR>/lib/libncclpin.so.bak-v7` | 隔离核绑定回退（v8→v7） |
| 网络快照 | 01 `<INSTALL_DIR>/backup/ring-fix-20260811/<host>/` | netplan/hosts/rules.v4 回退 |
| 补丁源码 | 01 `backup/tp4-20260812/`（P0 禁删） | 补丁重建唯一依据 |
| TP2 容器 | 01 `backup/tp2-node.json`、02 `backup/tp2-worker.json` | TP2 还原 |
| TP4 容器 | 四机 `backup/rollback_tp4-rank{0-3}.json` | 每次启停覆写（脚本生成） |

## 5. 镜像冗余

- 生产镜像 tag `0.2.1-v026.0` 在注册表 02:5000（<NODE_IP>），四机已本地缓存。
- 历史 tag：`<NODE_IP>:5000/...:0.1.1`、`0.2.1-v026.0`（34.2GB，01 已清副本）→ 镜像层在 02/58 仍有备份，可重新 pull。
- **原则**：镜像不落 /tmp；清理须走 SOP-四机镜像清理与整理-20260807.md。

## 6. 容错缺口（风险登记）

1. **单点**：head(01) 承担 TCPStore + 编排 + 主 NFS 源（恢复后）——01 整机故障 = 全集群不可用（TP4 架构性限制）。
2. QoS 无持久化：重启后 RoCE 拥塞控制回退默认 → 需手动 mlnx-qos-setup.sh（P2）。
3. rules.v4 固化依赖人工执行 iptables-save-custom.sh；漏跑则白名单丢失 → 建议 cron。
4. **NFS 依赖风险（R12 回归）**：03/04 推理权重回归依赖 NFS 双源——主源断时 03 侧硬挂载（hard）会阻塞 I/O，靠 04 备源 + `.local-backup` 兜底；恢复路径见 §3。
