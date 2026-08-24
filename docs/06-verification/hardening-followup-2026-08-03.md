# 加固遗留项补全报告（①autotune 缓存持久化 ②25000 口限制）

**日期**：2026-08-03
**工作流**：安全加固后续（工作流 4 变体）
**参与成员**：Rex（实施与自检）/ 主理人（编排）

---

## 📌 TL;DR（执行摘要）

- **① flashinfer autotune cache 持久化完成**：缓存卷扩为整 `/root/.cache/vllm`，冷启动 **325-340s**（省 ~36-45s vs 370s），autotune 复用实证（rank 0/1 cache loaded、0 次 nvcc 编译）
- **② 25000 master 口限制完成**：vLLM 侧绑定不可行（torch TCPStore 忽略 host 绑定 wildcard，已实证）→ **iptables 数据面白名单**（FW_25000 链：数据面/回环 ACCEPT + 其余 DROP）+ sidecar 容器持久化（无 sudo 等效方案）；验证 worker OPEN / 管理网 BLOCKED，回滚演练通过
- 自检 8 项全绿；生产 E-800k 服务正常（8001 内部 key 认证保持）
- 阻塞 / 非阻塞：**非阻塞（完成）**

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟢 完成（①②闭环） |
| ① 冷启动 | 325-340s（vs 370s 基线） |
| ② 限制方式 | iptables 数据面白名单（sidecar 持久化） |
| 遗留建议 | sudoers 免密后可用 iptables-persistent 级固化 |
| 回滚 | 脚本 .bak_v32 + rollback.sh（已验证） |

---

## ① flashinfer autotune cache 持久化

- **现状确认**：容器 /root/.cache/vllm 仅 3 子目录（deep_gemm 8.9M / flashinfer_autotune_cache 24K / modelinfos 16K），无 torch/triton 大缓存（compilation mode=NONE），磁盘无风险（head 3.0T / worker 2.5T 空闲）
- **变更**：`-v $HOME/vllm-cache/deep_gemm:...` → `-v $HOME/vllm-cache:/root/.cache/vllm:rw`（双机 BINDS 一致）
- **预置**：docker cp flashinfer_autotune_cache + modelinfos 出容器，chown 1000:1000（248 文件）
- **验证**：cold start 325-340s；`FlashInfer SM120 sparse MLA DSv4 decode autotune cache loaded on rank 0/1`；`Config cache hit for sparse_mla_sm120_decode_dsv4 (Loaded 24 configs)`；deep_gemm cache mtime 未变、0 次 nvcc

## ② 25000 master 口监听限制

- **方案评估**：a) vLLM/torch 绑定 MASTER_ADDR——**不可行**（实测 TCPStore host=<NODE_IP> 仍监听 `:::25000` 通配符，tcputil::listen 忽略 host）；b) **iptables 白名单——采用**
- **实施**（head 机，无宿主 sudo → 特权 host-network 容器操作宿主 netfilter）：
  - 镜像 aicad-fw:1.0（alpine + iptables/ip6tables）
  - FW_25000 链：ESTABLISHED,RELATED ACCEPT → 127.0.0.0/8 ACCEPT → <NODE_IP>/24 ACCEPT → DROP；INPUT 首条跳转；ip6tables 镜像（::1 ACCEPT→DROP）
  - **持久化**：sidecar 容器 aicad-fw-25000（--restart unless-stopped --network host --privileged），宿主重启后 docker 自动拉起恢复规则（等效持久化；无免密 sudo 无法写 /etc/iptables/rules.v4）
- **验证**：worker(<NODE_IP>)→25000 OPEN（分布式正常）；管理网源 <NODE_IP> → BLOCKED；head 本机回环/数据面 OPEN；回滚演练通过（rollback.sh → 管理网恢复 OPEN → 重新应用）

## ✅ 自检表（全绿）

| # | 项 | 结果 |
|---|---|---|
| 1 | 容器 healthy | ✅ 双机 Up healthy |
| 2 | /v1/models（内部 key） | ✅ 200，max_model_len=800000 |
| 3 | chat | ✅ 200 "pong" |
| 4 | 冷启动 | ✅ 325-340s（<370s） |
| 5 | 25000 可达性 | ✅ 数据面 OPEN / 管理网 BLOCKED（v4+v6） |
| 6 | rdma QP | ✅ 双链路 RTS/MIGRATED |
| 7 | deep_gemm | ✅ 0 次 nvcc，cache 未变 |
| 8 | flashinfer autotune | ✅ rank 0/1 cache loaded |

## ⚠️ 遗留建议（非阻塞）

1. iptables-persistent 级固化：为主机 <USER> 配置 iptables-restore NOPASSWD sudoers 或由 root 部署 /etc/iptables/rules.v4
2. fw 规则纳入部署清单（加固状态可审计）
3. head 管理网 SSH 不可达（现走数据面跳板）——运维网络拓扑备注

## 📚 数据来源 & 成员产出索引

- Rex：① 卷扩容 + 预置 + 重启验证（325-340s、autotune cache loaded 证据）② a 方案实证（TCPStore wildcard）、b 方案实施（aicad-fw 镜像/FW_25000 链/sidecar）、验证与回滚演练、自检 8 项
- 回滚：`~/start_{head,worker}_E.sh.bak_v32`、`~/fw/rollback.sh`、`docker rm -f aicad-fw-25000`

---

> 本报告由工程保障团队 AI 协作生成，关键决策（sudoers 配置、规则审计）请由人类工程负责人复核。
