# 全栈持久化完成报告

**日期**：2026-08-03
**工作流**：部署前检查 / 持久化（工作流 4 变体）
**参与成员**：Rex（盘点与补齐）/ 主理人（编排汇编）

---

## 📌 TL;DR（执行摘要）

- **全栈持久化盘点完成**：唯一缺口 = **8003 Responses 网关 nohup 进程无自启**（主机重启不拉起）——**已补齐 systemd 化**
- **补齐后全栈自启机制就位**：vllm-envE / embed-gpu / 25000 FW / 监控栈均 unless-stopped 或 systemd user + linger + docker.service enabled
- **验证通过**：网关 systemd enabled+active + restart 恢复模拟 200；全栈健康（8001/8003/8020/3000/8191 全 200）；远端与归档 md5 一致
- **交付**：全栈持久化清单（重启恢复顺序）+ 网关 systemd 单元归档 hardened/live/
- 严重度：🔴 0 / 🟠 0（网关缺口已补）/ 🟡 0 / 🟢 5（遗留）
- 阻塞 / 非阻塞：**非阻塞（持久化完成）**

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟢 持久化完成 |
| 关键缺口 | 8003 网关 nohup → systemd user 单元（已补齐） |
| 自启机制 | 容器 unless-stopped + systemd user + linger + docker.service |
| 验证 | 全栈健康 + restart 恢复模拟 |
| 建议下一步 | vllm 重启演练（需授权）；运行手册（Docu） |

---

## 📋 盘点矩阵

| 服务 | 节点 | 自启机制 | 状态 | 缺口 |
|---|---|---|---|---|
| vllm-envE-node（E-600k, 8001） | head60 | 容器 unless-stopped + docker.service enabled | ✅ Up 5h healthy | 无 |
| vllm-envE-worker | worker58 | 同上 | ✅ Up 5h healthy | 无 |
| **Responses 网关 8003** | worker58 | **原 nohup 无自启** | ⚠️ → ✅ | **✗ 已补齐 systemd** |
| Qwen3-Embed GPU 8020 | worker58 | systemd user（docker start）+ 容器 unless-stopped + linger | ✅ enabled+active | 无 |
| aicad-fw-25000 | head60 | 容器 unless-stopped | ✅ Up 6h | 无 |
| Grafana/Prometheus/dcgm/node | 双机 | aicad compose 容器 unless-stopped | ✅ 全 200 | 无 |
| docker.service | 双机 | systemd enabled | ✅ | 无 |

## 🔧 补齐动作（8003 网关 systemd 化）

- 单元：`~/.config/systemd/user/responses-gateway.service`（user 单元 + linger）
- Type=simple / WorkingDirectory=~/responses_gateway / ExecStart=venv/bin/python main.py（绝对路径）/ Restart=always / RestartSec=3
- **env 全量固化**：API_KEY（客户）、UPSTREAM_API_KEY（内部）、VLLM_URL=http://<NODE_IP>:8001、EMBED_URL=http://127.0.0.1:8020、SERVED_MODEL/PUBLIC_MODEL、GATEWAY_PORT=8003
- 日志 append:gateway.log（与原 nohup 一致）
- **pkill 兼容性**：旧 nohup 用 `[r]` 方括号技巧清理（避免 ssh 内联 pkill 自杀）；日常 `systemctl --user restart responses-gateway`，**勿再手动 start_gateway.sh**（脚本保留作回退）

## ✅ 验证

- systemd：responses-gateway / embed-qwen3-gpu / docker 均 enabled+active（双机）+ linger=yes
- restart 恢复模拟：网关 systemctl restart → /health 200
- 全栈健康：8003 /health + /v1/models + /v1/embeddings 透传 200、完整 /v1/responses 链路 200、8001（内部 key）200、8020 200、3000/8191/9400/9100 双机 200
- 远端单元 md5 == 本地归档 md5（0630814a...）

## 🔄 全栈持久化清单（重启恢复顺序）

1. docker.service（双机 systemd enabled）
2. 全部 unless-stopped 容器自动拉起（vllm-envE 双机、aicad 全家、embed 容器、fw-25000、exporter）
3. embed-qwen3-gpu（worker58 user 单元 + linger）
4. responses-gateway（worker58 user 单元 + Restart=always；依赖 8001/8020/网络）

**健康判定链路**：8003 /health → 8003 /v1/responses（客户 key）→ 8001 /v1/models（内部 key）→ 8020 → 3000/8191
**回滚**：`systemctl --user stop/disable responses-gateway` 后 start_gateway.sh 还原 nohup

## ✅ 行动清单（按优先级排序）

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | vllm-envE 双机重启恢复演练（低峰窗口，用户授权后 docker restart 验证） | Rex | P1 | 用户授权后 |
| 2 | start_gateway.sh 改为包装 systemctl restart（消除双管理冲突） | Rex | P2 | 下轮维护窗 |
| 3 | 监控双活确认（3000/8191 双机独立 compose，避免数据源双写混淆） | Rex | P2 | 1 周内 |
| 4 | 清理临时 alpine 容器（nice_visvesvaraya/lucid_edison） | Rex | P3 | 顺手 |
| 5 | 正式运行手册（基于 persistence-ledger） | Docu | P2 | 1 周内 |

## ⚠️ 待完善 / 已知局限

- vllm-envE 双机重启未做真机演练（生产约束，需授权低峰执行）
- 旧 8/2 网关系统级单元草稿（缺 UPSTREAM/EMBED env）已被本次 user 单元替代，勿回滚
- head60 linger=no 但无生产 user 单元，无影响

## 📚 数据来源 & 成员产出索引

- Rex：盘点矩阵、网关 systemd 化（env 固化 + pkill 处理）、验证全量、persistence-ledger-20260803.md（归档 hardened/live/）、遗留项 5 条
- 归档：hardened/live/responses-gateway.service、hardened/live/persistence-ledger-20260803.md

---

> 本报告由工程保障团队 AI 协作生成，关键决策（vllm 重启演练授权）请由人类工程负责人复核。
