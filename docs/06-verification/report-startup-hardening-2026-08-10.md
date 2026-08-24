# 启动编排容错与自恢复加固 — 2026-08-10 重启事故复盘

> 状态：加固已落地，TP2 验证中
> 触发：08-10 重启落地新配置时暴露系列启动问题

## ① 事故复盘（重启暴露的 4 个问题）

| # | 现象 | 根因 | 修复 |
|---|------|------|------|
| 1 | head 容器 Restarting(127) | SERVE_CMD 续行行尾加注释 `\   # comment` → 续行符被注释吞掉，`--max-num-seqs` 断裂成独立命令 | 移除 SERVE_CMD 行尾注释；新增自检工具防复发 |
| 2 | docker run 挂载失败 "not a directory" | `sudo bash` 把 `$HOME` 重置为 `/root` → `-v "$HOME/patch-v026/..."` 指向不存在的 /root 路径 | 统一用普通用户 `HOME=/home/<USER>` 执行（<USER> 在 docker 组，无需 sudo） |
| 3 | worker "docker run requires at least 1 argument" | LD_LIBRARY_PATH 行 `'...'  # comment \` → `#` 注释吞掉续行符 → docker run 语句在 LD_LIBRARY_PATH 后结束，无 IMAGE 参数 | 移除该行行尾注释；全脚本扫描同类模式（仅此 1 行） |
| 4 | vllm-cluster.service failed | 编排器 StartLimit（5min/3 次）耗尽后不再重试 | reset-failed + 开机自动重置；编排脚本加双脚本自检 |

**共性根因**：`行尾注释 + 反斜杠续行` 是 bash 头号坑（`#` 后全部是注释，`\` 无效）——head/worker/编排脚本全部受此影响。

## ② 加固清单（已落地）

### 工具：check_vllm_script.sh（四台部署 <INSTALL_DIR>/scripts/）
启动前自检 6 项：
1. `bash -n` 语法
2. **注释吞续行扫描**（`#` 在行尾 `\` 前的行）
3. **行尾反斜杠尾随空格检查**（`\ ` + 换行 ≠ 续行）
4. 关键参数存在性（max-model-len 768000 / max-num-seqs 12 / BREAKABLE=0 / LD_PRELOAD）
5. 依赖文件（libncclpin.so / 模型 config.json）
6. SERVE_CMD 展开长度（<300 判定断裂）+ $HOME 检查（sudo 下 /root 报错）

### 启动脚本（start_head/worker_v026r.sh）
- docker run 前插入**前置自检**（`check_vllm_script.sh "$0"`，失败终止）

### 编排脚本（start_v026r_cluster.sh）
- 挂载预检后插入**双脚本自检**（head 本地 + worker 经 SSH）
- 既有：幂等清理 / trap 诊断 / head-first / 双阶段就绪（TCPStore→API）保持不变

### systemd（vllm-cluster.service）
- failed 状态已 reset；开机自动重置（每次 boot 新 StartLimit 窗口）

### 执行规范（运维铁律，防复发）
- **禁止**在 `\` 续行行尾加注释（用独立 `#` 行）
- **禁止** sudo 执行启动脚本（HOME 重置）——统一 `HOME=/home/<USER> bash script.sh`（<USER> 在 docker 组）
- 脚本改动后必须跑 `check_vllm_script.sh` 自检 + `bash -n`

## ③ 验证状态

- [x] 自检工具 4 台部署 + head/worker 实测 6 项全绿
- [x] head/worker 脚本前置自检插入（SYNTAX OK）
- [x] 编排脚本双脚本自检（SYNTAX OK）
- [x] vllm-cluster.service reset-failed
- [ ] TP2 重启就绪验证（head 8001 + 768K + shim 线程落核）——进行中
- [ ] 基本功能测试（/v1/models + smoke chat）

## ④ 相关文件
- check_vllm_script.sh（本机副本 deliverables/engineering-assurance/）
- start_head_v026r.sh / start_worker_v026r.sh（01/02，含前置自检）
- start_v026r_cluster.sh（01，含双脚本自检）
