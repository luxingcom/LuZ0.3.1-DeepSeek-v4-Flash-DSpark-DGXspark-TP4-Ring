# KV 缓存 SSD 卸载统一 200G 磁盘缓存 —— 部署执行清单与风险核验报告

- **日期**：2026-08-19
- **作者**：Rex（SRE 工程师，工程保障团队）
- **变更主题**：TP4 生产集群（4×DGX Spark，vLLM 0.26 定制镜像，deepseek-v4-flash-0731，600K token 上下文）KV 缓存 SSD 卸载生产变更：
  1. 容器内 `vllm/v1/kv_offload/tiering/fs/io.py` 卷挂载覆盖补丁（去重 + trim + zstd-3 压缩）；
  2. 四节点磁盘缓存配额统一为 **200G/节点**（01/02 由 800G 重建为 200G，03/04 维持 200G），并修正 **配额实际未生效** 的 bind mount 偏差。
- **目标**：磁盘落盘从 382KB/token 降至 ≤10KB/token。
- **交付物性质**：**分析与核验文档，不含任何实机执行**（所有 SSH/挂载/重启命令由主理人/执行人执行；本文提供命令级要点与门禁标准供执行引用）。
- **状态**：🟡 **CONDITIONAL-GO**（见 §1，满足 8 项前置条件后进入单一维护窗口执行）

---

## 📌 TL;DR（执行摘要）

1. **整体评级**：🟡 **有条件 GO**。前置健康/备份/wheel 固化已就绪；"配额未生效"偏差（19G KV 写在根分区、配额镜像形同虚设）确认属实，bind mount 修正是正确且可逆的补救，不构成阻断项——**但它把变更从"纯容器内补丁"扩大为"宿主机存储拓扑变更"，必须纳入同一重启窗口**。
2. **核心结论（SRE 判断）**：
   - ⚠️ **阶段 2 与阶段 3 必须合并为单一停机窗口**：bind mount 修正仅容器重启后才对容器生效；清根分区残留需容器停止；两次重启 = 两次不必要停机。
   - ⚠️ **命令顺序纪律**：容器停止 → 清 `/opt/aicad-kvssd/*`（根分区残留）→ 建 bind → 写 fstab → 重启容器。顺序颠倒会导致 19G 残留被挂载点隐藏、根分区永久占用。
   - ⚠️ **回滚双备份存在歧义**：需明确 Tier-1（`.bak-kvssd200g-20260819-100628`，仅去补丁，保留 kv-transfer-config）为主回滚目标；`.bak-kvoffload-20260819-075524` 为深度回滚（可能整体关闭 SSD offload，600K ctx 下内存压力大，慎用）。
   - ✅ **补丁旧版本需单独留存**：原版 `io.py` 应从容器内复制归档 + 补丁文件 md5 固化，用于 diff 与 G-3 核验。
   - 🎯 **最大残余风险**：03/04 内存仅 6G available，重启 + zstd + 模型加载存在 OOM（SEV1）；800G→200G 重建中断导致开机挂载失败（SEV1）。两者均有明确缓解（见 §4）。
3. **建议下一步**：满足 §1 的 8 项前置条件 → 约定维护窗口 → 按 §3 阶段 2+3 合并执行 → G-4 冒烟 → G-5 benchmark（建议独立低谷窗口）。

---

## 1. Go / No-Go 决策卡片

### 1.1 决策结论

| 项目 | 结论 |
|------|------|
| **整体评级** | 🟡 **CONDITIONAL-GO（有条件进入变更）** |
| 变更必要性 | ✅ 高。当前磁盘写放大 38x（382 vs ≤10 KB/token），且 KV 数据落在**根分区**（配额未生效），存在根分区被写满、拖垮整机的长期风险——**本次变更同时修复该隐患**。 |
| 变更可逆性 | ✅ 主变更（补丁+脚本）通过恢复 `.bak` 脚本即可回滚（RTO ≈ 5–15 min）；宿主机 bind/fstab/镜像变更可逆（umount + 删 fstab 行 + 恢复 .bak 镜像）。 |
| 变更爆炸半径 | ⚠️ 中偏高。除容器内补丁外，还含宿主机存储拓扑修正（bind mount + 800G 镜像重建），须在**单一维护窗口**内执行。 |
| 前置就绪 | ✅ 健康/SSH/备份/wheel 固化已确认；⚠️ 以下 8 项前置条件需在执行窗口前逐项打勾。 |

### 1.2 前置条件（进入窗口前必须全部满足 → 任一不满足则 NO-GO）

| # | 前置条件 | 核验命令/方式（执行人） | 不满足后果 |
|---|----------|------------------------|-----------|
| P1 | 容器运行用户与 G-2「目录 0700」权限可调和 | `docker exec <ctr> id -u -n`；确认 UID=0（root）或按 UID 调整 `chown` | KV 写路径被拒 → 容器起不来 / 落盘失败 |
| P2 | 根分区余量 ≥ 850G（含 800G .bak + 模型 ~79GB + 日志余量） | `df -h /` | 重建镜像中途写满 → 根分区故障 |
| P3 | 容器内 Python = 3.12 且 aarch64，与 wheel 匹配 | `docker exec <ctr> python3 --version; uname -m` | zstd import 失败 → 补丁无压缩能力 |
| P4 | 容器内 `io.py` 真实路径与卷挂载目标路径一致（非 symlink/拼接） | `docker exec <ctr> readlink -f <容器io.py路径>` + `python -c "import ...io as m; print(m.__file__)"` | 补丁静默未生效（挂错路径） |
| P5 | 补丁 `io.py` 通过 `python -m py_compile` 且经 code-reviewer 审签 | 宿主机 `python -m py_compile <INSTALL_DIR>/kvpatch/io.py`；code-reviewer 签核 | 语法错 → 容器启动崩溃；逻辑错 → G-4 不达标 |
| P6 | 回滚双备份定级完成（Tier-1/Tier-2）并核对 md5 | `md5sum` 比对 `.bak-kvssd200g-20260819-100628` 与 `.bak-kvoffload-20260819-075524`；与实测 md5（a6b82155/9689b6d5/254a3c7c）一致 | 回滚时选错目标 → 过度回滚或无法恢复 |
| P7 | fstab 两行均加 `nofail`（loop 行 + bind 行） | 编辑 `/etc/fstab` 后 `mount -a` 冒烟 | 重建窗口内意外重启 → 开机卡挂载 / 节点起不来 |
| P8 | 维护窗口与相关方确认（网关 4000/8003 将短暂 502；容器 `--restart no`，停即停机） | 与主理人/业务确认窗口；预置人工拉起路径 | 停机影响未沟通 → 事故式上线 |

> **判定规则**：P1–P8 全部 ✅ → **GO（执行）**；≤2 项 ⚠️ 且非 P1/P3/P5 → **CONDITIONAL-GO（先补齐再进窗口）**；P1/P3/P5 任一不满足或 P2 根分区不足 → **NO-GO（先修前置）**。

### 1.3 对"配额未生效偏差"修正方案的风险评估（重点）

- **偏差确认**：`/opt/aicad-kvssd` 为根分区普通目录（inode 2369881，/dev/nvme0n1p2），与 `/mnt/kvssd-quota`（loop 挂载，inode 2）非同文件系统 → 容器 KV 写入并未落盘到配额镜像，**800G/200G 配额形同虚设**，19G KV 数据占根分区。判定：偏差属实，且为**当前生产隐患**（根分区写满风险），修复具有正收益。
- **修正方案评估**：bind mount（`/mnt/kvssd-quota` → `/opt/aicad-kvssd`）+ fstab 持久化 —— **方案正确、可逆、低风险**：
  - 正确：docker `-v /opt/aicad-kvssd:/opt/aicad-kvssd:rw` 在容器启动时绑定宿主机路径；宿主机 `/opt/aicad-kvssd` 一旦指向 loop，容器即落盘到 200G 配额。
  - 可逆：`umount /opt/aicad-kvssd` + 删 fstab bind 行即回退（代价：回到"写根分区"的现状 bug）。
  - **关键纪律（方案成败所在）**：
    1. **必须容器停止后再操作**：运行中容器持续向 `/opt/aicad-kvssd` 写 KV，清残留与建 bind 存在竞态。
    2. **清残留必须先于建 bind**：先 `rm -rf /opt/aicad-kvssd/*` 清根分区 19G，再 `mount --bind`（否则残留被挂载点隐藏、根分区永久占用）。
    3. **bind 必须先于容器重启**：bind 仅在容器重启后对容器可见。
    4. **fstab 加 `nofail`**：防止 loop/bind 未就绪导致开机挂起。
  - **残余风险**：容器若以非 root 运行，G-2 的 0700 权限会挡住 KV 写（P1 覆盖）；fstab 顺序（loop 行在前、bind 行在后）需确保，否则开机 bind 源未就绪。

---

## 2. 变更范围与已核验事实（采信主理人实地取证）

| 项 | 状态 | 说明 |
|----|------|------|
| 节点连通 | ✅ | 四节点 SSH 全通；TP4 生产健康（01:8001/health HTTP 200，容器 Up 2h healthy） |
| 启动脚本 | ✅ | `start_tp4_head.sh`(01) + `start_tp4_worker.sh`(02/03/04)；已带 `--kv-transfer-config`（OffloadingConnector + TieringOffloadingSpec，cpu_bytes_to_use=2147483648，fs root_dir=/opt/aicad-kvssd，读写线程各 4）；容器 `-v /opt/aicad-kvssd:/opt/aicad-kvssd:rw` |
| 镜像配额 | ⚠️ 偏差 | 01/02 = 800G（loop19，/mnt/kvssd-quota，已用 19G）；03/04 = 200G（196G）；fstab loop 行已写 |
| **关键偏差** | 🔴 | `/opt/aicad-kvssd` = 根分区普通目录（非挂载点，inode 2369881 属 /dev/nvme0n1p2），与 `/mnt/kvssd-quota`（loop，inode 2）**非同文件系统 → 配额未生效**，19G KV 数据占根分区 |
| zstd wheel | ✅ | zstandard-0.25.0-cp312-manylinux2014_aarch64，md5 f235b7d611e1e5fa4afb6208d8d63863，已固化四节点 `<INSTALL_DIR>/envs/zstd/` |
| 脚本备份 | ✅ | `.bak-kvssd200g-20260819-100628`（01 head md5 a6b82155、02 head 9689b6d5、02/03/04 worker 254a3c7c） |
| 内存 | ⚠️ 偏紧 | 01/02 available 11G；**03/04 available 6G**；CPU 主层 2GiB；swap 15Gi |
| 容器策略 | ⚠️ | `--restart no` → 停止即停机，重启完全依赖启动脚本（人工编排） |
| 回滚锚点 | ⚠️ 待定级 | 任务原文指向 `.bak-kvoffload-20260819-075524`；本文建议以 `.bak-kvssd200g-20260819-100628` 为 Tier-1（见 §5） |

> 本文所有门禁（G-2/G-3/G-4/G-5）与命令均基于以上事实；若执行中发现与上述不符，**先停**并回报主理人。

---

## 3. 分阶段执行检查清单

> 使用说明：执行人逐项打勾并回填证据（命令输出/时间戳）。**本文不执行任何命令**；命令供执行人引用。

### 3.0 阶段 0/1 前置确认（衔接他人产出，作为进入窗口的输入核验）

- [ ] P1–P8（§1.2）全部满足，主理人确认 **GO**
- [ ] code-reviewer 已对 `io.py` 补丁出具审查结论（无 Must-Fix）
- [ ] 补丁 `io.py` 已放 `<INSTALL_DIR>/kvpatch/io.py`，宿主机 `python -m py_compile` 通过，md5 已记录
- [ ] 原版 `io.py` 已从容器复制归档（`<INSTALL_DIR>/kvpatch/original/io.py`），与补丁形成 diff 基线
- [ ] zstd wheel 四节点 md5 复核 == f235b7d611e1e5fa4afb6208d8d63863
- [ ] 回滚 Tier-1/Tier-2 定级完成（§5），两份 .bak md5 与已知值一致
- [ ] 变更前 benchmark 基线已采集或明确接受"131072 组记录待定档"

### 3.1 阶段 2：配额重建 + bind mount 修正（G-2）

**⚠️ 建议：本阶段与阶段 3 合并到同一停机窗口执行（见 §6）。以下顺序以合并窗口为前提。**

**2a. 停机前核验（容器仍运行）**

- [ ] `df -h /` 根分区余量 ≥ 850G（P2）
- [ ] `findmnt /mnt/kvssd-quota` 确认 loop 挂载；`findmnt /opt/aicad-kvssd` 预期无输出（确认偏差现状）
- [ ] `lsof /mnt/kvssd-quota | head` 无进程占用（容器当前不写该路径，仍双确认）
- [ ] 记录容器状态与内存基线：`docker ps`、`free -h`（供 G-3 内存增量比对）
- [ ] 通知相关方：网关 4000/8003 对 TP4 的请求将 502；维护窗口开始

**2b. 停止容器（四节点）**

- [ ] `docker stop <ctr>`（01/02/03/04）——容器 `--restart no`，停止即维持停止，人工掌控
- [ ] `docker ps` 确认全部退出，TP4 已不可用（预期停机）

**2c. 01/02 配额重建 800G → 200G**

```bash
umount /mnt/kvssd-quota
mv <INSTALL_DIR>/kvssd-images/kvssd.img <INSTALL_DIR>/kvssd-images/kvssd.img.bak-kvssd800g-20260819
truncate -s 200G <INSTALL_DIR>/kvssd-images/kvssd.img   # 新建 200G 稀疏文件
mkfs.ext4 -q -F <INSTALL_DIR>/kvssd-images/kvssd.img
mount /mnt/kvssd-quota        # 读 fstab loop 行，含 nofail
```

- [ ] `df -h /mnt/kvssd-quota` 预期 ≈196G
- [ ] 挂载后 `findmnt /mnt/kvssd-quota` 显示 loop 挂载、fstype=ext4
- [ ] 若中途任何一步失败 → 按 §5 Host-state 回滚恢复 800G .bak 镜像，**不要带病继续**

**2d. 全节点：清根分区残留（容器已停，先于 bind）**

- [ ] `ls -la /opt/aicad-kvssd/ | head` 确认当前内容（约 19G KV 缓存）
- [ ] `rm -rf /opt/aicad-kvssd/*`（KV 缓存可再生成，删除可接受）
- [ ] `mkdir -p /opt/aicad-kvssd` 确保挂载点存在
- [ ] `df -h /` 复核根分区余量回升（回收 ~19G）

**2e. 全节点：bind mount 修正 + fstab 持久化**

```bash
mount --bind /mnt/kvssd-quota /opt/aicad-kvssd
# fstab 追加（位于 loop 行之后；两行均加 nofail）：
# /mnt/kvssd-quota /opt/aicad-kvssd none bind 0 0
```

- [ ] `findmnt /opt/aicad-kvssd` 显示 bind 挂载
- [ ] `stat -c '%i %n' /opt/aicad-kvssd` inode=2（loop 根，证明已切换文件系统）
- [ ] `df -h /opt/aicad-kvssd` ≈196G
- [ ] 权限：`chmod 700 /opt/aicad-kvssd && chown root:root /opt/aicad-kvssd`（**前提 P1：容器以 root 运行**；否则按容器 UID 调 chown）
- [ ] `mount -a` 冒烟通过（验证 fstab 语法无误，无报错）

**🛑 G-2 门禁（四节点逐项核验，任一失败不得进入阶段 3）**

- [ ] 01/02/03/04 `df -h /opt/aicad-kvssd` 均 ≈196G
- [ ] 01/02/03/04 `findmnt /opt/aicad-kvssd` 均为 bind 挂载
- [ ] 01/02/03/04 `stat -c '%a %i' /opt/aicad-kvssd` == `700 2`
- [ ] 01/02/03/04 fstab 含 bind 行且两行均带 `nofail`；`mount -a` 无报错

### 3.2 阶段 3：补丁灰度挂载 + 4 rank 重启（G-3）

**3a. 单节点预验证（01，throwaway 容器，不占生产）**

- [ ] 确认容器路径（P4）：
  `docker exec <ctr> python -c "import vllm.v1.kv_offload.tiering.fs.io as m; print(m.__file__)"` 已执行并记录
- [ ] throwaway 验证（不加载模型，仅验证 import 与补丁生效）：

```bash
docker run --rm \
  -v <INSTALL_DIR>/kvpatch/io.py:<容器io.py路径>:ro \
  -v <INSTALL_DIR>/envs/zstd:<INSTALL_DIR>/envs/zstd:ro \
  <生产镜像> bash -c "pip install --no-index --no-deps <INSTALL_DIR>/envs/zstd/*.whl \
    && python -c 'import zstandard; print(zstandard.__version__); \
                   import vllm.v1.kv_offload.tiering.fs.io as io; print(io.__file__); \
                   assert io.__file__ == \"<挂载容器路径>\"'"
```

- [ ] `zstandard.__version__` 打印 0.25.0，无 import 异常
- [ ] `io.__file__` 指向挂载路径（证明补丁已覆盖）
- [ ] 补丁自检函数/日志探针（若有）通过；语法/逻辑无异常

**3b. 启动脚本补丁化（四节点）**

- [ ] 启动脚本追加：`-v <INSTALL_DIR>/kvpatch/io.py:<容器io.py路径>:ro` + `-v <INSTALL_DIR>/envs/zstd:<INSTALL_DIR>/envs/zstd:ro`（**确认 envs/zstd 已挂入容器**，否则 pip install 看不到 wheel）
- [ ] entrypoint 前追加：`pip install --no-index --no-deps <INSTALL_DIR>/envs/zstd/*.whl`（`--no-index --no-deps` 杜绝外网拉取）
- [ ] 改动后的脚本另存新版本，原脚本以 `.bak-kvssd200g-20260819-100628` 保持不动

**3c. 全量 4 rank head-first 重启**

- [ ] **01 head 先**：`bash <INSTALL_DIR>/scripts/start_tp4_head.sh`
- [ ] **02/03/04 worker 后**：`bash <INSTALL_DIR>/scripts/start_tp4_worker.sh`
- [ ] GPU-gate：≤180s 内完成 ring 初始化；对端门禁通过
- [ ] 观察 `docker logs` 无 zstd/io.py/TieringOffloadingSpec 相关 error

**🛑 G-3 门禁（任一失败 → 触发 §5 回滚）**

- [ ] `curl -s -o /dev/null -w '%{http_code}' http://01:8001/health` == 200
- [ ] `docker logs <ctr> 2>&1 | grep -i TieringOffloadingSpec` 有输出（kv-transfer-config 生效）
- [ ] `docker logs <ctr> 2>&1 | grep -i 'io.py\|patch\|dedup\|trim\|zstd'` 确认补丁加载日志（**建议补丁内打印一行自证日志**，使本项可判定）
- [ ] `docker exec <ctr> python -c "import zstandard; import vllm.v1.kv_offload.tiering.fs.io as io; print(io.__file__)"` 指向挂载路径
- [ ] `free -h` 对比 §3.2-2a 基线：**内存增量 ≤3GB**（尤其 03/04，available 仅 6G）
- [ ] `dmesg` 无 OOM-kill 记录

### 3.3 阶段 4：存储效率复测（G-4）

- [ ] 记录基线：变更前已实测 382KB/token（问题复现口径）
- [ ] 受控触发落盘：发起 ≥16K ctx 长上下文请求（建议 32K–64K，覆盖多次换入换出），观察 `/opt/aicad-kvssd` 增长
- [ ] `du -sh /opt/aicad-kvssd` 记录请求前后增量；结合生成 token 数计算 bytes/token
- [ ] 优先采用指标口径：集群 dashboard 已有 bytes/token 类指标则直接读取，并记录指标名

**🛑 G-4 门禁**

- [ ] **bytes/token ≤ 10KB**（相对 382KB 基线改善 ≥ 38x；若 ≥20x 但未达 10KB，记 C-2 待定档，不阻断）
- [ ] 落盘路径确认仍为 `/opt/aicad-kvssd`（loop 200G），根分区无新增 KV 数据
- [ ] 请求正确性：响应内容与变更前一致（无质量回退）
- [ ] 采样 ≥3 次长上下文请求，结果稳定（非单次噪声）

### 3.4 阶段 5：全量 benchmark（G-5）

- [ ] 组合：ctx {512, 4096, 16384, 65536, 131072} × task {coding, json, prose} × conc {1, 3, 5} = **54 组合**
- [ ] 统一口径：随机前缀强制、温度统一（沿用集群方法学规范，避免 prefix-cache 假象）
- [ ] 每组记录：TTFT（P50/P95）、TBT/ITL、吞吐（token/s）、bytes/token
- [ ] **131072 组 TTFT**：与变更前基线对比，劣化 ≤10%；不满足则**记录待定档（C-3）**并附数据
- [ ] 全量结果回填 benchmark 报告（交付物交叉引用）

**🛑 G-5 门禁**

- [ ] 54 组合全部完成；无正确性错误
- [ ] 131072 组 TTFT 劣化 ≤10%，或已正式记录待定档（C-3）并同步主理人定夺
- [ ] 无 OOM / 无 ENOSPC / 无 zstd 异常

---

## 4. 风险清单与缓解（≥5 条，带 SEV 评级）

> SEV 口径：SEV1=服务宕机/全员不可用；SEV2=主要功能降级；SEV3=次要功能问题；SEV4=低影响。附加「发生概率」参考。

| # | 风险 | SEV | 概率 | 描述与触发 | 缓解措施 |
|---|------|-----|------|-----------|----------|
| R1 | **bind mount 顺序/持久化失败**：清残留后建 bind、bind 后重启的顺序破坏，或 fstab 无 `nofail` 导致开机挂起 | **SEV1** | 中 | 运行中容器持续写 `/opt/aicad-kvssd`；顺序颠倒 → 19G 残留被挂载点隐藏、根分区永久占用；fstab 无 nofail → 窗口内意外重启节点起不来 | ① 容器停止后再操作（§3.2-2b）；② 严格顺序：清残留→bind→fstab→重启；③ fstab 两行加 `nofail`；④ `mount -a` 冒烟；⑤ 恢复手册：umount bind + 删 fstab 行即回退 |
| R2 | **800G→200G 镜像重建中断**：umount/mv/truncate/mkfs 中途失败或节点重启，`kvssd.img` 损坏 → 开机挂载失败 | **SEV1** | 低-中 | mkfs 进行中重启、磁盘写满、误删 | ① **先 mv 保留 .bak 再 truncate**，重建成功前不删 .bak；② fstab `nofail` 兜底；③ 中断恢复手册：umount → 恢复 .bak 为 kvssd.img → mount；④ 窗口内禁止非计划重启 |
| R3 | **03/04 内存仅 6G，重启 + zstd + 模型加载 OOM** | **SEV1** | 中 | 内存紧张叠加压缩缓冲区；OOM-kill 容器 → TP4 宕 | ① G-3 内存增量 ≤3GB 硬门禁；② 重启全程监控 `free -h` + `dmesg`；③ 利用 15Gi swap 兜底；④ 提前释放非关键内存占用；⑤ 预案：OOM 即回滚脚本重启 |
| R4 | **补丁格式/加载缺陷**：`io.py` 语法错或挂载路径错 → 启动崩溃，或补丁静默未生效（bytes/token 仍 382KB） | SEV2 | 中 | 语法错误、容器内路径是 symlink、`__pycache__` 冲突 | ① 宿主机 `py_compile`；② P4 路径核验（readlink/import __file__）；③ throwaway 容器单节点预验证；④ 补丁内加**自证日志行**使 G-3 可判定；⑤ 未生效时 G-4 直接暴露 → 回滚 |
| R5 | **zstd wheel 加载失败**：pip install 失败（离线/依赖）或 Python/架构不匹配 | SEV2 | 低-中 | cp312 wheel 与容器 Python 不符、`pip` 尝试联网 | ① `--no-index --no-deps` 本地安装；② P3 预检 Python 3.12 + aarch64；③ md5 复核；④ G-3 import 验证；⑤ 补丁对 zstd import 失败需**优雅降级**（打日志跳过压缩，服务不挂） |
| R6 | **压缩 CPU 开销**：zstd-3 在 20 线程 GB10 上与 decode/attention 争 CPU，拉高 decode 延迟 / TTFT | SEV2 | 中 | 高并发（conc 5）+ 大 ctx 时压缩为瓶颈 | ① zstd-3 为快速档，先观察；② 读写线程 4 各保持可调；③ G-5 观测 TTFT/TBT，CPU 利用率；④ 备选回退：zstd-1 或关闭压缩（仅改补丁/参数，不需镜像） |
| R7 | **回滚方案缺口**：双 .bak 歧义 + 脚本回滚不覆盖宿主机 bind/fstab/镜像变更 | SEV2 | 中 | 回滚时恢复错备份，或回滚后容器仍指向 bind 造成"以为回退了其实没有" | ① §5 定级 Tier-1/Tier-2；② 脚本回滚 + Host-state 回滚分离文档化；③ 回滚后必须 `findmnt /opt/aicad-kvssd` 复核实际状态 |
| R8 | **4-rank 重启失败（NCCL init hang）**：head-first 顺序/时序竞态，历史上有 init hang 前科 | **SEV1** | 中 | ring 形成失败、TCPStore 时序 | ① 严格 head-first 顺序（01 先，02/03/04 后）；② GPU-gate ≤180s 硬门禁；③ 沿用已知良好启动脚本序列；④ 失败即回滚重启；⑤ 抓 `NCCL_DEBUG` 日志 |
| R9 | **200G 配额写满（ENOSPC）**：压缩后仍接近 200G 上限 → 落盘失败 | SEV3 | 低 | 极端长会话/大并发堆满缓存 | ① G-4/G-5 观察 `df /opt/aicad-kvssd` 水位；② 监控告警 ≥80%；③ 明确缓存可清（`rm -rf /opt/aicad-kvssd/*` 容器停后） |
| R10 | **benchmark 归因污染**：TTFT 劣化源于磁盘读压缩块噪声而非补丁 | SEV3 | 中 | 131072 ctx 单样本抖动 | ① 变更前基线先行；② 多采样取 P50/P95；③ 131072 组"劣化>10%"记为待定档（C-3）而非立即回滚，由主理人定夺 |

---

## 5. 回滚方案核验

### 5.1 现有方案是否充分？——**结论：基本充分，但存在 2 处必须补齐的缺口**

| 维度 | 评估 |
|------|------|
| 服务恢复能力 | ✅ 足够。脚本回滚 + 重启 TP4 可在 5–15 min 恢复服务 |
| 回滚粒度 | ⚠️ 需明确。当前存在两份 .bak，任务原文指向旧版 `.bak-kvoffload-20260819-075524`，存在**过度回滚**风险 |
| 覆盖范围 | ⚠️ 缺口 1：脚本回滚**不覆盖**宿主机 bind mount / fstab / 镜像重建（这些是 host-level 变更），需单独 Host-state 回滚手册 |
| 补丁版本 | ⚠️ 缺口 2：补丁卷挂载是只读 overlay，原版 `io.py` 天然在镜像内可恢复，但**未显式归档**。见 5.3 |

### 5.2 建议的回滚分层（执行前定级并核对 md5）

| 层级 | 目标 | 操作 | RTO | 适用场景 |
|------|------|------|-----|----------|
| **Tier-1（主回滚，推荐默认）** | 仅去本次补丁，保留 kv-transfer-config（SSD 卸载继续开） | 恢复 `.bak-kvssd200g-20260819-100628`（含 --kv-transfer-config、无补丁挂载、无 zstd install）→ 重启 TP4 → 缓存可留可清 | 5–15 min | G-3/G-4 不达标、补丁导致崩溃、压缩 CPU 回归 |
| **Tier-2（深度回滚，慎用）** | 整体关闭 SSD 卸载（回到 kv-transfer-config 之前） | 恢复 `.bak-kvoffload-20260819-075524` → 重启 | 5–15 min | SSD 卸载本身故障；**注意：600K ctx 无 offload 时 121GB 统一内存可能放不下，有 OOM/降级风险**，须由主理人明确授权 |
| **Host-state 回滚（非服务恢复，供完整撤销存储拓扑）** | 撤销 bind/fstab/镜像变更 | `umount /opt/aicad-kvssd`；删 fstab bind 行；如需恢复 800G：`umount /mnt/kvssd-quota && mv kvssd.img.bak-kvssd800g-20260819 kvssd.img && mount /mnt/kvssd-quota` | 10 min | 仅在需完整回到变更前存储拓扑时 |

> ⚠️ **待主理人确认**：`.bak-kvoffload-20260819-075524` 是否真的是"SSD 卸载之前"的状态？若是，Tier-2 在 600K ctx 下内存可行性需先评估。建议以 `.bak-kvssd200g-20260819-100628` 为默认 Tier-1。

### 5.3 补丁版本回滚是否需要单独保留旧 io.py？——**需要，且应在变更前归档**

1. **原版 `io.py`**：镜像内天然可恢复（只读 overlay 不改镜像层），但**应显式复制归档**到 `<INSTALL_DIR>/kvpatch/original/io.py`：
   - 提供 diff 基线（后续补丁演进、复盘）；
   - 防止镜像被重建/GC 后丢失参照。
2. **补丁 `io.py`**：`<INSTALL_DIR>/kvpatch/io.py` 记录 md5，作为 G-3 核验锚点。
3. 回滚 Tier-1 后，`<INSTALL_DIR>/kvpatch/` 目录保留但不再被挂载；**不删除**，便于二次灰度。

### 5.4 回滚触发条件（建议固化）

- G-3 任一门禁失败（health ≠200 / 无 TieringOffloadingSpec / 内存增量 >3GB / OOM）
- G-4 bytes/token 无改善（仍 >100KB/token）或出现正确性错误
- 生产报障：容器崩溃、502、decode 延迟显著劣化（> 基线 20% 持续 10 min）
- 决策时限：任何 SEV1 场景 → **立即回滚，不等待完整诊断**

---

## 6. 部署窗口建议

### 6.1 核心建议：阶段 2 + 阶段 3 合并为**单一停机窗口**

| 理由 | 说明 |
|------|------|
| bind 修正仅重启后生效 | bind mount 是宿主机变更，运行中容器持有旧挂载引用，必须重启才看到新文件系统 |
| 清残留需容器停止 | 运行中容器持续写 `/opt/aicad-kvssd`，清残留/建 bind 存在竞态 |
| 减少停机次数 | 两次重启 = 两次停机 + 两次 NCCL init 风险暴露 |
| 单一回滚点 | 合并后"失败=恢复 .bak 重启"一次完成，无需判断部分完成状态 |

### 6.2 建议的时间线

```
[窗口 A：主变更 + 冒烟，预留 2h（含回滚余量）]
 0:00  窗口开始，通知相关方
 0:05  §3.2-2a 停机前核验（基线采集）
 0:10  四节点 docker stop
 0:15  01/02 配额重建（umount→mv→truncate→mkfs→mount）
 0:25  全节点清残留 + bind + fstab + G-2 核验
 0:35  §3.3-3a throwaway 单节点预验证（01）
 0:45  脚本补丁化 + 全量 4 rank head-first 重启
 1:00  G-3 门禁核验（health/日志/内存）
 1:10  G-4 快速冒烟（≥3 次长上下文请求）
 1:40  若全部通过 → 窗口 A 结束，恢复对外声明
 2:00  窗口 A 截止（未通过 → 进入回滚流程）

[窗口 B：全量 benchmark，建议独立低谷窗口，预留 2–4h]
 执行 §3.4 阶段 5（54 组合），131072 组按 G-5 门禁判定
```

### 6.3 优化要点

- **不要"先重建 01/02 再等"**：03/04 虽已 200G，仍需 bind 修正；四节点应在同一窗口一次性完成 bind + 重启，避免 01/02/03/04 状态不一致。
- **throwaway 预验证可提前到窗口 A 之前**（不占生产）：用 `docker run --rm` 验证 zstd import + 补丁加载，**提前消除 P3/P4/P5 风险**，使窗口 A 只承担"重启+门禁"。
- **G-5 benchmark 独立窗口**：54 组合含 131072 ctx，耗时长且需无业务干扰；若必须与窗口 A 合并，需提前与业务方确认接受服务占用。
- **建议固化监控**（变更后 24h）：`/opt/aicad-kvssd` 磁盘水位告警 ≥80%、内存 available 告警、bytes/token 指标纳入 dashboard，防 200G 写满复发。

---

## 7. 待主理人确认事项（Open Questions）

1. `.bak-kvoffload-20260819-075524` 的准确内容？是否为"SSD 卸载前"状态？Tier-2 回滚在 600K ctx 下的内存可行性？
2. 变更前 54 组合 benchmark 基线是否已采集？若无，G-5"131072 组劣化 ≤10%"缺少对照，需先行补采或明确接受"记录待定档"。
3. 容器运行 UID（P1）：若为非 root，G-2「目录 0700」需调整为容器 UID 可写。
4. 补丁是否已内置"自证日志行"（如 `[kvpatch] io.py loaded dedup+trim+zstd3`）？建议补充，否则 G-3"补丁加载"判定依赖 import 路径核验。
5. 变更窗口时段是否已与业务方约定？网关 4000/8003 在停机期 502 的接受度。

---

## 附：变更预期收益与成功判据

| 指标 | 变更前（现状） | 目标 | 门禁 |
|------|---------------|------|------|
| 磁盘落盘 bytes/token | 382 KB | ≤10 KB（≥38x） | G-4 |
| 缓存文件系统 | 根分区（配额未生效） | loop 200G bind 挂载，四节点统一 | G-2 |
| 131072 ctx TTFT | 基线待确认 | 劣化 ≤10% 或记录待定档 | G-5 |
| 内存增量 | – | ≤3GB/节点 | G-3 |
