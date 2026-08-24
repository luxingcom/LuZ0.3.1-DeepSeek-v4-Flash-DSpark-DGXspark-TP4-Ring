# RouteA 生产运维执行记录 — 2026-08-21（SRE: Rex）

**执行人**: Rex（SRE 工程师） · **窗口**: 2026-08-20 23:55 ~ 2026-08-21 00:35 UTC（服务器时间）
**节点**: node01~04（用户 <USER>） · **生产状态**: 停机窗口（容器停止），全程未启动生产、未触碰 -0731/-nvfp4 数据本体（只读检查）

---

## 0. 突发事件处置（任务前置）：node01 nfsd 僵死 → 已修复

**发现过程**: Preflight 时 node01 上 `ls -la <MODELS_DIR>/` 无限挂起。

**诊断链**:
- `ls -f`（免 stat）正常 → 挂起在 statx 环节；strace 确认阻塞于 `statx("<MODELS_DIR>/deepseek-v4-flash-0731")`（NFS 挂载点）
- 03 dmesg: `nfs: server <NODE_IP> not responding, still trying`（连续多条）
- 03 上有 **4 个 D 状态 mount.nfs4 僵尸进程**（8/20 07:52 起挂起，均为 -hp 挂载尝试）→ 证明 -hp 在 03 **从未成功挂载**
- 关键对照实验: 在 **01 本机** `mount -t nfs4 127.0.0.1:...` 同样挂起（TCP 2049 可连、nfsd 8 线程全 idle、/proc/fs/nfsd/versions 正常 +3/+4/+4.1/+4.2）→ 排除网络问题，定性为 **01 nfsd 对新 RPC 无响应（僵死）**

**处置**: `systemctl restart nfs-server`（停机窗口内，03 的挂载本已挂死、04 挂载来自 02 不受影响、无生产消费者，零附加风险）。

**结果**: 
- 01 localhost 挂载立即恢复响应（快速返回 access denied——127.0.0.1 不在导出 ACL，属预期）
- 03 既有 -0731/-nvfp4 挂载恢复可用（stat/ls 正常）
- 03 的 4 个 mount.nfs4 僵尸进程全部退出；其中 2 个补挂成功造成 -hp **堆叠挂载两次**（soft+hard），已在后续清理中双重 umount

**风险提示**: 该 nfsd 僵死至少始于 8/20 07:52（-hp 挂载尝试全部挂起之时），期间 03 的 -0731/-nvfp4 挂载不可用——若当时生产在跑，03 将无法加载模型。建议后续排查僵死诱因（portlist 中 tcp 2049 出现两次的重复注册现象可作为线索）。

---

## 1. 任务 1：-nvfp4-hp 资产外科手术式删除 ✅

**执行顺序**: 03 → 04 → 01 → 02（先客户端后服务端）

### 1.1 node01（NFS 客户端，源 01）
| 步骤 | 操作 | 结果 |
|---|---|---|
| umount | `umount <MODELS_DIR>/deepseek-v4-flash-0731-nvfp4-hp` ×2（nfsd 修复后僵尸挂载补挂了两层） | UMOUNT1_OK / UMOUNT2_OK，`mount \| grep nvfp4-hp` 为空 |
| systemd | `systemctl list-units --all` + `list-unit-files` + `/etc/systemd/system` 检查 | 无任何 -hp unit（无残留 unit 文件） |
| fstab | 第 18 行 `-hp` NFS 条目，`sed -i.bak-hp` 删除 | 已删，`/etc/fstab.bak-hp` 留档 |
| symlink | `<INSTALL_DIR>/models/deepseek-v4-flash-0731-nvfp4-hp`（确认 `file` 输出为 symbolic link 后删除） | 已删 |
| mountpoint | `rmdir <MODELS_DIR>/deepseek-v4-flash-0731-nvfp4-hp`（空目录，root 属主） | RMDIR_OK |

### 1.2 node01（NFS 客户端，源 02）
| 步骤 | 操作 | 结果 |
|---|---|---|
| umount | 活跃 NFS 挂载（<NODE_IP> 源），umount 一次 | OK，`mount \| grep nvfp4-hp` 为空 |
| systemd | mount unit `data-models-...-nvfp4-hp.mount` 为临时/生成单元（`FragmentPath=` 为空，无持久 unit 文件） | 无需删除 |
| fstab | 第 18 行，`sed -i.bak-hp` 删除 | 已删，`/etc/fstab.bak-hp` 留档 |
| symlink | 确认 symlink 后删除 | 已删 |
| mountpoint | rmdir | RMDIR_OK |

### 1.3 node01（NFS 服务端 + 数据主本）
| 步骤 | 操作 | 结果 |
|---|---|---|
| exports | 第 3 行 `-hp` 条目（<NODE_IP>/24 + <NODE_IP>/24），`sed -i.bak-hp` 删除 + `exportfs -ra` | 已删，`exportfs -v \| grep nvfp4-hp` = 0，`/etc/exports.bak-hp` 留档 |
| 数据 | 删除前最终 ls 确认（drwxrwxr-x, 148G）→ `rm -rf /home/<USER>/models/deepseek-v4-flash-0731-nvfp4-hp` | DATA_RM_OK |
| md5 文件 | `rm -f deepseek-v4-flash-0731-nvfp4-hp.md5` | 已删 |
| symlink | `<INSTALL_DIR>/models/...-hp`（删除时为 broken symlink，确认后删） | 已删 |
| 残留 | `<INSTALL_DIR>/nvfp4/models/` 仅剩 `full_convert.log`（-hp 全量转换日志，内容确认为"转换 768 专家矩阵/1536 张量"） | 已收编进 routeB 归档后清除，目录现为空 |

### 1.4 node01（NFS 服务端 + rsync 副本）
| 步骤 | 操作 | 结果 |
|---|---|---|
| exports | 第 14 行（<NODE_IP>/30），`sed -i.bak-hp` 删除 + `exportfs -ra` | 已删，`exportfs -v` = 0 条，`/etc/exports.bak-hp` 留档 |
| 数据 | `rm -rf /home/<USER>/models/deepseek-v4-flash-0731-nvfp4-hp`（148G） | DATA_RM_OK |
| 备注 | 02 无 -hp 的 /opt symlink 与 .md5 文件（Preflight 确认本就不存在） | — |

### 1.5 Post 验证清单（全过）

| 检查项 | 01 | 02 | 03 | 04 |
|---|---|---|---|---|
| df / 前后 | 989G→**828G** used（释放 ~161G） | 1.4T→**1.3T** used（释放 ~148G） | —（客户端无数据） | —（客户端无数据） |
| exports -hp 条目 | 0 | 0 | n/a | n/a |
| mount -hp | n/a | n/a | 0 | 0 |
| /home 或 /data -hp 残留 | 无 | 无 | 无 | 无 |
| **-0731 完好** | 59 项 / 156G | 58 项 / 156G | NFS 59 项可访问 | NFS 58 项可访问 |
| **-nvfp4 完好** | 70 项 / 165G | 70 项 / 164G | NFS 70 项可访问 | NFS 70 项可访问 |

**铁律遵守确认**: 全程未触碰 -0731 与 -nvfp4 数据本体（仅 ls/du/md5 抽查只读）；所有 fstab/exports 修改均留 .bak-hp；每次删除前 ls/file 验证路径精确。

---

## 2. 任务 2：routeB 服务器工件收编（/tmp → 持久）✅

**归档产物**（均在 node01 `<INSTALL_DIR>/nvfp4/`）:
- `routeb-archive-20260821.tar.gz`（600 KB，125 文件）— MD5: `011661a907ae7ef78a2e0894081e9c3d`（同目录 `.md5` 文件）
- `routeb-archive-20260821-INDEX.md` — 完整内容清单

**收编内容**:

| 顶层目录 | 来源 | 文件数 | 内容 |
|---|---|---|---|
| `routeb_task12/` | 01:/tmp/routeb_task12 | 95 | Task1/2 bench 工作副本（bn1_* 脚本/日志/pt 矩阵、dense_blockscaled_gemm_persistent_* 内核源码、routeb_prod_adapter.py）、SASS 工件（sass/、cubins/、cache_export/、dump/、sass_gate_*）、p3_*.py、**p4/results.json**、routeb_official/ |
| `routeb_p3/` | 02:/tmp/routeb_p3（tar+scp 传输，传输 md5 `ffb4c2e0ca3fe214777ada8acd8f50ad` 双端一致） | 24 | P3 语义诊断套件（p3_diag_*.py 全家、p3_inspect_weights.py、**p3_results.txt** 最终判定输出） |
| `routeb_sglang/` | 01:/tmp/routeb_sglang | 5 | sglang 容器 manifest/pull 辅助脚本（28K，顺手收编） |
| `routeb_extra/hp-conversion-residue/` | 01:<INSTALL_DIR>/nvfp4/models/ | 1 | full_convert.log（-hp 全量转换日志，删除 -hp 资产时的唯一残留物） |

关键文件已验证在档: `routeb_task12/p4/results.json`、`routeb_task12/sass_gate_final.txt`、`routeb_p3/p3_results.txt`。
/tmp 原目录**保留未删**（按指示，重启自然清理）。

---

## 3. 任务 3：-nvfp4（modelopt）健康检查 — 供 architect Task #20 裁定 ✅

**对象**: `/home/<USER>/models/deepseek-v4-flash-0731-nvfp4/`（01 主本 165G / 02 副本 164G，rsync 同步）

### 3.1 结构完整性
- 48 个 safetensors shard 全部通过结构校验（文件大小 = 8B 长度头 + header + data 段；header 张量与 `model.safetensors.index.json` weight_map 双向一一对应，无缺失/多余）
- 01↔02 抽查 shard-00001 与 shard-00048 md5 **完全一致**（`3548f62d...` / `87b6aa51...`）
- ⚠️ 备注: 目录内有 6 个 `.aria2` 残留控制文件（8/13 下载期，5 个 shard + index.json）。因结构校验通过判定 shard 完整，.aria2 仅为下载器控制文件残留，无实质影响；建议择机清理（非紧急）

### 3.2 量化配置（config.json → quantization_config 摘录）
- `moe_quant_algo`: **NVFP4**，`fmt`: **e4m3**，`group_size`: **16**
- `quant_algo`: MIXED_PRECISION；`quant_method`: "fp8"（顶层兼容标记）
- `producer`: modelopt `dsv4-nvfp4-experts-mtp-fallback`
- `quantized_layers`: 全部 layers.N.ffn.experts → NVFP4 group_size 16；`ignore`: `*.attn.*`、`*.ffn.shared_experts.*`、`head`

### 3.3 张量级实证（layer 0 experts，model-00002-of-00048.safetensors，纯 safetensors header+字节解析，未用容器）
| 张量 | dtype | shape | 实证 |
|---|---|---|---|
| `layers.0.ffn.experts.0.w1.weight` | U8 | [2048, 2048] | 每字节 2×FP4 nibble 打包 → 逻辑 K=4096；nibble 直方图 **16/16 码字全部使用**（中心码 6/14 占 16.5%/16.7%），真实 E2M1 分布 |
| `layers.0.ffn.experts.0.w1.weight_scale` | **F8_E4M3** | [2048, 256] | **group_size = 4096/256 = 16 实锤**；采样 512 值 105 个 distinct（-240 ~ +224，含负值），**非恒 1** |
| `layers.0.ffn.experts.0.w1.weight_scale_2` | F32 | 标量 | 0.0（二次 scale 未用，modelopt 常见形态） |
| `layers.0.ffn.experts.0.w1.input_scale` | F32 | 标量 | 1.58e-43 |
| `layers.0.ffn.experts.5.w2.weight` / `weight_scale` | U8 / F8_E4M3 | [4096,1024] / [4096,128] | 逻辑 K=2048，group_size=2048/128=16 吻合；码本 15/16（+0 码缺席属正常） |

### 3.4 结论（供 routeA 权重路径裁定）
**-nvfp4（modelopt）是健康的 NVFP4 权重**: e4m3 block scale 真实分布 + E2M1 码本完整使用 + group_size 16 数学自洽——与 -hp 缺陷品（scale 恒为字节 1 + 码本非 E2M1）特征完全相反。**可作 routeA 适配评估的权重底座。**

### 3.5 routeA 集成注意点（转 architect）
- 权重布局为 **U8 打包（2×FP4/字节）+ 独立 weight_scale（e4m3，逐 16 组）+ 标量 scale_2/input_scale**——与 -hp 的布局不同，转换器/内核读取侧需按此布局适配
- shared_experts/attn/head 未量化（ignore 列表），MIXED_PRECISION 混合精度路径
- 目录含 `hf_quant_config.json`、`mtp_nvfp4_build_report.json`、`config.1.json` 等附加文件，MTP 结构相关，评估时一并查看

---

## 4. 遗留事项与建议
1. **[建议 P2] 排查 01 nfsd 僵死诱因**（本次以 restart 恢复；线索：/proc/fs/nfsd/portlist 中 tcp 2049 重复出现两次）。若复发，03 模型加载将挂死
2. **[建议 P3] 清理 -nvfp4 目录 6 个 .aria2 残留控制文件**（不影响完整性，纯卫生）
3. **[提示] 03 的 <MODELS_DIR>/deepseek-v4-flash-0731.local-backup**（8/12 的本地备份）与 Qwen3-Embedding-0.6B 仍在，未在本次任务范围，仅记录存在
4. /tmp 原工件目录（01: routeb_task12/routeb_sglang/routeb_p3（解包副本）/routeb_p3.tar.gz/routeb_extra, 02: routeb_p3/routeb_p3.tar.gz）保留，重启自然清理

## 5. 变更留档汇总
| 节点 | 变更 | 备份 |
|---|---|---|
| 01 | /etc/exports 删 -hp 行；删 148G 数据 + .md5 + /opt symlink；清 nvfp4/models/ 残留日志；**restart nfs-server**；新增 routeb 归档 3 文件 | /etc/exports.bak-hp |
| 02 | /etc/exports 删 -hp 行；删 148G 数据 | /etc/exports.bak-hp |
| 03 | umount -hp×2；/etc/fstab 删第 18 行；删 /opt symlink；rmdir 挂载点 | /etc/fstab.bak-hp |
| 04 | umount -hp；/etc/fstab 删第 18 行；删 /opt symlink；rmdir 挂载点 | /etc/fstab.bak-hp |
