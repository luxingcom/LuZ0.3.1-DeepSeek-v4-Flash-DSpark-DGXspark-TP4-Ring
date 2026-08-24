# 服务器治理盘点：临时文件/测试副本 与 敏感信息源清单（阶段 C 预研·只读）

- 日期：2026-08-24
- 角色：SRE 工程师（雷克斯）
- 范围：四机 node01~04（SSH 只读）+ 本地 `deliverables/engineering-assurance/`
- 纪律：**只读盘点，未删除/修改任何文件**；跳过 >50MB 大文件（checkpoint/镜像/日志仅看元数据）
- 交付物：本文件（临时文件处置建议表 + 敏感信息源清单），供阶段 D 清理执行 / 脱敏阶段引用

---

## 一、总览

### 1.1 /tmp 占用总览（四机）

| 主机 | /tmp 总大小 | 项目相关目录数 | 空文件(0B) | check_vllm_err.* |
|------|------------|--------------|-----------|-----------------|
| node01 | ~17G | 64 | 231 | 223 |
| node01 | ~455M | 21 | 186 | 178 |
| node01 | ~232M | 20 | 145 | 143 |
| node01 | ~232M | 20 | 147 | 146 |
| **合计** | **~17.9G** | — | **709** | **690** |

> node01 占绝对大头（_routea_work/mini0731 单模型 16G）。

### 1.2 <INSTALL_DIR>/backup 占用（四机）

| 主机 | backup 大小 | 主要项 |
|------|------------|--------|
| node01 | ~52G | vllm027-img 23G / v027-nvfp4-archive 9.8G / vllm027-src 5G / prod-snapshot 3×200M / 2hop-archive 389M / luz031 checkpoints ~41M |
| node01 | ~40G | 同上（vllm027-img 23G / v027-nvfp4 9.8G / vllm027-src 5G / snapshots 3×197M） |
| node01 | ~20M | luz031-bench-checkpoint / retired-scripts / rollback json / scripts-archive |
| node01 | ~20M | 同上 |

---

## 二、临时文件处置建议表

分类说明：
- **[归档]** = 结果数据/报告/可复用工具脚本，建议归入本地 deliverables 对应 `_xxx_assets` 或正式资产目录后再清服务器副本
- **[删除]** = 中间产物/日志副本/安装包/空错误文件，确认无引用后可直接清理
- **[保留]** = 回滚资产/在役配置，暂不动

### 2.1 /tmp 项目目录（按主机关联）

| 主机 | 路径 | 大小 | 内容概览 | 建议 |
|------|------|------|----------|------|
| 01 | `/tmp/_routea_work/` | 16G | `mini0731/model.safetensors` 16G（模型副本）+ 结果 json/log + ir_dump(mlir) + plugin_merged | **[删除]**：模型副本可从正式模型路径重建；结果 json 已由本地 `_routea_work/` 覆盖，其余日志可删。**ir_dump 796K 如需留证先归档** |
| 01 | `/tmp/_fp8_f1/` | 384M | head_bf16/bin 253M + head_fp8_payload 127M + head_scale 4M + f1 json/log + baseline*.py | **[归档]**：f1_*.json + log 是 fp8-f1 结果；.bin 是临时 LM-head 权重副本，结果入库后可删 bin |
| 01 | `/tmp/fi_rebase/` | 348M | flashinfer rebase 源码 + wheels + tar.gz + build | **[删除/部分归档]**：`flashinfer-0.6.16-rebased-experimental.tar.gz`(20M) 已是资产；build/wheels 可删 |
| 01 | `/tmp/nccl-v5-src/` | 295M | NCCL v5 源码 build（复制自 backup/vllm027-src） | **[删除]**：backup/ 下有 5G 同源，/tmp 副本可删 |
| 01 | `/tmp/_ringopt/` | 117M | v5/libnccl*.so 117M + results/18 + logs/115 | **[归档]**：results/*.json 是 ringopt-v5 结果；libnccl.so 是测试构建产物，确认 v5 已固化后可删 |
| 01 | `/tmp/_slowround/` | 102M | irq_percpu.tsv 45M + irq_source.tsv 60M + expD 结果 | **[删除]**：tsv 为诊断采样（大文件），已出报告 `slowround-rootcause`；如需留证先归档 |
| 01 | `/tmp/_luz031/` | 120M | buildctx + Dockerfile + restore/setup 脚本 | **[删除]**：构建上下文；checkpoint 已入 backup/ |
| 01 | `/tmp/_expverdict/` | 8.2M | 73 个子目录 + 45 顶层文件（arm_* forensic.tsv） | **[归档]**：expverdict 结果，本地已有 `_expverdict_assets/` 覆盖，可归档后删服务器副本 |
| 01 | `/tmp/_fp8_qg_toolchain/` | 8.4M | assets/runs/*.json + reference_set + collect_manifest | **[归档]**：质量门禁结果，本地已有 `_fp8_qg_toolchain/` |
| 01 | `/tmp/v18_*.bin` | 4×33M | v18t/v18_186/b/c.bin | **[删除]**：v18 二进制构建副本（backup/ 有 vllm027-src 与 prod-snapshot） |
| 01 | `/tmp/_thr2048_retest/` | 428K | arm_a1/a2/b1/b2 + bench 脚本 | **[归档]**：thr2048 retest 数据 |
| 01 | `/tmp/_bench_luz031/` | 604K | data/(g1_w4a16 json+md) + logs + official + p0 | **[归档]**：luz031 bench 数据；official 包已入 backup/ |
| 01 | `/tmp/nccl-abB-*` | 7×~44K | nccl-abB B0~B4 结果 | **[归档]**：nccl A/B 测试结果 |
| 01 | `/tmp/_mtp_tune/` `/tmp/_w4a4_ext/` `/tmp/_wsdedup_l3/` `/tmp/_thr4096/` `/tmp/_bprime*` `/tmp/_thrst/` `/tmp/_wA/` `/tmp/_ws_dedup/` `/tmp/_fi016/` `/tmp/_ar_opt/` `/tmp/_eugr_ab/` `/tmp/_e2e_stage/` | 每项 50K~370K | 各实验脚本+日志+结果 | **[部分归档]**：结果 json/md 已有本地 `_*_assets` 对应；脚本+日志可删 |
| 01 | `/tmp/evidence-*` `/tmp/sass_evidence` `/tmp/nvfp4-delivery-*` `/tmp/routeb_*` `/tmp/nccl_b1_e2e` | 总计 ~26M | nvfp4/sass/routeb 证据包（含 tgz） | **[归档]**：确认 tgz 已入 deliverables 后可删展开目录 |
| 01 | `/tmp/check_vllm_err.*` | 223×0B | 空错误占位文件 | **[删除]** |
| 01 | `/tmp/*.log` `*.md5` `*.py` `*.sh`（约 150 个散文件） | 各 4K | bench/probe/debug 一次性脚本与日志 | **[删除]**：多为当次探测脚本；关键脚本已在 deliverables/scripts 有版本 |
| 02 | `/tmp/_ringopt/` | 117M | 同 01（libnccl.so + results/logs） | **[归档后删]** |
| 02 | `/tmp/_slowround/` | 146M | irq tsv 大文件 + expD | **[删除]** |
| 02 | `/tmp/_expverdict/` | 8.1M | 72 子目录 | **[归档后删]** |
| 02 | `/tmp/v18_*.bin` `m17_fix.bin` `v18_docker.tgz`(26M) `v18_crates.tgz`(2.2M) `flashinfer-*.tar.gz`(20M) | ~200M | v18 构建产物 | **[删除]**：tar.gz 已是资产，bin/docker/crates 副本可删 |
| 02 | `/tmp/Cargo.toml` `/tmp/Cargo.lock` `/tmp/ragas-build/` | ~160K | rust/ragas 构建 | **[删除]** |
| 02 | `/tmp/check_vllm_err.*` + `backup_err.*` + 0B | 186 | 空错误文件 | **[删除]** |
| 03/04 | `/tmp/_ringopt/`(117M) `/tmp/_slowround/`(87M) `/tmp/_expverdict/`(8.1M) `flashinfer-*.tar.gz`(20M) | ~232M | 同 02 结构 | **[同 02]**：ringopt 归档后删 / slowround 删 / expverdict 归档后删 / tar.gz 资产 |
| 03/04 | `/tmp/check_vllm_err.*` + 0B | 145/147 | 空错误文件 | **[删除]** |
| 全机 | `/tmp/systemd-private-*` `/tmp/.X11-unix` `.font-unix` 等 | 系统 | 系统私有 tmp | **[保留]**（系统自管理） |

### 2.2 <INSTALL_DIR> 测试副本/备份（四机）

| 主机 | 路径 | 大小 | 内容 | 建议 |
|------|------|------|------|------|
| 01/02 | `<INSTALL_DIR>/backup/vllm027-img/` | 23G/台 | vllm027 镜像备份 | **[保留]**：回滚资产，确认 v027 固化后方可评估删除 |
| 01/02 | `<INSTALL_DIR>/backup/v027-nvfp4-archive-20260815/` | 9.8G/台 | nvfp4 归档 | **[保留]**：回滚资产 |
| 01/02 | `<INSTALL_DIR>/backup/vllm027-src/` | 5G/台 | vllm027 源码构建 | **[保留]**：回滚资产 |
| 01/02 | `<INSTALL_DIR>/backup/prod-snapshot-B1-*.tar.gz` | 3×~200M/台 | 生产快照 | **[保留]**：回滚资产 |
| 01 | `<INSTALL_DIR>/backup/2hop-s1-archive-20260817.tar.gz` | 389M | 2hop NCCL 归档 | **[保留]**：回滚资产 |
| 01/02 | `<INSTALL_DIR>/backup/nccl-*-archive*`（多个） | 每项 260~308M | NCCL 各版本归档 | **[保留]**：回滚资产 |
| 01/02 | `<INSTALL_DIR>/backup/luz031-checkpoint-20260823` + `luz031-bench-checkpoint-20260823` | ~41M | luz031 回滚检查点 | **[保留]**：回滚资产（含 secrets_vllm_perms / overlay 备份） |
| 03/04 | `<INSTALL_DIR>/backup/luz031-bench-checkpoint-20260823` | 20M/台 | luz031 bench 回滚 | **[保留]**：回滚资产 |
| 01~04 | `<INSTALL_DIR>/scripts/*.bak*` | 01:63 / 02:78 / 03:61 / 04:61，总 ~556K | start_tp4_head/check_vllm 等脚本历史 .bak | **[谨慎]**：.bak 是回滚锚点；建议仅保留近 7~14 天或关键发布点（如 luz031/b12xgate/r12），更早的归档到 backup/retired-scripts 后可清 |
| 01 | `<INSTALL_DIR>/nvfp4.bak-20260820-1605/` | 420K | nvfp4 旧版回滚 | **[保留]**：回滚资产（确认 nvfp4/ 稳定后可归档） |
| 01~04 | `<INSTALL_DIR>/overlay-mask/` `/overlay-wsdedup/` | 各 20K | 在役 overlay patch | **[保留]**：在役补丁 |
| 01 | `<INSTALL_DIR>/docs/*.bak-preB1` `*.bak-*` | 少量 | docs 回滚副本 | **[保留]**：回滚锚点 |
| 01 | `<INSTALL_DIR>/build/` | 1.9G | 构建产物 | **[删除]**：构建中间产物，可重建 |
| 01 | `<INSTALL_DIR>/results_benchopt_smoke/` `results_bt4096_c6verify/` `results_kvssd_200g/` | 合计 ~140K | 结果目录 | **[归档]**：结果入库后可删 |
| 01 | `<INSTALL_DIR>/verification-logs/` | 2M | 验证日志 | **[保留/归档]**：验证证据，归档后可清 |
| 01 | `<INSTALL_DIR>/archi-test/` | 72K | archi 测试 | **[确认]**：无引用可删 |

### 2.3 汇总

- **可直接删除（纯临时）**：/tmp 空文件 709 个 + check_vllm_err 690 个 + v18 构建副本 ~200M + nccl-v5-src 295M + fi_rebase 大部分 300M + _slowround tsv ~330M + _routea_work 模型 16G + /opt build 1.9G + 各散文件 → **粗估可回收 ≥18.5G**（其中 16G 模型副本 + 1.9G build + 约 700M 其余）
- **需归档后删除（结果数据）**：_expverdict 8.1M×4、_fp8_f1 json、_fp8_qg_toolchain、_ringopt results、_bench_luz031 data、_thr2048_retest、nccl-abB、_mtp_tune/_w4a4_ext/_wsdedup_l3 等结果、evidence/nvfp4/routeb 证据包 ~26M、results_* ~140K
- **需保留（回滚资产）**：backup/ 全部（~92G 两机镜像 + 20M×2）、scripts/*.bak*（近期）、nvfp4.bak、overlay-*、docs .bak、secrets/vllm.env、verification-logs
- **高风险留意**：`<INSTALL_DIR>/secrets/vllm.env`（root 600 权限，含当前密钥，**绝对不能外泄/不得进入任何归档包**）；`luz031-bench-checkpoint` 内的 `secrets_vllm_perms_*.txt` 同样含敏感项

---

## 三、敏感信息源清单（供脱敏阶段）

扫描方式：本地用 Grep 工具扫 `deliverables/engineering-assurance/` 全量；服务器用 `grep -rIl` 扫 `<INSTALL_DIR>/{docs,deliverables,scripts}`（仅文本类扩展名，跳过模型/镜像/大日志）。

### 3.1 敏感项分类与文件命中数（本地 deliverables/engineering-assurance）

| 敏感类型 | 模式 | 命中文件数 | 代表文件（高命中） |
|----------|------|-----------|-------------------|
| sudo 密码 | `<PASSWORD>` | 40 | fix-batch-summary-2026-08-13.md, migration-tp2-nccl-2026-08-08.md, cleanup-restore-roce-test-traces-2026-08-09.md, _w4a4_ext_assets/_wsdedup_l3_assets/_ar_opt/_eugr_ab_assets 脚本, _rex/rex_ssh.py |
| API key A | `<API_KEY>-64b0374c6f2840fe` | 24 | litellm-api-key-manual-2026-08-05.md(7), _tessa_v026_bench_plan.md(7), _tessa_v026_raw_2026-08-05.txt(6), hardened/live/persistence-ledger(3), responses_gateway/README(7) |
| API key B/C | `<KEY_PREFIX_NEW>...` / `<KEY_PREFIX_OLD>...` | 1 / 53 | key-rotate-r12-restart-2026-08-24.md(5), _fix_20260813/key-env-refactor.md, _prde_bn/evidence_run*.sh, _fix_20260813/sre_work/*.txt 大量, v027-nvfp4-acceptance(2) |
| API key（综合） | `<KEY_PREFIX_NEW>\|<KEY_PREFIX_OLD>\|<API_KEY>` | ~90 | 上表各文件 |
| 内网 IP | `192.168.5.x` / `10.20.0.x` / `10.100.x` | ~200 | 分布极广：benchmark-nccl-*、runbook-distribution-*、tp4-service-deployment-guide(25)、report-nccl-tcp-firewall(20)、plan-isolcpus(14)、sglang-nvfp4-arch-design(29)、_raw_audit/arch-verify(21) |
| 单机 IP | `<NODE_IP>~189` | ~120 | 同 IP 段分布 |
| 主机名 | `dgxspark0[1-4]` | ~200+ | 项目主标识，几乎全量文档/脚本 |
| 用户 | `<USER>` | ~160 | ops/server-maintenance-handbook、setup-newnode-env-guide、runbook-*、nvfp4-download-deploy-plan(10)、_audit/audit2-*.txt |
| 内部路径 | `<INSTALL_DIR>` | ~200 | 分布极广：architecture-nvfp4(13)、file-registry-4node(20)、handoff-tp4(15)、sglang-*、tp4-service-deployment-guide(10)、_luz031_official_bench/bench_preflight_backup.sh(20) |
| registry | `<NODE_IP>:5000` | ~120 | runbook-distribution-ops(10)、sglang-nvfp4-arch-design(14)、tp4-service-deployment-guide(5)、deploy-sglang-ngc(5)、_ar_opt/*.sh、_luz031_official_bench/config.env(3) |
| Grafana/Prometheus 端口 | `:3000`/`:9090`/`prometheus` | ~100 | incident-grafana-unreachable(11)、hardened/configs/prometheus-alerts.yml(4)、grafana-*、audit-grafana-dashboard-data(14) |
| secrets/vllm.env 引用 | `vllm.env`/`VLLM_API_KEY`/`GRAFANA_PASSWORD` | ~120 | key-env-refactor.md(16)、key-rotate-r12-restart(13)、_tessa_*、hardened/deploy-profiles/CREDENTIALS-FIX.md(5)、responses-gateway.service |

### 3.2 服务器端命中（<INSTALL_DIR> docs+deliverables+scripts，仅文本）

| 敏感项 | node01 | node01 | node01 | node01 |
|--------|-----------|-----------|-----------|-----------|
| `<PASSWORD>` | 0 | **8** | 0 | 0 |
| `<KEY_PREFIX_NEW>/<KEY_PREFIX_OLD>/<API_KEY>` | 0 | 0 | 0 | 0 |
| `<NODE_IP>:5000` | 45 | 51 | 20 | 20 |
| `<USER>` | 56 | 59 | 14 | 14 |
| `<INSTALL_DIR>` | 135 | 144 | 33 | 33 |
| `vllm.env` | 48 | 56 | 24 | 24 |
| `dgxspark0[1-4]`（docs 内） | — | 64 | — | — |

**服务器端关键发现**：
1. **node01 `<INSTALL_DIR>/docs/.qa-comment-bak-20260817/`（53 个文件，392K）含 8 个带 sudo 密码 `<PASSWORD>` 的脚本**（如 `opt__2hop-s1__*.sh`、`opt__aicad-prod__scripts__v027-test__preflight_v027.sh` 等）。**脱敏优先级最高**：这是唯一在服务器文本目录内明文出现 sudo 密码的位置。
2. 实际密钥值在 `<INSTALL_DIR>/secrets/vllm.env`（root:root 600）中，未被文本扫描命中（符合预期）；但**该文件绝不能进入任何待开源归档包**。`luz031-bench-checkpoint` 内 `secrets_vllm_perms_node01.txt` 亦含相关权限/密钥引用。
3. 服务器 docs/deliverables/scripts 内大量命中 registry IP、用户、内部路径、vllm.env 引用 → 若整目录同步开源，须全量替换。

### 3.3 脱敏处理建议（供阶段 D 参考，本阶段不改文件）

1. **必须掩码/删除（最高危）**：`<PASSWORD>`、三个 API key（`<API_KEY>-64b0...`、`<KEY_PREFIX_NEW>...`、`<KEY_PREFIX_OLD>...`）、`secrets/vllm.env` 内容、`vllm.env` 引用行。
2. **建议替换为占位符**：IP `192.168.5.x`/`10.20.0.x`/`10.100.x` → `192.0.2.x`/`10.0.0.x`；主机名 `dgxspark0N` → `node0N`；用户 `<USER>` → `<user>`；路径 `<INSTALL_DIR>` → `/opt/app`；registry → `registry.example.com`。
3. **替换策略**：批量 sed/grep 替换并建立替换映射表；替换后全量重扫验证零残留；二进制/日志/镜像不在替换范围（不随文档开源）。
4. 本地已开源包 `dgxspark-tp4-open-kit-2026-08-13/` 需独立复核（本次未纳入深度扫描）。

---

## 四、风险提示

- **容量**：/tmp 中 16G 模型副本（_routea_work/mini0731）+ 1.9G build + 镜像备份 23G×2 为主要占用；清理临时项可显著缓解。
- **回滚保护**：scripts/*.bak 系列（63/78/61/61 个）切勿整体清删——需按"最近关键发布点保留、其余归档"策略执行。
- **泄密红线**：sudo 密码与 API key 明文出现位置已定位，脱敏阶段须全量覆盖；任何归档/开源压缩包生成前须先对包内文件重扫。
- **只读承诺**：本盘点未做任何删除/修改；清理执行须在盘点确认与批准后另行派单。

---

*附：本文件为只读盘点交付物，落盘于本地 deliverables/engineering-assurance/task1-server-governance-inventory-2026-08-24.md。*
