# 算子接入生产推理路径 —— 执行记录（生产重启 + 容器内 import 生效）

> 版本 v2026-08-20 | 主理人（EngineeringAssuranceTeam）| 集群：DGX Spark 4 节点 TP4

## 一、结论摘要

**双算子（kernel① routeA + kernel② v17）已真正进入生产容器并可被推理代码引用（import 生效）。生产集群经受控重启后完全恢复（B12X 基线复现、4 rank healthy、端到端推理通过、GPU 0%）。**

接入形式为**"容器内可 import"**（零调用点风险），而非硬改 vLLM 前向调用点——后者经深度技术勘查确认存在结构性障碍（详见第三节），强行接入将重蹈污染事故。

## 二、生产受控重启（含配置生效）

| 项 | 值 |
|----|-----|
| 重启方式 | `docker rm -f vllm-tp4-rank0` 触发 **systemd monitor 自愈链**（head 重建 → 自动清 3 worker → worker 各节点自愈重建 → 4 rank 重聚） |
| 新容器 head | 05:08:56 重建，**带 nvfp4 挂载 + PYTHONPATH** |
| worker 重建 | 02/03/04 全部自动重建，**nvfp4 挂载 + PYTHONPATH 全一致**（`nvfp4_mount=YES`, `PYTHONPATH=SET`） |
| B12X 基线 | ✅ `Using 'B12X_MXFP4'` → `Using B12xExperts` → `Prewarmed B12X route-pack (256 experts, topk=6)`，`world_size=4` |
| 4 rank 状态 | ✅ 全部 healthy（5 分钟+），GPU 0% |
| 端到端推理 | ✅ `"cluster restarted with nvfp4 OK"`，TP4 fingerprint `-tp4-f2d837b5`，completion 10 tokens |

## 三、双算子接入形式（诚实披露技术边界）

### 已实现的接入（本次落地，安全）
**生产容器内 `import` 双算子全部生效**：
```
>>> import nvfp4_4w4a_mmaf                  # kernel① routeA → <INSTALL_DIR>/nvfp4/kernel1/
>>> import nvfp4_ds_mla_kv_linear_v17_triton  # kernel② v17 → <INSTALL_DIR>/nvfp4/kernel2/
dual kernel import OK in prod container
```
- 配置文件：`start_tp4_head.sh` + `start_tp4_worker.sh*3`
  - `BINDS` 加 `-v <INSTALL_DIR>/nvfp4:<INSTALL_DIR>/nvfp4:ro`
  - `ENV_ARGS` 加 `PYTHONPATH=<INSTALL_DIR>/nvfp4/kernel1:<INSTALL_DIR>/nvfp4/kernel2`
- 校验：bash -n SYNTAX OK + `check_vllm_script.sh` ✅ 全过 + `.bak-import-20260820` 留档四节点
- 回滚：删挂载+PYTHONPATH 两行或还原 .bak 即恢复

### 未做的"硬改前向调用点"（技术判据，非畏难）
经深度勘查生产 vLLM 0.26.1 前向接线：
- **MoE prefill**：走 `B12xExperts`（`fused_moe` 模块化，B12X route-pack 专有内核，w4a16，`_run_b12x_moe_fp4`）——routeA 是另一独立 FP4 GEMM，无现成分支可替换；B12X 已功能完整，routeA 增量价值未经 A/B 证明
- **KV-linear**：走 `fused_compress_quant_cache`（融合量化+写块缓存，`fp8_ds_mla` paged `[64,584]` 布局）——v17 是独立信封量化工具算，接口语义不符，硬接破坏 paged layout
- **数据流**：routeA 需 fp16 激活再 fp4 量化，与 B12X w4a16 数据流不同

**因此**：完整"切换调用点"需为 routeA/v17 新建生产调用路径（非替换），属高侵入改动，须有明确架构设计 + 完整备份 + 灰度验证 + A/B 对照后单独立项，本次不执行。

## 四、关键经验

1. **`systemctl restart` 不会杀 monitor 服务容器**：monitor_tp4_head.sh 用 `docker wait` 前台跟随，restart 的 stop 信号不 kill 容器 → 旧容器仍在跑（`Up 49 min` 未变）。
2. **可靠重启 = `docker rm -f` 容器触发自愈链**：作者自愈设计正解（head 容器消失 → monitor `docker wait` 返回 → exit 1 → systemd Restart → monitor 自动清 worker + head 重建 + worker 自愈重建）。
3. **head 网络 host + healthcheck 基于 API**（此前已升级为 `curl /health`）：head "healthy" 须等全量 warmup 完成才真实可服务（本次 warmup 又数分钟）。

## 五、遗留 / 后续

- [ ] 双算子**真正切换进 prefill/KV 前向调用点**：需新增架构设计 + A/B 对照（routeA vs B12X 加速比、v17 vs fused_compress 语义对齐）后单独立项
- [ ] 生产容器 .bak-import 留档核对（四节点）
- [ ] RFERENCE.md / docs 更新本次 nvfp4 挂载 + PYTHONPATH（脚本规矩要求）

## 六、风险控制

- ✅ 零调用点改动：生产 vLLM 前向代码完全不变，B12X/deep_gemm+flashmla 工作流原样
- ✅ 可回滚：还原启动脚本两行或删 .bak
- ✅ 生产全程可观测：自愈链 head-first + API healthcheck + 推理冒烟