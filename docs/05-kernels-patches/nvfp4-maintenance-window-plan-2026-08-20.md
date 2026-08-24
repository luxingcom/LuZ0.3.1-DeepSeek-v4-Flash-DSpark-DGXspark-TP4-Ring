# NVFP4 双算子停机窗口执行方案（一次做对）

> 日期：2026-08-20 下午 | 依据：`nvfp4-kernels-delivery-final(1).zip`（用户交付）+ `entrypoint_activation_plan.md` + 生产实测
> 状态：**执行中**（权重全量转换进行中；本方案为停机窗口内 Step0~5 的执行蓝图）

---

## 〇、三项决策（用户 2026-08-20 确认）

1. ✅ **同意生产停机窗口** → 用于干净性能 A/B（routeA vs 真实 B12X）+ entrypoint 注册 + 灰度
2. ✅ **生产权重非 NVFP4** → 已落地高精度转换器，正在**全量生成 NVFP4 专用权重**（保存到磁盘复用，等同生产权重的 NVFP4 版本）
3. ✅ **entrypoint 注册 + 性能测试要求** → 按用户 zip（README/activation_plan/perf_diag/improvement）执行

---

## 一、权重转换（进行中，head 一次性容器）

- **脚本**：`convert_high_precision_nvfp4_stream.py`（流式逐层，43 层，`--mode high` 高精度）
- **产物**：`<INSTALL_DIR>/nvfp4/models/dsv4f-0731-nvfp4-hp/`（等同生产权重的 NVFP4 版）
- **已校验**：layer0 768 矩阵 roundtrip 7e-34；输出格式与 routeA 内核零缝对齐（w1→[4096,1024]uint8+scale[128,16]）
- **收尾（转换完成后）**：四节点 md5 一致性 + 三方备份（01/02/local + md5 校验）

## 二、entrypoint 注册（方式 A：pip 非 -e，绕开 `:ro`）

```bash
pip install --no-deps <INSTALL_DIR>/nvfp4/plugin-src   # 包复制进 site-packages，非 editable
# 验证（三行全过才继续）：
python - <<'PY'
from importlib.metadata import entry_points
eps=[ep for ep in entry_points(group="vllm.general_plugins")]
print("general_plugins:", eps)
assert any("nvfp4" in ep.name for ep in eps)
import nvfp4_vllm_plugin, nvfp4_vllm_plugin.quant_config
print("nvfp4_vllm_plugin import OK")
PY
```
> 备选：B=`site-packages/*.pth`（import 钩子）；C=侵入 api_server（兜底，不推荐）

## 三、数据采集（停机窗口内，空闲 GPU）

### 3a. 权重格式核验（Step 1，5 分钟）
确认 FusedMoE 权重为 NVFP4 打包（uint8 [K,N//2]）；否则 routeA 需转换器（本次已产出）

### 3b. routeA vs B12X 干净 A/B（Phase 0，空闲 GPU，15 分钟）
- **脚本**：`ab_routeA_vs_b12x.py`（已改 routeA 口径 + RouteA 类缓存 W 修复测量失真）
- **口径**：routeA = RouteA 类缓存（GEMM-only）；B12X = 反量化+fp16 matmul（下界）
- **判据**：routeA/B12X prefill ≥ 1.5× 才启用 K1；否则保持默认关

### 3c. perf_diag_fp4_gemm 归因（shape 扫描 + GEMM/量化拆分）
- **脚本**：`perf_diag_fp4_gemm.py --backend cutlass`（需接入真实 routeA 量化，非 zeros 占位）
- **目标**：定位 60~187 TFLOPS 与 350 目标的差距（shape 依赖 / A 量化开销 / W 缓存 / tile）
- **按归因走 P1~P5**：A 量化占比>20%→P1；shape 依赖明显→P3；首调/稳态差大→P2；其余→P4 graph/P5 内核对照

### 3d. 灰度启用（Step 4，先 K2 后 K1，各观察 10 分钟）
```bash
# K2（KV 写回 v17，风险低）
VLLM_NVFP4_K2=1 vllm serve ... --kv-cache-dtype nvfp4_ds_mla   # 观测 128K needle + 写回耗时 -73~79%
# K1（prefill routeA，需 A/B ≥1.5× 后开）
VLLM_NVFP4_K1=1 vllm serve ... --quantization nvfp4_4w4a_sm121  # 观测 prefill 吞吐/TTFT/数值 ≤1%
```

## 四、验收与判据

| 项 | 判据 |
|---|---|
| 转换权重 | 43 层全转 + 4 节点 md5 一致 + 三方备份 |
| entrypoint 注册 | entry_points 含 nvfp4 + import OK + `--quantization nvfp4_4w4a_sm121` 被识别 |
| routeA A/B | vs 真实 B12X ≥1.5×（空闲 GPU，无共享噪声） |
| perf 归因 | 记录 60~187 的 shape 表 + P1~P5 定位结论 |
| K2 灰度 | 128K needle 全绿 + 写回耗时 -73~79% |
| K1 灰度 | prefill 吞吐提升 + 数值 ≤1% |
| 回滚 | `pip uninstall nvfp4-vllm-plugin` / env 关 / 还原备份（≤5 min） |

## 五、当前进展与下一步

- ✅ 转换器落地 + 单层验证通过（roundtrip 7e-34, 格式对齐）
- 🔄 全量 43 层转换进行中（约 21/43 层）
- 🔄 ab 脚本已改 routeA 口径 + 缓存修复（待停机窗口跑）
- [ ] 转换完成后：4 节点 md5 + 三方备份
- [ ] 停机窗口：entrypoint 方式A → 干净 A/B + perf 归因 → 灰度 K2→K1
- [ ] 回填 Runbook + 归档 v3 源码/diff 到 `backup/tp4-<date>/`