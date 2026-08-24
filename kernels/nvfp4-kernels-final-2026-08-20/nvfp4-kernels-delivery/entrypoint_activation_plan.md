# nvfp4-vllm-plugin 最终启用方案（entrypoint 确认 + 停机窗口执行清单）

> 日期：2026-08-20 | 依据：nvfp4-plugin-execution-2026-08-20（生产团队执行记录）
> 现状：插件源码已持久化 `<INSTALL_DIR>/nvfp4/plugin-src/`（挂载 `:ro`）；entrypoint 未注册（`pip install -e` 因只读失败）
> 约束：生产停机窗口稀缺 → 方案必须"一次做对"，含验证与回滚

---

## 一、entrypoint 注册方式确认（三选一，按可靠性排序）

vLLM 0.26.1 插件机制：启动时自动扫描 `vllm.general_plugins` entry point group 并 import 其中模块 → 模块内 `@register_quantization_config` 完成注册。**关键：必须让该 entry point 出现在 vLLM 进程可见的包元数据中**。

### 方式 A（首选）：pip 非 editable 安装 —— 绕开 `:ro`

```bash
# :ro 只影响 -e（editable，需写源码目录）；普通安装把包复制进 site-packages，可行
pip install --no-deps <INSTALL_DIR>/nvfp4/plugin-src
# 或指定目标（若容器 pip 无写 site-packages 权限）：
pip install --no-deps --target "$(python -c 'import site; print(site.getsitepackages()[0])')" \
    <INSTALL_DIR>/nvfp4/plugin-src
```

**验证注册生效**（vLLM 进程内可查）：
```bash
python - <<'PY'
from importlib.metadata import entry_points
eps = [ep for ep in entry_points(group="vllm.general_plugins")]
print("general_plugins entry points:", eps)
assert any("nvfp4" in ep.name for ep in eps), "插件 entry point 未注册！"
import nvfp4_vllm_plugin, nvfp4_vllm_plugin.quant_config
print("nvfp4_vllm_plugin import OK")
PY
```

### 方式 B（备选，无 pip）：site-packages `.pth` 自动 import 钩子

```bash
SP=$(python -c 'import site; print(site.getsitepackages()[0])')
echo "<INSTALL_DIR>/nvfp4/plugin-src"        > "$SP/nvfp4_plugin.pth"   # 加入 sys.path
echo "import nvfp4_vllm_plugin"               >> "$SP/nvfp4_plugin.pth"  # 解释器启动即 import
# .pth 的 import 行在 Python 启动时执行 → vLLM 进程启动即注册（无需 entry_points）
```

> ⚠️ .pth 的 import 行是 site 机制，**vLLM 进程内生效**；验证同方式 A（entry_points 可能查不到，
> 但 `import nvfp4_vllm_plugin` 成功 + `--quantization nvfp4_4w4a_sm121` 被识别即证明注册）

### 方式 C（兜底）：vLLM 入口显式 import（侵入一行）

```bash
# 修改 vllm/entrypoints/openai/api_server.py 头部加一行（停机窗口内可做，需重进镜像验证）
import nvfp4_vllm_plugin  # noqa: F401  ← 在 api_server 所有 import 之前
```
> 不推荐（侵入 vLLM 源码，升级会丢）；仅 A/B 均不可行时使用。

**推荐：方式 A**（pip 非 -e，最干净、可被 entry_points 机制自动加载、回滚 = `pip uninstall`）。

---

## 二、停机窗口执行清单（一次做对）

### Step 0 — 备份（2 分钟）
```bash
cp -r <INSTALL_DIR>/nvfp4 <INSTALL_DIR>/nvfp4.bak-$(date +%Y%m%d-%H%M)
# 并确认 .bak-import-20260820 留档仍在四节点
```

### Step 1 — 生产权重格式核验（执行记录遗留项 2，5 分钟）
```bash
python - <<'PY'
# 确认 FusedMoE 权重是否为 NVFP4 打包（w13_packed/w2_packed），否则 routeA 需转换器
import torch
from safetensors import safe_open
# 以 rank0 的模型目录为例（实际路径按生产）
import glob
f = sorted(glob.glob("<INSTALL_DIR>/models/*/model-*.safetensors"))[0]
with safe_open(f, framework="pt") as sf:
    keys = [k for k in sf.keys() if "moe" in k.lower() or "experts" in k.lower()]
    print("MoE 张量样例:", keys[:5])
    for k in keys[:3]:
        t = sf.get_tensor(k)
        print(f"  {k}: dtype={t.dtype} shape={tuple(t.shape)}")
PY
# 判定：
#   dtype=uint8 且 shape=[K, N//2]  → NVFP4 已打包（routeA 直接用）
#   dtype 含 F8_E8M0 / uint8 + scale  [out, in//32] → MXFP4 原版 → 需跑 convert_high_precision_nvfp4.py
```

### Step 2 — entrypoint 注册（方式 A，5 分钟）
```bash
pip install --no-deps <INSTALL_DIR>/nvfp4/plugin-src
# 验证（见 §一 方式 A 的验证脚本，三行全过才继续）
```

### Step 3 — 注册生效验证 + A/B 性能门槛（停机窗口内，空闲 GPU，15 分钟）
```bash
# 3a 注册验证：--quantization nvfp4_4w4a_sm121 能被识别（vLLM 启动日志出现 Quantization 行）
# 3b 干净 A/B（执行记录遗留项 1）：空闲 GPU 上 routeA vs 真实 B12X kernel
python ab_run.py   # 生产校准后的 A/B 脚本（隔离容器，无共享噪声）
# 判据：routeA/B12X prefill ≥ 1.5× 才启用 K1；否则保持默认关（插件原则）
```

### Step 4 — 灰度启用（先 K2 后 K1，各观察 10 分钟）
```bash
# KV 写回（K2，风险低：v17 已逐字节 A/B 通过）
VLLM_NVFP4_K2=1 vllm serve ... --kv-cache-dtype nvfp4_ds_mla
# 观测：128K needle 召回 + KV 写回耗时（预期 -73~79%）

# prefill（K1，需 A/B ≥1.5× 后才开）
VLLM_NVFP4_K1=1 vllm serve ... --quantization nvfp4_4w4a_sm121
# 观测：prefill 吞吐 / TTFT / 数值一致性 ≤1%
```

### Step 5 — 回滚预案（随时可用）
```bash
# 方式 A 回滚：pip uninstall nvfp4-vllm-plugin
# 开关回滚：VLLM_NVFP4_K1=0 / K2=0（默认即关）
# 终极回滚：还原 Step 0 备份 + 删 plugin-src（执行记录 §七 已确认）
```

---

## 三、验收与判据汇总

| 项 | 判据 |
|---|---|
| entrypoint 注册 | entry_points 含 nvfp4 + import OK + `--quantization nvfp4_4w4a_sm121` 被识别 |
| 权重格式 | NVFP4 打包（uint8 [K,N//2]）→ 直用；MXFP4 → 先转换器 |
| routeA A/B | vs 真实 B12X ≥1.5×（空闲 GPU，无共享噪声） |
| K2 灰度 | 128K needle 全绿 + 写回耗时 -73~79% |
| K1 灰度 | prefill 吞吐提升 + 数值 ≤1% |
| 回滚 | pip uninstall / env 关 / 还原备份（≤5 分钟） |

## 四、与执行记录对齐

- ✅ 执行记录 §五"entrypoint 未启用"→ 本方案 Step 2 解决（pip 非 -e 绕开 :ro）
- ✅ 执行记录 §六 遗留 1（干净 A/B）→ Step 3b（停机窗口空闲 GPU）
- ✅ 执行记录 §六 遗留 2（权重核验）→ Step 1
- ✅ 执行记录 §六 遗留 3（启用路径确认）→ 本方案 §一（方式 A 首选）
- ✅ 执行记录 §六 遗留 4（kv_writer R3 paged）→ 不阻塞启用（torch 索引过渡）
