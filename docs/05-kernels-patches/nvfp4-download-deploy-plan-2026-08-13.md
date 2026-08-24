# DeepSeek-V4-Flash NVFP4 权重下载与使用方案

**日期**：2026-08-13
**背景**：项目当前使用 deepseek-v4-flash-0731（FP8，156GB，01/02 各一份本地 serving）；目标引入 **NVFP4 权重**以解锁 CUDA 13.2 cuBLASLt NVFP4 加速路径。
**要求**：下载到服务器 01/02，与原版模型同一位置、独立文件夹，**不动原版资料**。

---

## 1. 下载通道实测结论（2026-08-13 实测）

| 通道 | 服务器 01/02 | 本地 Windows | 结论 |
|---|---|---|---|
| huggingface.co | HTTP=000 不通 | 不通 | ✗ |
| hf-mirror.com（API） | 不通 | API 通，**文件下载 307→CDN 超时** | ✗ |
| **modelscope.cn（魔搭）** | **HTTP=302 可达，SDK 1.39.0 可用（02）** | — | ✓ **唯一可行通道** |

> 首选候选 `MJPansa/DeepSeek-V4-Flash-0731-NVFP4`（48 shard 标准格式，0731 版）**在现有网络条件下无法获取**（HF 系文件下载全不通）。可获取的 NVFP4 为魔搭 `FlagRelease/DeepSeek-V4-Flash-nvidia-FlagOS`。

---

## 2. 候选取舍

| 维度 | MJPansa 0731-NVFP4（首选，不可达） | **FlagRelease nvidia-FlagOS（魔搭，可达）** |
|---|---|---|
| 版本 | DeepSeek-V4-Flash-**0731**（最新正式） | nvidia NVFP4（preview 基础，5/28 版） |
| 格式 | 标准 48 shard safetensors | **4× mp4 预分片**（model0~3-mp4，各 75.2GB） |
| 总大小 | ~160GB | **300.8GB**（mp4 冗余较大） |
| vLLM 兼容 | ✓ 实测（2×DGX Spark TP2） | ⚠️ **mp4 为部署优化格式，需冒烟验证**（vLLM 加载可能需转换/特定 load-format） |
| 硬件验证 | 2×DGX Spark | 魔搭 FlagOpen（智源系）发布，部署格式 |

---

## 3. 推荐执行方案（魔搭通道）

### 3.1 目录规划（与原版共存，互不影响）

```
/home/<USER>/models/
├── deepseek-v4-flash-0731/          ← 原版 FP8（156GB，勿动）
└── deepseek-v4-flash-nvfp4/         ← 新 NVFP4（FlagRelease，300.8GB）
                                       下载时 modelscope 解包到此目录
```

- 02 机同结构（`/home/<USER>/models/deepseek-v4-flash-nvfp4`）
- 03/04：NFS 挂载 `<MODELS_DIR>/deepseek-v4-flash-nvfp4` ← 01/02（需更新 exports，见 §5）

### 3.2 下载命令（02 机，后台 + 断点续传）

```bash
# 02 机（已装 modelscope 1.39.0）
nohup python3 -c "
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download('FlagRelease/DeepSeek-V4-Flash-nvidia-FlagOS',
                  local_dir='/home/<USER>/models/deepseek-v4-flash-nvfp4')
" > /home/<USER>/models/nvfp4-download.log 2>&1 &

# 01 机（先装 modelscope）
pip install -U modelscope
# 方案 A（推荐）：01 不重复下载，由 02 rsync 内网分发（200G RoCE/管理网，GB/s 级）
# 方案 B：01 独立下载同命令
```

> 300.8GB 预计耗时：modelscope 国内 CDN 按 50-100MB/s 估 **50-100 分钟**；下载中可通过 `du -sh` / log 监控，断点续传自动支持。

### 3.3 校验

```bash
# 下载完成后对比魔搭 API Sha256（repo/files API 返回各文件 Sha256）
# 全文件 md5/sha256 清单生成并比对 01/02
find /home/<USER>/models/deepseek-v4-flash-nvfp4 -type f -exec sha256sum {} + | sort > nvfp4.sha256
# 01/02 比对一致
```

---

## 4. 权重使用方案（vLLM）

### 4.1 兼容性冒烟（第一步，关键关卡）

```bash
# 02 机单机验证能否加载 mp4 权重（若失败需转换，见 §4.3）
vllm serve /home/<USER>/models/deepseek-v4-flash-nvfp4 \
  --trust-remote-code --tokenizer-mode deepseek_v4 \
  --tensor-parallel-size 1 --load-format safetensors \
  --kv-cache-dtype fp8 --block-size 256 \
  --max-model-len 8192 --max-num-seqs 2 --enforce-eager
```

### 4.2 四机 TP4 启动（若冒烟通过）

```
start_tp4_head/worker.sh 基础上：
--model /home/<USER>/models/deepseek-v4-flash-nvfp4（或软链 <INSTALL_DIR>/models/deepseek-v4-flash-nvfp4）
--kv-cache-dtype fp8 --block-size 256
--speculative-config '{"method":"dspark",...}'（若 NVFP4 保留 MTP）
--gpu-memory-utilization 0.65（沿用基线）
NCCL 配置不变（RING + PEER_HCA 双口）
```

### 4.3 mp4 格式不兼容时的转换路径（预案）

```
NVIDIA Model Optimizer（nvidia-modelopt 容器）：
  modelopt 转换为标准 shard 后 vLLM 加载
或 nvidia/NeMo Converter 工具
或回退：仍用原 FP8（本项目基线），NVFP4 仅作实验分支
```

### 4.4 性能验证（A/B）

```
用项目 bench_matrix（54 组合）：
A 组：现 FP8 0731 基线（已有 R12 数据）
B 组：NVFP4 权重同参数
关键指标：MoE GEMM 是否走 NVFP4 kernel（CUDA 13.2 cuBLASLt）、decode/prefill 对比
注意：NVFP4 解算需 CUDA 13.2+ 镜像（8/7 评估），当前 0.2.1-v026 镜像需确认/升级
```

---

## 5. NFS 与 03/04 同步（本地 serving 架构）

```
03/04 当前：<MODELS_DIR>/deepseek-v4-flash-0731 ← NFS(01/02)
新增：/etc/exports 追加 nvfp4 目录
  <NODE_IP>:/home/<USER>/models/deepseek-v4-flash-nvfp4 (03)
  <NODE_IP>:/home/<USER>/models/deepseek-v4-flash-nvfp4 (04)
03/04: mount 或 /etc/fstab 追加；门禁 GPU-gate 适配
若 NVFP4 仅实验：03/04 可暂不挂，先双机验证
```

---

## 6. 风险与回退

| 风险 | 等级 | 缓解 |
|---|---|---|
| mp4 预分片 vLLM 不兼容 | 🟡 中 | 先单机冒烟；不兼容走 §4.3 转换或回退 FP8 |
| 版本非 0731（preview 基础） | 🟡 中 | 精度/能力可能低于 0731；A/B 实测评估 |
| 300GB×2 磁盘/带宽 | 🟢 低 | 磁盘余量 3.1T/2.5T 充足；后台下载 |
| 下载中断 | 🟢 低 | modelscope 断点续传 + nohup |
| 原版受影响 | 🟢 无 | 独立文件夹，命令不触碰 deepseek-v4-flash-0731 |

---

## 7. 待确认决策（用户）

1. **版本取舍**：接受 FlagRelease（nvidia NVFP4，preview 版，魔搭可达）？还是先解决 HF 通道（提供代理/内网中转）拿 MJPansa 0731？
2. **下载模式**：02 下载 → rsync 01（省外部带宽）vs 双机各自下载（并行快）？
3. **03/04 NFS**：本轮是否同步挂载？

---

## 8. 通道决策补充（16:55 实测后定案）

| 通道 | 实测 | 结论 |
|---|---|---|
| 服务器 → HF / hf-mirror 文件 | 000 不通 | ✗ |
| 本地 → hf-mirror API | ✓ 通 | 仅元数据 |
| 本地 → hf-mirror/HF 文件 | 000 不通（307 后 CDN 超时） | ✗ |
| 本地 Clash 代理 127.0.0.1:7890 | **配置存在但进程未运行**（ProxyEnable=0，无监听） | ⏸ 需用户启动 |
| 魔搭官方 deepseek-ai/DeepSeek-V4-Flash-0731 | ✓ repo/files 确认存在（FP4+FP8 混合，**非 NVFP4**） | 备选（不符 NVFP4 需求） |
| 魔搭 FlagRelease nvidia NVFP4 | ✓ 300.8GB mp4（preview 版） | 用户已否决（坚持 0731） |

**用户决策**：坚持 MJPansa 0731-NVFP4；下载模式 02→01 内网分发；NFS 本轮同步。

**唯一可执行路径**：用户启动本地代理（Clash 7890 或提供其他代理）→ 本地下载 MJPansa（HF/hf-mirror 经代理）→ 校验 SHA256 → **scp 内网传输 02**（/home/<USER>/models/deepseek-v4-flash-nvfp4）→ **rsync 01** → 更新 03/04 NFS exports+fstab 挂载 → vLLM 冒烟 → TP4 A/B。

**待办（用户动作）**：启动代理后告知，本方案即进入执行。

---

## 9. GitCode 探索结论（17:10 定案）

- `ai.gitcode.com/hf_mirrors/MJPansa/...` 页面 200 = **SPA 前端假象**（任意路径返回 200）；raw API 真实查询 = **Project not found**
- GitCode 文件服务全不可用：git clone 403/302、raw.gitcode.com 404、HF 兼容 API 非 JSON、resolve 返回 HTML
- GitCode 可用镜像：官方 0731（FP4+FP8）、nvidia NVFP4（preview）——**MJPansa 未镜像**
- **终局结论**：MJPansa 0731-NVFP4 在所有国内可达渠道（魔搭/GitCode/hf-mirror/其他镜像）均不可达；唯一获取方式 = 外网代理/特殊网络

**用户决策（"等 MJPansa 镜像"）已落地为自动化监控**：每日探测 MJPansa 在魔搭/GitCode/hf-mirror 的可用性，出现即通知并启动下载流水线（§8 编排）。

**当前可立即下载的 0731 版**：魔搭 `deepseek-ai/DeepSeek-V4-Flash-0731`（166.9GB，FP4+FP8 混合 = MJPansa NVFP4 的源权重），如接受 4-bit MoE（MXFP4）路线可先行落地。
