# Embed 服务 GPU 切换交付报告

**日期**：2026-08-03
**工作流**：部署前检查（工作流 4）
**参与成员**：Rex（切换实施与验证）/ 主理人（编排汇编）

---

## 📌 TL;DR（执行摘要）

- **embed 服务已切 GPU 模式**（复用 Anemll 镜像容器，满足"复用现有 LLM 运行环境"）；8020 端口与网关 8003 透传不变
- **显存受控**：`set_per_process_memory_fraction(0.04)` 硬上限 5.22GB（负向测试确认强制生效）；常规负载仅 1.25GB reserved；极端 8192×8 被拦截保护 E-600k
- **性能跃升**：batch16 **472 条/s**（vs CPU 3.36 条/s，**快 ~140×**）——**10 万条 embedding 从 CPU 8.8h → ~3.5min**（batch16）
- **关键修正**：fp16 加载致 Normalize 全 NaN → 必须 **bf16**（config 原生 dtype，内存同 fp16 无代价）
- **E-600k 零扰动**：vllm GPU 97776MiB 不变、/v1/models + chat 200、容器 MEM 波动内
- 严重度：🔴 0 / 🟠 0 / 🟡 1（极端 8192×8 500 属预期保护）/ 🟢 3（遗留：告警探针、镜像固化、OOM reserved）
- 阻塞 / 非阻塞：**非阻塞（交付）**

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🟢 交付（GPU 模式就绪） |
| 环境 | Anemll 镜像容器（embed-gpu:anemll-0.1.1-st5.6.1） |
| 显存控制 | fraction 0.04 → 5.22GB 硬上限（常规 1.25GB） |
| 性能 | batch16 472 条/s（10 万条 ~3.5min） |
| E-600k | 零扰动（GPU 97776MiB 不变） |
| 建议下一步 | 固化镜像 tag、补 embed 告警探针 |

---

## 🔧 切换实施（Rex）

### 环境选择（复用现有 LLM 运行环境）
- Anemll 镜像（E-600k 同款）实测：python3.12.13 / torch 2.11.0+cu130 / transformers 5.13.1 / pip 26.1.2 / cuda=True ✓
- 容器内装 sentence-transformers 5.6.1（与 CPU venv 同版本 → 语义一致）→ commit 本地镜像 `embed-gpu:anemll-0.1.1-st5.6.1`
- embed-gpu-venv 保留未启用（容器方案更贴合"复用 LLM 运行环境"）

### 关键修正：bf16 而非 fp16
- 实测 `model.half()` 后 Normalize 溢出 → **全 NaN**；config 原生 dtype=bfloat16 → **必须 bf16**（内存同 fp16，2 字节/参数，无代价）

### 显存控制（GB10 unified memory 共卡）
- 方案：`torch.cuda.set_per_process_memory_fraction(0.04, 0)` = **5.22GB 硬上限**（0.04×130.59）+ `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- fraction 语义已在 GB10 实测确认强制生效（负向测试 0.002 → 正确抛 OOM）
- 实测：常规 reserved 1.25GB / RSS ~1.96GB；极端 8192tok×8（需 10.64GB）被拦截 → 优雅 500，保护 E-600k

### E-600k 无扰动对照

| 指标 | 切换前 | 切换后 |
|---|---|---|
| worker vllm GPU | 97776MiB | 97776MiB（不变） |
| head vllm GPU | 97774MiB | 97774MiB（不变） |
| worker free | 11.37GB | 7.9GB（embed ~2GB） |
| vllm 容器 MEM | 3.32GiB | 2.89GiB（波动内） |
| head 8001 /v1/models / chat | 200 | 200 |

## ✅ 验证结果

| 项 | 结果 |
|---|---|
| /health | device=cuda、gpu_alloc 1.19GB、reserved 1.20GB、peak 1.23GB、fraction 0.04/cap 5.22GB |
| 正确性 | 同输入 1024 维 cos_vs_CPU = **0.999795/0.999754**（bf16 精度）、norm≈1.0 |
| 性能 | batch16 **472 条/s**、batch8 297 条/s（远超预期 20+） |
| 网关回归 | 8003 客户 key→8020 200（n=2 dim=1024）、/v1/models 200、无 key 401 |
| 极端容错 | 8192×8 → 500 "Encoding out of memory"（OOM 已 empty_cache 处理，短文本恢复 200） |

## 📊 性能对比（10 万条 embedding 生成）

| 模式 | 吞吐 | 10 万条耗时 |
|---|---|---|
| CPU（旧 8020） | 3.36 条/s（batch16） | **8.8h** |
| GPU 临时进程（早前基准） | 30.7 条/s（batch1） | 54min |
| **GPU 容器化（当前，bf16+batch16）** | **472 条/s** | **~3.5min** |

> 优化路径：容器复用（torch cu130 + ST 内置）→ bf16（无 NaN）→ batch16 吞吐最大化；对比早前 GPU 临时进程基准（未优化 HTTP/批处理）提升 ~15×

## 🔄 systemd 与回滚

- 新单元 `embed-qwen3-gpu.service`（Type=oneshot+RemainAfterExit，docker start/stop embed-qwen3-gpu；容器 --restart unless-stopped 自动拉起）
- 关键 env：DEVICE=cuda、EMBED_MEM_FRACTION=0.04、expandable_segments、CUDA_VISIBLE_DEVICES=0；8020 端口保持
- **回滚**：`/home/<USER>/embed-svc/rollback_cpu.sh`（停 GPU → 启 CPU 单元 → health 验证）；原 main.py 备份 main.py.bak.cpu.v1.0.0

## ✅ 行动清单（按优先级排序）

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | 固化本地镜像 tag `embed-gpu:anemll-0.1.1-st5.6.1`（勿覆盖） | Rex | P2 | 已 commit |
| 2 | embed-gpu 容器纳入告警探针（当前无专用探针） | Rex | P2 | 1 周内 |
| 3 | 若业务需长文大批量 embedding（8192×8+）：评估调 fraction/MAX_CTX（当前 free ~8GB） | Archi+Rex | P2 | 按需 |

## ⚠️ 待完善 / 已知局限

- 极端 8192token×batch8 → 500（显存上限保护属预期；常规不受影响）
- OOM 后 torch reserved 至 4.23GB（cap 内无风险，常规回落 1.2GB）
- E-600k 有流量时 embedding 高并发仍需关注共卡（当前 free ~8GB）

## 📚 数据来源 & 成员产出索引

- Rex：环境复用实测、bf16 修正、显存控制（fraction 负向验证）、切换时间线（模型加载 7.45s）、验证全量、systemd diff、回滚脚本、遗留项

---

> 本报告由工程保障团队 AI 协作生成，关键决策（长文大批量调参）请由人类工程负责人复核。
