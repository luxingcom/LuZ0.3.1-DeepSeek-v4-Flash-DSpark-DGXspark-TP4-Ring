# DGXspark02（.58）负载偏高核查报告

**日期**：2026-08-07
**工作流**：工作流 3（事故响应/调查）——负载异常核查
**参与成员**：主理人（实机核查）

---

## 📌 TL;DR（执行摘要）

- DGXspark02（.58）负载确实全面偏高（用户观察正确），已定位**三大根因**
- **根因 1（核心）**：comfyui 视频工作流独占 GPU 统一内存 103.75GiB/121.6GiB（GPU 96% 满载）→ **embed 服务 encode 请求 OOM（连 108-290MiB 都分配失败）→ 500 错误循环** → CPU/负载虚高
- **根因 2**：GPU 满载 96% + 温度 84-86°C（1h 峰值 86°C）+ 功耗 65.75W（他机 4-12W）
- **根因 3**：磁盘 util 85%（NFS 服务端 + registry + comfy 写盘）
- 四机对比：DGXspark02 是唯一高负载机（CPU 6% vs <2%、load1 1.04 vs <0.2、GPU 96% vs 0%、内存 12.2% vs <5%）
- 严重度：🔴 1 项（embed 服务不可用 + GPU 显存争抢）

---

## 🎯 核心结论卡片

| 项目 | 内容 |
|------|------|
| 整体评级 | 🔴 不健康（embed 服务 OOM 故障中） |
| 阻塞项数量 | 1（embed /health 无响应） |
| 关键行动项 | 4 条（见行动清单） |
| 建议下一步 | comfyui 限内存或迁移 embed 到空闲机（.55/.59） |

---

## 1. 四机负载对比（2026-08-07 16:36 实测）

| 指标 | DGXspark01(.60) | **DGXspark02(.58)** | DGXspark03(.55) | DGXspark04(.59) |
|------|----------------|---------------------|----------------|----------------|
| CPU 占用 | 2.0% | **6.0%** | 0.3% | 0.2% |
| load1 | 0.19 | **1.04** | 0.04 | 0.20 |
| GPU 占用 | 0% | **96%** | 0% | 0% |
| GPU 温度 | 53°C | **84°C**（1h 峰值 86°C） | 42°C | 43°C |
| GPU 功耗 | 12.4W | **60.2W** | 5.1W | 4.1W |
| 内存占用 | 4.7% | **12.2%** | 2.9% | 2.6% |
| 磁盘 util | - | **85%**（nvme0n1） | - | - |

## 2. 根因分析

### 根因 1（核心）：GPU 显存争抢 → embed OOM 错误循环
- comfyui-h3 容器：GPU 96% 满载，**占用统一内存 103.75GiB/121.63GiB**
- embed-qwen3-gpu（Qwen3-Embedding-0.6B）同机运行，encode 请求时：
  ```
  ERROR:embed-svc:encode OOM: CUDA out of memory. Tried to allocate 290.00 MiB.
  GPU 0 has a total capacity of 121.63 GiB of which 103.75 GiB is...
  ```
  **连 108-290MiB 分配都失败** → `/v1/embeddings` 返回 500 → 客户端重试 → CPU/负载虚高
- embed `/health` 当前**无响应**（服务不可用）

### 根因 2：GPU 满载 + 热负载
- comfyui 视频工作流 96% GPU 占用、功耗 60.2W（他机 4-12W）、温度 84-86°C（GB10 被动散热，长期 96% 满载有热降频风险；SM 时钟 2431/3003MHz 尚未严重降频）

### 根因 3：磁盘高 IO
- nvme0n1 util 85.01%、await 33ms：NFS 服务端（.55/.59 挂载权重）+ registry + comfyui 写盘叠加

### 与历史审查的关联
- 2026-08-07 12:15 四机审查已记录："🔴 .58 内存 110/121G 可用仅 10G（comfyui 占 50% 无上限）"——**comfyui 无显存/内存上限问题是已知遗留**，本次 OOM 事件是其直接后果

## 3. 影响范围

| 影响 | 说明 |
|------|------|
| embed 服务（8020） | 🔴 /health 无响应，embeddings 请求 500 循环 |
| 上层依赖 | litellm 网关等依赖 embed 的服务受影响 |
| 视频工作流 | ✅ comfyui 正常（独占 GPU） |
| 其余节点 | 无影响（.55/.59 GPU 完全空闲） |

---

## ✅ 行动清单（按优先级排序）

| # | 行动 | 负责角色 | 紧急度 | 预期完成 |
|---|------|---------|--------|---------|
| 1 | **comfyui 设置显存/内存上限**（容器 --memory 或 comfy 参数），防止独占 103GB 挤掉 embed（12:15 审查遗留 P0） | SRE | P0 | 今日 |
| 2 | **embed 迁移到空闲机**（.55/.59 GPU 空闲、镜像/权重已就绪）或 comfyui 与 embed 错峰运行 | SRE | P0 | 今日 |
| 3 | 恢复 embed 后验证 /health 200 + embeddings 正常 | SRE | P1 | 修复后 |
| 4 | 监控 comfyui 内存上限生效（GPU 显存水位 < 80%）与温度（< 85°C） | SRE | P2 | 持续 |
| 5 | 磁盘 util 85% 观察：NFS 流量错峰（.55/.59 同步调度避开 comfy 高峰） | SRE | P2 | 本周 |

---

## ⚠️ 待完善 / 已知局限

- docker stats 与 docker top 的 CPU 采样存在瞬时差异（stats 99.68% 为瞬时峰值，top 8.8% 为稳态）——两者均指向 embed 异常活跃
- comfyui 显存占用 103.75GiB 为 nvidia-smi 报告值（含统一内存缓存），实际进程占用需 comfy 日志确认
- embed 迁移方案需确认 litellm 配置指向（网关 base_url 变更）

---

## 📚 数据来源

- Prometheus dcgx:* 预聚合（四机 CPU/GPU/内存对比）
- docker stats / docker top（容器级 CPU/内存）
- nvidia-smi（显存占用 103.75GiB/121.6、温度、时钟）
- embed 容器日志（OOM 错误循环、500 响应）
- /proc/diskstats + iostat（磁盘 util 85%）

---

> 本报告由工程保障团队 AI 协作生成，关键决策请由人类工程负责人复核。
