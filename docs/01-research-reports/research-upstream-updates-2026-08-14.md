# 生产镜像上游更新核查报告

**日期**：2026-08-14 ｜ **范围**：生产链路镜像（anemll 推理 / litellm 网关）+ 在跟踪的 vLLM / SGLang ｜ **数据源**：GitHub API / HF 实时

---

## 1. 结论速览

| 镜像 | 生产版本 | 上游最新 | 差距 | 行动建议 |
|---|---|---|---|---|
| anemll/dspark-vllm-gx10 | 0.2.1-v026.0（本地构建） | **v0.1.1**（2026-07-15） | 生产已领先上游 | ✅ 无需升级 |
| litellm | v1.83.7 | **v1.96.2**（2026-08-11） | 落后 13 个版本 | ⚠️ 列入升级待办 |
| vLLM | 0.26.1.dev0（anemll 内置） | **v0.27.1 已发布**（2026-08-11） | 落后 1 个大版本 | 🔄 跟踪 01 构建中 |
| SGLang（NGC 26.07） | 0.5.14（容器内置） | **v0.5.17**（2026-08-08） | 落后 3 个 minor | 观察，遇 bug 再升级 |

---

## 2. 分项详情

### 2.1 anemll/dspark-vllm-gx10 —— 上游无更新，生产已领先

- 上游仓库：`github.com/Anemll/dspark-vllm-gx10`（54 stars，描述：Two-node DGX Spark/ASUS GX10 DeepSeek V4 Flash DSpark NVFP4 port for **vLLM 0.25**）
- 最新 release：**v0.1.1**（2026-07-15 发布，vLLM 0.25.2 内核）
- main 分支 pushed_at 2026-08-05（有提交但未发新 release）
- **关键事实**：生产镜像 `0.2.1-v026.0`（digest `e100ddad56...`）经镜像 Labels 反查确认源自该仓库，但 **0.2.1-v026.0 是本地构建版本（内置 vLLM 0.26.1）**，已超越上游 v0.1.1（vLLM 0.25.2）
- **结论：anemll 上游没有比生产更新的版本，无需追新**

### 2.2 litellm —— 差距最大，建议评估升级

- 上游最新：**v1.96.2**（2026-08-11 22:09 发布）
- 生产：v1.83.7（02 litellm-proxy，ghcr.io/berriai/litellm）
- 差距：v1.84 → v1.96，**13 个版本**
- 建议：litellm 迭代极快（安全修复/新模型路由/网关功能），升级前需回归验证：API key 鉴权（3 套 key）、embed 池路由（8022）、上游 vLLM 转发（8001）、限流/重试配置

### 2.3 vLLM —— 0.27.1 已发布，正对应 01 构建任务

- 上游最新：**v0.27.1**（2026-08-11 发布）
- 生产：anemll 内置 vLLM 0.26.1.dev0
- **联动**：01 上 `vllm027-build` 容器（基于 NGC 26.07 / CUDA 13.3.1 环境）正在源码构建 vLLM 0.27 → 与上游 0.27.1 时间线吻合
- 0.27 含 SM120 系列修复（#29711 cutlass_scaled_fp4_mm sm120a dispatch、#35568 sm120 family guard）——NVFP4 升级路径的关键
- 跟踪点：`/vllm-src/build.log` 构建结果

### 2.4 SGLang —— 上游 0.5.17，NGC 26.07 内置 0.5.14

- 上游最新：**v0.5.17**（2026-08-08 发布）
- NGC 26.07 内置：0.5.14+nv26.7（PR #25820 NVFP4 DSV4 支持已含 ✓）
- 若 NVFP4 验证遇 0.5.14 特定 bug：可换 `lmsysorg/sglang:v0.5.17`（上游自建）或等 NGC 26.08（8 月底预期）

---

## 3. 建议动作清单

1. **litellm 升级评估**（P1）：v1.83.7 → v1.96.2，先核对 changelog 中 breaking changes + 3 套 key 回归
2. **vLLM 0.27.1**：等 01 构建结果 → 若成功，按既有 0.27 升级预案（aarch64 源码编译 + SM120 验证）评估灰度
3. **SGLang 0.5.17**：仅在 NVFP4 验证遇阻时升级；NGC 26.08 发布后复查内置版本
4. **anemll**：无需动作（生产已领先上游）

## 4. 数据源

- GitHub API（api.github.com）：Anemll/dspark-vllm-gx10、BerriAI/litellm、vllm-project/vllm、sgl-project/sglang
- 生产镜像 Labels 反查（02 docker image inspect）：`org.opencontainers.image.source = github.com/anemll/dspark-vllm-gx10`
