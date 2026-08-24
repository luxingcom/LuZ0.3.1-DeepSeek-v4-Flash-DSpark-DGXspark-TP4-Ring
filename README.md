# LuZ0.3.1 — DeepSeek V4 Flash · DSpark · DGX Spark TP4 · Ring

4× DGX Spark（GB10/sm_121a）TP4 环网部署 DeepSeek V4 Flash 的生产调优与算子工程开源归档。

**生产形态基线（LuZ0.3.1）**：W4A4 full（`VLLM_MOE_W4A4=2`）+ 池补丁（`VLLM_B12X_SHARED_WRAPPER=1`）+ FlashInfer 0.6.16 + threshold 4096 + util 0.82 + DSpark MTP n7 + CUMEM=0。

**最终性能指标**：见 [`docs/03-final-metrics/FINAL-METRICS-LuZ0.3.1.md`](docs/03-final-metrics/FINAL-METRICS-LuZ0.3.1.md)。

## 目录结构

```
docs/
  01-research-reports/      研究报告（架构/设计/根因/上游核对）
  02-performance-benchmarks/性能测试报告与基准数据
  03-final-metrics/          最终性能指标汇总（FINAL-METRICS + CSV）
  04-issues/                 缺陷/事故/根因调查
  05-kernels-patches/        算子/kernel/补丁相关报告
  06-verification/           验证/QA/恢复演练/验收
  07-deployment/             部署/runbook/运维手册/回滚锚点
  08-tools/                  工具链说明
kernels/                    算子源码交付（W4A4 插件 / kernel1 / kernel2 / routeB / b′）
patches/                    补丁包（ws-dedup / ringonly-v5 / FP8 质量门 / routeA/routeB）
scripts/                    启动/部署/基准/巡检脚本（脱敏版）
data/                       基准原始数据（json/csv）
```

## 快速导航

- **最终指标**：`docs/03-final-metrics/FINAL-METRICS-LuZ0.3.1.md`
- **部署指导教程（新手入口）**：`docs/07-deployment/DEPLOYMENT-GUIDE.md`（环境要求 / 镜像构建 / head-first 启动 / 参数表 / 验证 / 回滚 / 安全）
- **生产镜像获取（网盘分发）**：https://pan.baidu.com/s/1l8-1-9PoAEcNrgIq88fs0g?pwd=luzi （LuZ0.3.1 自包含镜像 11.48 GiB，SHA256 `abad90a9…`，校验数据见 `docs/07-deployment/DEPLOYMENT-GUIDE.md §3.7`）
- **LuZ0.3.1 落地**：`docs/07-deployment/luz031-deployment-2026-08-23.md`
- **W4A4 翻案链**：`docs/02-performance-benchmarks/threshold-retest-2026-08-22.md` → `docs/07-deployment/threshold-4096-adoption-2026-08-22.md` → `docs/05-kernels-patches/wsdedup-l3-combo-2026-08-23.md`
- **算子源码**：`kernels/`（plugin_a1 = W4A4 生产插件；kernel2 = MLA KV linear；routeB = FP8 稠密 GEMM）
- **缺陷处置**：`docs/04-issues/`（issue22、环境 stall、AR 调查）
- **部署手册**：`docs/07-deployment/`（runbook、回滚锚点、服务部署指南）

## 说明

- 本仓库为工程保障团队在生产攻坚过程中沉淀的报告与资产归档，所有报告均为当时实验/生产实测记录，含 [实测]/[推断] 口径标注。
- 涉密信息（内网 IP、主机名、内部路径、凭证）已脱敏为占位符；若发现遗漏请提交 issue。
- 详细盘点与脱敏规则见 `docs/01-research-reports/task1-asset-inventory-2026-08-24.md` 与 `task1b-citation-revision-2026-08-24.md`。
