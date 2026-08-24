# vLLM A/B 调度参数持久化变更记录（2026-08-09）

> **2026-08-09 17:50 修订**：镜像实测不支持 Concurrent Partial Prefill，最终生效参数调整为 **3 个**（移除 partial 两参数，详见 §补充：partial prefill 不支持）。以下参数表为**最终生效版本**。

## 变更概述

根据用户决策（priority 分级可行、先 AB 后 DCP、batched tokens 调整到 4096），在四台服务器的 vLLM LLM 启动脚本中持久化新增调度参数。**未实际重启任何服务**（重启计划另行安排），本变更仅修改启动脚本，待下次重启时生效。

## 变更参数（最终生效）

| 参数 | 值 | 作用 |
|------|-----|------|
| `--max-num-batched-tokens` | **4096** | 调度预算（原 vLLM 按 spec decoding 自动限制为 2024），允许更大 prefill 批处理 |
| `--long-prefill-token-threshold` | **2048** | 长 prefill 分块阈值（默认 0/disabled），>2048 token 的 prefill 每步只调度 2048 token → 短请求/decode 可插队（V1 调度器 scheduler.py:507/861 已实现） |
| `--scheduling-policy` | **priority** | 优先级调度（默认 fcfs），配合请求级 priority 分级（方案 B） |

> **已移除**：`--max-num-partial-prefills` / `--max-long-partial-prefills`（镜像不支持，见下方补充说明）。

参数名已在 anemll 镜像内源码级验证（EngineArgs/SchedulerConfig 字段存在，CLI kebab-case 正确）。

## 涉及文件（5 个脚本，四台服务器）

| 机器 | 脚本 | 角色 |
|------|------|------|
| 01 (<MGMT_OCTET>) | `<INSTALL_DIR>/scripts/start_head_v026r.sh` | A 组 TP2 head |
| 02 (<MGMT_OCTET>) | `<INSTALL_DIR>/scripts/start_worker_v026r.sh` | A 组 TP2 worker |
| 03 (<MGMT_OCTET>) | `<INSTALL_DIR>/scripts/start_head_groupB.sh` | B 组 TP2 head |
| 03 (<MGMT_OCTET>) | `<INSTALL_DIR>/scripts/start_worker_groupB.sh` | B 组 worker 副本（分发源） |
| 04 (<MGMT_OCTET>) | `<INSTALL_DIR>/scripts/start_worker_groupB.sh` | B 组 TP2 worker |

01/02 的 `~/start_head_v026r.sh` / `~/start_worker_v026r.sh` 为软链，自动跟随。
`~/start_head_E.sh` / `~/start_worker_E.sh`（vLLM 0.25 旧版、无引用）未修改。

## 备份

每个修改文件均有 `.bak-sched-<时间戳>` 备份（`shutil.copy2` 完整副本），位于同目录。

## 验证结果

- **语法**：5 文件 `bash -n` 全部 SYNTAX OK
- **参数计数**：每文件 5 个参数各出现 1 次，无重复
- **幂等**：补丁脚本复跑全部 `[skip]`（已存在则跳过）
- **格式**：每行一个参数（`--max-num-seqs 6` 之后、`--gpu-memory-utilization` 之前），无挤行
- **本机副本**：`deliverables/engineering-assurance/` 下 3 个脚本副本已同步

## priority 分级约定（业务侧）

`--scheduling-policy priority` 启用后，请求通过 OpenAI 兼容 API 的 `extra_body` 传优先级（**数值越小优先级越高**）：

| priority | 业务场景 |
|----------|----------|
| **0** | 交互式请求（前端问答，TTFT 敏感） |
| **10** | RAG/知识库检索（中优先级） |
| **20** | 后台任务（批量、离线索引） |

示例：
```python
client.chat.completions.create(..., extra_body={"priority": 0})
```

## 重启后验证清单

1. 启动日志确认参数生效：`grep -iE 'max_num_partial_prefills|scheduling_policy|max_num_batched_tokens' <vllm日志>`
2. 调度策略显示 `priority` 而非 `fcfs`
3. 长 prefill（如 32K/131K）+ 短请求混合压测：短请求 TTFT 不再被长 prefill head-of-line 阻塞
4. priority 0 请求在 priority 20 之前被调度
5. `max-num-batched-tokens 4096` 与 spec decoding 兼容性观察（如有异常回退 2024）

## 未变更项

- **DCP（decode context parallel）**：按用户决策后置，单独验证后再上（分 C1 单独 / C2 叠加 MTP）
- 服务未重启，参数待下次维护窗口生效

## 回滚

任一脚本 `cp <脚本>.bak-sched-<时间戳> <脚本>` 即可恢复原配置。

---

## 补充：Concurrent Partial Prefill 镜像不支持（2026-08-09 17:50）

### 现象

`--max-num-partial-prefills 4`（或 `--max-long-partial-prefills` 非默认值）启动时，vLLM 在 `create_engine_config()` → `_check_feature_supported()` 直接抛：

```
NotImplementedError: Concurrent Partial Prefill is not supported.
We recommend to remove Concurrent Partial Prefill from your config.
```

（用户 2026-08-09 实测，已写入脚本注释）

### 根因（源码级核实）

anemll 镜像 `vllm/engine/arg_utils.py:2403-2411`：

```python
def _check_feature_supported(self):
    """Raise an error if the feature is not supported."""
    # No Concurrent Partial Prefills so far.
    if (
        self.max_num_partial_prefills != SchedulerConfig.max_num_partial_prefills
        or self.max_long_partial_prefills
        != SchedulerConfig.max_long_partial_prefills
    ):
        _raise_unsupported_error(feature_name="Concurrent Partial Prefill")
```

- CLI 解析层支持（参数可解析）✅
- `SchedulerConfig` 构建层支持（校验通过）✅
- **`_check_feature_supported` 明确拒绝**（源码注释 "No Concurrent Partial Prefills so far"）❌
- V1 调度器 `vllm/v1/core/sched/scheduler.py` **零引用** `max_num_partial_prefills`（无执行逻辑）❌

与社区情况一致：官方 vLLM V1 文档中 "Concurrent Partial Prefills" 状态为 **🟡 In Progress**；GitHub Issue #39737 报告同样报错（vLLM 部分版本直接拒绝）。

### 处置

| 参数 | 状态 | 说明 |
|------|------|------|
| `--max-num-partial-prefills 4` | ❌ 移除 | 镜像不支持，启动即崩溃 |
| `--max-long-partial-prefills 1` | ❌ 移除 | 同上 |
| `--max-num-batched-tokens 4096` | ✅ 保留 | 生效 |
| `--long-prefill-token-threshold 2048` | ✅ 保留 | **单独有效**：不被 `_check_feature_supported` 拦截（镜像内实测通过），且 V1 调度器 scheduler.py:507/861 实现单长 prefill 分块 → 短请求/decode 插队（方案 A 核心机制） |
| `--scheduling-policy priority` | ✅ 保留 | 生效（方案 B） |

> 社区（dev.to《Chunked Prefill: Why One Long Prompt Stalls Every Decode》）同样确认：`long-prefill-token-threshold` + `max-num-batched-tokens` 是 head-of-line blocking 控制的核心，partial 参数只是"多 prefill 并发"的增强——增强部分本镜像不支持，但**核心分块机制可用**。

### 最终生效参数（五脚本统一）

```bash
--max-num-batched-tokens 4096 \
--long-prefill-token-threshold 2048 \
--scheduling-policy priority \
```

### 回滚

- partial 移除前备份：`*.bak-rempartial-*`
- threshold 位置修复前备份：`*.bak-fixpos-*`、`*.bak-addthreshold-*`
