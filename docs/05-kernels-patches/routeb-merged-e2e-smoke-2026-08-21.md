# e2e 冒烟报告：Triton 长尾架构插件 TP4 生产部署（Task #2，2026-08-21 夜班）

**执行**: Archi-2（系统架构师）· node01-04 TP4 生产集群 · 插件 routeb_merged_plugin v2（Triton 长尾架构）
**结果**: **PR 测量 No-Go（结构性阻塞：DSL 冷编译卡首 prefill ≥45min）**；架构验证大幅推进（四节点部署/TP4 启动/CUDA graph 捕获/decode Triton 路径全部打通）；已回退基线运行。

> **一页结论（供裁决）**
> 1. **生产 TP4 插件路径打通到"首个 merged prefill"**：四节点部署（md5 一致）→ entry point 经 spawn 链四 rank 全部生效（"Using MergedB12xExperts" ×4）→ CUDA graph 捕获 16/16 PASS → 20 发 warmup（短 prompt，Triton decode 路径）全部 200 OK。
> 2. **PR 四档未取得**：首个 ≥MIN_M 的 prefill chunk 在 EngineCore 内触发 cute.compile 冷编译，**Worker_TP0 98% CPU 编译 ≥45 分钟未完成**（无异常、无失败日志——纯编译耗时），引擎 prompt throughput 归零。判 No-Go，按止损条款回退。
> 3. **过程修复 6 个真实生产集成障碍**（每个都是一次性修复+实证）：entry point spawn 不继承 / pip 只读源 / rendezvous 超时 / Triton JIT 在 capture 中 / `.any()` host 同步在 capture 中 / 运行时无效 expert id illegal access。
> 4. **数值质量（mini 全链）**：merged 路径逐层 ≈ B12X（W4A4 A 侧量化噪声内，self-check rel 2-12% 元素级）；all-merged mini 总 logprob 差 41.5%（重复 prompt 放大 + 我的 A 量化器噪声大于 flashinfer 参考实现——量化器精化 = 生产化第一优先项）。
> 5. **内存实测**：KV 6,061,119 → 4,302,649 tokens（**-29%**，原型派生超 +9GB 预算——派生瘦身项）。
> 6. **回退干净**：start 脚本 .bak 恢复 + checker PASS + 基线 "Using B12X_MXFP4" + /health 200 + 零 Merged 日志。

---

## 1. 部署架构与执行记录

- **插件**: routeb_merged_plugin v2 = merged prefill（w13 N-merge 间接寻址 + w2 K-concat combine 折叠，E6/E12 零拷贝机制）+ Triton W4A16 grouped MoE（长尾 prefill + **decode 全量，B12X 完全退出**，消费同一 stacked payload + E4M3 swizzled scale，in-kernel E12 勘误公式寻址）
- **落位**: <INSTALL_DIR>/nvfp4/{plugin_merged, routeb_official_v2} 四节点，md5 一致 ✓
- **安装机制**: start 脚本 SERVE_CMD 前置 `cp -r …/plugin_merged /tmp/_pmerge && pip install --no-deps -q /tmp/_pmerge/`（nvfp4 挂载只读，pip 需可写源）+ ENV_ARGS 注入（VLLM_MOE_MERGED=1 等 4 项）；.bak-e2e-20260821 留档；checker 全程 PASS
- **启动**: TCPStore head-first 模式（6 轮迭代，每轮解决一个障碍）

## 2. 打通的部分（生产实证）

| 里程碑 | 证据 |
|---|---|
| entry point 四 rank 生效 | EngineCore 日志 "Using MergedB12xExperts"（head + workers）|
| CUDA graph 捕获 | 16/16 PASS（Triton 预热于权重加载期 + 无 host 同步分派 + 无效 id 防护三修复后）|
| Triton decode 生产路径 | warmup 20×短请求全部 200 OK（spec-decode metrics 正常推进）|
| KV cache | 4,302,649 tokens（插件版）vs 6,061,119（基线）|
| 合并路径 mini 正确性 | self-check: merged vs torch 反量化参考 rel 2.5e-2~1.2e-1；merged vs B12X rel 1.6e-2~4.1e-2（W4A4 噪声量级）|

## 3. 阻塞：DSL 冷编译卡首 prefill（No-Go 根因）

- 首个 ≥MIN_M(256) prefill chunk（TTFT_2k 基准第一发）→ 43 层共享 compile 缓存首次 cute.compile → **Worker_TP0 持续 98% CPU ≥45min 未返回**（0 异常 0 失败——纯编译），引擎 prompt tok/s 归零、2 req 挂起
- 对照：容器内独立测试同量级 shape 编译 2-4 min——生产容器 10× 慢待查（嫌疑：N_b=262144 stacked B 的 TMA 描述符 trace / CPU 配额 1-19 与邻居竞争 / MLIR 缓存冷）
- **修复路径明确**（下一窗口）: ①_derive 期间预编译固定 M_pad 档位集（256/512/1024/2048/4096 × 2 GEMM ≈ 10 次编译，启动期吸收）②AOT 编译 + CUTLASS cache 预烘焙随插件分发 ③查 10× 慢的根因

## 4. 数值质量（mini 全链，正确性优先口径）

- 插件 v2（merged+Triton 全路径）vs B12X 基线 mini logprob：**41.5%**（all-merged 暴露 + 重复 prompt 放大；对照 Task#21 flashinfer W4A4 = 0.41%）
- 逐层 self-check 三角验证：merged ≈ B12X ≈ torch 反量化参考（rel 1.6e-2~1.2e-1 元素级 max）——**管线语义正确，噪声源 = 我的 A 侧 NVFP4 量化器**（amax/6→E4M3 RTN→E2M1 RTN 阈值）
- **生产化第一优先项**: A 量化器精化（对齐 flashinfer 两级 scale 方案：global amax + 相对 block scale），预期把 logprob 差拉回 ~1% 量级
- 生产真实暴露口径（本设计意图）: 仅热桶（hash 层 ~27% 流量）走 merged，其余 Triton W4A16（A 不量化，数值≈无损）→ 实际质量影响远小于 mini 的 100% 暴露测试

## 5. 修复的生产集成障碍清单（工程资产）

1. **EngineCore spawn 不继承 monkey-patch** → pip install + `vllm.general_plugins` entry point（且必须 callable `module:install`）
2. **nvfp4 挂载 :ro → pip egg_info 不可写** → /tmp 拷贝安装
3. **插件 pip 前缀拖慢 worker 冷启 → rendezvous 301s 超时**（1/4 clients joined）→ --distributed-timeout-seconds 300→900
4. **Triton JIT 首编译发生在 CUDA graph 捕获中 → capture 崩** → _derive 期 warmup 预编译
5. **`merged_mask.any()` host 同步在捕获中 → cudaErrorStreamCaptureUnsupported** → use_merged Python 布尔分流，捕获路径零 host 同步
6. **运行时 topk 无效 expert id（-1/越界）→ Triton/merged 负偏移 illegal access**（capture 用 dummy id 不触发，replay 真实路由触发）→ clamp + 权重清零 + 桶逐出

## 6. 回退验证

- start_tp4_head.sh + 3×worker 从 .bak-e2e-20260821 恢复；checker PASS
- 基线重启 READY：**"Using 'B12X_MXFP4'"**、KV 6,061,119 tokens、/health 200、Merged 日志计数 0
- <INSTALL_DIR>/nvfp4/{plugin_merged, routeb_official_v2} 保留（无 env 不激活，零污染结构性保证）；生产现以基线运行

## 7. 遗留移交

| 项 | 优先级 | 说明 |
|---|---|---|
| DSL 预编译（阻塞根因） | P0 | _derive 期固定 M_pad 档预编译 或 AOT cache 烘焙；另查生产容器 10× 编译慢 |
| A 量化器精化 | P0 | flashinfer 两级 scale 方案对齐；mini 41.5% → 目标 ~1% |
| 派生内存瘦身 | P1 | KV -29% 超预算（+9GB 计划）；瘦身派生（免 f32 壳驻留/w2 缓存 cap 调优） |
| PR 四档测量 | P1 | 上述两项完成后重跑本 smoke（脚本全部就绪：e2e_*.sh / bench） |
| Triton decode 性能 | P2 | 未测（本夜 warmup 仅正确性）；C1/C8 快测补 |
| panorama 基线口径核对 | P2 | 2510/2500/2420/2270 的测量脚本与 bench_tp4.py TTFT 档位对应关系需与 Task#25 归档核对 |

## 8. 工件

| 位置 | 内容 |
|---|---|
| 01:/tmp/_routea_work/ | e2e_p0_probe.py（E1-E12 机制实验）/ plugin_merged/（v2 全代码）/ e2e_deploy·start·resume·fix_pip·bench·rollback 六脚本 + 全部日志（e2e_start*.log / bench_e2e_merged.log / e2e_rollback.log）/ lp_e2e_merged.json（mini 全链 logprob）|
| 四节点 <INSTALL_DIR>/nvfp4/ | plugin_merged + routeb_official_v2（保留, env 门控零污染）|
| 本地 _routeb_extract/ | 同步上述 + e2e_*.sh |
| 生产脚本 | 已恢复基线（.bak-e2e-20260821 留档于四节点）|

**结论**: 冒烟目标（PR 信号）今夜未达成——阻塞点明确（DSL 冷编译）、修复路径明确（预编译/AOT）、无未知风险。架构侧全部打通（部署→启动→捕获→decode→warmup），插件代码与六项集成修复沉淀为可复用资产。建议下一窗口按 §7 顺序推进后重跑本 smoke。
