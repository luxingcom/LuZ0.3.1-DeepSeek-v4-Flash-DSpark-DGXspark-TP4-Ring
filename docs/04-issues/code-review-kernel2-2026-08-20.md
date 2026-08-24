# Code Review: kernel② 交付包（NVFP4 DS-MLA KV Linear/Paged 写回）

> 审查人：科迪（Cody）· 代码审查师 ｜ 日期：2026-08-20
> 对象：`C:\Users\novAI\WorkBuddy\集群部署\_kernel2_extract\kernel2-delivery\`（15 文件，逐行全量审查）
> 背景：DGX Spark 4 节点 TP4 / DeepSeek V4 Flash / 停机窗口；kernel2 = v17（md5 a795b2b4a486f8bd2b07366890e928af，与生产四节点已部署版本一致，此前克隆环境 7 组 T 逐字节全过）
> 结论评级：**Request Changes（交付包层面）／ v17 内核本体 Approve（生产留任）**

---

## TL;DR

**v17 内核本体没有发现任何正确性/内存安全缺陷**——列界推导全部成立、store 全掩码、int64 寻址、T=0 早退、确定性成立，且与生产 md5 一致，此前的克隆实测可信。**但交付包本身不合格**：安全测试套件上一轮已被指出的 3 处断言缺陷**一处未修**（README §四 的验证指令在真机上仍会 3/7 失败，"安全可靠全过"的结论因此不成立）；安全报告 §一/§八 的两处数值断言错误原样保留；benchmark 头条口径（262.3 GB/s = "理论 96%"）是 L2 驻留膨胀 + memset 漏计的合成结果，真实 HBM 口径应取 T=65536 的 194.3 GB/s（71%）；paged 与 linear 存在语义漂移（同一输入产出不同 scale 字节），且 paged 的 `BLOCK_SIZE=256` 硬编码无断言——若生产块大小不是 256，将是静默 KV 损坏（已列 SRE 验证项，代码角度证实风险真实存在）。

---

## 严重度分布

| 严重度 | 数量 | 说明 |
|---|---|---|
| Critical | 0（1 项条件升级，待 SRE） | paged BLOCK_SIZE 若与生产不一致则升 Critical |
| High | 2 | 安全测试套件 3 处断言缺陷（未修复）；paged BLOCK_SIZE 硬编码无断言 |
| Medium | 4 | 安全报告数值错误；paged↔linear 语义漂移；benchmark L2 膨胀口径；memset 漏计 |
| Low | 5 | 陈旧注释/版本号、README 计数与数字出入、v11 缺 shape 断言、死参数/死导入、测试覆盖缺口 |

---

## 发现总表

| # | 严重度 | 文件:行 | 问题 | 修复建议 |
|---|---|---|---|---|
| 1 | **High** | test_nvfp4_ds_mla_kv_linear_v17_safety.py:44 | `assert (out[:, 512:576] == 255)` — kv=1e6 时实际 scale 字节 = **144**（floor(log2(1e6/6))=17，17+127=144）；255 需 amax≥6×2^128≈2.06e39 > fp32 max，**数学上不可达**（fp32 有限输入可达上限为 252） | 改 `== 144`；docstring 第 37 行 "scale 字节 255" 同步改 144 |
| 2 | **High** | 同上:52 | `assert (a[:, 512:576] == 1)` — 零输入 amax 被 clamp 到 1e-30 → floor(log2(1e-30/6)) = -103 → 字节 = **24**；注释 "e8m0=127-126=1" 误以为触发 exp 下界 clamp，实际未触发 | 改 `== 24`，删除错误注释 |
| 3 | **High** | 同上:59-60 | `v17_impl(torch.randn(1,1024))` 与 `ref_impl(torch.randn(1,1024))` 两次**独立、未 seed 的 randn** —— 比较的是两个不同张量，必失败 | 生成一次 `kv` 后两个实现共用 |
| 4 | **High**（待 SRE 条件升级 Critical） | nvfp4_ds_mla_kv_linear_paged_triton.py:158 | `BLOCK_SIZE = 256` 硬编码：wrapper 手握 `kv_cache.shape[1]`（第 153 行只取了 shape[0]）却不派生、不断言；kernel 用它算 `block_idx = position // BLOCK_SIZE`（:57）与 `slot = position % BLOCK_SIZE`（:58）。若生产块大小≠256 → **每个 token 写错块+错槽，静默 KV 损坏且无任何报错**；测试自建 256 槽缓存，永远测不出 | 最小修复：`assert kv_cache.shape[1] == BLOCK_SIZE`；正确修复：`BLOCK_SIZE = kv_cache.shape[1]` 参数化下传。paged_torch 参考（:28）同样硬编码，需一并改。**SRE 确认生产 block_size 后定级**：=256 → 降 Low（补断言即可）；≠256 → 升 Critical（paged 禁用） |
| 5 | **Medium** | kernel2_v17_safety_reliability.md:15, 79 | §一第 15 行 "零输入 → scale 字节 1"（实为 **24**）；§八第 79 行 "kv=1e6 → scale 字节 255"（实为 **144**）。安全认证文档对被认证内核的语义给出错误数值，而 §九 "全过" 的结论部分建立在这些错误期望值上 | 按正确值勘误；顺带修 :14 "E8M0 字节 clamp [0,255]" 的误导表述（exp clamp [-126,127] 下实际可达域是 [1,254]，上下界 clamp 在 fp32 下均为死代码） |
| 6 | **Medium** | paged_triton.py:90,96,100 vs v17_triton.py:78,82 | **paged↔linear 语义漂移**（详见线索 4 分析）：safe_max 1e-38 vs 1e-30、exp clamp [-127,128] vs [-126,127]、paged 独有 `tl.maximum(scale_f32, 1e-38)` 除数防护 → 同一输入两实现产出**不同 scale 字节**（零输入 0 vs 24） | 统一 paged 为 linear v17 语义（1e-30 / [-126,127] / 去掉除数防护），或在 README 明示两路径编码差异域及不可交叉校验 |
| 7 | **Medium** | benchmark_..._v17.py:13,31 + v17_triton.py:131 | `BYTES_PER_TOKEN = 1024*4+584` 只计 kernel 写 584B，未计 wrapper `torch.zeros((T,584))` 的 memset —— 实际写流量 584+576=**1160B/token**，GB/s 虚高约 **12.3%**（5256/4680）；且 v11 用 `torch.empty` 无 memset，同一公式下 **v11 vs v17 对比对 v17 有利** | 计入 memset（1160B/token），或 v17 改 `torch.empty`+kernel 写 pad；两版本口径必须一致 |
| 8 | **Medium** | benchmark_..._v17.py:31,36 + README.md:16 | 小 T 档工作集 < L2：T=1024 仅 kv 4MB + out 0.6MB ≈ 4.7MB，do_bench 100 次重复同一缓冲 → 全 L2 命中，262.3 GB/s 是 **L2 带宽而非 HBM**（克隆实测 436 GB/s > 273 物理上限即铁证）；T=4096（~21MB）仍可能驻留。"理论 96%" 的头条口径失真 | 输入多缓冲轮换（总集 > L2）或仅以 T=65536（194.3 GB/s = 71%）作 HBM 口径；README §一 "T=1024 → 262.3（理论 96%）" 加注 L2 驻留 |
| 9 | Low | paged_triton.py:92（另 linear_torch.py:36、paged_torch.py:50、linear_triton.py:70 同病） | "与生产 v5 逐字节一致" — v5 为过时版本号（现为 v11/v17）；benchmark docstring 自称 "v4"（benchmark_linear.py:1）/"v6"（benchmark_paged.py:1），与实际内容（v11）不符 | 全部更新为现行版本号，或改为"与 torch 参考/floor 语义一致"的版本无关表述 |
| 10 | Low | README.md:12,18,55 + safety md:35,80 | 声明与实际出入：①"包结构（14 文件）"实为 **15**；②"8/8 逐字节"实为 **7 组 bit-exact + 1 个仅查 shape/dtype 的 smoke**（test_v17.py:38-45）；③safety "6 组"实为 **7 个测试函数**；④safety §八写 T=65536，实际测试用 **65535**（test:61）；⑤safety §三 "T=65536 → 最优 213.5" 与 README "T=65536 → 194.3" **同包两口径打架** | 逐项勘误；213.5 vs 194.3 需注明运行条件差异或统一 |
| 11 | Low | nvfp4_ds_mla_kv_linear_triton.py:178-180 | v11 wrapper **无 `shape[1]==1024` 断言**（v17:129 有）——窄输入直接 OOB 读（kernel 假定列宽 1024） | 补 assert，对齐 v17 |
| 12 | Low | paged_triton.py:5-9,29-31,34,41-42,84 | 清理项：`tle` 导入从未使用（v11 文件的导入路径还不一致）；`num_blocks`/`DIM`/`ENVELOPE` 传参后内核未用（死参数）；`bid` 无 `< num_blocks` 界检查（block_table 脏项 → OOB 写）；k/v 读取用 `evict_last`（:84）与优化文档 M4 "输入 evict_first" 相悖 | 删死代码；可选加 bid 界断言（调试期）；evict 改 evict_first |
| 13 | Low | test_paged.py / benchmark_paged.py:23-25 / benchmark_linear.py:45 | 测试与基准口径缺口：①paged 测试无"同一 (seq,position) 重复写"用例（triton 并行冲突未定义 vs torch 顺序后者胜——生产行为未验证）；②paged benchmark 的 ref 计时含每次 `torch.zeros_like` 分配，speedup 虚高；③benchmark_linear.py:45 残留 "speedup=42.17x（v4）" 营销句，与优化文档自己的教训③（对标慢参考无意义）自相矛盾 | 补冲突用例（或文档化 last-wins 语义）；ref 预分配复用；删营销句 |

---

## 预扫线索逐条结论

### 线索 1 —— 安全套件 3 处断言缺陷：**证实（一处未修）**

逐行核对 `test_nvfp4_ds_mla_kv_linear_v17_safety.py`：

- **①test_saturation（:44）**：仍在。`assert (out[:, 512:576] == 255)`。推导：amax=1e6 → floor(log2(166666.67)) = floor(17.346) = 17 → 字节 17+127 = **144**。255 需 scale_exp=128，即 amax ≥ 6×2^128 ≈ 2.06×10^39，超出 fp32 max（3.40×10^38）约 6 倍——**不可达**；fp32 全域可达上限为字节 252（amax=3.4e38 → floor(125.41)=125）。附带：:42-43 两个 nibble 断言（全 0x07）本身会通过（1e6/2^17≈7.6 → clamp 6 → mag 7），失败点只在 :44。
- **②test_sign_zero（:52）**：仍在。`assert (a[:, 512:576] == 1)`。零输入 → amax 被 v17:78 clamp 至 1e-30 → floor(log2(1e-30/6)) = floor(-102.24) = -103 → 字节 **24**。注释 "127-126=1" 的错误在于：1e-30 远未触发 -126 下界。注意 :51 的 `torch.equal(a, b)`（-0.0 与 +0.0 输出一致）**会通过**（`(-0.0 < 0.0)=false`，sign=0），失败点只在 :52。
- **③test_boundary_T（:59-60）**：仍在。两次连续 `torch.randn(1, 1024, device="cuda")` 未 seed 且分别喂给两个实现——比较的是两个**不同的随机张量**，逐字节相等概率为零，必失败。

**影响**：README §四 将该套件列为生产验证指令（"安全/可靠 6 组"），真机执行将 3/7 失败；safety 报告 §九 "安全可靠全过" 与 README §一 "8/8" 的验收结论**不成立**。此前一轮审查已指出，本轮交付未修——**回退到用户的整改未被执行**。

### 线索 2 —— 安全报告数值断言错误：**证实（仍在）**

- §一 :15 "零输入 → … scale 字节 1（e8m0=127-126=1）——与 v11/torch 一致"：**值错误**（实为 24），讽刺的是"与 v11/torch 一致"这句倒是真的——v11 与 torch 参考同样产出 24。
- §八 :79（代码块第 2 行）"饱和：kv=1e6 → nibble 全 7 + scale 字节 255"：**值错误**（实为 144）。

两处原样保留，与线索 1 的测试缺陷互为镜像（测试按报告的错误期望值写死），说明"测试脚本缺陷"与"文档数值错误"是同一根因的两面，必须一起修。

### 线索 3 —— paged BLOCK_SIZE=256 硬编码：**证实（代码角度风险真实）**

- :158 `BLOCK_SIZE = 256` 字面量；wrapper 在 :153 已取 `kv_cache.shape[0]` 却不取 `shape[1]` 派生块大小，**无任何断言**。
- kernel :57-58 用它做 `position // BLOCK_SIZE` / `position % BLOCK_SIZE`——块大小错一个数，**每个 token 的物理写入位置整体错位**，且是静默的（无越界报错，只要 bid 落在合法 block 范围内）。
- README :64 更以文字形式将 256 固化进语义声明（"bid=block_table[seq,pos//256]，slot=pos%256"）。vLLM 社区默认 block_size 常见为 16/32；256（584B×256 槽 ≈ 149.5KB/块）可能是本集群为带宽故意选择的，但**代码不设防意味着任何配置漂移都会变成数据损坏而非报错**。
- 附带：`num_blocks`（:153，传参 :177）在 kernel 内**从未使用**——本可用于 bid 界检查的死参数。
- paged_torch 参考 :28 同样硬编码 256，参考与实现"一致性"建立在共同的假设上，测试（自建 256 槽缓存）对该假设**零覆盖**。

**定级**：待 SRE 验证生产块大小。=256 → Low（补 assert）；≠256 → **Critical**（paged 路径禁止上线）。无论结果如何，"派生或断言"都是必改项。

### 线索 4 —— paged↔linear 语义漂移：**证实，判定为 Medium（reader 层面基本安全，但编码层不可互换）**

三处差异（行号均已核实）：

| 维度 | linear v17 | paged | 后果 |
|---|---|---|---|
| amax 下限 | `tl.maximum(amax, 1e-30)`（:78） | `tl.maximum(max_abs, 1e-38)`（:90） | amax<1e-30 的组选不同 scale |
| exp clamp | [-126, 127]（:82） | [-127, 128]（:96） | 下界差 1 档；上界 128 在 fp32 下双方均为死代码 |
| 除数防护 | 无（scale=exp2(exp) 直接除，:83-85） | `tl.maximum(scale_f32, 1e-38)`（:100） | 见下，paged 特有偏差 |

数值验证（零输入）：linear → amax=1e-30 → exp=-103 → **字节 24**；paged → safe=1e-38 → floor(log2(1.667e-39)) = -129 → clamp -127 → **字节 0**。**同一输入、两套生产写回、不同 scale 字节**——证实。

**对 reader 一致性的分层判定**：

1. **解码值层面（reader 实际消费的东西）**：reader 按 `val = e2m1(nibble) × 2^(byte-127)` 反量化。零值组 nibble=0 → 两边解码都是 0，**完全一致**。对 amax ≥ 1e-30 的组（正常 KV 幅度），两边 floor(log2(amax/6)) 相同 → 编码一致。差异域仅限 **amax < 1e-30**（含零）的组：此时 linear 用 2^-103 一档、paged 用真实 amax 对应档，nibble 与 byte 均不同，解码幅值可有约 2× 差异——但两者对原值的逼近误差都在量化误差量级内，且幅度 <1e-30 的 KV 对 attention 输出的贡献可忽略。
2. **paged 内部自洽性缺陷**：当 exp 被 clamp 到 -127（组 amax < 6×2^-126 ≈ 7.05e-38）时，编码字节说 scale=2^-127≈5.88e-39，但**实际量化除数是防护值 1e-38**——写侧与读侧 scale 不一致，解码值系统性偏低至 **0.588×**（2^-127/1e-38）。linear 无此问题（除数=编码 scale 恒成立）。该偏差仅影响 amax<7e-38 的组，实际影响趋近于零，但这是"靠数值侥幸正确"而非设计正确。
3. **工程层面**：字节流不一致意味着 linear 与 paged 写回**不可交叉校验/不可互换**（任何跨路径 bit-exact 断言会失败）；TP4 多 rank 因各 rank 走同一路径而不受影响。paged_torch 参考（:44,:51,:58）与 paged_triton 逐项对齐（1e-38 / [-127,128] / 除数 clamp 1e-38），**paged 对内部自洽**，5/5 测试通过是真实的——漂移只存在于两个产品族之间。

**建议**：以 linear v17 语义为金标准统一 paged（改动 3 行 + 重跑 5/5），或在交付文档明示差异域。若 vLLM 生产同时启用两条路径（README 部署矩阵确实将 paged 列为"维持"），统一是更稳妥的选择。

### 线索 5 —— "与生产 v5 一致"陈旧注释 + paged 额外防护：**证实**

- "v5" 出现 4 处：paged_triton.py:92、linear_torch.py:36、paged_torch.py:50、linear_triton.py:70。v5 不在交付矩阵中（现行为 v11/v17），注释指向不存在的基准。benchmark docstring 的 "v4"（benchmark_linear.py:1）/"v6"（benchmark_paged.py:1）同类问题。
- `tl.maximum(scale_f32, 1e-38)`（paged:100）确为 linear 没有的额外防护，其后果已并入线索 4 分析（0.588× 解码偏差）。

### 线索 6 —— 小 T 档 GB/s 为 L2 膨胀：**证实**

- T=1024 工作集 = kv 4MB + out 0.6MB ≈ 4.7MB，远小于 Blackwell 级 L2；benchmark_v17.py:31 只分配**一份** kv，do_bench（warmup=25, rep=100）重复命中同一缓冲 → L2 常驻。克隆实测 **436 GB/s > 273 GB/s HBM 物理上限**，是 L2 驻留的直接物证。
- README :16 头条 "T=1024 → 262.3 GB/s（理论 96%）" 因此是 **L2 带宽口径**；T=4096（~21MB）大概率仍部分驻留（248.9 存疑）；唯一可信的 HBM 数字是 T=65536（工作集 ~300MB）的 **194.3 GB/s = 71%**。
- 判定：不算造假——交付区间写成 "194~262" 档住了下界——但 "96% 理论" 的单点强调和 safety 报告 :3 的 "已达标" 都以膨胀端点为卖点。**性能结论的量级判断（194+ GB/s、3.5×+ 于 v11 的 53.4）依然成立**，不影响部署决策，影响的是口径严谨性。

### 线索 7 —— BYTES_PER_TOKEN 漏计 memset：**证实**

- v17 wrapper :131 `torch.zeros((T,584))` 每次调用做全量 memset（584B/token），kernel 再写 576B（data 512 + scale 64；pad 8B 只由 memset 覆盖）→ 实际写流量 **1160B/token**，而 benchmark_v17.py:13 只计 584。
- 计算的影响：口径流量 4680B/token vs 实际 ≥5256B/token（读 4096 + 写 1160）→ GB/s 虚高 **~12.3%**（L2 可部分吸收 memset→覆写的二次写，故说"≥"）。194.3（T=65536）按全流量口径修正后约 **173 GB/s（63%）**。
- 更重要的是**对比公平性**：v11 wrapper 用 `torch.empty` + 独立 pad kernel（只写 8B），无 memset；同一公式下 v11 口径准确、v17 口径漏计 → v11 vs v17 的 GB/s 差距被夸大 ~12%。

---

## 其他发现（线索外）

1. **v11 wrapper 无形状断言**（linear_triton.py:178-180）：kernel 假定列宽 1024（K 偏移 ≤511、V 偏移 512+511），窄输入直接越界读。v17:129 有断言，v11 作为回退路径应对齐。Low。
2. **torch 参考正确性核验通过**：两个参考对各自 triton 内核建模精确——E8M0 公式（floor(log2(max/6))+127）、阈值链 tie 语义（triton strict `>` 取低档 vs torch argmin 取首个最小值=低档，全部 7 个阈值一致）、pack 布局（偶元素低半字节，`0::2` vs reshape[8,2] 首列，一致）、clamp 域逐项对齐（linear 对 1e-30/[-126,127]，paged 对 1e-38/[-127,128]）。**参考是各自内核的忠实镜像**——这既是优点（测试可信），也是线索 4 漂移得以潜伏的原因（没有跨族参考）。
3. **v17 内核逐项安全核验通过**：列读界 `col_base+BLOCK_G*16 ≤ 1024`、写界 data ≤512 / scale ≤576、pad 由 zeros 保证、全部 store 带 `mask_t`、指针算术全程 int64（T=65536 时偏移 ~38M×584 需 int64，已正确处理）、T=0 早退（:133-134）、`tl.multiple_of` 提示与标准 Triton 用法一致且有逐字节实测背书。
4. **测试有效性**：test_linear/test_v17/test_paged 均 fix seed（manual_seed），断言用 `torch.equal`/atol=0，强度足够；test_v17 的 `_rand_kv` 注入大小值考验 scale 边界，设计好。缺口：paged 无重复 (seq,position) 冲突用例（triton 并行写同槽未定义 vs torch 后者胜——生产行为未验证）；无跨 linear/paged 一致性用例（若有，会立刻暴露线索 4）。
5. **benchmark_paged ref 口径**：`run_ref`（:23-25）每 rep 新建 `torch.zeros_like(kv_cache)`，分配+memset 计入 ref 时间 → speedup 虚高（不影响 triton 的 GB/s 数字，paged 本身 in-place 无 memset，公式对其准确）。

---

## 做得好的地方

- **v17 内核本体质量过硬**：与生产 md5 一致；负载设计（BLOCK_G×TPP 组合、连续 1D load、`tl.split` 打包）思路清晰；pad 内联消除了 v11 的独立 pad kernel；int64 寻址、全掩码 store、T=0 早退、wrapper 形状断言（:129）一应俱全。
- **正确性验证方法论正确**：逐字节 atol=0 对照 torch 金标准 + 固定 seed + 边界值注入（test_v17 的 `_rand_kv`），而不是浮点容差糊弄。
- **benchmark 主动修复并留档了历史 1000× 单位错误**（benchmark_linear.py:27-29 注释），这种"记录踩坑"的作风值得肯定。
- **优化空间文档（kv_linear_optimization_space.md）质量高**：瓶颈定性（微型 block 杀手）、"GB/s 实测为准、对标慢参考无意义"的教训、S/M/C/A/E 分层手段与量化预期，与 v17 实际架构（S1+M1+S3+E1）一一对应，可追溯性好。
- **CUDA Graph（R2 warmup）与 NaN/Inf 前提（R1）在报告和 README 中均有明确交代**，集成风险提示到位。

---

## 结论与修复清单

**评级：Request Changes（交付包层面）。**

- **v17 内核本体：Approve**——与生产部署版本 md5 一致，本轮未发现任何内核缺陷，生产留任不受影响。
- **交付包：不通过**——README §四 的自验指令在真机上会失败（3/7），安全报告含错误数值断言，性能头条口径失真；该包**不能作为合规验收依据**直至下列 P0 修复完成。

| 优先级 | 项 | 工作量 |
|---|---|---|
| P0 | 修 safety 测试 3 处断言（#1/#2/#3）+ safety 报告 2 处数值（#5），真机重跑 7/7 | ~30 分钟 |
| P0 | paged BLOCK_SIZE：SRE 确认生产块大小后，加断言或参数化（#4） | 断言 5 分钟 / 参数化半天 |
| P1 | benchmark 口径：计入 memset + 多缓冲轮换防 L2 驻留（#7/#8），重跑并修订 README 数字 | 半天 |
| P1 | paged 语义统一到 v17（1e-30 / [-126,127] / 去除除数防护）+ 重跑 5/5（#6） | 半天（含回归） |
| P2 | 注释版本号、README 计数、v11 断言、死代码清理（#9-#13） | 顺手 |

> 审查方法说明：全静态审查（15 文件逐行）+ 数值语义手工推导复算（144/24/224/252/0/0.588× 等关键值均独立推导验证）；未在真机执行（停机窗口内核生产容器已停，安全套件结论基于断言期望值与内核语义的确定性推导，置信度高）。
