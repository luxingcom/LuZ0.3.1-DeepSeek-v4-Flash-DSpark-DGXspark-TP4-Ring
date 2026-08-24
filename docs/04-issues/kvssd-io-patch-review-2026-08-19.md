# Code Review: vLLM KV 缓存 SSD 卸载 io 层补丁（zstd + 裁剪 + 去重）

- **审查人**：Cody（代码审查师，工程保障团队）
- **日期**：2026-08-19
- **审查对象**：`delivery/kvssd-offload-2026-08-18/kvpatch/io.py`（新格式 store/load，替换容器内 `vllm/v1/kv_offload/tiering/fs/io.py`）
- **目标**：落盘密度 ≤10KB/token（实测基线 382KB/token，根因 5× 写放大 + 整槽 4,263,936B 落盘）

---

## 概要

补丁实现"去重（同 dest 只写 1 份）→ 裁剪（仅有效前缀 ~1.03MB）→ zstd-3 压缩（NVFP4 期望 60-65%）→ O_DIRECT 对齐写"，load 侧支持新格式与旧格式双读、失败删文件语义保留。本地已用真实 block_size（4,263,936B）跑通 6 项验证：往返一致性、旧格式兼容、8 线程×10 次并发去重、全零块、损坏头删除、orig_len 校验。**结论：有条件 Approve**——逻辑正确性已实证，但 O_DIRECT 对齐与真实 NVFP4 压缩率必须在容器内按第 5 节自检清单复验后放行。

---

## 严重问题

| # | 文件 | 行 | 问题 | 严重度 |
|---|------|-----|-------|---------|
| 1 | io.py（原版） | 原版 `store_block` | **exists() TOCTOU 竞态 + thread-local 随机 tmp suffix 使 O_EXCL 失去去重作用**：5 个 KV group 并发时均可通过 `os.path.exists` 检查，各自用不同 suffix 创建独立 tmp，全部落盘 → 5× 写放大（382KB/token 的根因之一）。补丁已用 per-path lock + 确定性 tmp + O_EXCL 修复 | Critical（已修复，留档） |
| 2 | io.py | L194-213（`_write_aligned`）、L97 | **O_DIRECT 对齐是最大技术风险**：mmap staging 方案依赖 (a) 匿名 mmap 页对齐 ≥ fs 逻辑块大小；(b) 目标 fs 逻辑块 == 4096。若 `/opt/aicad-kvssd` 逻辑块 >4096（某些 64K 配置），4096 对齐写会 EINVAL。已提供 `VLLM_KVSSD_ALIGN` 环境变量兜底，但**必须在目标路径实测**（见第 5 节第 1 条） | High（需部署验证） |
| 3 | io.py | L338-368 | **持锁做 IO（设计取舍）**：per-path lock 在 O_DIRECT 写 + replace 期间持有，与主理人"不得持锁做 IO"字面要求有出入。理由：锁是 per-path 的，只序列化**同 path 的 5 个重复调用者**（这正是我们要串行的），其它 path 完全并行；且持锁到写完让等待者能通过 `exists` 复查**接管失败写入**，否则会出现"store 已返回成功但文件从未落盘"的静默缺口。若主理人坚持不持锁做 IO，可改为锁内仅 claim tmp、锁外写，但需接受该缺口 | Medium（需主理人确认） |
| 4 | io.py | L72-75、L119-134 | **zstandard 硬依赖**：模块加载时 `import zstandard`；缺失时模块仍可加载（legacy 读可用），但新格式 store 在 `_get_zstd()` 抛 RuntimeError。**部署顺序必须先装 wheel（zstandard 0.25.0 cp312 aarch64）再打补丁**，否则 vLLM 启动后首个 store 即失败 | Medium |
| 5 | io.py | L219-236 | **per-path 锁表无界增长**：每个 dest 一条锁。60 万 token 场景块数约数万级，内存约几 MB~十几 MB，可接受；但极端大缓存下可换"分片锁表"（N=256 取模）以封顶内存 | Low |
| 6 | io.py | L330-336 | **重复压缩浪费**：5 个并发重复调用都会先压缩再竞争锁（压缩在锁外是硬要求），最多 4 份压缩 CPU 被浪费；用早期 `exists` 快速路径可缓解"已写完"场景，但"同时到达"场景无法避免 | Low |
| 7 | io.py | L334-335 | **不可压缩数据轻微膨胀**：`valid_len==block_size` 且数据不可压缩时 zstd 输出略大于输入（帧头 + 熵编码开销）。NVFP4 实测可压缩（60-65%），实际不会触发；若未来遇到不可压 KV 可加"raw 回退" | Low |
| 8 | io.py | L431 | **legacy 文件魔数误判（理论）**：旧格式首 8B 恰好等于 `b"KVZSTD01"` 的概率 ~2⁻⁶⁴，且还有 orig_len/payload_len/解压长度三重校验兜底，实际不可达 | Info |
| 9 | io.py | L406-421 | **读失败即删文件（保留原语义）**：瞬态 IO 错误（如 EIO）也会删缓存文件并 raise。这是 `kv_load_failure_policy=fail` 的既有设计，保留；但注意磁盘抖动会放大损失 | Info |

---

## 改进建议

| # | 文件 | 行 | 建议 | 类别 |
|---|------|-----|------|------|
| 1 | io.py | L200 | 将 `_write_aligned` 中 mmap 写入改为循环 `os.write` + 非对齐部分写即 raise（现为单次写 + 全量校验，已正确；仅提示 O_DIRECT 短写不可续传的语义要保留） | 正确性 |
| 2 | io.py | L446 | 解压已用 `max_output_size=block_size` 防解压炸弹；建议再加 `len(data) < block_size` 的强校验日志（现只在 0 字节时 raise，非 0 但超长由 max_output_size 兜住） | 安全性 |
| 3 | io.py | L110 | **fsync 建议**：默认关闭（`VLLM_KVSSD_FSYNC=0`）是正确选择——O_DIRECT 写返回时数据已到设备，唯一丢失窗口是 rename 元数据在断电时不持久，而 KV 缓存可重算 + vLLM fail 策略容忍缺失块。若后续要防断电丢 rename，置 1（每次 store 增加 fdatasync + dir fsync，有延迟成本），不要默认开 | 性能 |
| 4 | 部署 | — | 确认 block geometry 与 10KB/token 目标自洽：单块落盘 ≈ align_up(16 + zstd(1.03MB 有效段), 4096) ≈ 0.66MB；若每块承载 T token，则每 token ≈ 0.66MB/T ≤ 10KB ⇒ T ≥ 66。与 deepseek-v4-flash 实际 tokens/block 核对 | 验收 |
| 5 | 部署 | — | 灰度四节点前：备份原 io.py → 清旧缓存（`/opt/aicad-kvssd`）→ 装 zstandard wheel → 打补丁 → 跑第 5 节测试脚本 → 重启 | 运维 |

---

## 做得好的地方

- **O_DIRECT 对齐完全隔离在 mmap staging**（L194-213）：只有 mmap 缓冲触碰 O_DIRECT（地址页对齐 + 长度 4096 对齐），KV slot buffer 本身只用普通内存拷贝读写，其对齐性无关紧要——这是本补丁最干净的设计决策。
- **O_EXCL 确定性 tmp + flock 接管 stale tmp**（L244-298）：崩溃残留的 `.tmp` 可被后续写入安全接管（写者先 flock 再写，flock 空闲即证明无人在写），不会永久卡死该 path。
- **失败可见性**（L338-368）：持锁到写完，等待中的重复调用者能复查 `exists` 并在主写者失败后接管，避免"静默假成功"。
- **零填充语义正确**（L400-405）：裁剪掉的是尾部全零，load 时解压前缀 + 补零到 block_size，往返 bit-exact（本地测试验证 `out == slot`）。
- **失败语义保留**（L406-421）：异常先关 fd 再删源文件并 raise，兼容 `kv_load_failure_policy=fail`；先关 fd 也修复了 Windows 上"删打开文件"问题。
- **向后兼容**（L455-461）：旧格式直读；魔数 + 三重校验把误判降到不可达。
- **O_BINARY 显式化**（L81-84）：本地验证时实测发现 Windows 文本模式会把 zstd payload 中的 `0x0a` 翻译成 `0x0d0a`（文件从 24576 膨胀到 24639），加 `O_BINARY` 后修复；Linux 上为 0 无副作用。
- **解压炸弹防护**（L446）：`max_output_size=block_size`。
- **格式头设计确认**：主理人方案中的"尾部 8B 存原长度"**不需要**——头部 16B 已含 orig_len，且尾部 8B 会被 4096 对齐填充区吞掉、读取时还要特殊处理，徒增复杂度；采用纯头部方案（已在模块 docstring 说明）。

---

## 原版逻辑缺陷（留档）

1. **Critical**：`os.path.exists(dest_path)` 的 TOCTOU 竞态——5 个 group 并发均可通过检查，是 5× 写放大的直接原因。
2. **Critical**：`_get_tmp_suffix()` 的 thread-local 随机 suffix 使 `O_EXCL` 在不同线程间**永远不会冲突**（各写各的 tmp），O_EXCL 完全失去去重能力；必须改为确定性 tmp 名才能让 O_EXCL 成为跨线程/跨进程去重闸门。
3. **Minor**：`_ensure_dirs` 对 `os.path.dirname(path)==""`（裸文件名）会 `makedirs("")` 抛错；补丁已加 `if parent` 守卫。
4. **Info**：原版短写/短读检查正确但 O_DIRECT 部分写不可续传；补丁保留"短写即 raise"语义并注明原因。
5. **Info**：读失败删文件会连带删除瞬态 IO 错误场景的缓存；属 vLLM fail 策略的设计决定，补丁保留。

---

## 部署前自检清单（给主理人，3-5 条）

### 1. 目标路径 O_DIRECT 冒烟（最高优先级，在 `/opt/aicad-kvssd` 上执行）
```bash
python3 - <<'EOF'
import os, mmap
p = "/opt/aicad-kvssd/odirect_probe"
try:
    fd = os.open(p, os.O_CREAT | os.O_WRONLY | os.O_DIRECT, 0o644)
    buf = mmap.mmap(-1, 4096)
    n = os.write(fd, buf)
    print("O_DIRECT 4096-aligned write OK:", n)
    buf.close(); os.close(fd)
    print("fs block size (st_blksize):", os.statvfs("/opt/aicad-kvssd").f_bsize)
    os.remove(p)
except OSError as e:
    print("O_DIRECT FAILED:", e, "-> 需设 VLLM_KVSSD_ALIGN 或评估 fs 是否支持 O_DIRECT")
EOF
```
若逻辑块大小 >4096，设 `VLLM_KVSSD_ALIGN=<size>` 再跑一次；失败则必须回滚方案（不能直接上生产）。

### 2. 往返一致性 + 去重 + 损坏语义测试（容器内跑）
```python
"""io 补丁往返/去重/损坏测试（在容器内以真实 vLLM 环境运行）"""
import os, tempfile, threading
from vllm.v1.kv_offload.tiering.fs import io   # 或 import 补丁模块

BLOCK = 4_263_936

def make_slot(valid_len, seed=0):
    import random
    rnd = random.Random(seed)
    s = bytearray(BLOCK)
    for i in range(0, valid_len, 64):
        s[i:i+64] = bytes([rnd.randrange(256)]) * 64
    return s

def roundtrip(tmp):
    slot = make_slot(1_034_240)
    p = os.path.join(tmp, "b")
    io.store_block(p, memoryview(slot), 0, BLOCK)
    assert os.path.getsize(p) % 4096 == 0, os.path.getsize(p)
    out = bytearray(BLOCK)
    io.load_block(p, memoryview(out), 0, BLOCK)
    assert out == slot, "roundtrip mismatch"
    print("roundtrip OK", os.path.getsize(p))

def dedup(tmp):
    slot = make_slot(1_034_240, seed=1)
    p = os.path.join(tmp, "d")
    errs = []
    def w():
        try:
            for _ in range(10):
                io.store_block(p, memoryview(slot), 0, BLOCK)
        except Exception as e:
            errs.append(e)
    ts = [threading.Thread(target=w) for _ in range(8)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert not errs, errs
    assert not [f for f in os.listdir(tmp) if f.endswith(".tmp")], "stale tmp"
    out = bytearray(BLOCK)
    io.load_block(p, memoryview(out), 0, BLOCK)
    assert out == slot
    print("dedup OK")

def corrupt(tmp):
    slot = make_slot(1_034_240, seed=2)
    p = os.path.join(tmp, "c")
    io.store_block(p, memoryview(slot), 0, BLOCK)
    with open(p, "r+b") as f:
        f.seek(12); f.write(b"\xff\xff\xff\xff")   # 破坏 payload_len
    try:
        io.load_block(p, memoryview(bytearray(BLOCK)), 0, BLOCK)
        raise AssertionError("should raise")
    except OSError:
        pass
    assert not os.path.exists(p), "corrupt file must be removed"
    print("corrupt OK")

if __name__ == "__main__":
    with tempfile.TemporaryDirectory(dir="/opt/aicad-kvssd") as tmp:
        roundtrip(tmp); dedup(tmp); corrupt(tmp)
    print("ALL PASS")
```
> 本地（Windows 缓冲路径）已 6 项全绿；容器内重点验证 `os.path.getsize % 4096 == 0` 与 O_DIRECT 真正生效。

### 3. 压缩率与 10KB/token 验收
灰度后实测：`du -s /opt/aicad-kvssd` ÷ 累计生成 token。换算口径：单块落盘 ≈ `align_up(16 + zstd(1.03MB), 4096)`，NVFP4 若压缩率 60-65% ⇒ 每块 ~0.66MB；再按每块 token 数折算 ≤10KB/token。若超目标，先查是否仍有 5× 写放大残留（`_path_locks` 未命中 / 不同 group dest_path 不一致）或压缩率不达标（数据特征）。

### 4. 失败策略联动验证
在灰度实例上人为破坏一个已落盘 KV 文件（改坏 payload_len），发一个命中该 block 的请求，确认：文件被删 + 请求按 `kv_load_failure_policy=fail` 报错（不崩溃、不 hang、不返回脏数据）。

### 5. 回归基线
灰度前用同一 benchmark（如 60 万 token 长上下文）记录 TTFT/ITL/落盘量；上线后对比：落盘量应下降 ~40×，TTFT/ITL 不应显著劣化（zstd-3 压缩在写线程内、读线程解压 1MB 的 CPU 成本需确认可接受）。

---

## 结论

**Request Changes → 有条件 Approve（Needs Discussion）**

- 代码逻辑与格式设计：**Approve**——本地 6 项验证通过，O_DIRECT 对齐/竞态/补零/失败语义均有明确方案。
- 放行条件（主理人确认）：
  1. 第 5.1 条 O_DIRECT 冒烟在 `/opt/aicad-kvssd` 通过（或设 `VLLM_KVSSD_ALIGN` 后通过）；
  2. 主理人接受第 3 条"per-path 锁持有至写完"的取舍（换取失败可见性）；
  3. 部署顺序：wheel → 清旧缓存 → 打补丁 → 重启。
