#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""merged_experts.py v2.1 — Triton 长尾架构（fix-rerun 版, 2026-08-22）。
B12X 完全退出:
  - process_weights_after_loading: 不调 super()（B12X 就地重打包会毁掉零拷贝 payload 视图）
    —— 派生: stacked payload 零拷贝视图 + w13 swizzled E4M3 scale 壳（壳背衬即
    物理存储, merged DSL 与 Triton 共用字节）
  - prefill (M ≥ MIN_M): exact-set 桶 ≥ MIN_M → merged GEMM（w13 N-merge + w2
    K-concat combine 折叠, M_pad 档位量化 + 两级 scale A 量化）; 其余行 → Triton T3′
  - decode / 小 M: 全部 → Triton per-expert（E8M0 逻辑直寻址, 性能未验证口径）

v2.1 修复（对照 v2.0 冒烟版）:
  [P0-A] _derive 期 M_pad 档预编译（~14 档 × ~2.5s ≈ 35s 启动吸收;
         VLLM_MOE_MERGED_WARMUP=0 可关）
  [P0-B] A 量化两级 scale（dsl_gemm.nvfp4_quant two_level, GEMM 输出 ÷g）
  [P1-A] 派生内存瘦身:
         - w2 E4M3 派生驻留删除（Triton 改吃 E8M0 逻辑 scale 零拷贝视图;
           combo SF 建 combo 时一次派生）
         - _s2_logical 驻留删除（combo 创建时按需派生 6 expert 切片）
         - SF 壳/c 壳复用（v2.0 _HOLDS 每调用泄漏 ~67MB 的根因修复）
内存: payload 零拷贝 + w13 E4M3 壳 67MB/层（即物理存储本体）+ combo LRU
      cap 8/层（pay 3MB + SF 壳 0.75MB each）。"""
import os

import torch

from vllm.model_executor.layers.fused_moe.experts.b12x_mxfp4_moe import B12xExperts
from vllm.logger import init_logger

logger = init_logger(__name__)

_dg = None
_tm = None
_WARMED = {}       # 进程级: DSL M_pad 档预编译只做一次


def _libs():
    global _dg, _tm
    if _dg is None:
        from routeb_merged_plugin import dsl_gemm as _d
        from routeb_merged_plugin import triton_moe as _t
        _dg, _tm = _d, _t
    return _dg, _tm


class MergedB12xExperts(B12xExperts):
    def __init__(self, moe_config, quant_config):
        super().__init__(moe_config, quant_config)
        self._mode = os.environ.get("VLLM_MOE_MERGED", "0")
        self._min_m = int(os.environ.get("VLLM_MOE_MERGED_MIN_M", "128"))
        self._ready = False
        self._w2_cap = int(os.environ.get("VLLM_MOE_MERGED_W2_CAP", "8"))
        self._stats = {"apply": 0, "merged_apply": 0, "merged_rows": 0,
                       "triton_rows": 0, "buckets": 0}

    # ---------------- 权重派生（零拷贝 + scale 物理） ----------------
    def _derive(self, layer) -> None:
        dg, _ = _libs()
        w13 = layer.w13_weight.data            # [E, N13, K/2] u8（[w1;w3] 行序）
        w2 = layer.w2_weight.data              # [E, N2, K2/2] u8
        s13 = layer.w13_weight_scale.data      # [E, N13, K/32] E8M0
        s2 = layer.w2_weight_scale.data        # [E, N2, K2/32] E8M0
        E, N13, Kh = w13.shape
        K = Kh * 2
        _, N2, Kh2 = w2.shape
        K2 = Kh2 * 2
        self._E, self._N13, self._K = E, N13, K
        self._N2, self._K2 = N2, K2

        # payload 零拷贝 flat 视图
        self._w13_pay = w13.reshape(E * N13, Kh)          # 视图（参数存储）
        self._w2_pay = w2.reshape(E * N2, Kh2)

        # w13 scale: E8M0 → E4M3 K/16 → swizzle 壳（一次; 壳背衬 = 物理存储,
        # back flat 视图给 Triton 共用; 免 v2.0 的 phys+壳双份驻留）
        s13_e4 = dg.e8m0_to_e4m3_k16(s13.reshape(E * N13, K // 32))
        s13_phys, rm13, rk13 = dg.swizzle_scales_u8(s13_e4)
        self._w13_sf_cute, self._w13_sf = dg.make_sf_cute(s13_phys, rm13, rk13)
        del s13_e4, s13_phys

        # w2 scale: 不做全量 E4M3 派生驻留——Triton 吃 E8M0 逻辑零拷贝;
        # combo 的 E4M3 在 _w2_combo_get 按需派生（6 expert 切片, 小)
        self._s2_src = s2.reshape(E * N2, K2 // 32)       # E8M0 零拷贝视图
        self._s13_src = s13                               # E8M0 原始（selfcheck）

        self._w2_combo = {}
        self._w2_combo_order = []
        self._tiles13 = N13 // 128
        self._ready = True

        # Triton 内核预编译（权重加载期, CUDA graph 捕获前——首次 JIT 编译若发生在
        # capture 中会破坏捕获, 实证 2026-08-21 e2e）
        try:
            _, tm = _libs()
            dev = w13.device
            x2 = torch.zeros(2, K, dtype=torch.bfloat16, device=dev)
            ti2 = torch.zeros(2, 6, dtype=torch.int32, device=dev)
            tw2 = torch.ones(2, 6, dtype=torch.float32, device=dev)
            tm.triton_moe(x2, ti2, tw2, self._w13_pay, self._w13_sf, N13,
                          self._w2_pay, self._s2_src, N2, K,
                          E * N13, E * N2)
            torch.cuda.synchronize()
            logger.info("[routeb_merged] Triton kernels warmed up (capture 安全)")
        except Exception:
            logger.exception("[routeb_merged] Triton warmup 失败")

        # [P0-A] DSL M_pad 档预编译（进程级一次; ~2.5s/档实测）
        if os.environ.get("VLLM_MOE_MERGED_WARMUP", "1") == "1" \
                and not _WARMED.get("dsl"):
            _WARMED["dsl"] = True
            try:
                t0 = __import__("time").time()
                self._dsl_warmup()
                logger.info("[routeb_merged] DSL M_pad 档预编译完成 %.1fs "
                            "(tiers=%s stats=%s)", __import__("time").time() - t0,
                            list(dg.M_TIERS), dg.stats())
            except Exception:
                logger.exception("[routeb_merged] DSL 预编译失败（首 chunk 将现场编译）")

        logger.info("[routeb_merged] layer ready: E=%d N13=%d K=%d N2=%d K2=%d "
                    "(payload 零拷贝; w13 SF 壳 %.0fMB)",
                    E, N13, K, N2, K2, self._w13_sf.numel() / 1e6)

    def _dsl_warmup(self):
        """M_pad 档 × (w13, w2) 预编译。w13 用真实权重 + dummy A; w2 用 dummy
        combo（shape 同真实: b_rows=N2, K=nc·K2 → 编译 key 内容无关, 真实 combo
        复用同一编译产物）。"""
        dg, _ = _libs()
        dev = self._w13_pay.device
        nc = 6   # topk=6 精确集桶的标称档（nc<6 罕见, 现场编译兜底 ~2.5s）
        for tier in dg.M_TIERS:
            a = torch.randn(tier, self._K, device=dev).bfloat16() * 0.5
            a_pay, a_sc, _ = dg.nvfp4_quant(a)
            tm13 = torch.cat([
                torch.arange(e * self._tiles13, (e + 1) * self._tiles13,
                             dtype=torch.int32) for e in range(nc)]).cuda()
            dg.merged_gemm(a_pay, a_sc, self._w13_pay, self._w13_sf_cute,
                           self._K, nc * self._N13, tm13, m_pad_tier=tier)
            # w2: dummy combo
            K2g = nc * self._K2
            b2 = torch.randint(0, 256, (self._N2, K2g // 2), dtype=torch.uint8,
                               device=dev)
            b2_sc = torch.randint(120, 136, (self._N2, K2g // 16), dtype=torch.uint8,
                                  device=dev)
            b2_phys, rm2, rk2 = dg.swizzle_scales_u8(b2_sc)
            b2_cute, _ = dg.make_sf_cute(b2_phys, rm2, rk2)
            a2 = torch.randn(tier, K2g, device=dev).bfloat16() * 0.5
            a2_pay, a2_sc, _ = dg.nvfp4_quant(a2)
            dg.merged_gemm(a2_pay, a2_sc, b2, b2_cute, K2g, self._N2,
                           None, m_pad_tier=tier)
        torch.cuda.synchronize()

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self._mode != "1":
            super().process_weights_after_loading(layer)
            return
        if self._ready:
            return
        self._derive(layer)
        # 不调 super(): B12X 重打包退出（原始 payload 保持零拷贝共享）

    # ---------------- w2 K-concat combo 缓存 ----------------
    def _w2_combo_get(self, combo):
        if combo in self._w2_combo:
            return self._w2_combo[combo]
        dg, _ = _libs()
        N2, K2 = self._N2, self._K2
        pay = torch.cat([self._w2_pay[e * N2:(e + 1) * N2] for e in combo], dim=1)
        # scale: 每 expert 的 E8M0 → E4M3 K/16 切片 cat → swizzle 壳（一次）
        sc_e4 = dg.e8m0_to_e4m3_k16(
            torch.cat([self._s2_src[e * N2:(e + 1) * N2] for e in combo], dim=1))
        sc_phys, rm, rk = dg.swizzle_scales_u8(sc_e4)
        sf_cute, _ = dg.make_sf_cute(sc_phys, rm, rk)
        item = (pay.contiguous(), sf_cute)
        self._w2_combo[combo] = item
        self._w2_combo_order.append(combo)
        while len(self._w2_combo_order) > self._w2_cap:
            old = self._w2_combo_order.pop(0)
            self._w2_combo.pop(old, None)
        return item

    # ---------------- merged 桶前向 ----------------
    def _merged_bucket(self, hidden_rows, combo, w_rows):
        dg, _ = _libs()
        M_g = hidden_rows.shape[0]
        nc = len(combo)
        tier = dg.pick_tier(M_g)
        a_pay, a_sc, g1 = dg.nvfp4_quant(hidden_rows)
        tile_map = torch.cat([
            torch.arange(e * self._tiles13, (e + 1) * self._tiles13,
                         dtype=torch.int32) for e in combo])
        C13 = dg.merged_gemm(a_pay, a_sc, self._w13_pay, self._w13_sf_cute,
                             self._K, nc * self._N13, tile_map,
                             m_pad_tier=tier) / g1
        # act + 权重折叠 + K 拼接
        I = self._N13 // 2
        acts = []
        for i in range(nc):
            blk = C13[:, i * self._N13:(i + 1) * self._N13]
            acts.append(torch.nn.functional.silu(blk[:, :I]) * blk[:, I:])
        A2 = torch.cat([acts[i] * w_rows[:, i:i + 1] for i in range(nc)], dim=1)
        a2_pay, a2_sc, g2 = dg.nvfp4_quant(A2)
        b2_pay, b2_sf_cute = self._w2_combo_get(combo)
        C2 = dg.merged_gemm(a2_pay, a2_sc, b2_pay, b2_sf_cute,
                            nc * self._K2, self._N2, None,
                            m_pad_tier=tier) / g2
        return C2

    def _selfcheck(self, x_rows, combo, w_rows, out_m):
        """torch 反量化参考 vs merged 输出（首个桶, 一次性）。"""
        try:
            E, N13, K = self._E, self._N13, self._K
            N2, K2 = self._N2, self._K2
            I = N13 // 2
            x = x_rows.float()
            ref = torch.zeros(x.shape[0], N2, device=x.device)
            def _scale_f32(src, e, rows, k):
                s = src[e].reshape(rows, k // 32).float()      # E8M0 字节
                s = torch.exp2(s - 127.0)
                return s.repeat_interleave(2, dim=1)            # [rows, k/16]

            for i, e in enumerate(combo):
                w13 = self._dequant(self._w13_pay[e * N13:(e + 1) * N13], K) \
                    * _scale_f32(self._s13_src, e, N13, K).repeat_interleave(16, dim=1)
                h = x @ w13.t()
                act = torch.nn.functional.silu(h[:, :I]) * h[:, I:]
                w2 = self._dequant(self._w2_pay[e * N2:(e + 1) * N2], K2) \
                    * _scale_f32(self._s2_src.view(self._E, N2, K2 // 32), e, N2, K2).repeat_interleave(16, dim=1)
                ref += w_rows[:, i:i + 1] * (act @ w2.t())
            rel = (out_m - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
            print(f"[routeb_merged SELFCHECK] merged vs torch反量化参考 "
                  f"rel={rel:.3e} combo={combo}", flush=True)
        except Exception:
            import traceback
            traceback.print_exc()

    def _dequant(self, payload_rows, K):
        """[N, K/2] u8 → f32 [N, K]（scale 由调用方另乘）。"""
        tab = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                           device=payload_rows.device)
        lo = tab[(payload_rows & 0x0F).long()]
        hi = tab[(payload_rows >> 4).long()]
        out = torch.empty(payload_rows.shape[0], K, dtype=torch.float32,
                          device=payload_rows.device)
        out[:, 0::2] = lo
        out[:, 1::2] = hi
        return out

    # ---------------- forward ----------------
    def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids,
              activation, global_num_experts, expert_map, a1q_scale, a2_scale,
              workspace13, workspace2, expert_tokens_meta,
              apply_router_weight_on_input):
        M, K = hidden_states.shape
        self._stats["apply"] += 1
        if self._mode != "1" or not self._ready or expert_map is not None:
            return super().apply(
                output, hidden_states, w1, w2, topk_weights, topk_ids, activation,
                global_num_experts, expert_map, a1q_scale, a2_scale, workspace13,
                workspace2, expert_tokens_meta, apply_router_weight_on_input)
        dg, tm = _libs()
        dev = hidden_states.device
        ti = topk_ids.long()
        tw = topk_weights.float()
        _dbg = os.environ.get("VLLM_MOE_MERGED_DEBUG", "0") == "1"

        use_merged = (M >= self._min_m) and (not apply_router_weight_on_input)
        if use_merged:
            merged_mask = torch.zeros(M, dtype=torch.bool, device=dev)
            # ---- 桶分组（exact-set; sorted 权重对齐）—— 仅 prefill（M≥min_m）,
            # CUDA graph 捕获（decode M<min_m）不进入此分支 ----
            sorted_ids, sort_idx = torch.sort(ti, dim=1)
            w_sorted = torch.gather(tw, 1, sort_idx)
            # 无效 expert（<0 或 ≥E）的行整行逐出桶（Triton 路径零权重兜底）
            _bad = ((sorted_ids < 0) | (sorted_ids >= self._E)).any(dim=1)
            # key = 纯标量乘加（capture 安全, 无 CPU→GPU 张量创建）
            key_bytes = torch.zeros_like(sorted_ids[:, 0])
            for _i in range(sorted_ids.shape[1]):
                key_bytes = key_bytes * 256 + sorted_ids[:, _i]
            key_bytes = torch.where(_bad, torch.full_like(key_bytes, -1), key_bytes)
            uniq, inv, cnt = torch.unique(
                key_bytes, return_inverse=True, return_counts=True)
            if _dbg:
                print(f"[routeb_merged] apply M={M} uniq={len(uniq)} "
                      f"cnt_top={int(cnt.max())} arwoi={apply_router_weight_on_input}",
                      flush=True)
            for u in (cnt >= self._min_m).nonzero().flatten().tolist():
                if int(uniq[u]) < 0:
                    continue    # 无效行桶跳过（走 Triton 零权重）
                rows = (inv == u).nonzero().flatten()
                combo = tuple(sorted_ids[rows[0]].cpu().tolist())
                try:
                    out_m = self._merged_bucket(
                        hidden_states[rows], combo, w_sorted[rows])
                    if os.environ.get("VLLM_MOE_MERGED_SELFCHECK", "0") == "1" \
                            and not getattr(self, "_selfchecked", False):
                        self._selfchecked = True
                        self._selfcheck(hidden_states[rows], combo,
                                        w_sorted[rows], out_m)
                    output[rows] = out_m.to(output.dtype)
                    merged_mask[rows] = True
                    if _dbg:
                        print(f"[routeb_merged] bucket combo={combo} M_g={rows.numel()}"
                              f" 合并完成", flush=True)
                    self._stats["buckets"] += 1
                    self._stats["merged_rows"] += rows.numel()
                except Exception:
                    logger.exception("[routeb_merged] merged bucket 失败, 该桶回退 Triton")
            if merged_mask.any():
                self._stats["merged_apply"] += 1

        # ---- Triton 路径: 非 merged 行（长尾 prefill / decode 全量） ----
        # 捕获安全: 全程无 host 同步（.any()/.item() 会 cudaErrorStreamCaptureUnsupported）
        try:
            if use_merged:
                lt = ~merged_mask
                out_t = tm.triton_moe(
                    hidden_states[lt].contiguous(), topk_ids[lt].contiguous(),
                    tw[lt].contiguous(),
                    self._w13_pay, self._w13_sf, self._N13,
                    self._w2_pay, self._s2_src, self._N2,
                    self._K, self._E * self._N13, self._E * self._N2)
                output[lt] = out_t.to(output.dtype)
                self._stats["triton_rows"] += int(lt.sum())
            else:
                out_t = tm.triton_moe(
                    hidden_states, topk_ids, tw,
                    self._w13_pay, self._w13_sf, self._N13,
                    self._w2_pay, self._s2_src, self._N2,
                    self._K, self._E * self._N13, self._E * self._N2)
                output.copy_(out_t.to(output.dtype))
                self._stats["triton_rows"] += M
        except Exception:
            logger.exception("[routeb_merged] Triton 路径失败, 回退 B12X 生产路径")
            return super().apply(
                output, hidden_states, w1, w2, topk_weights, topk_ids, activation,
                global_num_experts, expert_map, a1q_scale, a2_scale, workspace13,
                workspace2, expert_tokens_meta, apply_router_weight_on_input)
        return None

    def merged_stats(self):
        return dict(self._stats)
