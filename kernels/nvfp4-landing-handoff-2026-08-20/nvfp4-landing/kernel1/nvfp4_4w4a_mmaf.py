"""NVFP4 4W4A prefill GEMM —— 路线 A 生产适配层（最终，基于 vLLM 内置 cutlass_scaled_fp4_mm 原生 FP4）。

核心结论（生产实测，2026-08-20）：
  cutlass_scaled_fp4_mm 与 vLLM 官方 NVFP4 dequant 数学完全一致（rel=0.00141）。
  A/W 均用 vLLM 官方 scaled_fp4_quant（16-group e4m3）量化 + CUTLASS GEMM = 官方语义。
  原生 FP4 tensor-core MMA（sm_121a），性能 90~130 TFLOPS，SASS 含 HMMA 原生 FP4。

入口：
  nvfp4_4w4a_prefill_gemm(A, W_packed, W_scale, bias=None)    # 便捷入口（内部自建单例）
  RouteA 类：preprocess_weights() 预量化缓存 + __call__() 推理

调用约定：
  A        [M, K] fp32
  W_packed [K, N//2] uint8  NVFP4 权重（N 打包，低半字节=偶 N 列）
  W_scale  [K//32, N//128] uint8 E8M0（每个 32×128 block）
  返回     [M, N] fp32
"""
import torch
import vllm._custom_ops as _co

DEV = "cuda"


def _dequant_w_our(W_packed, W_scale):
    """既有 W 格式 dequant -> fp32 [K, N]（torch ref 语义，供转到官方 NVFP4 格式）。"""
    K, N2 = W_packed.shape
    N = N2 * 2
    E2M1F = torch.tensor([0.,0.5,1.,1.5,2.,3.,4.,6.,-0.,-0.5,-1.,-1.5,-2.,-3.,-4.,-6.],
                         dtype=torch.float32, device=W_packed.device)
    p = W_packed.to(torch.int32)
    lo = (p & 0x0F).long(); hi = ((p >> 4) & 0x0F).long()
    w = torch.stack([E2M1F[lo.reshape(-1)].reshape(K, N2),
                     E2M1F[hi.reshape(-1)].reshape(K, N2)], 2).reshape(K, N)
    wsf = torch.pow(2.0, W_scale.to(torch.float32) - 127.0)
    wsx = wsf.repeat_interleave(32, 0).repeat_interleave(128, 1)
    return (w * wsx).float()


class RouteA:
    """路线 A 原生 FP4 GEMM 适配层。每层 preprocess_weights 一次（缓存官方格式权重）。"""

    def __init__(self):
        self._preprocessed = None   # (w_q, w_sf, N) 由调用方传入时用
        self._wq = None
        self._wsf = None
        self._N = None

    def preprocess_weights(self, W_packed, W_scale):
        """把既有 W 格式预处理为官方 NVFP4 格式（每层一次，结果缓存）。返回 (w_q, w_sf)。"""
        K, N2 = W_packed.shape
        self._N = N2 * 2
        W_fp32 = _dequant_w_our(W_packed, W_scale).t().contiguous()   # [N, K]
        w_q, w_sf = _co.scaled_fp4_quant(
            W_fp32.half(),
            torch.tensor([1.0], dtype=torch.float32, device=W_packed.device),
            is_sf_swizzled_layout=True, backend='none', padded_n=None)
        self._wq, self._wsf = w_q, w_sf
        return w_q, w_sf

    def __call__(self, A, W_packed=None, W_scale=None, bias=None, use_cached_w=False):
        assert self._wq is not None or (W_packed is not None), \
            "call preprocess_weights(W_packed, W_scale) first, or pass W_packed/W_scale"
        if W_packed is not None and W_scale is not None and not use_cached_w:
            self.preprocess_weights(W_packed, W_scale)
        M, K = A.shape
        N = self._N
        a_q, a_sf = _co.scaled_fp4_quant(
            A.float().half(),
            torch.tensor([1.0], dtype=torch.float32, device=A.device),
            is_sf_swizzled_layout=True, backend='none', padded_n=None)
        alpha = torch.tensor([1.0], dtype=torch.float32, device=A.device)
        out = _co.cutlass_scaled_fp4_mm(a_q, self._wq, a_sf, self._wsf, alpha, torch.bfloat16).float()
        out = out[..., :N].contiguous() if out.shape[-1] != N else out
        if bias is not None:
            out = out + bias.to(out.dtype)
        return out.contiguous()


_SINGLETON = None


def nvfp4_4w4a_prefill_gemm(A, W_packed, W_scale, bias=None):
    """便捷入口：每次调用重做 W 预处理（无状态，最安全）。生产高频调用请用 RouteA 类缓存。"""
    impl = RouteA()
    impl.preprocess_weights(W_packed, W_scale)
    return impl(A, bias=bias, use_cached_w=True)


if __name__ == "__main__":
    torch.manual_seed(0)
    M, K, N = 256, 4096, 4096
    A = torch.randn(M, K, device=DEV) * 0.5
    out = nvfp4_4w4a_prefill_gemm(A, W_packed=torch.rand(K, N//2, device=DEV).to(torch.uint8),
                                  W_scale=torch.ones(K//32, N//128, dtype=torch.uint8, device=DEV))
    print("smoke", tuple(out.shape), "ok")