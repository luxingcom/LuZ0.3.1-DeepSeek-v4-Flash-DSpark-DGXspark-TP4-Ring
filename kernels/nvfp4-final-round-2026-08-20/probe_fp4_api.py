"""方案 B 探针：FlashInfer FP4 API 签名（mm_fp4 / nvfp4_quantize / block_scale_interleave / prepare_bf16_fp4_weights）。"""
import inspect
import flashinfer

for fn_name in ["mm_fp4", "mm_bf16_fp4", "nvfp4_quantize", "nvfp4_batched_quantize",
                "block_scale_interleave", "nvfp4_block_scale_interleave",
                "prepare_bf16_fp4_weights", "scaled_fp4_grouped_quantize"]:
    fn = getattr(flashinfer, fn_name, None)
    if fn is None:
        print(f"{fn_name}: NOT FOUND")
        continue
    try:
        sig = inspect.signature(fn)
        doc = (fn.__doc__ or "").strip().splitlines()
        print(f"{fn_name}{sig}")
        for d in doc[:4]:
            print(f"    {d}")
    except Exception as e:
        print(f"{fn_name}: sig err {type(e).__name__} {str(e)[:60]}")
    print()
