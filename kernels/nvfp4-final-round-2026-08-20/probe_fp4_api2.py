"""mm_fp4 + nvfp4_attention_sm120 签名探针。"""
import inspect
import flashinfer

for fn_name in ["mm_fp4", "nvfp4_attention_sm120", "nvfp4_attention_sm120_fwd",
                "nvfp4_attention_sm120_quantize_qkv", "nvfp4_kv_quantize"]:
    fn = getattr(flashinfer, fn_name, None)
    if fn is None:
        print(f"{fn_name}: NOT FOUND")
        continue
    try:
        sig = inspect.signature(fn)
        doc = (fn.__doc__ or "").strip().splitlines()
        print(f"{fn_name}{sig}")
        for d in doc[:6]:
            print(f"    {d}")
    except Exception as e:
        print(f"{fn_name}: {type(e).__name__} {str(e)[:80]}")
    print()
