"""FlashInfer NVFP4/FP4 能力探针（方案 B 首选路径可行性）。"""
import importlib

try:
    import flashinfer
    print("flashinfer version:", flashinfer.__version__)
    print("modules:", [m for m in dir(flashinfer) if not m.startswith("_")])
except Exception as e:
    print("flashinfer import fail:", type(e).__name__, str(e)[:100])
    raise SystemExit

for modname in ["flashinfer.quant", "flashinfer.nvfp4", "flashinfer.fp4", "flashinfer.fp8"]:
    try:
        m = importlib.import_module(modname)
        names = [x for x in dir(m) if not x.startswith("_")]
        print(f"{modname} OK: {names[:20]}")
    except Exception as e:
        print(f"{modname} FAIL: {type(e).__name__} {str(e)[:80]}")
