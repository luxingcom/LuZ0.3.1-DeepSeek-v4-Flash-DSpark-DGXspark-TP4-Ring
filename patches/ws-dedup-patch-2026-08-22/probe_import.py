"""Test CPU-only import of the vllm flashinfer_b12x_moe module (read-only probe)."""
import sys
import importlib

try:
    m = importlib.import_module(
        "vllm.model_executor.layers.fused_moe.experts.flashinfer_b12x_moe")
    print("IMPORT_OK", m.__name__)
    print("HAS_CLASS:", hasattr(m, "FlashInferB12xExperts"))
    import inspect
    print("_ensure_wrapper src:")
    print(inspect.getsource(m.FlashInferB12xExperts._ensure_wrapper))
except Exception as e:
    import traceback
    traceback.print_exc()
    print("IMPORT_FAIL:", type(e).__name__, e)
    sys.exit(1)
