"""Inspect B12xMoEWrapper in the production image (read-only)."""
import inspect
from flashinfer.fused_moe import B12xMoEWrapper

sig = inspect.signature(B12xMoEWrapper.__init__)
print("INIT_SIG:", sig)
print("FILE:", inspect.getsourcefile(B12xMoEWrapper))

src = inspect.getsource(B12xMoEWrapper).splitlines()
for pat in ["_allocate_buffers", "max_num_tokens", "def __init__",
            "self._workspace", "def run", "_moe_output", "data_ptr"]:
    lines = [i for i, l in enumerate(src) if pat in l]
    print(f"{pat!r} -> lines {lines[:10]}")

# show __init__ body
import re
in_init = False
for i, l in enumerate(src):
    if l.strip().startswith("def __init__"):
        in_init = True
    elif in_init and l.strip().startswith("def "):
        break
    if in_init:
        print(f"{i:4d}| {l}")
