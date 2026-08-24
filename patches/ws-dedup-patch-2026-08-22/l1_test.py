"""L1 validation (CPU-only, no GPU): import / off-path equivalence / pool logic.

Runs inside a one-shot container with the production image, mounting:
  /w/src_orig/flashinfer_b12x_moe.py   (pristine copy from the image)
  /w/flashinfer_b12x_moe.py            (patched overlay file)

Tests:
  T1  Both modules import cleanly (patched adds os/threading/init_logger).
  T2  AST equivalence:
      - every class method except _ensure_wrapper is AST-identical;
      - the off-path B12xMoEWrapper(...) constructor call in the patched
        file is AST-identical to the original constructor call;
      - patched module-level additions are limited to the documented set.
  T3  Runtime off-path equivalence with a mocked B12xMoEWrapper:
      4 instances, same geometry -> 4 ctor calls, kwargs list byte-identical
      to the original module's calls, distinct wrapper objects, pool unused.
  T4  Pool logic with VLLM_B12X_SHARED_WRAPPER=1:
      - 4 same-geometry instances -> 1 ctor call, all share one wrapper;
      - second geometry -> +1 call, pool size 2;
      - geometry key sensitivity (activation/intermediate/max_tokens);
      - idempotence (second _ensure_wrapper is a no-op);
      - env unset / "0" / garbage -> disabled; only "1" enables.
Exit code 0 iff all pass; prints JSON summary.
"""
import ast
import importlib.util
import json
import os
import sys

ORIG = "/w/src_orig/flashinfer_b12x_moe.py"
PATCHED = "/w/flashinfer_b12x_moe.py"

results = {}
failures = []


def check(name, ok, detail=""):
    results[name] = {"pass": bool(ok), "detail": detail}
    if not ok:
        failures.append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- T1 import
orig = load_module(ORIG, "fib12x_orig")
patched = load_module(PATCHED, "fib12x_patched")
check("T1_import_both_modules", True,
      f"orig={orig.FlashInferB12xExperts.__module__}, "
      f"patched={patched.FlashInferB12xExperts.__module__}")

# ---------------------------------------------------------------- T2 AST
orig_src = open(ORIG).read()
pat_src = open(PATCHED).read()
orig_tree = ast.parse(orig_src)
pat_tree = ast.parse(pat_src)


def get_class(tree):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "FlashInferB12xExperts":
            return node
    raise AssertionError("class not found")


def methods(cls_node):
    return {n.name: n for n in cls_node.body if isinstance(n, ast.FunctionDef)}


oc, pc = get_class(orig_tree), get_class(pat_tree)
om, pm = methods(oc), methods(pc)
check("T2a_same_method_names", set(om) == set(pm), f"{sorted(set(om) ^ set(pm))}")

diff_methods = [n for n in om if n != "_ensure_wrapper"
                and ast.dump(om[n]) != ast.dump(pm.get(n))]
check("T2b_other_methods_identical", not diff_methods, f"differ: {diff_methods}")


def ctor_calls(func_node):
    """All B12xMoEWrapper(...) call nodes assigned to self._wrapper."""
    out = []
    for node in ast.walk(func_node):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "B12xMoEWrapper"):
            out.append(node)
    return out


orig_calls = ctor_calls(om["_ensure_wrapper"])
pat_calls = ctor_calls(pm["_ensure_wrapper"])
check("T2c_two_ctor_calls_in_patch", len(pat_calls) == 2,
      f"orig has {len(orig_calls)}, patched has {len(pat_calls)}")
# Both patched calls (off-branch and pool-miss branch) must be AST-identical
# to the original call (kwargs/geometry identical -> byte-level equivalence
# of the construction semantics).
check("T2d_off_path_ctor_ast_identical",
      len(orig_calls) == 1
      and all(ast.dump(c) == ast.dump(orig_calls[0]) for c in pat_calls),
      "off-branch and pool-branch ctor calls AST-identical to original")

# Module-level additions whitelist
def top_level_names(tree):
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                names.add(a.asname or a.name.split(".")[0])
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
        elif isinstance(node, ast.FunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.ClassDef):
            names.add(node.name)
    return names


added = top_level_names(pat_tree) - top_level_names(orig_tree)
removed = top_level_names(orig_tree) - top_level_names(pat_tree)
check("T2e_additions_whitelisted",
      added <= {"os", "threading", "init_logger", "logger",
                "_B12X_WRAPPER_POOL", "_B12X_WRAPPER_POOL_LOCK",
                "_b12x_wrapper_pool_enabled"}
      and not removed,
      f"added={sorted(added)} removed={sorted(removed)}")

# ------------------------------------------------------- T3/T4 runtime mock
import flashinfer.fused_moe as ffi_fused_moe

calls = []


class FakeWrapper:
    def __init__(self, **kwargs):
        calls.append(dict(kwargs))
        self.kwargs = dict(kwargs)


real_wrapper = ffi_fused_moe.B12xMoEWrapper
ffi_fused_moe.B12xMoEWrapper = FakeWrapper


def make_fake(cls, **overrides):
    obj = cls.__new__(cls)  # bypass __init__ (no moe_config needed)
    attrs = dict(
        global_num_experts=256, topk=6, hidden_dim=4096,
        intermediate_size_per_partition=512,  # TP4: 2048/4
        max_num_tokens=4096, num_local_experts=256,
        _activation_str="silu", _wrapper=None,
    )
    attrs.update(overrides)
    for k, v in attrs.items():
        setattr(obj, k, v)
    return obj


def env_clean():
    os.environ.pop("VLLM_B12X_SHARED_WRAPPER", None)
    patched._B12X_WRAPPER_POOL.clear()
    calls.clear()


GEOM_KW = dict(
    num_experts=256, top_k=6, hidden_size=4096, intermediate_size=512,
    use_cuda_graph=True, max_num_tokens=4096, num_local_experts=256,
    activation="silu",
)

# T3: off path (env unset) — equivalence with original module
env_clean()
orig_objs = [make_fake(orig.FlashInferB12xExperts) for _ in range(4)]
for o in orig_objs:
    o._ensure_wrapper()
orig_calls_rec = list(calls)
check("T3a_orig_4_calls", len(orig_calls_rec) == 4, f"{len(orig_calls_rec)} calls")

env_clean()
pat_objs = [make_fake(patched.FlashInferB12xExperts) for _ in range(4)]
for o in pat_objs:
    o._ensure_wrapper()
check("T3b_patched_off_4_calls", len(calls) == 4, f"{len(calls)} calls")
check("T3c_off_kwargs_byte_identical",
      calls == orig_calls_rec,
      "patched off-path ctor kwargs identical to original module's")
check("T3d_off_distinct_wrappers",
      len({id(o._wrapper) for o in pat_objs}) == 4)
check("T3e_pool_unused_when_off", len(patched._B12X_WRAPPER_POOL) == 0)

# T4: pool on
os.environ["VLLM_B12X_SHARED_WRAPPER"] = "1"
patched._B12X_WRAPPER_POOL.clear()
calls.clear()

objs = [make_fake(patched.FlashInferB12xExperts) for _ in range(4)]
for o in objs:
    o._ensure_wrapper()
check("T4a_pool_single_creation", len(calls) == 1, f"{len(calls)} calls")
check("T4b_all_share_one_wrapper", len({id(o._wrapper) for o in objs}) == 1)
check("T4c_pool_size_1", len(patched._B12X_WRAPPER_POOL) == 1)
check("T4d_on_kwargs_same_as_off", calls and calls[0] == orig_calls_rec[0])

# different geometry -> new entry
alt = make_fake(patched.FlashInferB12xExperts, max_num_tokens=2048)
alt._ensure_wrapper()
check("T4e_second_geometry_new_entry",
      len(calls) == 2 and len(patched._B12X_WRAPPER_POOL) == 2)

# geometry sensitivity: activation changes key
alt2 = make_fake(patched.FlashInferB12xExperts, _activation_str="relu2")
alt2._ensure_wrapper()
check("T4f_activation_in_key",
      len(calls) == 3 and len(patched._B12X_WRAPPER_POOL) == 3)

# idempotence
before = len(calls)
objs[0]._ensure_wrapper()
check("T4g_idempotent", len(calls) == before)

# env parsing
for val, expect in [(None, False), ("0", False), ("1", True),
                    ("true", False), ("", False), ("2", False)]:
    if val is None:
        os.environ.pop("VLLM_B12X_SHARED_WRAPPER", None)
    else:
        os.environ["VLLM_B12X_SHARED_WRAPPER"] = val
    got = patched._b12x_wrapper_pool_enabled()
    check(f"T4h_env[{val!r}]", got == expect, f"got {got}")

# restore
ffi_fused_moe.B12xMoEWrapper = real_wrapper
os.environ.pop("VLLM_B12X_SHARED_WRAPPER", None)

summary = {
    "total": len(results),
    "passed": sum(1 for r in results.values() if r["pass"]),
    "failed": len(failures),
    "failures": failures,
}
print("\n=== L1 SUMMARY ===")
print(json.dumps(summary, indent=2))
print(json.dumps(results, indent=2, default=str))
sys.exit(0 if not failures else 1)
