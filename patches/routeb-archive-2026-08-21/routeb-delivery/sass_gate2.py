#!/usr/bin/env python3
"""SASS gate v3: intercept cute.compile return object, introspect for cubin/PTX,
save artifacts, then disassemble with cuobjdump/nvdisasm (done by shell wrapper).
"""
import os
import sys

os.environ.setdefault("CUTE_DSL_ARCH", "sm_121a")
os.environ.setdefault("ROUTEB_DIAG", "ones")
os.environ.pop("ROUTEB_DUMP", None)
os.environ.pop("ROUTEB_SENTINEL", None)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "routeb_official"))

import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
import cutlass.cute.testing as _cute_testing  # noqa: E402

if not hasattr(cutlass, "testing"):
    cutlass.testing = _cute_testing
    sys.modules["cutlass.testing"] = _cute_testing

CAPTURED = []


def _spy_compile(*args, **kwargs):
    obj = _orig_compile(*args, **kwargs)
    CAPTURED.append(obj)
    return obj


_orig_compile = cute.compile
cute.compile = _spy_compile

# also spy cutlass_torch.convert / testing.convert kernels? main target is the GEMM.
import dense_blockscaled_gemm_persistent_pingpong as pp  # noqa: E402


def find_blobs(obj, prefix, out_dir, report):
    """Recursively look for cubin bytes / paths / PTX strings on the object."""
    import gzip
    import zlib

    seen = set()

    def walk(o, path, depth):
        if depth > 4 or id(o) in seen:
            return
        seen.add(id(o))
        # bytes blobs -> ELF?
        if isinstance(o, (bytes, bytearray)):
            b = bytes(o)
            if len(b) > 1024:
                off = b.find(b"\x7fELF")
                tag = "ELF" if off >= 0 else ("GZ" if b[:2] == b"\x1f\x8b" else ("ZLIB" if b[:2] == b"\x78\x9c" else "raw"))
                report.append(f"  blob {path}: len={len(b)} head={b[:8].hex()} tag={tag}")
                data = b
                try:
                    data = zlib.decompress(b)
                except Exception:
                    try:
                        data = gzip.decompress(b)
                    except Exception:
                        pass
                off = data.find(b"\x7fELF")
                if off >= 0:
                    fn = os.path.join(out_dir, f"{prefix}_{len(os.listdir(out_dir))}.cubin")
                    open(fn, "wb").write(data[off:])
                    report.append(f"    -> wrote {fn} ({len(data) - off} bytes)")
                return
        # strings -> PTX or file path?
        if isinstance(o, str):
            if len(o) > 4096 and (".version" in o[:2000] or "ptx" in o[:200].lower()):
                fn = os.path.join(out_dir, f"{prefix}_{len(os.listdir(out_dir))}.ptx")
                open(fn, "w").write(o)
                report.append(f"  ptx-string {path}: len={len(o)} -> {fn}")
            elif o.endswith(".cubin") and os.path.exists(o):
                import shutil
                fn = os.path.join(out_dir, f"{prefix}_{len(os.listdir(out_dir))}.cubin")
                shutil.copy(o, fn)
                report.append(f"  cubin-path {path}: {o} -> {fn}")
            return
        # dict / list / tuple / object attrs
        if isinstance(o, dict):
            for k, v in list(o.items())[:200]:
                if isinstance(k, str):
                    walk(v, f"{path}.{k}", depth + 1)
            return
        if isinstance(o, (list, tuple)):
            for i, v in enumerate(list(o)[:200]):
                walk(v, f"{path}[{i}]", depth + 1)
            return
        # plain object: walk attributes
        for name in dir(o):
            if name.startswith("_") and not name.startswith("__"):
                continue
            try:
                val = getattr(o, name)
            except Exception:
                continue
            if callable(val) and not isinstance(val, (bytes, str)):
                continue
            walk(val, f"{path}.{name}", depth + 1)

    walk(obj, prefix, 0)


def probe_obj(obj, idx, out_dir, report):
    """Explicitly probe known attribute names for cubin/ptx artifacts."""
    import inspect

    def rec(name, val):
        if isinstance(val, (bytes, bytearray)) and len(val) > 512:
            b = bytes(val)
            off = b.find(b"\x7fELF")
            report.append(f"  [obj{idx}] {name}: bytes len={len(b)} head={b[:8].hex()} ELF@{off}")
            if off >= 0:
                fn = os.path.join(out_dir, f"obj{idx}_{name.strip('_')}.cubin")
                open(fn, "wb").write(b[off:])
                report.append(f"    -> wrote {fn} ({len(b) - off} bytes)")
        elif isinstance(val, str) and len(val) > 4096:
            tag = "PTX" if ".version" in val[:2000] or "target sm_" in val[:4000] else "STR"
            fn = os.path.join(out_dir, f"obj{idx}_{name.strip('_')}.{tag.lower()}")
            open(fn, "w").write(val)
            report.append(f"  [obj{idx}] {name}: str len={len(val)} tag={tag} -> {fn}")
        elif isinstance(val, dict):
            report.append(f"  [obj{idx}] {name}: dict keys={list(val.keys())[:20]}")
            for k, v in val.items():
                rec(f"{name}[{k!r}]", v)
        else:
            report.append(f"  [obj{idx}] {name}: {type(val).__name__} "
                          f"{str(val)[:120] if not isinstance(val, (list, tuple)) else f'len={len(val)}'}")

    for attr in ("function_name", "__cubin__", "artifacts", "kernel_info",
                 "jit_module", "engine", "has_gpu_module", "prefix"):
        try:
            val = getattr(obj, attr, None)
        except Exception as e:
            report.append(f"  [obj{idx}] {attr}: ERR {e}")
            continue
        if val is None:
            report.append(f"  [obj{idx}] {attr}: None")
            continue
        rec(attr, val)

    # dump_to_object: try to serialize
    try:
        sig = inspect.signature(obj.dump_to_object)
        report.append(f"  [obj{idx}] dump_to_object sig={sig}")
    except Exception:
        pass


def main():
    out_dir = "/work/cubins"
    os.makedirs(out_dir, exist_ok=True)
    report = []

    pp.run_bs(
        mnkl=(256, 256, 512, 1),
        a_dtype=cutlass.Float4E2M1FN,
        b_dtype=cutlass.Float4E2M1FN,
        sf_dtype=cutlass.Float8E8M0FNU,
        sf_vec_size=32,
        c_dtype=cutlass.Float16,
        acc_dtype=cutlass.Float32,
        a_major="k",
        b_major="k",
        c_major="n",
        tile_shape_mnk=(128, 128, 128),
        epi_tile=(128, 128),
        tolerance=1e-1,
        warmup_iterations=0,
        iterations=1,
        skip_ref_check=True,
    )

    print(f"[sass] captured {len(CAPTURED)} compiled object(s)")
    for i, obj in enumerate(CAPTURED):
        print(f"[sass] ===== obj[{i}] type={type(obj)}")
        probe_obj(obj, i, out_dir, report)

    # dump_to_object: serialize the compiled artifact (obj file with cubin inside)
    for i, obj in enumerate(CAPTURED):
        for pfx in ("gemm", obj.function_name if hasattr(obj, "function_name") else "k"):
            try:
                data = obj.dump_to_object(pfx)
                if isinstance(data, (bytes, bytearray)) and len(data) > 0:
                    fn = os.path.join(out_dir, f"obj{i}_dump.o")
                    open(fn, "wb").write(bytes(data))
                    report.append(f"  [obj{i}] dump_to_object({pfx[:40]!r}) -> {fn} "
                                  f"len={len(data)} head={bytes(data[:8]).hex()}")
                    break
            except Exception as e:
                report.append(f"  [obj{i}] dump_to_object({pfx[:40]!r}) ERR {e}")

    # probe jit_module of each obj (where the GEMM cubin lives)
    for i, obj in enumerate(CAPTURED):
        jm = getattr(obj, "jit_module", None)
        if jm is None:
            continue
        report.append(f"  [obj{i}] jit_module dir={[n for n in dir(jm) if not n.startswith('_') or n in ('__cubin__',)]}")
        for attr in ("__cubin__", "artifacts", "kernel_info", "function_names"):
            try:
                val = getattr(jm, attr, None)
            except Exception as e:
                report.append(f"  [obj{i}].jit_module.{attr}: ERR {e}")
                continue
            if val is None:
                continue
            rec = None
            if isinstance(val, (bytes, bytearray)) and len(val) > 512:
                b = bytes(val)
                off = b.find(b"\x7fELF")
                report.append(f"  [obj{i}].jit_module.{attr}: bytes len={len(b)} ELF@{off}")
                if off >= 0:
                    fn = os.path.join(out_dir, f"obj{i}_jm_{attr.strip('_')}.cubin")
                    open(fn, "wb").write(b[off:])
                    report.append(f"    -> wrote {fn}")
            elif isinstance(val, dict):
                report.append(f"  [obj{i}].jit_module.{attr}: dict keys={list(val.keys())[:10]}")
                for k, v in val.items():
                    if isinstance(v, (bytes, bytearray)) and len(v) > 512:
                        b = bytes(v)
                        off = b.find(b"\x7fELF")
                        report.append(f"    [{k[:60]}]: bytes len={len(b)} ELF@{off}")
                        if off >= 0:
                            fn = os.path.join(out_dir, f"obj{i}_jm_kernel.cubin")
                            open(fn, "wb").write(b[off:])
                            report.append(f"      -> wrote {fn}")
            else:
                report.append(f"  [obj{i}].jit_module.{attr}: {str(val)[:200]}")

    print("[sass] blob report:")
    for line in report:
        print(line)
    print("[sass] out_dir:", os.listdir(out_dir))


if __name__ == "__main__":
    main()
