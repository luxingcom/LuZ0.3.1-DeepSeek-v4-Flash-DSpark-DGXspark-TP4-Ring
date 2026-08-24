#!/usr/bin/env python3
"""B-N1 diagnostic driver v2: run the probe-patched official pingpong kernel.

usage:
  python bn1_run.py M,N,K [epi_m,epi_n] [tile_m,tile_n,tile_k] [c_dtype]
env:
  ROUTEB_DIAG   "" (exact-representable random + real ref check) | ones | scales
  ROUTEB_DUMP   1 (default: dump C and exit) | 0 (run through, incl. ref check)
"""
import os
import sys

os.environ.setdefault("CUTE_DSL_ARCH", "sm_121a")
os.environ.setdefault("ROUTEB_DIAG", "ones")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "routeb_official"))

import cutlass  # noqa: E402

# 4.5.2 compat shim: cutlass.testing -> cutlass.cute.testing (bench 同款)
import cutlass.cute.testing as _cute_testing  # noqa: E402

if not hasattr(cutlass, "testing"):
    cutlass.testing = _cute_testing
    sys.modules["cutlass.testing"] = _cute_testing

import dense_blockscaled_gemm_persistent_pingpong as _pp_probe  # noqa: E402

# ROUTEB_MODULE=official -> use the patched (guarded) official module
if os.environ.get("ROUTEB_MODULE") == "official":
    import importlib  # noqa: E402

    pp = importlib.import_module("dense_blockscaled_gemm_persistent_pingpong")
else:
    pp = _pp_probe


def _parse(s, n):
    vals = tuple(int(x) for x in s.split(","))
    assert len(vals) == n, f"expected {n} values, got {s}"
    return vals


def main():
    mnk = _parse(sys.argv[1], 3)
    epi = _parse(sys.argv[2], 2) if len(sys.argv) > 2 else (128, 128)
    tile = _parse(sys.argv[3], 3) if len(sys.argv) > 3 else (128, 128, 128)
    c_dt_name = sys.argv[4] if len(sys.argv) > 4 else "Float32"
    c_dtype = getattr(cutlass, c_dt_name)
    sf_vec = int(sys.argv[5]) if len(sys.argv) > 5 else 32
    sf_dt_name = sys.argv[6] if len(sys.argv) > 6 else (
        "Float8E4M3FN" if sf_vec == 16 else "Float8E8M0FNU")
    sf_dtype = getattr(cutlass, sf_dt_name)
    dump = os.environ.get("ROUTEB_DUMP", "1") == "1"
    tag = (f"{mnk[0]}x{mnk[1]}x{mnk[2]}_epi{epi[0]}x{epi[1]}"
           f"_tile{tile[0]}x{tile[1]}x{tile[2]}_{c_dt_name}")
    if dump:
        os.environ["ROUTEB_DUMP"] = os.path.join(HERE, f"bn1_C_{tag}.pt")
    else:
        os.environ.pop("ROUTEB_DUMP", None)
    print(f"[bn1_run] shape={mnk} epi={epi} tile={tile} c_dtype={c_dt_name} "
          f"sf_vec={sf_vec} sf_dtype={sf_dt_name} "
          f"diag={os.environ.get('ROUTEB_DIAG','')!r} dump={dump}")

    pp.run_bs(
        mnkl=(mnk[0], mnk[1], mnk[2], 1),
        a_dtype=cutlass.Float4E2M1FN,
        b_dtype=cutlass.Float4E2M1FN,
        sf_dtype=sf_dtype,  # MXF4: E8M0+vec32; NVFP4: E4M3+vec16
        sf_vec_size=sf_vec,
        c_dtype=c_dtype,
        acc_dtype=cutlass.Float32,
        a_major="k",
        b_major="k",
        c_major="n",
        tile_shape_mnk=tile,
        epi_tile=epi,
        tolerance=1e-1,
        warmup_iterations=0,
        iterations=1,
        skip_ref_check=False,
    )
    print("[bn1_run] COMPLETED (incl. ref check if enabled)")


if __name__ == "__main__":
    main()
