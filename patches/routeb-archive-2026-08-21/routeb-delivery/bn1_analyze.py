#!/usr/bin/env python3
"""B-N1 zero-form analysis: classify each C element as sentinel / zero / value.

usage: python bn1_analyze.py <dump.pt> <K> [epi_m] [tile_m]
"""
import sys

import torch


def main():
    path, K = sys.argv[1], int(sys.argv[2])
    epi_m = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    tile_m = int(sys.argv[4]) if len(sys.argv) > 4 else 128
    SENT = -777.0

    c = torch.load(path)["c"].float()
    if c.dim() == 3:
        c = c.squeeze(-1) if c.shape[-1] == 1 else c.squeeze(0)
    m, n = c.shape
    print(f"[analyze] C shape=({m},{n}) K={K} epi_m={epi_m} tile_m={tile_m}")

    sent = c == SENT
    zero = c == 0.0
    good = (c == float(K))
    print(f"[analyze] sentinel={sent.sum().item()} ({100*sent.float().mean():.2f}%)  "
          f"zero={zero.sum().item()} ({100*zero.float().mean():.2f}%)  "
          f"==K:{good.sum().item()} ({100*good.float().mean():.2f}%)  "
          f"other={(~(sent|zero|good)).sum().item()}")

    # per-row classification
    row_sent = sent.all(dim=1)
    row_zero = zero.all(dim=1)
    row_good = good.all(dim=1)
    print(f"[analyze] rows: all-sentinel={row_sent.sum().item()}  all-zero={row_zero.sum().item()}  "
          f"all-correct={row_good.sum().item()}  mixed={(~(row_sent|row_zero|row_good)).sum().item()}")

    # per-column classification
    col_sent = sent.all(dim=0)
    col_zero = zero.all(dim=0)
    col_good = good.all(dim=0)
    print(f"[analyze] cols: all-sentinel={col_sent.sum().item()}  all-zero={col_zero.sum().item()}  "
          f"all-correct={col_good.sum().item()}  mixed={(~(col_sent|col_zero|col_good)).sum().item()}")

    # per tile_m band: pattern summary
    for b in range(0, m, tile_m):
        band = c[b:b + tile_m]
        bs = (band == SENT).float().mean().item()
        bz = (band == 0).float().mean().item()
        bg = good[b:b + tile_m].float().mean().item()
        # boundary: last correct row index within band
        rows_ok = good[b:b + tile_m].any(dim=1)
        last_ok = (torch.nonzero(rows_ok).max().item()
                   if rows_ok.any() else -1)
        first_ok = (torch.nonzero(rows_ok).min().item()
                    if rows_ok.any() else -1)
        print(f"[analyze] tile-band rows[{b:4d},{b+tile_m:4d}): sent={bs:.2%} zero={bz:.2%} "
              f"good={bg:.2%} first_ok_row={first_ok} last_ok_row={last_ok}")

    # per epi_m sub-band inside first tile band
    b0 = c[0:tile_m]
    good0 = good[0:tile_m]
    for eb in range(0, tile_m, epi_m):
        rows_ok = good0[eb:eb + epi_m].any(dim=1)
        nz = torch.nonzero(rows_ok)
        print(f"[analyze]   epi rows[{eb:3d},{eb+epi_m:3d}): correct rows "
              f"{(nz.min().item() if len(nz) else -1)}..{(nz.max().item() if len(nz) else -1)} "
              f"(abs {eb + (nz.min().item() if len(nz) else -1)}..{eb + (nz.max().item() if len(nz) else -1)})")

    # column profile of first bad region
    if row_sent.any() or row_zero.any():
        bad_rows = torch.nonzero(row_sent | row_zero).flatten()
        print(f"[analyze] bad rows: {bad_rows[:8].tolist()} ... {bad_rows[-8:].tolist()} (count {len(bad_rows)})")

    # if mixed columns exist, show N-position pattern for row 0
    mixed_cols = (~(col_sent | col_zero | col_good))
    if mixed_cols.any():
        idx = torch.nonzero(mixed_cols).flatten()
        print(f"[analyze] mixed cols count={len(idx)} first={idx[:16].tolist()}")

    # show actual values of the bad region
    bad_mask = ~(sent | zero | good)
    if bad_mask.any():
        bad_vals = c[bad_mask]
        uniq, cnt = torch.unique(bad_vals, return_counts=True)
        order = cnt.argsort(descending=True)
        print(f"[analyze] bad-region unique values: {len(uniq)} "
              f"min={bad_vals.min().item():.4f} max={bad_vals.max().item():.4f}")
        for i in order[:12].tolist():
            print(f"[analyze]   value={uniq[i].item():.6g}  count={cnt[i].item()}")
        print(f"[analyze] sample block rows 62..70, cols 0..7:")
        for r in range(62, min(71, m)):
            print(f"[analyze]   row {r:3d}: {['%g' % v for v in c[r, 0:8].tolist()]}")
        print(f"[analyze] sample block rows 126..134, cols 0..7:")
        for r in range(126, min(135, m)):
            print(f"[analyze]   row {r:3d}: {['%g' % v for v in c[r, 0:8].tolist()]}")


if __name__ == "__main__":
    main()
