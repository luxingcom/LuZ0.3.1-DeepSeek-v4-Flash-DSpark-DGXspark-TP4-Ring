#!/usr/bin/env python3
"""compare2_lp.py — 四方对照: b12x-W4A4 vs EMU-W4A4 vs W4A16(noclamp) vs W4A16(clamp)。"""
import json
import sys
sys.path.insert(0, "/tmp/_routea_work")
from compare_lp import compare, load

b2 = load("/tmp/_routea_work/lp_w4a16_noclamp.json")
e = load("/tmp/_routea_work/lp_emu.json")
t = load("/tmp/_routea_work/lp_w4a4.json")
b1 = load("/tmp/_routea_work/lp_0731.json")

compare(e, b2, "E: EMU-W4A4", "B2: W4A16(noclamp)")   # 固有 W4A4 量化差
print()
compare(t, e, "T: b12x-W4A4", "E: EMU-W4A4")           # b12x 相对正确 W4A4 的偏差
print()
compare(t, b2, "T: b12x-W4A4", "B2: W4A16(noclamp)")   # 端到端差
