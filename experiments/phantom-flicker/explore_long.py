"""For one box-1 block: every stim-recorded long frame interval (>= 2 refreshes) whose span
contains more diode edges than expected, printed as edge times relative to vsync[k] (falling
edge of the held frame k) with polarity, plus vsync[k+1].

Edges counted for interval k: those in [v[k] + 0.5 R, v[k+1] + 0.5 R) where R is one refresh; a
clean held frame has exactly one edge there (frame k's own toggle, ~20 ms after v[k]).

Run: PYTHONPATH="../../../npc_sync/src;../../../npc_stim/src;../../src" ../../.venv/Scripts/python.exe explore_long.py DRpilot_716718_20240730 0
"""

from __future__ import annotations

import collections
import sys

import numpy as np

from common import block_data

folder, bi = sys.argv[1], int(sys.argv[2])
n_show = int(sys.argv[3]) if len(sys.argv) > 3 else 40
b = block_data(folder, bi)
v = b.vsyncs
R = b.refresh
nref = np.round(b.frame_intervals / R).astype(int)
print(f"{folder} b{bi} {b.stim_name}: box {b.box}, {len(v)} vsyncs / {b.n_stim_frames} frames, {len(b.edges)} edges, R={R * 1e3:.3f} ms")
tab = collections.Counter()
shown = 0
for k in np.flatnonzero(nref >= 2):
    lo, hi = v[k] + 0.5 * R, v[k + 1] + 0.5 * R
    sel = (b.edges >= lo) & (b.edges < hi)
    n = int(sel.sum())
    tab[(int(nref[k]), n)] += 1
    if n > 1 and shown < n_show:
        shown += 1
        ctx = (b.edges >= v[k] - 0.02) & (b.edges < v[k + 1] + 0.045)
        txt = " ".join(
            ("[" if s else "") + ("R" if r else "F") + f"{(t - v[k]) * 1e3:.1f}" + ("]" if s else "")
            for t, r, s in zip(b.edges[ctx], b.rising[ctx], sel[ctx])
        )
        print(f"frame {k:>7} n={nref[k]:>2} v[k+1]@{(v[k + 1] - v[k]) * 1e3:6.1f}: {txt}")
print("(refreshes, edges in interval): count")
for key, c in sorted(tab.items()):
    print(" ", key, c)
