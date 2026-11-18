"""Count diode edges per toggle-frame window in one block and print windows with != 1 edge.

Window of toggle frame k (box 1: every frame; box N: f % N == 0) = [v[k] + L - h, v[k'] + L - h),
where k' is the next toggle frame, L = the block's median (edge - vsync of its toggle frame) and
h = half a refresh. Prints, for every window with 0 or >= 2 edges, the stim frame interval(s)
spanned by the window (refreshes) and the edges relative to v[k] with polarity.

Run: PYTHONPATH="../../../npc_sync/src;../../../npc_stim/src;../../src" ../../.venv/Scripts/python.exe explore_windows.py DRpilot_762526_20250320 1
"""

from __future__ import annotations

import sys

import numpy as np

from common import block_data

folder, bi = sys.argv[1], int(sys.argv[2])
n_show = int(sys.argv[3]) if len(sys.argv) > 3 else 40
b = block_data(folder, bi)
box = b.box or 1
v = b.vsyncs
assert len(v) == b.n_stim_frames, (len(v), b.n_stim_frames)
R = b.refresh
nref = np.round(b.frame_intervals / R).astype(int)
tog = np.arange(0, len(v), box)
vt = v[tog]
# latency: nearest toggle vsync before each edge
j = np.clip(np.searchsorted(vt, b.edges - 0.010) - 1, 0, len(vt) - 1)
L = float(np.median(b.edges - vt[j]))
h = 0.5 * R
bounds = np.r_[vt + L - h, vt[-1] + L - h + box * R]
c = np.histogram(b.edges, bins=bounds)[0]
print(f"{folder} b{bi} {b.stim_name}: box {box}, {len(tog)} toggle frames, {len(b.edges)} edges; L={L * 1e3:.2f} ms")
print("edges per window:", dict(zip(*np.unique(c, return_counts=True))))
bad = np.flatnonzero(c != 1)
print(f"windows with != 1 edge: {len(bad)}")
for w in bad[:n_show]:
    k0 = tog[w]
    k1 = tog[w + 1] if w + 1 < len(tog) else len(v)
    spanned = nref[k0:min(k1, len(nref))]
    sel = (b.edges >= bounds[w]) & (b.edges < bounds[w + 1])
    lo = bounds[w] - 0.03
    hi = bounds[w + 1] + 0.03
    ctx = (b.edges >= lo) & (b.edges < hi)
    txt = " ".join(
        ("[" if s else "") + ("R" if r else "F") + f"{(t - v[k0]) * 1e3:.1f}" + ("]" if s else "")
        for t, r, s in zip(b.edges[ctx], b.rising[ctx], sel[ctx])
    )
    print(f"win {w:>7} frame {k0:>7} refreshes {spanned.tolist()} n_edges {c[w]}: {txt}")
