"""Print diode edges around stim-recorded long frame intervals in one box-1 block.

For frame k with frameIntervals[k] >= 1.5 refreshes (frame k held for n refreshes), list every
diode edge between vsync[k] and vsync[k+1] + 40 ms, relative to vsync[k] (falling edge), with
polarity. Read-only on the cache.

Run: PYTHONPATH="../../../npc_sync/src;../../../npc_stim/src;../../src" ../../.venv/Scripts/python.exe explore_block.py DRpilot_762526_20250320 1
"""

from __future__ import annotations

import sys

import numpy as np

from common import block_data

folder, bi = sys.argv[1], int(sys.argv[2])
n_show = int(sys.argv[3]) if len(sys.argv) > 3 else 30
b = block_data(folder, bi)
print(f"{folder} block {bi} ({b.stim_name}): {len(b.vsyncs)} vsyncs, {b.n_stim_frames} stim frames, box {b.box}")
print(f"refresh period {b.refresh * 1e3:.3f} ms; diode edges in block window: {len(b.edges)}")
fi = b.frame_intervals
nref = np.round(fi / b.refresh).astype(int)
long_idx = np.flatnonzero(nref >= 2)
print(f"long frame intervals: {len(long_idx)}; refresh counts: {dict(zip(*np.unique(nref[long_idx], return_counts=True)))}")
lat = b.edges - b.vsyncs[np.clip(np.searchsorted(b.vsyncs, b.edges) - 1, 0, None)]
print(f"edge - preceding vsync: median {np.median(lat) * 1e3:.2f} ms")
for k in long_idx[:n_show]:
    lo, hi = b.vsyncs[k], b.vsyncs[k + 1] + 0.040
    sel = (b.edges >= lo) & (b.edges < hi)
    rel = (b.edges[sel] - lo) * 1e3
    pol = np.where(b.rising[sel], "R", "F")
    vs = (b.vsyncs[k + 1] - lo) * 1e3
    print(f"frame {k:>7} n={nref[k]:>2} fi={fi[k] * 1e3:6.1f} ms vsync[k+1]@{vs:6.1f}: " + " ".join(f"{p}{t:.1f}" for p, t in zip(pol, rel)))
