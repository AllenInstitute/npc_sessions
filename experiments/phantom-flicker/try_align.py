"""Run the prototype alignment on one block and print diagnostics.

Run: PYTHONPATH="../../../npc_sync/src;../../../npc_stim/src;../../src" ../../.venv/Scripts/python.exe try_align.py DRpilot_716718_20240730 0
"""

from __future__ import annotations

import sys
import time

import numpy as np

import phantom as P
from common import block_data


def context(b, k, a, mark=None, after_ms=45):
    v = b.vsyncs
    k0 = int(np.clip(k, 0, len(v) - 1))
    k1 = min(k0 + 1, len(v) - 1)
    ctx = np.flatnonzero((b.edges >= v[k0] - 0.02) & (b.edges < v[k1] + after_ms / 1e3))
    out = []
    for i in ctx:
        c = a.edge_class[i]
        tag = {P.TOGGLE: "", P.GLITCH: "g", P.PHANTOM: "p", P.UNEXPLAINED: "u", P.OUTSIDE: "o"}[c]
        out.append(f"{tag}{'R' if b.rising[i] else 'F'}{(b.edges[i] - v[k0]) * 1e3:.1f}")
    nn = a.n_refreshes[k0] if k0 < len(a.n_refreshes) else -1
    return f"frame {k} n={nn} v[k+1]@{(v[k1] - v[k0]) * 1e3:.1f}: " + " ".join(out)


if __name__ == "__main__":
    folder, bi = sys.argv[1], int(sys.argv[2])
    n_show = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    b = block_data(folder, bi)
    t0 = time.time()
    a = P.align_box1(b.vsyncs, b.frame_intervals, b.edges, b.rising, vsync_rising=b.vsync_rise_per_frame)
    print(f"{folder} b{bi} {b.stim_name}: {len(b.vsyncs)} frames, {len(b.edges)} edges ({time.time() - t0:.1f} s)")
    print(f"first frame rising: {a.first_frame_rising}; landmarks {a.n_landmarks} agree / {a.n_landmarks_disagree} disagree")
    lr, lf = a.latency_rising[np.isfinite(a.latency_rising)], a.latency_falling[np.isfinite(a.latency_falling)]
    print(f"latency rising {np.median(lr) * 1e3:.2f} ms ({lr.min() * 1e3:.2f}-{lr.max() * 1e3:.2f}); falling {np.median(lf) * 1e3:.2f} ({lf.min() * 1e3:.2f}-{lf.max() * 1e3:.2f})")
    print("edge classes:", {P.CLASS_NAMES[c]: a.count(c) for c in P.CLASS_NAMES})
    print(f"missing frames: {len(a.missing_frames)} {a.missing_frames[:10].tolist()}; displaced: {len(a.displaced)} {a.displaced[:10].tolist()}")
    r = a.residual[np.isfinite(a.residual)] * 1e3
    print(f"toggle residual ms: p0.1 {np.percentile(r, 0.1):.2f} p1 {np.percentile(r, 1):.2f} p99 {np.percentile(r, 99):.2f} p99.9 {np.percentile(r, 99.9):.2f} max|r| {np.abs(r).max():.2f}")
    print(f"stim long intervals: {int((a.n_refreshes >= 2).sum())}; same-polarity long mismatches: {len(a.long_mismatch)} {a.long_mismatch[:10].tolist()}")
    print(f"second-pass matches: {len(a.second_pass)}; literal long mismatches: " + str({k: len(v) for k, v in a.literal_long_mismatch.items()}))
    print(f"VALIDATED: {a.validated}")
    for c in (P.PHANTOM, P.UNEXPLAINED, P.GLITCH):
        idx = np.flatnonzero(a.edge_class == c)
        ks = []
        for j in idx:
            k = a.edge_frame[j] if a.edge_frame[j] >= 0 else int(np.searchsorted(b.vsyncs, b.edges[j]) - 2)
            if k not in ks:
                ks.append(k)
        print(f"{P.CLASS_NAMES[c]}: {len(idx)} edges at {len(ks)} places")
        for k in ks[:n_show]:
            print("  " + context(b, k, a))
    for k in a.missing_frames[:n_show]:
        print("  missing " + context(b, k - 1, a))
    for k in a.literal_long_mismatch["other"][:n_show]:
        print("  literal mismatch (other) " + context(b, k, a, after_ms=60))
