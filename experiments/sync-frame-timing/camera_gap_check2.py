"""Refinement of camera_gap_check.py: estimate the camera period by least-squares slope of
exposure time vs index over 10 s of measured edges on each side (the median of 10-us-quantised
intervals is too coarse over ~500 periods).

Run: ../../../npc_sync/.venv/Scripts/python.exe camera_gap_check2.py > camera_gap_check2.log 2>&1
"""
from __future__ import annotations

import logging
import pathlib

import numpy as np

import npc_sync

logging.disable(logging.WARNING)
DATA = pathlib.Path(__file__).resolve().parents[3] / "npc_sync" / "experiments" / "vsync-frame-count" / "data"
WIN = 10.0

worst = 0.0
for sess_dir in sorted(p for p in DATA.iterdir() if p.is_dir()):
    s = npc_sync.SyncDataset(next(sess_dir.glob("*T*.h5")))
    iv = list(s.invalid_intervals)
    if not iv:
        continue
    print(f"\n===== {sess_dir.name} invalid={[(round(a, 2), round(z, 2)) for a, z in iv]}")
    for label in s.line_labels:
        if not label.endswith("_cam_exposing"):
            continue
        r = s.get_rising_edges(label, units="seconds")
        for a, z in iv:
            inside = (r >= a) & (r < z)
            b = r[(r < a) & (r > a - WIN)]
            f = r[(r >= z) & (r < z + WIN)]
            pb = np.polyfit(np.arange(b.size), b, 1)[0]
            pf = np.polyfit(np.arange(f.size), f, 1)[0]
            p = 0.5 * (pb + pf)
            t0, t1 = b[-1], f[0]
            ratio = (t1 - t0) / p
            n = int(round(ratio))
            resid_ms = (t1 - (t0 + n * p)) * 1e3
            # within-segment fit residuals (how regular is the camera clock?)
            rb = b - np.polyval(np.polyfit(np.arange(b.size), b, 1), np.arange(b.size))
            rf = f - np.polyval(np.polyfit(np.arange(f.size), f, 1), np.arange(f.size))
            worst = max(worst, abs(resid_ms))
            print(
                f"  {label:18s} period={p * 1e3:.5f} ms (before {pb * 1e3:.5f}, after {pf * 1e3:.5f}); "
                f"(t1-t0)/p={ratio:.4f}; gap resid={resid_ms:+.3f} ms; in-segment fit resid max={max(np.abs(rb).max(), np.abs(rf).max()) * 1e3:.3f} ms; "
                f"copies={int(inside.sum())} reconstructed={n - 1}"
            )
print(f"\nworst |gap resid| = {worst:.3f} ms")
