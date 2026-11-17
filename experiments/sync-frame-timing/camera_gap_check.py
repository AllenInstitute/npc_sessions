"""Can camera exposures inside invalid (replayed) sync intervals be reconstructed by
regular-period interpolation between the measured exposures on either side?

For each camera exposing line: drop rising edges inside the invalid interval, take the last
measured edge before (t0) and first after (t1), estimate the local period from the 2 s of
measured edges on each side, and check how close (t1 - t0) / period is to an integer.
Also report how many exposures the replayed copy contained versus the integer estimate.

Run: ../../../npc_sync/.venv/Scripts/python.exe camera_gap_check.py > camera_gap_check.log 2>&1

Checks:
  F1 fractional part of (t1 - t0)/period is within 0.05 of an integer (unambiguous count).
  F2 residual of the integer-period model at t1 (ms).
  F3 copy count vs reconstructed count (does the replay change the exposure count?).
"""

from __future__ import annotations

import logging
import pathlib

import numpy as np

import npc_sync

logging.disable(logging.WARNING)
DATA = pathlib.Path(__file__).resolve().parents[3] / "npc_sync" / "experiments" / "vsync-frame-count" / "data"

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
            before, after = r[r < a], r[r >= z]
            if before.size < 10 or after.size < 10:
                continue
            t0, t1 = before[-1], after[0]
            p_before = np.median(np.diff(before[before > t0 - 2]))
            p_after = np.median(np.diff(after[after < t1 + 2]))
            p = 0.5 * (p_before + p_after)
            ratio = (t1 - t0) / p
            n_int = int(round(ratio))
            resid_ms = (t1 - (t0 + n_int * p)) * 1e3
            # largest deviation of measured intervals from the period (jitter), 2 s each side
            jit = np.max(np.abs(np.r_[np.diff(before[before > t0 - 2]), np.diff(after[after < t1 + 2])] - p)) * 1e3
            print(
                f"  {label:22s} [{a:.1f},{z:.1f}) period={p * 1e3:.4f} ms (before {p_before * 1e3:.4f}, after {p_after * 1e3:.4f}); "
                f"(t1-t0)/p={ratio:.4f} -> frac {abs(ratio - n_int):.4f}; resid={resid_ms:+.3f} ms; jitter max={jit:.3f} ms; "
                f"copies={int(inside.sum())} vs reconstructed={n_int - 1}"
            )
