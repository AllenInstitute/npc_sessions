"""For multi-run files: are barcodes with +-k*0.5 s interval anomalies *outside* invalid_intervals?
Prints, for each anomalous barcode onset, the nearest invalid interval and the distance to it.

Run: ../../../npc_sync/.venv/Scripts/python.exe displaced_barcode_check.py > displaced_barcode_check.log 2>&1
"""
import logging
import pathlib

import numpy as np

import npc_sync

logging.disable(logging.WARNING)
SHARE = pathlib.Path("//allen/programs/mindscope/workgroups/dynamicrouting/PilotEphys/Task 2 Pilot")
for folder in ("DRpilot_714748_20240628", "DRpilot_703880_20240418", "DRpilot_708016_20240426", "DRpilot_859582_20260612"):
    s = npc_sync.SyncDataset(next((SHARE / folder).glob("????????T??????.h5")))
    iv = np.array(s.invalid_intervals)
    on = s.get_rising_edges("barcode_ephys", units="seconds")
    inside = np.zeros(on.size, bool)
    for a, z in iv:
        inside |= (on >= a) & (on < z)
    on = on[~inside]
    starts = on[np.r_[True, np.diff(on) > 20]]
    itv = np.diff(starts)
    dev = itv - np.median(itv)
    print(f"\n{folder}: {len(iv)} invalid intervals")
    # onset i is displaced by d if interval (i-1 -> i) has dev -d and (i -> i+1) has +d
    for i in range(1, len(starts) - 1):
        d1, d2 = dev[i - 1], dev[i]
        if abs(d1) > 0.01 and abs(d1 + d2) < 0.01:
            t = starts[i]
            dist = np.min(np.r_[np.abs(iv[:, 0] - t), np.abs(iv[:, 1] - t)])
            a, z = iv[np.argmin(np.minimum(np.abs(iv[:, 0] - t), np.abs(iv[:, 1] - t)))]
            print(
                f"  barcode onset t={t:.3f} s displaced by {d1 * 1e3:+.1f} ms (k={d1 / 0.5:+.2f}); "
                f"nearest invalid interval [{a:.1f}, {z:.1f}) at {dist:.2f} s; inside an interval: False (masked)"
            )
