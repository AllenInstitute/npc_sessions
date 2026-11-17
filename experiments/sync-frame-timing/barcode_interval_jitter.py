"""How regular are sync barcode onset intervals? (Basis for a sync-only timebase-integrity check:
a displaced/dropped 0.5 s buffer shifts one interval by ~k * 0.5 s.)

Run: ../../../npc_sync/.venv/Scripts/python.exe barcode_interval_jitter.py > barcode_interval_jitter.log 2>&1

Check H1: |interval - median| for all barcode intervals not adjacent to an invalid interval;
report max (ms) and any interval deviating by > 10 ms.
"""

import logging
import pathlib

import numpy as np

import npc_sync

logging.disable(logging.WARNING)
DATA = pathlib.Path(__file__).resolve().parents[3] / "npc_sync" / "experiments" / "vsync-frame-count" / "data"

for d in sorted(p for p in DATA.iterdir() if p.is_dir()):
    s = npc_sync.SyncDataset(next(d.glob("*T*.h5")))
    iv = list(s.invalid_intervals)
    on = s.get_rising_edges("barcode_ephys", units="seconds")
    for a, z in iv:
        on = on[~((on >= a) & (on < z))]
    starts = on[np.r_[True, np.diff(on) > 20]]
    itv = np.diff(starts)
    near = np.array([any(a - 35 < t < z + 35 for a, z in iv) for t in starts[:-1]], dtype=bool)
    dev = (itv - np.median(itv)) * 1e3
    big = [(round(float(t), 2), round(float(x), 1)) for t, x in zip(starts[:-1], dev) if abs(x) > 10]
    clean = dev[~near] if near.size else dev
    print(
        f"{d.name}: n_intervals={itv.size} median={np.median(itv):.4f} s; "
        f"max |dev| (not within 35 s of an invalid interval) = {np.abs(clean).max():.2f} ms; "
        f"intervals with |dev| > 10 ms: {big}"
    )
