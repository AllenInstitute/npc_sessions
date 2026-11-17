"""Follow-up to line_audit.py: include *unlabelled* bits (e.g. bit 1 = sound on sync since
2023-08-31, unlabelled in these files) and measure sync barcode durations.

Run: ../../../npc_sync/.venv/Scripts/python.exe all_bits_audit.py > all_bits_audit.log 2>&1

Checks:
  E1 per-bit edge counts inside invalid intervals, all 32 bits (label or '<unlabelled>').
  E2 distribution of barcode durations on sync (first rising edge to last falling edge of each
     barcode): can a whole barcode fit inside one 0.5 s acquisition buffer?
"""

from __future__ import annotations

import logging
import pathlib

import numpy as np

import npc_sync

logging.disable(logging.WARNING)
DATA = pathlib.Path(__file__).resolve().parents[3] / "npc_sync" / "experiments" / "vsync-frame-count" / "data"


def in_any(t, intervals):
    m = np.zeros(len(t), bool)
    for a, z in intervals:
        m |= (t >= a) & (t < z)
    return m


durations_all = []
for sess_dir in sorted(p for p in DATA.iterdir() if p.is_dir()):
    s = npc_sync.SyncDataset(next(sess_dir.glob("*T*.h5")))
    iv = list(s.invalid_intervals)
    rows = []
    for bit in range(32):
        r = s.get_rising_edges(bit, units="seconds")
        f = s.get_falling_edges(bit, units="seconds")
        if not (r.size or f.size):
            continue
        nr, nf = int(in_any(r, iv).sum()), int(in_any(f, iv).sum())
        label = s.line_labels[bit] if bit < len(s.line_labels) and s.line_labels[bit] else "<unlabelled>"
        rows.append(f"bit{bit}:{label}: total={r.size} in_invalid={nr}/{nf}")
    print(f"\n===== {sess_dir.name} invalid={[(round(a, 2), round(z, 2)) for a, z in iv]}")
    print("E1 " + "\n   ".join(rows))
    on = s.get_rising_edges("barcode_ephys", units="seconds")
    off = s.get_falling_edges("barcode_ephys", units="seconds")
    starts = np.flatnonzero(np.r_[True, np.diff(on) > 20])
    d = []
    for i, k in enumerate(starts):
        t0 = on[k]
        offs = off[(off > t0) & (off < t0 + 2)]
        if offs.size:
            d.append(offs[-1] - t0)
    d = np.array(d)
    durations_all.extend(d.tolist())
    print(f"E2 barcode durations (s): n={d.size} min={d.min():.3f} median={np.median(d):.3f} max={d.max():.3f}")
da = np.array(durations_all)
print(f"\nE2 all sessions: n={da.size} min={da.min():.3f} median={np.median(da):.3f} max={da.max():.3f}; fraction <0.5 s: {np.mean(da < 0.5):.2f}")
