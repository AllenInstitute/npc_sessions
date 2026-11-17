"""Which *real* task events fell inside replayed (invalid) sync intervals, and does the
barcode decode change if events in invalid intervals are masked?

The replayed copies show the genuine last buffer's events, not what happened, so the
sync lines can't tell us what was lost. Instead, map stim-file frame indices to sync time
with the prototype `reconcile_replay` (npc_sync/experiments/vsync-frame-count/repair.py),
then count stim-file events whose frames fall inside invalid intervals.

Run with the npc_sync dev venv:
    ../../../npc_sync/.venv/Scripts/python.exe lost_events.py > lost_events.log 2>&1

Checks (defined before running):
  D1 reconcile_replay succeeds (anchor residual <= 3 ms) for every block overlapping an interval.
  D2 count of stim-file events (licks, rewards, manual rewards, trial stim onsets, opto onsets)
     whose frame time is inside an invalid interval.
  D3 barcode decode (sync side) with vs without masking events inside invalid intervals:
     same set of (start time, value) pairs?
"""

from __future__ import annotations

import logging
import pathlib
import sys

import h5py
import numpy as np

import npc_stim
import npc_sync

logging.disable(logging.WARNING)

ROOT = pathlib.Path(__file__).resolve().parents[3] / "npc_sync" / "experiments" / "vsync-frame-count"
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT))
from repair import reconcile_replay  # noqa: E402

try:
    from npc_ephys import extract_barcodes_from_times  # noqa: E402
except ImportError:
    from npc_ephys_barcodes import extract_barcodes_from_times  # local copy in ROOT  # noqa: E402


def in_any(t, intervals):
    t = np.asarray(t, dtype=float)
    m = np.zeros(t.shape, bool)
    for a, z in intervals:
        m |= (t >= a) & (t < z)
    return m


def frames(h5: h5py.File, key: str) -> np.ndarray:
    if key not in h5:
        return np.array([], dtype=float)
    v = np.asarray(h5[key][()], dtype=float).ravel()
    return v[np.isfinite(v)]


def main() -> None:
    for sess_dir in sorted(p for p in DATA.iterdir() if p.is_dir()):
        sync_path = next(sess_dir.glob("*T*.h5"))
        s = npc_sync.SyncDataset(sync_path)
        iv = list(s.invalid_intervals)
        print(f"\n===== {sess_dir.name}: invalid_intervals={[(round(a, 3), round(z, 3)) for a, z in iv]}")
        if not iv:
            continue
        blocks = s.vsync_times_in_blocks
        for stim_path in sorted(sess_dir.glob("*.hdf5")):
            h5 = npc_stim.get_h5_stim_data(stim_path)
            n_stim = npc_stim.get_total_stim_frames(h5)
            fi = h5["frameIntervals"][:]
            # match block by frame count within +-10 and start-time proximity (diagnostic only)
            st = (npc_stim.get_stim_start_time(h5) - s.start_time).total_seconds()
            cands = [i for i, b in enumerate(blocks) if abs(len(b) - n_stim) <= 10]
            if not cands:
                continue
            bi = min(cands, key=lambda i: abs(blocks[i][0] - st))
            v = blocks[bi]
            if not any((v[0] <= z) and (v[-1] >= a) for a, z in iv):
                continue
            try:
                rec = reconcile_replay(v, fi, iv)
            except ValueError as exc:
                print(f"D1 {stim_path.stem} block {bi}: reconcile FAILED: {exc}")
                continue
            print(f"D1 {stim_path.stem} block {bi}: n_sync={len(v)} n_stim={n_stim}; {'; '.join(rec.notes)}")
            t = rec.times
            out = []
            for key in (
                "lickFrames",
                "rewardFrames",
                "manualRewardFrames",
                "trialStimStartFrame",
                "stimStartFrame",
                "trialOptoOnsetFrame",
            ):
                f = frames(h5, key)
                if key == "trialOptoOnsetFrame" and f.size:
                    f = np.asarray(
                        [x for x in npc_stim.get_stim_trigger_frames(h5, stim_type="opto") if x is not None],
                        dtype=float,
                    )
                f = f[(f >= 0) & (f < len(t))].astype(int)
                n_in = int(in_any(t[f], iv).sum()) if f.size else 0
                out.append(f"{key}: {n_in}/{f.size}")
            print("D2 stim events with frame time inside invalid intervals:", "; ".join(out))

        # D3 barcode decode with/without masking
        on = s.get_rising_edges("barcode_ephys", units="seconds")
        off = s.get_falling_edges("barcode_ephys", units="seconds")
        t0, b0 = extract_barcodes_from_times(on, off, total_time_on_line=s.total_seconds)
        on_m, off_m = on[~in_any(on, iv)], off[~in_any(off, iv)]
        t1, b1 = extract_barcodes_from_times(on_m, off_m, total_time_on_line=s.total_seconds)
        a0 = set(zip(np.round(t0, 6).tolist(), b0.tolist()))
        a1 = set(zip(np.round(t1, 6).tolist(), b1.tolist()))
        n_edges_in = int(in_any(on, iv).sum() + in_any(off, iv).sum())
        print(
            f"D3 barcodes: unmasked n={len(t0)}, masked n={len(t1)}, edges inside intervals={n_edges_in}, "
            f"only-unmasked={sorted(a0 - a1)[:5]}, only-masked={sorted(a1 - a0)[:5]}"
        )
        if n_edges_in:
            near = [(round(float(x), 3), int(y)) for x, y in zip(t0, b0) if any(a - 3 < x < z + 3 for a, z in iv)]
            prev = t0[t0 < iv[0][0]]
            print(f"    barcode starts within 3 s of an interval: {near}; last start before first interval: {prev[-1]:.3f} s")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
