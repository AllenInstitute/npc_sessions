"""Independent check of sync files with many replay runs (read directly from the share).

Note: the prototype `repair.reconcile_replay` assigns ALL remaining missing frames to the first
gap, so it is only valid for blocks with a single invalid interval. Here each gap's frame count
is solved from elapsed time instead:
    n_g = argmin_n | (t_after - t_before) - (tf[k + 1 + n] - tf[k]) |
where tf = [0, cumsum(frameIntervals)] (stim clock) and k = stim frame index of the last measured
vsync before the gap. Balance = n_stim - (index of last measured vsync + 1) must be 0.

Per file: invalid_intervals (npc_sync 0.2.8), sync barcode-interval anomalies (|dev| > 10 ms from
the per-file median after masking edges inside invalid intervals), and the task block (DynamicRouting1):
raw count difference, number of gaps, per-gap residual distribution, balance.

Run: ../../../npc_sync/.venv/Scripts/python.exe multi_gap_check.py > multi_gap_check.log 2>&1
"""

from __future__ import annotations

import logging
import pathlib
import sys
import time

import numpy as np

import npc_stim
import npc_sync

logging.disable(logging.WARNING)
SHARE = pathlib.Path("//allen/programs/mindscope/workgroups/dynamicrouting/PilotEphys/Task 2 Pilot")
LOCAL = pathlib.Path(__file__).resolve().parents[3] / "npc_sync" / "experiments" / "vsync-frame-count" / "data"
FOLDERS = (
    # many runs in the 2026-11-03 survey (npc_sync/sync_replay_issues.csv)
    "DRpilot_714748_20240628",  # 137 runs
    "DRpilot_703880_20240418",  # 47
    "DRpilot_708016_20240426",  # 29
    "DRpilot_859582_20260612",  # 27
    # a few runs
    "DRpilot_746439_20250103",  # 5
    "DRpilot_813586_20251117",  # 6
    # single run, known (control)
    "DRpilot_836629_20260430",
)


def in_any(t, intervals):
    t = np.asarray(t, dtype=float)
    m = np.zeros(t.shape, bool)
    for a, z in intervals:
        m |= (t >= a) & (t < z)
    return m


def remove_invalid_and_splices(v: np.ndarray, iv) -> np.ndarray:
    v = v[~in_any(v, iv)]
    half = 0.5 * np.median(np.diff(v))
    for a, z in iv:  # same splice handling as repair.reconcile_replay
        j = int(np.searchsorted(v, z))
        while j + 1 < len(v) and v[j] - z < half and v[j + 1] - v[j] < half:
            v = np.delete(v, j)
        j = int(np.searchsorted(v, a)) - 1
        while j >= 1 and a - v[j] < half and v[j] - v[j - 1] < half:
            v = np.delete(v, j)
            j -= 1
    return v


def solve_gaps(v_raw: np.ndarray, frame_intervals: np.ndarray, iv):
    tf = np.concatenate([[0.0], np.cumsum(frame_intervals)])
    n_stim = len(tf)
    v = remove_invalid_and_splices(v_raw, iv)
    gaps = [(a, z) for a, z in sorted(iv) if v[0] < z and a < v[-1]]
    k = 0  # stim index of v[0] (assume block starts at frame 0)
    idx_v = 0
    resid, second = [], []
    for a, z in gaps:
        j_before = int(np.searchsorted(v, a)) - 1  # last measured before gap
        j_after = j_before + 1
        k_before = k + (j_before - idx_v)
        dt = v[j_after] - v[j_before]
        n_max = int(dt / 0.016) + 5
        cand = np.arange(0, n_max)
        cand = cand[k_before + 1 + cand < n_stim]
        err = np.abs(dt - (tf[k_before + 1 + cand] - tf[k_before]))
        o = np.argsort(err)
        n_g = int(cand[o[0]])
        resid.append(err[o[0]])
        second.append(err[o[1]] if len(o) > 1 else np.nan)
        k = k_before + 1 + n_g
        idx_v = j_after
    last_k = k + (len(v) - 1 - idx_v)
    balance = n_stim - (last_k + 1)
    return len(gaps), np.array(resid), np.array(second), balance, len(v_raw) - n_stim


for folder in FOLDERS:
    t0 = time.time()
    root = LOCAL / folder if (LOCAL / folder).exists() else SHARE / folder
    try:
        sync_path = next(root.glob("????????T??????.h5"))
        s = npc_sync.SyncDataset(sync_path)
        iv = list(s.invalid_intervals)
        # barcode intervals
        on = s.get_rising_edges("barcode_ephys", units="seconds")
        on = on[~in_any(on, iv)]
        starts = on[np.r_[True, np.diff(on) > 20]]
        dev = (np.diff(starts) - np.median(np.diff(starts))) * 1e3
        near = np.array([any(a - 35 < t < z + 35 for a, z in iv) for t in starts[:-1]], bool)
        big = np.abs(dev) > 10
        # task block
        task = next(root.glob("DynamicRouting1_*.hdf5"))
        h5 = npc_stim.get_h5_stim_data(task)
        fi = h5["frameIntervals"][:]
        st = (npc_stim.get_stim_start_time(h5) - s.start_time).total_seconds()
        blocks = s.vsync_times_in_blocks
        bi = min(range(len(blocks)), key=lambda i: abs(blocks[i][0] - st))
        n_gaps, resid, second, balance, raw_diff = solve_gaps(blocks[bi], fi, iv)
        r = resid * 1e3
        print(
            f"\n===== {folder} ({time.time() - t0:.0f} s): invalid_intervals={len(iv)} "
            f"({sum(z - a for a, z in iv):.1f} s)\n"
            f"  barcodes: n={starts.size}, median interval {np.median(np.diff(starts)):.4f} s; "
            f"|dev|>10 ms: {int(big.sum())} ({int((big & near).sum())} within 35 s of an invalid interval, "
            f"{int((big & ~near).sum())} not); examples (t, dev ms): "
            f"{[(round(float(t), 1), round(float(d), 1)) for t, d in zip(starts[:-1][big], dev[big])][:8]}\n"
            f"  task block {bi}: raw vsync-stim diff={raw_diff}, gaps={n_gaps}, balance={balance}; "
            + (
                f"per-gap residual ms: median={np.median(r):.3f} max={r.max():.3f} n>0.5ms={int((r > 0.5).sum())} "
                f"n>3ms={int((r > 3).sum())}; min second-best={np.nanmin(second) * 1e3:.1f} ms"
                if n_gaps
                else "no gaps"
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(f"\n===== {folder}: ERROR {type(exc).__name__}: {exc}")
    sys.stdout.flush()
