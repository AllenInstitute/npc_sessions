"""Photodiode prediction gate on every replay-reconcilable block in the manifest, old check
(max error over 3 s after the span) vs new (npc_sync.patch.DIODE_CHECK_WINDOW_SEC = 0.5 s).

Stim files without `diodeBoxFrameInterval` use 1 (npc_stim.get_diode_box_frame_interval).
Checks: count of gate failures under each; for failures, error of the first 4 edges after the
span and when the worst edge occurs.

Run: PYTHONPATH=../../../npc_sync/src ../../.venv/Scripts/python.exe diode_gate_check.py > diode_gate_check.log 2>&1
"""

from __future__ import annotations

import csv
import logging

import numpy as np
import pandas as pd

import npc_sync
from npc_sync import patch as P

from common import match_stims_to_blocks, read_stim, sync_start

logging.disable(logging.WARNING)
man = pd.read_csv("manifest.csv")
rows = {r["session"]: r for r in csv.DictReader(open("sessions.csv"))}
todo = man[man.cls == "replay-reconcilable"]
out = []
for session, grp in todo.groupby("session"):
    row = rows[session]
    raw = npc_sync.SyncDataset(row["sync_path"], on_invalid="ignore")
    blocks = list(raw.vsync_times_in_blocks)
    if row["source"] == "s3":
        import io

        import npc_io

        stims = [read_stim(io.BytesIO(npc_io.from_pathlike(p).read_bytes()), sync_start(raw), name=npc_io.from_pathlike(p).stem)
                 for p in row["stim_paths"].split(";") if p]
    else:
        stims = [read_stim(p, sync_start(raw)) for p in row["stim_paths"].split(";") if p]
    matches = match_stims_to_blocks(blocks, stims)[0]
    fs = raw.sample_freq
    spans = [(a - 1 / fs, z) for a, z in raw.invalid_intervals]
    dr = raw.get_rising_edges("stim_photodiode", units="seconds")
    df = raw.get_falling_edges("stim_photodiode", units="seconds")
    for bi in grp.block_index.astype(int):
        m = matches[bi]
        box = m.stim.diode_box_frame_interval or 1
        b = blocks[bi]
        ov = [(a, z) for a, z in spans if a <= b[-1] and z > b[0]]
        rec = P.solve_vsync_gaps(b, m.stim.frame_intervals, ov)
        for g in rec.gaps:
            _, _, sol = P.predict_diode_edges(rec.times, dr, df, g.span, box)
            # old check: max over 3 s after
            a, z = g.span
            e = np.sort(np.r_[dr, df])
            post = e[(e >= z) & (e < z + 3)]
            pred = np.sort(np.r_[rec.times[::box] + sol.latency_rising_sec, rec.times[::box] + sol.latency_falling_sec]) if sol.status == "reconstructed" else np.array([])
            err = np.array([np.min(np.abs(pred - x)) for x in post]) if len(pred) else np.array([np.nan])
            old_fail = sol.status != "reconstructed" or not (np.nanmax(err) <= sol.baseline_error_sec + P.DIODE_MAX_RESIDUAL_ABOVE_BASELINE_SEC)
            out.append(dict(session=session, block_index=bi, box=box, box_from_file=m.stim.diode_box_frame_interval is not None,
                            span=f"{a:.3f}-{z:.3f}", status=sol.status, reason=sol.reason,
                            latency_rising_ms=sol.latency_rising_sec * 1e3, latency_falling_ms=sol.latency_falling_sec * 1e3,
                            baseline_ms=sol.baseline_error_sec * 1e3, after_0p5s_ms=sol.after_error_sec * 1e3,
                            after_3s_ms=float(np.nanmax(err)) * 1e3, worst_edge_after_z_s=float(post[np.nanargmax(err)] - z) if len(post) and np.isfinite(err).any() else np.nan,
                            first4_ms=np.round(err[:4] * 1e3, 2).tolist(), polarity_errors=sol.n_polarity_errors_after,
                            n_inserted=sol.n_inserted, n_removed=sol.n_removed,
                            fail_old=bool(old_fail), fail_new=bool(sol.failures()), failures_new=sol.failures()))
d = pd.DataFrame(out)
d.to_csv("diode_gate_check.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 80)
print(f"spans: {len(d)} in {d.session.nunique()} sessions; box from file: {d.box_from_file.sum()}, defaulted to 1: {(~d.box_from_file).sum()}")
print(f"status: {d.status.value_counts().to_dict()}")
print(f"gate failures: old (3 s) {d.fail_old.sum()}, new (0.5 s) {d.fail_new.sum()}")
print(f"baseline ms quantiles: {d.baseline_ms.quantile([.5, .9, .99, 1]).round(3).to_dict()}")
print(f"after (0.5 s) ms quantiles: {d.after_0p5s_ms.quantile([.5, .9, .99, 1]).round(3).to_dict()}")
print(f"after (0.5 s) - baseline ms quantiles: {(d.after_0p5s_ms - d.baseline_ms).quantile([.5, .9, .99, 1]).round(3).to_dict()}")
print("\nold-gate failures:")
print(d[d.fail_old][["session", "block_index", "box", "span", "baseline_ms", "after_0p5s_ms", "after_3s_ms", "worst_edge_after_z_s", "first4_ms", "fail_new"]].to_string())
print("\nnew-gate failures:")
print(d[d.fail_new][["session", "block_index", "box", "span", "status", "baseline_ms", "after_0p5s_ms", "first4_ms", "failures_new"]].to_string())
