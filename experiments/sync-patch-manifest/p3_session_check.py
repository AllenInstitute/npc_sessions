"""P3 check: npc_sessions with the timing module on real sessions with a replayed interval.

Checks (defined before running):
  C1 sync_patch exists; sync_data is patched; no edge query raises.
  C2 sync_timing_report: the block overlapping the replay is 'patched' (display times from the
     photodiode, interpolated frames > 0); other blocks 'clean'.
  C3 invalid_times contains each replayed interval (reason names the unknown lines).
  C4 trials: built; none overlaps an invalid_times entry; `timing_source` column present.
  C5 licks: `timing_source` column; reconstructed ('stim_frame') licks lie inside the lick
     line's unknown spans; sync licks don't.
  C6 rewards (_reward_frame_times): `timing_source` present; reconstructed rewards inside the
     reward line's unknown spans (if any rewardFrames fall there).
  C7 stim_frame_time_sources: the patched stim has interpolated frames.

Sessions: 856847_2026-06-04 (replay inside the task block), 862025_2026-08-05 (inside RFMapping).
Read-only (patches are built in memory: NPC_SESSIONS_SYNC_PATCH_DIR unset).

Run: PYTHONPATH="../../../npc_sync/src;../../../npc_stim/src;../../src" ../../.venv/Scripts/python.exe p3_session_check.py > p3_session_check.log 2>&1
"""

from __future__ import annotations

import logging
import os
import sys
import time

import numpy as np

os.environ.pop("NPC_SESSIONS_SYNC_PATCH_DIR", None)
logging.basicConfig(level=logging.ERROR)
import npc_sessions  # noqa: E402
import npc_sync  # noqa: E402

print("npc_sessions", npc_sessions.__file__, "| npc_sync", npc_sync.__file__)
OK = {True: "PASS", False: "FAIL"}
results: dict[str, list[bool]] = {}


def check(name, cond, msg):
    results.setdefault(name, []).append(bool(cond))
    print(f"  {name} {OK[bool(cond)]}: {msg}", flush=True)


for sid in sys.argv[1:] or ("856847_2026-06-04", "862025_2026-08-05"):
    t0 = time.time()
    print(f"\n===== {sid}", flush=True)
    s = npc_sessions.DynamicRoutingSession(sid)
    raw = s._raw_sync_data
    patch = s.sync_patch
    check("C1", patch is not None and s.sync_data.patch is patch, f"patch: {patch.summary() if patch else None}")
    errs = []
    for bit in range(32):
        try:
            s.sync_data.get_rising_edges(bit, units="seconds")
        except Exception as exc:  # noqa: BLE001
            errs.append((bit, repr(exc)[:60]))
    check("C1", not errs, f"edge queries on all 32 bits: errors {errs}")
    rep = s.sync_timing_report
    print(rep[["block_index", "stim", "n_stim_frames", "n_vsyncs_recorded", "status", "display_time_source", "n_interpolated_frames", "anchor_residuals_ms", "reason"]].to_string())
    inv = raw.invalid_intervals
    for _, r in rep.iterrows():
        ov = any(a <= r.stop_time and z >= r.start_time for a, z in inv)
        if ov:
            check("C2", r.status == "patched" and r.display_time_source == "photodiode" and r.n_interpolated_frames > 0,
                  f"block {r.block_index} ({r.stim}) overlapping replay: {r.status}, {r.display_time_source}, {r.n_interpolated_frames} interpolated")
        else:
            check("C2", r.status == "clean", f"block {r.block_index} ({r.stim}): {r.status} {r.reason}")
    it = s.invalid_times[:] if s.invalid_times is not None else None
    print(it)
    for a, z in inv:
        check("C3", it is not None and any((x.start_time <= a) & (x.stop_time >= z) for x in it.itertuples()), f"invalid_times covers {a:.3f}-{z:.3f}")
    trials = s.trials[:]
    bad = [t for t in trials.itertuples() if any(x.start_time < t.stop_time and x.stop_time > t.start_time for x in it.itertuples())]
    check("C4", "timing_source" in trials and not bad, f"{len(trials)} trials, overlapping invalid_times: {len(bad)}; timing_source: {trials.timing_source.value_counts().to_dict()}")
    licks = s._licks[:]
    uk = s.sync_data.get_unknown_intervals("lick_sensor")
    rec = licks[licks.timing_source == "stim_frame"]
    meas = licks[licks.timing_source == "sync"]
    check("C5", "timing_source" in licks and npc_sync.in_intervals(rec.timestamps, uk).all() and not npc_sync.in_intervals(meas.timestamps, uk).any(),
          f"licks: {licks.timing_source.value_counts().to_dict()}; reconstructed {np.round(rec.timestamps.to_numpy(), 3).tolist()} in {uk}")
    rw = s._reward_frame_times[:]
    uk15 = s.sync_data.get_unknown_intervals(15)
    if "timing_source" in rw:
        rr = rw[rw.timing_source == "stim_frame"]
        check("C6", npc_sync.in_intervals(rr.timestamps, uk15).all(), f"rewards: {rw.timing_source.value_counts().to_dict()}")
    else:
        check("C6", False, f"rewards: no timing_source column ({list(rw.columns)})")
    src = s.stim_frame_time_sources
    n = {k: int((v == 1).sum()) for k, v in src.items()}
    check("C7", any(v > 0 for v in n.values()), f"interpolated frames per stim: {n}")
    print(f"  ({time.time() - t0:.0f} s)")

print("\n===== summary")
for k in sorted(results):
    print(f"{k}: {sum(results[k])}/{len(results[k])} pass")
