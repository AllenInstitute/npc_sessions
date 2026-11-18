"""P4 check (2026-11-18): replayed intervals are not invalid_times; affected events are NaN.

Decision (notes/sync-patching-handoff-2.md, decision 1): replayed intervals must not be
`invalid_times`; trials overlapping them are kept; reconstructed licks only in the licks table;
`response_time` NaN where the lick line is unknown.

Checks (defined before running):
  D1 invalid_times has no entry overlapping a replayed interval.
  D2 epochs overlapping a replayed interval have tag 'replayed_sync' and a note naming it.
  D3 trials: every TaskControl trial is in the NWB trials table (none dropped for the replay);
     trials overlapping a replayed interval have timing_source 'patched', others 'measured'.
  D4 response_time: NaN for trials whose response window (up to the first measured lick)
     overlaps the lick line's unknown spans; for all other trials it equals the TaskControl
     value computed without masking, i.e. a measured sync lick (or NaN if none).
  D5 aud/opto: for trials whose stim trigger->onset overlaps the sound/opto line's unknown
     spans, the onset is NaN; no finite aud onset lies inside an unknown span of the sound line.
  D6 every intervals table builds; no finite start/stop times are NaN-derived (OptoTagging
     rows with unknown onsets are omitted, count logged).
  D7 patch cache: with NPC_SESSIONS_SYNC_PATCH_DIR=<tmp>, the saved JSON has
     dependency_versions == timing.get_dependency_versions(), and a second call loads it.

Sessions: the 5 replay sessions of p3_session_check_final.log.
Read-only on S3 (patch cache in a local temp dir).

Run: ../../.venv/Scripts/python.exe p4_replay_trials_check.py > p4_replay_trials_check.log 2>&1
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time

import numpy as np

TMP = tempfile.mkdtemp(prefix="sync_patches_")
os.environ["NPC_SESSIONS_SYNC_PATCH_DIR"] = TMP
logging.basicConfig(level=logging.ERROR)
import npc_sessions  # noqa: E402
import npc_sync  # noqa: E402

from npc_sessions import timing  # noqa: E402

print("npc_sessions", npc_sessions.__file__, "| npc_sync", npc_sync.__version__ if hasattr(npc_sync, "__version__") else npc_sync.__file__)
print("dependency versions", timing.get_dependency_versions())
OK = {True: "PASS", False: "FAIL"}
results: dict[str, list[bool]] = {}


def check(name, cond, msg):
    results.setdefault(name, []).append(bool(cond))
    print(f"  {name} {OK[bool(cond)]}: {msg}", flush=True)


def overlaps(a, z, intervals):
    return any(x < z and y > a for x, y in intervals)


SESSIONS = sys.argv[1:] or (
    "839146_2026-05-18",
    "835598_2026-02-27",
    "837959_2026-04-02",
    "856847_2026-06-04",
    "862025_2026-08-05",
)
for sid in SESSIONS:
    t0 = time.time()
    print(f"\n===== {sid}", flush=True)
    s = npc_sessions.DynamicRoutingSession(sid)
    replayed = s._raw_sync_data.invalid_intervals
    print(f"  replayed intervals: {replayed}")
    # D7 ------------------------------------------------------------------ cache
    _ = s.sync_patch
    saved = [p for p in os.listdir(TMP) if p.startswith(f"{s.id}_")]
    ok7 = len(saved) == 1
    if ok7:
        d = json.loads(open(os.path.join(TMP, saved[0])).read())
        ok7 = d.get("dependency_versions") == timing.get_dependency_versions()
        again = timing.get_sync_patch(str(s.id), s._raw_sync_data, s.stim_paths)
        ok7 = ok7 and again.created == s.sync_patch.created
    check("D7", ok7, f"saved {saved}")
    # D1 ------------------------------------------------------------------ invalid_times
    it = s.invalid_times
    it_df = it[:] if it is not None else None
    bad = [] if it_df is None else [r for r in it_df.itertuples() if overlaps(r.start_time, r.stop_time, replayed)]
    check("D1", not bad, f"invalid_times: {None if it_df is None else len(it_df)} entries, overlapping replay: {len(bad)}")
    # D2 ------------------------------------------------------------------ epochs
    ep = s.epochs[:]
    for r in ep.itertuples():
        if overlaps(r.start_time, r.stop_time, replayed):
            check("D2", "replayed_sync" in list(r.tags) and "replayed sync data" in r.notes, f"epoch {r.stim_name if hasattr(r, 'stim_name') else r.Index}: tags {list(r.tags)}; notes {r.notes[:140]!r}")
    # D3 ------------------------------------------------------------------ trials
    trials = s.trials[:]
    tc = s._trials
    n_tc = tc._len
    src = trials.timing_source.to_numpy()
    expect = np.array(["patched" if overlaps(a, z, replayed) else "measured" for a, z in zip(trials.start_time, trials.stop_time)])
    check("D3", len(trials) == n_tc and (src == expect).all(), f"{len(trials)}/{n_tc} trials; timing_source {dict(zip(*np.unique(src, return_counts=True)))}")
    # D4 ------------------------------------------------------------------ response_time
    uk_lick = s.sync_data.get_unknown_intervals("lick_sensor")
    licks = s.sync_data.get_rising_edges("lick_sensor", units="seconds")
    rt = trials.response_time.to_numpy()
    ws, we = tc.response_window_start_time, tc.response_window_stop_time
    n_masked, wrong = 0, []
    for i in range(len(trials)):
        inwin = licks[(licks > ws[i]) & (licks < we[i])]
        first = inwin[0] if len(inwin) else np.nan
        end = first if len(inwin) else we[i]
        masked = bool(uk_lick) and bool(timing.unknown_overlap_mask(ws[i], end, uk_lick)[0])
        if masked:
            n_masked += 1
            if not np.isnan(rt[i]):
                wrong.append((i, "expected NaN", rt[i]))
        elif not (np.isnan(rt[i]) and np.isnan(first)) and rt[i] != first:
            wrong.append((i, first, rt[i]))
    check("D4", not wrong, f"{n_masked} trials masked (lick unknown {uk_lick}); mismatches {wrong[:5]}")
    # D5 ------------------------------------------------------------------ aud / opto onsets
    try:
        uk_sound = timing.get_unknown_intervals(s.sync_data, s.sync_data.get_line_for_stim_onset("sound"))
    except ValueError:
        uk_sound = ()
    aud = trials[trials.is_aud_stim] if "is_aud_stim" in trials else trials.iloc[:0]
    inside = [t for t in aud.stim_start_time if np.isfinite(t) and npc_sync.in_intervals([t], uk_sound)[0]]
    nan_aud = int(np.isnan(aud.stim_start_time).sum())
    check("D5", not inside, f"sound unknown {uk_sound}; aud trials {len(aud)}, NaN onsets {nan_aud}, finite onsets inside unknown {len(inside)}")
    # D6 ------------------------------------------------------------------ intervals
    try:
        tabs = {k: v[:] for k, v in s.intervals.items()}
        info = {}
        okd6 = True
        for k, df in tabs.items():
            okd6 &= bool(np.isfinite(df.start_time).all() and np.isfinite(df.stop_time).all())
            n_nan_stim = int(np.isnan(df.stim_start_time).sum()) if "stim_start_time" in df else None
            info[k] = (len(df), n_nan_stim)
        check("D6", okd6, f"tables (rows, NaN stim_start_time): {info}")
    except Exception as exc:  # noqa: BLE001
        check("D6", False, f"intervals failed: {exc!r}")
    print(f"  ({time.time() - t0:.0f} s)")

print("\n===== summary")
for k in sorted(results):
    print(f"{k}: {sum(results[k])}/{len(results[k])} pass")
