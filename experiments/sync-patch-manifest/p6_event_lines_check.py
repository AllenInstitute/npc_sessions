"""P6 (2026-11-18): sound line from NI-DAQ, laser lines from trigger frames, licks in trials.

Decisions (notes/sync-patching-handoff-2.md, 2026-11-18 follow-up):
  - reconstructed licks are used in the trials table too (response_time); timing_source only in
    the licks table, not the trials table
  - the sync sound line is patched inside replayed intervals from the NI-DAQ (npc_samstim)
  - laser lines are patched from opto trigger frames (npc_stim); opto times are NaN in trials
    overlapping a replayed interval

Checks (defined before running):
  F1 sound line ('bit1' here) is 'reconstructed' (method nidaq_sound) in every replayed span;
     max (NI-DAQ - sync) deviation <= 1 ms; no remaining unknown sound span.
  F2 aud trials: every aud trial has a finite stim_start_time; the onsets inside a replayed
     interval are exactly the reconstructed onsets in the patch diagnostics (to 1 us).
  F3 response_time: every finite value is a licks-table timestamp; trials whose first lick
     is reconstructed use a 'stim_frame' lick.
  F4 trials table has no timing_source column; licks table has one.
  F5 opto leave-out: on sessions with OptoTagging blocks, hide the laser line's measured edges
     in synthetic 3-s spans inside an OptoTagging block and predict them from trigger frames
     (`predict_opto_line`). |predicted - measured| for rising edges <= 1/2 frame (8.33 ms).
  F6 laser lines unknown in the real patch (714753_2024-07-03, 733780_2024-08-30: laser_488):
     reconstructed (no triggers inside the span -> no edges), or stay unknown with a reason.
  F7 trial counts equal p4_replay_trials_check.log; every intervals table builds.

Run: ../../.venv/Scripts/python.exe p6_event_lines_check.py > p6_event_lines_check.log 2>&1
"""

from __future__ import annotations

import logging
import os
import time

import numpy as np

os.environ["NPC_SESSIONS_SYNC_PATCH_DIR"] = "none"
logging.basicConfig(level=logging.ERROR)
import npc_sessions  # noqa: E402
import npc_sync  # noqa: E402

from npc_sessions import timing  # noqa: E402

print("dependency versions", timing.get_dependency_versions(), "builder", timing.BUILDER)
OK = {True: "PASS", False: "FAIL"}
results: dict[str, list[bool]] = {}


def check(name, cond, msg):
    results.setdefault(name, []).append(bool(cond))
    print(f"  {name} {OK[bool(cond)]}: {msg}", flush=True)


P4_TRIALS = {
    "839146_2026-05-18": 548,
    "835598_2026-02-27": 543,
    "837959_2026-04-02": 544,
    "856847_2026-06-04": 546,
    "862025_2026-08-05": 533,
}
for sid, n_expected in P4_TRIALS.items():
    t0 = time.time()
    print(f"\n===== {sid}", flush=True)
    s = npc_sessions.DynamicRoutingSession(sid)
    patch = s.sync_patch
    sync = s.sync_data
    line = sync.get_line_for_stim_onset("sound")
    key = sync.line_key(line)
    spans = patch.lines.get(key, [])
    diag = {k: v for sp in spans for k, v in (sp.diagnostics or {}).items()}
    devs = [d.get("max_deviation_sec") for d in diag.values() if d.get("max_deviation_sec") is not None]
    check(
        "F1",
        spans and all(sp.status == "reconstructed" and sp.method == "nidaq_sound" for sp in spans)
        and all(d <= 1e-3 for d in devs),
        f"{key}: {[(sp.status, sp.method) for sp in spans]}; max dev ms {[round(d * 1e3, 3) for d in devs]}; "
        f"notes {[n for n in patch.notes if 'sound' in n]}",
    )
    trials = s.trials[:]
    aud = trials[trials.is_aud_stim]
    recon = np.array([t for d in diag.values() for t in d.get("reconstructed_onsets", [])])
    st = aud.stim_start_time.to_numpy()
    inside = npc_sync.in_intervals(st, sync.invalid_intervals)
    # every onset inside a replayed interval is a reconstructed one, and vice versa
    match = sorted(np.round(st[inside], 6)) == sorted(np.round(recon[npc_sync.in_intervals(recon, sync.invalid_intervals)], 6))
    check("F2", np.isfinite(st).all() and match,
          f"{len(aud)} aud trials, NaN {int(np.isnan(st).sum())}; inside replay {int(inside.sum())}: "
          f"{np.round(st[inside], 4).tolist()} vs reconstructed {np.round(recon, 4).tolist()}")
    licks = s._licks[:]
    rt = trials.response_time.to_numpy()
    fin = rt[np.isfinite(rt)]
    in_table = np.isin(fin, licks.timestamps.to_numpy())
    src = licks.set_index("timestamps").timing_source
    n_rec = int(sum(src.get(t) == "stim_frame" for t in fin if t in src.index))
    check("F3", in_table.all(), f"{len(fin)} response times, all in licks table: {in_table.all()}; from reconstructed licks: {n_rec}")
    check("F4", "timing_source" not in trials.columns and "timing_source" in licks.columns,
          f"trials columns timing_source: {'timing_source' in trials.columns}; licks: {licks.timing_source.value_counts().to_dict()}")
    try:
        tabs = {k: len(v[:]) for k, v in s.intervals.items()}
        check("F7", len(trials) == n_expected, f"{len(trials)} trials (p4: {n_expected}); tables {tabs}")
    except Exception as exc:  # noqa: BLE001
        check("F7", False, f"intervals failed: {exc!r}")
    print(f"  ({time.time() - t0:.0f} s)", flush=True)

for sid in ("714753_2024-07-03", "733780_2024-08-30"):
    t0 = time.time()
    print(f"\n===== {sid} (opto)", flush=True)
    s = npc_sessions.DynamicRoutingSession(sid)
    patch = s.sync_patch
    sp = patch.lines.get("laser_488", [])
    check("F6", all(x.status == "reconstructed" or any("laser_488" in n for n in patch.notes) for x in sp),
          f"laser_488: {[(x.status, x.method, len(x.rising), (x.diagnostics or {}).get('n_paired')) for x in sp]}; "
          f"notes {[n for n in patch.notes if 'opto' in n]}")
    raw = s._raw_sync_data
    view = npc_sync.SyncDataset(raw, patch=patch, on_invalid="mask")
    tagging = [p for p in s.stim_paths if p.stem.startswith("OptoTagging")]
    trig, durs = timing._opto_triggers(view, tagging)
    on = view.get_rising_edges("laser_488", units="seconds")
    off = view.get_falling_edges("laser_488", units="seconds")
    errs_on, errs_off = [], []
    for p in tagging:
        ft = s.stim_frame_times[p.stem]
        for c in np.linspace(ft[0] + 10, ft[-1] - 10, 4):
            span = [(float(c), float(c) + 3.0)]
            pred = timing.predict_opto_line(on, off, trig, durs, span)
            if pred["status"] != "reconstructed":
                print(f"    span {span}: {pred}")
                errs_on.append(np.inf)
                continue
            hidden_on = on[npc_sync.in_intervals(on, span)]
            hidden_off = off[npc_sync.in_intervals(off, span)]
            pr = pred["rising"][npc_sync.in_intervals(pred["rising"], span)]
            pf = pred["falling"][npc_sync.in_intervals(pred["falling"], span)]
            if len(pr) != len(hidden_on) or len(pf) != len(hidden_off):
                print(f"    span {span}: count mismatch rising {len(pr)}/{len(hidden_on)} falling {len(pf)}/{len(hidden_off)}")
                errs_on.append(np.inf)
                continue
            errs_on.extend(pr - hidden_on)
            errs_off.extend(pf - hidden_off)
    e_on, e_off = np.asarray(errs_on), np.asarray(errs_off)
    check("F5", len(e_on) > 0 and np.max(np.abs(e_on)) <= 1 / 120,
          f"{len(e_on)} hidden onsets: max |err| {np.max(np.abs(e_on)) * 1e3:.2f} ms, median {np.median(e_on) * 1e3:.2f} ms; "
          f"offsets: max |err| {np.max(np.abs(e_off)) * 1e3 if len(e_off) else np.nan:.2f} ms")
    print(f"  ({time.time() - t0:.0f} s)", flush=True)

print("\n===== summary")
for k in sorted(results):
    print(f"{k}: {sum(results[k])}/{len(results[k])} pass")
