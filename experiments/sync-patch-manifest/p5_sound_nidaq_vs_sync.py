"""P5 (2026-11-18): can NI-DAQ sound onsets replace the sync sound line inside replayed intervals?

Question: in replayed intervals the sync sound line is unknown, so sound onsets are NaN in the
trials tables. Ephys sessions also record the sound on the NI-DAQ (npc_samstim
`get_stim_latencies_from_nidaq_recording`, analog channel from
`get_nidaq_channel_for_stim_onset`), timed on sync through the barcode fit (npc_ephys), which
doesn't need barcodes inside the replayed interval.

Measurements (defined before running), per session and stim file with sound (task, RFMapping):
  E1 on trials with both: d = onset(NI-DAQ) - onset(sync line): n, median, p1, p99, max |d - median|.
  E2 trials whose sync onset is masked (sound line unknown): NI-DAQ onset finite? And is it
     inside the window [trigger, trigger + 0.1 s]?
Acceptance for using NI-DAQ onsets (+ the session's median d removed) in masked trials:
  max |d - median| <= 1 ms on every stim file, and every masked trial gets a finite onset.

Run: ../../.venv/Scripts/python.exe p5_sound_nidaq_vs_sync.py > p5_sound_nidaq_vs_sync.log 2>&1
"""

from __future__ import annotations

import logging
import os
import sys
import time

import numpy as np

os.environ["NPC_SESSIONS_SYNC_PATCH_DIR"] = "none"
logging.basicConfig(level=logging.ERROR)
import npc_samstim  # noqa: E402
import npc_sessions  # noqa: E402

from npc_sessions import timing  # noqa: E402

SESSIONS = sys.argv[1:] or (
    "839146_2026-05-18",
    "835598_2026-02-27",
    "837959_2026-04-02",
    "856847_2026-06-04",
    "862025_2026-08-05",
)


def onsets(recs):
    return np.array([np.nan if r is None else float(r.onset_time_on_sync) for r in recs], dtype=float)


def triggers(recs):
    return np.array(
        [np.nan if r is None else float(r.presentation.trigger_time_on_sync) for r in recs],
        dtype=float,
    )


for sid in SESSIONS:
    t0 = time.time()
    print(f"\n===== {sid}", flush=True)
    s = npc_sessions.DynamicRoutingSession(sid)
    print(f"  is_ephys {s.is_ephys}; replayed {s._raw_sync_data.invalid_intervals}")
    if not s.is_ephys:
        continue
    sync = s.sync_data
    line = sync.get_line_for_stim_onset("sound")
    for stem in s.stim_data_without_timing_issues:
        if not any(k in stem for k in ("DynamicRouting1", "RFMapping")):
            continue
        path = next(p for p in s.stim_paths if p.stem == stem)
        try:
            rs = npc_samstim.get_stim_latencies_from_sync(path, sync, waveform_type="sound")
        except Exception as exc:  # noqa: BLE001
            print(f"  {stem}: sync recordings failed: {exc!r}")
            continue
        masked = timing.stim_recordings_unknown_mask(rs, sync, line, include_offset=False)
        rn = npc_samstim.get_stim_latencies_from_nidaq_recording(
            path, sync, s.ephys_recording_dirs, waveform_type="sound"
        )
        a, b = onsets(rs), onsets(rn)
        both = np.isfinite(a) & np.isfinite(b) & ~masked
        d = b[both] - a[both]
        med = float(np.median(d)) if len(d) else np.nan
        dev = np.abs(d - med)
        print(
            f"  {stem}: E1 n={both.sum()} median d={med * 1e3:.3f} ms "
            f"p1={np.percentile(d, 1) * 1e3:.3f} p99={np.percentile(d, 99) * 1e3:.3f} ms "
            f"max|d-med|={dev.max() * 1e3:.3f} ms; n>1ms={int((dev > 1e-3).sum())}"
            if len(d)
            else f"  {stem}: E1 n=0"
        )
        if len(d) and (dev > 1e-3).any():
            worst = np.argsort(dev)[::-1][:5]
            print(f"      worst d-med (ms): {np.round((d[worst] - med) * 1e3, 3).tolist()}")
        trig = triggers(rs)
        for i in np.flatnonzero(masked):
            ok = np.isfinite(b[i]) and trig[i] <= b[i] <= trig[i] + 0.1
            print(
                f"    E2 trial {i}: trigger {trig[i]:.4f}; sync onset (masked) {a[i]:.4f}; "
                f"NI-DAQ onset {b[i]:.4f} (latency {(b[i] - trig[i]) * 1e3:.2f} ms) {'OK' if ok else 'BAD'}"
            )
    print(f"  ({time.time() - t0:.0f} s)", flush=True)
