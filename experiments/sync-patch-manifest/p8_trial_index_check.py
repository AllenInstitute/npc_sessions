"""P8 (2026-11-18): `trial_index` continues across stim files; `trial_index_in_block` is per file.

Decision (GitHub #195 discussion): in intervals tables built from several stim files,
`trial_index` is the offset-adjusted number (doesn't reset to 0), `trial_index_in_block` is the
index within the stim file; trials left out of a table change neither.

Checks (defined before running), per intervals table with a `trial_index_in_block` column:
  H1 trial_index is unique and increasing with start_time.
  H2 per stim file (rows grouped by a reset of trial_index_in_block, in time order):
     trial_index - trial_index_in_block is constant within the file, 0 for the first file, and
     equal to the cumulative total trial count of the earlier files (from the TaskControl
     objects: `_n_all_trials` / `_len_all_trials`).
  H3 trial_index_in_block equals the TaskControl object's own `trial_index_in_block` rows.
  H4 with one OptoTagging file's trials simulated as overlapping a replayed interval (via
     `timing.replayed_overlap_mask`, as in p7), the removed trials' indices are absent and the
     other rows' trial_index / trial_index_in_block are unchanged.

Sessions: 714753_2024-07-03 (2 OptoTagging), 733780_2024-08-30 (2 OptoTagging),
712141_2024-06-11 (2 RFMapping blocks in the manifest but only 1 RFMapping file in the session), 712141_2024-06-05 (2 RFMapping files).

Run: ../../.venv/Scripts/python.exe p8_trial_index_check.py > p8_trial_index_check.log 2>&1
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

os.environ["NPC_SESSIONS_SYNC_PATCH_DIR"] = "none"
logging.basicConfig(level=logging.ERROR)
import npc_session  # noqa: E402
import npc_sessions  # noqa: E402

from npc_sessions import timing  # noqa: E402

OK = {True: "PASS", False: "FAIL"}
results: dict[str, list[bool]] = {}


def check(name, cond, msg):
    results.setdefault(name, []).append(bool(cond))
    print(f"  {name} {OK[bool(cond)]}: {msg}", flush=True)


def table_checks(s, df: pd.DataFrame, name: str) -> None:
    df = df.sort_values("start_time").reset_index(drop=True)
    ti = df.trial_index.to_numpy()
    check("H1", len(np.unique(ti)) == len(ti) and np.all(np.diff(ti) > 0), f"{name}: {len(df)} rows, trial_index {ti.min()}..{ti.max()}")
    tcs = sorted(
        [(k, v) for k, v in s._all_trials.items() if npc_sessions.utils.get_taskcontrol_intervals_table_name(v.__class__.__name__) == name],
        key=lambda kv: npc_session.DatetimeRecord(kv[0]).dt,
    )
    offset, ok2, ok3, info = 0, True, True, []
    for k, v in tcs:
        n_total = getattr(v, "_n_all_trials", None) or getattr(v, "_len_all_trials")
        t0, t1 = float(np.nanmin(v.start_time)), float(np.nanmax(v.stop_time))
        rows = df[(df.start_time >= t0 - 1e-9) & (df.stop_time <= t1 + 1e-9)]
        d = (rows.trial_index - rows.trial_index_in_block).unique()
        ok2 &= len(d) == 1 and d[0] == offset
        ok3 &= np.array_equal(np.sort(rows.trial_index_in_block.to_numpy()), np.sort(np.asarray(v.trial_index_in_block)[np.isin(np.asarray(v.trial_index_in_block), rows.trial_index_in_block.to_numpy())])) and len(rows) <= len(v.trial_index_in_block)
        info.append((k, int(n_total), int(offset), d.tolist()))
        offset += int(n_total)
    check("H2", ok2, f"{name}: per file (stim, total trials, expected offset, observed offsets) {info}")
    check("H3", ok3, f"{name}: trial_index_in_block matches the TaskControl objects")


for sid in ("714753_2024-07-03", "733780_2024-08-30", "712141_2024-06-11", "712141_2024-06-05"):
    print(f"\n===== {sid}", flush=True)
    s = npc_sessions.DynamicRoutingSession(sid)
    tabs = {k: v[:] for k, v in s.intervals.items()}
    for name, df in tabs.items():
        if "trial_index_in_block" in df and name != "trials" and name != "performance":
            table_checks(s, df, name)

# H4: removal simulated in the second OptoTagging file of 714753
print("\n===== 714753_2024-07-03 (simulated replay in the 2nd OptoTagging file)", flush=True)
s0 = npc_sessions.DynamicRoutingSession("714753_2024-07-03")
base = s0.intervals["optotagging_trials"][:].sort_values("start_time").reset_index(drop=True)
p2 = sorted([p for p in s0.stim_paths if p.stem.startswith("OptoTagging")], key=lambda p: p.stem)[1]
ft = s0.stim_frame_times[p2.stem]
spans = [(float(ft[0]) + 30, float(ft[0]) + 32)]
real = timing.replayed_overlap_mask
timing.replayed_overlap_mask = lambda sync, a, z: timing.unknown_overlap_mask(a, z, spans)
try:
    s1 = npc_sessions.DynamicRoutingSession("714753_2024-07-03")
    sim = s1.intervals["optotagging_trials"][:].sort_values("start_time").reset_index(drop=True)
finally:
    timing.replayed_overlap_mask = real
removed = sorted(set(base.trial_index) - set(sim.trial_index))
kept = base[base.trial_index.isin(sim.trial_index)]
same = np.array_equal(kept.trial_index.to_numpy(), sim.trial_index.to_numpy()) and np.array_equal(
    kept.trial_index_in_block.to_numpy(), sim.trial_index_in_block.to_numpy()
)
check("H4", len(removed) > 0 and len(sim) == len(base) - len(removed) and same,
      f"span {spans}: removed trial_index {removed} (in block {base[base.trial_index.isin(removed)].trial_index_in_block.tolist()}); others unchanged: {same}")

print("\n===== summary")
for k in sorted(results):
    print(f"{k}: {sum(results[k])}/{len(results[k])} pass")
