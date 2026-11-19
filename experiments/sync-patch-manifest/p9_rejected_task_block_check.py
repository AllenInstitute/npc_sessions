"""P9 (2026-11-18): a stim file without usable frame times gets no table and no epoch.

Session 715706_2024-06-26: the task block (DynamicRouting1_715706_20240626_144020) is rejected by
npc_stim ("1 frame(s) have a diode flip less than 15 ms after their vsync rising edge").
Before: `epochs` (and everything that uses it: `intervals`, keywords, NWB) raised, because
`stim_frame_times` asserted every stim unless `ignore_stim_errors`.

Checks (defined before running):
  J1 `stim_frame_times` has no entry for the rejected stim; every other stim is present.
  J2 `epochs` builds; no epoch for the rejected stim; one per other stim.
  J3 `is_task` is False (existing behaviour: `task_data` only considers stims with frame times);
     `trials` raises AttributeError naming the reason.
  J4 `intervals` builds, without `trials`/`performance`; other tables present.
  J5 `keywords` builds (no good/bad behavior keyword); `_all_processing` modules without
     `quiescent_interval_violations`.
  J6 control 714753_2024-07-03 (task block accepted): `is_task` True, trials and
     performance present.

Run: ../../.venv/Scripts/python.exe p9_rejected_task_block_check.py > p9_rejected_task_block_check.log 2>&1
"""

from __future__ import annotations

import logging
import os

os.environ["NPC_SESSIONS_SYNC_PATCH_DIR"] = "none"
logging.basicConfig(level=logging.ERROR)
import npc_sessions  # noqa: E402

OK = {True: "PASS", False: "FAIL"}
results: dict[str, list[bool]] = {}


def check(name, cond, msg):
    results.setdefault(name, []).append(bool(cond))
    print(f"  {name} {OK[bool(cond)]}: {msg}", flush=True)


print("\n===== 715706_2024-06-26", flush=True)
s = npc_sessions.DynamicRoutingSession("715706_2024-06-26")
rejected = [k for k, v in s._stim_frame_times.items() if isinstance(v, Exception)]
print(f"  rejected stims: {rejected}")
ft = s.stim_frame_times
check("J1", rejected and not set(rejected) & set(ft) and set(ft) | set(rejected) == set(s._stim_frame_times),
      f"stim_frame_times: {sorted(ft)}")
ep = s.epochs[:]
n_ep = len(ep)
check("J2", n_ep == len(ft), f"{n_ep} epochs for {len(ft)} stims with frame times; script names {ep.script_name.tolist()}")
try:
    _ = s.trials
    check("J3", False, "trials did not raise")
except AttributeError as exc:
    check("J3", not s.is_task and "usable frame times" in str(exc), f"is_task={s.is_task}; {str(exc)[:200]}")
tabs = list(s.intervals.keys())
check("J4", "performance" not in tabs and "trials" not in tabs and len(tabs) > 0, f"intervals: {tabs}")
kw = s.keywords
try:
    mods = [m.name for m in s._behavior]  # processing modules list
except Exception as exc:  # noqa: BLE001
    mods = repr(exc)[:100]
check("J5", "good_behavior" not in kw and "bad_behavior" not in kw and (isinstance(mods, list) and "quiescent_interval_violations" not in mods),
      f"keywords {kw}; behavior modules {mods}")

print("\n===== 714753_2024-07-03 (control)", flush=True)
c = npc_sessions.DynamicRoutingSession("714753_2024-07-03")
check("J6", c.is_task and "performance" in c.intervals and len(c.trials[:]) > 0,
      f"is_task={c.is_task}; intervals {list(c.intervals.keys())}; {len(c.trials[:])} trials")

print("\n===== summary")
for k in sorted(results):
    print(f"{k}: {sum(results[k])}/{len(results[k])} pass")
