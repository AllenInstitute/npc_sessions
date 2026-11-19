"""P7 (2026-11-18): opto handling of trials overlapping a replayed interval, simulated.

No npc_lims session has a reconcilable replay inside an OptoTagging block or an opto task
block, so replayed intervals are simulated: `timing.replayed_overlap_mask` is pointed at synthetic spans (sync data unchanged, so the laser
edges are the real ones; the laser line is not unknown there).

Decisions (notes/sync-patching-handoff-2.md, 2026-11-18):
  - OptoTagging: trials overlapping a replayed interval are removed
  - task trials table: such trials are kept; opto times are NaN; `is_opto` still True

Checks (defined before running):
  G1 OptoTagging: trial_index = all trials minus exactly those whose analysis period
     [onset - period, offset + period] overlaps a synthetic span; remaining trials' columns
     (stim_start_time, stim_stop_time, location, power, wavelength, nominal_duration) equal the
     run without spans at the same trial_index.
  G2 task trials (opto session): number of trials unchanged; opto_start/stop_time NaN exactly
     in trials overlapping a span; is_opto identical to the run without spans; opto times
     identical elsewhere.

Run: ../../.venv/Scripts/python.exe p7_opto_replay_simulation.py > p7_opto_replay_simulation.log 2>&1
"""

from __future__ import annotations

import logging
import os

import numpy as np

os.environ["NPC_SESSIONS_SYNC_PATCH_DIR"] = "none"
logging.basicConfig(level=logging.ERROR)
import npc_sessions  # noqa: E402

from npc_sessions import timing  # noqa: E402
from npc_sessions.trials import DynamicRouting1, OptoTagging  # noqa: E402

OK = {True: "PASS", False: "FAIL"}
results: dict[str, list[bool]] = {}


def check(name, cond, msg):
    results.setdefault(name, []).append(bool(cond))
    print(f"  {name} {OK[bool(cond)]}: {msg}", flush=True)


_real_mask = timing.replayed_overlap_mask


def simulate(spans):
    """Make `timing.replayed_overlap_mask` (used by the TaskControl classes) see `spans` as
    the replayed intervals; None restores it."""
    if spans is None:
        timing.replayed_overlap_mask = _real_mask
    else:
        timing.replayed_overlap_mask = lambda sync, a, z: timing.unknown_overlap_mask(a, z, spans)


def overlaps(a, z, spans):
    return np.array([any(x < zz and y >= aa for x, y in spans) for aa, zz in zip(a, z)])


# ---------------------------------------------------------------- G1 OptoTagging
for sid in ("714753_2024-07-03", "733780_2024-08-30"):
    print(f"\n===== {sid} OptoTagging", flush=True)
    s = npc_sessions.DynamicRoutingSession(sid)
    for p in [p for p in s.stim_paths if p.stem.startswith("OptoTagging")]:
        base = OptoTagging(p, sync=s.sync_data)
        ft = s.stim_frame_times[p.stem]
        spans = [(float(c), float(c) + 2.0) for c in np.linspace(ft[0] + 20, ft[-1] - 20, 3)]
        simulate(spans)
        sim = OptoTagging(p, sync=s.sync_data)
        _ = sim.trial_index
        simulate(None)
        period = base._analysis_period
        expect_removed = overlaps(base.stim_start_time - period, base.stim_stop_time + period, spans)
        expect_index = base.trial_index[~expect_removed]
        same_index = np.array_equal(sim.trial_index, expect_index)
        cols = ("stim_start_time", "stim_stop_time", "location", "power", "wavelength", "nominal_duration")
        same_cols = {
            c: bool(np.array_equal(np.asarray(getattr(sim, c)), np.asarray(getattr(base, c))[~expect_removed]))
            for c in cols
        }
        check(
            "G1",
            same_index and all(same_cols.values()) and expect_removed.any(),
            f"{p.stem}: {base._len} trials, {int(expect_removed.sum())} overlap spans {spans}; "
            f"removed indices {sorted(set(base.trial_index) - set(sim.trial_index))}; columns equal {same_cols}",
        )

# ---------------------------------------------------------------- G2 opto task trials
for sid in ("744740_2024-11-14", "818720_2025-12-19"):
    print(f"\n===== {sid} task", flush=True)
    s = npc_sessions.DynamicRoutingSession(sid)
    p = s.task_path
    base = DynamicRouting1(p, sync=s.sync_data)
    if not base._is_opto:
        check("G2", False, "not an opto task file")
        continue
    opto_trials = np.flatnonzero(base.is_opto)
    pick = opto_trials[[len(opto_trials) // 4, len(opto_trials) // 2, 3 * len(opto_trials) // 4]]
    spans = [(float(base.opto_start_time[i]) - 0.01, float(base.opto_start_time[i]) + 0.01) for i in pick]
    simulate(spans)
    sim = DynamicRouting1(p, sync=s.sync_data)
    _ = (sim.opto_start_time, sim.opto_stop_time, sim.is_opto)
    simulate(None)
    hit = overlaps(base.start_time, base.stop_time, spans)
    on_nan_ok = np.array_equal(np.isnan(sim.opto_start_time), np.isnan(base.opto_start_time) | hit)
    off_nan_ok = np.array_equal(np.isnan(sim.opto_stop_time), np.isnan(base.opto_stop_time) | hit)
    same_else = np.array_equal(sim.opto_start_time[~hit], base.opto_start_time[~hit], equal_nan=True)
    check(
        "G2",
        sim._len == base._len and on_nan_ok and off_nan_ok and same_else
        and np.array_equal(sim.is_opto, base.is_opto) and hit.sum() == len(pick),
        f"{base._len} trials ({sim._len} simulated), {len(opto_trials)} opto; replayed trials {np.flatnonzero(hit).tolist()}; "
        f"opto NaN there: {on_nan_ok and off_nan_ok}; is_opto unchanged: {np.array_equal(sim.is_opto, base.is_opto)} "
        f"(True in replayed: {sim.is_opto[hit].tolist()}); other opto times identical: {same_else}",
    )

print("\n===== summary")
for k in sorted(results):
    print(f"{k}: {sum(results[k])}/{len(results[k])} pass")
