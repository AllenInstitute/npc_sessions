# QC: moving frame display-time estimation from npc_sync to npc_stim

2026-11-13 · Status: **implemented on branches, QC complete on 441 of 444 Task 2 pilot ephys sessions; not pushed or released.**
Plan: `outputs/sync-frame-timing-upgrade-plan.md` (section "Moves from npc_sync to npc_stim").
Evidence: `experiments/display-times-move/` (scripts, logs, CSVs, `qc_tables.md`, which has all the generated tables).

## Summary

- The moves in the plan's "Moves" table are done (except two items listed under [Deviations from the plan](#deviations-from-the-plan)). npc_sync no longer depends on npc_stim. The estimator was copied verbatim: `diff` of the function bodies against npc_sync 0.2.9 is empty.
- **Every per-block display-time array (2,000 blocks in 441 sessions) is byte-identical** (same dtype, shape and values) before and after. Per-block provenance and block-to-stim matching are identical in every session. So are vsync blocks, `stim_onsets`, and `get_stim_frame_times(..., "vsync")`. The deprecated npc_sync shim returns exactly the npc_stim result in 441/441 sessions.
- The npc_sessions code path (`get_stim_frame_times(..., "display_time")`) differs in 35 sessions, and every difference is a fix. In all 35, baseline **crashed with `TypeError`**: a block without photodiode times was stored as `None`, and sorting the results then failed. The new code returns arrays for the stims that do have times (34 sessions; 0→1…7 arrays each), plus an explicit `ValueError` for the failing stim. Where both versions return an array, the arrays are identical (0 changed).
- The `stim_offsets` output changes as designed in 3 sessions from 2022 that have no stim_running line: it now falls back to the last vsync, not the last display time. One of those 3 also crashed in baseline.
- Session-level failures are otherwise unchanged: 64 replayed-data and 2 timebase-discontinuity rejections, 3 `IndexError`s (2 sync files with no vsync edges, 1 empty diode block) and 1 sync `AssertionError`, all in both versions.
- Sanity checks on the new photodiode times (1,938 blocks): the count equals the vsync count and the stim-file frame count in 1,938/1,938 blocks, and times are strictly increasing in all of them. The median latency from vsync to display is 27.66 ms (the constant-lag model is 27.55 ms). **These describe behaviour that already existed, not anything the move introduced:** 78 blocks (3,941 frames) have frames displayed less than 15 ms after their vsync *rising* edge (physically impossible), and 39 blocks have a median latency of over 35 ms. (The original metric, display before the vsync *falling* edge, gives 169 blocks and 4,065 frames, but the falling edge can be late: see the 2026-11-16 note under Pre-existing issues.) The details are under [Pre-existing issues](#pre-existing-issues-unchanged-by-the-move-input-to-p2).

## What changed (branches `move-display-times-to-npc-stim`)

| Repo | Commit | Change |
|---|---|---|
| npc_sync | `59aaf91` (from `55dfefe` = 0.2.9) | drops the `npc-stim` dependency; adds `SyncDataset.get_edges_in_blocks(line, edge, end_padding)`. `frame_display_time_blocks`, `is_block_using_diode`, `constant_lag_frame_display_time_blocks`, `expected_diode_flip_rate`, `block_index_to_stim_path`, `expected_vsyncs_per_block` and the 3 stim plots become deprecated shims (lazy npc_stim import, `DeprecationWarning`). The moved constants and functions remain as deprecated module attributes (`__getattr__`). `stim_offsets` falls back to the last vsync. `validate()` hands its diode check to npc_stim and skips it with a warning if npc_stim isn't installed. The frame-timing tests moved out |
| npc_stim | `945f1f6`, `960f282` (from `1daf580` = 0.1.23) | new `display_times` module: `get_display_times(sync, stim_paths, fallback=None, raise_on_error=False)` returns one `BlockDisplayTimes` per vsync block (`display_times`, `source` ∈ photodiode / constant_lag / unavailable, `reason`, `diode_box_frame_interval`, `stim_path`). Also `get_diode_edges_in_blocks`, `get_diode_box_frame_interval`, `get_expected_diode_flip_rate`, `check_stim_photodiode`, `get_constant_lag_display_time_blocks`, the plots, and the verbatim estimator. `get_stim_frame_times` uses it and no longer sets the process-global `SYNC_ALLOW_MISSING_STIM_FILES`. New `get_stim_display_times`. Requires `npc-sync>=0.3.0` |
| npc_sessions | `0ec6d59` (from `origin/main` `65ca4bd`; the 2 unpushed local-main commits are **not** included) | `_stim_display_times` comes from `npc_stim.get_stim_display_times`, and `_stim_frame_times` is derived from it. `is_photodiode` reads `source` instead of comparing arrays against constant-lag blocks. `npc-stim>=0.1.24` |
| npc_sessions_cache | `8ba46c2` | plots use `npc_stim.get_expected_diode_flip_rate`, `plot_diode_measured_sync_square_flips` and `plot_stim_onsets/offsets(sync, get_display_times(...))`. `plot_frametime_intervals` uses `session.stim_frame_times` |

Behaviour changes that are intentional (the plan's "no defaults / explicit provenance"):
1. **Stim files are an input.** Nothing globs `*.hdf5` next to the sync file, except the deprecated npc_sync shim, which keeps the old behaviour.
2. **A missing `diodeBoxFrameInterval` key means 1, on any date** (npc_stim `960f282`). Whether the key is present depends on which TaskControl version wrote the file, not the date: code older than `11b32f5` flipped the box on every frame, even when run after 2025-07-14. An earlier version of this branch used a date cutoff instead. Its first form rejected 2 sessions from 2025-07-14 and was corrected; it has now been removed. Scan of all 2,415 cached stim files (`diode_box_key_dates.log`): every file up to 2025-07-14 17:50 lacks the key, every file from 2025-07-15 has it (value 3), and 11 dev-mouse files (366122) from 2025-01-09 have it set to 2. If a missing key ever hid an interval other than 1, the estimator's check that flips are within 5 of the vsync count would mark the block `unavailable`, rather than return wrong times.
3. **The constant-lag fallback and re-raising are parameters** (`fallback=`, `raise_on_error=`). `get_stim_frame_times` still honours the env vars `SYNC_FALLBACK_CONSTANT_LAG` and `SYNC_DEBUG` (with a deprecation warning), because `np_codeocean/scripts/upload_dynamic_routing_ecephys.py` sets them.
4. **Blocks without display times give a `ValueError` in `get_stim_frame_times`**, where baseline stored `None` and then crashed.
5. **`stim_offsets` without stim_running uses the last vsync.**

## Method

- **Population:** every top-level `DRpilot_<mouse>_<date>` folder in `//allen/.../PilotEphys/Task 2 pilot` that has an OpenEphys subfolder (`<name>*` or `Record Node*`). That gives 444 sessions from 2022–2026 (`ephys_sessions.csv`). 2 of them have no sync or stim files.
- **Inputs:** the sync `.h5` and stim `.hdf5` files were copied once to a local cache: 2,872 files, 59.1 GB, 0 failures (`cache_files.log`). Both versions read identical bytes. Where a folder has several sync files, the largest is used, as in npc_sessions. 8 sessions have more than one.
- **Environments:** two venvs with identical third-party packages (`base_freeze.txt`).
  - baseline: git worktrees at npc_sync `55dfefe` and npc_stim `1daf580`
  - new: the branch working trees, which match the commits above. All 444 new outputs were written after the last source edit.
  - npc_sessions itself isn't installed in either venv.
- **Code paths** compared per session (`qc_run.py`):
  - A: per-block display times
  - A2: the deprecated shim
  - B: `get_stim_frame_times(display_time)` with all stim files (the npc_sessions call)
  - C: the same with `vsync`
  - D: `stim_onsets` and `stim_offsets`
- **Comparison and checks:** `qc_compare.py` checks equality as dtype + shape + `np.array_equal` and computes sanity metrics. `qc_report.py` renders the tables. Per-session timeout: 1,800 s.

## Results

### Coverage

| status | sessions |
|:--|--:|
| compared | 441 |
| not run: no sync/stim files on the share (`DRpilot_724612_20240723`, `DRpilot_772655_20250305`) | 2 |
| not run: timeout in **both** versions (`DRpilot_366122_20250109`, dev mouse, `diodeBoxFrameInterval=2`) | 1 |

`py-spy` shows both versions spinning in the same pre-existing loop: `while any(anomalous_interval_indices(...))`, which deletes one flip pair per iteration (O(n²)). The runs were stopped after >30 min.

### Consistency, baseline vs new

| code path | identical | different |
|:--|--:|--:|
| vsync blocks (unchanged code; sanity check) | 441 | 0 |
| A: per-block display times | 441 | 0 |
| A: per-block source (photodiode / none) | 441 | 0 |
| A: block-to-stim-file matching | 441 | 0 |
| A2: deprecated npc_sync shim == npc_stim result | 441 | 0 |
| B: `get_stim_frame_times(display_time)` (npc_sessions path) | 406 | 35 |
| C: `get_stim_frame_times(vsync)` | 441 | 0 |
| D: `stim_onsets` | 441 | 0 |
| D: `stim_offsets` | 438 | 3 |
| **blocks:** display-time arrays | **2,000** | **0** |

**The 37 sessions that differ, and why:**

| Group | Sessions | Baseline → new |
|---|---|---|
| B: baseline crash fixed | 35: `660023_20230808`, `666986_20230816`, `667252_20230927`, `674562_20231002…05` (4), `699847_20240415/17/18`, `702136_20240305/06`, `703881_20240509`, `703882_20240424`, `706401_20240422…25` (4), `708016_20240501/02`, `712141_20240612`, `712815_20240523`, `715706_20240626/27`, `716718_20240730`, `726088_20240620`, `733780_20240905`, `733891_20240916/17/19`, `741148_20241016`, `761583_20241219`, `762526_20250320`, `818720_20251219`, `626791_20220816` | `TypeError: 'NoneType' object is not subscriptable`, so no frame times for any stim, becomes arrays for every stim that has display times (0 → 1–7 per session). The failing stim gets a `ValueError` carrying its reason. `716718_20240730` gets 0 arrays and an explicit `ValueError` for its only matched stim. |
| D: designed change | `626791_20220815/16/17` (2022, no stim_running) | `stim_offsets` = last vsync, not last display time. `…0816` also crashed in baseline (`TypeError`) and now works. |

In all 37 sessions, no array present in both versions changed (`B_diff_arrays` is empty everywhere; see `qc_sessions.csv`).

### Session-level errors

| error | A baseline | A new | B baseline | B new |
|:--|--:|--:|--:|--:|
| none | 371 | 371 | 337 | 372 |
| AssertionError: replayed sync data (`invalid_intervals`) | 64 | 64 | 64 | 64 |
| AssertionError: timebase discontinuity | 2 | 2 | 1 | 1 |
| IndexError: no vsync edges (`664566_20231121`, `772656_20250425`), or an empty diode block (`796848_20250715`) | 3 | 3 | 3 | 3 |
| AssertionError (sync `stim_running_edges`, `716718_20240731`) | 1 | 1 | 1 | 1 |
| TypeError: `None` display-time block | 0 | 0 | 35 | 0 |

### Block sources (new; baseline sources are identical)

| source | reason | blocks |
|:--|:--|--:|
| photodiode | | 1,938 |
| unavailable | N diode flips (after filtering) vs N vsyncs, off by more than 5 | 36 |
| unavailable | no stim file matched to the block | 13 |
| unavailable | estimator: "Too many frames are misaligned with vsyncs" | 8 |
| unavailable | estimator: first-frame latency assertion | 2 |
| unavailable | estimator: other assertion | 2 |
| unavailable | vsync count ≠ stim-file frame count | 1 |

Baseline logged these reasons but didn't return them. They now come back in `BlockDisplayTimes.reason`.

### Sanity checks on the new photodiode display times

| check | value |
|:--|:--|
| blocks with photodiode display times | 1,938 (1,151 with box interval 1, 787 with box interval 3) |
| n display times == n vsyncs | 1,938/1,938 |
| n display times == n frames in stim file | 1,938/1,938 |
| strictly increasing | 1,938/1,938 |
| vsync→display latency: median over blocks of the block medians | 27.66 ms (range 27.17–109.66) |
| latency p0.1, minimum over blocks | −6.38 ms |
| latency p99.9, maximum over blocks | 138.75 ms |
| blocks with at least 1 frame displayed < 15 ms after its vsync rising edge (replaces the falling-edge metric, 2026-11-16) | 78 (3,941 frames; 77 blocks / 112 frames without `733891_20240918` b4) |
| superseded: blocks with at least 1 frame displayed before its vsync falling edge | 169 (4,065 frames) |

## Pre-existing issues (unchanged by the move; input to P2)

Every array in this section is byte-identical in baseline. None of it is evidence about the move. All of it is evidence for the plan's "tolerances to remove" work.

1. **Frames displayed before their vsync:** 169 blocks, 4,065 frames.
   > **Update (2026-11-16): this metric is replaced by the rising-edge metric.** The vsync falling edge comes after `flip()` returns and can be delayed by a stall, while the frame is already on screen. 33 such frames checked against the NI-DAQ analog photodiode were on time (|err| ≤2.2 ms) although displayed up to 156 ms before their falling edge (`outputs/display-times-analog-validation.md`). The sanity metric is now "display < vsync rising edge + 15 ms", where display includes the 8 ms `MONITOR_CENTER_REFRESH_TIME`: baseline 3,941 frames in 78 blocks, final branch 0. The branch's hard check is stricter: diode flip < rising edge + `MIN_VSYNC_DIODE_FLIP_SEPARATION_SEC` (15 ms), i.e. display < rising + 23 ms, which is also 0 in the final branch (`experiments/display-times-issues/eval_rise.log`). The causes are in `outputs/display-times-issues.md` §1.

   Original text: Of those frames, 4,054 are mid-block, 11 are within the last 10 frames, and 0 are at the start (`negative_latency_positions.log`). The extreme is −147.7 ms in `744279_20250116` OptoTagging. This matches the disabled assert in the estimator ("display ≥ vsync … may be violated"). It is a misassignment of diode flips and should become a hard failure in P2.
2. **Blocks with median latency over 35 ms:** 39 blocks, 38 of them Spontaneous or SpontaneousRewards, and 38 from 2023–2024 (`latency_outliers.log`). Most sit around 44 ms (about 1 frame later than 27.6 ms). That fits the "sometimes shifts to ~0.035" note on `AVERAGE_VSYNC_TO_DIODE_FLIP_TIME`, or a whole-block 1-frame offset. `702131_20240227` block 5 (Spontaneous) has a 109.66 ms median, about 5 frames late, which is almost certainly a wrong assignment. These blocks need the NI-DAQ analog-diode check from plan P4.
3. **O(n²) anomalous-flip removal** hangs on `366122_20250109` (box interval 2).
4. Things I inspected but didn't change in this pass, because they are P2 tolerances:
   - the dead ±60-frame branch in `get_stim_frame_times` (unmatched stims silently get no key)
   - the `atol=1` count match and double-claiming in `get_stim_block_to_path`
   - the `.seconds` / `.days` start-time comparison
   - the ±5 flip tolerance
   - the ≤3 invented frames and ≤4 truncated frames at block ends
   - the 25% misalignment tolerance

## Deviations from the plan

- **`get_stim_block_to_path` was not made exact or one-to-one.** The plan's Moves table makes it "the only matcher", which it now is; making it strict is P2. I kept that out of this pass so that before/after could be byte-identical.
- **The deprecated shims cover every moved member** (including `block_index_to_stim_path` and `stim_paths`, which the table says to delete), for one release, as the plan's npc_sync row says.
- **I didn't bump versions.** The floors `npc-sync>=0.3.0` (in npc_stim) and `npc-stim>=0.1.24` (in npc_sessions) assume those releases. **npc_sync must be released before npc_stim.** `uv.lock` files were not regenerated because those versions aren't on PyPI yet.

## Not checked / blocked

| item | status |
|---|---|
| S3-backed tests (npc_sync `test_core` ×2; npc_stim `test_slow_processing_file`, `test_flips_outside_stim_running`, `test_standard_file`, `test_plots`) | **blocked**: no valid AWS credentials on this machine (`PermissionError`). The non-S3 tests pass: npc_sync 33, npc_stim 8 (including doctests). `test_missing_diode_flips_at_start_of_block` "passes" without checking anything, because an exception raised in its thread is swallowed (also pre-existing) |
| npc_sessions `Session` end-to-end (NWB, `is_photodiode`, keywords) | **not run**. The code change is small and compiles; path B covers the `get_stim_frame_times` call it wraps |
| npc_sessions_cache plots with a real `Session` | **not run**. The npc_stim plot functions and the shims were smoke-tested on `DRpilot_839146_20260512` (`plot_smoke.py`, `plots/`) |
| accuracy against the NI-DAQ analog photodiode | **not run** (plan P4). Identical outputs mean accuracy is unchanged, not that it is verified |
| npc_lims | its local main is 1 commit ahead of and 27 behind origin. The rebase conflicted in `ibl-orchestration/ibl_tracking.md` (real IBL tracking edits on both sides), so I **aborted it, leaving the repo as it was**. Not needed for this change |

## Next steps

1. Review the four branches, release npc_sync 0.3.0, then npc_stim, then update the npc_sessions and npc_sessions_cache locks.
2. P2: make the pre-existing issues above hard failures, using `qc_blocks.csv` as the regression list, and fix the O(n²) loop.
3. Rerun `qc_run.py` / `qc_compare.py` after each P2 change. Expected diffs should then be limited to the blocks listed above.

## Reproduce

```
cd experiments/display-times-move
<venv>/python list_ephys_sessions.py; <venv>/python cache_files.py C:/data/t2p_cache
C:/data/t2p_qc/venv_base/Scripts/python qc_run.py baseline --workers 6
C:/data/t2p_qc/venv_new/Scripts/python  qc_run.py new --workers 8
C:/data/t2p_qc/venv_new/Scripts/python  qc_compare.py && PYTHONIOENCODING=utf-8 C:/data/t2p_qc/venv_new/Scripts/python qc_report.py > qc_tables.md
```
Raw outputs: `C:/data/t2p_qc/out/{baseline,new}/<session>.{json,npz}`. Local cache: `C:/data/t2p_cache` (59 GB; delete when no longer needed).

## Sources

- Plan: `outputs/sync-frame-timing-upgrade-plan.md`; note: `notes/sync-frame-timing-upgrade.md`
- TaskControl `diodeBoxFrameInterval` commit: samgale/DynamicRoutingTask `11b32f5` (2025-07-14), local clone `../DynamicRoutingTask`
- QC artifacts: `experiments/display-times-move/{qc_sessions.csv, qc_blocks.csv, qc_summary.json, qc_tables.md, diode_box_key_dates.log, negative_latency_positions.log, latency_outliers.log, qc_run_*.log, cache_files.log}`
