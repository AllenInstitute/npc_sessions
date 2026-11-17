# Handoff: targeted P4, validate display times against the NI-DAQ analog photodiode

Written 2026-11-15. Status: **validation done 2026-11-16** (`outputs/display-times-analog-validation.md`). Afterwards steps 1–4 done, with no branch change needed. Steps 5–7 (release, P2 leftovers, cleanup) are open and up to the user.

## Task

Run a targeted version of P4 from `outputs/sync-frame-timing-upgrade-plan.md`: check frame display times against the NI-DAQ analog photodiode. This is a sample-based validation of the fixes branch. It is **not** the full rollout described in the plan.

## Read first

1. `outputs/display-times-issues.md`: issues §1–§8, the fixes, and the remaining open questions. This is the main context.
2. `outputs/display-times-move-qc.md`: the move QC (baseline is byte-identical to the move branch).
3. `outputs/sync-frame-timing-upgrade-plan.md`: section P4 and the "Open questions" (the analog channel mapping).
4. `experiments/display-times-issues/README.md`: scripts, run labels, CSVs.

## State

- **Code under test:** npc_stim branch `display-times-p2-fixes`, commit `1431002` (6 commits on `move-display-times-to-npc-stim` @ `960f282`).
  - Worktree: `C:/data/t2p_qc/wt/npc_stim_fix`. The `../npc_stim` working tree does **not** contain the fixes.
  - Use it with `PYTHONPATH=C:/data/t2p_qc/wt/npc_stim_fix/src` and `C:/data/t2p_qc/venv_new/Scripts/python`.
  - Not pushed.
- **Display-time outputs** (per session: `<folder>.npz` with `A_block{i}`, plus a `.json` with sources and reasons):
  - baseline: `C:/data/t2p_qc/out/baseline/` (vsync blocks are under `vsync_block{i}`)
  - branch: `C:/data/t2p_qc/out_issues/fixE/`
  - **Don't use `C:/data/t2p_qc/out/new/`**: a concurrent run overwrote it.
- **Local cache** of sync and stim files: `C:/data/t2p_cache/<folder>/`. It does **not** contain Open Ephys / NI-DAQ data. That data is on the share: `//allen/.../PilotEphys/Task 2 pilot/<folder>/` (OpenEphys subfolder `<name>*` or `Record Node*`). Read-only.
- **Prior analog work to reuse:** `experiments/sync-frame-timing/` (see its README; `lost_events.log` D1 compared rebuilt vsyncs with the analog photodiode, reporting 0.16–2.02 ms errors). Reuse its loading and barcode-alignment code.
  - Also `npc_ephys` (barcode alignment; NI-DAQ channels) and `npc_samstim`.
  - **Using AI0 as the photodiode channel is an unconfirmed guess.** Confirm it per rig and date before trusting any result.

## Sample (about 10–20 sessions; pick ones that have NI-DAQ data)

Block lists are in `experiments/display-times-issues/compare_fixE.csv`, `neg_classify.csv`, `flip_count.csv`, `flip_extra.csv` and `misaligned.log`.

| Group | Why | Examples |
|---|---|---|
| §2 shifted 1 frame | is the +1 frame correction right? | `662892_20230821` b2, `703333_20240408` b2; and the two left high: `706401_20240423` b2 (32.8 ms), `703882_20240425` b5 (30.9 ms) |
| §1 box-3 re-placed | is placement using the rising-edge bound right? | `807082_20250930` b3 frame 81436, `813586_20251205` b5 frame 27877, `878903_20261028` b0 |
| §1a late falling edge | is the display really on time when the falling edge is late? | `744279_20250116` b1 frame 1827, `702131_20240226` b3 frame 73754 |
| §1 hard-check rejects | were the rejections justified? | `733891_20240918` b4, `742903_20241024` b4, `714748_20240624` b4 |
| §4 extra flips in stalls | which flip is the real frame? | OptoTagging `699847_20240415` b1; DR1 `716718_20240730` b0 |
| §5 misalignment steps | where do the steps come from? | `733891_20240917` b4, `761583_20241219` b3 |
| §5 recovered | are the recovered blocks right? | `818720_20251219` b3, `626791_20220816` b0 |
| clean controls | baseline accuracy | 3–4 unchanged blocks, box intervals 1 and 3, across eras |

## Method

1. Load the analog diode for each session and map it to sync time through the barcodes (Open Ephys ↔ sync).
2. Detect the analog transitions (box-flip edges) and assign them to frames.
3. Compare them with the baseline and branch display times. Remember that display times include `MONITOR_CENTER_REFRESH_TIME` (8 ms): either subtract it or compare against flip times.
4. For each block, report:
   - the error distribution (median, p1, p99, max)
   - any shifts in landmark (dropped-frame) positions. Shifts that are a multiple of `diodeBoxFrameInterval` are invisible to diode phase, so use landmarks.
5. **Pass criterion:** errors of about 1–2 ms, and zero landmark shift.

## Rules

- Don't modify the branch until the results are in. Read-only on the share.
- Code and logs go to `experiments/display-times-analog-validation/`; the report goes to `outputs/display-times-analog-validation.md`. Include figures, and mark any unchecked items `not run` or `blocked`.
- If NI-DAQ data or the channel mapping can't be established for a session, record it as blocked; don't guess.

## Afterwards: completing the task

1. **Act on the results, in the branch** (`C:/data/t2p_qc/wt/npc_stim_fix`):
   - If a fix fails validation, revert or revise that commit.
   - If the §4 extra flips are confirmed as spurious, implement "keep the first flip in a long vsync interval when the next window has its own flip" and the §5 segment re-alignment, if the analog data shows how to do it.
   - If a constant needs recalibrating (the 15 ms min separation, the 19.55 ms average latency), change it only with the analog evidence cited.
2. **Re-run the population check** after any code change:
   - `experiments/display-times-issues/run_display_times.py <label>`
   - `compare_fix.py <label>`
   - `eval_rise.py <label>`

   Every diff against fixE should be explained. Also run `pytest tests/test_display_times.py` plus the doctests in the worktree.
3. **Check what the move QC didn't cover:**
   - path B (`get_stim_frame_times(..., "display_time")`, the npc_sessions call) and `stim_offsets` on the branch
   - a `Session` end-to-end in npc_sessions for one or two sessions
4. **Update the docs:**
   - append a "Validation" section to `outputs/display-times-issues.md`, with a link to the new report
   - change the move QC's "frames displayed before vsync" metric to the rising-edge metric
   - tick or adjust P4 in the plan
5. **Release order** (the user does this or approves it): npc_sync 0.3.0 → npc_stim move branch → `display-times-p2-fixes` (rebase onto the released move and push for review) → update the npc_sessions and npc_sessions_cache locks. Then regenerate NWB and cache for sessions with changed blocks (the list is in `compare_<final label>.csv`).
6. **Leftovers for P2:**
   - make the stim-file matcher exact and one-to-one (§6)
   - clear npc_sync errors for files with no vsync edges and for unbalanced stim_running (§7)
   - a replacement file for the xfail test (§8)
   - remove end-of-block truncation and invention now that the hard check exists
7. **Clean up** when no longer needed: `C:/data/t2p_cache` (59 GB), `C:/data/t2p_qc/out_issues`, and the worktree (`git -C ../npc_stim worktree remove C:/data/t2p_qc/wt/npc_stim_fix` once the branch is pushed).
