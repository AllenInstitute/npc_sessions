# display-times-issues (2026-11-15)

Follow-up to `../display-times-move/`: causes and fixes for the pre-existing issues listed in
`outputs/display-times-move-qc.md`. Report: `../../outputs/display-times-issues.md`.
All scripts are read-only on the data (local cache `C:/data/t2p_cache`).

Fix branch: npc_stim `display-times-p2-fixes` (worktree `C:/data/t2p_qc/wt/npc_stim_fix`, based on
`move-display-times-to-npc-stim` @ `960f282`). Run anything below that imports the fixed code with
`PYTHONPATH=C:/data/t2p_qc/wt/npc_stim_fix/src` and `C:/data/t2p_qc/venv_new/Scripts/python`.

| File | What |
|---|---|
| `common.py` | cache paths, sync line access, `plot_window` (vsync / diode / display-time figure) |
| `neg_latency.py` → `neg_latency.csv/.log` | issue 1: every frame with display < vsync falling edge (baseline) |
| `vsync_rise_check.py` → `vsync_rise_check.csv/.log` | issue 1: same blocks, against the vsync *rising* edge |
| `neg_classify.py` → `neg_classify.csv/.log` | issue 1: consistent (delayed falling edge) vs impossible frames |
| `fig_neg_latency.py`, `fig_733891.py` | figs `neg_latency_examples.png`, `neg_latency_733891.png` |
| `fig_latency_outliers.py` | issue 2: `latency_outliers.png` |
| `spontaneous_vote_trial.log` | issue 2: rejected alternative fix (transcribed) |
| `equiv_anomalous.py` → `equiv_anomalous.csv/.log` | issue 3: new flip-removal == old loop (fuzz + all blocks) |
| `fig_box2.py` | issue 3: `box2_intervals.png` |
| `flip_count.py` → `flip_count.csv/.log` | issue 4: per-frame diode flip counts in rejected blocks; `flip_count_cumulative.png` |
| `flip_extra.py`, `pingpong.py` → `.csv/.log` | issue 4: where extra flips are; `pingpong_examples.png` |
| `misaligned.py` → `misaligned.log` | issue 5: alignment vote in rejected blocks; `misaligned.png` |
| `unmatched.py` → `unmatched.log`, `fig_unmatched.py` | issues 6/7: blocks vs stim files; `unmatched_timelines.png` |
| `run_display_times.py` | run `get_display_times` on all cached sessions → `C:/data/t2p_qc/out_issues/<label>/` |
| `compare_fix.py` → `compare_<label>.csv/.log` | per-block comparison with baseline (`C:/data/t2p_qc/out/baseline`) |
| `eval_rise.py` → `eval_rise_<label>.csv`, `eval_rise.log` | rising-edge lower-bound check on any run |
| `fig_fixes.py` | `fix_box3_placement.png`, `fix_latency_before_after.png` |

Run labels (each = the branch at a commit, all 442 cached sessions with sync files):
`fixB` = `a4160d8`, `fixC` = `6b90a52`, `fixD` = `42f39a3`, `fixE` = `64eccca` (final code; the
later commit `1431002` only edits a docstring). `run_fix*.log` are the run logs.

`cfae3e3` reverts the linear-time flip removal. Scripts that import `remove_anomalous_diode_flips` need `1431002` checked out. `deleted_sync_file.sha256`: local cache copy of `DRpilot_366122_20250109/20250109T153706.h5`, deleted 2026-11-15.
