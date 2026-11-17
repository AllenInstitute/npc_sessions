# sync-frame-timing experiments (2026-11-12)

Evidence for `outputs/sync-frame-timing-upgrade-plan.md`. All scripts are read-only on data.

Data: local copies of 7 Task 2 pilot sessions (sync `.h5` + stim `.hdf5`) in
`../../../npc_sync/experiments/vsync-frame-count/data/`. `multi_gap_check.py` and
`displaced_barcode_check.py` also read directly from
`//allen/programs/mindscope/workgroups/dynamicrouting/PilotEphys/Task 2 Pilot`.
Interpreter: the npc_sync dev venv (`../../../npc_sync/.venv/Scripts/python.exe`; editable npc_sync
@ `8974ba0` = 0.2.8 source, npc_stim 0.1.23). `survey_summary.py` uses this repo's venv.

| Script | Log | Question |
|---|---|---|
| `survey_summary.py` | `survey_summary.log` | How common and how long are replay windows in the 2,916-file survey (`npc_sync/sync_replay_issues.csv`; consecutive-chunk detector, pre-0.2.8)? |
| `line_audit.py` | `line_audit.log` | Which labelled lines have edges inside `invalid_intervals`, and which code paths does the 0.2.7 fail-fast guard? |
| `all_bits_audit.py` | `all_bits_audit.log` | Same question for all 32 bits, including unlabelled bit 1 (sound); also sync barcode durations |
| `lost_events.py` | `lost_events.log` | Which real stim-file events (licks, rewards, trial onsets, opto) fall inside intervals? Does barcode decoding change if those intervals are masked? |
| `camera_gap_check.py` | `camera_gap_check.log` | First pass at interpolating camera exposures: period taken from the median interval. Too coarse, because intervals are quantised to 10 us |
| `camera_gap_check2.py` | `camera_gap_check2.log` | Can camera exposures across intervals be filled by integer-period interpolation, with a least-squares period? |
| `event_latency_check.py` | `event_latency_check.log` | Outside intervals, how do reward, sound and lick times relate to frame times? |
| `lick_count_check.py` | `lick_count_check.log` | Do sync lick onsets match stim `lickFrames` when both are restricted to the task block? |
| `barcode_interval_jitter.py` | `barcode_interval_jitter.log` | How regular are sync barcode onset intervals? This is the basis for a sync-only timebase check |
| `multi_gap_check.py` | `multi_gap_check.log` | For files with many runs: barcode-onset anomalies, and task-block reconciliation with each gap's frame count solved from elapsed time |
| `displaced_barcode_check.py` | `displaced_barcode_check.log` | In multi-run files, are barcode onsets *outside* `invalid_intervals` displaced by k × 0.5 s? |

Reused prototype: `npc_sync/experiments/vsync-frame-count/repair.py::reconcile_replay`. It is only
valid for blocks with a single gap, because it assigns every missing frame to the first gap.
`multi_gap_check.py` solves each gap separately.
