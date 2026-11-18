# sync-patch-manifest (2026-11-17)

Evidence for `notes/sync-patching-handoff.md` (P0 manifest, re-derived gates, P1–P3 acceptance,
step 6 validation). Report: `../../outputs/sync-patching-report.md`. All scripts are read-only on
the share and S3.

Code under test (local branches, **not pushed**): npc_sync `sync-patch`, npc_stim `sync-patch`,
npc_sessions `sync-patch`. Interpreters:
- `../../.venv/Scripts/python.exe` (npc_sessions venv; S3 works) with
  `PYTHONPATH="../../../npc_sync/src;../../../npc_stim/src;../../src"` (or just npc_sync/src)
- `../../../npc_sync/.venv/Scripts/python.exe` (npc_sync dev venv, editable npc_sync; no S3)
- `C:/data/t2p_qc/venv_analog/Scripts/python` (s3fs + zarr + wavpack; analog stage 2 only)

| Script | Output | What |
|---|---|---|
| `common.py` | – | paths; exact one-to-one stim↔block matching by start time only (count not used) |
| `list_sessions.py` | `sessions.csv`, `list_sessions.log` | every sync file: Task 2 Pilot share (2,945 files) + npc_lims-tracked sessions not on the share (26 with raw data on S3) |
| `manifest.py` | `manifest_files.jsonl`, `manifest_blocks.jsonl`, `intervals.jsonl`, `manifest_run.log` | P0 survey: per file invalid intervals + timebase discontinuities; per block strict count, per-gap solve, lost stim events, class; per interval camera/diode diagnostics |
| `summarize.py` | `manifest.csv`, `gap_residuals.csv`, `gates.json`, `summary.log` | reclassification (truncated; stim files of other sync files dropped), class tables, gate distributions, proposed vsync gate |
| `diode_gate_check.py` | `diode_gate_check.csv/.log` | photodiode prediction gate, 3 s vs 0.5 s post-span check, on all replay-reconcilable blocks |
| `ephys_barcode_residuals.py` | `ephys_barcode_residuals.jsonl`, `_run.log` | barcode residuals, every device of 446 npc_lims ephys sessions |
| `ephys_barcode_summary.py`, `ephys_barcode_robust.py` | `ephys_barcode_summary.log`, `ephys_barcode_robust.log/.csv` | residual distributions; robust all-barcode fit; current first/last alignment error; proposed ephys gate |
| `camera_gate_check.py` | `camera_gate_check.log/.csv` | camera gate re-check with the startup PWM section excluded from fits |
| `acceptance_local7.py` | `acceptance_local7.log` | P1 acceptance on the 7 local sessions (A1–A8) vs the prototype |
| `p2_display_times_local.py` | `p2_display_times_local.log` | npc_stim display times with a patch (B1–B5) |
| `p3_session_check.py` | `p3_session_check.log`, `p3_session_check_events.log`, `p3_session_check_final.log` (all 5 sessions, final code) | npc_sessions end to end on 5 real replay sessions (C1–C7) |
| `analog_windows_prep.py` → `analog_windows_compare.py` | `analog_windows_prep.log`, `analog_windows_compare.log/.json` | step 6: patched photodiode edges and display times inside every replay window vs NI-DAQ AI0, npc_lims ephys sessions from 2025-01-13 with reconcilable blocks (34) |
| `p4_replay_trials_check.py` | `p4_replay_trials_check.log` | 2026-11-18 close-out: replayed intervals not in `invalid_times`, trials kept (`timing_source='patched'`), `response_time`/sound/opto NaN where their line is unknown, patch cache with `dependency_versions` (D1–D7, 5 replay sessions) |
| `npc_sessions_tests.log` | – | npc_sessions test suite with the branch libraries |
| `debug_diode.py`, `debug_diode2.py`, `debug_diode3.py` | (stdout) | diagnosis of the polarity-dependent diode latency, the splice event at sample start−1, and post-span outliers |

Library unit tests: npc_sync `tests/test_patch.py` (12), npc_stim `tests/test_patched_display_times.py`
(3), npc_sessions `tests/test_timing.py` (4).
