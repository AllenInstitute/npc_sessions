# display-times-move QC (2026-11-13)

Before/after QC for moving frame display-time estimation from npc_sync to npc_stim.
Report: `../../outputs/display-times-move-qc.md`. All scripts are read-only on the share.

| File | What |
|---|---|
| `list_ephys_sessions.py` → `ephys_sessions.csv` | the 444 Task 2 pilot ephys sessions |
| `cache_files.py` → `cache_files.log` | copy sync + stim files to `C:/data/t2p_cache` |
| `qc_run.py` → `qc_run_{baseline,new}.log`, `C:/data/t2p_qc/out/` | run code paths A/A2/B/C/D per session in one env |
| `qc_compare.py` → `qc_sessions.csv`, `qc_blocks.csv`, `qc_summary.json`, `qc_compare.log` | byte-level comparison + sanity metrics |
| `qc_report.py` → `qc_tables.md` | generated tables used by the report |
| `diode_box_key_dates.py` → `diode_box_key_dates.log` | when stim files started having `diodeBoxFrameInterval` |
| `negative_latency_positions.py` → `negative_latency_positions.log` | where display-before-vsync frames are in their blocks |
| `latency_outliers.log` | blocks with median latency > 35 ms (one-liner in report session) |
| `plot_smoke.py` → `plots/` | smoke test of moved plots + deprecated shims |
| `qc_run_new_superseded.log` | first "new" run, before the diodeBoxFrameInterval cutoff fix (outputs overwritten) |
