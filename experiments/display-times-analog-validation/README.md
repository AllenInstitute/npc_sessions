# display-times-analog-validation (2026-11-16)

Targeted P4: display times (baseline vs npc_stim `display-times-p2-fixes`) against the NI-DAQ analog
photodiode. Report: `../../outputs/display-times-analog-validation.md`. Handoff: `../../notes/p4-analog-validation-handoff.md`.
Read-only on the share, S3 and the cache. Intermediate data: `C:/data/t2p_qc/analog/<folder>/`
(`sync.npz`, `analog.json`, `barcodes.npz`, `traces.json`, `trace_b*.npy` (AI0 at 5 kHz, x4 int32), `analog.npz`).

Environments: `C:/data/t2p_qc/venv_new` (npc_sync dev, for `prep_sync.py`) and `C:/data/t2p_qc/venv_analog`
(created here: numpy, zarr<3, numcodecs, wavpack-numcodecs, s3fs, boto3, h5py, scipy, pandas, matplotlib;
needs AWS credentials). The npc_sessions `.venv` with `PYTHONPATH=<wt npc_stim_fix>/src;<npc_sync>/src` for `session_e2e.py`.

| File | What |
|---|---|
| `locate.py` | find a session's ecephys upload on S3 (3 buckets, by sync start time) and its NI-DAQ zarr / clipped files |
| `probe_s3.py`, `probe_scan.py` | first format / throughput / channel probes (S3 zarr, 30x real time) |
| `survey_channels.py` → `survey_channels.jsonl/.log`, `survey_summary.log` | per session: AI0..AI7 band fractions (10/30 Hz) and range in 4 x 2 s windows; 87 sessions (`survey_sessions_all.txt`) |
| `prep_sync.py` → `prep_sync.log` | sync side: barcodes, vsync and TTL diode edges, blocks -> `sync.npz` |
| `fetch_analog.py` → `logs_fetch_*.log`, `channel_confirmation.log` | barcode map, channel confirmation (all 8 channels vs TTL edges, per block), 5 kHz AI0 traces; `--confirm-only` re-runs the confirmation with the current detector (`channel_confirmation.log` is the final one; `logs_fetch_*` transition counts are from earlier detector versions) |
| `detect.py` | Schmitt-trigger transition detection on local levels -> `analog.npz` |
| `compare.py` → `results/<folder>.csv`, `results/<folder>_frames.npz`, `compare_all.log` | DP assignment of analog transitions to toggle frames (vsync + polarity only), errors of baseline and branch, misassignment test, latency calibration |
| `run_all.sh` | detect + compare for the 11 sessions |
| `summarize.py` → `summary.log`, `results_blocks.csv`, `results_changed_frames.csv`, `results_late_fall.csv`, `figs/err_*.png`, `figs/latency_calibration.png` | pooled results and group checks |
| `fig_examples.py` → `figs/ex_*.png`, `figs/channel_survey.png` | report figures (reads S3) |
| `raw_window.py` | plot one frame window (analog, vsync, TTL, both display times); `figs/scratch/` are the diagnostic plots made with it |
| `rule_sim.py` → `rule_sim.log` | offline test of the §4 "keep the first flip in a stall" rule on TTL edges, scored against the analog |
| `compare_pathB.py` → `compare_pathB.log/.csv` | path B / A2 / D on the branch (`qc_run.py fixE_qc`, log `qc_run_fixE_qc.log`) vs baseline and vs path A |
| `session_e2e.py` → `session_e2e.log` | npc_sessions `DynamicRoutingSession` end to end on 2 sessions with the branch |
| `pytest_cfae3e3.log` | branch tests + doctests at HEAD `cfae3e3` |
| `npc_ephys_barcodes.py` | copy of `../../../npc_sync/experiments/vsync-frame-count/npc_ephys_barcodes.py` |

Detector history (why counts in early logs differ): v1 = Schmitt 0.6/0.3 on 2.5 s levels (fell for the
white-phase overshoot step); v2 = 0.5/0.15/0.25 on 2.5 s levels (missed low-amplitude OptoTagging stretches);
v3 = 80 ms local levels + guards (final). Only v3 results are reported.
