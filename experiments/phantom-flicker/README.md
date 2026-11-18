# phantom-flicker (2026-11-17)

Evidence for `notes/sync-patching-handoff-2.md` ("Critical before proceeding - phantom flicker").
Report: `../../outputs/phantom-flicker-report.md`. Read-only on the cache, the share and S3.

Code under test (local branches `sync-patch`): npc_sync (`align_diode_toggles`,
`predict_box1_diode_edges`, `SyncPatch.removed` / `diode_blocks`, `build_sync_patch`), npc_stim
(`get_display_times` consumes a cleaned line), npc_sessions (`timing.get_sync_patch` for every
session). Interpreter: `../../.venv/Scripts/python.exe` with
`PYTHONPATH="../../../npc_sync/src;../../../npc_stim/src;../../src;."`.

| Script | Output | What |
|---|---|---|
| `common.py` | – | cache paths, exact stim matching (from `../sync-patch-manifest/common.py`), block data |
| `explore_block.py`, `explore_windows.py`, `explore_long.py` | (stdout) | first look at diode edges around held frames (fixed windows: superseded by the alignment) |
| `phantom.py`, `try_align.py` | (stdout) | prototype of the box-1 alignment (ported to `npc_sync.patch.align_diode_toggles`) |
| `survey.py` → `summarize_survey.py` | `survey_blocks.csv`, `survey_holds.csv.gz` (gzip; 52 MB as CSV), `survey_run.log`, `survey_summary.log` | prototype alignment on every cached box-1 block (1,990 blocks with a stim file, 442 sessions): edge classes per block, per held frame |
| `crosscheck_npc_stim.py` | `crosscheck_npc_stim.csv/.log` | frame <-> edge assignment vs npc_stim's on 1,029 accepted blocks (34.0 M frames) |
| `diag_polarity.py` | `diag_polarity.csv/.log` | cause of the 21 box-1 "wrong polarity after the span" failures (old `predict_diode_edges`) |
| `run_population.py` → `compare_population.py` | `C:/data/t2p_qc/out_phantom/`, `run_population.log`, `compare_population.csv/.log`, `compare_population_sessions.csv` | npc_stim display times with/without the npc_sessions-built patch, 441 sessions (branch code), vs the released behaviour (`C:/data/t2p_qc/out_issues/fixE`) |
| `diode_gate_recheck.py` | `diode_gate_recheck.csv/.log` | photodiode gate on the 155 replay spans of the P0 manifest with the new builder (old: `../sync-patch-manifest/diode_gate_check.csv`) |
| `section4_accounting.py` | `section4_accounting.log` | every extra flip of the 24 §4 blocks (`../display-times-issues/flip_extra.csv`) |
| `analog_check_762526.py` | `analog_check_762526.log` | NI-DAQ AI0 check of the removed edges and recovered display times, DRpilot_762526_20250320 block 1 |
| `npc_sessions_tests.log` | – | npc_sessions test suite with the branch libraries: 23 passed |
| `fig_phantom.py` | `figs/phantom_examples.png`, `figs/phantom_population.png`, `figs/box1_gate.png` | report figures |
| `_edit_builder.py`, `_builder_*.txt` | – | one-off edit of npc_sync's builder (provenance only) |

`run_population.log` is the last re-run only: 4 sessions, after making the builders tolerant of sync-level failures and fixing npc_stim's rising/falling balance assert. The full run (441 sessions) was stopped after 2 h, waiting on the one session below, before its in-order log was written. Its per-session results are the JSON/NPZ files in `C:/data/t2p_qc/out_phantom`, and `compare_population.py` reads those. Sessions not affected by either fix were not re-run: the first fix only changes sessions whose vsync blocks can't be computed, and the second only files with 'unknown' photodiode spans.

Not run: `DRpilot_366122_20250109` (dev mouse, diode box interval 2): npc_stim's O(n²) flip
removal (restored in `cfae3e3`) takes > 2 h, with or without a patch; phantom-flicker handling
doesn't apply to box interval 2.
