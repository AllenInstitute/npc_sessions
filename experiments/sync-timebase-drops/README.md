# sync-timebase-drops (2026-11-19)

Question: can sync timebase discontinuities (dropped acquisition buffers) be located precisely
and patched like replays? Task: `notes/sync-dropped-chunks-task.md`. Report:
`outputs/sync-dropped-chunks-report.md`. All scripts are read-only on the share and S3.

Code under test: npc_sync branch `timebase` (local), npc_sessions working tree. Run with the
npc_sessions venv and `PYTHONPATH="../../../npc_sync/src;../../src"` unless noted.

## Result in one paragraph

Drops and replays are one mechanism. The sync program drops 0.5 s buffers without leaving a
gap in sample numbers, and later writes copies of a buffer to make up the count. Recorded
buffer p holds true buffer p + s(p): s rises by k at a drop of k buffers and falls by 1 at
each copy. Between a drop and the run of copies that makes up for it (0-20 s), every line is
recorded early by s x 0.5 s. A buffer is 30 camera periods minus 12-26 us, so the camera phase
of each buffer reads s directly; the jump at every buffer boundary is an integer.

## Scripts and evidence

| Script | Output | What |
|---|---|---|
| `step_fit_explore.py` | `step_fit_explore.log` | Known-size step fit per camera line at every 0.1 s boundary of the barcode window, 5 drops in the 4 files without replays: 26/26 lines pick the same boundary, all on the 0.5 s grid; fitted step within 5 us of the prediction for exactly k x 50,000 samples |
| `pair_explore.py`, `chunk_identity.py`, `pair_step_fit.py` | `.log` | Complementary barcode pairs in replay files: the "early" barcode is real data (not a copy), recorded 1.0-1.5 s early, 2.5-3.5 s before a replayed run |
| `stretch_fit.py` | `stretch_fit_703880_0418.log` | One drop position per replayed run (first version of the displaced-stretch model): about half the runs of 703880_20240418 have the drop 0.5-8 s before the run |
| `shift_state.py` | `shift_state_703880_0418.log`, `shift_state_others.log` | Prototype of `npc_sync.timebase.find_chunk_jumps`: jump at every buffer boundary. 703880_20240418: 58 jumps of +1, 58 of -1.00 (exactly the 58 copies), nothing else in 15,437 boundaries. Pure drops: +4, +1, +3, +3, +2 at single boundaries |
| `survey.py`, `survey_summary.py` | `survey.jsonl`, `survey.log`, `survey_summary.log` (`*_v1.*`: first run, 0.1 ms interval tolerance) | `SyncDataset.find_timebase()` on all 156 manifest files with replays or discontinuities + 100 clean files |
| `lick_validation.py` | `lick_validation.log` | Licks recorded in displaced data vs the stim file's `lickFrames` (stim PC clock) |
| `e2e_check.py`, `e2e_summary.py` | `e2e_new.jsonl`, `e2e_old.jsonl`, `e2e_*_run.log` | npc_sessions end to end on 5 ephys sessions, new code vs `HEAD` + released packages |
| `patch_debug.py` | (stdout) | Patch for one session without the NI-DAQ sound step: span status per line |
| `locate_drops.py` + inline | `locate_drops.log`, `exact_slip_check.log` | Earlier session (same day): single-interval slip, vsync cost (see "Earlier results") |

### Survey (`survey_summary.log`)

- **100 clean files:** 818,155 buffer boundaries observed, 0 jumps. Largest |jump - integer| 0.118.
- **156 files with replays or barcode discontinuities:**
  - 1,475 replayed copies; 1,487 dropped buffers located on the cameras; 1 assumed at the run
    start (759435_20250508: a 30,000-sample "replay" 1.5 s into the file, before the cameras
    start). 1,487 = 1,475 + 13 (net shift of the 4 pure-drop files) - 1: every copy is
    accounted for by a located drop, and there are no other jumps.
  - Located drops: 1,040 single buffers; 2-19 buffers at the start of longer runs.
  - Lag from a drop to the start of the next run: 0 for 34% of drops, median 2.0 s for the
    rest, max 20.5 s. 16 of 152 replay files have at least one lag > 0; in the other 136 every
    drop is at the start of its run (the old "copies overwrite data" model was right there).
  - Recorded time between a drop and its run, over all files: 2,361 s (upper bound on the
    data that was mis-timed by 0.5-6.5 s before this change).
  - Barcode onset discontinuities: 37 intervals in 11 files on the recorded clock, **0 on the
    timebase**. Barcodes are not used to build the timebase.
- First run (`survey_summary_v1.log`): the 0.1 ms interval tolerance rejected the phase step
  of drops of 7-19 buffers (81 buffers at 7 runs assumed). Fixed by allowing any step below a
  quarter period.

### Licks (`lick_validation.log`)

3,901 licks recorded in displaced data in 11 sessions: 6% are within 25 ms of a stim-file lick
at their recorded time, 99% at their true time (on-time licks: 99.8%). Not usable:
832032_20260223 (lick line noise, 202,800 edges), 717473_20240916 (drop before the task).

### End to end (`e2e_summary.py new old`)

See the report. 840160_2026-04-28 and 715709_2024-08-08 were refused before (no frame times;
715709: 6/6 ephys devices failed timing QC on a 1.97 s step). Now: task blocks patched, ephys
devices pass with leave-one-out barcode residuals <= 0.4 ms.

## Earlier results (first session of the day, superseded where they differ)


### Q5 / Q1: drop sizes (all 11 files, 38 discontinuities)
- **Files with no replays (4 files, 5 drops):** every shift is −k × 0.5 s, with k = 4, 1, 3, 3, 2. The largest difference from a whole number of buffers is 0.14 ms, which is within barcode jitter.
- **Files with replays (7 files):** discontinuities mostly come in −/+ pairs that cancel, eg −1.0 then +1.0. Several pairs are not whole buffers (±1.7035, ±1.8049, +1.3301/+0.1696, ±0.9958). These look like barcode misdetection near replays (a barcode moved, so net shift 0), not real drops. **Not investigated further.** Those files should keep being rejected until someone looks at them.

### Q3: cameras locate the drop independently
The cameras record at 60.00x Hz. A whole-buffer drop isn't a whole number of camera periods, so it shows up as a phase slip of a few tens of µs in exactly one inter-frame interval.

| session | shift (s) | camera consensus | lines agreeing (top anomaly) | on 0.5 s grid | vsync locate (w=600) |
|---|---|---|---|---|---|
| 715709_20240808 | −2.0 | 258.5 | 6/6 | yes | skipped (block holds both drops) |
| 715709_20240808 | −0.5 | 2078.0 | 3/6 | yes | skipped |
| 717473_20240916 | −1.5 | 782.5 | 4/6 | yes | 782.5 (agree) |
| 832032_20260223 | −1.5 | 3598.0 | 6/6 | yes | 3598.0 (agree) |
| 840160_20260428 | −1.0 | 2472.0 | 6/7 | yes | 2472.0 (agree) |

The −0.5 s drop gives the smallest slip (about 0.026 ms, close to the 10 µs sample size), so it's the weakest result. Only the beh/face lines see it. The 717473 lines that miss it have unrelated larger anomalies (beh readout ±2.5 ms at 814.4 s).

`exact_slip_check.log`: the observed slip matches the prediction for an **exactly** k × 50,000-sample drop. 24 of 25 lines agree within ±17 µs (20 of 25 within ±10 µs). Each interval spans two edges quantized to 10 µs, so up to about 20 µs is expected. The one outlier is +40 µs (832032 nose readout). This is independent evidence for the size: the barcode shift alone (eg −0.99998 s) predicts noticeably worse (`expected slip` column in `locate_drops.log`, eg 840160 nose: −0.0068 predicted vs −0.0262 ms observed).

### Q2: the vsync/frameIntervals cost (`locate_drop`) is weak alone
With w=600 the cost is nearly flat across neighbouring boundaries: runner-up/best ratio is 1.03 for both 717473 and 840160, and 1.34 for 832032. With w=30 it picks off-grid boundaries in 2 of 3 files. Its earlier 2472.0 result for 840160 is correct (the cameras agree), but it could not be told apart from 2471.8 using vsync alone. It also assumes one drop per block, and 715709 breaks that.

### Q4: raw events at the located boundary
All boundaries land on multiples of 50,000 samples, which is the 0.5 s grid. The 715709 boundaries were located by the cameras without assuming any grid, so this is real evidence there. For the vsync picks it's partly by construction. There's no event gap in sample numbers beyond the local maximum, and no line has same-polarity edges on either side of the boundary.

Limitation: a line that toggles an even number of times during the missing second passes this check, so it can't detect that.

## Implications for a drop patch (as written before the displaced-stretch finding)
1. Use the **cameras** as the main locator. Take each camera line's largest modulo-period residual in the barcode window, snap it to the 0.5 s grid, and require most lines to agree. Check that the slip matches the exact k × 0.5 s prediction within about 20 µs. This works between stim blocks too.
2. Use vsync + frameIntervals as a **check**, not the locator. Handle more than one drop per block by fitting all located drops together.
3. Then: add k × 0.5 s to every edge at or after the boundary. Insert `[t, t + k·0.5)` as an unknown interval on all lines. Fill it with the existing replay reconstruction (vsync from frameIntervals, photodiode, licks, laser marked unknown).
4. Keep rejecting files with replays plus paired/non-quantized discontinuities until the barcode-pair behaviour is understood.

## Not done
- Ephys barcode alignment after the shift. This is the decisive end-to-end check.
- Patch implementation.
- Whether 10,000-sample buffers ever occur in drops. All 5 drops sit on the 50,000-sample grid.
