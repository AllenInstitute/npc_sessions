# Plan: patch dropped sync chunks (task: notes/sync-dropped-chunks-task.md)

Started 2026-11-19. Evidence dir: experiments/sync-timebase-drops/.

## Key finding (changes the design)
Camera phase per 0.5 s chunk (shift_state.py) gives an integer jump J at every chunk boundary:
- pure drops: one +k jump (715709: +4 @258.5, +1 @2078.0; 717473: +3 @782.5; 832032: +3 @3598.0; 840160: +2 @2472.0), equal to the barcode shift.
- replay files: every copy boundary is -1.00 (58/58 in 703880_20240418), and the matching +1 drops sit 0-8 s BEFORE the run. Data between a drop and its run is recorded early by k chunks ("displaced stretch"). This is what the complementary barcode pairs are (barcode inside a stretch).
- no other nonzero jumps in ~15,000 boundaries per file (max |d - round(d)| 0.18).
Model: recorded chunk p holds true chunk p + s(p), s = cumsum(J). Replays and drops are one mechanism.

## Task ledger
- [x] T1 investigate locator (step_fit_explore, pair_explore, chunk_identity, stretch_fit, shift_state)
- [x] T2 independent validation: barcodes 37 -> 0 irregular intervals; licks 6% -> 99% within 25 ms; vsync gap residual median 0.385 -> 0.024 ms
- [x] T3 survey: 156 faulty + 100 clean files (survey_summary.log): 0 false jumps, 1,487 buffers located, 1 assumed
- [x] T4 npc_sync 0.3.8: timebase.py, SyncDataset(timebase=), SyncPatch format 3, builder on the timebase, tests/test_timebase.py (10 tests)
- [x] T5 npc_sessions 0.0.294: timing.true_clock_view, builder 5, floor npc-sync>=0.3.8, test_dropped_buffers_are_located_and_reconstructed
- [x] T6 end to end on 5 sessions (e2e_release.json): 4 fully usable, 714748 task block rejected by the photodiode check
- [x] T7 released: npc_sync CI run 44695310796 (72 passed x 3 OS), npc_sessions CI run 44695811631 (65 passed, mypy clean)
- [x] T8 S3 patch cache for the 5 sessions; reloaded in a new process (cache_reload_check.md)
- [x] T9 report outputs/sync-dropped-chunks-report.md; notes checklist closed

## Verification log
- npc_sync local: 83 passed, 4 failed (S3 doctests/tests that fail the same way on `main` in this checkout: its `.env` credentials are refused). CI is the reference: 72 passed, 16 skipped.
- npc_sessions local: tests/ 38 passed (test_aind_metadata not collected: optional deps missing locally); tests/test_timing.py 17 passed with released npc_sync 0.3.8. CI: 65 passed, 2 skipped.
- Stalled runs: the first e2e pass and the old-code baseline both stopped making progress (no CPU) after ~40 min; killed and rerun (new) or replaced by old_refusal_check.log (old).
- Not verified: analog photodiode inside dropped intervals; the 15 share-only files end to end.
