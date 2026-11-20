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
- [ ] T2 independent validation of the model: barcodes after remap; licks vs stim lickFrames; vsync gap residuals
- [ ] T3 survey: all cached files with replays/discontinuities + clean sample (false positives, s < 0, uncovered runs)
- [ ] T4 npc_sync: `timebase.py` (find jumps, Timebase remap), SyncDataset applies it, SyncPatch carries it, builder uses remapped view; tests
- [ ] T5 npc_sessions: builder on remapped view, BUILDER bump, notes/tags; npc_stim/npc_mvr/npc_ephys audit
- [ ] T6 end-to-end on problem sessions (715709_2024-08-08, 840160_2026-04-28, 703880_2024-04-18, 708020_2024-05-17, 714748_2024-06-28): barcodes/ephys residuals, display times, licks
- [ ] T7 release: push npc_sync -> CI publish -> floors (npc_stim?, npc_sessions) -> push
- [ ] T8 S3 patch cache for problem sessions; validate from cache
- [ ] T9 report outputs/sync-dropped-chunks-report.md; update notes checklist

## Verification log
(see experiments/sync-timebase-drops/README.md for per-script results)
