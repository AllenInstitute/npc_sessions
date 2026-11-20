# Task: patch dropped sync chunks ("discontinuities")

Original request (2026-11-19), verbatim:

> we recently implemented patches for replay chunks on sync - let's try to fix the "discontinuities" if we can. sync drops data at chunk boundaries, and in multiples of chunks - we can probably triangulate them more precisely than we're currently doing by looking at camera exposing line intervals and detecting discrepencies (perhaps vsyncs too for redundancy). they should explain short barcode intervals almost exactly (barcode intervals have a small amount of variability). when confident in their location and length, we can push all events after the drop forwards in time, and use the exact same principles used for patching replay chunks to reconstruct the dropped data, and npc_sync can apply them in a similar manner when the cache is populated. I think we'd have to apply replay patches first, then dropped patches. Be aware that complementary consecutive long/short or short/long barcode pairs are not discontinuities and aren't an issue - they're caused by replay intervals and the timing of barcodes immediately before/after should be correct. multiple replays or discontinuities per block also aren't necessarily an issue, we may be able to explain everything now. investigate, implement fixes in relevant npc_ repos, push to trigger ci/cd releases, raise floors in npc_session, create cache for problematic sessions and validate.

## Checklist (derived) - closed 2026-11-20

Report: `outputs/sync-dropped-chunks-report.md`. Evidence: `experiments/sync-timebase-drops/`.

- [x] Investigate: drops located on camera exposing lines (phase per 0.5 s buffer). They explain every short barcode interval: 37 irregular intervals in 11 files, 0 on the corrected clock. Vsyncs are used as a check (per-gap frame solve), not to locate
- [x] Complementary barcode pairs: not excluded but explained. The middle barcode is recorded in data displaced between a drop and the replayed run that makes up for it; that data is now moved to its true time
- [x] Multiple replays/drops per block: handled (up to 230 spans in one block)
- [x] npc_sync: `npc_sync.timebase`; events after a drop are pushed forwards; dropped intervals reconstructed with the replay machinery
- [x] Apply order: one step. Replays and drops are one mechanism, so the timebase is applied first and the reconstruction runs once on the corrected clock
- [x] npc_sync applies the timebase and the spans from the cached patch (`SyncDataset(path, patch=...)`)
- [x] Released: npc_sync 0.3.8, npc_sessions 0.0.294 (CI green)
- [x] Floor raised in npc_sessions: `npc-sync>=0.3.8`. (The request said "npc_session"; I took that to mean npc_sessions. npc_stim, npc_ephys, npc_mvr and npc_samstim needed no change.)
- [x] Patch cache written for the 5 npc_lims sessions with displaced data or a net shift, reloaded and validated (`cache_reload_check.md`, `e2e_release_summary.md`)

## Open

- 714748_2024-06-28 task block: rejected by the box-1 photodiode alignment (4,072 unexplained diode edges from 4129.37 s, a 22 ms-late frame-signal edge; not near a dropped interval)
- 703880_20240412 task block (share only): first frames of the block fall in a dropped interval, so the frame count is off by 16 and the block is rejected
- Vsync rising edges don't alternate at a reconstructed span (one RR and one FF pair per span); predates this work
- Files with fewer than two cameras can't have drops located
- npc_sessions_cache: not touched. NWB/cache regeneration for the affected sessions not done
- The 15 share-only files with displaced data were surveyed but not run end to end
- The local npc_sessions venv was synced to `uv.lock` during the release (polars 0.20.28 -> 1.39.3, ruff)
