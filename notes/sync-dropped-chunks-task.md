# Task: patch dropped sync chunks ("discontinuities")

Original request (2026-11-19), verbatim:

> we recently implemented patches for replay chunks on sync - let's try to fix the "discontinuities" if we can. sync drops data at chunk boundaries, and in multiples of chunks - we can probably triangulate them more precisely than we're currently doing by looking at camera exposing line intervals and detecting discrepencies (perhaps vsyncs too for redundancy). they should explain short barcode intervals almost exactly (barcode intervals have a small amount of variability). when confident in their location and length, we can push all events after the drop forwards in time, and use the exact same principles used for patching replay chunks to reconstruct the dropped data, and npc_sync can apply them in a similar manner when the cache is populated. I think we'd have to apply replay patches first, then dropped patches. Be aware that complementary consecutive long/short or short/long barcode pairs are not discontinuities and aren't an issue - they're caused by replay intervals and the timing of barcodes immediately before/after should be correct. multiple replays or discontinuities per block also aren't necessarily an issue, we may be able to explain everything now. investigate, implement fixes in relevant npc_ repos, push to trigger ci/cd releases, raise floors in npc_session, create cache for problematic sessions and validate.

## Checklist (derived)

- [ ] Investigate: locate drops via camera exposing-line interval discrepancies (vsyncs as redundancy); check they explain short barcode intervals almost exactly
- [ ] Exclude complementary consecutive long/short or short/long barcode pairs (replay artefacts, not drops)
- [ ] Handle multiple replays/drops per block
- [ ] Implement drop patches in npc_sync: shift all events after a drop forward by the drop length; reconstruct dropped data using the replay-patch principles
- [ ] Apply order: replay patches first, then drop patches
- [ ] npc_sync applies drop patches when the cache is populated
- [ ] Push to trigger CI/CD releases in relevant npc_ repos
- [ ] Raise dependency floors in npc_sessions
- [ ] Create cache for problematic sessions and validate
