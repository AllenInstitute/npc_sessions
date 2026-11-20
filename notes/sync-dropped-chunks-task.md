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

## Cause in the sync program (added 2026-11-20)

From reading the local clone `../sync` (`git describe`: 4.1.0-8-g1614844; the affected files were written by 4.3.0-4.3.2, which is not in the clone). Inferred from the code and matched to the data; the program was not run.

**One shared buffer, queued by reference.**

- `src/sync/ni/DIU32.py` line 21: `self.daq_buffer` is one array for the whole recording.
- Line 44: `ReadDigitalU32` overwrites it in place on every callback.
- Line 46: `self.buffer_q.put((timestamp, self.daq_buffer))` queues a reference, not a copy.
- `src/sync/sync_controller.py` lines 300-305: the writer thread copies the data only when it writes (`data[1].astype(np.uint32).tostring()`), and reopens the `.sync` and `_timestamps.txt` files for every chunk.

So file position p gets whatever is in the buffer when item p is written. If the writer is behind by s callbacks, position p holds chunk p + s. That is the shift state measured from the cameras.

| Writer behaviour | What the file gets | What was observed |
|---|---|---|
| Keeps up | chunk p at position p | clean file |
| Falls one further behind | the chunk about to be written is overwritten first, and is lost | a +1 jump (single dropped chunk) |
| Stays behind by s | each position holds the chunk s later | displaced stretch, early by s x 0.5 s |
| Catches up by one | two consecutive items written before the next callback hold the same chunk | one copy (-1) |
| Stalls for n chunks, then drains the queue | n + 1 items all hold the newest chunk | run of n copies with the drop at its start |

Consistent with the survey: drops before a run are almost always single chunks, every copy is identical to the chunk after it, the shift state is never negative, and copies equal drops in every replay file.

**Not explained: the four pure drops** (840160_20260428, 715709_20240808, 717473_20240916, 832032_20260223). The writer drains the queue at stop, so a backlog always ends in copies. A net shift needs callbacks that never queued anything. Candidate (unverified, from `../npc_sync/outputs/sync-dropped-data.md`): `ReadDigitalU32` raises inside the callback, so `put` is never reached. `_timestamps.txt` would settle it: the timestamp is taken in the callback, so a pure drop shows a gap of k x 0.5 s there and writer lag shows nothing.

**Side points.**

- `write_h5_data` (`src/sync/datasets.py` line 124) stamps each event with the index of the last sample before the change, so every event in every file is one sample (10 us) early. This is why the event before a copy "already carried the copy's state".
- Upstream fix, one line: `self.buffer_q.put((timestamp, self.daq_buffer.copy()))`. A slow writer then only delays the file.
