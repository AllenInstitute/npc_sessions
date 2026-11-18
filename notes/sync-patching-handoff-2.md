# Handoff 2: sync patching follow-up

Written 2026-11-17. Previous handoff: `notes/sync-patching-handoff.md` (done).
Full report: `outputs/sync-patching-report.md`. Plan and status: `outputs/sync-frame-timing-upgrade-plan.md`. Evidence: `experiments/sync-patch-manifest/` (see its README).

## State

| Repo | Branch `sync-patch` | Pushed |
|---|---|---|
| npc_sync | `7e64286`: `npc_sync.patch` (`SyncPatch`, reconciliation, builder); strict `on_invalid` | no |
| npc_stim | `7db2ce4`: patch-aware `get_display_times`, `frame_source` | no |
| npc_sessions | `28f0c4f`: `npc_sessions.timing`, patched `sync_data`, `timing_source`, manifest, report | no |

- `main` is untouched in all three repos. npc_sessions local `main` is still 5 commits ahead of origin.
- The npc_sessions branch needs the unreleased npc_sync. Run it with `PYTHONPATH="../npc_sync/src;../npc_stim/src;src"`.
- Gates on the branch: vsync per-gap residual ≤0.2 ms; camera ≤0.1 ms; photodiode checked over the 0.5 s after each span.
## Critical before proceeding - phantom flicker
regarding "§4: what the extra flips in stalls are" - these are phantom/ghost flicker as documented in the pre-upgrade npc_sync sync.py. 
Does not apply to Box-3.
I'm not sure if they as TTL events on sync - their amplitude can change over time.
You must consider this ghost flicker handling as an explanation for extra flips in the sync line where they do not match the inter-frame interval predicted by the stim file. 

### Design agreed (2026-11-17): diode edges in dropped-frame intervals

- **Outside replay intervals:** in npc_sessions, use the stim `frameIntervals` (mapped to vsyncs) to find dropped frames (long intervals). A long interval should have no diode toggle. Edges inside it are phantom flicker, and the patch removes them.
- **Inside replay intervals:** keep no measured diode edges. Create new edges from the reconstructed frame times + per-polarity latency, spaced by the frame intervals. `predict_diode_edges` already does this; it should also remove phantom edges just outside the span. Check whether this explains the 21 box-1 "wrong polarity after the span" failures.
- **Don't use flicker as a catch-all.** All should hold:
  - they fall inside a Box-1 stim-recorded long interval (the display can still only flicker at 1Hz)
  - they are a state-restoring pair
  - Any other extra edge stays unexplained and the block is rejected. Report removal counts per session.
- **Patch scope changes:** removals apply to measured data, including files with no replays, so every session may get a patch.
- **Ownership:** npc_stim's own extra-flip heuristics (anomalous short intervals, §4 stalls) must not also remove edges. npc_stim should consume the cleaned line, under a test.
- **Validation:**
  - indices of long frame intervals from stim file should perfectly match those on the patched photodiode sync line.
  - Removed edges sometimes have less than full-amplitude AI0 transition (2025+) or a DC offset - but it's unreliable.
  - Rerun the box-1 gate (`diode_gate_check.py`).
  - Check whether §4 stall extra flips fall inside recorded long intervals.
  - ensure we're not double-fixing with earlier code that tries to address the same problem by other means.

### Status (2026-11-17): phantom flicker done, merged into local `main` in npc_sync, npc_stim, npc_sessions (not pushed)

Report: `outputs/phantom-flicker-report.md`. Evidence: `experiments/phantom-flicker/`.
- **Implemented as designed.** npc_sync `align_diode_toggles` checks every box-1 block. The patch (`SyncPatch.removed`, `.diode_blocks`) removes phantom flicker and glitches in every session. npc_stim consumes cleaned blocks without its own removal (tested). Unexplained edges reject the block.
- **The rule holds in the data.** Over 362,082 holds: an even number of phantom edges, at most n − 1, only in holds of 3 or more refreshes. AI0 confirms 3/3 removed pairs are real non-frame dips (762526_20250320). In all 24 §4 blocks every extra flip is accounted for.
- **Population.** 1,936/1,936 accepted blocks byte-identical, 0 lost, 11 DR1 blocks recovered. 20 cleaned blocks (14 OptoTagging) stay rejected: npc_stim's rise + 15 ms check catches its own after-hold frame placement (`asym_round`); next step in the report.
- **The 21 box-1 "wrong polarity" failures are not flicker.** They come from a pairing bug in `predict_diode_edges`: one frame late when the latency exceeds a refresh. Fixed; 53/53 box-1 spans with reconstructed vsyncs pass (21 before).
- **`test_opto` fixed.** The 636766_2023-01-23 failures came from the box-1 bug; the npc_sessions suite passes 23/23.
- **npc_mvr is not patch-aware** (question below). It gets the patched camera edges through the patched `SyncDataset`, but it doesn't consult `get_unknown_intervals`.
- **Not done:** decisions 1 (`invalid_times`) and 4 (ephys gate); "always update patches if AWS creds are available and library versions changed" (Gotchas); release and push.

## Next work (not started)

## Decisions needed

1. **Licks:** trials overlapping a replayed interval are currently excluded via `invalid_times`, and reconstructed licks (±8.3 ms) appear only in the licks table. Keep that, or keep the trials?
	comments:
	- replayed intervals should not be invalid times in the NWB. we're reconstructing them so they can be used, not ignored.
	- reconstructed licks should appear in licks table only for now. response time in trials table should be nan.
2. **Count ≠ stim with no explained fault:** currently `unavailable` plus a reason; nothing raises.
   comments: fine. 
3. **Release order and timing:** npc_sync → npc_stim (floor) → npc_sessions (floors). The strict default in npc_sync breaks rig QC and npc_sessions_cache on ~5% of files.
  comments: fine. 
4. **Ephys barcode gate** (≤0.5 ms AP/NI-DAQ, ≤0.8 ms LFP): apply it, or not?
	comment: only 1 gate is necessary (same clock for both). <1ms for AP is fine.

- **npc_ephys:** re
p
lace the first/last-barcode fit. It is >1 ms off on 104 devices in 43 sessions (`ephys_barcode_robust.log`), mostly ProbeD on PXI-116. 17 devices are off by ≥30 s because barcodes were mispaired: 751595_2024-11-20, 837959_2026-04-01, 649944_2023-02-27, 660023_2023-08-08. **Check against real NWB unit times first.**
comments: fixing with another agent - skip.
- is npc_mvr patch aware now? 
- **Box-1 photodiode prediction** (pre-2023 files): 35 of 60 spans are rejected.
- **Unexplained residua
l** in multi-run files: about −0.55 ms per second of span. The camera clock is continuous across those spans.
- **Test failures:** `t
est_opto` for 636766_2023-01-23 (2 failures, not regressions). The RF-mapping block is rejected by the box-1 gate.
- **Plan leftovers:** the npc_stim `atol=1` matcher, end-of-block truncation, generate patches, feynamn docs cleanup/preservation.

## Gotchas
- The npc_sync dev venv
 has no S3 access. Use the npc_sessions venv with `PYTHONPATH`.
- `ephys_barcode_residu
als.jsonl` (21 MB) is uncommitted, on disk only. The analog window data is in `C:/data/t2p_qc/analog_windows/`.
- Patches are written o
nly if `$NPC_SESSIONS_SYNC_PATCH_DIR` is set. Nothing has been written to S3.
comment: let's change this - we should always update patches if AWS creds are available and the versions of any of the libraries invovled have changed.

## Your notes

- 





