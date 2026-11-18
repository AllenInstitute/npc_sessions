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

## Decisions needed

1. **Licks:** trials overlapping a replayed interval are currently excluded via `invalid_times`, and reconstructed licks (±8.3 ms) appear only in the licks table. Keep that, or keep the trials?
	comments:
	- replayed intervals should not be invalid times in the NWB. we're reconstructing them so they can be used, not ignored.
	- reconstructed licks should appear in licks table only for now. response time in trials table should be nan.
2. **Count ≠ stim with no explained fault:** currently `unavailable` plus a reason; nothing raises.
   comments: fine. 
3. **Release order and timing:** npc_sync → npc_stim (floor) → npc_sessions (floors). The strict default in npc_sync breaks rig QC and npc_sessions_cache on ~5% of files.
4. **Ephys barcode gate** (≤0.5 ms AP/NI-DAQ, ≤0.8 ms LFP): apply it, or not?
	comment: only 1 gate is necessary (same clock for both). <1ms for AP is fine.

## Critical before proceeding - phantom flicker
regarding "§4: what the extra flips in stalls are" - these are phantom/ghost flicker as documented in the pre-upgrade npc_sync sync.py. 
I'm not sure if they as TTL events on sync - their amplitude can change over time.
You must consider this ghost flicker handling as an explanation for extra flips in the sync line where they do not match the inter-frame interval predicted by the stim file. 

## Next work (not started)

- **npc_ephys:** replace the first/last-barcode fit. It is >1 ms off on 104 devices in 43 sessions (`ephys_barcode_robust.log`), mostly ProbeD on PXI-116. 17 devices are off by ≥30 s because barcodes were mispaired: 751595_2024-11-20, 837959_2026-04-01, 649944_2023-02-27, 660023_2023-08-08. **Check against real NWB unit times first.**
comments: write a pdf report with figures to explain these phenomena, do not fix yet.
- **Camera spans:** 5/1,286 exposing spans fail the 0.1 ms gate, all in 813586_20251117 at 811–815 s. All three cameras agree on a +0.1–0.15 ms excess, so it's a timeline problem, and that block is already rejected.
  - The 857681_20260720 eye-cam failure was the startup PWM signature (10 × 2 ms exposures) skewing the falling-edge fit. It is now excluded from fits (`startup_pwm_section`, npc_sync `7e64286`), with no other change across 9,692 line × span × edge checks (`camera_gate_check.log`).
  - `info.json` is not used. Possible extra gate: patched exposure count = `FramesRecorded + LostFrames` (unverified).
  - npc_mvr is still not patch-aware for spans that end up unknown.
- **npc_samstim:** bounded onset search.
- **Box-1 photodiode prediction** (pre-2023 files): 35 of 60 spans are rejected.
- **Unexplained residual** in multi-run files: about −0.55 ms per second of span. The camera clock is continuous across those spans.
- **Test failures:** `test_opto` for 636766_2023-01-23 (2 failures, not regressions). The RF-mapping block is rejected by the box-1 gate.
- **Plan leftovers:** the npc_stim `atol=1` matcher, end-of-block truncation, NWB/cache regeneration, the MPE buffer fix, and cache cleanup.

## Gotchas

- npc_stim's `.env` holds stale AWS keys, so S3 tests fail. Move it aside while testing.
- The npc_sync dev venv has no S3 access. Use the npc_sessions venv with `PYTHONPATH`.
- `ephys_barcode_residuals.jsonl` (21 MB) is uncommitted, on disk only. The analog window data is in `C:/data/t2p_qc/analog_windows/`.
- Patches are written only if `$NPC_SESSIONS_SYNC_PATCH_DIR` is set. Nothing has been written to S3.

## Your notes

- 
