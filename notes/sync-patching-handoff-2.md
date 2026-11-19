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

## Decis
ions needed

1. **Licks:** trials overlapping a replayed interval are currently excluded via `invalid_times`, and reconstructed licks (±8.3 ms) appear only in the licks table. Keep that, or keep the trials?
	comments:
	- replayed intervals should not be invalid times in the NWB. we're reconstructing them so they can be used, not ignored.
	- reconstructed licks should appear in licks table only for now. response time in trials table should be nan.
2. **Count ≠ stim with no explained fault:** currently `unavailable` plus a reason; nothing raises.
   comments: fine. 
3. **Release order and timing:** npc_sync → npc_stim (floor) → npc_sessions (floors). The strict default in npc_sync breaks rig QC and npc_sessions_cache on ~5% of files.
  comments: these are released except for _cache
4. **Ephys
 barcode gate** (≤0.5 ms AP/NI-DAQ, ≤0.8 ms LFP): apply it, or not?
	comment: only 1 gate is necessary (same clock for both). <1ms for AP is fine.

- **npc_ephys:** re
p
lace the 
fi
rst/last-barcode fit. It is >1 ms off on 104 devices in 43 sessions (`ephys_barcode_robust.log`), mostly ProbeD on PXI-116. 17 devices are off by ≥30 s because barcodes were mispaired: 751595_2024-11-20, 837959_2026-04-01, 649944_2023-02-27, 660023_2023-08-08. **Check against real NWB unit times first.**
comments: 
fixing with another agent - skip.
- is npc_m
vr patch aware now? 
- **Box-1 
photodiode prediction** (pre-2023 files): 35 of 60 spans are rejected.
- **Unexpl
ained residua
l** in mult
i-run files: about −0.55 ms per second of span. The camera clock is continuous across those spans.
- **Test f
ailures:** `t
est_opto` f
or 636766_2023-01-23 (2 failures, not regressions). The RF-mapping block is rejected by the box-1 gate.
- **Plan l
eftovers:** the npc_stim `atol=1` matcher, end-of-block truncation, generate patches, feynamn docs cleanup/preservation.

## Gotcha
s
- The npc_
sync dev venv
 has no S3 
access. Use the npc_sessions venv with `PYTHONPATH`.
- `ephys_b
arcode_residu
als.jsonl` 
(21 MB) is uncommitted, on disk only. The analog window data is in `C:/data/t2p_qc/analog_windows/`.
- Patches 
are written o
nly if `$NP
C_SESSIONS_SYNC_PATCH_DIR` is set. Nothing has been written to S3.
comment: l
et's change this - we should always update patches if AWS creds are available and the versions of any of the libraries invovled have changed.

## Your n
o
tes

- ensure we have patch info written to s3 includs versions of deps that could affect it, for knowing when to regenerate. 

## Close-out (2026-11-18)

Applied in npc_sessions local `main` (committed, not pushed, no version bump). Evidence: `experiments/sync-patch-manifest/p4_replay_trials_check.log`. Summary: `outputs/sync-patching-report.md` § Close-out.

| Item | Status |
|---|---|
| 1. Replayed intervals in `invalid_times` | **Done.** They are no longer `invalid_times`. Overlapping trials are kept (`timing_source='patched'`). Epochs get the tag `replayed_sync` and a note. |
| 1. Reconstructed licks only in the licks table; `response_time` NaN | **Done.** `response_time` is NaN when the lick line is unknown in the response window before the first measured lick. Sound and opto onsets/offsets on an unknown line are NaN too. Without that, npc_samstim's unbounded search would return the next stimulus's edge. OptoTagging rows with an unknown onset are left out of the table (logged). |
| 2. Count ≠ stim | Unchanged (agreed). |
| 3. Release order | npc_sync 0.3.5, npc_stim 0.1.25, npc_ephys 0.1.52 are released, and the npc_sessions floors already require them. **npc_sessions_cache is not done:** its branch `move-display-times-to-npc-stim` has uncommitted changes. I didn't touch it. |
| 4. Ephys gate | **Already in npc_ephys 0.1.52.** The gate is 1 ms max leave-one-out residual for AP, LFP and NI-DAQ, and an LFP stream fails with its AP stream. The LFP stream still gets its own 1 ms check when AP passes. That's left to the npc_ephys work. |
| Patches always updated with AWS creds + dep versions | **Done.** The default cache is `s3://aind-scratch-data/dynamic-routing/cache/sync_patches` when botocore finds credentials. `$NPC_SESSIONS_SYNC_PATCH_DIR` overrides it; `none` disables it. The JSON adds `dependency_versions` (npc_sessions, npc_sync, npc_stim). A patch is rebuilt and overwritten when the builder or any version differs. Read and write errors are logged and the patch is built in memory. Still nothing on S3: test and check runs used `none` or a temp dir. |
| npc_mvr patch-aware? | **No** (unchanged). It reads the patched `SyncDataset`, so it gets reconstructed camera edges. Camera spans that fail the gate (6/1,274 exposing spans) return no edges there. It never calls `get_unknown_intervals`. Not checked: whether npc_mvr's frame-count tolerance then rejects those videos or shifts their frames. |
| Also fixed (found while testing) | Upstream breakage (pandas 3 / universal-pathlib 0.3 / zarr): remote UPaths passed to `pd.read_csv`/`read_pickle`/`read_excel` and `zarr.open` now go through file handles or fsspec mappers. 2 doctests were made independent of UPath repr, pandas Index repr and the last digits of the new ephys fit. Also fixed: the opto latency assert in `DynamicRouting1.get_trial_opto_onset`, which compared absolute onset times with 0.1 s. |
| Tests | Suite: 49 passed, 2 skipped, 1 failed. The failure is `test_surface_recording` ("703880_2024-04-17_1 has no main raw data assets"). It also fails on `HEAD`; it's a data lookup problem, not a code one. `tests/test_timing.py` has 8 tests (3 new). mypy is clean on the changed modules. |

Still open (not addressed here):
- 20 cleaned blocks (14 OptoTagging) are still rejected by npc_stim's rise + 15 ms check (`asym_round`; see `outputs/phantom-flicker-report.md`).
- The −0.55 ms/s residual in multi-run files is unexplained.
- Plan leftovers: the npc_stim `atol=1` matcher, end-of-block truncation, NWB/cache regeneration after release, and Feynman docs cleanup.
- Release steps: bump the npc_sessions version, push `main` (local `main` is ahead of origin), and release npc_sessions_cache after the strict-sync changes.

### Follow-up (2026-11-18): licks in trials, sound from NI-DAQ, opto NaN

Builder `npc_sessions.timing 3`. Evidence: `experiments/sync-patch-manifest/p5_sound_nidaq_vs_sync.log` and `p6_event_lines_check.log`. Summary: report § Follow-up.

| Request | Status |
|---|---|
| Reconstructed licks in the trials table | **Done.** `response_time` uses measured and reconstructed licks. |
| `timing_source` in the licks table, not the trials table | **Done.** The trials column is removed. The licks and rewards tables keep theirs. |
| Sound patched in sync from the NI-DAQ, via npc_samstim | **Done, for ephys sessions.** It wasn't before: the sound line was 'unknown' in every replay. The sound is on NI-DAQ AI1 (`npc_samstim.get_nidaq_channel_for_stim_onset`); AI0 is the photodiode. The patch now reconstructs the sync sound line from npc_samstim's NI-DAQ onsets, gated at ≤1 ms against the sync line on the other trials (observed ≤0.211 ms). It is reconstructed in 5/5 sessions. Behavior-only sessions have no NI-DAQ, so their sound line stays unknown and onsets there are NaN. |
| Opto patched from npc_stim; NaN in trials overlapping replays | **Done.** Laser lines are reconstructed from trigger frames + latency, only if the edges pair one-to-one with triggers outside the span. Leave-out error was ≤0.99 ms, but the stated bound is ±1/2 frame. Trial opto times are NaN in overlapping trials. OptoTagging rows are kept, with bounds from the reconstructed edges. Not testable on data: no opto task session with a replay exists. |

Notes:
- Building a patch for an ephys session with a replay now reads the NI-DAQ: 3–5 min per session (p5 timings), once per patch (cached).
- Fixed in passing: the sound reconstruction ignores pulses more than 60 s from the span, because 839146_20260518 has an extra pulse after the task's last trial.

### Follow-up 2 (2026-11-18): opto rows

- **OptoTagging:** trials overlapping a replayed interval are **removed**; `trial_index` keeps the stim-file indices.
- **Task trials:** trials with opto that overlap a replay are **kept**. Opto times are NaN there, and `is_opto` stays True. Before this fix, the NaN would have turned `is_opto` False.
- **Evidence:** `experiments/sync-patch-manifest/p7_opto_replay_simulation.log`. The intervals are simulated, because no real opto block has a replay.
- **Not changed (existing behaviour):** a session with 2 OptoTagging files has `trial_index` 0–149 twice in `optotagging_trials` (714753_2024-07-03). The sessions code computes an offset but never applies it.

- Filed as https://github.com/AllenInstitute/npc_sessions/issues/195 (2026-11-18). Common: 116/130 sessions with OptoTagging have 2+ files. The fix needs a decision (unique index vs per-file index + file column).
- **#195 implemented (2026-11-18, local `main`, not pushed).** In intervals tables built from several stim files (OptoTagging, RFMapping, LuminanceTest), `trial_index` now continues across the files in time order: each file is offset by the total trial count of the earlier files. `trial_index_in_block` is the new per-file index. Trials left out of a table change neither.
  - Evidence: `experiments/sync-patch-manifest/p8_trial_index_check.log` (H1–H4 pass). mypy is clean. The tests were not run locally; they'll run in CI on push.
  - The task `trials` table is unchanged. Its `trial_index_in_block` already means the index within a task block.
