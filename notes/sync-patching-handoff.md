# Handoff: sync patching (P0 survey → P1 `SyncPatch` → P3 `npc_sessions.timing`)

Written 2026-11-17. Status: **done on local branches, not pushed** (2026-11-17).
- Report: `outputs/sync-patching-report.md`. Status blocks: `outputs/sync-frame-timing-upgrade-plan.md` (P0–P4 "update" blocks).
- Branches named `sync-patch` in npc_sync, npc_stim and npc_sessions.
- Open: the three decisions below; npc_mvr, npc_samstim and npc_ephys patch-awareness; box-1 photodiode prediction.

## Task

Build what's needed to *patch* sync faults instead of rejecting the affected sessions:
1. the P0 survey manifest
2. re-derived gates
3. `SyncPatch` plus the ported reconciliation in npc_sync (P1)
4. `npc_sessions.timing` (P3)

Plan: `outputs/sync-frame-timing-upgrade-plan.md`. The per-phase status blocks (dated 2026-11-17) under each work-plan heading list exactly what is done and what isn't. Read them first.

## Read first

1. `outputs/sync-frame-timing-upgrade-plan.md`: the policy table, the tolerance table, the P0–P4 work plan with status, and the open questions.
2. `../npc_sync/outputs/vsync-frame-count-discrepancies.md`: replay and drop mechanics, the analog ground truth, and the recommended changes (items 1–6).
3. `experiments/sync-frame-timing/README.md`: the evidence scripts. `multi_gap_check.py` has the per-gap solve.
4. `../npc_sync/experiments/vsync-frame-count/repair.py`: the reconciliation prototype. It assumes a single gap.
5. `outputs/display-times-issues.md` and `outputs/display-times-analog-validation.md`: the estimator fixes, and what the analog diode can and can't validate.

## State (2026-11-17)

| Repo | `main` | Pushed |
|---|---|---|
| npc_sync | 0.3.1: detection only (`invalid_intervals`, `timebase_discontinuities`, `get_edges_in_blocks`); display times moved out | yes (PyPI) |
| npc_stim | 0.1.24: `get_display_times` with provenance; refuses files with replays or discontinuities | yes (PyPI) |
| npc_sessions | 5 commits ahead of `origin/main`: dependency floors, `is_photodiode` provenance, reports | **no (user's call)** |

- **No patching code exists.** Nothing in `npc_sync/src` matches `SyncPatch` or `reconcile`.
- **Unguarded consumers.** npc_sync's `vsync_times_in_blocks` and the edge getters don't raise on `invalid_intervals`. Only display times are guarded. Cameras, licks and barcodes are not.
- **The analog diode is QC only** (decided 2026-11-17): it isn't part of standard processing. It exists only from 2025-01-13, on AI0. Use it to validate patched frames inside replay windows for sessions from 2025 on (`experiments/display-times-analog-validation/`).
- **Local data.**
  - Cache of sync and stim files: `C:/data/t2p_cache` (442 Task 2 pilot sessions, 59 GB).
  - The 7 replay or drop sessions: `../npc_sync/experiments/vsync-frame-count/data/`.
  - The share: `//allen/programs/mindscope/workgroups/dynamicrouting/PilotEphys/Task 2 pilot` (read-only).
  - NI-DAQ continuous data are on S3 only (`experiments/display-times-analog-validation/locate.py`).

## Steps

1. **P0 manifest** (do this first; it scopes everything else).
   - Run on every session with sync: npc_lims-tracked sessions plus the share. The 442 cached sessions are a subset.
   - For each session record: `invalid_intervals`, `timebase_discontinuities`, and a strict check of each block's vsync count against its stim frames, using an exact one-to-one stim match rather than `atol=1`.
   - Output one CSV row per block: intervals, overlapping blocks, stim events lost by type, and a class (clean / replay-reconcilable / displaced / drop / unexplained).
   - *Accept:* every block whose count ≠ stim frames is classified.
   - Put the code and logs in `experiments/sync-patch-manifest/`.
2. **Re-derive the gates from the manifest population**: the anchor residual (now 0.5 ms), the camera residual (0.1 ms) and the ephys barcode residual (never measured). Record the distributions. Don't widen the gates to fit the unexplained ~0.4 ms per-gap residual (plan, open questions).
3. **P1 in npc_sync:**
   - Define `SyncPatch`: per-line patched event times, per-event provenance (measured / interpolated / missing), the fault, the anchors and their residuals, and a serialisable format.
   - Port `find_replay_runs` (detection is already in `src`) and `reconcile_replay` with a **per-gap** solve (`multi_gap_check.py`).
   - Port `locate_drop`/`reconcile_drop`, but the policy for drops is still to reject.
   - Add gap-aware blocks.
   - Do the P0 hotfix: getters raise, or mask explicitly, on `invalid_intervals`.
   - *Accept:* on the 7 local sessions, reproduces the prototype (anchor ≤0.05 ms, camera ≤0.032 ms). The synthetic replay tests pass.
   - Multi-run files with displaced barcode onsets stay **rejected**.
4. **P2 patch-awareness** for the consumers you need for P3. At minimum, npc_stim display times inside a gap: interpolated vsync plus the local latency (recommended change 4 in the npc_sync discrepancies report). Cameras (npc_mvr) and opto/sound (npc_samstim) only if P3 uses them now.
5. **P3 `npc_sessions.timing`:**
   - per-line builders that apply persisted patches
   - the gates, so a block that fails one is rejected with no fallback
   - `timing_source` columns in trials, licks, rewards and frame times
   - `invalid_times` for everything else
   - Needs a user decision on licks first (see below).
6. **Validate.** Where the analog diode exists (2025+), compare patched frames in replay windows against it with the existing pipeline (`fetch_analog.py` → `compare.py`). Everywhere else, use the prototype's anchor residuals.

## Decisions needed from the user

- Licks reconstructed at ±8.3 ms: keep those trials in response-time analyses or not.
- Whether a vsync count that differs from the stim frames (no explained fault) is `unavailable` plus a reason, as npc_stim does now, or a raise, as the plan says.
- When to push npc_sessions `main`. Steps 4–5 touch it.

## Rules

- Read-only on the share and S3. Don't push without approval. A push to `main` in npc_sync, npc_stim or npc_sessions publishes a release.
- Don't widen tolerances. Every removed tolerance needs a test that fails under the old behaviour (P2 acceptance).
- Mark anything unchecked `not run` or `blocked` in the report, and don't guess.
- Update the status blocks in the plan as items finish.

## Leftovers that are not part of this handoff

These are listed under P2/P4 in the plan:
- stim matcher exactness
- end-of-block truncation
- npc_mvr #176 and npc_ephys `match_barcodes`
- the full P4 analog check for 2025+
- NWB/cache regeneration for the changed blocks (`experiments/display-times-issues/compare_fixE.csv`)
- cleanup of `C:/data/t2p_cache`, `C:/data/t2p_qc/{out_issues,analog,venv_analog}` and the old worktree `C:/data/t2p_qc/wt/npc_stim`
