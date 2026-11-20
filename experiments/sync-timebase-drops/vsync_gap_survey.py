"""Vsync redundancy check: per-gap frame solve (npc_sync.solve_vsync_gaps) for every vsync block
with a stim file, in every manifest file with replays or discontinuities, on the recorded clock
(invalid intervals = replayed copies: the model before) and on the timebase (invalid intervals =
dropped intervals).

A gap's residual is (measured time across the gap) - (stim frameIntervals over the solved frame
count); balance is stim frames - (measured + solved) frames for the block. A block is
reconcilable if balance = 0, no gap touches the block ends, every |residual| <= 0.2 ms and the
second-best frame count is >= 10 ms worse.

Run: PYTHONPATH=../../../npc_sync/src;../../src ../../.venv/Scripts/python.exe vsync_gap_survey.py > vsync_gap_survey.log
Output: vsync_gap_survey.jsonl (one row per block and model)
"""
from __future__ import annotations
import concurrent.futures, json, logging, pathlib, sys, time
import numpy as np
HERE = pathlib.Path(__file__).parent
MANIFEST = HERE.parent / "sync-patch-manifest" / "manifest_files.jsonl"

def one(d: dict) -> list[dict]:
    import npc_sync, npc_stim
    from npc_sessions import timing
    logging.disable(logging.WARNING)
    rows = []
    try:
        path = pathlib.Path(d["sync_path"])
        stims = sorted(p for p in path.parent.glob("*.hdf5"))
        recorded = npc_sync.SyncDataset(path, on_invalid="ignore")
        views = {"recorded": recorded, "timebase": timing.true_clock_view(recorded)}
        for model, view in views.items():
            try:
                blocks = view.vsync_times_in_blocks
                matches, _ = timing.match_stims_to_blocks(view, stims)
            except Exception as exc:  # noqa: BLE001
                rows.append(dict(session=d["session"], model=model, error=f"{type(exc).__name__}: {exc}"[:200])); continue
            inv = view.invalid_intervals
            fs = view.sample_freq
            spans = [(a - 1.5 / fs, z + (0.5 / fs if model == "timebase" else 0)) for a, z in inv]
            for j, m in sorted(matches.items()):
                b = blocks[j]
                fi = np.asarray(npc_stim.get_h5_stim_data(m.stim_path)["frameIntervals"][:], dtype=float)
                over = [(a, z) for a, z in spans if a <= b[-1] and z > b[0]]
                row = dict(session=d["session"], model=model, block=j, stim=m.stim_path.stem, n_stim=len(fi) + 1, n_rec=len(b), n_spans=len(over))
                if over:
                    rec = npc_sync.solve_vsync_gaps(b, fi, over)
                    row |= dict(balance=rec.balance, boundary_gap=bool(rec.boundary_gap), residuals_ms=[round(g.residual_sec * 1e3, 4) for g in rec.gaps],
                                second_best_ms=[round(g.second_best_sec * 1e3, 2) for g in rec.gaps], n_frames=[g.n_frames for g in rec.gaps],
                                ok=not rec.failures(npc_sync.patch.VSYNC_MAX_GAP_RESIDUAL_SEC), ok_1ms=not rec.failures(npc_sync.patch.VSYNC_MAX_GAP_RESIDUAL_PLAIN_SEC))
                else:
                    row |= dict(balance=len(fi) + 1 - len(b), boundary_gap=False, residuals_ms=[], ok=len(fi) + 1 == len(b), ok_1ms=len(fi) + 1 == len(b))
                rows.append(row)
    except Exception as exc:  # noqa: BLE001
        rows.append(dict(session=d["session"], error=f"{type(exc).__name__}: {exc}"[:200]))
    return rows

if __name__ == "__main__":
    rows = [json.loads(l) for l in MANIFEST.open()]
    todo = [d for d in rows if d["n_invalid"] or d["n_discontinuities"]]
    print(len(todo), "files", flush=True)
    with concurrent.futures.ProcessPoolExecutor(5) as ex, open(HERE / "vsync_gap_survey.jsonl", "w") as f:
        for i, res in enumerate(ex.map(one, todo)):
            for r in res: f.write(json.dumps(r) + "\n")
            f.flush()
            print(i, res[0]["session"] if res else None, len(res), [r.get("error") for r in res if "error" in r][:1], flush=True)
