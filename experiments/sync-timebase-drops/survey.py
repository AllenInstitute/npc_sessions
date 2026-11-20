"""Survey: timebase (npc_sync.timebase, local branch) of every sync file in the P0 manifest
that has replays or discontinuities, plus a sample of clean files.

Per file: replayed runs/copies, drops located on camera lines vs assumed at the run start,
lag from each located drop to the next copy, notes (unexplained jumps), barcode discontinuities
as recorded and on the timebase, max |jump - integer| over observed boundaries.

Run: PYTHONPATH=../../../npc_sync/src ../../.venv/Scripts/python.exe survey.py [n_clean] > survey.log
Output: survey.jsonl
"""
from __future__ import annotations
import concurrent.futures, json, logging, pathlib, sys, time
import numpy as np
HERE = pathlib.Path(__file__).parent
MANIFEST = HERE.parent / "sync-patch-manifest" / "manifest_files.jsonl"

def one(d: dict) -> dict:
    import npc_sync
    logging.disable(logging.WARNING)
    out = dict(session=d["session"], n_invalid=d["n_invalid"], n_disc=d["n_discontinuities"])
    t0 = time.time()
    try:
        s = npc_sync.SyncDataset(d["sync_path"], on_invalid="ignore")
        assert npc_sync.__file__.replace("\\", "/").endswith("npc_sync/src/npc_sync/__init__.py"), npc_sync.__file__
        tb = s.find_timebase()
        fs = s.sample_freq
        # jump diagnostics
        from npc_sync.timebase import find_chunk_jumps
        from npc_sync.sync import get_bit
        ev = s.get_recorded_events(); times = ev[:, 0].astype(np.int64); bits = ev[:, -1]
        edges = {}
        for lab in [l for l in s.line_labels if l.endswith("_cam_exposing")]:
            ch = np.ediff1d(get_bit(bits, s._line_to_bit(lab)), to_begin=np.uint8(0))
            edges[f"{lab}:rising"] = times[ch == 1]; edges[f"{lab}:falling"] = times[ch == 255]
        j = find_chunk_jumps(edges, int(s.meta_data["total_samples"]), tb.chunk_samples)
        has = np.isfinite(j.value)
        out |= dict(
            chunk=tb.chunk_samples, n_chunks=j.n_chunks, n_lines=len(j.lines),
            n_boundaries_observed=int(j.observed.sum()), n_boundaries_with_estimate=int(has.sum()),
            max_frac_observed=float(np.abs(j.value - j.jump)[j.observed].max()) if j.observed.any() else None,
            n_unobserved_nonzero=int((has & ~j.observed & (np.abs(np.nan_to_num(j.value)) > 0.5)).sum()),
            n_copies=sum(x.n_chunks == -1 for x in tb.jumps),
            n_runs=len(s._replayed_buffers[0]),
            drops_measured=[(x.sample / fs, x.n_chunks, round(x.value, 3)) for x in tb.jumps if x.source == "periodic_lines"],
            drops_assumed=[(x.sample / fs, x.n_chunks) for x in tb.jumps if x.source == "assumed_at_run"],
            net_shift_s=tb.net_shift_samples / fs, notes=list(tb.notes), n_dropped_intervals=len(tb.dropped),
            dropped_s=sum(z - a for a, z in tb.dropped) / fs,
        )
        copies = np.array([x.sample for x in tb.jumps if x.n_chunks == -1])
        lags = []
        for x in tb.jumps:
            if x.n_chunks > 0 and copies.size:
                k = np.searchsorted(copies, x.sample, side="right")
                # first copy boundary after the drop is one chunk after the run start
                lags.append(float((copies[k] - tb.chunk_samples - x.sample) / fs) if k < len(copies) else None)
        out["lags_s"] = lags
        out["disc_recorded"] = [[round(v, 4) for v in x] for x in s.timebase_discontinuities]
        t = npc_sync.SyncDataset(s, on_invalid="ignore", timebase=tb)
        out["disc_true"] = [[round(v, 4) for v in x] for x in t.timebase_discontinuities]
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    out["sec"] = round(time.time() - t0, 1)
    return out

if __name__ == "__main__":
    n_clean = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    rows = [json.loads(l) for l in MANIFEST.open()]
    faulty = [d for d in rows if d["n_invalid"] or d["n_discontinuities"]]
    clean = [d for d in rows if not (d["n_invalid"] or d["n_discontinuities"])]
    rng = np.random.default_rng(0)
    clean = [clean[i] for i in sorted(rng.choice(len(clean), min(n_clean, len(clean)), replace=False))]
    todo = faulty + clean
    print(len(faulty), "faulty", len(clean), "clean", flush=True)
    with concurrent.futures.ProcessPoolExecutor(6) as ex, open(HERE / "survey.jsonl", "w") as f:
        for r in ex.map(one, todo):
            f.write(json.dumps(r) + "\n"); f.flush()
            print(r["session"], r.get("error") or (r["n_runs"], r["n_copies"], len(r["drops_measured"]), len(r["drops_assumed"]), r["net_shift_s"], len(r["disc_recorded"]), len(r["disc_true"]), r["notes"][:1]), r["sec"], flush=True)
