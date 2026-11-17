"""Ephys barcode residual (never measured before; step 2 of notes/sync-patching-handoff.md).

For every npc_lims ephys session in sessions.csv: decode barcodes on sync (edges inside invalid
intervals +-2 s removed, as `npc_sessions.timing` does) and on each Open Ephys device, pair
barcodes by value, and measure how well the current alignment (npc_ephys
`get_probe_time_offset`: an affine map from the FIRST and LAST matched barcodes) predicts every
other matched barcode:

    predicted sync time = probe_time / scale - translation      (npc_ephys convention)
    residual            = sync time - predicted

Also: residual of a least-squares affine fit to all matched barcodes (what a robust fit would
leave), and the number of matched barcodes.

Output: ephys_barcode_residuals.jsonl (one row per device). Read-only on S3 and the share.
Run: PYTHONPATH=../../../npc_sync/src ../../.venv/Scripts/python.exe ephys_barcode_residuals.py [workers] > ephys_barcode_residuals_run.log 2>&1
"""

from __future__ import annotations

import concurrent.futures
import csv
import json
import logging
import pathlib
import sys
import time
import traceback

import numpy as np

HERE = pathlib.Path(__file__).parent
OUT = HERE / "ephys_barcode_residuals.jsonl"
PAD = 2.0


def process(row: dict) -> list[dict]:
    logging.disable(logging.WARNING)
    import npc_ephys
    import npc_sessions
    import npc_sync

    t0 = time.time()
    base = dict(session=row["session"], key=row["key"], sync_path=row["sync_path"])
    try:
        s = npc_sessions.DynamicRoutingSession(row["key"])
        dirs = s.ephys_recording_dirs
        raw = npc_sync.SyncDataset(row["sync_path"], on_invalid="ignore")
        on = raw.get_rising_edges("barcode_ephys", units="seconds")
        off = raw.get_falling_edges("barcode_ephys", units="seconds")
        bad = [(a - PAD, z + PAD) for a, z in raw.invalid_intervals]
        on, off = on[~npc_sync.in_intervals(on, bad)], off[~npc_sync.in_intervals(off, bad)]
        st, sid = npc_ephys.barcodes.extract_barcodes_from_times(on_times=on, off_times=off, total_time_on_line=raw.total_seconds)
        disc = raw.timebase_discontinuities
        out = []
        for info in npc_ephys.get_ephys_timing_on_pxi(dirs):
            d = info.device
            ch = npc_ephys.openephys.BARCODE_TTL_CHANNEL
            pt, pid = npc_ephys.barcodes.extract_barcodes_from_times(
                on_times=d.ttl_sample_numbers[d.ttl_states == ch] / info.sampling_rate,
                off_times=d.ttl_sample_numbers[d.ttl_states == -ch] / info.sampling_rate,
                total_time_on_line=d.ttl_sample_numbers[-1] / info.sampling_rate if len(d.ttl_sample_numbers) else 0.0,
            )
            rec = dict(base, device=d.name, n_sync=int(len(st)), n_device=int(len(pt)), n_invalid=len(raw.invalid_intervals),
                       n_discontinuities=len(disc))
            try:
                pe, me = npc_ephys.barcodes.match_barcodes(st, sid, pt, pid)
                scale, translation = npc_ephys.barcodes.linear_transform_from_intervals(me, pe)
            except Exception as exc:  # noqa: BLE001
                rec["error"] = f"match: {type(exc).__name__}: {exc}"
                out.append(rec)
                continue
            # pair by value (unique values only)
            sv, sc = np.unique(sid, return_counts=True)
            pv, pc = np.unique(pid, return_counts=True)
            common = np.intersect1d(sv[sc == 1], pv[pc == 1])
            ms = np.array([st[sid == v][0] for v in common])
            mp = np.array([pt[pid == v][0] for v in common])
            o = np.argsort(ms)
            ms, mp = ms[o], mp[o]
            pred = mp / scale - translation
            res = ms - pred
            A = np.c_[mp, np.ones_like(mp)]
            coef, *_ = np.linalg.lstsq(A, ms, rcond=None)
            res_ls = ms - A @ coef
            # which matched barcodes are outside the device's first/last endpoints
            rec.update(
                n_matched=int(len(common)),
                scale=scale,
                translation=translation,
                endpoint_res_max_ms=float(np.max(np.abs(res)) * 1e3) if len(res) else None,
                endpoint_res_p99_ms=float(np.quantile(np.abs(res), 0.99) * 1e3) if len(res) else None,
                endpoint_res_median_ms=float(np.median(np.abs(res)) * 1e3) if len(res) else None,
                ls_res_max_ms=float(np.max(np.abs(res_ls)) * 1e3) if len(res) else None,
                ls_res_p99_ms=float(np.quantile(np.abs(res_ls), 0.99) * 1e3) if len(res) else None,
                worst_t=float(ms[np.argmax(np.abs(res))]) if len(res) else None,
                res_ms=np.round(res * 1e3, 4).tolist(),
                t=np.round(ms, 3).tolist(),
            )
            out.append(rec)
        for r in out:
            r["seconds"] = round(time.time() - t0, 1)
        return out or [dict(base, error="no devices")]
    except Exception as exc:  # noqa: BLE001
        return [dict(base, error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-1500:])]


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    rows = [r for r in csv.DictReader(open(HERE / "sessions.csv")) if r["is_ephys"] == "True" and r["sync_path"]]
    # one sync file per session key (folders with several sync files: the largest, as npc_sessions does)
    by_key: dict[str, dict] = {}
    for r in rows:
        if r["key"] not in by_key:
            by_key[r["key"]] = r
        else:
            try:
                if pathlib.Path(r["sync_path"]).stat().st_size > pathlib.Path(by_key[r["key"]]["sync_path"]).stat().st_size:
                    by_key[r["key"]] = r
            except OSError:
                pass
    done = set()
    if OUT.exists():
        done = {json.loads(x)["key"] for x in OUT.read_text().splitlines()}
    todo = [r for k, r in by_key.items() if k not in done]
    print(f"{len(by_key)} ephys sessions, {len(done)} done, {len(todo)} to do", flush=True)
    t0 = time.time()
    with concurrent.futures.ProcessPoolExecutor(n) as ex, open(OUT, "a") as fo:
        futs = {ex.submit(process, r): r for r in todo}
        for i, fut in enumerate(concurrent.futures.as_completed(futs)):
            r = futs[fut]
            try:
                res = fut.result()
            except Exception as exc:  # noqa: BLE001
                res = [dict(session=r["session"], key=r["key"], error=f"worker: {exc!r}")]
            for x in res:
                fo.write(json.dumps(x) + "\n")
            fo.flush()
            print(f"[{i + 1}/{len(todo)} {time.time() - t0:.0f} s] {r['key']}: " + "; ".join(
                f"{x.get('device', '')[-12:]} n={x.get('n_matched')} max={x.get('endpoint_res_max_ms') and round(x['endpoint_res_max_ms'], 3)} {x.get('error', '')[:80]}" for x in res), flush=True)


if __name__ == "__main__":
    main()
