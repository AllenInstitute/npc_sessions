"""Run the frame-timing code paths on one or all cached Task 2 pilot ephys sessions and save
the outputs, so that baseline (npc_sync 0.2.9 + npc_stim 0.1.23) and new (branch
move-display-times-to-npc-stim) results can be compared byte-for-byte.

Run once per environment:
    <venv>/Scripts/python.exe qc_run.py <label> [--cache C:/data/t2p_qc...] [--workers N] [--only FOLDER]

Per session writes <out>/<label>/<folder>.npz (arrays) and <folder>.json (metadata,
exceptions, warnings). Read-only on data.

Code paths (same inputs for both versions; stim files = every .hdf5 in the session folder,
sync = the only/largest sync .h5, as in npc_sessions):
  A  per-block display times for every vsync block
       baseline: SyncDataset(sync).frame_display_time_blocks (+ is_block_using_diode)
       new:      npc_stim.get_display_times(sync, stim_paths)  (+ .source)
  A2 new only: deprecated shim SyncDataset(sync).frame_display_time_blocks (must equal A)
  B  npc_stim.get_stim_frame_times(*stims, sync=sync, frame_time_type="display_time")
       - the call npc_sessions makes (Session._stim_frame_times)
  C  same with frame_time_type="vsync"
  D  SyncDataset.stim_onsets / stim_offsets
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.metadata
import json
import logging
import os
import pathlib
import re
import subprocess
import sys
import time
import traceback
import warnings

HERE = pathlib.Path(__file__).parent
DEFAULT_CACHE = "C:/data/t2p_cache"
DEFAULT_OUT = "C:/data/t2p_qc/out"
TIMEOUT_S = 1800


class ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage().splitlines()[0][:300]
        self.records.append(f"{record.levelname}:{record.name}:{msg}")


def exc_info(exc: BaseException) -> dict:
    return {"type": type(exc).__name__, "msg": str(exc).splitlines()[0][:500] if str(exc) else ""}


def run_one(label: str, folder: pathlib.Path, out: pathlib.Path) -> None:
    import numpy as np
    import npc_stim
    import npc_sync

    warnings.simplefilter("ignore", DeprecationWarning)
    handler = ListHandler()
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.WARNING)
    for v in ("SYNC_FALLBACK_CONSTANT_LAG", "SYNC_ALLOW_MISSING_STIM_FILES", "SYNC_DEBUG", "SYNC_PLOT_DIR"):
        os.environ.pop(v, None)

    syncs = sorted(
        (p for p in folder.iterdir() if re.match(r"^\d{8}T\d{6}\.h5$", p.name)),
        key=lambda p: p.stat().st_size,
    )
    stims = sorted(p for p in folder.iterdir() if p.suffix == ".hdf5")
    meta: dict = {
        "label": label,
        "folder": folder.name,
        "npc_sync": importlib.metadata.version("npc_sync"),
        "npc_stim": importlib.metadata.version("npc_stim"),
        "npc_sync_file": npc_sync.__file__,
        "npc_stim_file": npc_stim.__file__,
        "sync_files": [p.name for p in syncs],
        "stim_files": [p.name for p in stims],
    }
    arrays: dict[str, np.ndarray] = {}
    if not syncs:
        meta["error"] = {"type": "NoSyncFile", "msg": ""}
        _save(out, folder.name, meta, arrays)
        return
    sync_path = syncs[-1]
    meta["sync_file"] = sync_path.name

    # sync-only facts (identical code in both versions: sanity check)
    t0 = time.time()
    try:
        s = npc_sync.SyncDataset(sync_path)
        meta["invalid_intervals"] = [list(x) for x in s.invalid_intervals]
        meta["timebase_discontinuities"] = [list(x) for x in s.timebase_discontinuities]
        vb = s.vsync_times_in_blocks
        meta["vsync_block_lengths"] = [int(len(b)) for b in vb]
        for i, b in enumerate(vb):
            arrays[f"vsync_block{i}"] = b
    except Exception as exc:
        meta["sync_error"] = exc_info(exc)

    # A: per-block display times
    n0 = len(handler.records)
    t = time.time()
    try:
        if label == "baseline":
            s = npc_sync.SyncDataset(sync_path)
            blocks = s.frame_display_time_blocks
            using = s.is_block_using_diode
            meta["A_sources"] = [
                ("photodiode" if u else ("unavailable" if b is None else "constant_lag"))
                for b, u in zip(blocks, using)
            ]
            meta["A_stim_paths"] = [
                None if p is None else p.name for p in (s.block_index_to_stim_path[i] for i in range(len(blocks)))
            ]
            meta["A_reasons"] = [None] * len(blocks)
            meta["A_diode_box_frame_interval"] = [None] * len(blocks)
        else:
            res = npc_stim.get_display_times(npc_sync.SyncDataset(sync_path), stims)
            blocks = [b.display_times for b in res]
            meta["A_sources"] = [b.source for b in res]
            meta["A_stim_paths"] = [None if b.stim_path is None else b.stim_path.name for b in res]
            meta["A_reasons"] = [b.reason for b in res]
            meta["A_diode_box_frame_interval"] = [b.diode_box_frame_interval for b in res]
        for i, b in enumerate(blocks):
            if b is not None:
                arrays[f"A_block{i}"] = np.asarray(b)
        meta["A_n_blocks"] = len(blocks)
    except Exception as exc:
        meta["A_error"] = exc_info(exc)
        meta["A_traceback"] = traceback.format_exc()[-2000:]
    meta["A_seconds"] = round(time.time() - t, 2)
    meta["A_warnings"] = handler.records[n0:]

    # A2: new-only shim
    if label != "baseline":
        t = time.time()
        try:
            s = npc_sync.SyncDataset(sync_path, stim_paths=stims)
            for i, b in enumerate(s.frame_display_time_blocks):
                if b is not None:
                    arrays[f"A2_block{i}"] = np.asarray(b)
            meta["A2_using_diode"] = list(s.is_block_using_diode)
        except Exception as exc:
            meta["A2_error"] = exc_info(exc)
        meta["A2_seconds"] = round(time.time() - t, 2)

    # B / C: get_stim_frame_times (the npc_sessions path)
    for key, ftype in (("B", "display_time"), ("C", "vsync")):
        n0 = len(handler.records)
        t = time.time()
        try:
            res = npc_stim.get_stim_frame_times(*stims, sync=sync_path, frame_time_type=ftype)
            meta[f"{key}_keys"] = [pathlib.Path(str(k)).name for k in res]
            meta[f"{key}_exceptions"] = {
                pathlib.Path(str(k)).name: exc_info(v) for k, v in res.items() if isinstance(v, Exception)
            }
            for k, v in res.items():
                if not isinstance(v, Exception):
                    arrays[f"{key}_{pathlib.Path(str(k)).stem}"] = np.asarray(v)
        except Exception as exc:
            meta[f"{key}_error"] = exc_info(exc)
            meta[f"{key}_traceback"] = traceback.format_exc()[-2000:]
        meta[f"{key}_seconds"] = round(time.time() - t, 2)
        meta[f"{key}_warnings"] = handler.records[n0:]

    # D: stim onsets/offsets
    try:
        s = npc_sync.SyncDataset(sync_path)
        arrays["D_onsets"] = np.asarray(s.stim_onsets)
        arrays["D_offsets"] = np.asarray(s.stim_offsets)
    except Exception as exc:
        meta["D_error"] = exc_info(exc)

    meta["total_seconds"] = round(time.time() - t0, 2)
    _save(out, folder.name, meta, arrays)


def _save(out: pathlib.Path, name: str, meta: dict, arrays: dict) -> None:
    import numpy as np

    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{name}.npz", **arrays)
    (out / f"{name}.json").write_text(json.dumps(meta, indent=1, default=str))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = pathlib.Path(args.out) / args.label
    cache = pathlib.Path(args.cache)
    if args.only:
        run_one(args.label, cache / args.only, out)
        return
    import csv

    folders = [r["folder"] for r in csv.DictReader(open(HERE / "ephys_sessions.csv"))]
    todo = [f for f in folders if args.force or not (out / f"{f}.json").exists()]
    print(f"{args.label}: {len(todo)} of {len(folders)} sessions to run", flush=True)

    def sub(folder: str) -> str:
        if not (cache / folder).exists():
            _save(out, folder, {"label": args.label, "folder": folder, "error": {"type": "NotCached", "msg": ""}}, {})
            return f"{folder}: not cached"
        t = time.time()
        try:
            p = subprocess.run(
                [sys.executable, __file__, args.label, "--cache", args.cache, "--out", args.out, "--only", folder],
                capture_output=True, text=True, timeout=TIMEOUT_S,
            )
            status = "ok" if p.returncode == 0 else f"exit {p.returncode}: {p.stderr.strip().splitlines()[-1:]}"
            if p.returncode != 0:
                _save(out, folder, {"label": args.label, "folder": folder,
                                    "error": {"type": "ProcessExit", "msg": p.stderr[-1500:]}}, {})
        except subprocess.TimeoutExpired:
            status = "timeout"
            _save(out, folder, {"label": args.label, "folder": folder, "error": {"type": "Timeout", "msg": f">{TIMEOUT_S}s"}}, {})
        return f"{folder}: {status} ({time.time() - t:.0f}s)"

    with concurrent.futures.ThreadPoolExecutor(args.workers) as ex:
        for i, msg in enumerate(ex.map(sub, todo)):
            print(i, msg, flush=True)


if __name__ == "__main__":
    main()
