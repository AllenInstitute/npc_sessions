"""npc_stim display times with and without the session's SyncPatch, for every cached session.

For each session in C:/data/t2p_cache (sync file: the largest, as in earlier QC):
- patch = npc_sessions.timing.build_session_sync_patch(raw sync, stim files) (branch code)
- A = npc_stim.get_display_times(SyncDataset(sync), stims)          (no patch)
- B = npc_stim.get_display_times(SyncDataset(sync, patch=patch), stims)
Files with replayed intervals raise in A (unpatched), as on main.

Output per session: C:/data/t2p_qc/out_phantom/<folder>.npz (A_block<i>, B_block<i>) and .json
(sources, reasons, diode records, patch summary, timings). Compare with compare_population.py.

Run: PYTHONPATH="../../../npc_sync/src;../../../npc_stim/src;../../src;." ../../.venv/Scripts/python.exe run_population.py [folder substrings...]
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import logging
import pathlib
import sys
import time
import traceback

import numpy as np

OUT = pathlib.Path("C:/data/t2p_qc/out_phantom")


def run(folder: str) -> str:
    import npc_stim
    import npc_sync

    import common
    from npc_sessions import timing

    logging.disable(logging.WARNING)
    t0 = time.time()
    rec: dict = {"folder": folder}
    arrays: dict[str, np.ndarray] = {}
    try:
        sync_path = common.sync_path(folder)
        stims = common.stim_paths(folder)
        rec["sync"] = sync_path.name
        rec["stims"] = [p.name for p in stims]
        raw = npc_sync.SyncDataset(sync_path, on_invalid="ignore")
        rec["invalid_intervals"] = [list(x) for x in raw.invalid_intervals]
        try:
            patch = timing.build_session_sync_patch(raw, stims)
            rec["patch_summary"] = patch.diode_summary()
            rec["diode_blocks"] = [b.to_dict() for b in patch.diode_blocks]
            rec["removed"] = {k: [r.to_dict() for r in v] for k, v in patch.removed.items()}
        except Exception:  # noqa: BLE001
            patch = None
            rec["patch_error"] = traceback.format_exc().splitlines()[-1]
        for label, sync in (("A", npc_sync.SyncDataset(sync_path)), ("B", npc_sync.SyncDataset(sync_path, patch=patch) if patch else None)):
            if sync is None:
                continue
            try:
                res = npc_stim.get_display_times(sync, stims)
            except Exception:  # noqa: BLE001
                rec[f"{label}_error"] = traceback.format_exc().splitlines()[-1]
                continue
            rec[f"{label}_sources"] = [r.source for r in res]
            rec[f"{label}_reasons"] = [r.reason for r in res]
            rec[f"{label}_stims"] = [r.stim_path.name if r.stim_path else None for r in res]
            rec[f"{label}_box"] = [r.diode_box_frame_interval for r in res]
            for r in res:
                if r.display_times is not None:
                    arrays[f"{label}_block{r.block_index}"] = r.display_times
                arrays[f"{label}_vsync{r.block_index}"] = r.vsync_times
    except Exception:  # noqa: BLE001
        rec["error"] = traceback.format_exc().splitlines()[-1]
    rec["seconds"] = round(time.time() - t0, 1)
    np.savez_compressed(OUT / f"{folder}.npz", **arrays)
    (OUT / f"{folder}.json").write_text(json.dumps(rec, indent=1, default=str))
    return f"{folder}: {rec['seconds']} s" + (f" ERROR {rec['error']}" if "error" in rec else "")


def main() -> None:
    import common

    OUT.mkdir(parents=True, exist_ok=True)
    folders = sorted(p.name for p in common.CACHE.iterdir() if p.is_dir())
    if len(sys.argv) > 1:
        folders = [f for f in folders if any(s in f for s in sys.argv[1:])]
    t0 = time.time()
    with open("run_population.log", "w") as log, cf.ProcessPoolExecutor(max_workers=10) as ex:
        for msg in ex.map(run, folders):
            log.write(msg + "\n")
            log.flush()
        log.write(f"done: {len(folders)} sessions in {time.time() - t0:.0f} s\n")
    print(f"{len(folders)} sessions in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
