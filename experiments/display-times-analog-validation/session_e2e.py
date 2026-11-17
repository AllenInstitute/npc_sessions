"""End-to-end check in npc_sessions with the branch (read-only on the cache).
    PYTHONPATH=<wt npc_stim_fix>/src;<npc_sync>/src  <npc_sessions .venv>/python session_e2e.py <folder> ...
For each session: DynamicRoutingSession(local cache folder) -> stim_frame_times, is_photodiode
provenance, and the trials table of the task; frame times are compared with the fixE run
(get_display_times) for the same blocks."""
import json, pathlib, sys, time, traceback, warnings
import numpy as np
warnings.simplefilter("ignore")
import npc_sessions, npc_stim, npc_sync
FIX = pathlib.Path("C:/data/t2p_qc/out_issues/fixE")
print("npc_stim", npc_stim.__file__, "\nnpc_sync", npc_sync.__file__, "\nnpc_sessions", npc_sessions.__file__)
for folder in sys.argv[1:]:
    t0 = time.time()
    print(f"===== {folder}")
    try:
        s = npc_sessions.DynamicRoutingSession(pathlib.Path("C:/data/t2p_cache") / folder)
        fj = json.loads((FIX / f"{folder}.json").read_text()); fz = np.load(FIX / f"{folder}.npz")
        name_to_block = {pathlib.Path(p).stem: i for i, p in enumerate(fj["stim_paths"]) if p}
        sdt = s._stim_display_times
        for k, v in sdt.items():
            if isinstance(v, Exception):
                print(f"  {k}: EXCEPTION {type(v).__name__}: {str(v)[:150]}"); continue
            b = name_to_block.get(k)
            ref = fz[f"A_block{b}"] if b is not None and f"A_block{b}" in fz else None
            same = None if ref is None or v.display_times is None else bool(np.array_equal(ref, v.display_times))
            print(f"  {k}: source={v.source} block={v.block_index} n={None if v.display_times is None else len(v.display_times)} "
                  f"identical_to_fixE_A={same}")
        tr = s.trials[:]  # hdmf DynamicTable -> pandas DataFrame
        cols = [c for c in ("start_time", "stim_start_time", "response_time") if c in tr.columns]
        print(f"  trials: {len(tr)} rows, {len(tr.columns)} columns; NaN counts {dict((c, int(tr[c].isna().sum())) for c in cols)}")
    except Exception:
        traceback.print_exc(limit=6)
    print(f"  [{time.time() - t0:.0f} s]")
