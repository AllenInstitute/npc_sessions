"""Issue 6: blocks with no matched stim file, and the block whose vsync count != stim frames.
Lists each session's stim files (frames, start time) and vsync blocks (n, start time).
Output: unmatched.log"""
import datetime, h5py, numpy as np, pandas as pd
import npc_stim.stim
from common import sync, stim_paths, sync_path
B = pd.read_csv("../display-times-move/qc_blocks.csv")
U = B[B.reason.fillna("").str.contains("no stim file|expected")]
for folder, g in U.groupby("folder"):
    s = sync(folder)
    print(f"== {folder} (sync {sync_path(folder).name}, start {s.start_time}); unmatched/mismatched blocks {g.block.tolist()}")
    for i, v in enumerate(s.vsync_times_in_blocks):
        print(f"   block {i}: {len(v)} vsyncs, t={v[0]:.1f}-{v[-1]:.1f} s ({s.start_time + datetime.timedelta(seconds=float(v[0]))})")
    for p in stim_paths(folder):
        try:
            with h5py.File(p) as h:
                n = npc_stim.stim.get_total_stim_frames(h)
                st = h["startTime"][()].decode() if "startTime" in h else "?"
        except Exception as e:
            n, st = f"ERR {type(e).__name__}", "?"
        print(f"   stim {p.name}: {n} frames, startTime {st}")
