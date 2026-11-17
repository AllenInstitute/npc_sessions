"""Restrict sync lick onsets to the task block time range and compare with stim lickFrames
(both excluding invalid intervals). Companion to event_latency_check.py.

Run: ../../../npc_sync/.venv/Scripts/python.exe lick_count_check.py > lick_count_check.log 2>&1
"""

import logging
import pathlib
import sys

import numpy as np

import npc_stim
import npc_sync

logging.disable(logging.WARNING)
ROOT = pathlib.Path(__file__).resolve().parents[3] / "npc_sync" / "experiments" / "vsync-frame-count"
sys.path.insert(0, str(ROOT))
from repair import reconcile_replay  # noqa: E402

tot_s = tot_f = tot_unmatched = 0
for d in sorted(p for p in (ROOT / "data").iterdir() if p.is_dir()):
    s = npc_sync.SyncDataset(next(d.glob("*T*.h5")))
    iv = list(s.invalid_intervals)
    h5 = npc_stim.get_h5_stim_data(next(d.glob("DynamicRouting1_*.hdf5")))
    n = npc_stim.get_total_stim_frames(h5)
    st = (npc_stim.get_stim_start_time(h5) - s.start_time).total_seconds()
    b = s.vsync_times_in_blocks
    v = b[min(range(len(b)), key=lambda i: abs(b[i][0] - st))]
    if len(v) != n:
        if not iv:
            continue
        v = reconcile_replay(v, h5["frameIntervals"][:], iv).times
    on = s.get_rising_edges("lick_sensor", units="seconds")
    m = (on >= v[0]) & (on <= v[-1])
    for a, z in iv:
        m &= ~((on >= a) & (on < z))
    on = on[m]
    lf = np.asarray(h5["lickFrames"][()], int).ravel()
    lf = lf[(lf >= 1) & (lf < len(v))]
    rt = v[lf - 1]
    keep = np.ones(len(rt), bool)
    for a, z in iv:
        keep &= ~((rt >= a) & (rt < z))
    rt = rt[keep]
    # sync onsets with no stim lickFrame read within the following 2 frames
    j = np.searchsorted(rt, on, side="left")
    unmatched = int(np.sum((j >= len(rt)) | (np.take(rt, np.minimum(j, len(rt) - 1)) - on > 0.034)))
    tot_s += on.size
    tot_f += rt.size
    tot_unmatched += unmatched
    print(
        f"{d.name}: sync lick onsets in task block={on.size}, stim lickFrames={rt.size}, "
        f"sync onsets with no lickFrame read within 34 ms={unmatched}"
    )
print(f"ALL: sync={tot_s} stim={tot_f} unmatched sync onsets={tot_unmatched} ({tot_unmatched / tot_s:.3f})")
