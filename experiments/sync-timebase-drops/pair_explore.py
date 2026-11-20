"""What are the complementary barcode-interval pairs in replay files? Print barcode edges and
replayed intervals around each flagged window (703880_20240418, 708020_20240517)."""
import json, logging, pathlib, sys
import numpy as np, npc_sync
logging.disable(logging.WARNING)
for path in ("C:/data/t2p_cache/DRpilot_703880_20240418/20240418T130656.h5", "C:/data/t2p_cache/DRpilot_708020_20240517/20240517T100128.h5"):
    s = npc_sync.SyncDataset(path, on_invalid="ignore")
    r = s.get_rising_edges("barcode_ephys", units="seconds"); f = s.get_falling_edges("barcode_ephys", units="seconds")
    starts = r[np.r_[True, np.diff(r) > 20]]
    print(path); print(" discont", [(round(a,3), round(z,3), round(d,4)) for a,z,d in s.timebase_discontinuities])
    inv = s.invalid_intervals
    done=set()
    for a, z, d in s.timebase_discontinuities:
        for t in (a, z):
            if round(t,2) in done: continue
            done.add(round(t,2))
            near = [(round(x,3), round(y,3)) for x, y in inv if y > t - 6 and x < t + 6]
            e = r[(r > t - 4) & (r < t + 4)]; ef = f[(f > t - 4) & (f < t + 4)]
            print(f"  barcode onset {t:.4f}: n rising in +-4 s {len(e)}, first {e[0]:.4f} last falling {ef[-1]:.4f}; replayed near: {near}")
            i = np.searchsorted(starts, t - 1e-6)
            print(f"     onsets around: {np.round(starts[max(i-2,0):i+3],4).tolist()} intervals {np.round(np.diff(starts[max(i-2,0):i+3]),4).tolist()}")
