"""Issue 5: blocks rejected with 'Too many frames are misaligned with vsyncs'. Re-run the estimator
with check_vsyncs=False (as get_display_times otherwise would), then repeat its long-interval
alignment vote and look at the misaligned long intervals.
Output: misaligned.log, figs/misaligned.png"""
import numpy as np, pandas as pd, scipy.signal
import npc_stim.display_times as dt
from common import sync, stim_paths
B = pd.read_csv("../display-times-move/qc_blocks.csv")
U = B[B.reason.fillna("").str.contains("misaligned")]
FI = 1 / 59.95
res = {}
for _, r in U.iterrows():
    s = sync(r.folder); v = s.vsync_times_in_blocks[r.block]
    ri, fa = dt.get_diode_edges_in_blocks(s); a, b = ri[r.block], fa[r.block]
    lo, hi = v[0] + 0.015, v[-1] + 5 / 59.95
    a, b = a[(a > lo) & (a < hi)], b[(b > lo) & (b < hi)]
    on = a[0] < b[0]
    f, on, _ = dt.remove_anomalous_diode_flips(np.sort(np.concatenate([a, b])), 1, on)
    rise, fall = (f[::2], f[1::2]) if on else (f[1::2], f[::2])
    d = dt.get_frame_display_times(v, rise, fall, 1, check_lengths=False, check_vsyncs=False)
    med = np.median(np.diff(rise)) / 2
    long_d = np.where(np.diff(d) > 1.1 * med)[0]
    dv = np.diff(v + dt.AVERAGE_VSYNC_TO_DIODE_FLIP_TIME)
    mis = []
    for i in long_d:
        sh = [x for x in list(range(0, -6, -1)) + list(range(1, 6)) if 0 < i + x < len(dv)]
        k = int(np.argmax([dv[i + x] for x in sh])); mis.append(-sh[k] if dv[i + sh[k]] >= 1.1 * med else 99)
    mis = np.array(mis); sm = scipy.signal.medfilt(mis, 3)
    n_long_v = int((dv > 1.1 * med).sum())
    # vsync long intervals followed by a short one (< 0.5 frame): delayed falling edge signature
    delayed = int(((dv[:-1] > 1.1 * med) & (dv[1:] < 0.5 * FI)).sum())
    res[(r.folder, r.block)] = (long_d, mis, len(v))
    first_bad = long_d[np.argmax(sm != 0)] if np.any(sm != 0) else None
    print(f"{r.folder} b{r.block}: frames {len(v)}, display {len(d)}, long display intervals {len(long_d)}, long vsync intervals {n_long_v} "
          f"(of which delayed-falling-edge signature {delayed}); vote: 0={np.mean(sm == 0):.2f}, +1={np.mean(sm == 1):.2f}, -1={np.mean(sm == -1):.2f}, "
          f"other={np.mean(~np.isin(sm, [0, 1, -1])):.2f}; first sustained misalignment at frame {first_bad} "
          f"({first_bad / len(v):.2f} of block)" if first_bad is not None else "")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(11, 4))
for k, (ld, mis, n) in res.items():
    m = np.where(np.abs(mis) > 6, np.nan, mis)
    ax.plot(ld / n, scipy.signal.medfilt(np.nan_to_num(m, nan=7), 3), lw=0.7, label=f"{k[0][8:]} b{k[1]}", drawstyle="steps-post")
ax.set_xlabel("position in block (fraction of frames)"); ax.set_ylabel("smoothed misalignment of long display\nintervals vs long vsync intervals (frames)")
ax.set_title("blocks rejected with 'Too many frames are misaligned' (7 = no long vsync within ±5)", fontsize=9); ax.legend(fontsize=6, ncol=2)
fig.tight_layout(); fig.savefig("figs/misaligned.png", dpi=110)
