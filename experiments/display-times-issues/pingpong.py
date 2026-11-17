"""Issue 4b: diode flips during long vsync intervals ("ping-pong"). For the blocks with excess
flips, tabulate for each frame window with >=2 flips (and no 0-flip neighbour): vsync interval in
refreshes, flips in window, flips in the next window. Also count such windows in *accepted*
photodiode blocks of the same sessions. Output: pingpong.log, figs/pingpong_examples.png"""
import collections
import numpy as np, pandas as pd
import npc_stim.display_times as dt
from common import sync, plot_window
df = pd.read_csv("flip_extra.csv")
B = pd.read_csv("../display-times-move/qc_blocks.csv")
tab = collections.Counter(); acc = collections.Counter()
def windows(folder, blk):
    s = sync(folder); v = s.vsync_times_in_blocks[blk]
    r, fa = dt.get_diode_edges_in_blocks(s); a, b = r[blk], fa[blk]
    lo, hi = v[0] + 0.015, v[-1] + 5 / 59.95
    f = np.sort(np.concatenate([a[(a > lo) & (a < hi)], b[(b > lo) & (b < hi)]]))
    f, _, _ = dt.remove_anomalous_diode_flips(f, 1, True)
    c = np.histogram(f, bins=np.append(v + 0.015, np.inf))[0]
    nref = np.round(np.diff(v, append=v[-1] + 1 / 59.95) * 59.95).astype(int)
    return c, nref
for (folder, blk), _ in df.groupby(["folder", "block"]):
    c, nref = windows(folder, blk)
    for k in np.where(c >= 2)[0]:
        if k + 1 < len(c) and c[k - 1] != 0 and c[k + 1] != 0:
            tab[(int(nref[k]), int(c[k]), int(c[k + 1]))] += 1
print("rejected blocks with excess flips: (vsync interval in refreshes, flips in window, flips in next window): count")
for k, n in sorted(tab.items(), key=lambda x: -x[1])[:25]:
    print(" ", k, n)
# accepted box-1 photodiode blocks in the same sessions
for folder in df.folder.unique():
    for blk in B[(B.folder == folder) & (B.source == "photodiode") & (B.diode_box_frame_interval == 1)].block:
        c, nref = windows(folder, blk)
        long = nref >= 3
        acc["long windows"] += int(long.sum())
        acc["long windows with flips == refreshes"] += int((long & (c == nref)).sum())
        acc["long windows with 1 flip"] += int((long & (c == 1)).sum())
print("accepted box-1 blocks in the same sessions:", dict(acc))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ex = [("DRpilot_699847_20240415", 1, 306, "OptoTagging: 10-refresh vsync interval with 10 diode flips"),
      ("DRpilot_716718_20240730", 0, 78612, "DynamicRouting1: 3-refresh vsync interval with 3 diode flips")]
fig, axes = plt.subplots(2, 1, figsize=(11, 5.5))
for ax, (f, b, i, t) in zip(axes, ex):
    plot_window(ax, f, b, i, before=2, after=3, title=f"{f} block {b} frame {i}: {t} (block rejected; display times shown are not computed)")
axes[0].legend(fontsize=7, ncol=5, loc="lower right")
fig.tight_layout(); fig.savefig("figs/pingpong_examples.png", dpi=110)
