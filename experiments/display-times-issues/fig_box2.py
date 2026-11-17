"""Issue 3: diode-flip intervals in DRpilot_366122_20250109 (diodeBoxFrameInterval=2) vs the
anomaly threshold ((N-1) + 0.3 frames), and a box-3 session for comparison."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
import npc_stim.display_times as dt
from common import sync
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
for ax, (f, blks, n) in zip(axes, [("DRpilot_366122_20250109", (0, 1, 3), 2), ("DRpilot_366122_20260306", (0, 1), 3)]):
    s = sync(f); vb = s.vsync_times_in_blocks; r, fa = dt.get_diode_edges_in_blocks(s)
    for b in blks:
        v = vb[b]; x = np.sort(np.concatenate([r[b], fa[b]])); x = x[(x > v[0] + .015) & (x < v[-1] + .08)]
        ax.hist(np.diff(x) * 59.95, bins=np.arange(0, 6, 0.05), histtype="step", label=f"block {b}", log=True)
    thr = (n - 1) + 0.3
    ax.axvline(thr, c="r", lw=1, label=f"'anomalous' if < {thr} frames")
    ax.set_xlabel("interval between consecutive diode edges (frames)"); ax.set_title(f"{f} (diodeBoxFrameInterval={n})", fontsize=9)
    ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig("figs/box2_intervals.png", dpi=110)
