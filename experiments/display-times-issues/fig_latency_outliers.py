import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from common import plot_window, baseline_block
B = pd.read_csv("../display-times-move/qc_blocks.csv"); P = B[B.source == "photodiode"]
fig = plt.figure(figsize=(12, 11))
gs = fig.add_gridspec(4, 2)
ax = fig.add_subplot(gs[0, :])
P = P.assign(year=P.folder.str[-8:-4], kind=np.where(P.stim.str.startswith("Spont"), "Spontaneous*", "other"))
for k, (kind, g) in enumerate(P.groupby("kind")):
    ax.scatter(pd.to_datetime(g.folder.str[-8:]), g.lat_ms_median, s=6, label=kind, alpha=0.6)
ax.axhline(27.66, c="k", lw=0.5); ax.axhline(27.66 + 1000 / 59.95, c="k", lw=0.5, ls="--")
ax.set_ylabel("block median latency\n(display - vsync, ms)"); ax.set_ylim(20, 60); ax.legend(fontsize=8)
ax.set_title("median vsync->display latency per photodiode block (1 point clipped: 702131_20240227 b5, 110 ms); dashed = +1 frame", fontsize=9)
ex = [("DRpilot_662892_20230821", 2, "Spontaneous, 44 ms"), ("DRpilot_662892_20230821", 1, "same session, OptoTagging/other block, 27.6 ms"),
      ("DRpilot_703333_20240408", 2, "Spontaneous, 44 ms"), ("DRpilot_702131_20240227", 5, "Spontaneous, 131 frames, 110 ms")]
for k, (f, b, t) in enumerate(ex):
    a = fig.add_subplot(gs[1 + k // 2, k % 2])
    plot_window(a, f, b, 0, before=0, after=6, title=f"{f} b{b} start: {t}")
    a.set_xlim(-40, 140)
a = fig.add_subplot(gs[3, :])
for f, b, _ in ex[:1] + ex[2:3]:
    v, d = baseline_block(f, b)
    a.plot((d - v) * 1e3, lw=0.5, label=f"{f} b{b}")
a.set_ylabel("latency (ms)"); a.set_xlabel("frame"); a.legend(fontsize=7); a.set_ylim(20, 70)
fig.tight_layout(); fig.savefig("figs/latency_outliers.png", dpi=110)
