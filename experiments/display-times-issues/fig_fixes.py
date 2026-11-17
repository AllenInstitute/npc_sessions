"""Before/after figures for the fixes (baseline vs out_issues/<label>)."""
import sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
import common
from common import plot_window
label = sys.argv[1] if len(sys.argv) > 1 else "fixD"
NEW = common.pathlib.Path("C:/data/t2p_qc/out_issues") / label
# issue 1: box-3 placement before/after
ex = [("DRpilot_807082_20250930", 3, 81435), ("DRpilot_813586_20251205", 5, 27876)]
fig, axes = plt.subplots(2, 2, figsize=(14, 5.5))
for row, (f, b, i) in zip(axes, ex):
    plot_window(row[0], f, b, i, before=4, after=5, title=f"{f} b{b} frame {i}: baseline")
    plot_window(row[1], f, b, i, before=4, after=5, title=f"{f} b{b} frame {i}: {label}", display=np.load(NEW / f"{f}.npz")[f"A_block{b}"])
axes[0][0].legend(fontsize=6, ncol=5, loc="lower left", bbox_to_anchor=(0, 1.12))
fig.tight_layout(); fig.savefig("figs/fix_box3_placement.png", dpi=110)

# issue 2: block median latency before/after
c = pd.read_csv(f"compare_{label}.csv"); c = c[c.block >= 0]
fig, ax = plt.subplots(figsize=(6, 4.5))
ok = c.lat_med_base.notna()
ax.scatter(c.lat_med_base[ok], c.lat_med_new.fillna(np.nan)[ok], s=6)
rej = c[ok & c.lat_med_new.isna()]
ax.scatter(rej.lat_med_base, np.full(len(rej), 22), marker="x", c="r", s=20, label="now unavailable (plotted at 22 ms)")
ax.plot([20, 115], [20, 115], c="k", lw=0.5); ax.set_xlim(25, 115); ax.set_ylim(20, 50)
ax.set_xlabel("baseline block median latency (ms)"); ax.set_ylabel(f"{label} block median latency (ms)")
ax.axhline(27.66 + 8.34, c="0.5", ls="--", lw=0.5); ax.legend(fontsize=7)
ax.set_title("photodiode blocks: median (display - vsync falling edge)", fontsize=9)
fig.tight_layout(); fig.savefig("figs/fix_latency_before_after.png", dpi=110)
