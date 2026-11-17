import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from common import plot_window, baseline_block
f, b = "DRpilot_733891_20240918", 4
v, d = baseline_block(f, b)
fig, axes = plt.subplots(2, 1, figsize=(11, 6))
axes[0].plot((d - v) * 1e3, lw=0.5); axes[0].set_ylabel("display - vsync (ms)"); axes[0].set_xlabel("frame")
axes[0].set_title(f"{f} block {b} (DynamicRouting1, box interval 1): latency drops by 2 frames at frame 213257 and stays", fontsize=9)
plot_window(axes[1], f, b, 213257, before=8, after=6)
axes[1].legend(fontsize=7, ncol=5, loc="lower right")
fig.tight_layout(); fig.savefig("figs/neg_latency_733891.png", dpi=110)
