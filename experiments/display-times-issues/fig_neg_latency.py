import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from common import plot_window
df = pd.read_csv("neg_latency.csv")
ex = [("DRpilot_744279_20250116", 1, 1827), ("DRpilot_702131_20240226", 3, 73754),
      ("DRpilot_807082_20250930", 3, 81435), ("DRpilot_813586_20251205", 5, 27876)]
fig, axes = plt.subplots(len(ex), 1, figsize=(11, 2.6 * len(ex)))
for ax, (f, b, i) in zip(axes, ex):
    box = df[(df.folder == f) & (df.block == b)].box.iloc[0]
    plot_window(ax, f, b, i, title=f"{f} block {b} frame {i} (diode box interval {int(box)}); red arrows: display < vsync")
axes[0].legend(fontsize=7, ncol=5, loc="upper left", bbox_to_anchor=(0, 1.35))
fig.tight_layout(); fig.savefig("figs/neg_latency_examples.png", dpi=120)
