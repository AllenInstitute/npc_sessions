"""Issue 4: blocks rejected because the diode-flip count differs from the vsync count by > 5.
For each, count diode flips (after the same filtering as get_display_times) in each frame's
window [vsync_k + 15 ms, vsync_k+1 + 15 ms) and summarise where extra/missing flips are.
Output: flip_count.csv, flip_count.log, figs/flip_count_cumulative.png"""
import concurrent.futures, json
import numpy as np, pandas as pd
import npc_stim.display_times as dt
from common import sync, stim_paths

B = pd.read_csv("../display-times-move/qc_blocks.csv")
U = B[B.reason.fillna("").str.contains("diode flips")]

def one(folder):
    s = sync(folder)
    vb = s.vsync_times_in_blocks
    r, fa = dt.get_diode_edges_in_blocks(s)
    rows, curves = [], {}
    for blk in U[U.folder == folder].block:
        v, a, b = vb[blk], r[blk], fa[blk]
        lo, hi = v[0] + dt.MIN_VSYNC_DIODE_FLIP_SEPARATION_SEC, v[-1] + dt.MAX_VSYNC_DIODE_FLIP_SEPARATION_SEC
        f = np.sort(np.concatenate([a[(a > lo) & (a < hi)], b[(b > lo) & (b < hi)]]))
        f, _, nrem = dt.remove_anomalous_diode_flips(f, 1, True)
        edges = np.append(v + 0.015, np.inf)
        c = np.histogram(f, bins=edges)[0]
        n = len(v)
        ex = np.cumsum(c - 1)
        q = lambda x: int(x[-1]) if len(x) else 0
        rows.append(dict(folder=folder, block=blk, stim=U[(U.folder == folder) & (U.block == blk)].stim.iloc[0].split("_")[0],
            n_vsync=n, n_flips=len(f), excess=len(f) - n, removed_pairs=nrem,
            frames_0_flips=int((c == 0).sum()), frames_2plus_flips=int((c >= 2).sum()),
            excess_first_1pct=int(ex[n // 100]), excess_last_1pct=int(ex[-1] - ex[-n // 100]),
            longest_run_0=int(max((len(x) for x in np.split(c, np.where(c != 0)[0]) ), default=0)),
            frac_frames_ok=float((c == 1).mean())))
        curves[f"{folder}|{blk}"] = (np.linspace(0, 1, 500), np.interp(np.linspace(0, n - 1, 500), np.arange(n), ex))
    return rows, curves

if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rows, curves = [], {}
    with concurrent.futures.ProcessPoolExecutor(6) as ex:
        for r, c in ex.map(one, sorted(U.folder.unique())):
            rows += r; curves.update(c)
    df = pd.DataFrame(rows).sort_values(["stim", "excess"]); df.to_csv("flip_count.csv", index=False)
    pd.set_option("display.width", 250); print(df.to_string(index=False))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    groups = {"OptoTagging": axes[0], "DynamicRouting1": axes[1]}
    for k, (x, y) in curves.items():
        stim = df[(df.folder + "|" + df.block.astype(str)) == k].stim.iloc[0]
        ax = groups.get(stim, axes[2])
        ax.plot(x, y, lw=0.8, label=k.replace("DRpilot_", "").replace("|", " b"))
    for ax, t in zip(axes, ["OptoTagging", "DynamicRouting1", "Spontaneous*"]):
        ax.set_title(t, fontsize=9); ax.set_xlabel("fraction of block"); ax.axhline(0, c="k", lw=0.5)
        ax.legend(fontsize=5, ncol=2)
    axes[0].set_ylabel("cumulative (diode flips - vsyncs)")
    axes[1].set_yscale("symlog", linthresh=10); axes[2].set_yscale("symlog", linthresh=10)
    fig.tight_layout(); fig.savefig("figs/flip_count_cumulative.png", dpi=110)
