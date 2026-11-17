"""Where in each block are frames whose estimated display time precedes their vsync?
Reads new-run outputs (identical to baseline for every block). Output: negative_latency_positions.log"""
import collections, json, pathlib
import numpy as np, pandas as pd
OUT = pathlib.Path("C:/data/t2p_qc/out/new")
B = pd.read_csv("qc_blocks.csv")
P = B[(B.source == "photodiode") & (B.n_display_before_vsync > 0)]
pos = collections.Counter(); per_block = collections.Counter()
for (folder, blk), _ in P.groupby(["folder", "block"]):
    z = np.load(OUT / f"{folder}.npz")
    d, v = z[f"A_block{blk}"], z[f"vsync_block{blk}"]
    idx = np.where(d < v)[0]
    kinds = ["first frame" if i == 0 else "last frame" if i == len(d) - 1 else "first 10" if i < 10 else "last 10" if i >= len(d) - 10 else "interior" for i in idx]
    pos.update(kinds)
    per_block[tuple(sorted(set(kinds)))] += 1
print("blocks:", len(P), "| frames:", sum(pos.values()))
print("frames by position:", dict(pos))
print("blocks by set of positions:", {"+".join(k): n for k, n in per_block.most_common()})
