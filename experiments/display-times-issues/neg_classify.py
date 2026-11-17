"""Issue 1: classify frames with display < vsync falling edge, using the vsync rising edge of the
same frame (TaskControl writes it before draw + flip, so a frame can't be displayed before
rising + ~15 ms (MIN_VSYNC_DIODE_FLIP_SEPARATION_SEC)). Output: neg_classify.csv, neg_classify.log"""
import numpy as np, pandas as pd
from common import lines, baseline_block
df = pd.read_csv("neg_latency.csv")
rows = []
for (folder, blk), g in df.groupby(["folder", "block"]):
    L = lines(folder); v, d = baseline_block(folder, blk)
    vr = L["v_rise"]; r = vr[np.searchsorted(vr, v) - 1]
    for i in g.idx:
        fall_delayed = i + 1 < len(v) and (v[i + 1] - v[i]) < 0.5 / 59.95   # next falling edge < half a frame later
        prev_fall_delayed = i >= 1 and (v[i] - v[i - 1]) < 0.5 / 59.95 and False
        lat_r = (d[i] - r[i]) * 1e3
        rows.append(dict(folder=folder, block=blk, idx=i, box=g.box.iloc[0], stim=g.stim.iloc[0],
            lat_fall=(d[i] - v[i]) * 1e3, lat_rise=lat_r, pulse=(v[i] - r[i]) * 1e3,
            next_fall_within_half_frame=bool(fall_delayed),
            prev_frame_pulse=(v[i - 1] - r[i - 1]) * 1e3 if i else np.nan,
            klass="733891_20240918 b4 (degraded diode)" if folder.endswith("733891_20240918") else
                  ("consistent: display >= rise + 15 ms" if lat_r >= 15 else "impossible: display < rise + 15 ms")))
x = pd.DataFrame(rows); x.to_csv("neg_classify.csv", index=False)
pd.set_option("display.width", 200)
print(x.groupby("klass").agg(frames=("idx", "size"), blocks=("block", lambda s: x.loc[s.index, ["folder", "block"]].drop_duplicates().shape[0])))
y = x[~x.folder.str.endswith("733891_20240918")]
print("\nby class x box:"); print(pd.crosstab(y.klass, y.box))
print("\nconsistent frames: long vsync pulse (>25 ms) fraction", (y[y.klass.str.startswith("cons")].pulse > 25).mean().round(3),
      "| next falling edge < 0.5 frame later fraction", y[y.klass.str.startswith("cons")].next_fall_within_half_frame.mean().round(3))
imp = y[y.klass.str.startswith("imp")]
print("impossible frames: previous frame has long pulse (>25 ms):", int((imp.prev_frame_pulse > 25).sum()), "of", len(imp))
print("impossible frames per block:", imp.groupby(["folder", "block"]).size().describe()[["count", "mean", "max"]].to_dict())
print("impossible frames lat_rise ms p50/min:", imp.lat_rise.median().round(1), imp.lat_rise.min().round(1))
