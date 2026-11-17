"""Issue 1: frames whose estimated display time precedes their vsync.
Uses baseline outputs (byte-identical to new) in C:/data/t2p_qc/out/baseline.
Output: neg_latency.csv (one row per negative frame), neg_latency.log"""
import pathlib, numpy as np, pandas as pd
OUT = pathlib.Path("C:/data/t2p_qc/out/baseline")
B = pd.read_csv("../display-times-move/qc_blocks.csv")
P = B[(B.source == "photodiode") & (B.n_display_before_vsync > 0)]
rows = []
for (folder, blk), r in P.groupby(["folder", "block"]):
    z = np.load(OUT / f"{folder}.npz")
    d, v = z[f"A_block{blk}"], z[f"vsync_block{blk}"]
    lat = (d - v) * 1e3
    med = np.median(lat)
    dv = np.diff(v) * 1e3
    dd = np.diff(d) * 1e3
    for i in np.where(lat < 0)[0]:
        rows.append(dict(folder=folder, block=blk, stim=r.stim.iloc[0].split("_")[0],
            box=r.diode_box_frame_interval.iloc[0], n=len(d), idx=i, lat=lat[i], block_median=med,
            lat_prev=lat[i-1] if i else np.nan, lat_next=lat[i+1] if i+1 < len(d) else np.nan,
            vsync_int_before=dv[i-1] if i else np.nan, vsync_int_after=dv[i] if i < len(dv) else np.nan,
            disp_int_before=dd[i-1] if i else np.nan, disp_int_after=dd[i] if i < len(dd) else np.nan,
            phase=(i % 3) if r.diode_box_frame_interval.iloc[0] == 3 else (i % 2)))
df = pd.DataFrame(rows); df.to_csv("neg_latency.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
x = df[df.folder != "DRpilot_733891_20240918"]
print("all negative frames", len(df), "blocks", df.groupby(["folder","block"]).ngroups)
print("excluding 733891_20240918:", len(x))
print(x.groupby("box").agg(n=("lat","size"), lat_med=("lat","median"), lat_min=("lat","min"),
      vin_before=("vsync_int_before","median"), vin_after=("vsync_int_after","median"),
      din_before=("disp_int_before","median"), din_after=("disp_int_after","median")))
print("box3 phase counts", x[x.box==3].phase.value_counts().to_dict())
print("box1 phase counts", x[x.box==1].phase.value_counts().to_dict())
print(x.describe().T[["mean","50%","min","max"]])
print(x.sort_values("lat").head(15).to_string())
