"""Compare get_display_times blocks from a fix run (out_issues/<label>) with baseline
(out/baseline, byte-identical to the move branch). Per block: source change, array equality,
median latency before/after, n frames before vsync falling edge.
Output: compare_<label>.csv and printed summary."""
import json, pathlib, sys
import numpy as np, pandas as pd
label = sys.argv[1]
NEW = pathlib.Path("C:/data/t2p_qc/out_issues") / label
BASE = pathlib.Path("C:/data/t2p_qc/out/baseline")
B = pd.read_csv("../display-times-move/qc_blocks.csv")
rows = []
for jf in sorted(NEW.glob("*.json")):
    m = json.loads(jf.read_text()); f = m["folder"]
    bm = json.loads((BASE / f"{f}.json").read_text())
    if "error" in m or "sources" not in m:
        rows.append(dict(folder=f, block=-1, change="session_error_new:" + m.get("error", {}).get("type", "?"),
                         base_error=bm.get("A_error", {}).get("type") if "A_sources" not in bm else None)); continue
    if "A_sources" not in bm:
        rows.append(dict(folder=f, block=-1, change="session_error_base:" + str(bm.get("A_error", bm.get("error")))[:80])); continue
    zn, zb = np.load(NEW / f"{f}.npz"), np.load(BASE / f"{f}.npz")
    for i, src in enumerate(m["sources"]):
        k = f"A_block{i}"; v = zb[f"vsync_block{i}"]
        a, b = (zn[k] if k in zn else None), (zb[k] if k in zb else None)
        r = dict(folder=f, block=i, stim=(m["stim_paths"][i] or "").split("_")[0], box=m["box"][i],
                 source_base=bm["A_sources"][i], source_new=src, reason_new=m["reasons"][i])
        if a is not None and b is not None:
            r["identical"] = a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)
        else:
            r["identical"] = a is None and b is None
        for tag, x in (("base", b), ("new", a)):
            if x is not None and len(x) == len(v):
                r[f"lat_med_{tag}"] = np.median(x - v) * 1e3; r[f"n_before_vsync_{tag}"] = int((x < v).sum())
        if a is not None and b is not None and a.shape == b.shape:
            r["max_abs_diff_ms"] = np.abs(a - b).max() * 1e3; r["n_frames_changed"] = int((a != b).sum())
        r["change"] = "identical" if r["identical"] else f"{r['source_base']}->{src}" + ("" if r["source_base"] != src else " (times changed)")
        rows.append(r)
df = pd.DataFrame(rows); df.to_csv(f"compare_{label}.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120); pd.set_option("display.max_rows", 200)
print("sessions:", df.folder.nunique(), "| blocks:", int((df.block >= 0).sum()))
print(df.change.value_counts().to_string())
ch = df[(df.block >= 0) & (df.change != "identical")]
cols = [c for c in ["folder", "block", "stim", "box", "change", "lat_med_base", "lat_med_new", "n_before_vsync_base", "n_before_vsync_new", "n_frames_changed", "max_abs_diff_ms", "reason_new"] if c in ch]
print(ch[cols].to_string())
