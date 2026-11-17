"""Physical lower-bound check on display times from any run: a frame can't be displayed before its
vsync rising edge (frame signal raised before draw + flip) + 15 ms (MIN_VSYNC_DIODE_FLIP_SEPARATION_SEC).
    python eval_rise.py <label>   (label 'baseline' = C:/data/t2p_qc/out/baseline)
Output: eval_rise_<label>.csv + summary"""
import concurrent.futures, json, pathlib, sys
import numpy as np, pandas as pd
from common import lines
label = sys.argv[1]
D = pathlib.Path("C:/data/t2p_qc/out/baseline") if label == "baseline" else pathlib.Path("C:/data/t2p_qc/out_issues") / label
BASE = pathlib.Path("C:/data/t2p_qc/out/baseline")
def one(folder):
    m = json.loads((D / f"{folder}.json").read_text())
    srcs = m.get("A_sources", m.get("sources"))
    if not srcs:
        return []
    z, zb = np.load(D / f"{folder}.npz"), np.load(BASE / f"{folder}.npz")
    vr = lines(folder)["v_rise"]; out = []
    for i, s in enumerate(srcs):
        if s != "photodiode" or f"vsync_block{i}" not in zb:
            continue
        v, d = zb[f"vsync_block{i}"], z[f"A_block{i}"]
        r = vr[np.searchsorted(vr, v) - 1]
        out.append(dict(folder=folder, block=i, n=len(v), lat_med=np.median(d - v) * 1e3,
                        n_lt_fall=int((d < v).sum()), n_lt_rise15=int((d - r < 0.015).sum()), n_lt_rise23=int((d - r < 0.023).sum()), n_lt_rise=int((d < r).sum())))
    return out
if __name__ == "__main__":
    folders = sorted(p.stem for p in D.glob("*.json"))
    rows = []
    with concurrent.futures.ProcessPoolExecutor(10) as ex:
        for r in ex.map(one, folders):
            rows += r
    df = pd.DataFrame(rows); df.to_csv(f"eval_rise_{label}.csv", index=False)
    x = df[df.folder != "DRpilot_733891_20240918"]
    print(f"{label}: photodiode blocks {len(df)}; median-latency range {df.lat_med.min():.2f}-{df.lat_med.max():.2f} ms; blocks >35 ms: {(df.lat_med > 35).sum()}")
    for c in ["n_lt_fall", "n_lt_rise23", "n_lt_rise15", "n_lt_rise"]:
        print(f"  {c}: frames {int(df[c].sum())} in {(df[c] > 0).sum()} blocks | excl. 733891_20240918 b4: {int(x[c].sum())} in {(x[c] > 0).sum()} blocks")
