"""Issue 1 population check: for every photodiode block, compare estimated display times with
the vsync *rising* edge of the same frame (written by TaskControl before draw+flip) as well as
the falling edge (written after flip() returns).
Output: vsync_rise_check.csv (one row per block), vsync_rise_check.log"""
import concurrent.futures, sys
import numpy as np, pandas as pd
from common import BASE, lines, baseline_block

B = pd.read_csv("../display-times-move/qc_blocks.csv")
P = B[B.source == "photodiode"]

def one(folder):
    out = []
    try:
        L = lines(folder)
    except Exception as e:
        return [dict(folder=folder, error=repr(e))]
    vr = L["v_rise"]
    for blk in P[P.folder == folder].block:
        v, d = baseline_block(folder, blk)
        i = np.searchsorted(vr, v) - 1          # last rising edge before each falling edge
        r = vr[i]
        pulse = (v - r) * 1e3
        lat_f = (d - v) * 1e3
        lat_r = (d - r) * 1e3
        neg = lat_f < 0
        out.append(dict(folder=folder, block=blk, n=len(v),
            pulse_med=np.median(pulse), pulse_p999=np.percentile(pulse, 99.9), n_pulse_gt_1p5_frames=int((pulse > 25).sum()),
            n_disp_lt_fall=int(neg.sum()),
            n_disp_lt_rise=int((lat_r < 0).sum()), n_disp_lt_rise_15ms=int((lat_r < 15).sum()),
            n_neg_with_long_pulse=int((neg & (pulse > 25)).sum()),
            min_lat_rise=lat_r.min(), med_lat_rise=np.median(lat_r)))
    return out

if __name__ == "__main__":
    folders = sorted(P.folder.unique())
    rows = []
    with concurrent.futures.ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as ex:
        for r in ex.map(one, folders):
            rows.extend(r)
    df = pd.DataFrame(rows); df.to_csv("vsync_rise_check.csv", index=False)
    print("blocks", len(df), "errors", df.get("error", pd.Series(dtype=str)).notna().sum())
    print("frames with display < vsync falling:", int(df.n_disp_lt_fall.sum()), "in", int((df.n_disp_lt_fall > 0).sum()), "blocks")
    print("  ...of which the vsync pulse (rise->fall) is > 25 ms:", int(df.n_neg_with_long_pulse.sum()))
    print("frames with display < vsync rising:", int(df.n_disp_lt_rise.sum()), "in", int((df.n_disp_lt_rise > 0).sum()), "blocks")
    print("frames with display < vsync rising + 15 ms:", int(df.n_disp_lt_rise_15ms.sum()), "in", int((df.n_disp_lt_rise_15ms > 0).sum()), "blocks")
    x = df[df.folder != "DRpilot_733891_20240918"]
    print("excluding 733891_20240918: <fall", int(x.n_disp_lt_fall.sum()), "| <fall & long pulse", int(x.n_neg_with_long_pulse.sum()),
          "| <rise", int(x.n_disp_lt_rise.sum()), "| <rise+15ms", int(x.n_disp_lt_rise_15ms.sum()), "in", int((x.n_disp_lt_rise_15ms > 0).sum()), "blocks")
    print("median vsync pulse width (ms): median over blocks", df.pulse_med.median(), "; blocks with any pulse > 25 ms:", int((df.n_pulse_gt_1p5_frames > 0).sum()), "frames:", int(df.n_pulse_gt_1p5_frames.sum()))
    pd.set_option("display.width", 250)
    print(df[df.n_disp_lt_rise_15ms > 0].sort_values("n_disp_lt_rise_15ms", ascending=False).head(25).to_string())
