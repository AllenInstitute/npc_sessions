"""Robust view of ephys_barcode_residuals.jsonl -> ephys_barcode_robust.log.

Per device, from the stored (sync time t, endpoint-fit residual r) of value-paired barcodes:
  1. robust line r ~ a + b t (iteratively drop |r - fit| > 5 ms; 5 passes): what an all-barcode
     fit would leave.
  2. outliers: pairs > 5 ms from the robust line (value pairing errors, e.g. mis-decoded
     barcodes, or real timebase steps).
  3. endpoint error: max |a + b t| over the device's barcodes = how far the current
     first/last-barcode alignment is from the robust fit (> 1 ms = the current alignment is wrong).
"""

import numpy as np
import pandas as pd
import scipy.stats

pd.set_option("display.width", 250)
d = pd.read_json("ephys_barcode_residuals.jsonl", lines=True)
d = d[d.error.isna() & d.n_matched.notna()].copy()
d["dev"] = np.where(d.device.str.endswith("-AP"), "AP", np.where(d.device.str.endswith("-LFP"), "LFP", np.where(d.device.str.contains("NI-DAQ"), "NI-DAQ", "other")))
d["pop"] = np.where(d.n_discontinuities > 0, "discontinuities", np.where(d.n_invalid > 0, "invalid intervals only", "clean"))
rows = []
for _, r in d.iterrows():
    t, res = np.asarray(r.t, float), np.asarray(r.res_ms, float)
    slope, icpt, *_ = scipy.stats.theilslopes(res, t)  # robust start (outliers can be ~1e7 ms)
    c = np.array([slope, icpt])
    keep = np.abs(res - np.polyval(c, t)) <= 5.0
    for _ in range(5):
        if keep.sum() < 3:
            break
        c = np.polyfit(t[keep], res[keep], 1)
        keep = np.abs(res - np.polyval(c, t)) <= 5.0
    rr = res - np.polyval(c, t)
    if keep.sum() < 3:
        rows.append(dict(key=r.key, device=r.device, dev=r.dev, pop=r["pop"], n=len(t), n_outliers=int(len(t)),
                         robust_res_max_ms=np.nan, robust_res_p99_ms=np.nan, endpoint_error_ms=np.nan,
                         endpoint_res_max_ms=r.endpoint_res_max_ms, note="no robust linear fit"))
        continue
    rows.append(dict(key=r.key, device=r.device, dev=r.dev, pop=r["pop"], n=len(t), n_outliers=int((~keep).sum()),
                     robust_res_max_ms=float(np.max(np.abs(rr[keep]))), robust_res_p99_ms=float(np.quantile(np.abs(rr[keep]), 0.99)),
                     endpoint_error_ms=float(np.max(np.abs(np.polyval(c, t)))),
                     endpoint_res_max_ms=r.endpoint_res_max_ms))
x = pd.DataFrame(rows)
q = [0.5, 0.9, 0.99, 1.0]
print(f"devices {len(x)}, sessions {x.key.nunique()}")
if "note" in x:
    print("no robust linear fit:", x[x.note.notna()][["key", "device", "pop", "n"]].to_string())
print("\nrobust all-barcode fit, max |residual| per device (ms), quantiles across devices:")
t = x.groupby(["dev", "pop"]).robust_res_max_ms.quantile(q).unstack().round(3)
t["n"] = x.groupby(["dev", "pop"]).size()
print(t.to_string())
print("\ndevices with value-pairing outliers (> 5 ms from the robust fit):", (x.n_outliers > 0).sum(),
      "in", x[x.n_outliers > 0].key.nunique(), "sessions")
print(x[x.n_outliers > 0].groupby("key").agg(devices=("device", "size"), outliers=("n_outliers", "sum"), pop=("pop", "first")).to_string())
print("\ncurrent first/last-barcode alignment vs robust fit: max error over the recording (ms), quantiles:")
t = x.groupby(["dev", "pop"]).endpoint_error_ms.quantile(q).unstack().round(3)
print(t.to_string())
bad = x[x.endpoint_error_ms > 1.0]
print(f"\ndevices whose current alignment is > 1 ms from the robust fit somewhere: {len(bad)} in {bad.key.nunique()} sessions")
print(bad.sort_values("endpoint_error_ms")[["key", "device", "pop", "n", "n_outliers", "endpoint_error_ms", "robust_res_max_ms"]].to_string())
x.to_csv("ephys_barcode_robust.csv", index=False)

clean = x[(x.n_outliers == 0)]
print("\n---- devices with no pairing outliers (single linear segment):", len(clean), "of", len(x))
for col in ("robust_res_max_ms", "robust_res_p99_ms", "endpoint_error_ms"):
    t = clean.groupby(["dev", "pop"])[col].quantile(q).unstack().round(3)
    t["n"] = clean.groupby(["dev", "pop"]).size()
    print(f"\n{col}:")
    print(t.to_string())
seg = x[x.n_outliers > 0]
print("\n---- devices with outliers: fraction of pairs that are outliers, quantiles:",
      (seg.n_outliers / seg.n).quantile(q).round(3).to_dict())

GATE = {"AP": 0.5, "NI-DAQ": 0.5, "LFP": 0.8}  # proposed: robust all-barcode fit, max |residual| (ms)
x["gate_ms"] = x.dev.map(GATE)
x["fails"] = (x.n_outliers > 0) | ~(x.robust_res_max_ms <= x.gate_ms)
print(f"\n---- proposed ephys gate {GATE} + no pairing outliers: devices failing {int(x.fails.sum())} of {len(x)}, "
      f"in {x[x.fails].key.nunique()} of {x.key.nunique()} sessions")
print(x[x.fails].groupby(["dev", "pop"]).size().to_string())
print("failing devices by name (top):", x[x.fails].device.str.split(".").str[-1].value_counts().head(8).to_dict())
print("failing PXI slot/probe (top):", x[x.fails].device.value_counts().head(8).to_dict())
