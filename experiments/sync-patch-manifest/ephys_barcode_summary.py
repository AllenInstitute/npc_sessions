"""Summarise ephys_barcode_residuals.jsonl -> ephys_barcode_summary.log.

Groups: device type (AP 30 kHz, LFP 2.5 kHz, NI-DAQ) x file population (clean / invalid
intervals only / timebase discontinuities). Residuals: endpoint fit (what npc_ephys does now:
first and last matched barcode) and least-squares fit to all matched barcodes.
"""

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
d = pd.read_json("ephys_barcode_residuals.jsonl", lines=True)
print(f"rows (devices): {len(d)}; sessions: {d.key.nunique()}; errors: {d.error.notna().sum()}")
print(d[d.error.notna()].groupby("error").key.nunique().to_string() if d.error.notna().any() else "")
d = d[d.error.isna() & d.n_matched.notna()].copy()
d["dev"] = np.where(d.device.str.endswith("-AP"), "AP", np.where(d.device.str.endswith("-LFP"), "LFP", np.where(d.device.str.contains("NI-DAQ"), "NI-DAQ", "other")))
d["pop"] = np.where(d.n_discontinuities > 0, "discontinuities", np.where(d.n_invalid > 0, "invalid intervals only", "clean"))
q = [0.5, 0.9, 0.99, 1.0]
for col in ("endpoint_res_max_ms", "ls_res_max_ms"):
    print(f"\n{col} (per device: max |residual| over its matched barcodes), quantiles across devices:")
    t = d.groupby(["dev", "pop"])[col].quantile(q).unstack().round(3)
    t["n_devices"] = d.groupby(["dev", "pop"]).size()
    t["n_sessions"] = d.groupby(["dev", "pop"]).key.nunique()
    print(t.to_string())
print("\nmatched barcodes per device: median", d.n_matched.median(), "min", d.n_matched.min())
ap = d[(d.dev == "AP") & (d["pop"] == "clean")]
print("\nAP, clean files: endpoint max > 0.2 ms:", (ap.endpoint_res_max_ms > 0.2).sum(), "of", len(ap))
print(ap.sort_values("endpoint_res_max_ms").tail(8)[["key", "device", "n_matched", "endpoint_res_max_ms", "ls_res_max_ms", "worst_t"]].to_string())
print("\nlargest endpoint residuals overall:")
print(d.sort_values("endpoint_res_max_ms").tail(15)[["key", "device", "pop", "n_invalid", "n_discontinuities", "n_matched", "endpoint_res_max_ms", "ls_res_max_ms", "worst_t"]].to_string())
dd = d[d["pop"] == "discontinuities"]
print("\nfiles with timebase discontinuities (AP devices):")
print(dd[dd.dev == "AP"].groupby("key")[["endpoint_res_max_ms", "ls_res_max_ms"]].max().round(1).to_string())
