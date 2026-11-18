"""Compare npc_stim display times per block: A (branch code, no patch) vs fixE (released 0.1.24
behaviour, C:/data/t2p_qc/out_issues/fixE), and B (with the session's SyncPatch) vs A.

Output: compare_population.csv (one row per block), compare_population.log
Run: ../../.venv/Scripts/python.exe compare_population.py > compare_population.log
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

OUT = pathlib.Path("C:/data/t2p_qc/out_phantom")
FIXE = pathlib.Path("C:/data/t2p_qc/out_issues/fixE")
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 110)
pd.set_option("display.max_rows", 500)

rows = []
sessions = []
for jf in sorted(OUT.glob("*.json")):
    d = json.loads(jf.read_text())
    z = np.load(OUT / f"{jf.stem}.npz")
    fx = json.loads((FIXE / f"{jf.stem}.json").read_text()) if (FIXE / f"{jf.stem}.json").exists() else {}
    zf = np.load(FIXE / f"{jf.stem}.npz") if (FIXE / f"{jf.stem}.npz").exists() else {}
    ps = d.get("patch_summary") or {}
    sessions.append(dict(folder=jf.stem, invalid=len(d.get("invalid_intervals", [])), A_error=d.get("A_error"), B_error=d.get("B_error"),
                         patch_error=d.get("patch_error"), error=d.get("error"), **{f"patch_{k}": v for k, v in ps.items()}))
    if "B_sources" not in d:
        continue
    recs = {b["block_index"]: b for b in d.get("diode_blocks", [])}
    for i, src_b in enumerate(d["B_sources"]):
        r = dict(folder=jf.stem, block=i, stim=d["B_stims"][i], box=d["B_box"][i], B=src_b, B_reason=d["B_reasons"][i])
        rec = recs.get(i)
        r["diode_status"] = rec["status"] if rec else None
        r["phantom"] = (rec or {}).get("diagnostics", {}).get("phantom_flicker", 0) if rec and rec["status"] == "cleaned" else 0
        r["glitch"] = (rec or {}).get("diagnostics", {}).get("glitch", 0) if rec and rec["status"] == "cleaned" else 0
        r["diode_reason"] = rec["reason"] if rec else None
        if "A_sources" in d:
            r["A"] = d["A_sources"][i]
            r["A_reason"] = d["A_reasons"][i]
            a_t, b_t = z.get(f"A_block{i}"), z.get(f"B_block{i}")
            if a_t is not None and b_t is not None:
                r["B_eq_A"] = bool(np.array_equal(a_t, b_t))
                r["B_minus_A_max_ms"] = float(np.max(np.abs(a_t - b_t)) * 1e3) if len(a_t) == len(b_t) else np.nan
        else:
            r["A"] = "raised: " + str(d.get("A_error"))[:60]
        if fx:
            srcs = fx.get("sources", [])
            r["fixE"] = srcs[i] if i < len(srcs) else None
            f_t = zf.get(f"A_block{i}") if hasattr(zf, "get") else None
            a_t = z.get(f"A_block{i}")
            if f_t is not None and a_t is not None:
                r["A_eq_fixE"] = bool(np.array_equal(a_t, f_t))
        rows.append(r)
df = pd.DataFrame(rows)
df.to_csv("compare_population.csv", index=False)
ss = pd.DataFrame(sessions)
ss.to_csv("compare_population_sessions.csv", index=False)

print(f"sessions: {len(ss)} (DRpilot_366122_20250109 not run: dev mouse, box interval 2, O(n^2) flip removal in npc_stim > 2 h)")
print(f"  patch build errors: {int(ss.patch_error.notna().sum())}; display-time errors A: {int(ss.A_error.notna().sum())} (replay/discontinuity files raise unpatched), B: {int(ss.B_error.notna().sum())}")
print("  A errors:", ss.A_error.dropna().str.slice(0, 80).value_counts().to_dict())
print("  B errors:", ss.B_error.dropna().str.slice(0, 120).value_counts().to_dict())
print(f"blocks: {len(df)}")
print("\n== A (branch code, no patch) vs fixE (released behaviour) ==")
both = df[df.A.isin(["photodiode", "unavailable", "constant_lag"]) & df.fixE.notna()]
print(pd.crosstab(both.fixE, both.A).to_string())
ph = both[(both.A == "photodiode") & (both.fixE == "photodiode")]
print(f"photodiode in both: {len(ph)}; identical arrays: {int(ph.A_eq_fixE.fillna(False).sum())}")
print(ph[~ph.A_eq_fixE.fillna(False).astype(bool)][["folder", "block", "stim"]].to_string())

print("\n== B (with SyncPatch) vs A (no patch) ==")
ab = df[df.A.isin(["photodiode", "unavailable", "constant_lag"])]
print(pd.crosstab(ab.A, ab.B).to_string())
same = ab[(ab.A == "photodiode") & (ab.B == "photodiode")]
print(f"photodiode in both: {len(same)}; identical: {int(same.B_eq_A.fillna(False).sum())}")
diff = same[~same.B_eq_A.fillna(False).astype(bool)]
print("changed (photodiode in both, arrays differ):")
print(diff[["folder", "block", "stim", "box", "diode_status", "phantom", "glitch", "B_minus_A_max_ms"]].to_string())
print("\nlost (A photodiode -> B not):")
print(ab[(ab.A == "photodiode") & (ab.B != "photodiode")][["folder", "block", "stim", "diode_status", "B_reason"]].to_string())
print("\nrecovered (A not photodiode -> B photodiode):")
print(ab[(ab.A != "photodiode") & (ab.B == "photodiode")][["folder", "block", "stim", "diode_status", "phantom", "glitch", "A_reason"]].to_string())
print("\nstill rejected, cleaned by the patch (phantom flicker removed but npc_stim rejects):")
print(ab[(ab.B != "photodiode") & (ab.diode_status == "cleaned") & (ab.phantom > 0)][["folder", "block", "stim", "phantom", "B_reason"]].to_string())
print("\nreplay files (A raised unpatched): B sources")
rp = df[df.A.astype(str).str.startswith("raised")]
print(rp.B.value_counts().to_string())

print("\n== diode records (box-1 blocks with a stim file) ==")
print(df.diode_status.value_counts(dropna=False).to_string())
print("rejected:")
print(df[df.diode_status == "rejected"][["folder", "block", "stim", "A", "B", "diode_reason"]].to_string())
print("\nphantom-flicker removal per session (sessions with any):")
per = df[df.phantom > 0].groupby("folder").agg(blocks=("block", "size"), phantom=("phantom", "sum"), glitch=("glitch", "sum"),
                                                 photodiode_after=("B", lambda x: int((x == "photodiode").sum())))
print(per.to_string())
print(f"total phantom-flicker edges removed: {int(df.phantom.sum())} in {int((df.phantom > 0).sum())} blocks, {per.shape[0]} sessions; glitch edges removed: {int(df.glitch.sum())} in {int((df.glitch > 0).sum())} blocks")
