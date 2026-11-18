"""Summaries of survey_blocks.csv / survey_holds.csv -> survey_summary.log

Run: ../../.venv/Scripts/python.exe summarize_survey.py > survey_summary.log
"""

from __future__ import annotations

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 60)

b = pd.read_csv("survey_blocks.csv")
h = pd.read_csv("survey_holds.csv.gz")
print(f"blocks with a matched stim file: {len(b)} in {b.folder.nunique()} sessions")
print("status:")
print(b.status.str.slice(0, 60).value_counts().to_string())
a = b[b.status == "aligned"].copy()
a["stim_type"] = a.stim.str.split("_").str[0]
print(f"\naligned box-1 blocks: {len(a)} in {a.folder.nunique()} sessions; validated: {int(a.validated.sum())}")
print("\nby npc_stim source (released 0.1.24 behaviour, fixE run) x validated:")
print(pd.crosstab(a.npc_stim_source.fillna("none"), a.validated).to_string())
cols = ["n_glitch", "n_phantom", "n_unexplained", "n_missing", "n_displaced", "n_second_pass", "n_literal_late_return", "n_literal_after_stall", "n_literal_other"]
print("\nblocks with >0 of each count / total of the count:")
for c in cols:
    print(f"  {c:24s} blocks {int((a[c] > 0).sum()):5d}  total {int(a[c].sum()):8d}")
print("\nphantom edges by stim type:")
g = a.groupby("stim_type").agg(blocks=("block", "size"), blocks_with_phantom=("n_phantom", lambda x: int((x > 0).sum())),
                               phantom=("n_phantom", "sum"), glitch=("n_glitch", "sum"), unexplained=("n_unexplained", "sum"),
                               missing=("n_missing", "sum"), validated=("validated", "sum"))
print(g.to_string())
print("\nnot validated blocks:")
nv = a[~a.validated.astype(bool)]
print(nv[["folder", "block", "stim", "npc_stim_source", "n_phantom", "n_glitch", "n_unexplained", "n_missing", "missing_first", "missing_last", "n_displaced", "n_literal_other", "npc_stim_reason"]].to_string())
print("\nblocks npc_stim rejects that validate (would be recovered if npc_stim consumes the cleaned line):")
rec = a[(a.npc_stim_source != "photodiode") & a.validated.astype(bool)]
print(rec[["folder", "block", "stim", "n_phantom", "n_glitch", "npc_stim_reason"]].to_string())
print("\nphotodiode blocks with phantom edges (npc_stim accepted them with phantoms present):")
ph = a[(a.npc_stim_source == "photodiode") & (a.n_phantom > 0)]
print(ph[["folder", "block", "stim", "n_phantom", "n_glitch", "validated"]].to_string())
print("\noffset undetermined:")
ou = b[b.status.str.startswith("offset")]
print(ou[["folder", "block", "stim", "n_frames", "npc_stim_source"]].to_string())

print("\n--- holds (stim frame interval >= 1.5 refreshes) ---")
h["on_grid"] = (h.fi_ms / (1000 / 59.95) - h.n).abs() < 0.1
print(f"holds: {len(h)}; on the refresh grid: {int(h.on_grid.sum())}")
t = h.groupby(["n"]).agg(holds=("k", "size"), with_phantom=("n_phantom", lambda x: int((x > 0).sum())),
                          phantom_edges=("n_phantom", "sum"), max_phantom=("n_phantom", "max"),
                          unexplained=("n_unexplained", "sum"))
print(t[t.holds > 0].to_string())
print("\nphantom edges per hold vs refreshes (holds with phantoms):")
print(pd.crosstab(h[h.n_phantom > 0].n, h[h.n_phantom > 0].n_phantom).to_string())
print("\nnext-toggle residual (ms) after on-grid holds, by refreshes and whether the hold had phantom edges:")
og = h[h.on_grid & np.isfinite(h.res_next_ms)]
print(og.groupby(["n", og.n_phantom > 0]).res_next_ms.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).round(2).to_string())
print("\noff-grid holds (late flip return at one end): next toggle matched in the second pass:")
fg = h[~h.on_grid]
print(f"  {len(fg)} off-grid holds; next toggle second-pass: {int(fg.second_pass_next.sum())}; displaced: {int(fg.displaced.sum())}")
