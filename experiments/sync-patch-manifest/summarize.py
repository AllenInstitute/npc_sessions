"""Summarise the P0 manifest and re-derive the gates (step 2 of notes/sync-patching-handoff.md).

Inputs: manifest_files.jsonl, manifest_blocks.jsonl, intervals.jsonl (from manifest.py).
Outputs:
  manifest.csv          one row per block (+ stim-without-block rows), with file-level fields
                        and the final class
  gap_residuals.csv     one row per solved vsync gap
  summary.log           counts, class tables, gate distributions (stdout)
  gates.json            the distributions and proposed gates

Reclassification on top of manifest.py (no files re-read):
  truncated             count deficit, and the block's last vsync is within 1 s of the end of the
                        sync recording (the sync file ended before the stimulus did)
  stim-without-block    kept only for stim files that start inside this sync file's recording;
                        others belong to another sync file in the same folder (dropped)
  gate re-derivation    blocks are re-classified with the proposed vsync gap gate (column
                        cls_proposed)

Run: ../../.venv/Scripts/python.exe summarize.py > summary.log 2>&1
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent
pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 70)
pd.set_option("display.max_rows", 300)

CURRENT_GAP_GATE_MS = 0.5
MARGIN_GATE_MS = 10.0

f = pd.read_json(HERE / "manifest_files.jsonl", lines=True)
b = pd.read_json(HERE / "manifest_blocks.jsonl", lines=True)
f["in_npc_lims"] = f.in_npc_lims.astype(str) == "True"
b = b.merge(
    f[["sync_path", "source", "in_npc_lims", "is_ephys", "project", "version", "total_seconds",
       "n_invalid", "invalid_s", "n_discontinuities", "has_barcodes"]],
    on="sync_path", how="left",
)

# ---------------------------------------------------------------- reclassification
trunc = (b.cls == "unexplained") & (b["diff"] < 0) & (b.n_overlap.fillna(0) == 0) & (b.t1 > b.total_seconds - 1.0)
b.loc[trunc, "cls"] = "truncated"

onsets = b[b.block_index >= 0].groupby("sync_path").t0.apply(lambda s: np.sort(s.to_numpy()))
keep = []
for i, r in b.iterrows():
    if r.cls != "stim-without-block":
        keep.append(True)
        continue
    on = onsets.get(r.sync_path, np.array([]))
    if not len(on) or not np.isfinite(r.match_offset):
        keep.append(True)  # no blocks at all: the stim can't be placed
        continue
    cands = [o - r.match_offset for o in on]
    start = next((s for s in cands if np.argmin(np.abs(on - s)) == list(cands).index(s)), cands[0])
    keep.append(bool(0 <= start <= r.total_seconds))
n_other = int((~np.array(keep)).sum())
b = b[np.array(keep)].copy()

# ---------------------------------------------------------------- per-gap residuals
rows = []
for _, r in b[b.n_gaps.fillna(0) > 0].iterrows():
    for (a, z), res, sec, n, nr in zip(r.overlap, r.residuals_ms, r.second_best_ms, r.n_frames_gaps, r.n_removed_gaps):
        rows.append(dict(session=r.session, block_index=r.block_index, stim=r.stim, cls=r.cls,
                         n_invalid=r.n_invalid, has_disc=r.n_discontinuities > 0, span_s=round(z - a, 3),
                         residual_ms=res, second_best_ms=sec, n_frames=n, n_removed=nr, balance=r.balance))
g = pd.DataFrame(rows)
g["population"] = np.where(g.has_disc, "discontinuities", np.where(g.n_invalid >= 20, ">=20 runs", "<20 runs, no discontinuity"))
g.to_csv(HERE / "gap_residuals.csv", index=False)

print("=" * 30, "population")
print(f"sync files: {len(f)} ({(f.source == 'share').sum()} share, {(f.source == 's3').sum()} npc_lims-only on S3); "
      f"files with errors: {f.error.notna().sum()}")
print(f"files with invalid intervals: {(f.n_invalid > 0).sum()} ({(f.n_invalid > 0).mean() * 100:.1f}%); "
      f"with timebase discontinuities: {(f.n_discontinuities > 0).sum()}; both: {((f.n_invalid > 0) & (f.n_discontinuities > 0)).sum()}")
print(f"files with >=20 invalid intervals: {(f.n_invalid >= 20).sum()}")
print(f"npc_lims-tracked files: {f.in_npc_lims.sum()}; of which with invalid intervals: {(f.in_npc_lims & (f.n_invalid > 0)).sum()}, "
      f"discontinuities: {(f.in_npc_lims & (f.n_discontinuities > 0)).sum()}")
print(f"stim-without-block rows dropped (stim starts outside this sync file's recording): {n_other}")
print("\nfiles with errors:")
print(f[f.error.notna()][["session", "error"]].to_string())

print("\n" + "=" * 30, "block classes")
print(b.cls.value_counts().to_string())
m = b[b.block_index >= 0]
mismatch = m[m["diff"].notna() & (m["diff"] != 0)]
print(f"\nmatched blocks: {m['diff'].notna().sum()}; count != stim frames: {len(mismatch)}; unclassified: {mismatch.cls.isna().sum()}")
print(mismatch.cls.value_counts().to_string())
ov = m[m.n_overlap.fillna(0) > 0]
print(f"\nblocks overlapping invalid intervals: {len(ov)} (count == stim frames despite copies: {(ov['diff'] == 0).sum()})")
print(ov.cls.value_counts().to_string())
print("\nnpc_lims-tracked sessions only:")
print(b[b.in_npc_lims.fillna(False).astype(bool)].cls.value_counts().to_string())

cols = ["session", "block_index", "stim", "n_vsync", "n_stim", "diff", "n_overlap", "n_disc_in_block", "n_invalid", "balance", "max_abs_resid_ms"]
for c in ("unexplained", "truncated", "drop", "displaced"):
    x = b[b.cls == c].copy()
    x["stim"] = x.stim.str.slice(0, 26)
    print(f"\n----- {c} ({len(x)})")
    print(x[cols].to_string())
    if c == "drop":
        print(x[["session", "drop_t", "drop_n_missing", "drop_sec", "drop_cost_ms", "notes"]].to_string())

# lost events
print("\n" + "=" * 30, "stim-file events inside invalid spans (blocks with a solved reconciliation)")
le = b[b.lost_events.apply(lambda d: isinstance(d, dict) and len(d) > 0)]
tot = {}
unknown = {}
for d in le.lost_events:
    for k, v in d.items():
        if v is None:
            unknown[k] = unknown.get(k, 0) + 1
        else:
            tot[k] = tot.get(k, 0) + v
print("events lost (sum over blocks):", tot)
print("blocks where frame times unknown (solve failed), by key:", unknown)
for cls in ("replay-reconcilable",):
    t2 = {}
    for d in le[le.cls == cls].lost_events:
        for k, v in d.items():
            if v is not None:
                t2[k] = t2.get(k, 0) + v
    print(f"  in {cls} blocks:", t2)

# ---------------------------------------------------------------- gates
print("\n" + "=" * 30, "gate: vsync per-gap anchor residual")
q = [0.5, 0.9, 0.95, 0.99, 1.0]
for k, x in g.groupby("population"):
    a = x.residual_ms.abs()
    sl = np.polyfit(x.span_s, x.residual_ms, 1) if len(x) > 2 else (np.nan, np.nan)
    print(f"{k:28s} n={len(x):4d}  |res| ms quantiles {dict(zip(q, np.round(a.quantile(q).to_numpy(), 3)))}  "
          f"fit res = {sl[0]:+.3f} ms/s x span {sl[1]:+.3f}")
base = g[g.population == "<20 runs, no discontinuity"]
edges = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 1, 5, 20]
print("\n|residual| histogram, <20 runs & no discontinuity:")
print(pd.cut(base.residual_ms.abs(), edges).value_counts().sort_index().to_string())
print("\ngaps with 0.1 < |residual| <= 0.5 ms (pass the current gate):")
print(base[(base.residual_ms.abs() > 0.1) & (base.residual_ms.abs() <= 0.5)].sort_values("session")[
    ["session", "block_index", "cls", "n_invalid", "span_s", "residual_ms", "second_best_ms", "balance"]].to_string())
# blocks whose other gaps fail
PROPOSED_GAP_GATE_MS = 0.2
print(f"\nsecond-best margin (gaps within the current gate): min {(base[base.residual_ms.abs() <= 0.5].second_best_ms - base[base.residual_ms.abs() <= 0.5].residual_ms.abs()).min():.1f} ms")

def reclass(r):
    if r.cls != "replay-reconcilable":
        return r.cls
    if any(abs(x) > PROPOSED_GAP_GATE_MS for x in r.residuals_ms):
        return "unexplained"
    return r.cls

b["cls_proposed"] = b.apply(reclass, axis=1)
changed = b[b.cls != b.cls_proposed]
print(f"\nwith a {PROPOSED_GAP_GATE_MS} ms gate: {len(changed)} replay-reconcilable blocks become unexplained:")
print(changed[["session", "block_index", "stim", "n_invalid", "max_abs_resid_ms"]].to_string())

print("\n" + "=" * 30, "gate: camera integer-period residual")
iv = pd.read_json(HERE / "intervals.jsonl", lines=True)
cam = iv[iv.kind == "camera"].merge(f[["session", "n_discontinuities", "n_invalid"]], on="session", how="left")
cam["population"] = np.where(cam.n_discontinuities > 0, "discontinuities", np.where(cam.n_invalid >= 20, ">=20 runs", "<20 runs, no discontinuity"))
cam["line_type"] = np.where(cam.line.str.endswith("_exposing"), "exposing", "readout")
print(cam.groupby(["population", "line_type"]).status.value_counts().unstack(fill_value=0).to_string())
rc = cam[cam.status == "reconstructed"]
for (k, lt), x in rc.groupby(["population", "line_type"]):
    a = x.residual_sec.abs() * 1e3
    print(f"{k:28s} {lt:9s} n={len(x):5d} |res| ms {dict(zip(q, np.round(a.quantile(q).to_numpy(), 4)))}; "
          f"frac p99 {x.fraction.quantile(.99):.4f} max {x.fraction.max():.4f}; "
          f"in-segment fit residual max p99 {x.fit_residual_max_sec.quantile(.99) * 1e3:.3f} ms")
cb = rc[rc.population == "<20 runs, no discontinuity"]
print("\n|residual| histogram (ms), exposing, <20 runs & no discontinuity:")
print(pd.cut(cb[cb.line_type == "exposing"].residual_sec.abs() * 1e3, [0, 0.01, 0.02, 0.032, 0.05, 0.1, 0.2, 0.5, 5]).value_counts().sort_index().to_string())
print("readout:")
print(pd.cut(cb[cb.line_type == "readout"].residual_sec.abs() * 1e3, [0, 0.01, 0.02, 0.032, 0.05, 0.1, 0.2, 0.5, 5]).value_counts().sort_index().to_string())
failed = cam[cam.status == "failed"]
print("\nfailed camera spans by reason:")
print(failed.reason.str.slice(0, 40).value_counts().to_string())

print("\n" + "=" * 30, "gate: photodiode prediction (replay-reconcilable blocks)")
di = iv[iv.kind == "diode"]
print(di.status.value_counts().to_string())
dr = di[di.status == "reconstructed"]
print(f"n={len(dr)}; baseline error ms quantiles {dict(zip(q, np.round(dr.baseline_error_sec.quantile(q).to_numpy() * 1e3, 3)))}")
print(f"after-span error ms quantiles {dict(zip(q, np.round(dr.after_error_sec.quantile(q).to_numpy() * 1e3, 3)))}")
print(f"(after - baseline) ms quantiles {dict(zip(q, np.round((dr.after_error_sec - dr.baseline_error_sec).quantile(q).to_numpy() * 1e3, 3)))}")
print(f"polarity errors after span: {int(dr.n_polarity_errors_after.sum())}; gate failures: {int(dr.failures.apply(len).gt(0).sum())}")
print(dr[dr.failures.apply(len) > 0][["session", "block_index", "baseline_error_sec", "after_error_sec", "failures"]].to_string())
print("failed:", di[di.status == "failed"].reason.value_counts().to_string())

out_cols = ["session", "source", "in_npc_lims", "is_ephys", "project", "sync_path", "version", "block_index", "t0", "t1",
            "stim", "n_vsync", "n_stim", "diff", "box", "match_offset", "match_reason", "n_overlap", "overlap",
            "n_disc_in_block", "disc_in_block", "n_disc_before_block", "n_gaps", "balance", "residuals_ms",
            "second_best_ms", "n_frames_gaps", "max_abs_resid_ms", "failures", "lost_events", "drop_t",
            "drop_n_missing", "drop_cost_ms", "notes", "cls", "cls_proposed", "n_invalid", "n_discontinuities"]
b[[c for c in out_cols if c in b]].to_csv(HERE / "manifest.csv", index=False)
gates = dict(
    vsync_gap_residual_ms={k: dict(zip(map(str, q), np.round(x.residual_ms.abs().quantile(q).to_numpy(), 4).tolist()))
                           | {"n": int(len(x))} for k, x in g.groupby("population")},
    vsync_gap_current_gate_ms=CURRENT_GAP_GATE_MS,
    vsync_gap_proposed_gate_ms=PROPOSED_GAP_GATE_MS,
    camera_residual_ms={f"{k}|{lt}": dict(zip(map(str, q), np.round(x.residual_sec.abs().quantile(q).to_numpy() * 1e3, 4).tolist()))
                        | {"n": int(len(x))} for (k, lt), x in rc.groupby(["population", "line_type"])},
    diode_after_minus_baseline_ms=dict(zip(map(str, q), np.round((dr.after_error_sec - dr.baseline_error_sec).quantile(q).to_numpy() * 1e3, 3).tolist())),
)
(HERE / "gates.json").write_text(json.dumps(gates, indent=1))
print("\nwrote manifest.csv, gap_residuals.csv, gates.json")
