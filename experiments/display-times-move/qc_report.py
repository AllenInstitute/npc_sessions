"""Render the tables for the QC report from qc_sessions.csv / qc_blocks.csv (written by
qc_compare.py). Prints Markdown to stdout; the report embeds this output verbatim.

Run: <venv>/Scripts/python.exe qc_report.py > qc_tables.md
"""

from __future__ import annotations

import json
import pathlib
import re

import pandas as pd

HERE = pathlib.Path(__file__).parent
S = pd.read_csv(HERE / "qc_sessions.csv")
B = pd.read_csv(HERE / "qc_blocks.csv")
C = S[S.status == "compared"].copy()


def md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def b(col) -> pd.Series:
    return C[col].fillna(False).astype(bool)


def cat(e) -> str:
    if not isinstance(e, str):
        return "none"
    t = e.split(":")[0]
    if "replayed sync data" in e:
        return f"{t}: replayed sync data (invalid_intervals)"
    if "timebase discontinuity" in e:
        return f"{t}: timebase discontinuity"
    if "diodeBoxFrameInterval" in e:
        return f"{t}: diodeBoxFrameInterval missing"
    if "NoneType" in e:
        return f"{t}: None display-time block (no photodiode times)"
    return f"{t}: {re.sub(r'[0-9]+', 'N', e.split(':', 1)[1].strip())[:70]}" if ":" in e else t


print("### Coverage\n")
print(md(S.status.value_counts().rename_axis("status").reset_index(name="sessions")))

print("\n### Consistency, baseline vs new (sessions)\n")
rows = []
for key, label in [
    ("vsync_blocks", "vsync blocks (`vsync_times_in_blocks`, unchanged code: sanity)"),
    ("A", "A: per-block display times (`frame_display_time_blocks` vs `get_display_times`)"),
    ("A_sources", "A: per-block source (photodiode / none)"),
    ("A_stim_paths", "A: block-to-stim-file matching"),
    ("A2", "A2: deprecated npc_sync shim == npc_stim result (new only)"),
    ("B", "B: `get_stim_frame_times(display_time)` (npc_sessions path)"),
    ("C", "C: `get_stim_frame_times(vsync)`"),
    ("D_onsets", "D: `stim_onsets`"),
    ("D_offsets", "D: `stim_offsets`"),
]:
    col = "A2_equals_A" if key == "A2" else (f"{key}_identical" if key in "ABC" or key.startswith("D_") or key == "vsync_blocks" else f"{key}_same")
    v = b(col)
    rows.append({"code path": label, "identical": int(v.sum()), "different": int((~v).sum())})
print(md(pd.DataFrame(rows)))

print("\n### Block-level display times (all blocks in compared sessions where A ran)\n")
print(md(pd.DataFrame({
    "blocks": [len(B)],
    "identical to baseline": [int(B.identical_to_baseline.astype(bool).sum())],
    "different": [int((~B.identical_to_baseline.astype(bool)).sum())],
})))

print("\n### Sessions that differ, and why\n")
d = C[~(b("A_identical") & b("B_identical") & b("C_identical") & b("D_onsets_identical") & b("D_offsets_identical"))]
out = []
for _, r in d.iterrows():
    what = [k for k in ("A", "B", "C") if not bool(r[f"{k}_identical"])] + [k for k in ("D_onsets", "D_offsets") if not bool(r[f"{k}_identical"])]
    out.append({
        "session": r.folder,
        "differs in": ", ".join(what),
        "baseline A / B error": f"{cat(r.base_A_error)} / {cat(r.base_B_error)}",
        "new A / B error": f"{cat(r.new_A_error)} / {cat(r.new_B_error)}",
        "B arrays base→new": f"{int(r.B_n_arrays_base)}→{int(r.B_n_arrays_new)}",
        "B arrays changed": r.B_diff_arrays if isinstance(r.B_diff_arrays, str) else "",
    })
print(md(pd.DataFrame(out)) if out else "none")

print("\n### Session-level errors (A: all blocks; B: npc_sessions path)\n")
err = pd.DataFrame({
    "A baseline": C.base_A_error.map(cat).value_counts(),
    "A new": C.new_A_error.map(cat).value_counts(),
    "B baseline": C.base_B_error.map(cat).value_counts(),
    "B new": C.new_B_error.map(cat).value_counts(),
}).fillna(0).astype(int).rename_axis("error").reset_index()
print(md(err))

print("\n### Block sources, new (blocks in sessions where A ran)\n")
src = B.groupby(["source", B.reason.fillna("").str.replace(r"[0-9]+", "N", regex=True).str[:90]]).size()
print(md(src.rename("blocks").reset_index().rename(columns={"reason": "reason (numbers → N)"})))

print("\n### Sanity checks on new photodiode display times\n")
P = B[B.source == "photodiode"].copy()
P["count_eq_vsync"] = P.n_display == P.n_vsync
P["count_eq_stim"] = P.n_display == P.n_stim_frames
checks = pd.DataFrame([
    {"check": "blocks with photodiode display times", "value": len(P)},
    {"check": "n display times == n vsyncs", "value": f"{int(P.count_eq_vsync.sum())}/{len(P)}"},
    {"check": "n display times == n frames in stim file", "value": f"{int(P.count_eq_stim.sum())}/{len(P)}"},
    {"check": "strictly increasing", "value": f"{int(P.monotonic.astype(bool).sum())}/{len(P)}"},
    {"check": "blocks with >=1 display time before its vsync", "value": int((P.n_display_before_vsync > 0).sum())},
    {"check": "total frames with display time before vsync", "value": int(P.n_display_before_vsync.sum())},
    {"check": "median vsync→display latency, median over blocks (ms)", "value": round(P.lat_ms_median.median(), 2)},
    {"check": "median latency, range over blocks (ms)", "value": f"{P.lat_ms_median.min():.2f} – {P.lat_ms_median.max():.2f}"},
    {"check": "p0.1 latency, min over blocks (ms)", "value": round(P.lat_ms_p001.min(), 2)},
    {"check": "p99.9 latency, max over blocks (ms)", "value": round(P.lat_ms_p999.max(), 2)},
    {"check": "min display-time interval, min over blocks (ms)", "value": round(P.min_interval_ms.min(), 2)},
])
print(md(checks))

print("\nBy diode-box frame interval:\n")
g = P.groupby("diode_box_frame_interval").agg(
    blocks=("block", "size"),
    median_latency_ms=("lat_ms_median", "median"),
    min_p001_ms=("lat_ms_p001", "min"),
    max_p999_ms=("lat_ms_p999", "max"),
    blocks_display_before_vsync=("n_display_before_vsync", lambda x: int((x > 0).sum())),
).reset_index()
print(md(g.round(2)))

print("\nBlocks with any display time before its vsync, or p99.9 latency > 60 ms (pre-existing behaviour, identical in baseline):\n")
o = P[(P.n_display_before_vsync > 0) | (P.lat_ms_p999 > 60)][
    ["folder", "block", "stim", "diode_box_frame_interval", "n_display_before_vsync", "lat_ms_min", "lat_ms_median", "lat_ms_p999", "lat_ms_max", "identical_to_baseline"]
].round(2)
print(f"{len(o)} blocks" + ("" if len(o) <= 40 else " (first 40 shown; full list in qc_blocks.csv)") + "\n")
print(md(o.head(40)) if len(o) else "none")

print("\n### Runtime (s per session, median)\n")
print(md(pd.DataFrame([{
    "A baseline": C.A_seconds_base.median(), "A new": C.A_seconds_new.median(),
    "B baseline": C.B_seconds_base.median(), "B new": C.B_seconds_new.median(),
}]).round(2)))
