"""Compare baseline vs new outputs from qc_run.py and compute sanity metrics on the new
display times. Writes qc_sessions.csv, qc_blocks.csv and qc_summary.json next to this file.

Run: <venv_new>/Scripts/python.exe qc_compare.py [--out C:/data/t2p_qc/out] [--cache C:/data/t2p_cache]
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
from collections import Counter

import h5py
import numpy as np

HERE = pathlib.Path(__file__).parent


def load(d: pathlib.Path, folder: str):
    j = d / f"{folder}.json"
    if not j.exists():
        return None, {}
    meta = json.loads(j.read_text())
    z = d / f"{folder}.npz"
    arrays = dict(np.load(z)) if z.exists() else {}
    return meta, arrays


def same(a, b) -> bool:
    return a is not None and b is not None and a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)


def err(meta: dict, key: str) -> str | None:
    e = meta.get(key)
    return None if e is None else f"{e['type']}: {e['msg'][:160]}"


def stim_frames(cache: pathlib.Path, folder: str, stim_name: str | None) -> int | None:
    if stim_name is None:
        return None
    with h5py.File(cache / folder / stim_name, "r") as f:
        n = f["frameIntervals"].shape[0]
    return 0 if n == 0 else n + 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="C:/data/t2p_qc/out")
    ap.add_argument("--cache", default="C:/data/t2p_cache")
    args = ap.parse_args()
    out, cache = pathlib.Path(args.out), pathlib.Path(args.cache)
    folders = [r["folder"] for r in csv.DictReader(open(HERE / "ephys_sessions.csv"))]

    sess_rows, block_rows = [], []
    for folder in folders:
        mb, ab = load(out / "baseline", folder)
        mn, an = load(out / "new", folder)
        row: dict = {"folder": folder}
        if mb is None or mn is None:
            row["status"] = "missing_output"
            sess_rows.append(row)
            continue
        for lab, m in (("base", mb), ("new", mn)):
            row[f"{lab}_error"] = err(m, "error")
            for k in ("A", "B", "C", "D", "sync"):
                row[f"{lab}_{k}_error"] = err(m, f"{k}_error")
        if mb.get("error") or mn.get("error"):
            row["status"] = "no_run:" + (mb.get("error") or mn.get("error"))["type"]
            sess_rows.append(row)
            continue
        row["n_invalid_intervals"] = len(mn.get("invalid_intervals", []))
        row["n_timebase_discontinuities"] = len(mn.get("timebase_discontinuities", []))
        row["vsync_blocks_identical"] = mb.get("vsync_block_lengths") == mn.get("vsync_block_lengths") and all(
            same(ab.get(k), an.get(k)) for k in ab if k.startswith("vsync_block")
        )
        row["n_blocks"] = len(mn.get("vsync_block_lengths", []))

        # A: per-block display times
        a_keys = sorted({k for k in list(ab) + list(an) if k.startswith("A_block")})
        row["A_errors_same"] = (mb.get("A_error") or {}).get("type") == (mn.get("A_error") or {}).get("type")
        a_diff = [k for k in a_keys if not same(ab.get(k), an.get(k))]
        row["A_n_arrays_base"] = sum(k.startswith("A_block") for k in ab)
        row["A_n_arrays_new"] = sum(k.startswith("A_block") for k in an)
        row["A_identical"] = row["A_errors_same"] and not a_diff
        row["A_diff_blocks"] = ";".join(a_diff)
        row["A_sources_base"] = ";".join(mb.get("A_sources", []))
        row["A_sources_new"] = ";".join(mn.get("A_sources", []))
        row["A_sources_same"] = mb.get("A_sources") == mn.get("A_sources")
        row["A_stim_paths_same"] = mb.get("A_stim_paths") == mn.get("A_stim_paths")
        # A2: shim equals A
        a2 = [k for k in an if k.startswith("A2_block")]
        row["A2_equals_A"] = (not mn.get("A2_error") and not mn.get("A_error")
                               and len(a2) == row["A_n_arrays_new"]
                               and all(same(an[k], an.get(k.replace("A2_", "A_"))) for k in a2)) or (
            bool(mn.get("A2_error")) and bool(mn.get("A_error")))
        # B, C
        for key in ("B", "C"):
            kb = [k for k in ab if k.startswith(f"{key}_")]
            kn = [k for k in an if k.startswith(f"{key}_")]
            arrays_same = set(kb) == set(kn) and all(same(ab[k], an[k]) for k in kb)
            exc_b = {k: v["type"] for k, v in (mb.get(f"{key}_exceptions") or {}).items()}
            exc_n = {k: v["type"] for k, v in (mn.get(f"{key}_exceptions") or {}).items()}
            row[f"{key}_identical"] = (arrays_same and exc_b == exc_n and mb.get(f"{key}_keys") == mn.get(f"{key}_keys")
                                       and (mb.get(f"{key}_error") or {}).get("type") == (mn.get(f"{key}_error") or {}).get("type"))
            row[f"{key}_n_arrays_base"] = len(kb)
            row[f"{key}_n_arrays_new"] = len(kn)
            row[f"{key}_exc_base"] = json.dumps(exc_b) if exc_b else ""
            row[f"{key}_exc_new"] = json.dumps(exc_n) if exc_n else ""
            row[f"{key}_keys_same"] = mb.get(f"{key}_keys") == mn.get(f"{key}_keys")
            row[f"{key}_only_new_arrays"] = ";".join(sorted(set(kn) - set(kb)))
            row[f"{key}_only_base_arrays"] = ";".join(sorted(set(kb) - set(kn)))
            row[f"{key}_diff_arrays"] = ";".join(k for k in set(kb) & set(kn) if not same(ab[k], an[k]))
        for k in ("onsets", "offsets"):
            x, y = ab.get(f"D_{k}"), an.get(f"D_{k}")
            both_err = x is None and y is None and err(mb, "D_error") == err(mn, "D_error")
            row[f"D_{k}_identical"] = same(x, y) or both_err
        row["A_seconds_base"] = mb.get("A_seconds")
        row["A_seconds_new"] = mn.get("A_seconds")
        row["B_seconds_base"] = mb.get("B_seconds")
        row["B_seconds_new"] = mn.get("B_seconds")
        row["A_reasons_new"] = " | ".join(sorted({r for r in mn.get("A_reasons", []) if r}))[:400]

        # sanity metrics on new per-block display times
        for i, src in enumerate(mn.get("A_sources", [])):
            v = an.get(f"vsync_block{i}")
            d = an.get(f"A_block{i}")
            stim_name = mn["A_stim_paths"][i]
            br = {
                "folder": folder, "block": i, "stim": stim_name, "source": src,
                "source_base": (mb.get("A_sources") or [None] * (i + 1))[i] if i < len(mb.get("A_sources") or []) else None,
                "diode_box_frame_interval": mn["A_diode_box_frame_interval"][i],
                "reason": mn["A_reasons"][i],
                "n_vsync": None if v is None else len(v),
                "n_stim_frames": stim_frames(cache, folder, stim_name),
                "identical_to_baseline": same(ab.get(f"A_block{i}"), d) or (d is None and f"A_block{i}" not in ab),
            }
            if d is not None and v is not None:
                br["n_display"] = len(d)
                if len(d) == len(v):
                    lat = (d - v) * 1000
                    br |= {
                        "lat_ms_min": float(lat.min()), "lat_ms_p001": float(np.percentile(lat, 0.1)),
                        "lat_ms_median": float(np.median(lat)), "lat_ms_p999": float(np.percentile(lat, 99.9)),
                        "lat_ms_max": float(lat.max()), "n_display_before_vsync": int((lat < 0).sum()),
                    }
                di = np.diff(d)
                br["monotonic"] = bool(np.all(di > 0))
                br["min_interval_ms"] = float(di.min() * 1000)
            block_rows.append(br)
        row["status"] = "compared"
        sess_rows.append(row)

    def write(name, rows):
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with open(HERE / name, "w", newline="") as f:
            w = csv.DictWriter(f, keys)
            w.writeheader()
            w.writerows(rows)

    write("qc_sessions.csv", sess_rows)
    write("qc_blocks.csv", block_rows)
    comp = [r for r in sess_rows if r["status"] == "compared"]
    summary = {
        "n_sessions_listed": len(folders),
        "status": Counter(r["status"] for r in sess_rows),
        "n_compared": len(comp),
        **{f"{k}_identical": Counter(bool(r.get(k + "_identical")) for r in comp) for k in ("A", "B", "C")},
        "A2_equals_A": Counter(bool(r["A2_equals_A"]) for r in comp),
        "A_sources_same": Counter(bool(r["A_sources_same"]) for r in comp),
        "A_stim_paths_same": Counter(bool(r["A_stim_paths_same"]) for r in comp),
        "vsync_blocks_identical": Counter(bool(r["vsync_blocks_identical"]) for r in comp),
        "D_onsets_identical": Counter(bool(r["D_onsets_identical"]) for r in comp),
        "D_offsets_identical": Counter(bool(r["D_offsets_identical"]) for r in comp),
        "A_error_base": Counter((r["base_A_error"] or "none").split(":")[0] for r in comp),
        "A_error_new": Counter((r["new_A_error"] or "none").split(":")[0] for r in comp),
        "B_error_base": Counter((r["base_B_error"] or "none").split(":")[0] for r in comp),
        "B_error_new": Counter((r["new_B_error"] or "none").split(":")[0] for r in comp),
        "block_sources_new": Counter(b["source"] for b in block_rows),
        "blocks_identical": Counter(bool(b["identical_to_baseline"]) for b in block_rows),
    }
    (HERE / "qc_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
