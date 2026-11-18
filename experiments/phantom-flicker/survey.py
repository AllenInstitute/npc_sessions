"""Population survey of diode edge classes in box-1 blocks (442 cached Task 2 pilot sessions).

For every vsync block with an exactly matched stim file (start time only, ../sync-patch-manifest
matcher), diodeBoxFrameInterval 1 (key absent = 1), vsync count == stim frames, in sync files
without invalid intervals or timebase discontinuities: run the prototype alignment
(`phantom.align_box1`) and record edge class counts. Also, for every held frame (stim interval
>= 1.5 refreshes): refreshes, phantom edges in the hold, the residual of the next frame's toggle.

Outputs: survey_blocks.csv, survey_holds.csv.gz, survey_run.log
Run: PYTHONPATH="../../../npc_sync/src;../../../npc_stim/src;../../src" ../../.venv/Scripts/python.exe survey.py
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import logging
import pathlib
import sys
import time
import traceback

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
FIXE = pathlib.Path("C:/data/t2p_qc/out_issues/fixE")


def npc_stim_result(folder: str) -> dict:
    p = FIXE / f"{folder}.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    return d


def run_session(folder: str) -> tuple[list[dict], list[dict], str]:
    import common
    import phantom as P

    logging.disable(logging.WARNING)
    rows, holds = [], []
    try:
        s = common.raw_sync(folder)
        if s.invalid_intervals or s.timebase_discontinuities:
            return [], [], f"{folder}: skipped (invalid intervals {len(s.invalid_intervals)}, discontinuities {len(s.timebase_discontinuities)})"
        blocks, matches = common.matches(folder)
        ns = npc_stim_result(folder)
        ns_by_stim = dict(zip(ns.get("stim_paths", []), zip(ns.get("sources", []), ns.get("reasons", []))))
        e_all, q_all = common.diode_edges(s)
        vr_all = s.get_rising_edges("vsync_stim", units="seconds")
        for bi, m in enumerate(matches):
            if m.stim is None:
                continue
            box = m.stim.diode_box_frame_interval
            v = blocks[bi]
            row = dict(folder=folder, block=bi, stim=m.stim.name, box=box if box is not None else -1,
                       n_frames=m.stim.n_frames, n_vsyncs=len(v))
            src = ns_by_stim.get(m.stim.path.name)
            row["npc_stim_source"] = src[0] if src else None
            row["npc_stim_reason"] = src[1] if src else None
            if (box or 1) != 1 or len(v) != m.stim.n_frames:
                row["status"] = "skipped: box != 1" if (box or 1) != 1 else "skipped: vsyncs != stim frames"
                rows.append(row)
                continue
            sel = (e_all >= v[0]) & (e_all < v[-1] + 0.5)
            e, q = e_all[sel], q_all[sel]
            row["n_edges"] = len(e)
            try:
                jr = np.searchsorted(vr_all, v) - 1
                vr = np.where(jr >= 0, vr_all[np.clip(jr, 0, None)], np.nan)
                a = P.align_box1(v, m.stim.frame_intervals, e, q, vsync_rising=vr)
            except P.OffsetUndetermined as exc:
                row["status"] = f"offset undetermined: {exc}"
                rows.append(row)
                continue
            R = a.refresh_sec
            res = a.residual
            row.update(
                status="aligned",
                first_frame_rising=a.first_frame_rising,
                n_landmarks=a.n_landmarks,
                n_landmarks_disagree=a.n_landmarks_disagree,
                lat_r_ms=float(np.nanmedian(a.latency_rising) * 1e3),
                lat_f_ms=float(np.nanmedian(a.latency_falling) * 1e3),
                n_toggle=a.count(P.TOGGLE),
                n_glitch=a.count(P.GLITCH),
                n_phantom=a.count(P.PHANTOM),
                n_unexplained=a.count(P.UNEXPLAINED),
                n_outside=a.count(P.OUTSIDE),
                n_missing=len(a.missing_frames),
                missing_first=int((a.missing_frames < 3).sum()),
                missing_last=int((a.missing_frames >= len(v) - 3).sum()),
                n_displaced=len(a.displaced),
                n_second_pass=len(a.second_pass),
                n_literal_late_return=len(a.literal_long_mismatch["late_return"]),
                n_literal_after_stall=len(a.literal_long_mismatch["after_stall"]),
                n_literal_other=len(a.literal_long_mismatch["other"]),
                n_same_pol_mismatch=len(a.long_mismatch),
                n_long=int((a.n_refreshes >= 2).sum()),
                n_holds_with_phantom=int(len(np.unique(a.edge_frame[a.edge_class == P.PHANTOM]))),
                max_abs_res_ms=float(np.nanmax(np.abs(res)) * 1e3),
                p999_abs_res_ms=float(np.nanpercentile(np.abs(res), 99.9) * 1e3),
                validated=a.validated,
                # unexplained edges: in which kind of frame interval?
                n_unexpl_in_hold=int(((a.edge_class == P.UNEXPLAINED) & (a.edge_frame >= 0) & (np.take(np.r_[a.n_refreshes, 1], np.clip(a.edge_frame, 0, None)) >= 2)).sum()),
            )
            rows.append(row)
            ph_k = a.edge_frame[a.edge_class == P.PHANTOM]
            un_k = a.edge_frame[(a.edge_class == P.UNEXPLAINED) & (a.edge_frame >= 0)]
            for k in np.flatnonzero(a.n_refreshes >= 2):
                holds.append(dict(folder=folder, block=bi, stim=m.stim.name, k=int(k), n=int(a.n_refreshes[k]),
                                  fi_ms=float(m.stim.frame_intervals[k] * 1e3),
                                  n_phantom=int((ph_k == k).sum()), n_unexplained=int((un_k == k).sum()),
                                  res_k_ms=float(res[k] * 1e3), res_next_ms=float(res[k + 1] * 1e3) if k + 1 < len(res) else np.nan,
                                  displaced=bool(k in set(a.displaced.tolist())),
                                  second_pass_next=bool(k + 1 in set(a.second_pass.tolist())),
                                  polarity_next="R" if ((k + 1) % 2 == 0) == a.first_frame_rising else "F"))
        return rows, holds, f"{folder}: {len(rows)} blocks"
    except Exception:  # noqa: BLE001
        return rows, holds, f"{folder}: ERROR {traceback.format_exc()}"


def main() -> None:
    import common

    folders = sorted(p.name for p in common.CACHE.iterdir() if p.is_dir())
    if len(sys.argv) > 1:
        folders = [f for f in folders if any(s in f for s in sys.argv[1:])]
    t0 = time.time()
    all_rows, all_holds = [], []
    with open(HERE / "survey_run.log", "w") as log, cf.ProcessPoolExecutor(max_workers=10) as ex:
        for rows, holds, msg in ex.map(run_session, folders):
            all_rows += rows
            all_holds += holds
            log.write(msg + "\n")
            log.flush()
        log.write(f"done: {len(folders)} sessions in {time.time() - t0:.0f} s\n")
    pd.DataFrame(all_rows).to_csv(HERE / "survey_blocks.csv", index=False)
    pd.DataFrame(all_holds).to_csv(HERE / "survey_holds.csv.gz", index=False)
    print(f"{len(folders)} sessions, {len(all_rows)} blocks, {len(all_holds)} holds in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
