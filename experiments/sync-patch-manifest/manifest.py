"""P0 survey manifest: every sync file in sessions.csv -> per-file and per-block records.

Per file: invalid_intervals (replayed buffers), timebase_discontinuities (barcode onsets),
blocks. Per block: exact one-to-one stim match by start time (frame count NOT used, see
common.match_stims_to_blocks), strict vsync count vs stim frames, the per-gap vsync solve
(npc_sync.patch.solve_vsync_gaps) for blocks overlapping invalid intervals, stim-file events
lost inside invalid spans, and a class:

  clean                 no invalid interval or discontinuity in the block; count == stim frames
  replay-reconcilable   overlaps invalid intervals; per-gap solve passes every gate; no
                        timebase discontinuity in the block
  displaced             a timebase discontinuity (barcode onset displaced by k x 0.5 s) lies in
                        the block and it isn't a located drop; or the block overlaps invalid
                        intervals, fails the solve, and the file has discontinuities
  drop                  count deficit explained by a located drop (npc_sync.patch.locate_drop:
                        duration deficit = k x 0.5 s, located at a buffer boundary)
  unexplained           anything else with count != stim frames, or a failed solve with no
                        discontinuity in the file
  no-stim               block without a matched stim file (count can't be checked)
  stim-without-block    stim file with frames and no block (row has block_index = -1)

Per interval (intervals.jsonl): camera-line integer-period interpolation and photodiode
prediction diagnostics, for re-deriving gates.

Run (npc_sessions venv for S3 access; npc_sync from the sync-patch branch):
    PYTHONPATH=../../../npc_sync/src ../../.venv/Scripts/python.exe manifest.py [n_workers] > manifest_run.log 2>&1
Resumable: files already in manifest_files.jsonl are skipped.
"""

from __future__ import annotations

import concurrent.futures
import csv
import dataclasses
import json
import logging
import pathlib
import sys
import time
import traceback

import numpy as np

HERE = pathlib.Path(__file__).parent
FILES = HERE / "manifest_files.jsonl"
BLOCKS = HERE / "manifest_blocks.jsonl"
INTERVALS = HERE / "intervals.jsonl"

def _j(x):
    from npc_sync.patch import _jsonable

    return _jsonable(x)


def process(row: dict) -> dict:
    import npc_io

    import npc_sync
    from npc_sync import patch as P

    from common import match_stims_to_blocks, read_stim, sync_start

    logging.disable(logging.WARNING)
    t0 = time.time()
    out_file = dict(session=row["session"], source=row["source"], sync_path=row["sync_path"],
                    in_npc_lims=row["in_npc_lims"], is_ephys=row["is_ephys"], project=row["project"])
    blocks_out, intervals_out = [], []
    try:
        path = row["sync_path"]
        raw = npc_sync.SyncDataset(path, on_invalid="ignore")
        inv = raw.invalid_intervals
        disc = raw.timebase_discontinuities
        out_file.update(
            version=raw.meta_data.get("version"),
            start_time=str(raw.start_time),
            total_seconds=raw.total_seconds,
            has_barcodes=raw.has_barcodes,
            n_invalid=len(inv),
            invalid_s=float(sum(z - a for a, z in inv)),
            invalid_intervals=[list(x) for x in inv],
            n_discontinuities=len(disc),
            discontinuities=[list(x) for x in disc],
        )
        try:
            blocks = list(raw.vsync_times_in_blocks)
        except Exception as exc:  # noqa: BLE001
            out_file["error"] = f"vsync_times_in_blocks: {type(exc).__name__}: {exc}"
            blocks = []
        out_file["n_blocks"] = len(blocks)
        stim_paths = [p for p in row["stim_paths"].split(";") if p]
        stims = []
        for p in stim_paths:
            if row["source"] == "s3":
                import io

                pp = npc_io.from_pathlike(p)
                s = read_stim(io.BytesIO(pp.read_bytes()), sync_start(raw), name=pp.stem)
            else:
                s = read_stim(pathlib.Path(p), sync_start(raw))
            stims.append(s)
        out_file["n_stims"] = sum(s.n_frames > 0 for s in stims)
        matches, unmatched = match_stims_to_blocks(blocks, stims)
        fs = raw.sample_freq
        spans = [(a - 1 / fs, z) for a, z in inv]
        labels = raw.line_labels

        for m in matches:
            b = blocks[m.block_index]
            rec_b = dict(session=row["session"], sync_path=path, block_index=m.block_index,
                         t0=float(b[0]), t1=float(b[-1]), n_vsync=int(len(b)),
                         stim=m.stim.name if m.stim else None,
                         n_stim=m.stim.n_frames if m.stim else None,
                         box=m.stim.diode_box_frame_interval if m.stim else None,
                         match_offset=m.offset_sec, match_reason=m.reason)
            ov = [(a, z) for a, z in spans if a <= b[-1] and z > b[0]]
            dis_in = [d for d in disc if d[0] <= b[-1] and d[1] >= b[0]]
            rec_b.update(n_overlap=len(ov), overlap=[list(x) for x in ov], n_disc_in_block=len(dis_in),
                         disc_in_block=[list(x) for x in dis_in],
                         n_disc_before_block=sum(d[1] < b[0] for d in disc))
            if m.stim is None:
                rec_b["diff"] = None
                rec_b["cls"] = "no-stim"
                blocks_out.append(rec_b)
                continue
            diff = int(len(b) - m.stim.n_frames)
            rec_b["diff"] = diff
            rec = None
            cls = None
            notes = []
            if ov:
                rec = P.solve_vsync_gaps(b, m.stim.frame_intervals, ov)
                f = rec.failures()
                rec_b.update(
                    n_gaps=len(rec.gaps), balance=rec.balance, boundary_gap=rec.boundary_gap,
                    residuals_ms=[g.residual_sec * 1e3 for g in rec.gaps],
                    second_best_ms=[g.second_best_sec * 1e3 for g in rec.gaps],
                    n_frames_gaps=[g.n_frames for g in rec.gaps],
                    n_removed_gaps=[g.n_removed for g in rec.gaps],
                    max_abs_resid_ms=rec.max_abs_residual_sec * 1e3,
                    failures=f,
                )
                if not f and not dis_in:
                    cls = "replay-reconcilable"
                elif dis_in or disc:
                    cls = "displaced"
                    if not f:
                        notes.append("solve passes, but a timebase discontinuity is in the block/file")
                else:
                    cls = "unexplained"
            elif dis_in:
                if diff < 0:
                    try:
                        i0, n_missing, drop, cost = P.locate_drop(b, m.stim.frame_intervals)
                        rec_b.update(drop_i0=i0, drop_t=float(b[i0]), drop_n_missing=n_missing, drop_sec=drop, drop_cost_ms=cost)
                        cls = "drop"
                    except P.ReconciliationError as exc:
                        notes.append(f"locate_drop: {exc}")
                        cls = "displaced"
                else:
                    cls = "displaced"
            elif diff == 0:
                cls = "clean"
            else:
                if diff < 0:
                    try:
                        i0, n_missing, drop, cost = P.locate_drop(b, m.stim.frame_intervals)
                        rec_b.update(drop_i0=i0, drop_t=float(b[i0]), drop_n_missing=n_missing, drop_sec=drop, drop_cost_ms=cost)
                        if cost < 0.2:
                            cls = "drop"
                            notes.append("no barcode evidence of the drop (no discontinuity flagged)")
                    except P.ReconciliationError as exc:
                        notes.append(f"locate_drop: {exc}")
                cls = cls or "unexplained"
            rec_b["cls"] = cls
            rec_b["notes"] = notes

            # stim events lost inside invalid spans (only where frame times are known)
            if ov and m.stim is not None:
                lost = {}
                times = rec.times if (rec is not None and rec.balance == 0) else None
                for k, v in m.stim.events.items():
                    if times is None:
                        lost[k] = None  # frame times unknown (solve failed)
                        continue
                    v = v[v < len(times)].astype(int)
                    lost[k] = int(P.in_intervals(times[v], ov).sum())
                rec_b["lost_events"] = lost

                # photodiode prediction diagnostics per gap (gate derivation)
                if rec is not None and not rec_b["failures"] and m.stim.diode_box_frame_interval and "stim_photodiode" in labels:
                    dr = raw.get_rising_edges("stim_photodiode", units="seconds")
                    df = raw.get_falling_edges("stim_photodiode", units="seconds")
                    for g in rec.gaps:
                        _, _, sol = P.predict_diode_edges(rec.times, dr, df, g.span, m.stim.diode_box_frame_interval)
                        intervals_out.append(dict(session=row["session"], kind="diode", line="stim_photodiode",
                                                  block_index=m.block_index,
                                                  failures=sol.failures(), **_j(dataclasses.asdict(sol))))
            blocks_out.append(rec_b)
        for m in unmatched:
            blocks_out.append(dict(session=row["session"], sync_path=path, block_index=-1, stim=m.stim.name,
                                   n_stim=m.stim.n_frames, match_offset=m.offset_sec, match_reason=m.reason,
                                   cls="stim-without-block"))

        # camera lines: integer-period interpolation diagnostics for every interval
        if inv:
            for bit, lab in enumerate(labels):
                if not lab.endswith(P.CAMERA_LINE_SUFFIXES):
                    continue
                r = raw.get_rising_edges(bit, units="seconds")
                f = raw.get_falling_edges(bit, units="seconds")
                for sp in spans:
                    for kind, e in (("rising", r), ("falling", f)):
                        _, sol = P.interpolate_periodic_edges(e, sp)
                        intervals_out.append(dict(session=row["session"], kind="camera", line=lab, edge=kind,
                                                  failures=sol.failures(), **_j(dataclasses.asdict(sol))))
    except Exception as exc:  # noqa: BLE001
        out_file["error"] = f"{type(exc).__name__}: {exc}"
        out_file["traceback"] = traceback.format_exc()[-2000:]
    out_file["seconds"] = round(time.time() - t0, 1)
    return dict(file=_j(out_file), blocks=_j(blocks_out), intervals=_j(intervals_out))


def main() -> None:
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    rows = [r for r in csv.DictReader(open(HERE / "sessions.csv")) if r["sync_path"]]
    done = set()
    if FILES.exists():
        for line in FILES.read_text().splitlines():
            done.add(json.loads(line)["sync_path"])
    todo = [r for r in rows if r["sync_path"] not in done]
    print(f"{len(rows)} sync files, {len(done)} done, {len(todo)} to do, {n_workers} workers", flush=True)
    t0 = time.time()
    with concurrent.futures.ProcessPoolExecutor(n_workers) as ex, open(FILES, "a") as ff, open(BLOCKS, "a") as fb, open(INTERVALS, "a") as fi:
        futs = {ex.submit(process, r): r for r in todo}
        for i, fut in enumerate(concurrent.futures.as_completed(futs)):
            r = futs[fut]
            try:
                res = fut.result()
            except Exception as exc:  # noqa: BLE001  (worker crash)
                res = dict(file=dict(session=r["session"], sync_path=r["sync_path"], source=r["source"], error=f"worker: {type(exc).__name__}: {exc}"), blocks=[], intervals=[])
            ff.write(json.dumps(res["file"]) + "\n")
            for b in res["blocks"]:
                fb.write(json.dumps(b) + "\n")
            for x in res["intervals"]:
                fi.write(json.dumps(x) + "\n")
            ff.flush(); fb.flush(); fi.flush()
            if i % 50 == 0 or res["file"].get("n_invalid") or res["file"].get("error"):
                f = res["file"]
                print(f"[{i + 1}/{len(todo)} {time.time() - t0:.0f} s] {f['session']}: invalid={f.get('n_invalid')} "
                      f"disc={f.get('n_discontinuities')} blocks={f.get('n_blocks')} "
                      f"classes={[b.get('cls') for b in res['blocks']]} {f.get('error', '')} ({f.get('seconds')} s)", flush=True)
    print(f"done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
