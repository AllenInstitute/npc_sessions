"""Audit which sync lines carry events inside `SyncDataset.invalid_intervals`, and which
current code paths are guarded by the npc_sync 0.2.7 fail-fast.

Data: local copies of 7 Task 2 pilot sessions (sync + stim files) in
npc_sync/experiments/vsync-frame-count/data/ (see npc_sync/outputs/vsync-frame-count-discrepancies.md).

Run with the npc_sync dev venv (editable npc_sync @ HEAD 0.2.8 + npc_stim 0.1.23):
    ../../../npc_sync/.venv/Scripts/python.exe line_audit.py > line_audit.log 2>&1

Checks (defined before running):
  C1 invalid_intervals found for the 6 replay sessions, none for 840160 (drop, not replay).
  C2 per-line edge counts inside invalid intervals (which lines need reconstruction).
  C3 is frame_display_time_blocks guarded? (expect AssertionError when intervals exist)
  C4 is vsync_times_in_blocks guarded? (expect: NOT guarded, returns arrays)
  C5 npc_stim.get_stim_frame_times(..., 'vsync') per stim: array / exception / missing key.
  C6 barcode starts (sync) within 1 s of an invalid interval.
"""

from __future__ import annotations

import logging
import pathlib
import sys

import numpy as np

import npc_stim
import npc_sync

logging.disable(logging.WARNING)

DATA = pathlib.Path(__file__).resolve().parents[3] / "npc_sync" / "experiments" / "vsync-frame-count" / "data"


def in_any(t: np.ndarray, intervals) -> np.ndarray:
    m = np.zeros(len(t), bool)
    for a, z in intervals:
        m |= (t >= a) & (t < z)
    return m


def main() -> None:
    print("npc_sync source:", npc_sync.__file__)
    print("npc_stim:", npc_stim.__file__)
    for sess_dir in sorted(p for p in DATA.iterdir() if p.is_dir()):
        sync_path = next(sess_dir.glob("*T*.h5"))
        stim_paths = sorted(sess_dir.glob("*.hdf5"))
        print(f"\n===== {sess_dir.name}  sync={sync_path.name}  n_stim={len(stim_paths)}")
        s = npc_sync.SyncDataset(sync_path)
        iv = s.invalid_intervals
        print(f"C1 invalid_intervals: {[(round(a, 3), round(z, 3)) for a, z in iv]}")

        # C2: per-line edge counts inside the invalid intervals
        if iv:
            rows = []
            for bit, label in enumerate(s.line_labels):
                if not label:
                    continue
                r = s.get_rising_edges(bit, units="seconds")
                f = s.get_falling_edges(bit, units="seconds")
                nr, nf = int(in_any(r, iv).sum()), int(in_any(f, iv).sum())
                if nr or nf:
                    rows.append(f"{label}(bit{bit}): rise={nr} fall={nf}")
            print("C2 lines with edges in invalid intervals:", "; ".join(rows) or "none")

        # C3: frame_display_time_blocks guard
        s3 = npc_sync.SyncDataset(sync_path, stim_paths=stim_paths)
        try:
            _ = s3.frame_display_time_blocks
            print("C3 frame_display_time_blocks: returned (no guard triggered)")
        except AssertionError as exc:
            print(f"C3 frame_display_time_blocks: AssertionError ({str(exc)[:90]}...)")
        except Exception as exc:  # noqa: BLE001
            print(f"C3 frame_display_time_blocks: {type(exc).__name__}: {str(exc)[:120]}")

        # C4: vsync_times_in_blocks guard + which blocks overlap intervals
        try:
            blocks = s.vsync_times_in_blocks
            ov = [
                i
                for i, b in enumerate(blocks)
                if any((b[0] <= z) and (b[-1] >= a) for a, z in iv)
            ]
            print(
                f"C4 vsync_times_in_blocks: returned {len(blocks)} blocks, lens={[len(b) for b in blocks]}; "
                f"blocks overlapping invalid intervals={ov}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"C4 vsync_times_in_blocks: {type(exc).__name__}: {str(exc)[:120]}")

        # C5: npc_stim vsync path (used by npc_samstim latencies, _flip_times, _input_data_times)
        try:
            res = npc_stim.get_stim_frame_times(*stim_paths, sync=sync_path, frame_time_type="vsync")
            for p in stim_paths:
                v = res.get(p, "MISSING KEY")
                if isinstance(v, np.ndarray):
                    n_stim = npc_stim.get_total_stim_frames(p)
                    hit = bool(iv) and bool(in_any(v, iv).any())
                    desc = f"array len={len(v)} (stim frames={n_stim}){' OVERLAPS INVALID INTERVAL' if hit else ''}"
                elif isinstance(v, Exception):
                    desc = f"{type(v).__name__}: {str(v)[:80]}"
                else:
                    desc = str(v)
                print(f"C5 {p.stem}: {desc}")
        except Exception as exc:  # noqa: BLE001
            print(f"C5 get_stim_frame_times(vsync): {type(exc).__name__}: {str(exc)[:120]}")

        # C6: barcodes near invalid intervals
        try:
            import npc_ephys  # noqa: PLC0415

            on = s.get_rising_edges("barcode_ephys", units="seconds")
            off = s.get_falling_edges("barcode_ephys", units="seconds")
            t, ids = npc_ephys.extract_barcodes_from_times(on, off, total_time_on_line=s.total_seconds)
            near = [
                (round(float(tt), 3), int(i))
                for tt, i in zip(t, ids)
                if any((tt > a - 1.0) and (tt < z + 1.0) for a, z in iv)
            ]
            print(f"C6 sync barcodes: n={len(t)}; starting within 1 s of an invalid interval: {near}")
        except ImportError:
            # npc_ephys not installed in this venv: fall back to a crude start detector (>29 s gaps)
            on = s.get_rising_edges("barcode_ephys", units="seconds")
            starts = on[np.r_[True, np.diff(on) > 29]]
            near = [round(float(tt), 3) for tt in starts if any((tt > a - 1.0) and (tt < z + 1.0) for a, z in iv)]
            print(f"C6 sync barcode starts (crude): n={len(starts)}; within 1 s of an invalid interval: {near}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
