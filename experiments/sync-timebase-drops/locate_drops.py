"""Test the buffer model for sync timebase drops (see README.md).

For every timebase discontinuity in the "clean" files (no replayed intervals), check:
  Q1 shift is a whole number of 0.5 s (or 0.1 s) buffers
  Q2 vsync + stim frameIntervals: cost of a drop at every 0.1 s grid boundary in the barcode
     window (locate_drop's cost, but all candidates reported, two window sizes) -> best,
     runner-up, margin; is the best on the 0.5 s grid?
  Q3 camera lines: is the drop visible as a phase slip (interval residual modulo the frame
     period) at the same boundary, and is the expected slip above the jitter?
  Q4 raw events: no gap in sample numbers at the boundary; edge-polarity alternation on every
     line across it.
Q5 (corpus) uses the P0 manifest: shift quantization for all 11 files with discontinuities.

Read-only on the share. Run with the npc_sessions venv:
    ../../.venv/Scripts/python.exe locate_drops.py > locate_drops.log 2>&1
"""

from __future__ import annotations

import json
import logging
import pathlib
import sys

import numpy as np

import npc_sync

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "sync-patch-manifest"))
import common  # noqa: E402

logging.disable(logging.WARNING)

HERE = pathlib.Path(__file__).parent
MANIFEST = HERE.parent / "sync-patch-manifest" / "manifest_files.jsonl"
GRID = 0.1  # finest buffer size seen (10,000 samples at 100 kHz)
COARSE = 0.5  # 50,000 samples
WINDOWS = (600, 30)  # frames either side of a candidate boundary


def manifest_rows() -> list[dict]:
    return [json.loads(l) for l in MANIFEST.open() if json.loads(l)["n_discontinuities"]]


def q5_quantization(rows: list[dict]) -> None:
    print("=" * 100)
    print("Q5 shift quantization, all files with discontinuities (P0 manifest)")
    print(f"{'session':28s} {'n_inv':>5s} {'start':>9s} {'stop':>9s} {'shift':>9s} {'/0.5':>7s} {'/0.1':>7s}")
    for d in rows:
        for a, z, s in d["discontinuities"]:
            print(
                f"{d['session']:28s} {d['n_invalid']:5d} {a:9.2f} {z:9.2f} {s:+9.4f} "
                f"{s / COARSE - round(s / COARSE):+7.3f} {s / GRID - round(s / GRID):+7.3f}"
            )
    print("(columns /0.5, /0.1: shift in buffers minus nearest integer)")


def vsync_costs(v: np.ndarray, fi: np.ndarray, n_missing: int, a: float, z: float, w: int):
    vi = np.diff(v)
    out = []
    for t in np.arange(np.ceil(a / GRID), np.floor(z / GRID) + 1) * GRID:
        i0 = int(np.searchsorted(v, t))
        if i0 < w or i0 + w + n_missing + 1 >= len(v):
            continue
        before = np.abs(vi[i0 - w : i0 - 1] - fi[i0 - w : i0 - 1])
        after = np.abs(vi[i0 : i0 + w] - fi[i0 + n_missing : i0 + n_missing + w])
        out.append((round(float(t), 1), float(np.mean(before) + np.mean(after)) * 1e3))
    return out


def q2_vsync(sync, stims, a, z, shift) -> float | None:
    blocks = list(sync.vsync_times_in_blocks)
    matches, _ = common.match_stims_to_blocks(blocks, stims)
    for m in matches:
        v = blocks[m.block_index]
        if not (v[0] < a and v[-1] > z):
            continue
        others = [d for d in sync.timebase_discontinuities if d[0] != a and v[0] < d[0] and v[-1] > d[1]]
        if others:
            print(f"  Q2 block {m.block_index} also spans discontinuities {others}: single-drop fit invalid, skipped")
            return None
        if m.stim is None:
            print(f"  Q2 block {m.block_index} spans window but no stim matched ({m.reason})")
            return None
        fi = m.stim.frame_intervals
        n_missing = len(fi) + 1 - len(v)
        dur_def = fi.sum() - (v[-1] - v[0])
        print(
            f"  Q2 block {m.block_index} ({m.stim.name}): {n_missing} vsyncs missing, "
            f"stim-minus-sync duration {dur_def:+.4f} s (barcode shift {shift:+.4f})"
        )
        if n_missing <= 0:
            print("     not a drop in this block's frame count: skipped")
            return None
        best_t = None
        for w in WINDOWS:
            costs = sorted(vsync_costs(v, fi, n_missing, a, z, w), key=lambda x: x[1])
            if len(costs) < 2:
                print(f"     w={w}: too few candidates")
                continue
            (t1, c1), (t2, c2) = costs[0], costs[1]
            on_coarse = abs(t1 / COARSE - round(t1 / COARSE)) < 1e-6
            median = float(np.median([c for _, c in costs]))
            print(
                f"     w={w:3d}: {len(costs)} candidates; best {t1:.1f} s ({c1:.4f} ms), "
                f"runner-up {t2:.1f} s ({c2:.4f} ms), ratio {c2 / c1:.2f}, median {median:.4f} ms; "
                f"best on 0.5 s grid: {on_coarse}"
            )
            print(f"            5 best: {costs[:5]}")
            best_t = best_t if best_t is not None else t1
        return best_t
    print("  Q2 no single vsync block spans the barcode window")
    return None


def q3_cameras(sync, a, z, shift, t_best) -> float | None:
    labels = [l for l in sync.line_labels if "cam" in l]
    if not labels:
        print("  Q3 no camera lines")
        return
    tops_t: list[float] = []
    for label in labels:
        r = sync.get_rising_edges(label, units="seconds")
        r = r[(r > a - 5) & (r < z + 5)]
        if len(r) < 100:
            print(f"  Q3 {label:24s}: {len(r)} edges in window, skipped")
            continue
        iv = np.diff(r)
        # period from the 5 s before the window (median of intervals is quantized to 10 us)
        pre = sync.get_rising_edges(label, units="seconds")
        pre = pre[(pre > a - 5) & (pre < a)]
        period = float((pre[-1] - pre[0]) / np.round((pre[-1] - pre[0]) / np.median(np.diff(pre))))
        res = iv - np.round(iv / period) * period
        mad = float(np.median(np.abs(res - np.median(res)))) * 1e3
        expected = (shift - round(shift / period) * period) * 1e3
        order = np.argsort(-np.abs(res))[:3]
        tops = [(round(float(r[i]), 3), round(float(res[i]) * 1e3, 4)) for i in order]
        hit = ""
        if t_best is not None:
            i = int(np.searchsorted(r, t_best)) - 1  # interval straddling t_best
            if 0 <= i < len(res):
                rank = int(np.sum(np.abs(res) > abs(res[i]))) + 1
                hit = f"; interval straddling {t_best:.1f}: {res[i] * 1e3:+.4f} ms (rank {rank}/{len(res)})"
        tops_t.append(float(r[order[0]]))
        print(
            f"  Q3 {label:24s}: period {period * 1e3:.5f} ms, jitter MAD {mad:.4f} ms, "
            f"expected slip {expected:+.4f} ms ({abs(expected) / max(mad, 1e-9):.1f} MAD); "
            f"top |res| (t, ms): {tops}{hit}"
        )
    if tops_t:
        snapped = [round(t / GRID) * GRID for t in tops_t]
        vals, counts = np.unique(np.round(snapped, 1), return_counts=True)
        c = float(vals[np.argmax(counts)])
        print(
            f"  Q3 camera consensus: top anomaly of {counts.max()}/{len(tops_t)} lines snaps to {c:.1f} s "
            f"(on 0.5 s grid: {abs(c / COARSE - round(c / COARSE)) < 1e-6}); all: {sorted(np.round(tops_t, 3).tolist())}"
        )
        return c
    return None


def q4_raw(sync, t):
    fs = sync.sample_freq
    times = sync.get_all_times("samples")
    s0 = int(round(t * fs))
    i = int(np.searchsorted(times, s0))
    gap_before = (times[i] - times[i - 1]) / fs * 1e3 if 0 < i < len(times) else np.nan
    local = np.diff(times[max(i - 2000, 0) : i + 2000]) / fs * 1e3
    print(
        f"  Q4 boundary sample {s0} (mod 10000 = {s0 % 10000}, mod 50000 = {s0 % 50000}); "
        f"all-line event gap straddling it {gap_before:.3f} ms; local max gap {local.max():.3f} ms, "
        f"median {np.median(local):.3f} ms"
    )
    bad = []
    for label in sync.line_labels:
        if not label:
            continue
        try:
            ri = sync.get_rising_edges(label, units="seconds")
            fa = sync.get_falling_edges(label, units="seconds")
        except Exception:  # noqa: BLE001
            continue
        ev = np.r_[ri, fa]
        pol = np.r_[np.ones(len(ri)), np.zeros(len(fa))][np.argsort(ev)]
        ev = np.sort(ev)
        k = int(np.searchsorted(ev, t))
        if 0 < k < len(ev) and pol[k] == pol[k - 1]:
            bad.append(f"{label} ({ev[k - 1]:.4f}->{ev[k]:.4f})")
    print(f"  Q4 lines with same-polarity edges either side of boundary: {bad or 'none'}")


def main() -> None:
    rows = manifest_rows()
    q5_quantization(rows)
    clean = [d for d in rows if d["n_invalid"] == 0]
    print("=" * 100)
    print(f"Q1-Q4 on files with discontinuities and no replays: {[d['session'] for d in clean]}")
    for d in clean:
        path = pathlib.Path(d["sync_path"])
        print("-" * 100)
        print(d["session"], path)
        sync = npc_sync.SyncDataset(path)
        start = common.sync_start(sync)
        stims = [common.read_stim(p, start) for p in sorted(path.parent.glob("*.hdf5"))]
        for a, z, shift in sync.timebase_discontinuities:
            k = shift / COARSE
            print(
                f" discontinuity {a:.3f}-{z:.3f} s, shift {shift:+.5f} s: Q1 {k:+.4f} x 0.5 s buffers "
                f"(off by {abs(k - round(k)) * COARSE * 1e3:.3f} ms)"
            )
            t_best = q2_vsync(sync, stims, a, z, shift)
            t_cam = q3_cameras(sync, a, z, shift, t_best)
            if t_best is not None and t_cam is not None:
                print(f"  vsync vs camera: {t_best:.1f} vs {t_cam:.1f} s -> {'AGREE' if abs(t_best - t_cam) < 1e-6 else 'DISAGREE'}")
            for t in {x for x in (t_best, t_cam) if x is not None}:
                q4_raw(sync, t)


if __name__ == "__main__":
    main()
