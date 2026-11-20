"""Shift state of every 0.5 s chunk of a sync file from camera phase (prototype).

Model: recorded chunk p holds the data of true chunk p + s(p). s jumps by +k where k chunks
were dropped, and falls by 1 per chunk through a run of replayed copies (each copy holds the
chunk that follows the run). A whole chunk (0.5 s) is not a whole number of camera periods:
it is 30 periods minus ~12-26 us (delta1), so the camera phase residual of a chunk is
s(p) * delta1 relative to on-time data, and the jump J(p) = s(p) - s(p - 1) is read from the
difference of consecutive chunk means.

Per exposing line and edge kind: u = t - P * m (m: cumulative period count), chunk mean
u(p) over chunks whose intervals are all regular; D(p) = u(p) - u(p - 1) minus its rolling
median (clock drift); d(p) = D(p) / delta1. Lines are combined by the median.
"""
from __future__ import annotations
import logging, sys, json, pathlib
import numpy as np, npc_sync
logging.disable(logging.WARNING)
CH = 50_000
FS = 100_000.0

def line_chunk_means(t_samples: np.ndarray, n_chunks: int, ch: int = CH):
    """-> (u mean per chunk [s], NaN where the chunk has < 20 edges or an irregular interval), P, delta1"""
    t = np.asarray(t_samples, dtype=np.int64)
    out = np.full(n_chunks, np.nan)
    if len(t) < 1000:
        return out, np.nan, np.nan
    dt = np.diff(t)
    P0 = float(np.median(dt))
    regular = np.abs(dt - P0) < 0.006 * P0  # 0.1 ms of 16.7 ms
    k = np.round(dt / P0).astype(np.int64)
    k[~regular] = 0  # index restarts are handled by segment: D is not taken across them
    seg = np.r_[0, np.cumsum(~regular)]
    m = np.r_[0, np.cumsum(np.where(regular, 1, 0))]
    # period from regular intervals only
    P = float(dt[regular].mean())
    u = t - P * m - np.r_[0, np.cumsum(np.where(regular, 0, dt))]  # irregular intervals removed from u
    chunk = t // ch
    order_ok = np.r_[True, regular] & np.r_[regular, True]  # edge has regular intervals on both sides
    cnt = np.bincount(chunk, minlength=n_chunks)[:n_chunks]
    bad = np.bincount(chunk, weights=~order_ok, minlength=n_chunks)[:n_chunks]
    s = np.bincount(chunk, weights=u - u[0], minlength=n_chunks)[:n_chunks]
    segmin = np.full(n_chunks, np.iinfo(np.int64).max); np.minimum.at(segmin, chunk, seg)
    segmax = np.full(n_chunks, -1); np.maximum.at(segmax, chunk, seg)
    ok = (cnt >= 20) & (bad == 0) & (segmin == segmax)
    out[ok] = s[ok] / cnt[ok] / FS
    segid = np.where(ok, segmin, -1)
    delta1 = 30 * P / FS - ch / FS if abs(30 * P - ch) < 0.2 * P else np.nan  # generalise below
    q = round(ch / P); delta1 = (q * P - ch) / FS
    return out, P / FS, delta1, segid

def jumps(sync, ch: int = CH, drift_window: int = 201):
    n_chunks = int(np.ceil(sync.meta_data["total_samples"] / ch))
    labels = [l for l in sync.line_labels if l.endswith("_cam_exposing")]
    ds = []; names = []
    for l in labels:
        for kind in ("rising", "falling"):
            t = sync._get_recorded_edges(l, kind)
            r = line_chunk_means(t, n_chunks, ch)
            if len(r) == 3: continue
            u, P, d1, seg = r
            D = np.full(n_chunks, np.nan)
            same = (seg[1:] == seg[:-1]) & (seg[1:] >= 0)
            D[1:][same] = (u[1:] - u[:-1])[same]
            # drift: rolling median of D
            import pandas as pd
            med = pd.Series(D).rolling(drift_window, center=True, min_periods=20).median().to_numpy()
            ds.append((D - med) / d1); names.append(f"{l}:{kind[0]}")
    d = np.array(ds)
    return d, names

if __name__ == "__main__":
    path = sys.argv[1]
    s = npc_sync.SyncDataset(path, on_invalid="ignore")
    d, names = jumps(s)
    n_lines = np.sum(np.isfinite(d), axis=0)
    med = np.nanmedian(d, axis=0)
    spread = np.nanmax(np.abs(d - med), axis=0)
    J = np.round(med)
    print(path)
    print(" lines", names)
    print(" chunks with >= 2 lines:", int((n_lines >= 2).sum()), "of", d.shape[1])
    resid = med - J
    ok = n_lines >= 2
    print(" |d - round(d)| percentiles (50, 99, 99.9, max):", np.round(np.nanpercentile(np.abs(resid[ok]), [50, 99, 99.9, 100]), 3))
    print(" line spread percentiles (50, 99, 99.9, max):", np.round(np.nanpercentile(spread[ok], [50, 99, 99.9, 100]), 3))
    inv = s.invalid_intervals
    print(" replay runs:", len(inv))
    nz = np.flatnonzero(ok & (J != 0))
    print(" nonzero jumps:", len(nz))
    copy_bounds = {}
    for a, z in inv:
        for b in np.arange(round(a * 2) + 1, round(z * 2) + 1): copy_bounds[int(b)] = (a, z)
    for p in nz:
        tag = "copy" if int(p) in copy_bounds else ""
        print(f"   boundary {p/2:8.1f} s  J {int(J[p]):+d}  d {med[p]:+.2f}  spread {spread[p]:.2f} lines {n_lines[p]} {tag}")
    missing = [b / 2 for b in copy_bounds if not (ok[b] and J[b] == -1)]
    print(" copy boundaries without J = -1:", missing[:20], len(missing))
