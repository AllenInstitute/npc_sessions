"""Displaced-stretch model for a replay run: n chunks are dropped at X <= R (run start), so
recorded [X, R) holds data from n chunks later (events early by n*0.5 s), and the n copies at
[R, R + n*0.5) restore the timebase. Camera phase: u steps by delta = q*P - D for X <= t < R.

For each replay run (R, n) in the given files: SSE of the camera lines for every X on the
0.1 s grid in [R - BACK, R] (X = R: no displaced stretch), summed over exposing lines
(rising + falling), in units of each line's noise variance at its own best X.
"""
from __future__ import annotations
import logging, sys, json, pathlib
import numpy as np, npc_sync
logging.disable(logging.WARNING)
FS, SLOT = 100_000, 10_000
BACK, AFTER = 40.0, 5.0

def line_costs(t, R, Z, D, cands):
    """t: edges of one line/kind (replayed interval removed). -> (sse per cand, sse null, n, P, delta)"""
    P0 = float(np.median(np.diff(t)))
    m = np.r_[0, np.cumsum(np.round(np.diff(t) / P0))]
    post = t >= Z
    pre = t < R
    # period from a robust fit of the post-run + far-pre data is not needed: slope is fitted
    P = float((t[-1] - t[0]) / m[-1]); u = t - P * m
    delta = round(D / P) * P - D
    A = np.c_[np.ones(len(t)), m]
    def sse(y):
        coef, *_ = np.linalg.lstsq(A, y, rcond=None); r = y - A @ coef; return float(r @ r)
    null = sse(u)
    out = np.array([sse(u - delta * ((t >= c) & pre)) for c in cands])
    return out, null, len(t), P, delta

def analyse(path, only=None, back=BACK):
    s = npc_sync.SyncDataset(path, on_invalid="ignore")
    inv = list(s.invalid_intervals)
    labels = [l for l in s.line_labels if l.endswith("_cam_exposing")]
    edges = {(l, k): (s.get_rising_edges if k == "r" else s.get_falling_edges)(l, units="seconds") for l in labels for k in "rf"}
    res = []
    for i, (R, Z) in enumerate(inv):
        if only is not None and i not in only: continue
        D = Z - R
        lo = max(R - back, inv[i - 1][1] if i else 0.0)  # don't look past the previous run
        cands = np.arange(np.ceil(lo * FS / SLOT), np.floor(R * FS / SLOT) + 1) * SLOT / FS
        if len(cands) < 2: continue
        total = np.zeros(len(cands)); null_total = 0.0; nlines = 0; deltas = []
        for key, t in edges.items():
            t = t[(t > lo - 2) & (t < Z + AFTER)]
            t = t[~((t >= R - 0.02) & (t < Z + 0.02))]
            for a, z in inv:  # other runs in the window
                if (a, z) != (R, Z): t = t[~((t >= a - 0.02) & (t < z + 0.02))]
            if len(t) < 300 or (t < R).sum() < 100 or (t >= Z).sum() < 100: continue
            c, null, n, P, delta = line_costs(t, R, Z, D, cands)
            var = c.min() / n
            total += c / var; null_total += c[-1] / var; nlines += 1; deltas.append(delta * 1e6)
        if not nlines: continue
        j = int(np.argmin(total))
        order = np.argsort(total)
        res.append(dict(run=i, R=R, n_chunks=round(D / 0.5), X=float(cands[j]), lag_s=round(R - cands[j], 1),
                        dchi2_vs_null=round(float(total[-1] - total[j]), 1), dchi2_runner_up=round(float(total[order[1]] - total[j]), 1),
                        runner_up=float(cands[order[1]]), nlines=nlines, pred_step_us=[round(x, 1) for x in deltas[::2]]))
    return res

if __name__ == "__main__":
    for path in sys.argv[1:]:
        print(path)
        for r in analyse(path): print("  ", json.dumps(r))
