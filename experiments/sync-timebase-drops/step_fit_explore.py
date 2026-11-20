"""Explore: camera phase-step fit to locate timebase drops (see README, 'step fit').

For each drop in the 4 no-replay files: per exposing line and edge kind, u_i = t_i - P*m_i
(m_i: cumulative integer period count), linear fit in a +-W s window around each candidate
boundary c on the 10,000-sample grid inside the barcode window, with a step of known size
delta = q*P - D at c (D = k*0.5 s from the barcode shift, q = round(D / P)).
Reports per line: best c, SSE ratio runner-up/best, SSE(no step)/best, free-step estimate at
best c vs predicted delta.
"""
from __future__ import annotations
import json, logging, pathlib, sys
import numpy as np
import npc_sync
logging.disable(logging.WARNING)
HERE = pathlib.Path(__file__).parent
MANIFEST = HERE.parent / "sync-patch-manifest" / "manifest_files.jsonl"
FS = 100_000
SLOT = 10_000
W = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0

def fit(u, m, step_mask, delta=None):
    """LSQ of u = a + b*m (+ delta*step_mask). delta None: free step. -> (sse, step)"""
    if delta is None:
        A = np.c_[np.ones_like(m, float), m, step_mask.astype(float)]
        coef, *_ = np.linalg.lstsq(A, u, rcond=None)
        r = u - A @ coef
        return float(r @ r), float(coef[2])
    y = u - delta * step_mask
    A = np.c_[np.ones_like(m, float), m]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ coef
    return float(r @ r), delta

def main():
    rows = [json.loads(l) for l in MANIFEST.open()]
    rows = [d for d in rows if d["n_discontinuities"] and d["n_invalid"] == 0]
    for d in rows:
        s = npc_sync.SyncDataset(d["sync_path"], on_invalid="ignore")
        print("=" * 110); print(d["session"])
        for a, z, shift in s.timebase_discontinuities:
            D = round(-shift * FS / SLOT) * SLOT / FS
            print(f" window {a:.3f}-{z:.3f} shift {shift:+.5f} -> D {D}")
            cands = np.arange(np.ceil(a * FS / SLOT), np.floor(z * FS / SLOT) + 1) * SLOT / FS
            for label in [l for l in s.line_labels if l.endswith("_cam_exposing")]:
                for kind in ("rising", "falling"):
                    t = (s.get_rising_edges if kind == "rising" else s.get_falling_edges)(label, units="seconds")
                    t = t[(t > a - W - 1) & (t < z + W + 1)]
                    if len(t) < 200:
                        print(f"  {label:20s} {kind:7s}: {len(t)} edges, skipped"); continue
                    P0 = float(np.median(np.diff(t)))
                    m = np.r_[0, np.cumsum(np.round(np.diff(t) / P0))]
                    P = float((t[-1] - t[0]) / m[-1])  # ignores the step: error ~ delta / n, fitted out by b
                    u = t - P * m
                    q = round(D / P); delta = q * P - D  # recorded times after c are early by D; index continues -> u steps by +delta... sign checked below
                    out = []
                    for c in cands:
                        w = (t > c - W) & (t < c + W)
                        if w.sum() < 100: continue
                        sse, _ = fit(u[w], m[w], (t[w] >= c), delta)
                        sse0, _ = fit(u[w], m[w], (t[w] >= c), 0.0)
                        out.append((sse, c, sse0))
                    out.sort()
                    (b, cb, b0), (r2, c2, _) = out[0], out[1]
                    w = (t > cb - W) & (t < cb + W)
                    _, free = fit(u[w], m[w], (t[w] >= cb), None)
                    rms = np.sqrt(b / w.sum()) * 1e6
                    print(f"  {label:20s} {kind:7s}: P {P*1e3:.5f} ms pred step {delta*1e6:+7.1f} us | best c {cb:9.1f} rms {rms:5.2f} us; "
                          f"runner-up {c2:9.1f} x{r2/b:6.2f}; no-step x{b0/b:7.2f}; free step {free*1e6:+7.1f} us (diff {1e6*(free-delta):+5.1f})")


if __name__ == '__main__':
    main()
