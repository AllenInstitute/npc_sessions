"""Is the 'early' middle barcode of a complementary pair (replay files) a real shift of all
lines? Camera phase-step fit (as step_fit_explore.py) in the window before the early barcode,
with D = -shift, and print replayed intervals from the window start to 10 s after the barcode."""
import logging, sys
import numpy as np, npc_sync
from step_fit_explore import fit, FS, SLOT
logging.disable(logging.WARNING)
W = 5.0
CASES = [  # path, window start, window stop (early barcode onset), D
    ("C:/data/t2p_cache/DRpilot_703880_20240418/20240418T130656.h5", 896.5267, 926.4923, 1.0),
    ("C:/data/t2p_cache/DRpilot_703880_20240418/20240418T130656.h5", 1980.3239, 2009.7895, 1.5),
    ("C:/data/t2p_cache/DRpilot_714748_20240628/20240628T095859.h5", 3553.03264, 3582.9985, 1.0),
    ("C:/data/t2p_cache/DRpilot_714748_20240628/20240628T095859.h5", 5318.08787, 5347.55375, 1.5),
]
for path, a, z, D in CASES:
    s = npc_sync.SyncDataset(path, on_invalid="ignore")
    inv = s.invalid_intervals
    print("=" * 100); print(path.split("/")[3], f"window {a}-{z} D {D}")
    print("  replayed intervals in [a-10, z+10]:", [(round(x, 1), round(y, 1)) for x, y in inv if y > a - 10 and x < z + 10])
    cands = np.arange(np.ceil(a * FS / SLOT), np.floor(z * FS / SLOT) + 1) * SLOT / FS
    for label in [l for l in s.line_labels if l.endswith("_cam_exposing")]:
        t = s.get_rising_edges(label, units="seconds")
        t = t[(t > a - W - 1) & (t < z + W + 1)]
        keep = np.ones(len(t), bool)
        for x, y in inv:
            keep &= ~((t >= x - 0.02) & (t < y + 0.02))
        t = t[keep]
        if len(t) < 200:
            print(f"  {label}: {len(t)} edges"); continue
        P0 = float(np.median(np.diff(t)))
        m = np.r_[0, np.cumsum(np.round(np.diff(t) / P0))]
        P = float((t[-1] - t[0]) / m[-1]); u = t - P * m
        delta = round(D / P) * P - D
        out = []
        for c in cands:
            w = (t > c - W) & (t < c + W)
            if w.sum() < 100: continue
            sse, _ = fit(u[w], m[w], t[w] >= c, delta); sse0, _ = fit(u[w], m[w], t[w] >= c, 0.0)
            out.append((sse, c, sse0))
        out.sort(); (b, cb, b0), (r2, c2, _) = out[0], out[1]
        w = (t > cb - W) & (t < cb + W)
        _, free = fit(u[w], m[w], t[w] >= cb, None)
        print(f"  {label:20s}: pred {delta*1e6:+6.1f} us | best c {cb:8.1f} rms {np.sqrt(b/w.sum())*1e6:5.2f} us runner-up {c2:8.1f} x{r2/b:5.2f} no-step x{b0/b:6.2f} free {free*1e6:+6.1f} us")
