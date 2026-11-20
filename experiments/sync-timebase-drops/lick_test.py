"""Independent test of the displaced-stretch model with licks (stim clock = truth).

Stim file: lick times on the stim clock = cumsum(frameIntervals)[lickFrames]. Sync: lick_sensor
rising edges as recorded. For each replay run with a camera-located stretch [X, R) (stretch_fit),
the recorded-minus-stim-clock offset of the licks is measured in three windows: 10 s before X,
the stretch, 10 s after the run. Model: stretch offset = before/after offset - n * 0.5 s.
Offsets by histogram of all pairwise differences (sync lick - stim lick) within +-3 s, 5 ms bins.
"""
import logging, sys, pathlib, json
import numpy as np, h5py, npc_sync
from stretch_fit import analyse
logging.disable(logging.WARNING)
d = pathlib.Path(sys.argv[1]); task = next(d.glob("DynamicRouting1_*.hdf5")); syncp = next(d.glob("2*.h5"))
s = npc_sync.SyncDataset(syncp, on_invalid="ignore")
with h5py.File(task, "r") as f:
    fi = f["frameIntervals"][:]; lf = np.asarray(f["lickFrames"][:], dtype=int)
tau = np.r_[0, np.cumsum(fi)][lf]
blocks = s.vsync_times_in_blocks
bi = int(np.argmax([len(b) for b in blocks])); v0 = blocks[bi][0]; vend = blocks[bi][-1]
licks = s.get_rising_edges("lick_sensor", units="seconds")
print(d.name, "task block", bi, round(v0, 2), round(vend, 2), "stim frames", len(fi) + 1, "vsyncs", len(blocks[bi]), "stim licks", len(tau), "sync licks in block", int(((licks > v0) & (licks < vend)).sum()))
def offset(a, z):
    L = licks[(licks >= a) & (licks < z)]
    if len(L) == 0: return None, 0, 0
    diffs = (L[:, None] - (v0 + tau)[None, :]).ravel(); diffs = diffs[np.abs(diffs) < 3]
    if len(diffs) == 0: return None, len(L), 0
    h, e = np.histogram(diffs, bins=np.arange(-3, 3.005, 0.005)); j = int(np.argmax(h))
    sel = diffs[(diffs >= e[j] - 0.005) & (diffs < e[j + 1] + 0.005)]
    return float(np.median(sel)), len(L), int(h[j])
for r in analyse(str(syncp)):
    X, R, n = r["X"], r["R"], r["n_chunks"]; Z = R + n * 0.5
    if not (v0 < X - 10 and Z + 10 < vend) or r["lag_s"] == 0: continue
    ob, nb, hb = offset(X - 10, X); os_, ns, hs = offset(X, R); oa, na, ha = offset(Z, Z + 10)
    f = lambda o: "   none" if o is None else f"{o:+.4f}"
    exp = None if ob is None else ob - n * 0.5
    print(f" run {r['run']:3d} X {X:7.1f} R {R:7.1f} n {n} lag {r['lag_s']:4.1f} dchi2 {r['dchi2_vs_null']:8.0f} | before {f(ob)} ({hb}/{nb} licks)  stretch {f(os_)} ({hs}/{ns})  after {f(oa)} ({ha}/{na}) | model predicts stretch {f(exp)}")
