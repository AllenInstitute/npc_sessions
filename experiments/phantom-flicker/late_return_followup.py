"""Follow-up summaries of late_return_check.csv.gz (see late_return_check.py).

1. delay histogram of the 'stim on grid' class (PsychoPy's frame interval regular)
2. the 'stim == sync' candidates with a delay < 2 ms (mostly in diode-lost blocks)
3. frame counts: sync intervals vs display (from the diode toggles), per class
4. controls that weren't 'on time', by session
5. the main result without the 11 blocks the patch rejects (diode signal lost)

Run: ../../.venv/Scripts/python.exe late_return_followup.py > late_return_followup.log
"""

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
d = pd.read_csv("late_return_check.csv.gz", low_memory=False)
c = d[d.kind == "candidate"].copy()
c["stim_type"] = c.stim.str.split("_").str[0]
g = c[c.c1_class == "stim on grid"]
print("1. 'stim on grid' delay (ms) histogram:")
h = np.histogram(g.return_delay_ms.clip(0, 25), bins=[0, 1.67, 2.5, 3.5, 4.4, 4.5, 4.6, 4.7, 4.8, 5.0, 5.5, 10, 25])
for lo, hi, n in zip(h[1][:-1], h[1][1:], h[0]):
    print(f"   {lo:5.2f}-{hi:5.2f}: {n}")
print(f"   sessions with >=1: {g.folder.nunique()} of {c.folder.nunique()}; per-session median delay quantiles: "
      f"{g.groupby('folder').return_delay_ms.median().quantile([0.05, 0.5, 0.95]).round(2).to_dict()}")
w = c[(c.c1_class == "stim == sync") & (c.return_delay_ms < 2)]
print(f"\n2. 'stim == sync' with delay < 2 ms: {len(w)}; stim types {w.stim_type.value_counts().to_dict()}")
print("   status:", w.status_ttl.value_counts().to_dict())
print("   n_rise quantiles:", w.n_rise.quantile([0.05, 0.5, 0.95]).round(2).to_dict(),
      "(controls:", d[d.kind == "control"].n_rise.quantile([0.05, 0.5, 0.95]).round(2).to_dict(), ")")
print("   (n_prev, n_next):", w.groupby(["n_prev", "n_next"]).size().sort_values(ascending=False).head(5).to_dict())
print("   top sessions:", w.folder.value_counts().head(5).to_dict())
print("\n3. C5 by class (sync intervals -> display intervals), top patterns:")
for cls in ("stim == sync", "stim on grid"):
    x = c[c.c1_class == cls]
    print(f"  {cls}:", x.groupby(["n_prev", "n_next", "disp_prev", "disp_next"]).size().sort_values(ascending=False).head(6).to_dict())
x = c[(c.c1_class == "stim == sync") & c.disp_prev.notna() & c.disp_next.notna()]
long_stim = x[x.n_prev >= 2]
print(f"  'stim == sync' with a long stim interval before frame j (n_prev >= 2): {len(long_stim)}; display: no drop at all "
      f"{int(((long_stim.disp_prev == 1) & (long_stim.disp_next == 1)).sum())}, drop one frame later than the stim says "
      f"{int(((long_stim.disp_prev == 1) & (long_stim.disp_next >= 2)).sum())}, same place {int((long_stim.disp_prev == long_stim.n_prev).sum())}")
print("\n4. controls not on time:",
      d[(d.kind == "control") & (d.status_ttl != "on time")].groupby("folder").size().sort_values(ascending=False).head(5).to_dict())

cp = pd.read_csv("compare_population.csv")
rej = set(map(tuple, cp[cp.diode_status == "rejected"][["folder", "block"]].values))
d["rejected_block"] = [(f, b) in rej for f, b in zip(d.folder, d.block)]
print("\n5. excluding the 11 blocks the patch rejects (diode signal lost):")
x = d[~d.rejected_block]
c, k = x[x.kind == "candidate"], x[x.kind == "control"]
print(f"   candidates {len(c)} (removed {int((d.kind == 'candidate').sum()) - len(c)}), controls {len(k)}")
for cls in ("stim on grid", "stim == sync", "other"):
    y = c[c.c1_class == cls]
    print(f"   {cls:13s} n={len(y):6d}  " + ", ".join(f"{s}: {n}" for s, n in y.status_ttl.value_counts().items()))
print(f"   all candidates: on time {int((c.status_ttl == 'on time').sum())}/{len(c)} = {100 * (c.status_ttl == 'on time').mean():.2f}%; "
      f"follows the late fall {int((c.status_ttl == 'follows the late fall').sum())}; displayed after flip() returned {int(c.after_return_ttl.sum())}")
print(f"   controls: on time {int((k.status_ttl == 'on time').sum())}/{len(k)} = {100 * (k.status_ttl == 'on time').mean():.2f}%")
