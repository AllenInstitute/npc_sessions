import sys, pickle, numpy as np
from common import *
from npc_sessions import timing
sid = sys.argv[1]
raw, stim_paths, base = load(sid)
d = pickle.load(open(CACHE / f"{sid}_current.pkl", "rb"))
rr, ff = edges(raw); dt = 1/raw.sample_freq
spans = sorted(s.span for s in base.lines[LICK])
print("all lick spans == invalid intervals?", spans == sorted(map(tuple, raw.invalid_intervals)))
print("span lengths", sorted(set(round(z-a, 4) for a, z in spans)))
faults = base.faults
print("fault kinds", {k: sum(1 for f in faults if f.kind == k) for k in ("replay", "drop", "timebase_discontinuity")})
for a, z in spans:
    for t, k in [(x, "R") for x in rr] + [(x, "F") for x in ff]:
        if a - 3*dt <= t <= a + 3*dt: print(f"edge {k} at span start {a}: offset {(t-a)/dt:+.2f} samples")
        if z - 3*dt <= t <= z + 3*dt: print(f"edge {k} at span end {z}: offset {(t-z)/dt:+.2f} samples")
# groups: consecutive spans with no recorded lick edge between them (measured as patch.apply keeps them)
outside = lambda t: t[~npc_sync.in_intervals(t, spans)]
mr, mf = outside(rr), outside(ff)
allm = np.sort(np.r_[mr, mf])
groups = [[spans[0]]]
for s in spans[1:]:
    prev = groups[-1][-1]
    if np.any((allm >= prev[1]) & (allm < s[0])): groups.append([s])
    else: groups[-1].append(s)
print("n spans", len(spans), "n groups", len(groups), "groups with >1 span", sum(len(g) > 1 for g in groups), "max len", max(map(len, groups)))
# fake onsets per span diag
sp = d["info"]["spans"]
rec = [x for x in sp if x["status"] == "reconstructed"]
print("no-onset spans, high_at_exit & ~entry:", sum(1 for x in rec if x["n_onsets"] == 0 and x["high_at_exit"] and not x["high_at_entry"]),
      " entry & ~exit:", sum(1 for x in rec if x["n_onsets"] == 0 and x["high_at_entry"] and not x["high_at_exit"]))
# for the largest group print the context
g = max(groups, key=len)
a0, z1 = g[0][0], g[-1][1]
print("largest group", a0, z1, len(g))
m = lambda t: t[(t > a0 - 1) & (t < z1 + 1)]
print(" measured R", np.round(m(mr), 4)); print(" measured F", np.round(m(mf), 4))
kr = np.searchsorted(mr, z1); kf = np.searchsorted(mf, z1)
print(" first measured after group: R", mr[kr] if kr < len(mr) else None, "F", mf[kf] if kf < len(mf) else None)
