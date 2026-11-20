"""Post-hoc checks on results/*.pkl: (1) what the remaining 'fake' onsets of the fixed method on
injected spans are; (2) which real spans the fix changes."""
import pickle, json, numpy as np, npc_sync
from evaluate import nearest, TOL
out = {}
for l in open("results/summary.jsonl"):
    sid = json.loads(l)["session"]
    d = pickle.loads(open(f"results/{sid}.pkl", "rb").read())
    spans = [s for run in d["inj_runs"] for s in run]
    rr, ff, on = d["raw_r"], d["raw_f"], d["onsets"]
    res = {}
    for m in ("current", "fixed"):
        r = d[f"inj_{m}"]["r"]; r = r[npc_sync.in_intervals(r, spans)]
        fake = r[nearest(r, rr) > TOL]
        kinds = []
        for x in fake:
            a, z = next(s for s in spans if s[0] <= x < s[1])
            has_lf = nearest(np.array([x]), on)[0] <= TOL
            # true state just after z (lick in progress at exit)?
            tr = rr[rr < z]; tf = ff[ff < z]
            high_at_z = (tr[-1] if len(tr) else -1) > (tf[-1] if len(tf) else -1)
            kinds.append("lickFrames onset, no sync onset" if has_lf else ("lick in progress at span end" if high_at_z else "no lick (spurious)"))
        res[m] = {k: kinds.count(k) for k in set(kinds)}
    # real spans changed by the fix
    ch = {"changed": 0, "changed_first_in_group": 0, "changed_later_in_group": 0}
    real = d["real_spans"]
    allm = np.sort(np.r_[rr, ff])
    for i, (a, z) in enumerate(real):
        def sel(x): return x[(x >= a) & (x < z)]
        c, f_ = d["real_current"], d["real_fixed"]
        same = np.array_equal(sel(c["r"]), sel(f_["r"])) and np.array_equal(sel(c["f"]), sel(f_["f"]))
        if not same:
            ch["changed"] += 1
            first = i == 0 or np.any((allm >= real[i - 1][1]) & (allm < a))
            ch["changed_first_in_group" if first else "changed_later_in_group"] += 1
    out[sid] = dict(injected_fake_kinds=res, real_span_changes=ch, n_real_spans=len(real))
    print(sid, out[sid])
json.dump(out, open("results/checks.json", "w"), indent=1)
