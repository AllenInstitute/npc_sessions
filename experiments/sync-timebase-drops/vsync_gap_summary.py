"""Summary of vsync_gap_survey.jsonl -> vsync_gap_summary.log"""
import json, collections, pathlib
import numpy as np
rows = [json.loads(l) for l in open(pathlib.Path(__file__).parent / "vsync_gap_survey.jsonl")]
print("rows", len(rows), "errors", sum("error" in r for r in rows), sorted({r["session"] for r in rows if "error" in r})[:5])
rows = [r for r in rows if "error" not in r]
by = collections.defaultdict(dict)
for r in rows: by[(r["session"], r["stim"])][r["model"]] = r
both = {k: v for k, v in by.items() if len(v) == 2}
print("blocks with a stim file under both models:", len(both), "in", len({k[0] for k in both}), "files")
aff = {k: v for k, v in both.items() if v["recorded"]["n_spans"] or v["timebase"]["n_spans"] or not v["recorded"]["ok"] or not v["timebase"]["ok"]}
print("blocks with a fault or a count mismatch:", len(aff))
for model in ("recorded", "timebase"):
    ok = sum(v[model]["ok"] for v in aff.values())
    res = np.array([x for v in aff.values() for x in v[model]["residuals_ms"]])
    bal = collections.Counter("0" if v[model]["balance"] == 0 else "nonzero" for v in aff.values())
    print(f"  {model:9s}: reconcilable {ok}/{len(aff)}; balance {dict(bal)}; gaps {len(res)}; |residual| ms: median {np.median(np.abs(res)):.4f} p95 {np.percentile(np.abs(res), 95):.3f} p99 {np.percentile(np.abs(res), 99):.3f} max {np.abs(res).max():.3f}; > 0.2 ms: {(np.abs(res) > 0.2).sum()}; > 1 ms: {(np.abs(res) > 1).sum()}")
print("\nchange per block (recorded -> timebase):", dict(collections.Counter((v["recorded"]["ok"], v["timebase"]["ok"]) for v in aff.values())))
print("\nblocks not reconcilable on the timebase:")
for (sess, stim), v in sorted(aff.items()):
    t = v["timebase"]
    if not t["ok"]:
        r = np.array(t["residuals_ms"])
        print(f"  {sess} {stim[:40]:40s} stim {t['n_stim']} rec {t['n_rec']} spans {t['n_spans']} balance {t['balance']} boundary_gap {t['boundary_gap']} max|res| {np.abs(r).max() if len(r) else None} n>0.2: {(np.abs(r) > 0.2).sum() if len(r) else 0} | recorded model ok: {v['recorded']['ok']} balance {v['recorded']['balance']}")
print("\nblocks reconcilable on the recorded clock but not on the timebase:", [(k, ) for k, v in aff.items() if v["recorded"]["ok"] and not v["timebase"]["ok"]])
print("\ntimebase, 1 ms gate (used where every drop is located): reconcilable", sum(bool(v["timebase"].get("ok_1ms")) for v in aff.values()), "/", len(aff))
