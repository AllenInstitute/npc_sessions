"""Print e2e_<label>.jsonl (last record per session)."""
import json, sys, pathlib
import numpy as np
for label in sys.argv[1:]:
    recs = {}
    for l in open(pathlib.Path(__file__).parent / f"e2e_{label}.jsonl"):
        r = json.loads(l); recs[r["session"]] = r
    print("=" * 110); print(label.upper())
    for sid, r in recs.items():
        print("-" * 110); print(sid, "ok" if r.get("ok") else f"FAILED: {r.get('error', '')[:300]}", f"({r['sec']} s)")
        if not r.get("ok"): continue
        tb = r.get("timebase")
        if tb:
            drops = tb["drops"]; print(f"  timebase: {sum(d[1] for d in drops)} dropped buffers at {len(drops)} places ({sum(d[2] != 'periodic_lines' for d in drops)} assumed), {tb['n_copies']} copies, net shift {tb['net_shift_s']} s, notes {tb['notes']}")
        print(f"  invalid intervals: {r['n_invalid_intervals']} ({r['invalid_s']:.1f} s); barcode discontinuities: {r['barcode_discontinuities']}")
        print(f"  patch notes: {r.get('patch_notes')}")
        ls = r.get("line_summary") or {}
        print("  lines with unknown spans:", {k: v["unknown"] for k, v in ls.items() if v.get("unknown") and not k.startswith("bit")})
        print("  removed:", {k: {a: b for a, b in v.items() if a.startswith("removed")} for k, v in ls.items() if any(a.startswith("removed") for a in v)})
        for b in r.get("blocks", []):
            print(f"  block {b['i']} {str(b['stim'])[:42]:42s} {b['status']:8s} stim {b['n_stim']} rec {b['n_rec']} spans {b['n_spans']} max|resid| {b['resid_ms_max']} ms; {b['source']}, {b['n_interp']} interp. {b['reason'][:110]}")
        if "blocks_error" in r: print("  blocks error:", r["blocks_error"])
        for k, v in r.get("frame_times", {}).items():
            if isinstance(v, str): print(f"  frame times {k}: {v[:160]}")
        ft = [v for v in r.get("frame_times", {}).values() if isinstance(v, dict)]
        print(f"  frame times: {len(ft)} stims ok, {sum(v['nan'] for v in ft)} NaN frames")
        if "ephys" in r:
            e = r["ephys"]; res = [x["qc"].get("loo_max_ms") for x in e if x["qc"].get("loo_max_ms") is not None]
            print(f"  ephys: {sum(x['status'] == 'pass' for x in e)}/{len(e)} devices pass; leave-one-out max residual: max over devices {max(res, default=float('nan')):.3f} ms; n barcodes paired {sorted({x['qc'].get('n_paired') for x in e})}; unlogged steps {sum((x['qc'].get('n_unlogged_steps') or 0) for x in e)}")
            for x in e:
                if x["status"] != "pass": print("     FAIL", x["device"], x["reason"][:150])
        if "ephys_error" in r: print("  ephys error:", r["ephys_error"])
        print("  trials:", r.get("trials") or r.get("trials_error"), "is_task", r.get("is_task"))
        tags = [(e["name"], e["tags"]) for e in r.get("epochs", [])]
        print("  epochs:", len(tags), "with replayed_sync tag:", sum("replayed_sync" in t for _, t in tags), r.get("epochs_error", ""))
