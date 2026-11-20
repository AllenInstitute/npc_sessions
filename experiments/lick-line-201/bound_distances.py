"""Distances from each real span bound to the sync event whose data word gives the lick state
(fix.line_state_at_bounds), and lick edges within 3 samples of a bound."""
import json, numpy as np
from common import load, edges, LICK
import fix
out = {}
for sid in ["714748_2024-06-28", "708020_2024-05-17"]:
    raw, sp, base = load(sid)
    spans = sorted(s.span for s in base.lines[LICK])
    b = fix.line_state_at_bounds(raw, LICK, spans)
    dt = 1 / raw.sample_freq
    rr, ff = edges(raw); e = np.r_[rr, ff]
    out[sid] = dict(
        n_spans=len(spans),
        before_ms=[(x["span"][0] - x["t_before"]) * 1e3 for x in b],
        after_ms=[(x["t_after"] - x["span"][1]) * 1e3 for x in b],
        all_observed=all(x["before_observed"] and x["after_observed"] for x in b),
        n_edges_inside_at_start=int(sum(np.any((e - a > 0) & (e - a <= 3 * dt)) for a, z in spans)),
        n_edges_at_end=int(sum(np.any(np.abs(e - z) <= 3 * dt) for a, z in spans)),
    )
    print(sid, {k: v for k, v in out[sid].items() if not k.endswith("_ms")}, "max ms", max(out[sid]["before_ms"]), max(out[sid]["after_ms"]))
json.dump(out, open("results/bound_distances.json", "w"))
