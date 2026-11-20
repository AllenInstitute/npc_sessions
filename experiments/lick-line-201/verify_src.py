"""Check the src/ implementation (builder 9) against the prototype (fix.py) on real spans."""
import copy, json, pickle
import numpy as np, npc_sync
from npc_sessions import timing
from common import load, edges, LICK
from evaluate import nearest, TOL
import fix

rows = []
for l in open("results/summary.jsonl"):
    sid = json.loads(l)["session"]
    raw, stim_paths, base = load(sid)
    p = copy.deepcopy(base)
    info = timing.reconstruct_lick_line(raw, p, stim_paths)
    timing.widen_unknown_lick_spans(raw, p)
    view = npc_sync.SyncDataset(raw, patch=p, on_invalid="mask")
    r, f = edges(view)
    src = np.asarray(view.get_edge_sources(LICK, "rising"))
    d = pickle.loads(open(f"results/{sid}.pkl", "rb").read())
    proto = d["real_fixed"]
    row = dict(session=sid, builder=timing.BUILDER, n_rising=len(r), n_falling=len(f),
               alternation_errors=fix.alternates(r, f),
               synthetic_without_lick_frame=int(np.sum(nearest(r[src == 1], d["onsets"]) > TOL)),
               identical_to_prototype=bool(np.array_equal(r, proto["r"]) and np.array_equal(f, proto["f"])),
               notes=[n for n in p.notes if "lick" in n],
               info_alternation_errors=info.get("alternation_errors"),
               info_synthetic_without_onset=info.get("n_synthetic_onsets_without_lick_frame"))
    rows.append(row); print(row, flush=True)
json.dump(rows, open("results/verify_src.json", "w"), indent=1)
