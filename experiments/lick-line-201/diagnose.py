"""Run the CURRENT reconstruct_lick_line on a fresh base patch and inspect the result."""
import copy, json, sys, pickle
import numpy as np, npc_sync
from npc_sessions import timing
from common import *

sid = sys.argv[1]
raw, stim_paths, base = load(sid)
p = copy.deepcopy(base)
info = timing.reconstruct_lick_line(raw, p, stim_paths)
timing.widen_unknown_lick_spans(raw, p)
view = npc_sync.SyncDataset(raw, patch=p, on_invalid="mask")
r, f = edges(view)
rr, ff = edges(raw)
spans = [s.span for s in base.lines[LICK]]
print(sid, "lick spans", len(spans), "invalid", len(raw.invalid_intervals), "sample_freq", raw.sample_freq)
print("patched R/F", len(r), len(f), "alt errors", alternation_errors(r, f))
print("raw(ignore) R/F", len(rr), len(ff), "alt errors", alternation_errors(rr, ff))
# edges near span bounds
dt = 1/raw.sample_freq
near_a = sum(np.any(np.abs(np.r_[rr, ff] - a) <= 2*dt) for a, z in spans)
near_z = sum(np.any(np.abs(np.r_[rr, ff] - z) <= 2*dt) for a, z in spans)
print("spans with a raw lick edge within 2 samples of start / end:", near_a, near_z)
sts = [d for d in info.get("spans", [])]
print("diag spans", len(sts), {k: sum(1 for d in sts if d.get("status") == k) for k in ("reconstructed", "unknown")})
pickle.dump(dict(info=info, patch=p), open(CACHE / f"{sid}_current.pkl", "wb"))
