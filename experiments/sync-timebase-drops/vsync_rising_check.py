"""Do vsync rising and falling edges alternate in and around reconstructed spans of the patched
sync line? (rising edges are placed at reconstructed falling edge - median pulse width.)
Usage: vsync_rising_check.py <session id> [...]  (released packages, S3 patch cache)"""
import sys, logging
logging.basicConfig(level=logging.ERROR)
import numpy as np, npc_sessions
for sid in sys.argv[1:]:
    s = npc_sessions.DynamicRoutingSession(sid)
    sync = s.sync_data
    r = sync.get_rising_edges("vsync_stim", units="seconds"); f = sync.get_falling_edges("vsync_stim", units="seconds")
    t = np.r_[r, f]; kind = np.r_[np.ones(len(r), int), np.zeros(len(f), int)]
    o = np.argsort(t, kind="stable"); t, kind = t[o], kind[o]
    same = np.flatnonzero(kind[1:] == kind[:-1])
    spans = [sp.span for sp in sync.patch.lines["vsync_stim"] if sp.method == "frame_intervals"]
    near = lambda x: any(a - 0.05 <= x <= z + 0.05 for a, z in spans)
    n_near = sum(near(t[i]) for i in same)
    print(f"{sid}: {len(r)} rising, {len(f)} falling; {len(spans)} reconstructed spans; consecutive same-kind edge pairs: {len(same)} ({n_near} within 50 ms of a reconstructed span)")
    for i in same[:6]:
        print(f"    {'RR' if kind[i] else 'FF'} at {t[i]:.5f}, {t[i+1]:.5f} (dt {1e3 * (t[i+1] - t[i]):.2f} ms){' near span' if near(t[i]) else ''}")
