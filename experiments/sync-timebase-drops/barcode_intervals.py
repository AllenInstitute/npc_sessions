"""Barcode onset intervals of a session, on the recorded clock and on the patched sync line,
with no barcode excluded. Usage: barcode_intervals.py <session id>"""
import sys, logging
logging.basicConfig(level=logging.ERROR)
import numpy as np, npc_sessions, npc_sync
s = npc_sessions.DynamicRoutingSession(sys.argv[1])
views = {"recorded": s._raw_sync_data, "timebase, unpatched": npc_sync.SyncDataset(s.sync_data, on_invalid="ignore"), "patched": s.sync_data}
inv = s.sync_data.invalid_intervals
for name, v in views.items():
    r = v.get_rising_edges("barcode_ephys", units="seconds"); f = v.get_falling_edges("barcode_ephys", units="seconds")
    on = r[np.r_[True, np.diff(r) > 20]]
    d = np.diff(on); P = np.median(d); m = np.maximum(np.round(d / P), 1); dev = d - m * P
    print(f"== {name}: {len(on)} barcodes, period {P:.5f} s; multiples {dict(zip(*np.unique(m.astype(int), return_counts=True)))}; |dev| > 1 ms: {(np.abs(dev) > 1e-3).sum()}; max |dev| of the rest {np.abs(dev[np.abs(dev) <= 1e-3]).max() * 1e3:.3f} ms")
    for i in np.flatnonzero((np.abs(dev) > 1e-3) | (m > 1)):
        n_r = int(((r >= on[i + 1]) & (r < on[i + 1] + 2)).sum())
        near = [(a, z) for a, z in inv if z > on[i] - 3 and a < on[i + 1] + 3]
        print(f"   {on[i]:9.3f} -> {on[i+1]:9.3f}: {d[i]:8.4f} s = {int(m[i])} x period {dev[i]*1e3:+9.2f} ms; rising edges in the barcode at the end: {n_r}; dropped intervals within 3 s of either end: {near[:4]}")
