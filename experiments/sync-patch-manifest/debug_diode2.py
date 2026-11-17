import logging, sys, numpy as np, npc_sync
from npc_sync import patch as P
from common import LOCAL7, read_stim, sync_start, match_stims_to_blocks
logging.disable(logging.WARNING)
name, bi = sys.argv[1], int(sys.argv[2])
d = LOCAL7 / name
raw = npc_sync.SyncDataset(next(d.glob("*T*.h5")), on_invalid="ignore")
blocks = list(raw.vsync_times_in_blocks)
stims = [read_stim(p, sync_start(raw)) for p in sorted(d.glob("*.hdf5"))]
m = match_stims_to_blocks(blocks, stims)[0][bi]
rec = P.solve_vsync_gaps(blocks[bi], m.stim.frame_intervals, raw.invalid_intervals)
a, z = rec.gaps[0].span
r = raw.get_rising_edges("stim_photodiode", units="seconds"); f = raw.get_falling_edges("stim_photodiode", units="seconds")
nr, nf, sol = P.predict_diode_edges(rec.times, r, f, (a, z), m.stim.diode_box_frame_interval)
an = np.load(LOCAL7.parent / f"analog_{name}.npz")["t_an_b"]
lo, hi = a - 0.25, a + 0.15
ev = [(t, "frame%d%s" % (i, "*" if i % 3 == 0 else ""), "S" if rec.source[i] == 0 else "I") for i, t in enumerate(rec.times) if lo < t < hi]
ev += [(t, "dig_rise", "") for t in r if lo < t < hi] + [(t, "dig_fall", "") for t in f if lo < t < hi]
ev += [(t, "pred_rise", "") for t in nr if lo < t < hi] + [(t, "pred_fall", "") for t in nf if lo < t < hi]
ev += [(t, "ANALOG", "") for t in an if lo < t < hi]
for t, k, s in sorted(ev): print(f"{t:.4f} {k} {s}")
print(sol)
