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
inv = raw.invalid_intervals
rec = P.solve_vsync_gaps(blocks[bi], m.stim.frame_intervals, inv)
a, z = rec.gaps[0].span
r = raw.get_rising_edges("stim_photodiode", units="seconds"); f = raw.get_falling_edges("stim_photodiode", units="seconds")
e = np.sort(np.r_[r, f])
box = m.stim.diode_box_frame_interval
tt = rec.times[np.arange(len(rec.times)) % box == 0]
print("box", box)
for lo, hi in ((a - 3, a), (z, z + 3)):
    x = e[(e >= lo) & (e < hi)]
    j = np.searchsorted(tt, x + 0.002, side="right") - 1
    lat = (x - tt[j]) * 1e3
    print(np.round(lat, 1))
    fi = np.diff(rec.times[(rec.times >= lo) & (rec.times < hi)]) * 1e3
    print("frame intervals > 20 ms:", np.round(fi[fi > 20], 1))
an = np.load(LOCAL7.parent / f"analog_{name}.npz")["t_an_b"]
x = an[(an >= a - 3) & (an < a)]
j = np.searchsorted(tt, x + 0.002, side="right") - 1
print("analog lat", np.round((x - tt[j]) * 1e3, 1))
