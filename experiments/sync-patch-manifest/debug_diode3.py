"""For diode spans failing the after-span gate: where is the worst post-span edge? (splice artefact check)"""
import json, logging, sys, numpy as np, pandas as pd, npc_sync
from npc_sync import patch as P
from common import read_stim, sync_start, match_stims_to_blocks
logging.disable(logging.WARNING)
iv = pd.read_json("intervals.jsonl", lines=True); f = pd.read_json("manifest_files.jsonl", lines=True)
bad = iv[(iv.kind == "diode") & (iv.failures.apply(len) > 0)]
rows = [r for r in __import__('csv').DictReader(open('sessions.csv'))]
for _, r in bad.iterrows():
    row = next(x for x in rows if x["session"] == r.session)
    raw = npc_sync.SyncDataset(row["sync_path"], on_invalid="ignore")
    blocks = list(raw.vsync_times_in_blocks)
    stims = [read_stim(p, sync_start(raw)) for p in row["stim_paths"].split(";")]
    m = match_stims_to_blocks(blocks, stims)[0][int(r.block_index)]
    a, z = r.span
    spans = [(x - 1 / raw.sample_freq, y) for x, y in raw.invalid_intervals]
    rec = P.solve_vsync_gaps(blocks[int(r.block_index)], m.stim.frame_intervals, spans)
    dr = raw.get_rising_edges("stim_photodiode", units="seconds"); df = raw.get_falling_edges("stim_photodiode", units="seconds")
    _, _, sol = P.predict_diode_edges(rec.times, dr, df, (a, z), m.stim.diode_box_frame_interval)
    e = np.sort(np.r_[dr, df]); post = e[(e >= z) & (e < z + 3)]
    tt = rec.times[np.arange(len(rec.times)) % m.stim.diode_box_frame_interval == 0]
    pol_lat = [sol.latency_rising_sec, sol.latency_falling_sec]
    # error of each post edge vs nearest (tt + either latency)
    pred = np.sort(np.r_[tt + pol_lat[0], tt + pol_lat[1]])
    err = np.array([np.min(np.abs(pred - x)) for x in post])
    allt = raw.get_all_times("seconds"); first_ev = allt[np.searchsorted(allt, z)]
    k = int(np.argmax(err))
    fi = m.stim.frame_intervals; ftimes = rec.times
    near_frames = np.flatnonzero((ftimes > post[k] - 0.1) & (ftimes < post[k]))
    long = [round(fi[i - 1] * 1e3, 1) for i in near_frames if i > 0 and fi[i - 1] > 0.025]
    print(f"{r.session} block {int(r.block_index)} span {a:.3f}-{z:.3f}: after_err {r.after_error_sec*1e3:.2f} ms; worst post edge #{k} at {post[k]:.5f} "
          f"(z+{(post[k]-z)*1e3:.2f} ms), err {err[k]*1e3:.2f} ms; first event after z at z+{(first_ev-z)*1e3:.2f} ms; "
          f"first post edge is first event: {abs(post[0]-first_ev)<1e-9}; errors of first 4 post edges ms {np.round(err[:4]*1e3,2)}; long frame intervals in 100 ms before worst edge: {long}")
