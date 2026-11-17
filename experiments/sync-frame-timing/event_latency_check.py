"""How accurately could licks, rewards and sound onsets inside an invalid interval be
reconstructed from stim-file frame indices + frame times?

Measured OUTSIDE invalid intervals (where the sync lines are genuine), per session:
  licks : stim `lickFrames` f (frame whose getInputData() first saw the lick line high) vs sync
          `lick_sensor` rising edges. Model: onset precedes input-read time of frame f
          (= vsync falling edge of frame f-1). d = vsync[f-1] - onset.
  reward: stim `rewardFrames` f (solenoid triggered right after flip of frame f) vs sync
          `reward_1_actuator` rising edges. latency = edge - vsync[f].
  sound : stim `trialStimStartFrame` f vs first sync bit-1 rising edge in [vsync[f], vsync[f]+0.2 s]
          (bit 1 = sound running; unlabelled in these files). latency = edge - vsync[f].

Frame times: raw vsync block if its length equals the stim frame count, else the prototype
`reconcile_replay` output (npc_sync/experiments/vsync-frame-count/repair.py).

Run: ../../../npc_sync/.venv/Scripts/python.exe event_latency_check.py > event_latency_check.log 2>&1

Checks:
  G1 fraction of stim lickFrames with a sync lick onset in (vsync[f-2], vsync[f-1]] (one frame window).
  G2 distribution of d (licks), reward latency, sound latency: median, 5th-95th pct, min/max (ms).
"""

from __future__ import annotations

import logging
import pathlib
import sys

import numpy as np

import npc_stim
import npc_sync

logging.disable(logging.WARNING)
ROOT = pathlib.Path(__file__).resolve().parents[3] / "npc_sync" / "experiments" / "vsync-frame-count"
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT))
from repair import reconcile_replay  # noqa: E402


def in_any(t, intervals):
    t = np.asarray(t, dtype=float)
    m = np.zeros(t.shape, bool)
    for a, z in intervals:
        m |= (t >= a) & (t < z)
    return m


def q(x) -> str:
    x = np.asarray(x) * 1e3
    if x.size == 0:
        return "n=0"
    p5, p50, p95 = np.percentile(x, [5, 50, 95])
    return f"n={x.size} median={p50:.2f} p5={p5:.2f} p95={p95:.2f} min={x.min():.2f} max={x.max():.2f} ms"


all_d, all_rew, all_snd = [], [], []
for sess_dir in sorted(p for p in DATA.iterdir() if p.is_dir()):
    s = npc_sync.SyncDataset(next(sess_dir.glob("*T*.h5")))
    iv = list(s.invalid_intervals)
    task = next(sess_dir.glob("DynamicRouting1_*.hdf5"))
    h5 = npc_stim.get_h5_stim_data(task)
    n_stim = npc_stim.get_total_stim_frames(h5)
    st = (npc_stim.get_stim_start_time(h5) - s.start_time).total_seconds()
    blocks = s.vsync_times_in_blocks
    bi = min(range(len(blocks)), key=lambda i: abs(blocks[i][0] - st))
    v = blocks[bi]
    if len(v) != n_stim:
        if not iv:
            print(f"{sess_dir.name}: task block len {len(v)} != {n_stim} and no invalid intervals: skip (e.g. timebase drop)")
            continue
        v = reconcile_replay(v, h5["frameIntervals"][:], iv).times
    lick_on = s.get_rising_edges("lick_sensor", units="seconds")
    lick_on = lick_on[~in_any(lick_on, iv)]
    lf = np.asarray(h5["lickFrames"][()], dtype=int).ravel()
    lf = lf[(lf >= 2) & (lf < len(v))]
    read_t = v[lf - 1]
    ok = ~in_any(read_t, iv)
    lf, read_t = lf[ok], read_t[ok]
    # nearest preceding sync lick onset
    j = np.searchsorted(lick_on, read_t, side="right") - 1
    valid = j >= 0
    d = read_t[valid] - lick_on[j[valid]]
    frame_iv = (v[lf - 1] - v[lf - 2])[valid]
    within_one_frame = np.mean((d > 0) & (d <= frame_iv + 1e-4))
    # reward
    rew = s.get_rising_edges(15, units="seconds")
    rew = rew[~in_any(rew, iv)]
    rf = np.asarray(h5["rewardFrames"][()], dtype=int).ravel() if "rewardFrames" in h5 else np.array([], int)
    rf = rf[(rf >= 0) & (rf < len(v))]
    rt = v[rf]
    rt = rt[~in_any(rt, iv)]
    k = np.searchsorted(rew, rt, side="left")
    k = k[k < len(rew)]
    rl = rew[k] - rt[: len(k)]
    rl = rl[(rl >= 0) & (rl < 0.2)]
    # sound
    snd = s.get_rising_edges(1, units="seconds")
    snd = snd[~in_any(snd, iv)]
    sf = np.asarray(h5["trialStimStartFrame"][()], dtype=float).ravel()
    sf = sf[np.isfinite(sf)].astype(int)
    sf = sf[(sf >= 0) & (sf < len(v))]
    tt = v[sf]
    tt = tt[~in_any(tt, iv)]
    m = np.searchsorted(snd, tt, side="left")
    lat = np.array([snd[i] - t for i, t in zip(m, tt) if i < len(snd) and snd[i] - t < 0.2])
    all_d.extend(d.tolist())
    all_rew.extend(rl.tolist())
    all_snd.extend(lat.tolist())
    print(f"\n===== {sess_dir.name} task block {bi} (len {len(v)})")
    print(f"G1 licks: stim lickFrames={lf.size}, sync onsets={lick_on.size}; onset within the preceding frame interval: {within_one_frame:.3f}")
    print(f"G2 lick  d = read time - onset: {q(d)}")
    print(f"G2 reward latency (edge - vsync[rewardFrame]): {q(rl)}")
    print(f"G2 sound latency (bit1 edge - vsync[trialStimStartFrame], aud trials): {q(lat)}")
print("\nALL lick d:", q(all_d))
print("ALL reward latency:", q(all_rew))
print("ALL sound latency:", q(all_snd))
