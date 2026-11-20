"""Independent check of the timebase with licks: stim clock vs sync.

The task stim file gives each lick's frame (`lickFrames`); with `frameIntervals` that is a lick
time on the stim PC's clock, which no sync fault touches. For every sync lick (lick_sensor
rising edge) recorded in a displaced segment of the timebase (shift > 0), the residual to the
nearest stim lick is computed with the lick at its recorded time (H0: data not displaced) and
at its true time (H1: timebase). Reference: licks in segments with shift 0.

Stim-to-sync mapping: first vsync of the task block + stim time; slow drift removed with a
running median of the on-time residuals (31 licks).

Run: PYTHONPATH=../../../npc_sync/src ../../.venv/Scripts/python.exe lick_validation.py > lick_validation.log
"""
from __future__ import annotations
import logging, pathlib, sys
import h5py, numpy as np
import npc_sync
logging.disable(logging.WARNING)
SHARE = pathlib.Path("//allen/programs/mindscope/workgroups/dynamicrouting/PilotEphys/Task 2 Pilot")
SESSIONS = sys.argv[1:] or ["DRpilot_702131_20240223", "DRpilot_703880_20240418", "DRpilot_703880_20240412", "DRpilot_708016_20240426", "DRpilot_708020_20240516",
    "DRpilot_712139_20240510", "DRpilot_708020_20240517", "DRpilot_714748_20240628", "DRpilot_859582_20260612", "DRpilot_872974_20261105", "DRpilot_839144_20260429",
    "DRpilot_715709_20240808", "DRpilot_840160_20260428", "DRpilot_717473_20240916"]
# not usable: DRpilot_832032_20260223 (lick line has 202,800 rising edges in the task block: noise)
TOL = 0.025
tot = dict(n=0, h0=0, h1=0, ref=0, ref_ok=0)
print(f"{'session':26s} {'displaced licks':>15s} {'H0 within 25 ms':>16s} {'H1 within 25 ms':>16s} | {'on-time licks':>13s} {'within 25 ms':>13s} | shifts of displaced licks (s: n)")
for sess in SESSIONS:
    d = SHARE / sess
    try:
        syncp = next(p for p in d.glob("*.h5") if "T" in p.stem); task = sorted(d.glob("DynamicRouting1_*.hdf5"), key=lambda p: p.stat().st_size)[-1]
    except (StopIteration, IndexError):
        print(sess, "no sync/task file"); continue
    raw = npc_sync.SyncDataset(syncp, on_invalid="ignore")
    tb = raw.find_timebase()
    s = npc_sync.SyncDataset(raw, on_invalid="ignore", timebase=tb)
    fs = raw.sample_freq
    with h5py.File(task, "r") as f:
        fi = f["frameIntervals"][:]; lf = np.asarray(f["lickFrames"][:], dtype=int)
    tau = np.r_[0, np.cumsum(fi)][lf]
    blocks = s.vsync_times_in_blocks
    bi = int(np.argmax([len(b) for b in blocks])); v0, vend = blocks[bi][0], blocks[bi][-1]
    stim_licks = v0 + tau
    rec = np.asarray(raw._get_recorded_edges("lick_sensor", "rising"), dtype=np.int64)
    true, keep = tb.to_true(rec)
    rec, true = rec[keep] / fs, true[keep] / fs
    shift = true - rec
    inb = (true > v0) & (true < vend)
    rec, true, shift = rec[inb], true[inb], shift[inb]
    def resid(t):
        j = np.clip(np.searchsorted(stim_licks, t), 1, len(stim_licks) - 1)
        a, b = t - stim_licks[j - 1], t - stim_licks[j]
        return np.where(np.abs(a) < np.abs(b), a, b)
    on = shift == 0
    # frames at the start of the block can be missing (dropped), so that the first vsync is not
    # stim frame 0: offset = mode of (on-time sync lick - stim lick) over all pairs, 10 ms bins
    if on.sum() >= 10:
        diffs = (true[on][:, None] - stim_licks[None, :]).ravel(); diffs = diffs[np.abs(diffs) < 6]
        h, e = np.histogram(diffs, bins=np.arange(-6, 6.01, 0.01))
        o_best = float(e[int(np.argmax(h))]) + 0.005
        if abs(o_best) > 0.1:
            print(f"{sess:26s} (block start offset {o_best:+.2f} s applied: first frames of the block are in a dropped interval)")
            stim_licks = stim_licks + o_best
    if on.sum() < 10:
        print(f"{sess:26s} only {int(on.sum())} on-time licks in the task block (drop before it): no reference, skipped"); continue
    r_on = resid(true[on])
    # drift: running median of on-time residuals
    order = np.argsort(true[on]); t_on = true[on][order]; r_sorted = r_on[order]
    k = 15
    med = np.array([np.median(r_sorted[max(i - k, 0): i + k + 1]) for i in range(len(r_sorted))])
    base = lambda t: np.interp(t, t_on, med)
    ok_on = np.abs(r_on[order] - med) < TOL
    dis = shift > 0
    h1 = np.abs(resid(true[dis]) - base(true[dis])) < TOL
    h0 = np.abs(resid(rec[dis]) - base(rec[dis])) < TOL
    vals, cnt = np.unique(np.round(shift[dis], 1), return_counts=True)
    print(f"{sess:26s} {int(dis.sum()):15d} {int(h0.sum()):16d} {int(h1.sum()):16d} | {int(on.sum()):13d} {int(ok_on.sum()):13d} | {dict(zip(vals.tolist(), cnt.tolist()))}", flush=True)
    tot["n"] += int(dis.sum()); tot["h0"] += int(h0.sum()); tot["h1"] += int(h1.sum()); tot["ref"] += int(on.sum()); tot["ref_ok"] += int(ok_on.sum())
print(f"\nTOTAL displaced licks {tot['n']}: within 25 ms of a stim lick at recorded time (H0) {tot['h0']} ({100*tot['h0']/max(tot['n'],1):.0f}%), at true time (H1) {tot['h1']} ({100*tot['h1']/max(tot['n'],1):.0f}%); on-time reference {tot['ref_ok']}/{tot['ref']} ({100*tot['ref_ok']/max(tot['ref'],1):.0f}%)")
