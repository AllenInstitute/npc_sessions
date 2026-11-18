"""NI-DAQ analog photodiode (AI0) check of the phantom-flicker patch on DRpilot_762526_20250320
block 1 (DynamicRouting1, box interval 1): the only block from the analog era (2025-01-13 on)
that npc_stim rejected for extra flips and that the patch recovers. Block 0 (Spontaneous,
accepted with and without the patch) is the control.

Uses the analog ground truth of experiments/display-times-analog-validation (compare.py: a
monotone DP alignment of analog transitions to frames from vsync edges and box polarity only,
independent of the TTL diode line): results/DRpilot_762526_20250320_frames.npz, per analog
transition assigned to a frame: b1_a (time), b1_f (frame), b1_pol (+1/-1); b1_a_raw /
b1_unmatched: all transitions and those not assigned to a frame; lag_rise/lag_fall: session
analog -> TTL lag. The analog trace: C:/data/t2p_qc/analog/<folder>/trace_b1.npy (5 kHz).

Checks:
1. every TTL pair the patch removed as phantom flicker lies inside an analog dip (two
   consecutive non-frame analog transitions); every removed glitch is at a frame transition
2. depth of those analog dips vs the full black-white swing
3. alignment: the TTL edge assigned to each frame (npc_sync.align_diode_toggles) is the TTL edge
   nearest that frame's analog transition
4. recovered display times (C:/data/t2p_qc/out_phantom, B_block1) vs analog, compare.py's error:
   err = analog - (display - 8 ms) - session lag[polarity]

Run: PYTHONPATH="../../../npc_sync/src;../../../npc_stim/src;../../src;." ../../.venv/Scripts/python.exe analog_check_762526.py > analog_check_762526.log 2>&1
"""

from __future__ import annotations

import json
import logging

import numpy as np

import npc_sync

import common
from npc_sessions import timing

logging.disable(logging.WARNING)
FOLDER = "DRpilot_762526_20250320"
BI = 1
A = np.load(f"../display-times-analog-validation/results/{FOLDER}_frames.npz")
OUT = np.load(f"C:/data/t2p_qc/out_phantom/{FOLDER}.npz")
TR = json.load(open(f"C:/data/t2p_qc/analog/{FOLDER}/traces.json"))
LAG = {1: float(A["lag_rise"]), -1: float(A["lag_fall"])}
a_tr, f_tr, pol_tr = A[f"b{BI}_a"], A[f"b{BI}_f"].astype(int), A[f"b{BI}_pol"].astype(int)
a_raw = A[f"b{BI}_a_raw"]
unmatched = A[f"b{BI}_unmatched"].astype(bool)
print(f"{FOLDER} b{BI}: {len(a_tr)} analog transitions assigned to frames; all transitions {len(a_raw)}, not frames {int(unmatched.sum())}")
print(f"session analog -> TTL lag: rising {LAG[1] * 1e3:.2f} ms, falling {LAG[-1] * 1e3:.2f} ms")

raw = npc_sync.SyncDataset(common.sync_path(FOLDER), on_invalid="ignore")
stims = common.stim_paths(FOLDER)
patch = timing.build_session_sync_patch(raw, stims)
fs = raw.sample_freq
rec = [b for b in patch.diode_blocks if b.block_index == BI][0]
print(f"patch: diode block status {rec.status}; {rec.n_phantom_flicker} phantom-flicker and {rec.n_glitch} glitch edges removed")


def nearest(ref, x):
    j = np.clip(np.searchsorted(ref, x), 1, len(ref) - 1)
    return np.where(np.abs(x - ref[j - 1]) <= np.abs(ref[j] - x), j - 1, j)


# analog trace of the block (for dip depth)
tb = TR["blocks"][f"b{BI}"]
an_fs = TR["fs"]
y = np.load(f"C:/data/t2p_qc/analog/{FOLDER}/trace_b{BI}.npy", mmap_mode="r")
an = json.load(open(f"C:/data/t2p_qc/analog/{FOLDER}/analog.json"))
print(f"analog trace: {len(y)} samples at {an_fs} Hz; mapping keys: {sorted(an)[:8]}")

print("\n1-2. removed TTL edges vs analog:")
ph_pairs, glitches = [], []
for r in patch.removed.get("stim_photodiode", []):
    if r.block_index != BI:
        continue
    t = np.sort(np.r_[r.rising_samples, r.falling_samples] / fs)
    if r.reason == "phantom_flicker":
        ph_pairs += [tuple(t[i : i + 2]) for i in range(0, len(t), 2)]
    else:
        glitches += list(t)
non = np.sort(a_raw[unmatched])
n_ok = 0
for t0, t1 in ph_pairs:
    # an analog dip: two consecutive non-frame transitions bracketing the TTL pair (TTL lags 0-10 ms)
    i = np.searchsorted(non, t0) - 1
    ok = 0 <= i < len(non) - 1 and non[i] <= t0 + 0.002 and non[i + 1] >= t1 - 0.003 and non[i + 1] - non[i] < 0.03
    n_ok += ok
    msg = f"analog dip {non[i]:.4f}-{non[i + 1]:.4f} s ({(non[i + 1] - non[i]) * 1e3:.1f} ms)" if 0 <= i < len(non) - 1 else "no dip"
    print(f"  phantom pair TTL {t0:.4f}-{t1:.4f} s ({(t1 - t0) * 1e3:.1f} ms): {msg} -> {'ok' if ok else 'MISMATCH'}")
print(f"phantom-flicker TTL pairs inside an analog non-frame dip: {n_ok}/{len(ph_pairs)}")
fr_t = np.sort(a_tr)
gl_ok = 0
for t in glitches:
    d = (t - fr_t[nearest(fr_t, np.array([t]))[0]]) * 1e3
    gl_ok += -2 < d < 6
    print(f"  glitch TTL {t:.5f} s: nearest frame analog transition {d:+.2f} ms")
print(f"glitch edges at a frame transition: {gl_ok}/{len(glitches)}")
# dips not registered on the TTL line (sub-threshold) and unpaired non-frame transitions
used = set()
for t0, t1 in ph_pairs:
    i = np.searchsorted(non, t0) - 1
    used |= {i, i + 1}
rest = [i for i in range(len(non)) if i not in used]
print("non-frame analog transitions without a removed TTL pair: " + ", ".join(f"{non[i]:.4f}" for i in rest))
dig = np.sort(np.r_[raw.get_rising_edges("stim_photodiode", units="seconds"), raw.get_falling_edges("stim_photodiode", units="seconds")])
for i in rest:
    print(f"  {non[i]:.4f} s: nearest TTL edge {(dig[nearest(dig, np.array([non[i]]))[0]] - non[i]) * 1e3:+.2f} ms")


print("\n3. frame <-> TTL edge assignment vs the analog DP:")
v = raw.vsync_times_in_blocks[BI]
sb = [b for b in timing.get_stim_blocks(raw, stims) if b.block_index == BI][0]
d_r = raw.get_rising_edges("stim_photodiode", units="seconds")
d_f = raw.get_falling_edges("stim_photodiode", units="seconds")
d_s = np.r_[d_r, d_f]
d_q = np.r_[np.ones(len(d_r), bool), np.zeros(len(d_f), bool)]
o = np.argsort(d_s, kind="stable")
d_s, d_q = d_s[o], d_q[o]
sel = (d_s >= v[0]) & (d_s < v[-1] + 0.5)
vr_all = raw.get_rising_edges("vsync_stim", units="seconds")
vr = vr_all[np.searchsorted(vr_all, v) - 1]
al = npc_sync.align_diode_toggles(v, sb.frame_intervals, d_s[sel], d_q[sel], frame_signal_rising=vr)
e = d_s[sel]
fe = al.frame_edge[f_tr]
has = fe >= 0
ttl = e[fe[has]]
near_e = e[nearest(e, a_tr[has])]
agree = ttl == near_e
print(f"analog-assigned frames {len(f_tr)}; with an aligned TTL toggle {int(has.sum())}; "
      f"TTL toggle == TTL edge nearest the frame's analog transition: {int(agree.sum())}; disagree {int((~agree).sum())}")
lag = (ttl - a_tr[has]) * 1e3
print(f"TTL - analog (ms): p0.1 {np.percentile(lag, 0.1):.2f} p50 {np.median(lag):.2f} p99.9 {np.percentile(lag, 99.9):.2f} max {lag.max():.2f}")
for i in np.flatnonzero(has)[~agree][:10]:
    k = f_tr[i]
    print(f"  frame {k} (refreshes of frames k-3..k-1: {al.n_refreshes[max(k - 3, 0):k].tolist()}): TTL {e[al.frame_edge[k]]:.4f}, analog {a_tr[i]:.4f}")

print("\n4. display times vs analog (compare.py error: analog - (display - 8 ms) - session lag[polarity]):")
for bi in (0, BI):
    a_b, f_b, p_b = A[f"b{bi}_a"], A[f"b{bi}_f"].astype(int), A[f"b{bi}_pol"].astype(int)
    disp = OUT[f"B_block{bi}"] - 0.008
    err = (a_b - disp[f_b] - np.where(p_b == 1, LAG[1], LAG[-1])) * 1e3
    err = err[np.isfinite(err)]
    q = np.percentile(err, [1, 99])
    label = "control, accepted with and without the patch" if bi == 0 else "recovered by the patch"
    ref = ""
    if f"b{bi}_err_fix" in A.files:
        ef = A[f"b{bi}_err_fix"] * 1e3
        ef = ef[np.isfinite(ef)]
        ref = f"; compare.py on the released display times: p1/p99 {np.percentile(ef, 1):+.2f}/{np.percentile(ef, 99):+.2f}, max {np.abs(ef).max():.2f}"
    print(f"b{bi} ({label}): {len(err)} frames: p1/p99 {q[0]:+.2f}/{q[1]:+.2f} ms, max |err| {np.abs(err).max():.2f} ms, "
          f">2 ms {int((np.abs(err) > 2).sum())}, >4 ms {int((np.abs(err) > 4).sum())}, >8 ms {int((np.abs(err) > 8).sum())}{ref}")
print("block 1 had no display times without the patch:", "A_block1" not in OUT.files)
