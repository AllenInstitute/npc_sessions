"""Report figures: phantom-flicker examples (TTL, and AI0 for 762526_20250320), population rule,
box-1 photodiode gate before/after. Read-only (local cache, local analog traces).

Run: PYTHONPATH="../../../npc_sync/src;../../../npc_stim/src;../../src;." ../../.venv/Scripts/python.exe fig_phantom.py
"""

from __future__ import annotations

import json
import logging
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import npc_sync

import common

logging.disable(logging.WARNING)
HERE = pathlib.Path(__file__).parent
FIGS = HERE / "figs"
FIGS.mkdir(exist_ok=True)


def edges_panel(ax, folder, bi, k, before=2, after=3, title="", analog=None):
    b = common.block_data(folder, bi)
    v = b.vsyncs
    jr = np.searchsorted(b.vsync_rising, v) - 1
    al = npc_sync.align_diode_toggles(v, b.frame_intervals, b.edges, b.rising, frame_signal_rising=b.vsync_rising[jr])
    t0 = v[k]
    lo, hi = v[max(k - before, 0)] - 0.005, v[min(k + after, len(v) - 1)] + 0.045
    sel = (b.edges >= lo) & (b.edges < hi)
    # vsync frame signal as a square wave
    vr = b.vsync_rising[(b.vsync_rising >= lo - 0.02) & (b.vsync_rising < hi)]
    vf = v[(v >= lo - 0.02) & (v < hi)]
    tt = np.sort(np.r_[vr, vf])
    lvl = np.where(np.isin(tt, vr), 1, 0)
    ax.step((np.r_[lo, tt, hi] - t0) * 1e3, 3.2 + 0.6 * np.r_[1 - lvl[0], lvl, lvl[-1]], where="post", color="C0", lw=1)
    ax.text((lo - t0) * 1e3, 4.0, "frame signal (vsync line)", fontsize=7, color="C0", va="bottom")
    # diode TTL as a square wave, edges coloured by class
    e, q, c = b.edges[sel], b.rising[sel], al.edge_class[sel]
    state0 = 0 if q[0] else 1
    xs = np.r_[lo, e, hi]
    ys = np.r_[state0, q.astype(int), q[-1].astype(int)]
    ax.step((xs - t0) * 1e3, 1.6 + 0.6 * ys, where="post", color="k", lw=1)
    ax.text((lo - t0) * 1e3, 2.4, "photodiode (TTL)", fontsize=7, va="bottom")
    col = {npc_sync.DIODE_TOGGLE: "C2", npc_sync.DIODE_PHANTOM_FLICKER: "C3", npc_sync.DIODE_GLITCH: "C1",
           npc_sync.DIODE_UNEXPLAINED: "m", npc_sync.DIODE_OUTSIDE: "gray"}
    for t, cc in zip(e, c):
        ax.axvline((t - t0) * 1e3, ymin=0.28, ymax=0.52, color=col[int(cc)], lw=2 if cc == npc_sync.DIODE_PHANTOM_FLICKER else 1)
    # expected toggle times
    ks = np.arange(max(k - before, 0), min(k + after + 1, len(v)))
    ax.plot((al.expected[ks] - t0) * 1e3, np.full(len(ks), 1.35), "v", color="C2", ms=5, label="expected toggle (frame + latency)")
    for kk in ks:
        if kk < len(al.n_refreshes) and al.n_refreshes[kk] >= 2:
            ax.axvspan((v[kk] - t0) * 1e3, (v[kk + 1] - t0) * 1e3, color="C3", alpha=0.06)
            ax.text((v[kk] - t0) * 1e3 + 1, 4.25, f"frame {kk} held {al.n_refreshes[kk]} refreshes", fontsize=7, color="C3")
    if analog is not None:
        at, ay = analog
        m = (at >= lo) & (at < hi)
        yy = ay[m]
        yy = (yy - np.percentile(ay, 1)) / (np.percentile(ay, 99) - np.percentile(ay, 1))
        ax.plot((at[m] - t0) * 1e3, 0.1 + 1.0 * yy, color="C4", lw=1)
        ax.text((lo - t0) * 1e3, 1.15, "photodiode (NI-DAQ AI0)", fontsize=7, color="C4", va="bottom")
    ax.set_ylim(0, 4.8)
    ax.set_yticks([])
    ax.set_xlabel(f"ms from frame {k}'s vsync falling edge")
    ax.set_title(title, fontsize=9)
    from matplotlib.lines import Line2D

    ax.legend(handles=[Line2D([], [], color="C2", lw=2, label="toggle"), Line2D([], [], color="C3", lw=2, label="phantom flicker (removed)"),
                       Line2D([], [], color="C1", lw=2, label="glitch (removed)"), Line2D([], [], color="C2", marker="v", ls="", label="expected toggle")],
              fontsize=7, loc="upper right", ncol=4)


def analog_trace(folder, bi, t_lo, t_hi):
    d = pathlib.Path("C:/data/t2p_qc/analog") / folder
    tr = json.loads((d / "traces.json").read_text())
    info = tr["blocks"][f"b{bi}"]
    B = np.load(d / "barcodes.npz")
    pe, ps = B["pe"], B["ps"]
    A = np.polyfit(pe, ps, 1)

    def pw(t):
        return np.polyval(A, t) + np.interp(t, pe, ps - np.polyval(A, pe))

    y = np.load(d / f"trace_b{bi}.npy", mmap_mode="r")
    n = len(y)
    samp = info["sample0"] + np.arange(n) * info["decim"] + (info["decim"] - 1) / 2
    t_all_lo, t_all_hi = pw(samp[0] / 30000.0), pw(samp[-1] / 30000.0)
    i0 = int((t_lo - t_all_lo) / (t_all_hi - t_all_lo) * n) - 50
    i1 = int((t_hi - t_all_lo) / (t_all_hi - t_all_lo) * n) + 50
    i0, i1 = max(i0, 0), min(i1, n)
    return pw(samp[i0:i1] / 30000.0), np.asarray(y[i0:i1], float) * info.get("scale", 1.0)


# --- figure 1: examples -------------------------------------------------------------------
fig, axs = plt.subplots(3, 1, figsize=(13, 11))
edges_panel(axs[0], "DRpilot_716718_20240730", 0, 78612, 2, 4,
            "DynamicRouting1, 716718_20240730 block 0 (rejected by npc_stim: +12 flips): a state-restoring pair in a 3-refresh hold")
edges_panel(axs[1], "DRpilot_706401_20240425", 1, 506, 1, 2,
            "OptoTagging, 706401_20240425 block 1 (rejected: +613 flips): 5 pairs in an 11-refresh hold (one per two missed refreshes)")
b = common.block_data("DRpilot_762526_20250320", 1)
k = int(np.searchsorted(b.vsyncs, 1969.16)) - 1
at, ay = analog_trace("DRpilot_762526_20250320", 1, b.vsyncs[k - 3] - 0.03, b.vsyncs[k + 5] + 0.06)
edges_panel(axs[2], "DRpilot_762526_20250320", 1, k, 2, 4,
            "DynamicRouting1, 762526_20250320 block 1 (rejected: +11 TTL flips; recovered by the patch): the pair is a real non-frame dip in AI0",
            analog=(at, ay))
fig.tight_layout()
fig.savefig(FIGS / "phantom_examples.png", dpi=100)
plt.close(fig)
print("phantom_examples.png")

# --- figure 2: population rule ----------------------------------------------------------------
h = pd.read_csv(HERE / "survey_holds.csv.gz")
fig, axs = plt.subplots(1, 2, figsize=(13, 4))
ns = sorted(h.n.unique())
ns = [n for n in ns if n <= 13]
mat = np.zeros((11, len(ns)))
for j, n in enumerate(ns):
    x = h[h.n == n]
    for m in range(0, 11):
        mat[m, j] = int((x.n_phantom == m).sum())
ax = axs[0]
with np.errstate(divide="ignore"):
    im = ax.imshow(np.log10(mat + (mat == 0) * 0.0), aspect="auto", origin="lower", cmap="viridis")
for j, n in enumerate(ns):
    for m in range(11):
        if mat[m, j]:
            ax.text(j, m, f"{int(mat[m, j])}", ha="center", va="center", fontsize=6, color="w" if mat[m, j] < 1000 else "k")
ax.plot(np.arange(len(ns)), [n - 1 for n in ns], "r--", lw=1, label="n - 1 (one edge per missed refresh)")
ax.set_xticks(range(len(ns)))
ax.set_xticklabels(ns)
ax.set_xlabel("held frame: refreshes n (stim frameIntervals)")
ax.set_ylabel("phantom-flicker edges in the hold")
ax.set_ylim(-0.5, 10.5)
ax.legend(fontsize=7, loc="upper left")
ax.set_title("362,082 holds in 1,075 box-1 blocks (442 sessions): phantom edges are\nalways an even number <= n - 1, and absent in 2-refresh holds", fontsize=9)
ax = axs[1]
og = h[(h.fi_ms / (1000 / 59.95) - h.n).abs() < 0.1]
for lab, sel, c in (("no phantom edges", og.n_phantom == 0, "C0"), ("with phantom edges", og.n_phantom > 0, "C3")):
    x = og[sel & og.n.isin([3, 4, 10, 11])].res_next_ms.dropna()
    ax.hist(x.clip(-12, 3), bins=np.arange(-12, 3.25, 0.25), histtype="step", color=c, label=f"{lab} (n={len(x)})", log=True)
ax.set_xlabel("next frame's toggle - expected time (ms), holds of 3, 4, 10, 11 refreshes")
ax.set_ylabel("holds")
ax.legend(fontsize=7)
ax.set_title("after a hold the next toggle can come up to ~9 ms early (flicker below\nthe TTL threshold, or the last flicker transition merging with the frame's)", fontsize=9)
fig.tight_layout()
fig.savefig(FIGS / "phantom_population.png", dpi=110)
plt.close(fig)
print("phantom_population.png")

# --- figure 3: box-1 span gate ---------------------------------------------------------------
g = pd.read_csv(HERE / "diode_gate_recheck.csv")
d = pd.read_csv(HERE / "diag_polarity.csv")
b1 = g[g.box == 1].copy()
fig, axs = plt.subplots(1, 2, figsize=(13, 4))
ax = axs[0]
ax.hist(d.old_lat_r_ms.dropna(), bins=np.arange(0, 30, 0.5), alpha=0.6, label="old pairing, rising")
ax.hist(d.old_lat_f_ms.dropna(), bins=np.arange(0, 30, 0.5), alpha=0.6, label="old pairing, falling")
ax.hist(d.new_lat_r_ms.dropna(), bins=np.arange(0, 30, 0.5), histtype="step", lw=2, color="C2", label="alignment, rising")
ax.hist(d.new_lat_f_ms.dropna(), bins=np.arange(0, 30, 0.5), histtype="step", lw=2, color="C3", label="alignment, falling")
ax.axvline(1000 / 59.95, color="k", ls=":", lw=1)
ax.text(1000 / 59.95 + 0.3, ax.get_ylim()[1] * 0.9, "1 refresh", fontsize=7)
ax.set_xlabel("vsync -> TTL photodiode latency (ms), 60 box-1 replay spans")
ax.set_ylabel("spans")
ax.legend(fontsize=7)
ax.set_title("predict_diode_edges paired box-1 edges with the next frame\n(latency estimated one refresh short)", fontsize=9)
ax = axs[1]
x_old = b1.old_baseline_ms.to_numpy()
x_new = b1.new_baseline_ms.to_numpy()
ok = np.isfinite(x_old) & np.isfinite(x_new)
ax.scatter(x_old[ok], x_new[ok], s=12)
ax.plot([0.05, 40], [0.05, 40], "k:", lw=1)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("old pre-span baseline error (ms)")
ax.set_ylabel("new pre-span baseline error (ms)")
n_old = int((~b1.old_fail).sum())
n_new = int((b1.new_status == "reconstructed").sum())
ax.set_title(f"box-1 spans: {n_old}/60 passed the gate before, {n_new}/60 now\n(the other 7: block vsyncs not reconstructed, 0.2 ms gate)", fontsize=9)
fig.tight_layout()
fig.savefig(FIGS / "box1_gate.png", dpi=110)
plt.close(fig)
print("box1_gate.png")
