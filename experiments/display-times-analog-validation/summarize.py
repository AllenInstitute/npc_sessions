"""Pool compare.py results across sessions; checks for the handoff groups; figures.

    C:/data/t2p_qc/venv_analog/Scripts/python summarize.py > summary.log

Writes results_blocks.csv, results_changed_frames.csv, results_late_fall.csv and figs/*.png.
"""
from __future__ import annotations
import json, pathlib
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
A_DIR = pathlib.Path("C:/data/t2p_qc/analog")
BASE = pathlib.Path("C:/data/t2p_qc/out/baseline")
FIX = pathlib.Path("C:/data/t2p_qc/out_issues/fixE")
FR = 1 / 59.95
MCRT = 0.008
SESS = sorted(p.stem for p in (HERE / "results").glob("DRpilot_*.csv"))

rows = pd.concat([pd.read_csv(HERE / "results" / f"{s}.csv") for s in SESS], ignore_index=True)
rows.to_csv(HERE / "results_blocks.csv", index=False)
ok = rows[rows.box.notna()].copy()
print(f"sessions {len(SESS)}, blocks with analog {len(ok)}, toggles checked {int(ok.n_assigned.sum())}")

# ---- 1. per-block accuracy table ---------------------------------------------------------------- #
cols = ["folder", "block", "stim", "box", "source_base", "source_fix", "n_toggles", "n_analog_raw", "n_analog_unmatched",
        "n_toggles_unmatched", "fix_err_med", "fix_err_p1", "fix_err_p99", "fix_err_maxabs", "fix_n_gt2ms", "fix_n_gt4ms",
        "fix_n_gt8ms", "fix_n_edge_shifted", "fix_gt4_same_edge", "fix_gt4_other_edge", "fix_gt4_adjusted",
        "base_err_maxabs", "base_n_gt8ms", "base_n_edge_shifted", "lat_fall_median_ms", "lat_rise_median_ms",
        "lat_rise_min_ms", "lat_rise_p01_ms", "pulse_median_ms", "n_dp_vs_ordinal", "n_landmarks_analog"]
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
t = ok[cols].copy()
t["stim"] = t.stim.str.split("_").str[0]
t["folder"] = t.folder.str[8:]
print(t.round(2).to_string(index=False))

# ---- 2. pooled error distributions (branch), by box interval -------------------------------------- #
pool = {1: [], 3: []}
for s in SESS:
    z = np.load(HERE / "results" / f"{s}_frames.npz")
    for k in z.files:
        if k.endswith("_err_fix"):
            b = int(k.split("_")[0][1:])
            N = int(ok[(ok.folder == s) & (ok.block == b)].box.iloc[0])
            pool[N].append(z[k][~np.isnan(z[k])] * 1e3)
print("\npooled |err| (branch), toggles with a display time:")
for N, L in pool.items():
    if not L:
        continue
    e = np.concatenate(L)
    print(f"  box {N}: n={len(e)} median {np.median(e):+.3f} p1 {np.percentile(e, 1):+.2f} p99 {np.percentile(e, 99):+.2f} "
          f"p99.9 |e| {np.percentile(np.abs(e), 99.9):.2f} max |e| {np.abs(e).max():.2f}; "
          f">2 ms {np.sum(np.abs(e) > 2)} ({np.mean(np.abs(e) > 2) * 100:.3f} %), >4 ms {np.sum(np.abs(e) > 4)}, >8 ms {np.sum(np.abs(e) > 8)}")

# ---- 3. frames changed by the branch ------------------------------------------------------------- #
ch_rows = []
for s in SESS:
    zb, zf = np.load(BASE / f"{s}.npz"), np.load(FIX / f"{s}.npz")
    z = np.load(HERE / "results" / f"{s}_frames.npz")
    S = np.load(A_DIR / s / "sync.npz")
    for b in ok[ok.folder == s].block:
        k = f"A_block{b}"
        N = int(ok[(ok.folder == s) & (ok.block == b)].box.iloc[0])
        if k not in zb or k not in zf:
            if (k in zb) != (k in zf):
                ch_rows.append(dict(folder=s, block=b, kind="source changed", n=-1))
            continue
        tb, tf = zb[k], zf[k]
        if len(tb) != len(tf):
            continue
        d = np.where(np.abs(tb - tf) > 1e-6)[0]
        if not len(d):
            continue
        f_tog = z[f"b{b}_f"]; a = z[f"b{b}_a"]
        v = S[f"block{b}"]; vra = S["v_rise"]; vr = vra[np.clip(np.searchsorted(vra, v) - 1, 0, None)]
        for fr in d:
            is_t = fr % N == 0
            rec = dict(folder=s, block=b, frame=int(fr), box=N, toggle=bool(is_t), base_minus_fix_ms=(tb[fr] - tf[fr]) * 1e3,
                       base_minus_rise_ms=(tb[fr] - MCRT - vr[fr]) * 1e3, fix_minus_rise_ms=(tf[fr] - MCRT - vr[fr]) * 1e3)
            # the analog toggle that starts this frame's group and the next one
            g0 = fr - fr % N
            i0 = np.searchsorted(f_tog, g0)
            if i0 < len(f_tog) and f_tog[i0] == g0 and i0 + 1 < len(a):
                rec["group_start_analog_ms_rel_fix"] = (a[i0] - (tf[fr] - MCRT)) * 1e3
                rec["group_len_refreshes"] = (a[i0 + 1] - a[i0]) / FR
                rec["fix_slot"] = (tf[fr] - MCRT - a[i0]) / FR
                rec["base_slot"] = (tb[fr] - MCRT - a[i0]) / FR
            if is_t:
                j = np.searchsorted(f_tog, fr)
                if j < len(f_tog) and f_tog[j] == fr:
                    rec["err_base_ms"] = z[f"b{b}_err_base"][j] * 1e3 if f"b{b}_err_base" in z.files else np.nan
                    rec["err_fix_ms"] = z[f"b{b}_err_fix"][j] * 1e3
            ch_rows.append(rec)
ch = pd.DataFrame(ch_rows)
ch.to_csv(HERE / "results_changed_frames.csv", index=False)
print(f"\nframes changed baseline -> branch in the sample: {int((ch.get('frame', pd.Series(dtype=float)) >= 0).sum()) if len(ch) else 0}")
if len(ch):
    print(ch.groupby(["folder", "block"]).agg(n=("frame", "size"), toggles=("toggle", "sum")).to_string())
    c2 = ch[ch.frame.notna()]
    print("  changed frames that are box toggles:", int(c2.toggle.sum()), "of", len(c2))
    print("  base display - 8 ms - vsync rise (ms): min", round(c2.base_minus_rise_ms.min(), 1), "; branch:", round(c2.fix_minus_rise_ms.min(), 1))
    if "group_len_refreshes" in c2:
        print("  analog length of the box group holding each changed frame (refreshes):",
              np.round(np.sort(c2.group_len_refreshes.dropna().values), 2).tolist()[:40])
        print("  slot of frame in group (refreshes after the group's analog toggle): base", np.round(c2.base_slot.values, 2).tolist()[:20],
              "branch", np.round(c2.fix_slot.values, 2).tolist()[:20])

# ---- 4. frames displayed before their vsync falling edge (late falling edge, §1a) ---------------- #
lf_rows = []
for s in SESS:
    zf = np.load(FIX / f"{s}.npz"); z = np.load(HERE / "results" / f"{s}_frames.npz"); S = np.load(A_DIR / s / "sync.npz")
    for b in ok[ok.folder == s].block:
        k = f"A_block{b}"
        if k not in zf or f"b{b}_err_fix" not in z.files:
            continue
        v = S[f"block{b}"]; t = zf[k] - MCRT
        if len(t) != len(v):
            continue
        late = np.where(t < v)[0]
        f_tog = z[f"b{b}_f"]
        for fr in late:
            j = np.searchsorted(f_tog, fr)
            hit = j < len(f_tog) and f_tog[j] == fr
            lf_rows.append(dict(folder=s, block=b, frame=int(fr), toggle=bool(hit),
                                display_minus_fall_ms=(t[fr] - v[fr]) * 1e3,
                                err_fix_ms=z[f"b{b}_err_fix"][j] * 1e3 if hit else np.nan,
                                analog_minus_fall_ms=(z[f"b{b}_a"][j] - v[fr]) * 1e3 if hit else np.nan))
lf = pd.DataFrame(lf_rows)
lf.to_csv(HERE / "results_late_fall.csv", index=False)
if len(lf):
    tt = lf[lf.toggle]
    print(f"\nframes with branch display - 8 ms < vsync falling edge: {len(lf)} ({len(tt)} are box toggles, observable)")
    print(f"  toggles: analog - falling edge {tt.analog_minus_fall_ms.min():.1f} .. {tt.analog_minus_fall_ms.max():.1f} ms; "
          f"|err| median {tt.err_fix_ms.abs().median():.2f} max {tt.err_fix_ms.abs().max():.2f} ms")
    print(tt.round(2).to_string(index=False))

# ---- 5. latency calibration ------------------------------------------------------------------- #
print("\nlatency of the analog transition after the toggle frame's vsync edges (all checked toggles):")
allr, allf = [], []
for s in SESS:
    z = np.load(HERE / "results" / f"{s}_frames.npz")
    for k in z.files:
        if k.endswith("_lat_r"):
            allr.append(z[k]); allf.append(z[k.replace("_lat_r", "_lat_f")])
lr, lfa = np.concatenate(allr), np.concatenate(allf)
print(f"  from rising edge: n={len(lr)} min {lr.min():.1f} p0.01 {np.percentile(lr, 0.01):.1f} p0.1 {np.percentile(lr, 0.1):.1f} "
      f"p1 {np.percentile(lr, 1):.1f} median {np.median(lr):.1f} ms; < 15 ms: {int(np.sum(lr < 15))}")
print(f"  from falling edge: median {np.median(lfa):.2f} p1 {np.percentile(lfa, 1):.2f} p99 {np.percentile(lfa, 99):.2f} ms; "
      f"min {lfa.min():.1f} (negative = displayed before its falling edge: {int(np.sum(lfa < 0))})")
print(ok.groupby("folder")[["lat_fall_median_ms", "lat_rise_median_ms", "pulse_median_ms", "lat_rise_min_ms"]].agg(["median", "min"]).round(2).to_string())

# ---- figures ----------------------------------------------------------------------------------- #
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for i, (N, L) in enumerate(pool.items()):
    if not L:
        continue
    e = np.concatenate(L)
    ax[i].hist(np.clip(e, -15, 15), bins=np.arange(-15, 15.25, 0.25), color="C0")
    ax[i].set_yscale("log"); ax[i].set_xlabel("analog - (branch display - 8 ms) - lag  [ms, clipped at +/-15]")
    ax[i].set_title(f"box interval {N}: {len(e):,} toggle frames, {sum(len(x) > 0 for x in L)} blocks", fontsize=9)
    for x in (-2, 2):
        ax[i].axvline(x, color="k", ls=":", lw=0.8)
fig.tight_layout(); fig.savefig(HERE / "figs" / "err_hist.png", dpi=110)

fig, ax = plt.subplots(figsize=(12, 4.5))
lab = [f"{r.folder[8:]} b{r.block} {r.stim.split('_')[0][:12]}" for r in ok.itertuples()]
x = np.arange(len(ok))
ax.vlines(x, ok.fix_err_p1, ok.fix_err_p99, color="C0", lw=3, label="branch p1-p99")
ax.plot(x, ok.fix_err_maxabs, "v", color="C3", ms=4, label="branch max |err|")
ax.plot(x, ok.base_err_maxabs, "x", color="C1", ms=4, label="baseline max |err|")
ax.axhline(2, color="k", ls=":", lw=0.8); ax.axhline(-2, color="k", ls=":", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(lab, rotation=90, fontsize=5); ax.set_ylabel("ms"); ax.set_ylim(-5, 15)
ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(HERE / "figs" / "err_by_block.png", dpi=120)

fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
ax[0].hist(lr, bins=np.arange(0, 80, 0.5)); ax[0].set_yscale("log"); ax[0].axvline(15, color="r", ls="--", lw=1)
ax[0].set_xlabel("analog transition - vsync rising edge of its frame (ms)"); ax[0].set_title("red: MIN_VSYNC_DIODE_FLIP_SEPARATION 15 ms", fontsize=8)
ax[1].hist(lfa, bins=np.arange(-40, 80, 0.5)); ax[1].set_yscale("log"); ax[1].axvline(19.55, color="r", ls="--", lw=1)
ax[1].set_xlabel("analog transition - vsync falling edge of its frame (ms)"); ax[1].set_title("red: AVERAGE_VSYNC_TO_DIODE_FLIP_TIME 19.55 ms (TTL)", fontsize=8)
fig.tight_layout(); fig.savefig(HERE / "figs" / "latency_calibration.png", dpi=110)
print("\nfigs: err_hist.png err_by_block.png latency_calibration.png")
