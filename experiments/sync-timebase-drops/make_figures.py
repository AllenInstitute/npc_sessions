"""Figures for outputs/sync-dropped-chunks-evidence.pdf: is a "replay" a drop made up for by
copies? Real data only; released npc_sync 0.3.8.

Run: ../../.venv/Scripts/python.exe make_figures.py
Reads sync files from C:/data/t2p_cache and the Task 2 Pilot share, survey.jsonl and
vsync_gap_survey.jsonl from this folder. Writes ../../outputs/sync-dropped-chunks-evidence.pdf
and figure_numbers.json (every number quoted in a caption).
"""
from __future__ import annotations

import json
import logging
import pathlib

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

import npc_sync
from npc_sync.sync import get_bit
from npc_sync.timebase import find_chunk_jumps

logging.disable(logging.WARNING)
HERE = pathlib.Path(__file__).parent
OUT = HERE.parents[1] / "outputs" / "sync-dropped-chunks-evidence.pdf"
CACHE = pathlib.Path("C:/data/t2p_cache")
SHARE = pathlib.Path("//allen/programs/mindscope/workgroups/dynamicrouting/PilotEphys/Task 2 Pilot")
CH = 50_000
FS = 100_000.0
COLORS = {"beh": "#1f77b4", "face": "#d62728", "eye": "#2ca02c", "nose": "#9467bd"}
NUM: dict = {}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def sync_path(session: str) -> pathlib.Path:
    for root in (CACHE, SHARE):
        d = root / session
        if d.exists():
            return next(p for p in d.glob("*.h5") if "T" in p.stem)
    raise FileNotFoundError(session)


class File:
    def __init__(self, session: str) -> None:
        self.session = session
        self.raw = npc_sync.SyncDataset(sync_path(session), on_invalid="ignore")
        ev = self.raw.get_recorded_events()
        self.times = ev[:, 0].astype(np.int64)
        self.bits = ev[:, -1]
        self.tb = self.raw.find_timebase()
        n = int(np.ceil(self.raw.meta_data["total_samples"] / CH))
        jump = np.zeros(n + 1, dtype=int)
        for j in self.tb.jumps:
            jump[j.sample // CH] += j.n_chunks
        self.state = np.cumsum(jump)[:n]  # shift state of every recorded chunk
        self.copies = [(a / FS, z / FS) for a, z in self.raw._replayed_buffers[0]]

    def edges(self, label: str, kind: int = 1) -> np.ndarray:
        change = np.ediff1d(get_bit(self.bits, self.raw._line_to_bit(label)), to_begin=np.uint8(0))
        return self.times[change == kind]

    def cameras(self) -> list[str]:
        return [lab for lab in self.raw.line_labels if lab.endswith("_cam_exposing")]

    def phase(self, label: str, t0: float, t1: float):
        """(edge time s, phase in chunks) for rising edges in [t0, t1], as recorded.
        Phase = (edge time - period x index), detrended on chunks with shift state 0 (or the
        state at t0, for a file that never returns), divided by the slip of one chunk."""
        t = self.edges(label)
        t = t[(t >= (t0 - 20) * FS) & (t <= (t1 + 20) * FS)]
        dt = np.diff(t)
        p0 = np.median(dt)
        reg = np.abs(dt - p0) < 0.25 * p0
        period = dt[reg].mean()
        u = np.r_[0.0, np.cumsum(np.where(reg, dt - period, 0.0))]
        slip = round(CH / period) * period - CH
        st = self.state[np.clip(t // CH, 0, len(self.state) - 1)]
        ref = st == st[0]
        ref &= ~self.in_copies(t / FS)
        a = np.polyfit(t[ref], u[ref], 1)
        y = (u - np.polyval(a, t)) / slip + st[0]
        keep = (t >= t0 * FS) & (t <= t1 * FS)
        return t[keep] / FS, y[keep], slip / FS * 1e6

    def in_copies(self, t: np.ndarray) -> np.ndarray:
        m = np.zeros(len(t), bool)
        for a, z in self.copies:
            m |= (t >= a) & (t < z)
        return m


def phase_panel(ax, f: File, t0: float, t1: float, title: str) -> None:
    slips = {}
    for lab in f.cameras():
        t, y, slip = f.phase(lab, t0, t1)
        cam = lab.split("_")[0]
        slips[cam] = round(slip, 1)
        ax.plot(t, y, ".", ms=2.5, color=COLORS.get(cam, "k"), alpha=0.6, label=f"{cam} camera")
    p = np.arange(int(t0 * 2), int(t1 * 2) + 1)
    ax.step(p / 2, f.state[np.clip(p, 0, len(f.state) - 1)], where="post", color="k", lw=1.2, label="shift state s (timebase)")
    for a, z in f.copies:
        if z > t0 and a < t1:
            ax.axvspan(max(a, t0), min(z, t1), color="#ffcc66", alpha=0.45, lw=0)
    for x in np.arange(np.ceil(t0 * 2) / 2, t1 + 1e-9, 0.5):
        ax.axvline(x, color="0.9", lw=0.5, zorder=0)
    ax.set_xlim(t0, t1)
    ax.set_ylabel("camera phase\n(chunks early)")
    ax.set_title(title, loc="left", fontsize=9)
    NUM.setdefault("slip_us_per_chunk", {})[f.session] = slips


def barcode_onsets(f: File, true_clock: bool):
    r = f.edges("barcode_ephys") if "barcode_ephys" in f.raw.line_labels else np.array([])
    t = r / FS
    starts = t[np.r_[True, np.diff(t) > 20]]
    if true_clock:
        tr, keep = f.tb.to_true((starts * FS).astype(np.int64))
        starts = tr[keep] / FS
        bad = [(a / FS, z / FS) for a, z in f.tb.dropped]
    else:
        bad = f.copies
    ok = np.ones(len(starts), bool)
    for a, z in bad:  # a barcode cut by a copy / a dropped interval has no usable onset
        ok &= ~((starts + 2 > a) & (starts - 2 < z))
    total = f.raw.meta_data["total_samples"] / FS + (f.tb.net_shift_samples / FS if true_clock else 0)
    ok &= (starts > 2) & (starts + 2 < total)
    return starts[ok]


def barcode_deviation(starts: np.ndarray):
    iv = np.diff(starts)
    period = np.median(iv[np.round(iv / iv.min()) == 1])
    dev = iv - np.maximum(np.round(iv / period), 1) * period
    return starts[1:], dev, period


def page_text(pdf, title: str, lines: list[str]) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.93, title, fontsize=16, weight="bold", va="top")
    y = 0.87
    for line in lines:
        if line.startswith("#"):
            y -= 0.012
            fig.text(0.08, y, line[1:].strip(), fontsize=11, weight="bold", va="top")
            y -= 0.028
        elif line == "":
            y -= 0.012
        else:
            import textwrap

            indent = "   " if line[:2] == "- " or line[:1].isdigit() else ""
            for i, part in enumerate(textwrap.wrap(line, 104) or [""]):
                fig.text(0.08, y, (indent if i else "") + part, fontsize=9.5, va="top")
                y -= 0.0185
            y -= 0.004
    pdf.savefig(fig)
    plt.close(fig)


def caption(fig, text: str, y: float = 0.02) -> None:
    fig.text(0.07, y, text, fontsize=8.5, va="bottom", wrap=True)


# --------------------------------------------------------------------------------------- data
print("loading files")
F1 = File("DRpilot_703880_20240418")
F2 = File("DRpilot_840160_20260428")
F3 = File("DRpilot_715709_20240808")

figs = []

# ------------------------------------------------------------------ fig 1: one event in detail
fig, axes = plt.subplots(3, 1, figsize=(8.27, 11.0), gridspec_kw=dict(height_ratios=[3, 1.1, 3], hspace=0.6, top=0.95, bottom=0.17, left=0.13, right=0.96))
phase_panel(axes[0], F1, 920.0, 934.0, "A  703880_2024-04-18, 920-934 s: camera phase of every exposure, as recorded")
axes[0].legend(loc="lower center", ncol=4, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.27), markerscale=3)
axes[0].set_ylim(-0.7, 3.6)
axes[0].annotate("chunk lost\nhere", (923.0, 1.0), (920.4, 2.6), arrowprops=dict(arrowstyle="->", lw=0.7), fontsize=8)
axes[0].annotate("another", (926.0, 2.0), (923.9, 3.2), arrowprops=dict(arrowstyle="->", lw=0.7), fontsize=8)
axes[0].annotate("2 identical chunks\n(the 'replay')", (930.5, 1.5), (931.5, 2.9), arrowprops=dict(arrowstyle="->", lw=0.7), fontsize=8)

# barcode line raster
ax = axes[1]
r, fl = F1.edges("barcode_ephys", 1) / FS, F1.edges("barcode_ephys", 255) / FS
m = (r > 920) & (r < 934)
for a, z in zip(r[m], fl[np.searchsorted(fl, r[m])]):
    ax.plot([a, a, z, z], [0, 1, 1, 0], color="k", lw=0.8)
on_rec = barcode_onsets(F1, False)
_, _, period = barcode_deviation(on_rec)
before = on_rec[on_rec < 900][-1]
expected = before + round((926.49 - before) / period) * period
ax.axvline(expected, color="#d62728", ls="--", lw=1)
ax.text(expected + 0.1, 1.25, f"where this barcode is due ({expected:.2f} s:\na whole number of periods after the previous one)", color="#d62728", fontsize=7.5, va="bottom")
got = r[m][0]
ax.text(got - 0.1, 1.25, f"recorded at {got:.2f} s", fontsize=7.5, ha="right", va="bottom")
for a, z in F1.copies:
    if z > 920 and a < 934:
        ax.axvspan(a, z, color="#ffcc66", alpha=0.45, lw=0)
ax.set_xlim(920, 934)
ax.set_ylim(-0.2, 2.6)
ax.set_yticks([])
ax.set_title("B  barcode line, same window, as recorded", loc="left", fontsize=9)
NUM["fig1"] = dict(barcode_recorded_s=float(got), barcode_due_s=float(expected), early_by_s=float(expected - got), barcode_period_s=float(period))

phase_panel(axes[2], F1, 1640.0, 1660.0, "C  same file, 1640-1660 s: drops at the start of their run, and drops seconds before it")
axes[2].set_ylim(-0.7, 4.6)
axes[2].set_xlabel("time in file (s, recorded sample clock)")
axes[2].annotate("lost and replaced\nin place (gap 0)", (1642.5, 1.0), (1640.2, 3.0), arrowprops=dict(arrowstyle="->", lw=0.7), fontsize=8)
axes[2].annotate("lost here ...", (1643.5, 1.0), (1643.4, 3.9), arrowprops=dict(arrowstyle="->", lw=0.7), fontsize=8)
axes[2].annotate("... made up\nhere", (1646.75, 1.6), (1647.9, 3.6), arrowprops=dict(arrowstyle="->", lw=0.7), fontsize=8)
s1 = NUM["slip_us_per_chunk"][F1.session]
caption(fig, (
    "Figure 1. Each dot is one camera exposure. Its phase is how early it is against that camera's own strict period, in units of\n"
    f"the slip of one 0.5 s chunk (a chunk is 30 camera periods minus {abs(s1['beh'])}, {abs(s1['face'])}, {abs(s1['eye'])} us for the three cameras). Grey lines: chunk\n"
    "boundaries. Orange: chunks that are identical copies of the next chunk. Black: the shift state the timebase assigns.\n"
    "A: three independent cameras step together by exactly one unit at 923.0 and 926.0 s, stay there, and step back one unit per\n"
    "copy. Nothing in A uses the barcode. B: the barcode inside that stretch is recorded "
    f"{expected - got:.3f} s before it is due: two chunks.\n"
    "C: at 1642.5 s the step up and the copy coincide (a chunk lost and replaced in place). At 1643.5 and 1645.5 s chunks are lost\n"
    "with no copy; the two copies come at 1646.5 and 1647.5 s. Same signature, different distance."))
figs.append(fig)

# ------------------------------------------------------------------ fig 2: pure drops + integers
fig = plt.figure(figsize=(8.27, 11.0))
gs = fig.add_gridspec(3, 2, height_ratios=[2.2, 2.2, 2.6], hspace=0.5, wspace=0.28, top=0.94, bottom=0.16, left=0.12, right=0.96)
ax = fig.add_subplot(gs[0, :])
phase_panel(ax, F2, 2462.0, 2482.0, "A  840160_2026-04-28: 2 chunks lost at 2472.0 s, never made up for (no copies in this file)")
ax.set_ylim(-0.7, 3.6)
ax.legend(loc="upper left", ncol=5, fontsize=7.5, frameon=False, markerscale=3)
ax = fig.add_subplot(gs[1, :])
phase_panel(ax, F3, 250.0, 268.0, "B  715709_2024-08-08: 4 chunks lost at 258.5 s")
ax.set_ylim(-0.6, 4.9)
ax.set_xlabel("time in file (s, recorded sample clock)")

# histogram of jump values at every boundary
hist_files = [("703880_2024-04-18\n(58 drops, 58 copies)", F1), ("840160_2026-04-28\n(one drop of 2)", F2)]
for k, (name, f) in enumerate(hist_files):
    ax = fig.add_subplot(gs[2, k])
    edges = {}
    for lab in f.cameras():
        edges[f"{lab}:rising"] = f.edges(lab, 1)
        edges[f"{lab}:falling"] = f.edges(lab, 255)
    j = find_chunk_jumps(edges, int(f.raw.meta_data["total_samples"]), CH)
    v = j.value[j.observed]
    ax.hist(v, bins=np.arange(-1.5, 4.51, 0.02), color="0.25")
    ax.set_yscale("log")
    ax.set_xlabel("measured jump at a chunk boundary (chunks)")
    if k == 0:
        ax.set_ylabel("chunk boundaries")
    ax.set_title(("C  " if k == 0 else "D  ") + name, loc="left", fontsize=9)
    counts = {int(x): int(c) for x, c in zip(*np.unique(j.jump[j.observed], return_counts=True))}
    off = float(np.abs(v - np.round(v)).max())
    NUM.setdefault("fig2_jumps", {})[f.session] = dict(n_observed=int(j.observed.sum()), counts=counts, max_off_integer=off)
    ax.text(0.97, 0.95, "\n".join(f"{c:,} at {x:+d}" for x, c in counts.items()) + f"\nfarthest from an\ninteger: {off:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=7.5)
n1, n2 = NUM["fig2_jumps"][F1.session], NUM["fig2_jumps"][F2.session]
caption(fig, (
    "Figure 2. A, B: the same measurement in two files with no copies at all. The cameras step up by 2 and by 4 units at one chunk\n"
    "boundary and never come back: everything after is recorded 1.0 s and 2.0 s early. These are the files the barcodes flagged\n"
    "(-1.000 s and -2.000 s). C, D: the jump measured at every chunk boundary of a whole file (median over cameras), log scale.\n"
    f"C: {n1['n_observed']:,} boundaries: {n1['counts'].get(1, 0)} jumps of +1, {n1['counts'].get(-1, 0)} of -1 (one per copy), the rest 0. D: {n2['n_observed']:,} boundaries, one jump of +2.\n"
    "The values are integers, not a continuum, which is what whole chunks going missing predicts and what camera jitter\n"
    "or clock drift does not."))
figs.append(fig)

# ------------------------------------------------------------------ fig 3: barcodes, lag, vsync
fig = plt.figure(figsize=(8.27, 11.0))
gs = fig.add_gridspec(3, 2, height_ratios=[2.4, 2.2, 2.4], hspace=0.55, wspace=0.3, top=0.94, bottom=0.17, left=0.12, right=0.96)
bar_files = [File(s) for s in ("DRpilot_708020_20240517", "DRpilot_714748_20240628")] + [F1, F3, F2]
ax = fig.add_subplot(gs[0, :])
x0 = 0
ticks, labels = [], []
NUM["fig3_barcodes"] = {}
for f in bar_files:
    rec = {}
    for true_clock, color, mk in ((False, "#d62728", "o"), (True, "#1f77b4", ".")):
        starts = barcode_onsets(f, true_clock)
        t, dev, _ = barcode_deviation(starts)
        ax.plot(x0 + np.arange(len(dev)), dev, mk, ms=4 if not true_clock else 3, color=color, mfc="none" if not true_clock else color,
                label=("recorded clock" if not true_clock else "corrected clock") if f is bar_files[0] else None)
        rec["corrected" if true_clock else "recorded"] = dict(n=int(len(dev)), n_off_by_over_10ms=int((np.abs(dev) > 0.01).sum()), max_abs_s=float(np.abs(dev).max()))
    NUM["fig3_barcodes"][f.session] = rec
    n = rec["recorded"]["n"]
    ticks.append(x0 + n / 2)
    labels.append(f.session.replace("DRpilot_", "")[:6] + "\n" + f.session[-4:])
    x0 += n + 12
    ax.axvline(x0 - 6, color="0.85", lw=0.6)
ax.set_xticks(ticks, labels, fontsize=7.5)
ax.set_ylabel("barcode interval minus\nwhole periods (s)")
ax.set_yticks(np.arange(-2, 2.01, 0.5))
ax.set_ylim(-2.3, 2.6)
ax.grid(axis="y", color="0.92", lw=0.5)
ax.legend(loc="upper right", fontsize=7.5, frameon=False, ncol=2)
ax.set_title("A  every barcode onset interval in 5 files", loc="left", fontsize=9)

# lag histogram and drop sizes from the survey
rows = [json.loads(l) for l in open(HERE / "survey.jsonl")]
rows = [r for r in rows if "error" not in r]
lags = np.array([l for r in rows for l in r["lags_s"] if l is not None])
ax = fig.add_subplot(gs[1, 0])
ax.hist(lags, bins=np.arange(-0.25, 21, 0.5), color="0.25")
ax.set_yscale("log")
ax.set_xlabel("from a lost chunk to the run of copies\nthat makes up for it (s)")
ax.set_ylabel("drops")
ax.set_title("B  152 files with copies", loc="left", fontsize=9)
NUM["fig3_lags"] = dict(n=int(len(lags)), n_zero=int((lags == 0).sum()), median_nonzero=float(np.median(lags[lags > 0])), max=float(lags.max()))
ax = fig.add_subplot(gs[1, 1])
n_copies = np.array([r["n_copies"] for r in rows if r["n_copies"]])
n_located = np.array([sum(k for _, k, _ in r["drops_measured"]) for r in rows if r["n_copies"]])
ax.plot(n_copies, n_located, "o", ms=4, mfc="none", color="0.25")
lim = [0.8, n_copies.max() * 1.3]
ax.plot(lim, lim, color="0.7", lw=0.8, zorder=0)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("copies in the file")
ax.set_ylabel("chunks the cameras say were lost")
ax.set_title("C  one point per file with copies", loc="left", fontsize=9)
NUM["fig3_balance"] = dict(n_files=int(len(n_copies)), n_equal=int((n_copies == n_located).sum()), total_copies=int(n_copies.sum()), total_located=int(n_located.sum()),
                           unequal=[(r["session"], r["n_copies"], sum(k for _, k, _ in r["drops_measured"])) for r in rows if r["n_copies"] and r["n_copies"] != sum(k for _, k, _ in r["drops_measured"])])

# vsync gap residuals
gap = [json.loads(l) for l in open(HERE / "vsync_gap_survey.jsonl")]
res = {m: np.array([x for r in gap if r.get("model") == m for x in r.get("residuals_ms", [])]) for m in ("recorded", "timebase")}
ax = fig.add_subplot(gs[2, :])
bins = np.arange(-3.0, 3.001, 0.03)
ax.hist(np.clip(res["recorded"], -3, 3), bins=bins, color="#d62728", alpha=0.6, label=f"copies treated as overwritten data ({len(res['recorded'])} gaps)")
ax.hist(np.clip(res["timebase"], -3, 3), bins=bins, color="#1f77b4", alpha=0.7, label=f"corrected clock ({len(res['timebase'])} gaps)")
for k in range(-6, 7):
    if k:
        ax.axvline(-0.42 * k, color="0.6", lw=0.5, ls=":")
ax.set_yscale("log")
ax.set_xlabel("measured time across a gap minus the stim file's time for the missing frames (ms; beyond +-3 drawn at +-3)")
ax.set_ylabel("gaps")
ax.legend(fontsize=7.5, frameon=False, loc="upper right")
ax.set_title("D  display frames: per-gap residual in 160 stimulus blocks of 154 files", loc="left", fontsize=9)
NUM["fig3_vsync"] = {m: dict(n=int(len(v)), median_abs_ms=float(np.median(np.abs(v))), n_over_0p2=int((np.abs(v) > 0.2).sum()), n_over_1=int((np.abs(v) > 1).sum())) for m, v in res.items()}
b, lg, bal, vs = NUM["fig3_barcodes"], NUM["fig3_lags"], NUM["fig3_balance"], NUM["fig3_vsync"]
caption(fig, (
    "Figure 3. Three checks that do not use the cameras. A: barcodes come from a separate generator at a fixed period. On the\n"
    f"recorded clock (red) {sum(v['recorded']['n_off_by_over_10ms'] for v in b.values())} intervals in these 5 files are off by multiples of 0.5 s; in files with copies they come in -/+ pairs\n"
    f"(a barcode recorded early, the next interval long). On the corrected clock (blue) {sum(v['corrected']['n_off_by_over_10ms'] for v in b.values())} are off by more than 10 ms.\n"
    f"B: {lg['n_zero']} of {lg['n']} drops are at the start of their run (gap 0); the rest are up to {lg['max']} s before it (median {lg['median_nonzero']} s).\n"
    f"C: in {bal['n_equal']} of {bal['n_files']} files the chunks the cameras say were lost equal the number of copies ({bal['total_located']:,} and {bal['total_copies']:,} in total).\n"
    f"D: if copies simply overwrote data, the frames either side of a run would be on time and the residual would be vsync jitter.\n"
    f"Instead {vs['recorded']['n_over_0p2']} of {vs['recorded']['n']} gaps are off by more than 0.2 ms, clustered at multiples of 0.42 ms (dotted): a chunk is 30 display\n"
    f"refreshes minus 0.42 ms, so a frame recorded one chunk early shows exactly that. Corrected: {vs['timebase']['n_over_0p2']} of {vs['timebase']['n']}."))
figs.append(fig)

# ------------------------------------------------------------------ fig 4: licks
print("licks")
LICK_SESSIONS = ["DRpilot_702131_20240223", "DRpilot_712139_20240510", "DRpilot_708020_20240517", "DRpilot_714748_20240628", "DRpilot_859582_20260612",
                 "DRpilot_708016_20240426", "DRpilot_708020_20240516", "DRpilot_715709_20240808", "DRpilot_840160_20260428"]
pair_rec, pair_true, pair_on = [], [], []
raster = None
NUM["fig4_sessions"] = {}
for sess in LICK_SESSIONS:
    d = SHARE / sess
    f = {x.session: x for x in bar_files}.get(sess) or File(sess)
    task = sorted(d.glob("DynamicRouting1_*.hdf5"), key=lambda p: p.stat().st_size)[-1]
    with h5py.File(task, "r") as h:
        fi = h["frameIntervals"][:]
        lf = np.asarray(h["lickFrames"][:], dtype=int)
    view = npc_sync.SyncDataset(f.raw, on_invalid="ignore", timebase=f.tb)
    blocks = view.vsync_times_in_blocks
    b = blocks[int(np.argmax([len(x) for x in blocks]))]
    stim = b[0] + np.r_[0, np.cumsum(fi)][lf]
    rec = f.edges("lick_sensor")
    true, keep = f.tb.to_true(rec)
    rec, true = rec[keep] / FS, true[keep] / FS
    inb = (true > b[0]) & (true < b[-1])
    rec, true = rec[inb], true[inb]
    shift = np.round(true - rec, 3)
    on = shift == 0
    if on.sum() < 10:
        continue

    def pairs(t):
        dd = (t[:, None] - stim[None, :]).ravel()
        return dd[np.abs(dd) < 4.5]

    dd = pairs(true[on])
    h_, e_ = np.histogram(dd, bins=np.arange(-4.5, 4.501, 0.01))
    o = e_[int(np.argmax(h_))] + 0.005  # stim clock -> sync clock offset (first frames can be missing)
    stim = stim + o
    dis = shift > 0
    pair_rec.append(pairs(rec[dis])); pair_true.append(pairs(true[dis])); pair_on.append(pairs(true[on]))
    def n_matched(t):  # licks with a stim-file lick within 25 ms
        j = np.clip(np.searchsorted(stim, t), 1, len(stim) - 1)
        return int((np.minimum(np.abs(t - stim[j - 1]), np.abs(t - stim[j])) < 0.025).sum())

    NUM["fig4_sessions"][sess] = dict(n_copies=len(f.copies), matched_recorded=n_matched(rec[dis]), matched_corrected=n_matched(true[dis]),
                                      matched_on_time=n_matched(true[on]), n_displaced=int(dis.sum()), n_on_time=int(on.sum()), shifts={float(k): int(c) for k, c in zip(*np.unique(shift[dis], return_counts=True))})
    if sess == "DRpilot_708020_20240517" and raster is None:
        # densest window of displaced licks
        td = true[dis]
        best = max(td, key=lambda x: ((td >= x) & (td < x + 8)).sum())
        w0, w1 = best - 2.0, best + 8.0
        raster = dict(sess=sess, w=(w0, w1), stim=stim[(stim > w0) & (stim < w1)], rec=rec[(true > w0 - 5) & (true < w1 + 5)], true=true[(true > w0) & (true < w1)],
                      dropped=[(a / FS, z / FS) for a, z in f.tb.dropped if z / FS > w0 and a / FS < w1], f=f)
vals = list(NUM["fig4_sessions"].values())
groups = {
    "copies": [i for i, v in enumerate(vals) if v["n_copies"]],
    "pure": [i for i, v in enumerate(vals) if not v["n_copies"]],
}
OFFSETS = (0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -3.5)


def peak(dd, x):
    return int(((dd > x - 0.03) & (dd < x + 0.03)).sum())


fig = plt.figure(figsize=(8.27, 11.0))
gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 2.2, 2.2], hspace=0.55, wspace=0.22, top=0.95, bottom=0.19, left=0.14, right=0.97)
ax = fig.add_subplot(gs[0, :])
w0, w1 = raster["w"]
rows_ = [
    ("stim file licks\n(stim PC clock)", raster["stim"], "k"),
    ("sync lick line,\nas recorded", raster["rec"], "#d62728"),
    ("sync lick line,\ncorrected clock", raster["true"], "#1f77b4"),
]
for i, (name, t, c) in enumerate(rows_):
    ax.vlines(t, 2 - i - 0.35, 2 - i + 0.35, color=c, lw=1.1)
for a_, z_ in raster["dropped"]:
    ax.axvspan(a_, z_, color="0.85", lw=0)
ax.set_yticks([2, 1, 0], [r[0] for r in rows_], fontsize=7.5)
ax.set_xlim(w0, w1)
ax.set_xlabel("time (s)")
ax.set_title(f"A  {raster['sess'].replace('DRpilot_', '')} (a file with copies): licks around a run of lost chunks", loc="left", fontsize=9)
bins = np.arange(-4.5, 4.501, 0.02)
NUM["fig4"] = {}
for col, (g, gname) in enumerate((("copies", "files with copies"), ("pure", "files, no copies"))):
    idx = groups[g]
    n_l = sum(vals[i]["n_displaced"] for i in idx)
    NUM["fig4"][g] = dict(n_sessions=len(idx), n_displaced_licks=int(n_l), **{k: int(sum(vals[i][k] for i in idx)) for k in ("matched_recorded", "matched_corrected", "matched_on_time", "n_on_time")})
    for row, (parts, c, cname) in enumerate(((pair_rec, "#d62728", "as recorded"), (pair_true, "#1f77b4", "corrected clock"))):
        dd = np.concatenate([parts[i] for i in idx])
        ax = fig.add_subplot(gs[1 + row, col])
        ax.hist(dd, bins=bins, color=c)
        for x in np.arange(-4.5, 4.51, 0.5):
            ax.axvline(x, color="0.88", lw=0.5, zorder=0)
        ax.set_xlim(-4.5, 4.5)
        if col == 0:
            ax.set_ylabel("lick pairs per 20 ms")
        ax.set_title(f"{'BCDE'[2 * col + row]}  {len(idx)} {gname}, {n_l:,} licks: {cname}", loc="left", fontsize=8.5)
        if row == 1:
            ax.set_xlabel("sync lick minus stim-file lick (s)")
        NUM["fig4"][g][cname] = {str(x): peak(dd, x) for x in OFFSETS}
nc, npu = NUM["fig4"]["copies"], NUM["fig4"]["pure"]


def fmt(d, keys):
    return ", ".join(f"{d[str(k)]:,}" for k in keys)


caption(fig, (
    "Figure 4. The stim PC writes the frame of every lick it sees; with its frame intervals that is a lick time on a clock no sync\n"
    "fault can touch. A: one window. Grey: true time that no recorded chunk holds (stim-file licks there have no sync lick).\n"
    "Recorded licks (red) have the stim file's pattern, early; moved by the timebase (blue) they line up. B-E: every pairing of\n"
    "a sync lick with a stim-file lick, for licks the timebase says were recorded early. A lick and its own stim-file entry\n"
    "make a sharp peak; unrelated pairs make the smooth background.\n"
    f"Licks with a stim-file lick within 25 ms. Files with copies (B, C): {nc['matched_recorded']} of {nc['n_displaced_licks']} as recorded, {nc['matched_corrected']} of {nc['n_displaced_licks']} corrected.\n"
    f"Files with no copies (D, E): {npu['matched_recorded']:,} of {npu['n_displaced_licks']:,} as recorded, {npu['matched_corrected']:,} of {npu['n_displaced_licks']:,} corrected. For comparison, licks the timebase\n"
    f"leaves alone: {nc['matched_on_time'] + npu['matched_on_time']:,} of {nc['n_on_time'] + npu['n_on_time']:,}. Licks come in bouts, so some recorded licks land near an unrelated stim lick by chance.\n"
    "B is smeared because these files mix shifts of 0.5 to 6.5 s; D has three sharp peaks because its two files have only three shifts between them (1.0, 2.0, 2.5 s)."), y=0.012)
figs.append(fig)

# --------------------------------------------------------------------------------------- pdf
s = NUM
with PdfPages(OUT) as pdf:
    page_text(pdf, "Is a sync 'replay' a dropped chunk?", [
        "Evidence from recorded data, 2026-11-20. Code: npc_sync 0.3.8. Script: experiments/sync-timebase-drops/make_figures.py.",
        "",
        "# The claim",
        "The sync program loses whole 0.5 s chunks of data without leaving a gap in sample numbers. For each lost chunk it "
        "later writes one extra chunk, an identical copy of the chunk that follows. That run of copies is what we have been "
        "calling a replay.",
        "",
        "If the copy is written straight away, it sits where the lost chunk should have been. That is the 'gap of zero' case: "
        "a chunk of real data is gone and a copy occupies its place. Nothing else moves. This is the replay as we understood it.",
        "",
        "If the copy is written later, every line recorded in between is early by 0.5 s per chunk still owed.",
        "",
        "# Why believe it",
        "1. Three free-running cameras step together, by whole units, at chunk boundaries, and step back one unit per copy (Fig. 1).",
        "2. In files with no copies the step never comes back, and equals the barcode shift (Fig. 2 A, B).",
        "3. Over a whole file the step at each chunk boundary is an integer: 0, +k or -1, with nothing in between (Fig. 2 C, D).",
        "4. Barcodes, which the method does not use, are irregular on the recorded clock and regular on the corrected one (Fig. 3 A).",
        f"5. Chunks lost equal copies written, file by file: {s['fig3_balance']['n_equal']} of {s['fig3_balance']['n_files']} files (Fig. 3 C).",
        "6. Display frames next to a run are off by multiples of 0.42 ms, the signature of a frame recorded one chunk early (Fig. 3 D).",
        "7. Licks, timed independently by the stim PC, are early by whole chunks as recorded and on time when corrected, "
        "in files with copies as well as in files without (Fig. 4).",
        "",
        "# What this does not show",
        "- Why the sync program does this. The pattern is established; the cause in the acquisition code is not.",
        "- For a gap of zero, the file cannot distinguish 'lost, then padded with a copy' from 'overwritten by a copy'. They are "
        "the same bytes. Calling it a drop is a description that also covers the other two thirds, not a separate finding.",
        f"- {s['fig3_balance']['n_files'] - s['fig3_balance']['n_equal']} file(s) where chunks located and copies differ: "
        + "; ".join(f"{a.replace('DRpilot_', '')} ({b_} copies, {c} located)" for a, b_, c in s["fig3_balance"]["unequal"]) + ".",
        "- Figures 1, 2 and 4A show chosen examples. The counts over all 256 surveyed files are in "
        "experiments/sync-timebase-drops/survey_summary.log.",
        "- Fig. 4 covers 9 sessions with a usable lick line and task file, not every file.",
    ])
    for fig in figs:
        pdf.savefig(fig)
        plt.close(fig)
(HERE / "figure_numbers.json").write_text(json.dumps(NUM, indent=1, default=str))
print("wrote", OUT)
