"""Figures + text for outputs/issue-201-lick-line.pdf from results/ (evaluate.py, checks.py)."""
from __future__ import annotations

import json, pathlib, pickle, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = pathlib.Path(__file__).parent
RES = HERE / "results"
OUT = HERE.parents[1] / "outputs" / "issue-201-lick-line.pdf"
summary = {json.loads(l)["session"]: json.loads(l) for l in open(RES / "summary.jsonl")}
checks = json.load(open(RES / "checks.json"))
bounds = json.load(open(RES / "bound_distances.json")) if (RES / "bound_distances.json").exists() else None
D = {s: pickle.loads((RES / f"{s}.pkl").read_bytes()) for s in summary}

C_RAW, C_LF, C_CUR, C_FIX = "0.15", "tab:green", "tab:red", "tab:blue"


def page_text(pdf, title, body, size=9.5):
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.07, 0.95, title, fontsize=15, weight="bold", va="top")
    y = 0.91
    for para in body:
        mono = para.startswith("    ")
        lines = para.splitlines() if mono else textwrap.wrap(para, 92) or [""]
        for ln in lines:
            fig.text(0.07, y, ln, fontsize=size - (1 if mono else 0), va="top",
                     family="monospace" if mono else "sans-serif")
            y -= 0.0175 if not mono else 0.0155
        y -= 0.008
    pdf.savefig(fig); plt.close(fig)


def step(ax, r, f, t0, t1, y0, color, label, lw=1.2):
    t = np.r_[r, f]; k = np.r_[np.ones(len(r)), np.zeros(len(f))]
    m = (t >= t0) & (t <= t1); o = np.argsort(t[m], kind="stable"); t, k = t[m][o], k[m][o]
    before_r = r[r < t0]; before_f = f[f < t0]
    s0 = float((before_r[-1] if len(before_r) else -np.inf) > (before_f[-1] if len(before_f) else -np.inf))
    xs = np.r_[t0, t, t1]; ys = np.r_[s0, k, k[-1] if len(k) else s0]
    ax.step(xs, y0 + 0.7 * ys, where="post", color=color, lw=lw, label=label)


def edge_marks(ax, r, f, src_r, src_f, t0, t1, y0, color):
    for t, s, mk in [(r, src_r, "^"), (f, src_f, "v")]:
        m = (t >= t0) & (t <= t1) & (s == 1)
        ax.plot(t[m], np.full(m.sum(), y0 + 0.85), mk, color=color, ms=4, ls="none")


def panel(ax, sid, t0, t1, title, shade=True, annotate=None):
    d = D[sid]
    for a, z in d["real_spans"]:
        if z > t0 and a < t1 and shade:
            ax.axvspan(a, z, color="0.85", lw=0, zorder=0)
    rr, ff = d["raw_r"], d["raw_f"]
    step(ax, rr, ff, t0, t1, 3, C_RAW, "recorded (true clock, unpatched)")
    on = d["onsets"]; m = (on >= t0) & (on <= t1)
    ax.vlines(on[m], 2.05, 2.75, color=C_LF, lw=1.2, label="lickFrames onsets (stim file)")
    c, f_ = d["real_current"], d["real_fixed"]
    step(ax, c["r"], c["f"], t0, t1, 1, C_CUR, "patched: current (HEAD, builder 8)")
    edge_marks(ax, c["r"], c["f"], c["r_src"], c["f_src"], t0, t1, 1, C_CUR)
    step(ax, f_["r"], f_["f"], t0, t1, 0, C_FIX, "patched: proposed fix")
    edge_marks(ax, f_["r"], f_["f"], f_["r_src"], f_["f_src"], t0, t1, 0, C_FIX)
    ax.set_yticks([0.35, 1.35, 2.4, 3.35]); ax.set_yticklabels(["fixed", "current", "lickFrames", "recorded"])
    ax.set_xlim(t0, t1); ax.set_ylim(-0.2, 4.0); ax.set_title(title, fontsize=9.5, loc="left")
    ax.set_xlabel("time (s, true clock)")
    if annotate:
        annotate(ax)


with PdfPages(OUT) as pdf:
    s14, s08 = summary["714748_2024-06-28"], summary["708020_2024-05-17"]
    page_text(pdf, "npc_sessions #201: is it still an issue?", [
        "Answer: YES. reconstruct_lick_line is unchanged since 9c0279d (builder 4). Commits after the issue was "
        "filed (builders 6-8, up to HEAD 87614d8 / v0.0.296) only touched barcode spans. Re-running the current "
        "code on fresh patches reproduces the issue's numbers exactly:",
        f"    714748_2024-06-28  patched R/F {s14['real_current']['n_rising']}/{s14['real_current']['n_falling']}  "
        f"alternation errors {s14['real_current']['alternation_errors']}  (raw {s14['raw_rising']}/{s14['raw_falling']}, 0)\n"
        f"    708020_2024-05-17  patched R/F {s08['real_current']['n_rising']}/{s08['real_current']['n_falling']}  "
        f"alternation errors {s08['real_current']['alternation_errors']}  (raw {s08['raw_rising']}/{s08['raw_falling']}, 0)",
        f"Synthetic rising edges with no lickFrames onset within 1 frame (fake licks): 714748 "
        f"{s14['real_current']['n_synthetic_rising_without_lick_frame']}, 708020 {s08['real_current']['n_synthetic_rising_without_lick_frame']}.",
        "Root cause (confirmed): for each span, the lick state at entry/exit is inferred from the nearest MEASURED lick "
        "edges outside all spans. In a run of spans with no lick edges between them, every span sees the same "
        "'last edge before' (from before the run) and the same 'first edge after' (after the run). So each span repeats "
        "the same synthetic edge. The issue's second point, the edge at a span bound, sits at +0.5 sample INSIDE the "
        "span [a, z), so SyncPatch.apply removes it from the output anyway. It's only a problem because "
        "_measured_outside uses _mask_intervals (1.5 samples early), which differs from what apply() keeps.",
        "Fix (now in src/ as BUILDER 9; its output is identical to the prototype experiments/lick-line-201/fix.py on all 8 "
        "sessions, results/verify_src.json): read the lick state at "
        "each span bound directly from the sync data words. Every event on any line carries the lick bit, and "
        "gaps between spans (>= 0.5 s here) contain vsync/camera events every few ms. The nearest events used are "
        "at most 6.2 ms from a bound in both affected sessions. Fall back to carrying the previous span's exit "
        "state only if no event separates two spans. Use the same [a, z) as SyncPatch.apply for measured edges.",
        "Results of the prototype: on the real spans, the patched lick line alternates in every session tested (0 "
        "errors), and 0 synthetic onsets lack a lickFrames onset (except 1 legitimate lick in progress, 703880). "
        "On injected spans with known ground truth (40 runs of 3-8 spans per session, in valid data), current "
        "code: 25-133 alternation errors per session and 0-14 spurious onsets. Fix: 0 alternation errors, 0 "
        "spurious onsets, and higher line-state agreement in every session.",
        "Side finding (separate bug, not #201): in 644867_2023-02-21 (taskVersion 'stage 5 AMN ori moving'), "
        "lickFrames lists every frame the lick input is high (82% consecutive frames, 12208 entries vs 2275 sync "
        "licks). predict_lick_onsets treats each entry as an onset. Its real spans got no lick reconstruction, so "
        "no output is affected yet, but any lick span in such a session would get ~5x too many licks. Proposed: "
        "collapse consecutive lickFrames to their first frame.",
        "Pages: 2-4 examples; 5 summary on real spans; 6 injected-span validation; 7 bound-state evidence; "
        "8-9 proposed code changes and tests; 10 provenance.",
    ])

    # --- example: fake onsets 714748
    fig, axs = plt.subplots(2, 1, figsize=(11, 8.5))
    panel(axs[0], "714748_2024-06-28", 5903.5, 5913, "714748_2024-06-28: fake onsets at span end - duration in every span after the bout "
          "(grey = dropped/replayed spans; triangles = synthetic edges)")
    axs[0].legend(loc="upper right", fontsize=7, ncol=2)
    panel(axs[1], "714748_2024-06-28", 5900, 6142, "same run, 5900-6142 s: 80 spans, 0 measured lick edges and 0 lickFrames onsets until ~6136 s. "
          "Current: 82 fake licks; fix: none", shade=True)
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # --- example: falling edges at span starts 714748 + 708020
    fig, axs = plt.subplots(3, 1, figsize=(11, 8.5))
    def ann(ax):
        ax.annotate("recorded F at 3241.49999\n(+0.5 sample inside the span:\nremoved by SyncPatch.apply)", xy=(3241.49999, 3.0),
                    xytext=(3243.0, 3.3), fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.6))
    panel(axs[0], "714748_2024-06-28", 3240.5, 3265, "714748_2024-06-28: line low after 3241.5 s. Current: a falling edge at every later span start (F-F-F...); fix: one F, first span", annotate=ann)
    axs[0].legend(loc="upper right", fontsize=7, ncol=2)
    panel(axs[1], "708020_2024-05-17", 2846.5, 2858.5, "708020_2024-05-17: fake onsets at x.9218 / x.4218 in 4 spans (current); none (fix)")
    panel(axs[2], "708020_2024-05-17", 1665.5, 1674, "708020_2024-05-17: falling edges at span starts 1667, 1669, 1670, 1672 (current); one F in the first span (fix)")
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # --- summary on real spans
    sids = list(summary)
    fig, axs = plt.subplots(1, 3, figsize=(11, 4.6))
    x = np.arange(len(sids)); w = 0.38
    for ax, key, ttl in [(axs[0], "alternation_errors", "lick-line alternation errors\n(adjacent same-kind edges)"),
                         (axs[1], "n_synthetic_rising_without_lick_frame", "synthetic onsets with no\nlickFrames onset within 1 frame")]:
        cv = [summary[s]["real_current"][key] for s in sids]; fv = [summary[s]["real_fixed"][key] for s in sids]
        ax.bar(x - w / 2, cv, w, color=C_CUR, label="current"); ax.bar(x + w / 2, fv, w, color=C_FIX, label="fix")
        for i, (a, b) in enumerate(zip(cv, fv)):
            ax.text(i - w / 2, a, str(a), ha="center", va="bottom", fontsize=7); ax.text(i + w / 2, b, str(b), ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(sids, rotation=60, ha="right", fontsize=7); ax.set_title(ttl, fontsize=9)
    axs[0].legend(fontsize=8)
    cv = [summary[s]["real_current"]["n_rising"] - summary[s]["real_current"]["n_falling"] for s in sids]
    fv = [summary[s]["real_fixed"]["n_rising"] - summary[s]["real_fixed"]["n_falling"] for s in sids]
    axs[2].bar(x - w / 2, cv, w, color=C_CUR); axs[2].bar(x + w / 2, fv, w, color=C_FIX)
    for i, (a, b) in enumerate(zip(cv, fv)):
        axs[2].text(i - w / 2, a, str(a), ha="center", va="bottom", fontsize=7); axs[2].text(i + w / 2, b, str(b), ha="center", va="bottom", fontsize=7)
    axs[2].set_xticks(x); axs[2].set_xticklabels(sids, rotation=60, ha="right", fontsize=7); axs[2].set_title("n rising - n falling (patched line)", fontsize=9)
    fig.suptitle("Real dropped/replayed spans: the 5 sessions in #201 + 3 npc_lims sessions with >= 2 replays (raw line alternates in all)", fontsize=10)
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # --- injected validation
    fig, axs = plt.subplots(1, 3, figsize=(11, 4.8))
    sids_i = [s for s in sids if "inj_current" in summary[s]]
    x = np.arange(len(sids_i))
    def spurious(s, m):
        return checks[s]["injected_fake_kinds"][m].get("no lick (spurious)", 0)
    data = [("alternation_errors", "alternation errors (full line)", lambda s, m: summary[s][f"inj_{m}"]["alternation_errors"]),
            ("spurious", "synthetic onsets where the recorded line\nhad no lick (spurious licks)", spurious),
            ("state", "line-state agreement inside spans\n(1 ms grid, vs recorded)", lambda s, m: summary[s][f"inj_{m}"]["state_agreement"])]
    for ax, (_, ttl, fn) in zip(axs, data):
        cv = [fn(s, "current") for s in sids_i]; fv = [fn(s, "fixed") for s in sids_i]
        ax.bar(x - w / 2, cv, w, color=C_CUR, label="current"); ax.bar(x + w / 2, fv, w, color=C_FIX, label="fix")
        for i, (a, b) in enumerate(zip(cv, fv)):
            fmt = (lambda v: f"{v:.3f}") if isinstance(a, float) else str
            ax.text(i - w / 2, a, fmt(a), ha="center", va="bottom", fontsize=6, rotation=90 if isinstance(a, float) else 0)
            ax.text(i + w / 2, b, fmt(b), ha="center", va="bottom", fontsize=6, rotation=90 if isinstance(a, float) else 0)
        ax.set_xticks(x); ax.set_xticklabels(sids_i, rotation=60, ha="right", fontsize=7); ax.set_title(ttl, fontsize=9)
    axs[2].set_ylim(0.8, 1.02); axs[0].legend(fontsize=8)
    fig.suptitle("Injected spans with ground truth: 40 runs/session of 3-8 x 0.5 s 'unknown' lick spans in VALID data (gaps 0.5-1.5 s, half starting mid-bout), seed 201",
                 fontsize=9)
    fig.text(0.01, 0.01, "Agreement < 1 for both methods: synthetic offsets use the median duration; onsets are +-1/2 frame. "
             "644867: lickFrames = every high frame "
             "(side finding), so both methods add ~446 onsets there; not counted as 'spurious'.", fontsize=7)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95)); pdf.savefig(fig); plt.close(fig)

    # --- bound-state evidence
    if bounds:
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
        for ax, sid in zip(axs, ["714748_2024-06-28", "708020_2024-05-17"]):
            b = bounds[sid]
            ax.hist(np.array(b["before_ms"]), bins=40, alpha=0.6, label="last event before span start")
            ax.hist(np.array(b["after_ms"]), bins=40, alpha=0.6, label="first event at/after span end")
            ax.set_xlabel("distance from span bound (ms)"); ax.set_ylabel("n spans")
            ax.set_title(f"{sid}: {b['n_spans']} spans; max {max(max(b['before_ms']), max(b['after_ms'])):.1f} ms\n"
                         f"bound edges at +0.5 sample inside [a, z): {b['n_edges_inside_at_start']} (start), {b['n_edges_at_end']} (end)", fontsize=8.5)
            ax.legend(fontsize=7)
        fig.suptitle("Lick state at each span bound is OBSERVED: the nearest sync event on any line (its data word holds the lick bit) is a few ms away", fontsize=9.5)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    page_text(pdf, "Fixes (implemented in builder 9)", [
        "Fix 1: lick state at span bounds from the sync data words. In timing.py, replace the per-span inference in reconstruct_lick_line (and widen_unknown_lick_spans) with a "
        "direct read. Prototype: experiments/lick-line-201/fix.py::line_state_at_bounds.",
        "    t   = raw.get_all_times('seconds'); bit = raw.get_bit(raw._line_to_bit(LICK_LINE))\n"
        "    chg = np.ediff1d(bit, to_begin=0) != 0\n"
        "    i = searchsorted(t, a, 'left') - 1      # last event with t < a  (what SyncPatch.apply keeps)\n"
        "    j = searchsorted(t, z, 'left')          # first event with t >= z\n"
        "    high_at_entry = bool(bit[i])\n"
        "    high_at_exit  = bool(1 - bit[j]) if chg[j] else bool(bit[j])   # state just before that event\n"
        "    # only if t[i] is before the previous span's end / t[j] after the next span's start:\n"
        "    #   entry = previous span's exit state (carried), exit = low",
        "Why this, not 'merge consecutive spans into groups' (the issue's suggestion): grouping still has to guess "
        "where in a group a lick in progress starts or ends. The data words give the exact state in every gap. In "
        "714748's 80-span run, the fix adds no edge until lickFrames onsets resume at ~6136.9 s, and none of its "
        "synthetic onsets lacks a lickFrames onset.",
        "Fix 2: one definition of 'measured edges outside the spans'. Use the edges outside "
        "[a, z) of the lick spans (SyncPatch.apply's rule) for last_rising_before and the median duration. Don't use "
        "_mask_intervals (1.5 samples early, 0.5 late on a timebase). Otherwise the state can be computed from a "
        "different edge set than the one that ends up in the output. The 3241.49999 F edge is at +0.5 sample inside "
        "the span, so it's removed from the output. The fix synthesizes the line's fall inside the span from the "
        "state at the bounds. Don't 'keep' the bound edge as the issue suggests: npc_sync documents that the event "
        "at a drop/copy boundary can carry the state of the data after it, so its time isn't trustworthy.",
        "Fix 3: carry synthesized edges forward. When computing last_rising_before for span "
        "k, include edges synthesized for spans < k. This only matters if no event separates the spans, but it keeps "
        "the falling-edge time right when the line is high on entry.",
        "Fix 4 (a guard): after reconstruction, check that the patched lick line alternates. "
        "Count synthetic rising edges without a lickFrames onset within +-1 frame. Each must belong to a span with "
        "a lick in progress at exit and no onsets. Record both counts in the diagnostics. If the line doesn't "
        "alternate, leave the spans 'unknown' and widen them (never ship fake licks). Builder 9 records "
        "both counts in the diagnostics and implements the 'unknown' fallback (patch.notes gets a message).",
        "BUILDER bumped to 9 (lick content changes; affects 714748_2024-06-28 and 708020_2024-05-17, and in principle "
        "any session with >= 2 lick spans).",
    ])
    page_text(pdf, "Tests to add, and other findings", [
        "Unit tests added in tests/test_timing.py (a, b, d, e fail on builder 8; c passes on both):",
        "    a) >= 3 spans, events of another line in the gaps, lick low before, lick high in the 50 ms after the run\n"
        "       -> exactly 1 synthetic rising edge (last span), line alternates\n"
        "    b) lick high before the run, low after -> exactly 1 synthetic falling edge (first span)\n"
        "    c) lick low throughout -> 0 synthetic edges\n"
        "    d) a recorded falling edge at a + 0.5 sample (inside [a, z)) after a rising edge before a\n"
        "       -> output has R (measured), F (synthetic, in span), and alternates\n"
        "    e) two spans with no event between them -> state carried, line alternates",
        "Regression on real data (slow/S3 marker): for 714748_2024-06-28 and 708020_2024-05-17, the patched lick line "
        "alternates, and synthetic onsets without a lickFrames onset <= spans flagged 'lick in progress at exit, no "
        "onsets' (0 in both now). The injected-span harness (experiments/lick-line-201/evaluate.py) is a "
        "ground-truth check that could become a test.",
        "Side finding A (new, separate issue): lickFrames semantics differ by task version. In "
        "644867_2023-02-21, 82% of lickFrames are consecutive frames: every frame with the lick input high. "
        "predict_lick_onsets should collapse runs of consecutive frames to the first frame. Also check the date or "
        "taskVersion where this changed. In 708020/714748 (2024), 0% are consecutive.",
        "Side finding B (from the issue, not investigated here): 714748_2024-06-28 reward_1_actuator fails "
        "len(rising)==len(falling) in _reward_frame_times. That's possibly the same class of bug (per-span state "
        "from measured edges only) on the reward line. Status: unverified.",
        "Limits: the injected spans aren't at buffer seams, so they test fixes 1 and 3 but not the bound-edge artifact "
        "(fix 2). That's covered by the real 714748/708020 spans (7 + 6 bound edges, all at +0.5 sample inside the span). "
        "Sessions not tracked by npc_lims (e.g. 703880_2024-04-12, 702131_2024-02-23, with 122 and 73 replays) couldn't "
        "be loaded as sessions with sync, so they weren't tested.",
    ])
    page_text(pdf, "Provenance", [
        "Code under test: npc_sessions HEAD 87614d8 (v0.0.296, BUILDER 'npc_sessions.timing 8'), npc_sync 0.3.9 "
        "from .venv. Fresh base patches from npc_sync.build_sync_patch(true_clock_view(sync), get_stim_blocks(...)), "
        "cached in experiments/lick-line-201/cache/. The lick line was reconstructed by timing.reconstruct_lick_line + "
        "widen_unknown_lick_spans (current) or fix.py (proposed). Sound/opto steps not run (they don't touch the lick line).",
        "    experiments/lick-line-201/diagnose.py      reproduces the issue numbers on HEAD\n"
        "    experiments/lick-line-201/inspect_runs.py  span groups, bound edges (+0.5 sample)\n"
        "    experiments/lick-line-201/bound_states.py  data-word states; max 6.2 ms to nearest event\n"
        "    experiments/lick-line-201/fix.py           prototype fix\n"
        "    experiments/lick-line-201/evaluate.py      real + injected-span metrics -> results/summary.jsonl\n"
        "    experiments/lick-line-201/checks.py        classification of remaining onsets -> results/checks.json\n"
        "    experiments/lick-line-201/scan_cache.py    shared S3 patch cache: 54/65 still builder 3\n"
        "    experiments/lick-line-201/make_pdf.py      this document",
        "Session choice: the 5 sessions in the issue, plus npc_lims sessions with >= 2 replayed intervals in "
        "experiments/sync-patch-manifest/manifest.csv (644867_2023-02-21, 837959_2026-04-02, 798632_2025-08-29).",
    ])
print(OUT)
