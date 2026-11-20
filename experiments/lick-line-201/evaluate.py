"""#201: current vs prototype-fixed lick-line reconstruction.

A. Real spans (the sessions' dropped/replayed intervals): alternation, edge counts, synthetic
   rising edges without a lickFrames onset.
B. Injected spans (ground truth): runs of fake 'unknown' spans placed in VALID data, so the
   recorded lick edges inside them are known. Both methods reconstruct them; compared with the
   recorded truth.

Usage: python evaluate.py <session_id> [...]. Writes results/<sid>.pkl and results/summary.jsonl
"""
from __future__ import annotations

import copy, json, sys, pathlib, pickle
import numpy as np
import npc_sync
from npc_sessions import timing
from common import load, edges, lick_onsets, LICK
import fix

HERE = pathlib.Path(__file__).parent
RES = HERE / "results"; RES.mkdir(exist_ok=True)
TOL = 1 / 60  # one frame
RNG_SEED = 201


def run(method: str, raw, base, stim_paths):
    p = copy.deepcopy(base)
    if method == "current":
        info = timing.reconstruct_lick_line(raw, p, stim_paths)
        timing.widen_unknown_lick_spans(raw, p)
    else:
        info = fix.reconstruct_lick_line_fixed(raw, p, stim_paths)
        fix.widen_unknown_lick_spans_fixed(raw, p)
    view = npc_sync.SyncDataset(raw, patch=p, on_invalid="mask")
    r, f = edges(view)
    rs = np.asarray(view.get_edge_sources(LICK, "rising"))
    fs = np.asarray(view.get_edge_sources(LICK, "falling"))
    return dict(info=info, patch=p, r=r, f=f, r_src=rs, f_src=fs,
                spans=[(s.span, s.status) for s in p.lines[LICK]])


def nearest(x, ref):
    if not len(ref) or not len(x):
        return np.full(len(x), np.inf)
    ref = np.sort(ref)
    j = np.clip(np.searchsorted(ref, x), 1, max(1, len(ref) - 1))
    return np.minimum(np.abs(ref[j] - x), np.abs(ref[j - 1] - x))


def state_on_grid(r, f, grid):
    """1 where the line is high at each grid time (last edge before it is rising)."""
    t = np.r_[r, f]; k = np.r_[np.ones(len(r)), np.zeros(len(f))]
    o = np.argsort(t, kind="stable"); t, k = t[o], k[o]
    i = np.searchsorted(t, grid, side="right") - 1
    return np.where(i >= 0, k[np.clip(i, 0, None)], 0)


def metrics_real(res, onsets):
    r, f = res["r"], res["f"]
    synth_r = r[res["r_src"] == 1]
    return dict(
        n_rising=len(r), n_falling=len(f), alternation_errors=fix.alternates(r, f),
        n_synthetic_rising=len(synth_r),
        n_synthetic_rising_without_lick_frame=int(np.sum(nearest(synth_r, onsets) > TOL)),
        n_unknown_spans=sum(1 for _, st in res["spans"] if st == "unknown"),
    )


def inject_spans(raw, base, stim_paths, rng, n_runs=40):
    """Runs of 3-8 spans of 0.5 s separated by 0.5-1.5 s gaps, in valid task data away from
    real spans; half start 0-0.3 s after a measured lick onset (mid-bout)."""
    view = npc_sync.SyncDataset(raw, patch=base, on_invalid="mask")
    task = [p for p in stim_paths if "DynamicRouting" in str(p)][0]
    on, (t0, t1) = timing.predict_lick_onsets(view, task)
    real = sorted(s.span for s in base.lines[LICK])
    rr, _ = edges(raw)
    rr = rr[~npc_sync.in_intervals(rr, [(a - 30, z + 30) for a, z in real])]
    rr = rr[(rr > t0 + 10) & (rr < t1 - 30)]
    runs, tries = [], 0
    while len(runs) < n_runs and tries < 10000:
        tries += 1
        start = (rng.choice(rr) + rng.uniform(0, 0.3)) if (len(runs) % 2 == 0 and len(rr)) else rng.uniform(t0 + 10, t1 - 30)
        n = int(rng.integers(3, 9))
        spans, t = [], start
        for _ in range(n):
            spans.append((t, t + 0.5)); t += 0.5 + rng.choice([0.5, 1.0, 1.5])
        lo, hi = spans[0][0] - 3, spans[-1][1] + 3
        if any(a < hi and z > lo for a, z in real) or any(r[0][0] - 3 < hi and r[-1][1] + 3 > lo for r in runs):
            continue
        runs.append(spans)
    return runs


def with_spans(base, spans):
    p = copy.deepcopy(base)
    extra = [npc_sync.SpanPatch(span=(float(a), float(z)), status="unknown", method="mask", reason="injected (#201 test)") for a, z in spans]
    p.lines[LICK] = sorted([*p.lines[LICK], *extra], key=lambda s: s.span)
    return p


def metrics_injected(res, truth_r, truth_f, spans):
    m = npc_sync.in_intervals
    tr, tf = truth_r[m(truth_r, spans)], truth_f[m(truth_f, spans)]
    rr = res["r"][m(res["r"], spans)]
    grid = np.concatenate([np.arange(a, z, 0.001) for a, z in spans])
    agree = np.mean(state_on_grid(res["r"], res["f"], grid) == state_on_grid(truth_r, truth_f, grid))
    # edges outside spans unchanged -> alternation of the full line is a fair check
    return dict(
        n_true_onsets=len(tr), n_reconstructed_onsets=len(rr),
        n_matched=int(np.sum(nearest(rr, tr) <= TOL)),
        n_fake=int(np.sum(nearest(rr, tr) > TOL)),
        n_missed=int(np.sum(nearest(tr, rr) > TOL)),
        state_agreement=float(agree),
        alternation_errors=fix.alternates(res["r"], res["f"]),
    )


def main(sid):
    raw, stim_paths, base = load(sid)
    view0 = npc_sync.SyncDataset(raw, patch=base, on_invalid="mask")
    onsets = lick_onsets(view0, stim_paths)
    rr, ff = edges(raw)
    out = dict(session=sid, raw_rising=len(rr), raw_falling=len(ff), raw_alternation_errors=fix.alternates(rr, ff),
               n_lick_spans=len(base.lines[LICK]))
    store = dict(onsets=onsets, raw_r=rr, raw_f=ff, real_spans=sorted(s.span for s in base.lines[LICK]))
    for method in ("current", "fixed"):
        res = run(method, raw, base, stim_paths)
        out[f"real_{method}"] = metrics_real(res, onsets)
        if method == "fixed":
            out["real_fixed"]["n_spans_lick_in_progress_at_exit_without_onset"] = res["info"].get("n_spans_lick_in_progress_at_exit_without_onset")
        store[f"real_{method}"] = dict(r=res["r"], f=res["f"], r_src=res["r_src"], f_src=res["f_src"], spans=res["spans"], info=res["info"])
    # injected
    rng = np.random.default_rng(RNG_SEED)
    runs = inject_spans(raw, base, stim_paths, rng)
    spans = [s for run_ in runs for s in run_]
    pinj = with_spans(base, spans)
    store["inj_runs"] = runs
    if spans:
        for method in ("current", "fixed"):
            res = run(method, raw, pinj, stim_paths)
            out[f"inj_{method}"] = metrics_injected(res, rr, ff, spans)
            out[f"inj_{method}"]["n_runs"] = len(runs); out[f"inj_{method}"]["n_spans"] = len(spans)
            store[f"inj_{method}"] = dict(r=res["r"], f=res["f"], r_src=res["r_src"], f_src=res["f_src"])
    (RES / f"{sid}.pkl").write_bytes(pickle.dumps(store))
    with open(RES / "summary.jsonl", "a") as fh:
        fh.write(json.dumps(out) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    for sid in sys.argv[1:]:
        main(sid)
