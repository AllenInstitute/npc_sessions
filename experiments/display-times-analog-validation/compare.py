"""Compare baseline and branch (fixE) display times with NI-DAQ analog photodiode transitions.

    C:/data/t2p_qc/venv_analog/Scripts/python compare.py <folder> [<folder> ...]

Inputs: C:/data/t2p_qc/analog/<folder>/{sync.npz, analog.npz, analog.json} (prep_sync.py, fetch_analog.py),
display times C:/data/t2p_qc/out/baseline/<folder>.npz and C:/data/t2p_qc/out_issues/fixE/<folder>.npz.

Ground truth, independent of the estimator: the k-th analog transition in a block is the box toggle of
frame (k + o) * N, where N = diodeBoxFrameInterval. The offset o is chosen once per segment so that the
transition is 15 ms to 15 ms + N frames after that frame's vsync *rising* edge (frame signal raised
before draw + flip). Segments are split where the implied latency jumps by about N frames
(an extra or missing analog transition). Then, per toggle frame f:
    r = analog(f) - (display(f) - MONITOR_CENTER_REFRESH_TIME)
which is the analog -> sync-TTL comparator lag if the flip is assigned to the right frame. The lag is
taken per polarity from the median over all assigned toggles of the session (branch estimates), and
err = r - lag(polarity). A misassignment by j frames shows up as err ~ j * 16.7 ms.
Writes results/<folder>.csv (one row per block) and results/<folder>_frames.npz.
"""
from __future__ import annotations
import json, pathlib, sys
import numpy as np

HERE = pathlib.Path(__file__).parent
A_DIR = pathlib.Path("C:/data/t2p_qc/analog")
BASE = pathlib.Path("C:/data/t2p_qc/out/baseline")
FIX = pathlib.Path("C:/data/t2p_qc/out_issues/fixE")
FR = 1 / 59.95
MCRT = 0.008  # MONITOR_CENTER_REFRESH_TIME
MIN_SEP = 0.015  # MIN_VSYNC_DIODE_FLIP_SEPARATION_SEC
LAG_LO, LAG_HI = -0.005, 0.012
LF_PRIOR = 0.0205  # analog - vsync falling edge prior (AVERAGE_VSYNC_TO_DIODE_FLIP_TIME + ~1 ms); fallback only


def nearest(sorted_arr, x):
    j = np.clip(np.searchsorted(sorted_arr, x), 1, len(sorted_arr) - 1)
    left, right = sorted_arr[j - 1], sorted_arr[j]
    return np.where(np.abs(x - left) <= np.abs(x - right), left, right)


def assign(a, vr, vf, toggles, N, search=10, win=5000):
    """Ordinal assignment of analog transitions a (sorted) to toggle indices with a piecewise-constant
    offset o (transition k <-> toggle k + o), independent of the estimator.

    - Offset: chosen by landmarks. A dropped refresh makes both the analog interval and the vsync
      falling-edge interval of the same toggle group longer than N + 0.5 frames (the falling edge follows
      flip(); the rising edge of the *next* frame is the one that is late, so rising-edge intervals are
      shifted by one frame). Within the next
      `win` transitions, the offset (searched +/- `search` around the previous one) that maximises
      matched landmarks wins; the latency (analog - vsync rising edge of the toggle frame) breaks ties.
      Without any landmark, the latency window LF_PRIOR +/- N/2 frames (from the falling edge) is used.
    - Segments: after the first offset, the block's median latency L0 (from the falling edge) is measured; a segment ends where
      20 of 25 consecutive transitions are outside [L0 - N/2 frames - 5 ms, L0 + N/2 frames + 3 frames).
    Returns (toggle index per transition, -1 = unassigned; segments [(start, stop, offset, n_landmarks_matched,
    method)]; L0)."""
    vt = vf[toggles]  # latency is measured from the falling edge (flip() returned)
    n_t = len(vt)
    lm_v = np.r_[False, np.diff(vf[toggles]) > (N + 0.5) * FR]  # vsync group ending at toggle m is long
    lm_a = np.r_[False, np.diff(a) > (N + 0.5) * FR]

    def lat(k, o):
        m = k + o
        ok = (m >= 0) & (m < n_t)
        out = np.full(len(k), np.nan)
        out[ok] = a[k[ok]] - vt[m[ok]]
        return out

    def choose(k0, prev, window):
        k = np.arange(k0, min(k0 + win, len(a)))
        kl = k[lm_a[k]]
        def lm_score(o):
            m = kl + o
            ok = (m >= 0) & (m < n_t)
            return int(np.sum(lm_v[m[ok]]))
        def lat_score(o):
            L = lat(k, o)
            return np.nanmean((L >= window[0]) & (L < window[1])) if np.any(~np.isnan(L)) else -1
        cands = range(prev - search, prev + search + 1)
        if len(kl):
            best = max(cands, key=lambda o: (lm_score(o), lat_score(o)))
            return best, lm_score(best), "landmarks"
        best = max(cands, key=lat_score)
        return best, 0, "latency"

    idx = np.full(len(a), -1)
    segs = []
    k0, prev = 0, 0
    window = (LF_PRIOR - 0.5 * N * FR, LF_PRIOR + 0.5 * N * FR)
    L0 = None
    while k0 < len(a):
        best, n_lm, how = choose(k0, prev, window)
        if L0 is None:
            L0 = float(np.nanmedian(lat(np.arange(len(a)), best)))
            window = (L0 - 0.5 * N * FR, L0 + 0.5 * N * FR)
        k = np.arange(k0, min(k0 + 50000, len(a)))
        L = lat(k, best)
        good = (L >= window[0] - 0.005) & (L < window[1] + 3 * FR)
        bad_run = np.convolve(~good, np.ones(25), mode="full")[24:] >= 20
        bad_run[:1] = False
        stop = k0 + (int(np.argmax(bad_run)) if bad_run.any() else len(k))
        if segs and best == prev and stop - k0 < 25:
            # same offset still best: the implausible run isn't an offset change (e.g. a stall); keep the
            # offset through it, up to where 25 consecutive transitions are plausible again
            ok_run = np.convolve(good, np.ones(25), mode="full")[24:] >= 25
            stop = k0 + (int(np.argmax(ok_run)) if ok_run.any() else len(k))
        stop = max(stop, k0 + 1)
        m = np.arange(k0, stop) + best
        valid = (m >= 0) & (m < n_t)
        idx[np.arange(k0, stop)[valid]] = m[valid]
        if segs and segs[-1][2] == best and segs[-1][1] == k0:
            segs[-1] = (segs[-1][0], int(stop), best, segs[-1][3], segs[-1][4])
        else:
            segs.append((int(k0), int(stop), int(best), int(n_lm), how))
        k0, prev = stop, best
    return idx, segs, L0


def assign_dp(a, vr, vf, toggles, Lf=0.019, Lr=None, B=6, s_skip=12.0, cap=30.0, pol=None):
    """As _assign_dp, with the polarity constraint: the box alternates black/white at every toggle, so
    toggle i must be matched to a transition of polarity p0 * (-1)**i. Both p0 are tried; the cheaper
    alignment is returned. Without `pol`, polarity is ignored."""
    if pol is None:
        return _assign_dp(a, vr, vf, toggles, Lf, Lr, B, s_skip, cap)[:2]
    best = None
    for p0 in (1, -1):
        exp = p0 * (-1) ** np.arange(len(toggles))
        r = _assign_dp(a, vr, vf, toggles, Lf, Lr, B, s_skip, cap, pol=pol, exp_pol=exp)
        if best is None or r[2] < best[2]:
            best = r
    return best[:2]


def _assign_dp(a, vr, vf, toggles, Lf=0.019, Lr=None, B=6, s_skip=12.0, cap=30.0, pol=None, exp_pol=None):
    """Monotone alignment of analog transitions a (sorted) to toggle frames, independent of the
    estimator and of the TTL diode line. Match cost (ms) = min(|a - vf[f] - Lf|, |a - vr[f] - Lr|),
    capped at `cap`: the display follows the falling edge (flip() returned) by ~Lf, unless the falling
    edge was delayed by a stall after flip(), in which case it follows the rising edge by ~Lr.
    Unmatched frames and unmatched transitions (e.g. screen flicker during a stall) cost `s_skip` each.
    Band: transitions within +/- B (plus any gap) of the one expected from the falling edge.
    Returns frame index per transition (-1 = unmatched) and the per-frame matched transition (-1)."""
    tf, tr_ = vf[toggles], vr[toggles]
    if Lr is None:
        Lr = Lf + float(np.median(vf - vr))
    n_f, n_a = len(tf), len(a)
    jexp = np.searchsorted(a, tf + Lf)
    INF = 1e18
    rows = []  # (lo, D row, choice row) ; choice 0=match (diag), 1=skip frame (up), 2=skip transition (left)
    lo_prev, D_prev = 0, np.array([0.0])  # row 0: j = 0 only reachable with 0 cost... allow leading skips:
    hi0 = min(n_a, max(jexp[0] + B, 0))
    D_prev = np.arange(hi0 + 1) * s_skip
    rows.append((0, D_prev.copy(), np.full(hi0 + 1, 2, np.int8)))
    hi_prev = hi0
    for i in range(1, n_f + 1):
        lo = max(0, min(jexp[i - 1] - B, hi_prev))
        # the row must reach the transitions just before the next frame's (to skip stall flicker)
        hi = min(n_a, max(jexp[min(i, n_f - 1)] + B, jexp[i - 1] + B, lo))
        if i == n_f:
            hi = n_a
        j = np.arange(lo, hi + 1)
        # up: D[i-1][j] + s_skip
        up = np.full(len(j), INF)
        m = (j >= lo_prev) & (j <= hi_prev)
        up[m] = D_prev[j[m] - lo_prev] + s_skip
        # diag: D[i-1][j-1] + c(i-1, j-1)
        dg = np.full(len(j), INF)
        jm = j - 1
        m2 = (jm >= lo_prev) & (jm <= hi_prev) & (jm >= 0)
        if m2.any():
            x = a[jm[m2]]
            c = np.minimum(np.abs(x - tf[i - 1] - Lf), np.abs(x - tr_[i - 1] - Lr)) * 1e3
            c = np.minimum(c, cap)
            if pol is not None:
                c = np.where(pol[jm[m2]] == exp_pol[i - 1], c, INF)
            dg[m2] = D_prev[jm[m2] - lo_prev] + c
        X = np.minimum(up, dg)
        ch = np.where(dg <= up, 0, 1).astype(np.int8)
        # left: D[i][j] = min(X[j], D[i][j-1] + s_skip) -> cumulative
        Y = X - np.arange(len(j)) * s_skip
        cm = np.minimum.accumulate(Y)
        D = cm + np.arange(len(j)) * s_skip
        ch = np.where(D < X - 1e-9, 2, ch).astype(np.int8)
        rows.append((lo, D, ch))
        lo_prev, hi_prev, D_prev = lo, hi, D
    # backtrack from (n_f, n_a)
    f_of = np.full(n_a, -1)
    match_of_frame = np.full(n_f, -1)
    i, jj = n_f, n_a
    while i > 0 or jj > 0:
        lo, D, ch = rows[i]
        if i == 0:
            jj -= 1; continue
        c = ch[jj - lo]
        if c == 0:
            f_of[jj - 1] = toggles[i - 1]; match_of_frame[i - 1] = jj - 1; i -= 1; jj -= 1
        elif c == 1:
            i -= 1
        else:
            jj -= 1
    return f_of, match_of_frame, float(rows[n_f][1][n_a - rows[n_f][0]])


def load_times(path, block):
    if not path.exists():
        return None
    z = np.load(path)
    k = f"A_block{block}"
    return z[k] - MCRT if k in z else None


def main(folder):
    d = A_DIR / folder
    S = np.load(d / "sync.npz")
    meta = json.loads((d / "analog.json").read_text())
    if meta.get("status") != "ok":
        print(folder, meta.get("status")); return [dict(folder=folder, block=-1, status=meta.get("status"))]
    An = np.load(d / "analog.npz")
    fixj = json.loads((FIX / f"{folder}.json").read_text())
    basej = json.loads((BASE / f"{folder}.json").read_text())
    v_rise_all = S["v_rise"]
    dig = np.sort(np.r_[S["d_rise"], S["d_fall"]])
    rows, frames = [], {}
    n_blocks = len([k for k in S.files if k.startswith("block")])
    per_block = []
    for bi in range(n_blocks):
        v = S[f"block{bi}"]
        N = fixj["box"][bi] if fixj.get("box") and fixj["box"][bi] else None
        vr = v_rise_all[np.clip(np.searchsorted(v_rise_all, v) - 1, 0, None)]  # rising edge preceding each falling edge
        if f"b{bi}_t" not in An.files or N is None:
            rows.append(dict(folder=folder, block=bi, status="no analog for block" if N else "no box interval (no stim file)")); continue
        a_all, pol_all = An[f"b{bi}_t"], An[f"b{bi}_pol"]
        sel = (a_all > vr[0] + 0.005) & (a_all < v[-1] + 0.25)
        a, pol = a_all[sel], pol_all[sel]
        toggles = np.arange(0, len(v), N)
        # ground truth: monotone alignment of analog transitions to toggle frames from vsync edges only
        # (two passes: the second uses the block's own median latencies)
        f_dp, _ = assign_dp(a, vr, v, toggles, pol=pol)
        mt = f_dp >= 0
        Lf = float(np.median(a[mt] - v[f_dp[mt]])); Lr = float(np.median(a[mt] - vr[f_dp[mt]]))
        f_dp, match_of_frame = assign_dp(a, vr, v, toggles, Lf=Lf, Lr=Lr, pol=pol)
        mt = f_dp >= 0
        # cross-check: ordinal assignment (landmark-chosen offset) on the matched transitions only
        idx_o, segs, L0 = assign(a[mt], vr, v, toggles, N)
        f_ord = np.where(idx_o >= 0, toggles[np.clip(idx_o, 0, None)], -1)
        n_dp_vs_ordinal = int(np.sum(f_ord != f_dp[mt]))
        a_raw = a
        unmatched = ~mt
        a, pol, f_of = a[mt], pol[mt], f_dp[mt]
        assigned = np.ones(len(a), bool)
        # TTL vs analog (all analog transitions), independent of the estimator
        dw = dig[(dig > vr[0] + 0.005) & (dig < v[-1] + 0.25)]
        lag_a = (nearest(dw, a) - a) if len(dw) > 1 else np.full(len(a), np.nan)
        lag_raw = (nearest(dw, a_raw) - a_raw) if len(dw) > 1 else np.full(len(a_raw), np.nan)
        ok_raw = (lag_raw > LAG_LO) & (lag_raw < LAG_HI)
        lag_d = dw - nearest(a_raw, dw) if len(a_raw) > 1 else np.full(len(dw), np.nan)
        ok_d = (lag_d > LAG_LO) & (lag_d < LAG_HI)
        ai = np.diff(a)
        bins_valid = An[f"b{bi}_bins_valid"]; bins_t = An[f"b{bi}_bins_t"]
        inblk = (bins_t > v[0]) & (bins_t < v[-1])
        pulse = (v - vr) * 1e3
        row = dict(folder=folder, block=bi, stim=fixj["stim_paths"][bi], box=N, n_vsync=len(v),
                   n_toggles=len(toggles), n_analog_raw=int(len(a_raw)), n_assigned=int(mt.sum()),
                   n_analog_unmatched=int(unmatched.sum()), n_toggles_unmatched=int(np.sum(match_of_frame < 0)),
                   first_toggle_unmatched=bool(match_of_frame[0] < 0),
                   n_unmatched_analog_with_ttl=int(np.sum(unmatched & ok_raw)),
                   n_dp_vs_ordinal=n_dp_vs_ordinal, segments=json.dumps(segs), n_segments=len(segs),
                   first_offset=segs[0][2] if segs else None,
                   lat_fall_median_ms=Lf * 1e3, lat_rise_median_ms=Lr * 1e3,
                   pulse_median_ms=float(np.median(pulse)),
                   n_signal_lost_bins=int((~bins_valid[inblk]).sum()),
                   n_analog_short_intervals=int(np.sum(np.diff(a_raw) < 0.5 * FR)),
                   ttl_n=len(dw), ttl_unmatched=int((~ok_d).sum()),
                   source_base=basej["A_sources"][bi] if bi < len(basej.get("A_sources", [])) else None,
                   source_fix=fixj["sources"][bi], reason_fix=fixj["reasons"][bi])
        # analog latency relative to the toggle frame's own vsync edges (calibration)
        lat_r = (a - vr[f_of]) * 1e3; lat_f = (a - v[f_of]) * 1e3
        row.update(lat_rise_min_ms=float(lat_r.min()), lat_rise_p01_ms=float(np.percentile(lat_r, 0.1)),
                   lat_fall_p01_ms=float(np.percentile(lat_f, 0.1)), lat_fall_p999_ms=float(np.percentile(lat_f, 99.9)))
        lm_an = set(f_of[1:][ai > (N + 0.5) * FR].tolist())
        row["n_landmarks_analog"] = len(lm_an)
        fr_rec = dict(a=a, pol=pol, f=f_of, lag_ttl=lag_a, a_raw=a_raw, unmatched=unmatched, lat_r=lat_r, lat_f=lat_f)
        for lab, path in (("base", BASE / f"{folder}.npz"), ("fix", FIX / f"{folder}.npz")):
            t = load_times(path, bi)
            if t is None or len(t) != len(v):
                row[f"{lab}_status"] = "no display times" if t is None else f"{len(t)} times vs {len(v)} vsyncs"
                continue
            r = np.full(len(a), np.nan)
            r[assigned] = a[assigned] - t[f_of[assigned]]
            fr_rec[f"r_{lab}"] = r
            # assignment check, independent of timing deviations: is the TTL edge used for frame f the
            # TTL edge of the same physical transition as analog k (the TTL edge nearest to a_k)?
            if len(dw) > 1:
                i_an = np.clip(np.searchsorted(dw, a), 1, len(dw) - 1)
                i_an = np.where(np.abs(a - dw[i_an - 1]) <= np.abs(a - dw[i_an]), i_an - 1, i_an)
                te = t[np.clip(f_of, 0, None)]
                i_es = np.clip(np.searchsorted(dw, te), 1, len(dw) - 1)
                i_es = np.where(np.abs(te - dw[i_es - 1]) <= np.abs(te - dw[i_es]), i_es - 1, i_es)
                is_edge = np.abs(te - dw[i_es]) < 0.0005
                shift = np.where(assigned & is_edge, i_es - i_an, 0)
                fr_rec[f"edge_shift_{lab}"] = np.where(assigned, np.where(is_edge, i_es - i_an, 9999), -9999)
                row[f"{lab}_n_edge_shifted"] = int(np.sum(assigned & is_edge & (shift != 0)))
                row[f"{lab}_n_not_ttl_edge"] = int(np.sum(assigned & ~is_edge))
                fr_rec[f"is_edge_{lab}"] = is_edge & assigned
                fr_rec[f"shift_{lab}"] = shift
            # estimator landmarks at toggle level
            et = t[toggles]
            lm_est = set(toggles[1:][np.diff(et) > (N + 0.5) * FR].tolist())
            row[f"{lab}_landmarks_est"] = len(lm_est)
            row[f"{lab}_landmarks_only_analog"] = len(lm_an - lm_est)
            row[f"{lab}_landmarks_only_est"] = len(lm_est - lm_an)
            row[f"{lab}_status"] = "ok"
        frames[bi] = fr_rec
        per_block.append(row)
        rows.append(row)
    # session lag per polarity from branch estimates (fall back to baseline), all assigned toggles
    lag = {}
    for p in (1, -1):
        rs = [fr[k][fr["pol"] == p] for fr in frames.values() for k in ("r_fix",) if k in fr]
        rs = np.concatenate(rs) if rs else np.array([])
        rs = rs[~np.isnan(rs)]
        rs = rs[np.abs(rs) < 0.010]  # transitions matched to their own TTL edge (robust to misassigned blocks)
        lag[p] = float(np.median(rs)) if len(rs) else np.nan
    for row in per_block:
        fr = frames[row["block"]]
        row["lag_rise_ms"], row["lag_fall_ms"] = lag[1] * 1e3, lag[-1] * 1e3
        for lab in ("base", "fix"):
            if f"r_{lab}" not in fr:
                continue
            e = fr[f"r_{lab}"] - np.where(fr["pol"] == 1, lag[1], lag[-1])
            fr[f"err_{lab}"] = e
            e = e[~np.isnan(e)] * 1e3
            if not len(e):
                continue
            if f"is_edge_{lab}" in fr:
                big = ~np.isnan(fr[f"err_{lab}"]) & (np.abs(np.nan_to_num(fr[f"err_{lab}"])) > 0.004)
                ie, sh = fr[f"is_edge_{lab}"], fr[f"shift_{lab}"]
                row[f"{lab}_gt4_same_edge"] = int(np.sum(big & ie & (sh == 0)))
                row[f"{lab}_gt4_other_edge"] = int(np.sum(big & ie & (sh != 0)))
                row[f"{lab}_gt4_adjusted"] = int(np.sum(big & ~ie))
            row.update({f"{lab}_err_med": np.median(e), f"{lab}_err_p1": np.percentile(e, 1),
                        f"{lab}_err_p99": np.percentile(e, 99), f"{lab}_err_maxabs": np.max(np.abs(e)),
                        f"{lab}_n_gt2ms": int(np.sum(np.abs(e) > 2)), f"{lab}_n_gt4ms": int(np.sum(np.abs(e) > 4)),
                        f"{lab}_n_gt8ms": int(np.sum(np.abs(e) > 8))})
    out = HERE / "results"; out.mkdir(exist_ok=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(out / f"{folder}.csv", index=False)
    np.savez(out / f"{folder}_frames.npz", **{f"b{b}_{k}": v for b, fr in frames.items() for k, v in fr.items()},
             lag_rise=lag[1], lag_fall=lag[-1])
    for row in rows:
        if "box" not in row:
            print(f"{folder} b{row['block']}: {row.get('status')}"); continue
        def s(lab):
            if row.get(f"{lab}_status") != "ok" or f"{lab}_err_med" not in row:
                return f"{lab}: {row.get(f'{lab}_status')}"
            return (f"{lab}: err med {row[f'{lab}_err_med']:+.2f} p1 {row[f'{lab}_err_p1']:+.2f} p99 {row[f'{lab}_err_p99']:+.2f} "
                    f"max {row[f'{lab}_err_maxabs']:.2f} n>4ms {row[f'{lab}_n_gt4ms']} n>8ms {row[f'{lab}_n_gt8ms']} "
                    f"edge-shifted {row.get(f'{lab}_n_edge_shifted')} >4ms same/other/adj {row.get(f'{lab}_gt4_same_edge')}/{row.get(f'{lab}_gt4_other_edge')}/{row.get(f'{lab}_gt4_adjusted')}")
        print(f"{folder} b{row['block']} {row['stim'][:22]:22s} N={row['box']} toggles={row['n_toggles']} analog={row['n_analog_raw']} "
              f"unmatched analog/toggles={row['n_analog_unmatched']}/{row['n_toggles_unmatched']} dp-vs-ord={row['n_dp_vs_ordinal']} latR_min={row['lat_rise_min_ms']:.1f} "
              f"ttl_unmatched={row['ttl_unmatched']} lm_analog={row['n_landmarks_analog']} | "
              f"{s('base')} | {s('fix')} [{row['source_base']}->{row['source_fix']}]")
    print(f"{folder}: lag rise {lag[1]*1e3:.2f} ms, fall {lag[-1]*1e3:.2f} ms")
    return rows


if __name__ == "__main__":
    for f in sys.argv[1:]:
        main(f)
