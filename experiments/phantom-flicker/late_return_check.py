"""Is "late flip() return" real? Evidence check for the term used in
outputs/phantom-flicker-report.md (Method step 3) and npc_sync.patch (DIODE_OFF_GRID_FRACTION,
the second matching pass of align_diode_toggles, long_interval_mismatch['late_return']).

Claim (H1): for some frames j, the frame is displayed at its scheduled refresh but the
frame-signal falling edge comes late by D, so the interval before it is off the refresh grid.
TaskControl writes the falling edge right after `self._win.flip()` returns (DynamicRoutingTask
TaskControl.showFrame L413-416); PsychoPy records frameIntervals inside flip() (not checked
here: PsychoPy isn't installed locally).

Candidates, from the sync line only: frame j whose preceding falling-edge interval is off the
refresh grid (|frac| > 0.1 refresh), with a regular interval before it and an interval after
it that restores the grid (the two fractions sum to a whole number of refreshes within 0.1).
Controls: up to 100 frames per block with one-refresh intervals on both sides
(|frac| <= 0.02).

Checks (defined before running):
  C1 where is the delay? frameIntervals[j-1] - (v[j] - v[j-1]) (stim - sync, v = sync falling
     edges): 'stim == sync' (|diff| <= 0.2 ms): PsychoPy's frame timestamp is late too, so the
     delay is inside flip() before its timestamp ("late flip() return"); 'stim on grid' (diff =
     -offgrid part, +-0.3 ms): PsychoPy's interval is regular and only the falling edge is late,
     so the delay is after flip() took its timestamp (in or before the NI-DAQ write).
  C2 descriptive: pulse width excess dw and the gap from the previous falling edge to the
     rising edge.
  C3 when was frame j displayed (TTL photodiode)? Frame j's toggle = the one diode edge of its
     polarity in [v[j-1] + L - R/2, v[j] + L + R/2] (glitch and phantom-flicker edges excluded;
     L: frame j's local latency from align_diode_toggles). Display refresh
     m = (toggle - L - v[j-1]) / R on the grid of the previous falling edge (+-1 ms).
     'on time': m = first refresh after frame j's rising edge (H1). 'later': a refresh was
     missed. 'follows the late fall': the toggle is at v[j] + L, i.e. the display followed the
     late falling edge (H1 false). 'after the return': m R > v[j] - v[j-1] (flip() didn't block
     until the swap).
     (First version, superseded after a 2-session test: on-time position from the pulse width,
     T0 = v[j] - dw + L. Wrong when the rising edge is late and the pulse short: 40/200
     controls missed.)
  C4 same as C3 with the NI-DAQ AI0 photodiode (2025 box-1 sessions with local analog results),
     +-2 ms. Independent of the TTL photodiode line.
  C5 display intervals around j (latency-corrected toggles) vs the sync intervals' frame counts.

Outputs: late_return_check.csv.gz (candidates + controls), late_return_blocks.csv, the log; summaries: late_return_followup.py.
Run: PYTHONPATH="../../../npc_sync/src;../../../npc_stim/src;../../src;." ../../.venv/Scripts/python.exe late_return_check.py > late_return_check.log 2>&1
"""

from __future__ import annotations

import concurrent.futures as cf
import logging
import pathlib
import sys
import time
import traceback

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
A_DIR = pathlib.Path("C:/data/t2p_qc/analog")
RES = HERE.parent / "display-times-analog-validation" / "results"
ANALOG_FOLDERS = {"DRpilot_744279_20250116", "DRpilot_762526_20250320", "DRpilot_796012_20250714"}
OFF_GRID = 0.1
REGULAR = 0.02
TOL_TTL = 0.001
TOL_AN = 0.002
N_CONTROL = 100


def display_refresh(edges_pol, v, j, L, R, dv, m_first, tol):
    """(status, m, after_return) for frame j: its toggle is the one edge of its polarity in
    [v[j-1] + L - R/2, v[j] + L + R/2]; m = refreshes after frame j-1's falling edge."""
    lo, hi = v[j - 1] + L - 0.5 * R, v[j] + L + 0.5 * R
    i0, i1 = np.searchsorted(edges_pol, [lo, hi])
    if i1 - i0 != 1:
        return f"ambiguous ({i1 - i0} edges)", np.nan, False
    m_disp = (edges_pol[i0] - L - v[j - 1]) / R
    m = int(np.round(m_disp))
    if abs(m_disp - m) * R <= tol:
        st = "on time" if m == m_first else ("later" if m > m_first else "before the rise")
        return st, m, bool(m * R > dv[j - 1] + tol)
    if abs(m_disp - dv[j - 1] / R) * R <= tol:
        return "follows the late fall", np.nan, False
    return "off grid", np.nan, False


def load_analog(folder: str, bi: int):
    an, fr = A_DIR / folder / "analog.npz", RES / f"{folder}_frames.npz"
    if not (an.exists() and fr.exists()):
        return None
    A, F = np.load(an), np.load(fr)
    if f"b{bi}_t" not in A.files or f"b{bi}_a" not in F.files:
        return None
    return A[f"b{bi}_t"], A[f"b{bi}_pol"], F[f"b{bi}_a"], F[f"b{bi}_f"].astype(int), F[f"b{bi}_pol"]


def run(folder: str):
    import npc_sync

    import common

    logging.disable(logging.WARNING)
    rows, brows = [], []
    rng = np.random.default_rng(0)
    try:
        s = common.raw_sync(folder)
        if s.invalid_intervals or s.timebase_discontinuities:
            return rows, brows, f"{folder}: skipped (replays or discontinuities)"
        blocks, matches = common.matches(folder)
        e_all, q_all = common.diode_edges(s)
        vr_all = s.get_rising_edges("vsync_stim", units="seconds")
        for bi, m in enumerate(matches):
            if m.stim is None or (m.stim.diode_box_frame_interval or 1) != 1:
                continue
            v = np.asarray(blocks[bi], float)
            fi = np.asarray(m.stim.frame_intervals, float)
            n = len(v)
            if n != len(fi) + 1 or n < 10:
                continue
            sel = (e_all >= v[0]) & (e_all < v[-1] + 0.5)
            jr = np.searchsorted(vr_all, v) - 1
            vr = np.where(jr >= 0, vr_all[np.clip(jr, 0, None)], np.nan)
            try:
                al = npc_sync.align_diode_toggles(v, fi, e_all[sel], q_all[sel], frame_signal_rising=vr)
            except npc_sync.DiodeOffsetUndetermined:
                continue
            R = float(np.median(fi))
            dv = np.diff(v)
            ef = dv / R - np.round(dv / R)  # off-grid part of each sync falling-edge interval
            ef_stim = fi / R - np.round(fi / R)
            w = v - vr
            w_med = float(np.nanmedian(w))
            gap = vr[1:] - v[:-1]  # gap[j-1]: previous falling edge -> this rising edge
            gap_med = float(np.nanmedian(gap))
            L = al.expected - v
            keep = (al.edge_class != npc_sync.DIODE_GLITCH) & (al.edge_class != npc_sync.DIODE_PHANTOM_FLICKER)
            et, eq = al.edge_times[keep], al.edge_rising[keep]
            e_by_pol = {True: et[eq], False: et[~eq]}
            second = set(al.second_pass.tolist())
            tog = np.where(al.frame_edge >= 0, al.edge_times[np.clip(al.frame_edge, 0, None)], np.nan)
            ct = tog - L  # toggle-derived frame time
            c1 = (fi - dv) * 1e3
            pair = ef[:-1] + ef[1:]
            pair = pair - np.round(pair)
            js = np.arange(2, n - 1)
            cand = js[(np.abs(ef[js - 1]) > OFF_GRID) & (np.abs(pair[js - 1]) < OFF_GRID) & (np.abs(ef[js - 2]) <= REGULAR)]
            reg = js[(np.abs(ef[js - 1]) <= REGULAR) & (np.abs(ef[js]) <= REGULAR)
                     & (np.round(dv[js - 1] / R) == 1) & (np.round(dv[js] / R) == 1)]
            ctrl = rng.choice(reg, size=min(N_CONTROL, len(reg)), replace=False) if len(reg) else np.array([], int)
            an = load_analog(folder, bi) if folder in ANALOG_FOLDERS else None
            if an is not None:
                At, Apol, Fa, Ff, Fpol = an
                okf = (Ff >= 0) & (Ff < n)
                rising_is = 1 if np.mean((Fpol[okf] == 1) == al.frame_rising[Ff[okf]]) > 0.5 else -1
                regmask = np.zeros(n, bool)
                regmask[reg] = True
                La = {}
                for p in (1, -1):
                    mm = okf & (Fpol == p)
                    mm &= regmask[np.clip(Ff, 0, n - 1)]
                    La[p] = float(np.median(Fa[mm] - v[Ff[mm]]))
                A_by_pol = {p: np.sort(At[Apol == p]) for p in (1, -1)}
            for kind, frames in (("candidate", cand), ("control", ctrl)):
                for j in frames:
                    j = int(j)
                    dw = float(w[j] - w_med)
                    e1 = float(ef[j - 1] * R)
                    dmod = ((dw - e1) + R / 2) % R - R / 2
                    n_rise = (vr[j] - v[j - 1]) / R
                    m_first = int(np.floor(n_rise)) + 1
                    p = bool(al.frame_rising[j])
                    st_ttl, m_ttl, ar_ttl = display_refresh(e_by_pol[p], v, j, L[j], R, dv, m_first, TOL_TTL)
                    c1j = float(c1[j - 1])
                    e1_ms = e1 * 1e3
                    c1_class = ("stim == sync" if abs(c1j) <= 0.2 else
                                "stim on grid" if abs(c1j + e1_ms) <= 0.3 else "other")
                    r = dict(folder=folder, block=bi, stim=m.stim.name, kind=kind, j=j, R_ms=R * 1e3,
                             dv_prev_ms=dv[j - 1] * 1e3, dv_next_ms=dv[j] * 1e3,
                             n_prev=int(np.round(dv[j - 1] / R)), n_next=int(np.round(dv[j] / R)),
                             frac_prev=float(ef[j - 1]), frac_stim_prev=float(ef_stim[j - 1]),
                             offgrid_ms=e1_ms, c1_ms=c1j, c1_class=c1_class,
                             return_delay_ms=float((dv[j - 1] - m_first * R) * 1e3),
                             dw_ms=dw * 1e3, gap_excess_ms=float((gap[j - 1] - gap_med) * 1e3),
                             dmod_ms=float(dmod * 1e3), n_rise=float(n_rise), m_first=m_first,
                             status_ttl=st_ttl, m_ttl=m_ttl, after_return_ttl=ar_ttl,
                             matched=("second" if j in second else ("first" if al.frame_edge[j] >= 0 else "unmatched")),
                             res_ms=float(al.residual[j] * 1e3),
                             disp_prev=float(np.round((ct[j] - ct[j - 1]) / R)) if np.isfinite(ct[j] - ct[j - 1]) else np.nan,
                             disp_next=float(np.round((ct[j + 1] - ct[j]) / R)) if np.isfinite(ct[j + 1] - ct[j]) else np.nan)
                    if an is not None:
                        pa = rising_is if p else -rising_is
                        st_an, m_an, ar_an = display_refresh(A_by_pol[pa], v, j, La[pa], R, dv, m_first, TOL_AN)
                        r.update(status_an=st_an, m_an=m_an, after_return_an=ar_an)
                    rows.append(r)
            brows.append(dict(folder=folder, block=bi, stim=m.stim.name, n_frames=n, R_ms=R * 1e3, w_med_ms=w_med * 1e3,
                              gap_med_ms=gap_med * 1e3, n_off_grid_sync=int((np.abs(ef) > OFF_GRID).sum()),
                              n_off_grid_stim=int((np.abs(ef_stim) > OFF_GRID).sum()),
                              n_stim_only_off_grid=int(((np.abs(ef_stim) > OFF_GRID) & (np.abs(ef) <= REGULAR)).sum()),
                              n_sync_only_off_grid=int(((np.abs(ef) > OFF_GRID) & (np.abs(ef_stim) <= REGULAR)).sum()),
                              n_cand=len(cand), c1_n=len(c1), c1_n_gt02=int((np.abs(c1) > 0.2).sum()),
                              c1_p001=float(np.percentile(c1, 0.1)), c1_p999=float(np.percentile(c1, 99.9)),
                              c1_maxabs=float(np.abs(c1).max())))
        return rows, brows, f"{folder}: {len(brows)} blocks"
    except Exception:  # noqa: BLE001
        return rows, brows, f"{folder}: ERROR {traceback.format_exc().splitlines()[-1]}"


def summarize(d: pd.DataFrame, b: pd.DataFrame) -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(f"box-1 blocks aligned: {len(b)} in {b.folder.nunique()} sessions; frame intervals {int(b.c1_n.sum())}")
    print(f"off-grid sync falling-edge intervals (>0.1 refresh): {int(b.n_off_grid_sync.sum())}; "
          f"off-grid stim frameIntervals: {int(b.n_off_grid_stim.sum())}; off-grid in stim only (sync regular): "
          f"{int(b.n_stim_only_off_grid.sum())}; in sync only (stim regular): {int(b.n_sync_only_off_grid.sum())}")
    c = d[d.kind == "candidate"].copy()
    k = d[d.kind == "control"].copy()
    print(f"candidates (isolated off-grid falling edge): {len(c)} in {c.groupby(['folder', 'block']).ngroups} blocks, "
          f"{c.folder.nunique()} sessions; controls {len(k)}")
    print("\nC1 stim frameIntervals - sync falling-edge intervals (ms):")
    print(f"  all intervals: |diff| > 0.2 ms in {int(b.c1_n_gt02.sum())} of {int(b.c1_n.sum())}; per-block p0.1 min "
          f"{b.c1_p001.min():.3f}, p99.9 max {b.c1_p999.max():.3f}")
    print("  candidates: 'stim == sync' = the delay is also in PsychoPy's frame interval (inside flip(), before its timestamp);"
          " 'stim on grid' = PsychoPy's interval is regular, only the falling edge is late (after flip()'s timestamp)")
    print(c.c1_class.value_counts().to_string())
    R = float(c.R_ms.median()) if len(c) else 16.68
    c["delay"] = pd.cut(c.return_delay_ms, [-np.inf, 2, R - 1, R + 1, np.inf],
                        labels=["< 2 ms", "2 ms - R-1", "R-1 - R+1", ">= R+1"])
    print("  delay (falling edge - first refresh after the rise) by class, quantiles (ms):")
    print(c.groupby("c1_class").return_delay_ms.describe(percentiles=[0.05, 0.5, 0.95]).round(2).to_string())
    print("\nC2 pulse excess dw and rise gap excess (ms):")
    print(c.groupby("c1_class")[["dw_ms", "gap_excess_ms"]].median().round(2).to_string())
    print("\nC3 TTL photodiode: display refresh of frame j:")
    print(pd.crosstab([c.c1_class, c.delay], c.status_ttl, margins=True).to_string())
    print(f"  displayed after flip() returned (non-blocking): {int(c.after_return_ttl.sum())}")
    print("controls:", k.status_ttl.value_counts().to_dict())
    print("  how align_diode_toggles matched candidate toggles:", c.matched.value_counts().to_dict())
    if "status_an" in d:
        an = c[c.status_an.notna()]
        ak = k[k.status_an.notna()]
        print(f"\nC4 NI-DAQ AI0 ({an.folder.nunique()} sessions, {len(an)} candidates, {len(ak)} controls), +-2 ms:")
        if len(an):
            print(pd.crosstab([an.c1_class, an.delay], an.status_an, margins=True).to_string())
            print("  TTL vs analog verdict:", an.groupby(["status_ttl", "status_an"]).size().to_dict())
            print(f"  displayed after flip() returned: {int(an.after_return_an.sum())}")
        print("controls:", ak.status_an.value_counts().to_dict())
    print("\nC5 frame counts: sync intervals (n_prev, n_next) vs display from the toggles (disp_prev, disp_next), candidates:")
    print(c.groupby(["n_prev", "n_next", "disp_prev", "disp_next"], dropna=False).size().sort_values(ascending=False).head(12).to_string())
    cols = ["folder", "block", "j", "dv_prev_ms", "dv_next_ms", "c1_ms", "c1_class", "return_delay_ms", "dw_ms",
            "gap_excess_ms", "status_ttl", "m_ttl", "matched", "res_ms"]
    print("\nexamples, largest delays:")
    print(c.sort_values("return_delay_ms", ascending=False)[cols].head(8).round(2).to_string())
    print("examples, median delays:")
    print(c.sort_values("return_delay_ms")[cols].iloc[len(c) // 2 : len(c) // 2 + 5].round(2).to_string())
    odd = c[c.status_ttl != "on time"]
    print(f"\ncandidates not 'on time' (TTL): {len(odd)}")
    print(odd[cols].round(2).head(25).to_string())


def main() -> None:
    import common

    folders = sorted(p.name for p in common.CACHE.iterdir() if p.is_dir())
    folders = [f for f in folders if f != "DRpilot_366122_20250109"]
    if len(sys.argv) > 1:
        folders = [f for f in folders if any(s in f for s in sys.argv[1:])]
    t0 = time.time()
    rows, brows, msgs = [], [], []
    with cf.ProcessPoolExecutor(max_workers=10) as ex:
        for r, b, msg in ex.map(run, folders):
            rows += r
            brows += b
            msgs.append(msg)
    d, b = pd.DataFrame(rows), pd.DataFrame(brows)
    d.to_csv(HERE / "late_return_check.csv.gz", index=False)
    b.to_csv(HERE / "late_return_blocks.csv", index=False)
    print(f"{len(folders)} sessions in {time.time() - t0:.0f} s; errors: {[m for m in msgs if 'ERROR' in m]}")
    summarize(d, b)


if __name__ == "__main__":
    main()
