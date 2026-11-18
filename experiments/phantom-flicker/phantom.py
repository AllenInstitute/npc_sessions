"""Prototype: align photodiode (TTL) edges to the toggle frames of a box-1 block and classify
the edges that aren't a frame's toggle.

Model (TaskControl `showFrame`, diodeBoxFrameInterval = 1): every frame k inverts the diode box,
so frame k's toggle has polarity p_k = p_0 xor (k odd) and appears at v[k] + L(p_k) (v = vsync
falling edges; L = per-polarity latency, piecewise constant between stalls). A frame held for
n >= 2 refreshes (stim `frameIntervals[k]`: a dropped frame) shows no toggle until frame k+1's.

Edge classes:
- toggle: a frame's toggle edge
- glitch: comparator chatter, consecutive edges < GLITCH_MAX_SEC apart (removed in pairs, the
  first crossing of an odd cluster is kept). Not a display change: the display changes at most
  once per refresh.
- phantom: phantom flicker. State-restoring pairs strictly between the toggles of frame k and
  k+1, where frame k is held for n >= 3 refreshes, at most n - 1 edges (one per missed refresh).
- unexplained: anything else between the first and last matched toggles
- outside: before the first or after the last matched toggle

The frame <-> edge offset can't be determined from regular frames (v[k] + L ~ v[k+1] + L - R):
it is fixed by landmarks (long diode intervals matching long stim intervals).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import numpy.typing as npt

TOGGLE, GLITCH, PHANTOM, UNEXPLAINED, OUTSIDE = 0, 1, 2, 3, 4
CLASS_NAMES = {TOGGLE: "toggle", GLITCH: "glitch", PHANTOM: "phantom", UNEXPLAINED: "unexplained", OUTSIDE: "outside"}

GLITCH_MAX_SEC = 0.05 / 59.95
"""Same threshold as npc_stim's DEFINITE_SHORT_INTERVAL_THRESHOLD (0.83 ms)."""
LONG_INTERVAL_REFRESHES = 1.5
"""A stim frame interval at least this many refreshes long is a dropped frame (held frame)."""
MAX_POLARITY_LATENCY_DIFF_SEC = 0.008
LANDMARK_LATENCY_RANGE_SEC = (0.005, 0.035)
"""TTL latency range accepted for landmarks. Observed block medians: rising 19.5-20.4 ms, falling
23.4-25.9 ms (survey_blocks.csv); one refresh more (>= 36 ms) is excluded, which is the alias a
late `flip()` return produces (stim interval k long, diode interval k+1 long)."""
MIN_LANDMARKS = 3
MIN_LANDMARK_AGREEMENT = 0.9
LANDMARK_AGREEMENT_TOL_SEC = 0.008
"""A landmark agrees if its latency is within this of its polarity's mode. Latency drifts and can switch between two states ~5.5 ms apart within a block (e.g. falling 23 vs 28.5 ms in DRpilot_674562_20231004 b6); an alias is a refresh (16.7 ms) away. Was (latency drifts by a
few ms over a block)."""
LANDMARK_LENGTH_TOL_REFRESHES = 0.5
FIRST_PASS_TOL_SEC = 0.008
MATCH_TOL_SEC = 0.012
"""A toggle edge must be within this of its expected time (after-stall TTL edges can be
~8-10 ms early or late)."""
SAME_POLARITY_LONG_REFRESHES = 2.5
"""Same-polarity toggles (frames k, k+2) are 2 refreshes apart, >= 3 across a held frame."""
MIN_SEGMENT_EDGES = 3
MIN_DISPLAY_AFTER_RISE_SEC = 0.015
"""A frame can't be displayed before its frame-signal rising edge + this (npc_stim
MIN_VSYNC_DIODE_FLIP_SEPARATION_SEC; >= 20.9 ms in 2.2 M analog-checked toggles). Used as the
lower bound of the second matching pass, for frames whose falling edge is late (late `flip()`
return: the falling edge and the stim's frame interval are late, the frame was displayed on
time)."""
OFF_GRID_FRACTION = 0.1
"""A stim frame interval more than this fraction of a refresh from a whole number of refreshes
is off the refresh grid (a late `flip()` return at one of its ends)."""
AFTER_STALL_RESIDUAL_SEC = 0.25 / 59.95


class OffsetUndetermined(ValueError):
    """No long stim frame interval shows as a long diode interval: the frame <-> edge offset
    can't be fixed from the diode."""


@dataclasses.dataclass
class Alignment:
    frame_edge: npt.NDArray[np.intp]
    """Per frame: index of its toggle edge, or -1."""
    edge_class: npt.NDArray[np.int8]
    edge_frame: npt.NDArray[np.intp]
    """Per edge: the frame whose toggle it is (toggle), or the held frame k whose hold it is in
    (phantom; unexplained edges between two matched toggles: the earlier frame), or -1."""
    first_frame_rising: bool
    latency_rising: npt.NDArray[np.float64]
    latency_falling: npt.NDArray[np.float64]
    n_landmarks: int
    n_landmarks_disagree: int
    refresh_sec: float
    n_refreshes: npt.NDArray[np.int_]
    residual: npt.NDArray[np.float64]
    """Per frame: toggle edge - expected time (NaN if unmatched)."""
    long_mismatch: npt.NDArray[np.intp]
    """Frames k (same-polarity pairs k, k+2 with both matched) where diode and stim disagree on
    whether the pair spans a held frame."""
    displaced: npt.NDArray[np.intp]
    """Held frames k (>= 2 refreshes) where frame k+1's toggle is missing and an odd number of
    unmatched edges lies in the hold. Diagnostic only: not removable."""
    second_pass: npt.NDArray[np.intp]
    """Frames matched only in the second pass (edge earlier than expected - tol, but after the
    frame-signal rising edge + MIN_DISPLAY_AFTER_RISE_SEC)."""
    literal_long_mismatch: dict[str, npt.NDArray[np.intp]]
    """The literal check: stim long intervals (>= 1.5 refreshes) vs diode toggle-to-toggle
    intervals (latency-corrected) >= 1.5 refreshes. Mismatching frame intervals k, by cause:
    'late_return' (stim interval k or k-1/k+1 off the refresh grid), 'after_stall' (toggle k or
    k+1 off its expected time by > 1/4 refresh), 'other'."""

    def count(self, c: int) -> int:
        return int((self.edge_class == c).sum())

    @property
    def missing_frames(self) -> npt.NDArray[np.intp]:
        return np.flatnonzero(self.frame_edge < 0)

    @property
    def validated(self) -> bool:
        return (
            self.count(UNEXPLAINED) == 0
            and len(self.missing_frames) == 0
            and len(self.literal_long_mismatch["other"]) == 0
        )


def glitch_mask(e: npt.NDArray, max_sec: float = GLITCH_MAX_SEC) -> npt.NDArray[np.bool_]:
    """Edges to remove as comparator chatter: clusters of consecutive edges < max_sec apart.
    An even cluster is removed whole (a pulse); an odd cluster keeps its last edge (a transition
    with chatter; npc_stim's anomalous-interval removal also keeps the last).

    >>> glitch_mask(np.array([0.0, 1.0, 1.0001, 1.0002, 2.0, 3.0, 3.0003])).tolist()
    [False, True, True, False, False, True, True]
    """
    out = np.zeros(len(e), bool)
    if len(e) < 2:
        return out
    close = np.diff(e) < max_sec
    i = 0
    while i < len(close):
        if not close[i]:
            i += 1
            continue
        j = i
        while j < len(close) and close[j]:
            j += 1
        n = j - i + 1  # edges i..j
        out[i : j + 1 - (n % 2)] = True
        i = j + 1
    return out


def _landmarks(v, fi, e, q, R):
    # holds on the refresh grid, with on-grid neighbours: a late `flip()` return at either end
    # moves the stim's long interval by a frame
    frac = np.abs(fi / R - np.round(fi / R))
    on = frac <= OFF_GRID_FRACTION
    on_nb = on & np.r_[True, on[:-1]] & np.r_[on[1:], True]
    long_k = np.flatnonzero((fi >= LONG_INTERVAL_REFRESHES * R) & on_nb)
    d = np.diff(e)
    long_j = np.flatnonzero(d >= LONG_INTERVAL_REFRESHES * R)
    vs = v[long_k]
    lo_l, hi_l = LANDMARK_LATENCY_RANGE_SEC
    offs, pols, ks = [], [], []
    for j in long_j:
        lo = np.searchsorted(vs, e[j] - hi_l)
        hi = np.searchsorted(vs, e[j] - lo_l, side="right")
        for i in range(lo, hi):
            k = long_k[i]
            if abs(d[j] - fi[k]) < LANDMARK_LENGTH_TOL_REFRESHES * R:
                offs.append(e[j] - v[k])
                pols.append(q[j])
                ks.append(k)
    return np.array(offs), np.array(pols, bool), np.array(ks, int)


def landmark_offset(v, fi, e, q, R) -> tuple[int, float, float, int, int]:
    """(parity of rising frames, L_rising, L_falling, n landmarks agreeing, n disagreeing).

    A landmark: a diode interval >= 1.5 refreshes long whose length matches (within half a
    refresh) a long stim frame interval k starting 0-60 ms before it: its first edge is frame
    k's toggle, so L = e_j - v[k] with no one-refresh alias.
    """
    offs, pols, ks = _landmarks(v, fi, e, q, R)
    if len(offs) == 0:
        raise OffsetUndetermined("no long diode interval matches a long stim frame interval")
    lo_l, hi_l = LANDMARK_LATENCY_RANGE_SEC
    bins = np.arange(lo_l, hi_l + 0.0005, 0.0005)

    def mode(x):
        hs = np.convolve(np.histogram(x, bins)[0], np.ones(5), mode="same")
        return float(bins[np.argmax(hs)] + 0.00025)

    modes = {pol: mode(offs[pols == pol]) for pol in (True, False) if (pols == pol).any()}
    if len(modes) == 2 and abs(modes[True] - modes[False]) > MAX_POLARITY_LATENCY_DIFF_SEC:
        raise OffsetUndetermined(
            f"rising/falling landmark latencies differ by {abs(modes[True] - modes[False]) * 1e3:.1f} ms"
        )
    ref = np.array([modes.get(p, next(iter(modes.values()))) for p in pols])
    agree = np.abs(offs - ref) <= LANDMARK_AGREEMENT_TOL_SEC
    if agree.sum() < MIN_LANDMARKS or agree.mean() < MIN_LANDMARK_AGREEMENT:
        raise OffsetUndetermined(
            f"{int(agree.sum())} of {len(offs)} landmarks agree on the latency (need >= {MIN_LANDMARKS} and >= {MIN_LANDMARK_AGREEMENT:.0%})"
        )
    Lr = float(np.median(offs[agree & pols])) if (agree & pols).any() else modes.get(True, modes.get(False))
    Lf = float(np.median(offs[agree & ~pols])) if (agree & ~pols).any() else modes.get(False, modes.get(True))
    par = np.where(pols[agree], ks[agree] % 2, 1 - ks[agree] % 2)
    if 0 < par.mean() < 1 and min(par.mean(), 1 - par.mean()) > 1 - MIN_LANDMARK_AGREEMENT:
        raise OffsetUndetermined("landmarks disagree on the parity of rising frames")
    c_r = int(np.round(np.mean(par)))
    return c_r, Lr, Lf, int(agree.sum()), int((~agree).sum())


def _mutual_nearest(t, e, tol):
    """Pairs (i_t, i_e) of mutual nearest neighbours with |e - t| < tol (t, e sorted)."""
    if len(t) == 0 or len(e) == 0:
        return np.array([], int), np.array([], int)
    if len(e) == 1:
        near_e = np.zeros(len(t), int)
    else:
        j = np.clip(np.searchsorted(e, t), 1, len(e) - 1)
        near_e = np.where(np.abs(t - e[j - 1]) <= np.abs(e[j] - t), j - 1, j)
    if len(t) == 1:
        near_t = np.zeros(len(e), int)
    else:
        i = np.clip(np.searchsorted(t, e), 1, len(t) - 1)
        near_t = np.where(np.abs(e - t[i - 1]) <= np.abs(t[i] - e), i - 1, i)
    it = np.arange(len(t))
    ok = (near_t[near_e] == it) & (np.abs(e[near_e] - t) < tol)
    return it[ok], near_e[ok]


def _match(t, p_rising, e, q, tol, early=None):
    """frame -> edge index (or -1), matching each polarity separately.

    Each frame takes the edge of its polarity nearest its expected time t within
    [t - tol - early, t + tol]; an edge claimed by two frames goes to the nearer.
    """
    fe = np.full(len(t), -1, np.intp)
    early = np.zeros(len(t)) if early is None else early
    for pol in (True, False):
        ks = np.flatnonzero(p_rising == pol)
        js = np.flatnonzero(q == pol)
        if len(ks) == 0 or len(js) == 0:
            continue
        ee = e[js]
        tk = t[ks]
        lo = np.searchsorted(ee, tk - tol - early[ks])
        hi = np.searchsorted(ee, tk + tol, side="right")
        best = np.full(len(ks), -1, np.intp)
        dist = np.full(len(ks), np.inf)
        # candidates: at most a few per frame; iterate over offsets from lo
        width = int(np.max(hi - lo)) if len(ks) else 0
        for off in range(width):
            c = lo + off
            ok = c < hi
            d = np.where(ok, np.abs(ee[np.clip(c, 0, len(ee) - 1)] - tk), np.inf)
            better = d < dist
            best[better] = c[better]
            dist[better] = d[better]
        # conflicts: an edge chosen by more than one frame goes to the nearest
        m = best >= 0
        if m.any():
            order = np.lexsort((dist[m], best[m]))
            idx = np.flatnonzero(m)[order]
            dup = np.r_[False, best[idx][1:] == best[idx][:-1]]
            best[idx[dup]] = -1
        fe[ks[best >= 0]] = js[best[best >= 0]]
    return fe


def _segment_latency(res, p_rising, seg, n_seg, default_r, default_f):
    """Per-frame latency correction: median residual per (segment, polarity), carried forward
    from the previous segment where a segment has too few matched toggles."""
    out = np.zeros(len(res))
    for pol, default in ((True, default_r), (False, default_f)):
        sel = (p_rising == pol) & np.isfinite(res)
        med = np.full(n_seg, np.nan)
        if sel.any():
            s = seg[sel]
            r = res[sel]
            order = np.argsort(s, kind="stable")
            s, r = s[order], r[order]
            bounds = np.searchsorted(s, np.arange(n_seg + 1))
            for i in range(n_seg):
                a, z = bounds[i], bounds[i + 1]
                if z - a >= MIN_SEGMENT_EDGES:
                    med[i] = np.median(r[a:z])
        # carry forward, then backward
        last = np.nan
        for i in range(n_seg):
            if np.isnan(med[i]):
                med[i] = last
            else:
                last = med[i]
        last = np.nan
        for i in range(n_seg - 1, -1, -1):
            if np.isnan(med[i]):
                med[i] = last
            else:
                last = med[i]
        med = np.nan_to_num(med)
        out[p_rising == pol] = med[seg[p_rising == pol]]
    return out


def align_box1(
    vsyncs: npt.ArrayLike,
    frame_intervals: npt.ArrayLike,
    edges: npt.ArrayLike,
    rising: npt.ArrayLike,
    match_tol_sec: float = MATCH_TOL_SEC,
    vsync_rising: npt.ArrayLike | None = None,
) -> Alignment:
    """`vsync_rising`: the frame-signal rising edge before each frame's falling edge (optional;
    used to detect late `flip()` returns)."""
    v = np.asarray(vsyncs, float)
    fi = np.asarray(frame_intervals, float)
    e_all = np.asarray(edges, float)
    q_all = np.asarray(rising, bool)
    N = len(v)
    if len(fi) != N - 1:
        raise ValueError(f"{N} frame times but {len(fi)} frame intervals")
    R = float(np.median(fi))
    nref = np.round(fi / R).astype(int)
    # glitches first: they are not display changes
    gl = glitch_mask(e_all)
    keep = np.flatnonzero(~gl)
    e, q = e_all[keep], q_all[keep]
    c_r, Lr, Lf, n_agree, n_dis = landmark_offset(v, fi, e, q, R)
    p_rising = (np.arange(N) % 2) == c_r
    # segments: a new one starts after every held frame (latency can switch at a stall)
    seg = np.concatenate([[0], np.cumsum(fi >= LONG_INTERVAL_REFRESHES * R)])
    n_seg = int(seg[-1]) + 1
    base = np.where(p_rising, Lr, Lf)
    corr = np.zeros(N)
    for tol in (FIRST_PASS_TOL_SEC, match_tol_sec):
        t = v + base + corr
        fe = _match(t, p_rising, e, q, tol)
        res = np.where(fe >= 0, e[np.clip(fe, 0, None)] - t, np.nan)
        corr = corr + _segment_latency(res, p_rising, seg, n_seg, 0.0, 0.0)
    t = v + base + corr
    fe = _match(t, p_rising, e, q, match_tol_sec)
    second = np.array([], np.intp)
    if vsync_rising is not None:
        # second pass for unmatched frames: down to rising edge + MIN_DISPLAY_AFTER_RISE_SEC
        vr = np.asarray(vsync_rising, float)
        un = np.flatnonzero(fe < 0)
        used = np.zeros(len(e), bool)
        used[fe[fe >= 0]] = True
        early = np.clip(t[un] - match_tol_sec - (vr[un] + MIN_DISPLAY_AFTER_RISE_SEC), 0, None)
        early[~np.isfinite(early)] = 0
        free = np.flatnonzero(~used)
        fe2 = _match(t[un], p_rising[un], e[free], q[free], match_tol_sec, early=early)
        take = fe2 >= 0
        fe[un[take]] = free[fe2[take]]
        second = un[take]
    # matched toggles must be in frame order: unmatch any pair out of order (rare)
    m = np.flatnonzero(fe >= 0)
    bad = np.flatnonzero(np.diff(fe[m]) <= 0)
    fe[m[bad]] = -1
    fe[m[bad + 1]] = -1
    m = np.flatnonzero(fe >= 0)
    residual = np.where(fe >= 0, e[np.clip(fe, 0, None)] - t, np.nan)
    # classify the non-glitch edges
    M = len(e)
    cls = np.full(M, UNEXPLAINED, np.int8)
    ef = np.full(M, -1, np.intp)
    cls[fe[m]] = TOGGLE
    ef[fe[m]] = m
    displaced = []
    if len(m):
        cls[: fe[m[0]]] = OUTSIDE
        cls[fe[m[-1]] + 1 :] = OUTSIDE
        gaps = np.flatnonzero(np.diff(fe[m]) > 1)
        for g in gaps:
            ka, kb = m[g], m[g + 1]
            ja, jb = fe[ka], fe[kb]
            between = np.arange(ja + 1, jb)
            ef[between] = ka
            if kb == ka + 1 and nref[ka] >= 3 and len(between) <= nref[ka] - 1 and len(between) % 2 == 0:
                cls[between] = PHANTOM
            elif kb == ka + 2 and nref[ka] >= 2 and len(between) % 2 == 1:
                displaced.append(ka)
    # same-polarity long-interval check
    both = np.flatnonzero((fe[:-2] >= 0) & (fe[2:] >= 0))
    d_same = e[fe[both + 2]] - e[fe[both]]
    s_same = fi[both] + fi[both + 1]
    long_mismatch = both[(d_same >= SAME_POLARITY_LONG_REFRESHES * R) != (s_same >= SAME_POLARITY_LONG_REFRESHES * R)]
    # literal long-interval check (latency-corrected toggle-to-toggle intervals)
    both1 = np.flatnonzero((fe[:-1] >= 0) & (fe[1:] >= 0))
    lat = t - v
    ct = np.where(fe >= 0, e[np.clip(fe, 0, None)] - lat, np.nan)
    d_long = (ct[both1 + 1] - ct[both1]) >= LONG_INTERVAL_REFRESHES * R
    s_long = fi[both1] >= LONG_INTERVAL_REFRESHES * R
    mm = both1[d_long != s_long]
    frac = np.abs(fi / R - np.round(fi / R))
    off_grid = np.r_[frac > OFF_GRID_FRACTION, False]
    big_res = np.abs(np.nan_to_num(residual)) > AFTER_STALL_RESIDUAL_SEC
    is_late = off_grid[mm] | off_grid[np.clip(mm - 1, 0, None)] | off_grid[mm + 1]
    is_after = ~is_late & (big_res[mm] | big_res[mm + 1])
    literal = {"late_return": mm[is_late], "after_stall": mm[is_after], "other": mm[~is_late & ~is_after]}
    # map back to original edge indices (glitches included)
    frame_edge = np.where(fe >= 0, keep[np.clip(fe, 0, None)], -1)
    edge_class = np.full(len(e_all), GLITCH, np.int8)
    edge_class[keep] = cls
    edge_frame = np.full(len(e_all), -1, np.intp)
    edge_frame[keep] = ef
    return Alignment(
        frame_edge=frame_edge,
        edge_class=edge_class,
        edge_frame=edge_frame,
        first_frame_rising=bool(p_rising[0]),
        latency_rising=np.where(p_rising, t - v, np.nan),
        latency_falling=np.where(~p_rising, t - v, np.nan),
        n_landmarks=n_agree,
        n_landmarks_disagree=n_dis,
        refresh_sec=R,
        n_refreshes=nref,
        residual=residual,
        long_mismatch=long_mismatch,
        displaced=np.array(displaced, np.intp),
        second_pass=second,
        literal_long_mismatch=literal,
    )
