"""Analog photodiode transition detection (Schmitt trigger on the locally normalised trace).

    C:/data/t2p_qc/venv_analog/Scripts/python detect.py <folder> [<folder> ...]

Reads the 5 kHz traces saved by fetch_analog.py (C:/data/t2p_qc/analog/<folder>/trace_b{i}.npy) and
writes analog.npz: per block the transition times on the sync clock, polarity, swing, and 0.5 s bins
flagged where the signal swing is below MIN_SWING (signal lost).

Thresholds are fractions of the local swing (see `levels`). The white phase of the box has a lower
intermediate level for part of it (an overshoot on the first refreshes after black -> white, then the
steady white level), so falls are timed low in the swing: Schmitt high 0.5, low 0.15, timing at the
0.25 crossing (both polarities).
"""
from __future__ import annotations
import json, pathlib, sys
import numpy as np

A_DIR = pathlib.Path("C:/data/t2p_qc/analog")
SCHMITT_HI, SCHMITT_LO, MID = 0.5, 0.15, 0.25
MIN_SWING = 100  # counts


def levels(y, fs):
    """Local black/white levels: running minimum / maximum over 80 ms centred (always contains >= 30 ms
    of each phase when the box toggles every <= 3 frames). Local, because the white level can change
    abruptly (e.g. 1250 <-> 380 counts in OptoTagging, 744279_20250116) and with box interval 1 the
    trace doesn't settle to black between toggles (762526_20250320). Where the 80 ms minimum is > 35 %
    of the 5 s swing above the 5 s black level (window inside a long white phase), the 5 s black level
    is used. Where the local swing is only noise (< max(50 counts, 5 % of the 5 s swing): a long
    constant phase, e.g. a stall), the 5 s levels are used instead."""
    from scipy.ndimage import maximum_filter1d, minimum_filter1d, percentile_filter
    lo = minimum_filter1d(y, int(0.08 * fs), mode="nearest")
    hi = maximum_filter1d(y, int(0.08 * fs), mode="nearest")
    step = int(0.05 * fs)
    ys = y[::step]
    g_lo = np.interp(np.arange(len(y)), np.arange(len(ys)) * step, percentile_filter(ys, 1, size=100, mode="nearest"))
    g_hi = np.interp(np.arange(len(y)), np.arange(len(ys)) * step, percentile_filter(ys, 99, size=100, mode="nearest"))
    g_sw = g_hi - g_lo
    # an 80 ms window inside a long white phase has no black: its minimum is the white level after the
    # overshoot decays (would give a spurious fall), so use the 5 s black level there
    lo = np.where(lo - g_lo > 0.35 * g_sw, g_lo, lo)
    weak = (hi - lo) < np.maximum(50.0, 0.05 * g_sw)  # flat (only noise): long constant phase
    lo = np.where(weak, g_lo, lo)
    hi = np.where(weak, g_hi, hi)
    return lo, hi, g_sw


def detect(y, fs, smooth_ms=0.3, hi_thr=SCHMITT_HI, lo_thr=SCHMITT_LO, mid=MID):
    """Returns (positions in samples (float, interpolated MID crossing), polarity +1/-1, swing,
    per-0.5 s-bin validity (5 s swing >= MIN_SWING))."""
    y = y.astype(float)
    k = max(int(round(smooth_ms * 1e-3 * fs)), 1)
    if k > 1:
        y = np.convolve(y, np.ones(k) / k, mode="same")
    lo, hi, g_sw = levels(y, fs)
    sw = hi - lo
    u = (y - lo) / np.where(sw > 0, sw, 1)
    u[g_sw < MIN_SWING] = np.nan
    bn = int(0.5 * fs)
    valid = np.array([np.median(g_sw[i:i + bn]) >= MIN_SWING for i in range(0, len(y), bn)])
    state = np.full(len(u), np.nan)
    state[u > hi_thr] = 1
    state[u < lo_thr] = 0
    has = ~np.isnan(state)
    if not has.any():
        return np.array([]), np.array([], int), np.array([]), valid
    idx = np.where(has, np.arange(len(u)), 0)
    np.maximum.accumulate(idx, out=idx)
    st = state[idx]
    st[: np.argmax(has)] = np.nan
    ch = np.where(np.diff(st) != 0)[0] + 1
    ch = ch[~np.isnan(st[ch - 1]) & ~np.isnan(st[ch])]
    pol = np.where(st[ch] == 1, 1, -1)
    below = np.where(u < mid)[0]
    above = np.where(u >= mid)[0]
    pos = np.empty(len(ch))
    for n, (i, p) in enumerate(zip(ch, pol)):
        other = below if p == 1 else above
        s = np.searchsorted(other, i)
        j = other[s - 1] if s > 0 else i - 1
        a, b = u[j], u[min(j + 1, len(u) - 1)]
        pos[n] = j + ((mid - a) / (b - a) if b != a else 0.5)
    return pos, pol, sw[ch], valid


def _pw(t, x, y):
    A = np.polyfit(x, y, 1)
    return np.polyval(A, t) + np.interp(t, x, y - np.polyval(A, x))


def run(folder):
    d = A_DIR / folder
    m = json.loads((d / "analog.json").read_text())
    tr = json.loads((d / "traces.json").read_text())
    B = np.load(d / "barcodes.npz")
    pe, ps = B["pe"], B["ps"]
    arrays = dict(bc_pe=pe, bc_ps=ps)
    fs = tr["fs"]
    for key, info in tr["blocks"].items():
        y = np.load(d / f"trace_{key}.npy", mmap_mode="r")
        pos, pol, sw, valid = detect(np.asarray(y) * info.get("scale", 1.0), fs)
        samp = info["sample0"] + pos * info["decim"] + (info["decim"] - 1) / 2  # OE sample number
        arrays[f"{key}_t"] = _pw(samp / 30000.0, pe, ps)
        arrays[f"{key}_pol"] = pol.astype(np.int8)
        arrays[f"{key}_swing"] = sw.astype(np.float32)
        bc = info["sample0"] + (np.arange(len(valid)) + 0.5) * 0.5 * fs * info["decim"]
        arrays[f"{key}_bins_t"] = _pw(bc / 30000.0, pe, ps)
        arrays[f"{key}_bins_valid"] = valid
        print(f"{folder} {key}: {len(pos)} transitions, {int((~valid).sum())} low-swing bins of {len(valid)}")
    np.savez(d / "analog.npz", **arrays)
    m["detect"] = dict(SCHMITT_HI=SCHMITT_HI, SCHMITT_LO=SCHMITT_LO, MID=MID, MIN_SWING=MIN_SWING)
    (d / "analog.json").write_text(json.dumps(m, indent=1))


if __name__ == "__main__":
    for f in sys.argv[1:]:
        run(f)
