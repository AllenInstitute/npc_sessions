"""Offline test of the candidate §4 rule on the sync TTL diode edges, scored against the analog truth.

    C:/data/t2p_qc/venv_analog/Scripts/python rule_sim.py <folder> <block> [--min-refreshes 1.5]

Candidate rule ("keep the first flip in a long vsync interval when the next window has its own flip"):
window of frame f = [vf[f] + 8 ms, vf[f + 1] + 8 ms) on the vsync falling edges. If vf[f + 1] - vf[f]
>= min_refreshes frames and the window holds > 1 TTL edge, keep the first and drop the rest, provided
the next window holds >= 1 edge.

Truth: analog transitions matched to toggle frames by compare.py (results/<folder>_frames.npz); analog
transitions left unmatched there are screen changes that are not frames (stall flicker).
Each TTL edge is labelled by the analog transition nearest to it (TTL - analog in [-12, +15] ms):
'frame' (matched analog), 'flicker' (unmatched analog) or 'none'.
Prints counts before/after the rule and whether the kept edges are exactly the frame edges.
"""
import pathlib, sys
import numpy as np

HERE = pathlib.Path(__file__).parent
A_DIR = pathlib.Path("C:/data/t2p_qc/analog")
FR = 1 / 59.95


def label_ttl(d, a_frame, a_flick):
    lab = np.full(len(d), "none", dtype=object)
    allA = np.r_[a_frame, a_flick]
    kind = np.r_[np.zeros(len(a_frame)), np.ones(len(a_flick))]
    o = np.argsort(allA); allA, kind = allA[o], kind[o]
    j = np.clip(np.searchsorted(allA, d), 1, len(allA) - 1)
    j = np.where(np.abs(d - allA[j - 1]) <= np.abs(d - allA[j]), j - 1, j)
    lag = d - allA[j]
    ok = (lag > -0.012) & (lag < 0.015)
    lab[ok & (kind[j] == 0)] = "frame"
    lab[ok & (kind[j] == 1)] = "flicker"
    return lab


def apply_rule(d, vf, min_refreshes, offset=0.008, pairs_only=False):
    edges = np.r_[vf, vf[-1] + 10] + offset
    w = np.searchsorted(edges, d, side="right") - 1
    n_in = np.bincount(np.clip(w, 0, len(vf) - 1), minlength=len(vf))
    long_v = np.r_[np.diff(vf) >= min_refreshes * FR, False]
    first = np.r_[True, w[1:] != w[:-1]]
    nxt = np.r_[n_in[1:], 0]
    wc = np.clip(w, 0, len(vf) - 1)
    drop = (w >= 0) & (w < len(vf)) & ~first & long_v[wc] & (nxt[wc] >= 1)
    if pairs_only:  # keep polarity alternation: in each window drop an even number (the last extra kept)
        for win in np.unique(w[drop]):
            idx = np.where(drop & (w == win))[0]
            if len(idx) % 2:
                drop[idx[-1]] = False
    return ~drop


def main(folder, block, min_refreshes=1.5, offset=0.008, pairs_only=False, quiet=False):
    S = np.load(A_DIR / folder / "sync.npz")
    z = np.load(HERE / "results" / f"{folder}_frames.npz")
    v = S[f"block{block}"]
    a_frame = z[f"b{block}_a"]
    a_raw, unm = z[f"b{block}_a_raw"], z[f"b{block}_unmatched"]
    a_flick = a_raw[unm]
    dig = np.sort(np.r_[S["d_rise"], S["d_fall"]])
    d = dig[(dig > v[0] + 0.005) & (dig < v[-1] + 0.25)]
    lab = label_ttl(d, a_frame, a_flick)
    keep = apply_rule(d, v, min_refreshes, offset, pairs_only)
    def counts(m):
        return {k: int(np.sum(lab[m] == k)) for k in ("frame", "flicker", "none")}
    before, after, dropped = counts(np.ones(len(d), bool)), counts(keep), counts(~keep)
    n_frames = len(a_frame)
    if quiet:
        return dict(before=before, after=after, dropped=dropped, n_v=len(v), kept=int(keep.sum()))
    print(f"{folder} b{block}: vsyncs={len(v)} analog frame transitions={n_frames} analog flicker={len(a_flick)} "
          f"TTL edges={len(d)}")
    print(f"  TTL before rule: {before} (total {len(d)}; excess over vsyncs {len(d) - len(v):+d})")
    print(f"  dropped by rule (min {min_refreshes} refreshes, window offset {offset * 1e3:.0f} ms, pairs_only={pairs_only}): {dropped}")
    print(f"  TTL after rule: {after} (total {int(keep.sum())}; excess over vsyncs {int(keep.sum()) - len(v):+d})")
    # where are flicker TTL edges that survive, and frame edges dropped?
    surv = np.where(keep & (lab == "flicker"))[0]
    lost = np.where(~keep & (lab == "frame"))[0]
    fi = lambda x: np.searchsorted(v, d[x] - 0.02)  # noqa: E731
    print(f"  surviving flicker edges at ~frames {fi(surv)[:15].tolist()}; dropped frame edges at ~frames {fi(lost)[:15].tolist()}")
    return dict(before=before, after=after, dropped=dropped)


def grid(folder, block):
    print(f"{folder} b{block}: rule grid (TTL edges dropped that are frames / flicker; kept total - vsyncs)")
    for mr in (1.5, 2.5):
        for off in (0.008, 0.012, 0.016):
            for po in (False, True):
                r = main(folder, block, mr, off, po, quiet=True)
                print(f"  min {mr} refr, offset {off * 1e3:.0f} ms, pairs_only={po}: dropped frame={r['dropped']['frame']} "
                      f"flicker={r['dropped']['flicker']} none={r['dropped']['none']}; kept-vsyncs={r['kept'] - r['n_v']:+d}; "
                      f"flicker kept={r['after']['flicker']}")


if __name__ == "__main__":
    if "--grid" in sys.argv:
        grid(sys.argv[1], int(sys.argv[2])); sys.exit()
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    mr = float(sys.argv[sys.argv.index("--min-refreshes") + 1]) if "--min-refreshes" in sys.argv else 1.5
    if "--min-refreshes" in sys.argv:
        args = [a for a in args if a != str(sys.argv[sys.argv.index("--min-refreshes") + 1])]
    main(args[0], int(args[1]), mr)
