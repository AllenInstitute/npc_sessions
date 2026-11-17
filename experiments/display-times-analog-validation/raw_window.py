"""Plot the raw NI-DAQ photodiode trace around a frame, with sync vsync/diode edges, the analog
transitions, and baseline/branch display times (read-only).

    C:/data/t2p_qc/venv_analog/Scripts/python raw_window.py <folder> <block> <frame> [before_ms after_ms] [--out name.png]
"""
from __future__ import annotations
import json, pathlib, sys
import numpy as np, s3fs, zarr, wavpack_numcodecs  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
A_DIR = pathlib.Path("C:/data/t2p_qc/analog")
BASE = pathlib.Path("C:/data/t2p_qc/out/baseline")
FIX = pathlib.Path("C:/data/t2p_qc/out_issues/fixE")
FS = 30000.0
MCRT = 0.008
s3 = s3fs.S3FileSystem()


def _pw(t, x, y):
    A = np.polyfit(x, y, 1)
    return np.polyval(A, t) + np.interp(t, x, y - np.polyval(A, x))


def raw(folder, t0, t1):
    """Analog trace (sync time, counts) of the confirmed channel between sync times t0, t1."""
    m = json.loads((A_DIR / folder / "analog.json").read_text())
    An = np.load(A_DIR / folder / "analog.npz")
    pe, ps = An["bc_pe"], An["bc_ps"]
    z = zarr.open(s3fs.S3Map(f"{m['bucket']}/{m['zarr']}", s3=s3), mode="r")
    tr = z[m["segments"][0]]
    i0 = int(round(_pw(t0, ps, pe) * FS)) - m["first_sample"]
    i1 = int(round(_pw(t1, ps, pe) * FS)) - m["first_sample"]
    y = np.asarray(tr[i0:i1, m["channel"]]).astype(float)
    t = _pw((m["first_sample"] + np.arange(i0, i1)) / FS, pe, ps)
    return t, y


def plot(ax, folder, block, frame, before=0.12, after=0.15, title=None, show_legend=True):
    S = np.load(A_DIR / folder / "sync.npz")
    v = S[f"block{block}"]
    tc = v[frame]
    t0, t1 = tc - before, tc + after
    t, y = raw(folder, t0, t1)
    yn = (y - np.percentile(y, 1)) / (np.percentile(y, 99) - np.percentile(y, 1) + 1e-9)
    ax.plot((t - tc) * 1e3, yn * 1.0 + 3.2, color="k", lw=0.8, label="NI-DAQ analog diode")
    sel = lambda x: x[(x > t0) & (x < t1)]  # noqa: E731
    ax.vlines((sel(S["v_fall"]) - tc) * 1e3, 2.0, 2.9, color="C0", lw=1.4, label="vsync falling")
    ax.vlines((sel(S["v_rise"]) - tc) * 1e3, 2.0, 2.9, color="C0", lw=0.8, ls=":", label="vsync rising")
    ax.vlines((sel(S["d_rise"]) - tc) * 1e3, 1.0, 1.9, color="C2", lw=1.4, label="sync diode TTL rise")
    ax.vlines((sel(S["d_fall"]) - tc) * 1e3, 1.0, 1.9, color="C3", lw=1.4, label="sync diode TTL fall")
    An = np.load(A_DIR / folder / "analog.npz")
    if f"b{block}_t" in An.files:
        ta = sel(An[f"b{block}_t"])
        ax.vlines((ta - tc) * 1e3, 3.1, 4.3, color="m", lw=0.8, ls="--", label="analog transition")
    fi = np.arange(len(v))
    for lab, path, y0, c in (("baseline", BASE, 0.0, "C1"), ("branch", FIX, -1.0, "C4")):
        z = np.load(path / f"{folder}.npz")
        k = f"A_block{block}"
        if k not in z:
            ax.text(-before * 1e3, y0 + 0.4, f"{lab}: unavailable", fontsize=7); continue
        d = z[k] - MCRT
        w = (d > t0) & (d < t1)
        ax.vlines((d[w] - tc) * 1e3, y0, y0 + 0.9, color=c, lw=1.4, label=f"{lab} display - 8 ms")
        for f_, x in zip(fi[w], d[w]):
            if f_ % 1 == 0:
                ax.text((x - tc) * 1e3, y0 + 0.92, str(f_ - frame), fontsize=6, ha="center", color=c)
    for f_ in range(max(frame - 10, 0), min(frame + 12, len(v))):
        if t0 < v[f_] < t1:
            ax.text((v[f_] - tc) * 1e3, 2.95, str(f_ - frame), fontsize=6, ha="center", color="C0")
    ax.set_yticks([-0.5, 0.5, 1.5, 2.5, 3.7]); ax.set_yticklabels(["branch", "baseline", "sync diode", "vsync", "analog"], fontsize=7)
    ax.set_xlabel(f"ms from vsync falling edge of frame {frame}")
    ax.set_title(title or f"{folder} b{block} frame {frame}", fontsize=8)
    if show_legend:
        ax.legend(fontsize=6, loc="upper right", ncol=3)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    if out in args:
        args.remove(out)
    folder, block, frame = args[0], int(args[1]), int(args[2])
    before = float(args[3]) / 1e3 if len(args) > 3 else 0.12
    after = float(args[4]) / 1e3 if len(args) > 4 else 0.15
    fig, ax = plt.subplots(figsize=(13, 4.5))
    plot(ax, folder, block, frame, before, after)
    fig.tight_layout()
    p = HERE / "figs" / (out or f"raw_{folder[8:]}_b{block}_f{frame}.png")
    fig.savefig(p, dpi=110)
    print(p)
