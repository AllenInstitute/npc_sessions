"""Shared helpers for display-time issue analysis. Read-only on the cache."""
from __future__ import annotations
import functools, pathlib, re
import numpy as np

CACHE = pathlib.Path("C:/data/t2p_cache")
BASE = pathlib.Path("C:/data/t2p_qc/out/baseline")
FR = 1 / 59.95


def sync_path(folder: str) -> pathlib.Path:
    f = CACHE / folder
    syncs = sorted((p for p in f.iterdir() if re.match(r"^\d{8}T\d{6}\.h5$", p.name)), key=lambda p: p.stat().st_size)
    return syncs[-1]


def stim_paths(folder: str) -> list[pathlib.Path]:
    return sorted(p for p in (CACHE / folder).iterdir() if p.suffix == ".hdf5")


@functools.lru_cache(maxsize=4)
def sync(folder: str):
    import npc_sync
    return npc_sync.SyncDataset(sync_path(folder))


def lines(folder: str):
    s = sync(folder)
    return dict(
        v_rise=s.get_rising_edges("vsync_stim", units="seconds"),
        v_fall=s.get_falling_edges("vsync_stim", units="seconds"),
        d_rise=s.get_rising_edges("stim_photodiode", units="seconds"),
        d_fall=s.get_falling_edges("stim_photodiode", units="seconds"),
    )


def baseline_block(folder: str, block: int):
    z = np.load(BASE / f"{folder}.npz")
    d = z[f"A_block{block}"] if f"A_block{block}" in z else None
    return z[f"vsync_block{block}"], d


def plot_window(ax, folder, block, idx, before=6, after=8, title=None, display=None):
    """vsync (rise/fall), photodiode edges and estimated display times around frame idx."""
    L = lines(folder)
    v, d = baseline_block(folder, block)
    if display is not None:
        d = display
    t0 = v[idx]
    lo, hi = v[max(idx - before, 0)] - 0.01, v[min(idx + after, len(v) - 1)] + 0.06
    def sel(x):
        return x[(x > lo) & (x < hi)]
    for y, (k, c) in enumerate((("v_rise", "C0"), ("v_fall", "C0"))):
        pass
    # vsync as a square wave
    vr, vf = sel(L["v_rise"]), sel(L["v_fall"])
    ax.vlines((vf - t0) * 1e3, 2.0, 2.8, color="C0", lw=1.5, label="vsync falling (frame signal)")
    ax.vlines((vr - t0) * 1e3, 2.0, 2.8, color="C0", lw=0.8, ls=":", label="vsync rising")
    dr, df = sel(L["d_rise"]), sel(L["d_fall"])
    ax.vlines((dr - t0) * 1e3, 1.0, 1.8, color="C2", lw=1.5, label="diode rising")
    ax.vlines((df - t0) * 1e3, 1.0, 1.8, color="C3", lw=1.5, label="diode falling")
    if d is not None:
        sl = slice(max(idx - before, 0), min(idx + after, len(d)))
        ax.vlines((d[sl] - t0) * 1e3, 0.0, 0.8, color="k", lw=1.5, label="estimated display time")
        for j in range(sl.start, sl.stop):
            col = "r" if d[j] < v[j] else "0.6"
            ax.annotate("", xy=((d[j] - t0) * 1e3, 0.8), xytext=((v[j] - t0) * 1e3, 2.0),
                        arrowprops=dict(arrowstyle="->", color=col, lw=1.2 if col == "r" else 0.6))
    ax.set_yticks([0.4, 1.4, 2.4], ["display (est.)", "photodiode", "vsync"])
    ax.set_xlabel(f"time re vsync of frame {idx} (ms)")
    ax.set_title(title or f"{folder} block {block} frame {idx}", fontsize=8)
