"""Issues 6/7: vsync blocks on sync vs stim files (start time from file name, duration = frames / 60)."""
import datetime, re
import h5py, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
import npc_stim.stim
from common import sync, stim_paths, sync_path
sessions = ["DRpilot_660023_20230808", "DRpilot_726088_20240618", "DRpilot_716718_20240731"]
fig, axes = plt.subplots(len(sessions), 1, figsize=(12, 2.4 * len(sessions)))
for ax, f in zip(axes, sessions):
    s = sync(f)
    t0 = s.start_time
    dur = s.total_time if hasattr(s, "total_time") else None
    try:
        for i, v in enumerate(s.vsync_times_in_blocks):
            a = t0 + datetime.timedelta(seconds=float(v[0])); b = t0 + datetime.timedelta(seconds=float(v[-1]))
            ax.plot([a, b], [1, 1], lw=8, solid_capstyle="butt", c="C0")
            ax.text(a, 1.25, f"b{i}\n{len(v)}", fontsize=6)
    except AssertionError:
        ax.text(t0, 1.25, "vsync_times_in_blocks: AssertionError (stim_running edges unbalanced)", fontsize=7, color="r")
        L = s.get_falling_edges("vsync_stim", units="seconds")
        ax.plot([t0 + datetime.timedelta(seconds=float(L[0])), t0 + datetime.timedelta(seconds=float(L[-1]))], [1, 1], lw=8, c="C0", solid_capstyle="butt")
        sr, sf = s.get_rising_edges("stim_running", units="seconds"), s.get_falling_edges("stim_running", units="seconds")
        ax.text(t0 + datetime.timedelta(seconds=float(L[0])), 0.75, f"stim_running: {len(sr)} rising, {len(sf)} falling edges; first falling at {sf[0]:.1f} s, first rising at {sr[0] if len(sr) else np.nan:.1f} s", fontsize=7)
    ax.axvline(t0, c="k", lw=0.8); ax.text(t0, 0.3, f" sync start {t0:%H:%M:%S}", fontsize=7)
    for j, p in enumerate(stim_paths(f)):
        st = datetime.datetime.strptime(re.search(r"(\d{8}_\d{6})", p.stem).group(1), "%Y%m%d_%H%M%S")
        with h5py.File(p) as h:
            n = npc_stim.stim.get_total_stim_frames(h)
        ax.plot([st, st + datetime.timedelta(seconds=n / 60)], [0, 0], lw=8, solid_capstyle="butt", c="C1")
        ax.text(st, -0.45 - 0.2 * (j % 2), f"{p.stem.split('_')[0]}\n{n}", fontsize=6)
    ax.set_yticks([1, 0], ["sync vsync blocks", "stim files"]); ax.set_ylim(-1, 1.7); ax.set_title(f, fontsize=9)
fig.tight_layout(); fig.savefig("figs/unmatched_timelines.png", dpi=110)
