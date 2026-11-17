"""Issue 4: where are the extra flips in the small-excess DR1 blocks and the OptoTagging blocks?
'True extras' = frames with >=2 flips whose neighbours don't have 0 flips. Output: flip_extra.log"""
import numpy as np, pandas as pd
import npc_stim.display_times as dt
from common import sync
df = pd.read_csv("flip_count.csv")
sel = df[(df.excess > 0)]
out = []
for (folder, blk), g in sel.groupby(["folder", "block"]):
    s = sync(folder); v = s.vsync_times_in_blocks[blk]
    r, fa = dt.get_diode_edges_in_blocks(s); a, b = r[blk], fa[blk]
    lo, hi = v[0] + 0.015, v[-1] + 5 / 59.95
    f = np.sort(np.concatenate([a[(a > lo) & (a < hi)], b[(b > lo) & (b < hi)]]))
    f, _, _ = dt.remove_anomalous_diode_flips(f, 1, True)
    c = np.histogram(f, bins=np.append(v + 0.015, np.inf))[0]
    nb0 = np.convolve(c == 0, [1, 1, 1], "same") > 0
    extra = np.where((c >= 2) & ~nb0)[0]
    dv = np.diff(v, append=np.nan) * 59.95
    out.append(dict(folder=folder, block=blk, stim=g.stim.iloc[0], excess=g.excess.iloc[0],
        n_true_extra_frames=len(extra), extra_flips_in_them=int((c[extra] - 1).sum()), max_flips_in_frame=int(c.max()),
        vsync_interval_frames_at_extras_median=float(np.nanmedian(dv[extra])) if len(extra) else np.nan,
        frac_extras_at_long_vsync=float(np.mean(dv[extra] > 1.5)) if len(extra) else np.nan,
        example_frames=extra[:3].tolist()))
o = pd.DataFrame(out); pd.set_option("display.width", 250); print(o.to_string(index=False))
o.to_csv("flip_extra.csv", index=False)
