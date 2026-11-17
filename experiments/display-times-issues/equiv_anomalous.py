"""Fix A check: new remove_anomalous_diode_flips == original O(n^2) loop.
1) fuzz on synthetic flip trains; 2) every block of every cached session, flips built exactly as
in get_display_times (after the vsync-window filter). Skips DRpilot_366122_20250109 for the
original loop (it doesn't finish) and reports the new function's runtime there instead.
Run with PYTHONPATH=C:/data/t2p_qc/wt/npc_stim_fix/src. Output: equiv_anomalous.log"""
import concurrent.futures, sys, time
import numpy as np, pandas as pd
import npc_stim.display_times as dt
assert "npc_stim_fix" in dt.__file__, dt.__file__

def original(flips, n, on):
    while any(dt.anomalous_interval_indices(flips, n)):
        indices = dt.anomalous_interval_indices(flips, n)
        flips = np.delete(flips, slice(indices[0], indices[0] + 2))
        if 0 in indices:
            on = not on
    return flips, on

def same(a, b):
    return a[0].dtype == b[0].dtype and np.array_equal(a[0], b[0]) and a[1] == b[1]

def fuzz(seed):
    rng = np.random.default_rng(seed)
    bad = 0
    for _ in range(300):
        n = int(rng.choice([1, 1, 3, 2]))
        k = int(rng.integers(3, 60))
        iv = np.full(k, n * dt.FRAME_INTERVAL) * rng.choice([1, 1, 1, 2, 3], k) + rng.normal(0, 0.002, k)
        f = np.cumsum(np.abs(iv))
        # insert blips (short pairs) incl. at start/end
        for _ in range(int(rng.integers(0, 6))):
            i = int(rng.integers(0, len(f)))
            w = rng.choice([0.0005, 0.002, 0.004, 0.01])
            f = np.sort(np.concatenate([f, [f[i] + w / 3, f[i] + w]]))
        on = bool(rng.integers(0, 2))
        new = dt.remove_anomalous_diode_flips(f, n, on)
        if not same(original(f, n, on), new[:2]):
            bad += 1
    return bad

def session(folder):
    from common import sync, stim_paths
    s = sync(folder)
    rows = []
    try:
        vb = s.vsync_times_in_blocks
        r, fa = dt.get_diode_edges_in_blocks(s)
        m = dt._get_block_to_stim_path(s, stim_paths(folder))
    except Exception as e:
        return [dict(folder=folder, error=type(e).__name__)]
    for i, (v, a, b) in enumerate(zip(vb, r, fa)):
        p = m.get(i)
        if p is None:
            continue
        n = dt.get_diode_box_frame_interval(p)
        a = a[(a > v[0] + dt.MIN_VSYNC_DIODE_FLIP_SEPARATION_SEC) & (a < v[-1] + dt.MAX_VSYNC_DIODE_FLIP_SEPARATION_SEC)]
        b = b[(b > v[0] + dt.MIN_VSYNC_DIODE_FLIP_SEPARATION_SEC) & (b < v[-1] + dt.MAX_VSYNC_DIODE_FLIP_SEPARATION_SEC)]
        if not len(a) or not len(b):
            continue
        on = bool(a[0] < b[0])
        f = np.sort(np.concatenate((a, b)))
        t = time.time(); new = dt.remove_anomalous_diode_flips(f, n, on); t_new = time.time() - t
        row = dict(folder=folder, block=i, box=n, n_flips=len(f), n_removed=new[2], t_new=t_new)
        if folder != "DRpilot_366122_20250109":
            t = time.time(); old = original(f, n, on); row["t_old"] = time.time() - t
            row["identical"] = same(old, new[:2])
        rows.append(row)
    return rows

if __name__ == "__main__":
    print("fuzz mismatches:", sum(fuzz(s) for s in range(10)), "of 3000 synthetic trains", flush=True)
    folders = sorted(pd.read_csv("../display-times-move/qc_blocks.csv").folder.unique())
    rows = []
    with concurrent.futures.ProcessPoolExecutor(int(sys.argv[1])) as ex:
        for r in ex.map(session, folders):
            rows += r
    df = pd.DataFrame(rows); df.to_csv("equiv_anomalous.csv", index=False)
    c = df[df.identical.notna()] if "identical" in df else df
    print("blocks compared:", len(c), "identical:", int(c.identical.sum()), "different:", int((~c.identical.astype(bool)).sum()))
    print("blocks with >=1 removal:", int((df.n_removed > 0).sum()), "| removals per block p50/p99/max (excl. 366122_20250109):",
          np.percentile(c.n_removed, [50, 99]).tolist(), int(c.n_removed.max()))
    print("fraction of intervals removed, max over compared blocks:", float((2 * c.n_removed / c.n_flips).max()))
    print("total time old (s):", round(c.t_old.sum(), 1), "new:", round(c.t_new.sum(), 1))
    print(df[df.folder == "DRpilot_366122_20250109"].to_string())
    print("session errors:", df.get("error", pd.Series(dtype=str)).value_counts().to_dict())
