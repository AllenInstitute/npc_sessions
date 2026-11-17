"""Extract NI-DAQ analog photodiode transitions for one session and map them to sync time.
Read-only on S3 and the cache.

    C:/data/t2p_qc/venv_analog/Scripts/python fetch_analog.py <folder> [--force] [--confirm-only]

Needs C:/data/t2p_qc/analog/<folder>/sync.npz from prep_sync.py. Steps:
1. Locate the session on S3 (locate.py) and its NI-DAQ zarr (first record node) + clipped TTL events.
2. Barcodes from the NI-DAQ TTL line that best matches the sync barcodes -> piecewise-linear map
   NI-DAQ sample number -> sync seconds (linear between consecutive shared barcodes).
3. Channel confirmation: in a 30 s window in the middle of every block, detect transitions on all
   8 AI channels and match them to the sync digital photodiode edges (TTL - analog lag in
   [LAG_LO, LAG_HI]). A channel is confirmed for a block if >= 95 % of sync edges and >= 95 % of its
   transitions match; the session channel must be the only channel confirmed in any block, and be
   confirmed in at least half of the blocks.
4. Save the confirmed channel over every block span (first vsync - 1 s, last + 1 s) as a 5 kHz trace
   (trace_b{i}.npy, mean of 6 samples), then run detect.py on it.
Writes C:/data/t2p_qc/analog/<folder>/{analog.json, barcodes.npz, traces.json, trace_b*.npy, analog.npz}.
"""
from __future__ import annotations
import json, pathlib, sys, time
import numpy as np, s3fs, zarr, wavpack_numcodecs  # noqa: F401

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
from locate import find_session, nidaq_info  # noqa: E402
from npc_ephys_barcodes import extract_barcodes_from_times  # noqa: E402

OUT = pathlib.Path("C:/data/t2p_qc/analog")
FS = 30000.0
LAG_LO, LAG_HI = -0.005, 0.012  # accepted (sync TTL edge - analog transition), s
MIN_SWING = 100  # counts; below this the channel is treated as having no signal
DECIM = 6  # saved traces: mean of 6 samples -> 5 kHz
s3 = s3fs.S3FileSystem()


def npy_first_last(path):
    """First and last element of a 1-D .npy on S3 without downloading it."""
    with s3.open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        rd = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, _, dtype = rd(f)
        off = f.tell()
        first = np.frombuffer(f.read(dtype.itemsize), dtype)[0]
        f.seek(off + (shape[0] - 1) * dtype.itemsize)
        last = np.frombuffer(f.read(dtype.itemsize), dtype)[0]
    return int(first), int(last), int(shape[0])


def load_npy(path):
    with s3.open(path, "rb") as f:
        return np.load(f)


from detect import detect  # noqa: E402


# --- barcode map -------------------------------------------------------------------------- #
def barcode_map(ttl_states, ttl_sn, bc_t_sync, bc_v_sync):
    best = None
    for line in np.unique(np.abs(ttl_states)):
        on = ttl_sn[ttl_states == line] / FS
        off = ttl_sn[ttl_states == -line] / FS
        if len(on) < 20 or len(off) < 20:
            continue
        t, v = extract_barcodes_from_times(on, off)
        common, i_s, i_e = np.intersect1d(bc_v_sync, v, return_indices=True)
        if best is None or len(common) > best[0]:
            best = (len(common), int(line), np.asarray(bc_t_sync)[i_s], np.asarray(t)[i_e])
    n, line, ps, pe = best
    o = np.argsort(pe)
    return line, pe[o], ps[o]  # NI-DAQ seconds (sample_number / FS), sync seconds


def _pw(t, x, y, A):
    """Piecewise linear between shared barcodes (x -> y); global linear fit A outside."""
    t = np.asarray(t, float)
    return np.polyval(A, t) + np.interp(t, x, y - np.polyval(A, x))


# --- main --------------------------------------------------------------------------------- #
def main(folder, force=False, confirm_only=False):
    d = OUT / folder
    if (d / "traces.json").exists() and not force and not confirm_only:
        print(f"{folder}: exists"); return
    S = np.load(d / "sync.npz")
    blocks = [S[k] for k in sorted((k for k in S.files if k.startswith("block")), key=lambda k: int(k[5:]))]
    meta = dict(folder=folder)
    stem = pathlib.Path(str(S["sync_file"])).stem
    c = find_session(folder, stem)
    if not c:
        meta["status"] = "blocked: no ecephys upload on S3 within 10 min of the sync start"
        (d / "analog.json").write_text(json.dumps(meta, indent=1)); print(folder, meta["status"]); return
    _, bucket, prefix = c[0]
    comp, clipped = nidaq_info(bucket, prefix)
    if not comp:
        meta["status"] = "blocked: no NI-DAQ zarr on S3"
        (d / "analog.json").write_text(json.dumps(meta, indent=1)); print(folder, meta["status"]); return
    zpath = comp[0]
    import re as _re
    exp, node = _re.search(r"/(experiment\d+)_(Record Node \d+)#", zpath).groups()
    cl = [k for k in clipped if f"/{node}/{exp}/" in k]
    recs = sorted({k.split(f"/{exp}/")[1].split("/")[0] for k in cl if f"/{exp}/recording" in k})
    z = zarr.open(s3fs.S3Map(f"{bucket}/{zpath}", s3=s3), mode="r")
    segs = sorted(k for k in z.keys() if k.startswith("traces_seg"))
    meta.update(bucket=bucket, prefix=prefix, zarr=zpath, record_node=node, experiment=exp, recordings=recs, segments=segs,
                channel_ids=[str(x) for x in z["channel_ids"][:]])
    if len(segs) != 1 or len(recs) != 1:
        meta["status"] = f"blocked: {len(segs)} zarr segments / {len(recs)} clipped recordings (not handled)"
        (d / "analog.json").write_text(json.dumps(meta, indent=1)); print(folder, meta["status"]); return
    rec = recs[0]
    base = next(k for k in cl if k.endswith(f"{rec}/continuous/" + zpath.split('#')[1].replace('.zarr/', '') + "/sample_numbers.npy"))
    first, last, n_sn = npy_first_last(f"{bucket}/{base}")
    tr = z[segs[0]]
    meta.update(first_sample=first, last_sample=last, n_sample_numbers=n_sn, n_zarr=int(tr.shape[0]),
                contiguous=bool(last - first + 1 == n_sn == tr.shape[0]))
    ttl_dir = base.replace("/continuous/", "/events/").replace("sample_numbers.npy", "TTL/")
    states = load_npy(f"{bucket}/{ttl_dir}states.npy").astype(int)
    sn = load_npy(f"{bucket}/{ttl_dir}sample_numbers.npy").astype(np.int64)
    line, pe, ps = barcode_map(states, sn, S["bc_t"], S["bc_v"])
    A = np.polyfit(pe, ps, 1)
    res = ps - np.polyval(A, pe)
    meta.update(barcode_ttl_line=line, n_shared_barcodes=len(pe), global_fit=A.tolist(),
                global_resid_ms=[float(res.min() * 1e3), float(res.max() * 1e3)],
                max_resid_step_ms=float(np.max(np.abs(np.diff(res))) * 1e3),
                barcode_sync_range=[float(ps.min()), float(ps.max())])
    print(f"{folder}: {bucket}/{zpath} contiguous={meta['contiguous']} line={line} shared barcodes={len(pe)} "
          f"global resid {meta['global_resid_ms'][0]:.2f}..{meta['global_resid_ms'][1]:.2f} ms, max step {meta['max_resid_step_ms']:.2f} ms", flush=True)
    to_sync = lambda t_ni: _pw(t_ni, pe, ps, A)  # noqa: E731
    to_ni = lambda t_s: _pw(t_s, ps, pe, np.polyfit(ps, pe, 1))  # noqa: E731
    def idx_of(t_sync):
        return int(round(to_ni(t_sync) * FS)) - first

    dig = np.sort(np.r_[S["d_rise"], S["d_fall"]])
    # 3. channel confirmation
    conf = []
    for bi, v in enumerate(blocks):
        mid = 0.5 * (v[0] + v[-1]); w0, w1 = max(v[0], mid - 15), min(v[-1], mid + 15)
        i0, i1 = idx_of(w0), idx_of(w1)
        if i0 < 0 or i1 > tr.shape[0]:
            conf.append(dict(block=bi, status="outside NI-DAQ recording")); continue
        x = np.asarray(tr[i0:i1])
        dg = dig[(dig > w0 + 0.05) & (dig < w1 - 0.05)]
        row = dict(block=bi, window=[float(w0), float(w1)], n_sync_edges=int(len(dg)), channels=[])
        for ch in range(x.shape[1]):
            if np.ptp(x[:, ch]) < MIN_SWING:
                row["channels"].append(dict(ch=ch, n=0, frac_sync_matched=0.0, frac_analog_matched=0.0)); continue
            pos, pol, sw, _ = detect(x[:, ch], FS)
            ta = to_sync((first + i0 + pos) / FS)
            ta = ta[(ta > w0 + 0.05) & (ta < w1 - 0.05)]
            if not len(ta) or not len(dg):
                row["channels"].append(dict(ch=ch, n=int(len(ta)), frac_sync_matched=0.0, frac_analog_matched=0.0)); continue
            j = np.clip(np.searchsorted(ta, dg), 1, len(ta) - 1)
            near = np.where(np.abs(dg - ta[j - 1]) < np.abs(dg - ta[j]), ta[j - 1], ta[j])
            lag_s = dg - near
            ok_s = (lag_s > LAG_LO) & (lag_s < LAG_HI)
            k = np.clip(np.searchsorted(dg, ta), 1, len(dg) - 1)
            nd = np.where(np.abs(ta - dg[k - 1]) < np.abs(ta - dg[k]), dg[k - 1], dg[k])
            lag_a = nd - ta
            ok_a = (lag_a > LAG_LO) & (lag_a < LAG_HI)
            row["channels"].append(dict(ch=ch, n=int(len(ta)), frac_sync_matched=float(ok_s.mean()),
                                        frac_analog_matched=float(ok_a.mean()),
                                        lag_median_ms=float(np.median(lag_s[ok_s]) * 1e3) if ok_s.any() else None,
                                        lag_p1_p99_ms=[float(np.percentile(lag_s[ok_s], 1) * 1e3), float(np.percentile(lag_s[ok_s], 99) * 1e3)] if ok_s.any() else None))
        good = [c["ch"] for c in row["channels"] if c["frac_sync_matched"] >= 0.95 and c["frac_analog_matched"] >= 0.95]
        row["confirmed"] = good
        conf.append(row)
        print(f"  block {bi}: sync edges={len(dg)} confirmed={good} " +
              " ".join(f"AI{c['ch']}:{c['frac_sync_matched']:.2f}/{c['frac_analog_matched']:.2f}" for c in row["channels"]), flush=True)
    meta["channel_confirmation"] = conf
    # the wiring can't change within a recording: the session channel is the only channel confirmed in
    # any block, and it must be confirmed in at least half of the blocks. Blocks where it isn't
    # confirmed (e.g. TTL edges without an analog transition) are listed and reported.
    confirmed = [set(r["confirmed"]) for r in conf if "confirmed" in r]
    chans = set.union(*confirmed) if confirmed else set()
    unconf = [r["block"] for r in conf if "confirmed" in r and len(r["confirmed"]) != 1]
    if len(chans) == 1 and sum(len(c) == 1 for c in confirmed) < 0.5 * len(confirmed):
        chans = set()
    if len(chans) != 1:
        meta["status"] = f"blocked: photodiode channel not uniquely confirmed (per block: {[r.get('confirmed') for r in conf]})"
        (d / "analog.json").write_text(json.dumps(meta, indent=1)); print(folder, meta["status"]); return
    ch = chans.pop()
    meta.update(channel=ch, blocks_not_uniquely_confirmed=unconf)
    if confirm_only:  # re-run the channel confirmation with the current detector; keep everything else
        old = json.loads((d / "analog.json").read_text())
        assert old.get("channel") == ch, f"channel changed: {old.get('channel')} -> {ch}"
        old.update(channel_confirmation=conf, blocks_not_uniquely_confirmed=unconf)
        (d / "analog.json").write_text(json.dumps(old, indent=1))
        return
    # 4. save the confirmed channel over every block span as a 5 kHz trace (mean of DECIM samples)
    np.savez(d / "barcodes.npz", pe=pe, ps=ps)
    traces = dict(fs=FS / DECIM, channel=ch, blocks={})
    t0 = time.time()
    for bi, v in enumerate(blocks):
        i0, i1 = max(idx_of(v[0] - 1.0), 0), min(idx_of(v[-1] + 1.0), tr.shape[0])
        if i1 - i0 < FS:
            continue
        i1 = i0 + (i1 - i0) // DECIM * DECIM
        parts = []
        step = 60 * 30000
        for a in range(i0, i1, step):
            x = np.asarray(tr[a:min(a + step, i1), ch]).astype(np.float32)
            parts.append(x.reshape(-1, DECIM).mean(1))
        y = np.concatenate(parts)
        np.save(d / f"trace_b{bi}.npy", np.round(y * 4).astype(np.int32))  # 0.25-count resolution
        traces["blocks"][f"b{bi}"] = dict(sample0=int(first + i0), decim=DECIM, n=int(len(y)), scale=0.25)
        print(f"  block {bi}: saved {(i1 - i0) / FS:.0f} s [{time.time() - t0:.0f} s]", flush=True)
    (d / "traces.json").write_text(json.dumps(traces, indent=1))
    meta["status"] = "ok"
    (d / "analog.json").write_text(json.dumps(meta, indent=1))
    import detect as _det
    _det.run(folder)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    for f in args:
        try:
            main(f, force="--force" in sys.argv, confirm_only="--confirm-only" in sys.argv)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"{f}: ERROR {type(e).__name__}: {e}")
