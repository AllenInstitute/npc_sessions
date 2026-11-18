"""Why do box-1 replay spans fail the photodiode gate (experiments/sync-patch-manifest/
diode_gate_check.csv)? For every box-1 span:

1. reconcile the block's vsyncs (npc_sync.patch.solve_vsync_gaps), as the builder does
2. run the current npc_sync.patch.predict_diode_edges and find the measured edges after the span
   whose polarity disagrees with the nearest predicted toggle
3. align the measured edges outside the span to the reconciled frames with the prototype
   (phantom.align_box1) and say what each of those edges is: a phantom/glitch edge, or a real
   toggle (and of which frame, relative to the nearest held frame)
4. check the latency that predict_diode_edges paired edges with: its pairing rule ("latest
   toggle frame whose vsync is <= edge + 2 ms") is one frame late when the latency exceeds one
   refresh, as it does with box 1 (~20 ms vs 16.7 ms)

Output: diag_polarity.csv, diag_polarity.log
Run: PYTHONPATH="../../../npc_sync/src;../../../npc_stim/src;../../src;." ../../.venv/Scripts/python.exe diag_polarity.py > diag_polarity.log 2>&1
"""

from __future__ import annotations

import csv
import io
import logging

import numpy as np
import pandas as pd

import npc_sync
from npc_sync import patch as NP

import phantom as P
from common import manifest_common as mc

logging.disable(logging.WARNING)
MAN = "../sync-patch-manifest/"
gate = pd.read_csv(MAN + "diode_gate_check.csv")
rows = {r["session"]: r for r in csv.DictReader(open(MAN + "sessions.csv"))}
todo = gate[gate.box == 1]
out = []
for session, grp in todo.groupby("session"):
    row = rows[session]
    raw = npc_sync.SyncDataset(row["sync_path"], on_invalid="ignore")
    blocks = list(raw.vsync_times_in_blocks)
    if row["source"] == "s3":
        import npc_io

        stims = [mc.read_stim(io.BytesIO(npc_io.from_pathlike(p).read_bytes()), mc.sync_start(raw), name=npc_io.from_pathlike(p).stem)
                 for p in row["stim_paths"].split(";") if p]
    else:
        stims = [mc.read_stim(p, mc.sync_start(raw)) for p in row["stim_paths"].split(";") if p]
    matches = mc.match_stims_to_blocks(blocks, stims)[0]
    fs = raw.sample_freq
    spans = [(a - 1 / fs, z) for a, z in raw.invalid_intervals]
    dr = raw.get_rising_edges("stim_photodiode", units="seconds")
    df = raw.get_falling_edges("stim_photodiode", units="seconds")
    e_all = np.concatenate([dr, df])
    q_all = np.concatenate([np.ones(len(dr), bool), np.zeros(len(df), bool)])
    o = np.argsort(e_all, kind="stable")
    e_all, q_all = e_all[o], q_all[o]
    vr_all = raw.get_rising_edges("vsync_stim", units="seconds")
    for bi in sorted(set(grp.block_index.astype(int))):
        m = matches[bi]
        b = blocks[bi]
        ov = [(a, z) for a, z in spans if a <= b[-1] and z > b[0]]
        rec = NP.solve_vsync_gaps(b, m.stim.frame_intervals, ov)
        ft = rec.times
        # measured edges in the block, outside every replayed span
        sel = (e_all >= b[0]) & (e_all < b[-1] + 0.5) & ~NP.in_intervals(e_all, ov)
        e, q = e_all[sel], q_all[sel]
        jr = np.searchsorted(vr_all, ft) - 1
        vr = np.where((jr >= 0) & (rec.source == NP.MEASURED), vr_all[np.clip(jr, 0, None)], np.nan)
        try:
            al = P.align_box1(ft, m.stim.frame_intervals, e, q, vsync_rising=vr)
        except P.OffsetUndetermined as exc:
            al = None
            al_err = str(exc)
        for g in rec.gaps:
            a, z = g.span
            rise, fall, sol = NP.predict_diode_edges(ft, dr, df, g.span, 1)
            r = dict(session=session, block=bi, stim=m.stim.name, span=f"{a:.3f}-{z:.3f}", status=sol.status,
                     old_failures="; ".join(sol.failures()), old_lat_r_ms=sol.latency_rising_sec * 1e3,
                     old_lat_f_ms=sol.latency_falling_sec * 1e3, old_baseline_ms=sol.baseline_error_sec * 1e3,
                     old_after_ms=sol.after_error_sec * 1e3)
            if al is None:
                r["new"] = f"alignment failed: {al_err}"
                out.append(r)
                continue
            lr = float(np.nanmedian(al.latency_rising) * 1e3)
            lf = float(np.nanmedian(al.latency_falling) * 1e3)
            r.update(new_lat_r_ms=lr, new_lat_f_ms=lf, refresh_ms=al.refresh_sec * 1e3)
            # wrong-polarity edges of the old check (replicate predict_diode_edges' post check)
            if sol.status == "reconstructed":
                tt = ft[np.isfinite(ft)]
                pre = (e < a) & (e >= a - NP.DIODE_LATENCY_WINDOW_SEC)
                j_pre = np.searchsorted(tt, e[pre] + 0.002, side="right") - 1
                k0, p0 = int(j_pre[-1]), bool(q[pre][-1])
                tog_r = ((np.arange(len(tt)) - k0) % 2 == 0) == p0
                pred = tt + np.where(tog_r, sol.latency_rising_sec, sol.latency_falling_sec)
                oo = np.argsort(pred)
                pred, pr = pred[oo], tog_r[oo]
                post = np.flatnonzero((e >= z) & (e < z + NP.DIODE_LATENCY_WINDOW_SEC))
                jn = NP._nearest_index(e[post], pred)
                bad = post[pr[jn] != q[post]]
                kinds = []
                for i in bad:
                    c = P.CLASS_NAMES[al.edge_class[i]]
                    k = al.edge_frame[i]
                    if c == "toggle":
                        holds = np.flatnonzero(al.n_refreshes[: k + 1] >= 2)
                        prev_hold = int(holds[-1]) if len(holds) else -10**9
                        kinds.append(f"toggle of frame k, k - last hold = {k - prev_hold} (n={al.n_refreshes[prev_hold] if len(holds) else '-'}) {(e[i] - z):.2f}s after span")
                    else:
                        kinds.append(f"{c} ({(e[i] - z):.2f}s after span)")
                r["wrong_polarity_edges"] = " | ".join(kinds)
                # how many holds in the pre/post windows (the pairing rule breaks at holds)
                ks_pre = np.flatnonzero((ft[:-1] >= a - NP.DIODE_LATENCY_WINDOW_SEC) & (ft[:-1] < a) & (al.n_refreshes >= 2))
                ks_post = np.flatnonzero((ft[:-1] >= z) & (ft[:-1] < z + NP.DIODE_LATENCY_WINDOW_SEC) & (al.n_refreshes >= 2))
                r["holds_pre_3s"], r["holds_post_3s"] = len(ks_pre), len(ks_post)
            # classes of measured edges near the span
            near = (e >= a - 3) & (e < z + 3)
            r["phantom_near"] = int((al.edge_class[near] == P.PHANTOM).sum())
            r["glitch_near"] = int((al.edge_class[near] == P.GLITCH).sum())
            r["unexplained_near"] = int((al.edge_class[near] == P.UNEXPLAINED).sum())
            out.append(r)
d = pd.DataFrame(out)
d.to_csv("diag_polarity.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 200)
print(f"box-1 spans: {len(d)}")
print("old latency (ms, rising/falling) vs aligned latency:")
print(d[["session", "span", "old_lat_r_ms", "old_lat_f_ms", "new_lat_r_ms", "new_lat_f_ms"]].round(2).to_string())
pol = d[d.old_failures.str.contains("wrong polarity", na=False)]
print(f"\nspans failing on polarity: {len(pol)}")
print(pol[["session", "span", "holds_pre_3s", "holds_post_3s", "phantom_near", "glitch_near", "unexplained_near", "wrong_polarity_edges"]].to_string())
