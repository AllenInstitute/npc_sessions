"""Step 6 (validate), stage 1: for npc_lims ephys sessions from 2025-01-13 on with replay-
reconcilable blocks, build the session's SyncPatch (npc_sessions.timing) and save what stage 2
needs to compare against the NI-DAQ analog photodiode (AI0), around each replayed window only:

  C:/data/t2p_qc/analog_windows/<folder>/prep.npz
    spans (n, 2)                reconstructed photodiode spans
    d_t, d_pol, d_src           patched digital diode edges within 6 s of a span (pol 1 = rising;
                                src 1 = reconstructed)
    f_t, f_src                  display times (npc_stim, patched sync) of diode-toggle frames
                                within 6 s of a span, and their frame_source
    bc_t, bc_v                  sync barcodes (barcode line masked +-2 s around invalid intervals)
    sync_file                   sync file stem

Run (npc_sessions venv, branch libs):
  PYTHONPATH="../../../npc_sync/src;../../../npc_stim/src;../../src" ../../.venv/Scripts/python.exe analog_windows_prep.py > analog_windows_prep.log 2>&1
"""

from __future__ import annotations

import csv
import logging
import pathlib

import numpy as np
import pandas as pd

import npc_ephys
import npc_stim
import npc_sync
from npc_sessions import timing

logging.disable(logging.WARNING)
OUT = pathlib.Path("C:/data/t2p_qc/analog_windows")
W = 6.0
m = pd.read_csv("manifest.csv")
m["date"] = pd.to_datetime(m.session.str.extract(r"(\d{8})$")[0], format="%Y%m%d", errors="coerce")
sel = m[(m.cls_proposed == "replay-reconcilable") & (m.in_npc_lims == True) & (m.is_ephys == True) & (m.date >= "2025-01-13")]  # noqa: E712
rows = {r["session"]: r for r in csv.DictReader(open("sessions.csv"))}
print(f"{sel.session.nunique()} sessions")
for folder in sorted(sel.session.unique()):
    out = OUT / folder
    if (out / "prep.npz").exists():
        print(folder, "exists")
        continue
    r = rows[folder]
    try:
        stim_paths = [pathlib.Path(p) for p in r["stim_paths"].split(";") if p]
        raw = npc_sync.SyncDataset(r["sync_path"], on_invalid="ignore")
        patch = timing.build_session_sync_patch(raw, stim_paths)
        s = npc_sync.SyncDataset(raw, patch=patch)
        spans = [sp.span for sp in patch.lines.get("stim_photodiode", []) if sp.status == "reconstructed" and sp.method == "vsync+latency"]
        if not spans:
            print(folder, "no reconstructed photodiode span:", [(sp.span, sp.reason) for sp in patch.lines.get("stim_photodiode", [])])
            continue
        near = lambda t: npc_sync.in_intervals(t, [(a - W, z + W) for a, z in spans])  # noqa: E731
        dr, df = s.get_rising_edges("stim_photodiode", units="seconds"), s.get_falling_edges("stim_photodiode", units="seconds")
        sr, sf = s.get_edge_sources("stim_photodiode", "rising"), s.get_edge_sources("stim_photodiode", "falling")
        d_t = np.r_[dr, df]
        d_pol = np.r_[np.ones(len(dr), bool), np.zeros(len(df), bool)]
        d_src = np.r_[sr, sf]
        k = near(d_t)
        o = np.argsort(d_t[k])
        f_t, f_src = [], []
        for b in npc_stim.get_display_times(s, stim_paths):
            if b.display_times is None or not b.diode_box_frame_interval:
                continue
            t = b.display_times[:: b.diode_box_frame_interval]
            src = (b.frame_source if b.frame_source is not None else np.zeros(len(b.display_times), np.int8))[:: b.diode_box_frame_interval]
            kk = near(t)
            f_t.append(t[kk])
            f_src.append(src[kk])
        on = s.get_rising_edges("barcode_ephys", units="seconds")
        off = s.get_falling_edges("barcode_ephys", units="seconds")
        bc_t, bc_v = npc_ephys.barcodes.extract_barcodes_from_times(on_times=on, off_times=off, total_time_on_line=s.total_seconds)
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / "prep.npz", spans=np.array(spans), d_t=d_t[k][o], d_pol=d_pol[k][o], d_src=d_src[k][o],
                 f_t=np.concatenate(f_t) if f_t else np.array([]), f_src=np.concatenate(f_src) if f_src else np.array([]),
                 bc_t=bc_t, bc_v=bc_v, sync_file=pathlib.Path(r["sync_path"]).stem)
        print(folder, f"{len(spans)} spans, {int(k.sum())} diode edges, {sum(map(len, f_t))} toggle frames", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(folder, "ERROR", type(exc).__name__, exc, flush=True)
