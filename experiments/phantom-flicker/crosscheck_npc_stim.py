"""Cross-check the prototype frame <-> edge assignment against npc_stim's (released 0.1.24
behaviour, `fixE` run in C:/data/t2p_qc/out_issues/fixE) on every aligned box-1 block whose
npc_stim source is 'photodiode'.

npc_stim box 1: display_times[k] - MONITOR_CENTER_REFRESH_TIME is a raw diode edge for every
other frame ("on" flips) and an interpolated midpoint for the others. For every frame whose
npc_stim display time equals a raw edge (within 1 us), check that it is the edge the alignment
assigned to that frame. npc_stim's assignment was checked against the NI-DAQ analog photodiode
for 2025+ sessions (outputs/display-times-analog-validation.md: 0 misassigned toggles).

Output: crosscheck_npc_stim.csv, crosscheck_npc_stim.log
Run: PYTHONPATH="../../../npc_sync/src;../../../npc_stim/src;../../src;." ../../.venv/Scripts/python.exe crosscheck_npc_stim.py > crosscheck_npc_stim.log 2>&1
"""

from __future__ import annotations

import concurrent.futures as cf
import logging
import pathlib
import traceback

import numpy as np
import pandas as pd

FIXE = pathlib.Path("C:/data/t2p_qc/out_issues/fixE")
MONITOR_CENTER = 0.008


def run(folder: str) -> list[dict]:
    import common
    import phantom as P

    logging.disable(logging.WARNING)
    out = []
    try:
        z = np.load(FIXE / f"{folder}.npz")
        s = common.raw_sync(folder)
        blocks, matches = common.matches(folder)
        e_all, q_all = common.diode_edges(s)
        vr_all = s.get_rising_edges("vsync_stim", units="seconds")
        for bi in range(len(blocks)):
            key = f"A_block{bi}"
            m = matches[bi]
            if key not in z or m.stim is None or (m.stim.diode_box_frame_interval or 1) != 1:
                continue
            v = blocks[bi]
            if len(v) != m.stim.n_frames:
                continue
            D = z[key] - MONITOR_CENTER
            sel = (e_all >= v[0]) & (e_all < v[-1] + 0.5)
            e, q = e_all[sel], q_all[sel]
            jr = np.searchsorted(vr_all, v) - 1
            vr = np.where(jr >= 0, vr_all[np.clip(jr, 0, None)], np.nan)
            try:
                a = P.align_box1(v, m.stim.frame_intervals, e, q, vsync_rising=vr)
            except P.OffsetUndetermined:
                out.append(dict(folder=folder, block=bi, stim=m.stim.name, status="offset undetermined"))
                continue
            # frames whose npc_stim display time is a raw edge
            j = np.clip(np.searchsorted(e, D), 1, len(e) - 1)
            jj = np.where(np.abs(D - e[j - 1]) <= np.abs(e[j] - D), j - 1, j)
            is_edge = np.abs(e[jj] - D) < 1e-6
            mine = a.frame_edge
            both = is_edge & (mine >= 0)
            agree = both & (mine == jj)
            dis = np.flatnonzero(both & ~agree)
            out.append(dict(folder=folder, block=bi, stim=m.stim.name, status="ok", n_frames=len(v),
                            n_npc_stim_raw=int(is_edge.sum()), n_compared=int(both.sum()), n_agree=int(agree.sum()),
                            n_disagree=len(dis), disagree_frames=dis[:10].tolist(),
                            disagree_shift=(np.unique(np.round((e[jj[dis]] - e[mine[dis]]) / a.refresh_sec)).tolist() if len(dis) else []),
                            n_landmarks=a.n_landmarks, validated=a.validated))
    except Exception:  # noqa: BLE001
        out.append(dict(folder=folder, status="error: " + traceback.format_exc().splitlines()[-1]))
    return out


def main() -> None:
    import common

    folders = sorted(p.stem for p in FIXE.glob("*.npz"))
    folders = [f for f in folders if (common.CACHE / f).exists()]
    rows = []
    with cf.ProcessPoolExecutor(max_workers=10) as ex:
        for r in ex.map(run, folders):
            rows += r
    d = pd.DataFrame(rows)
    d.to_csv("crosscheck_npc_stim.csv", index=False)
    ok = d[d.status == "ok"]
    print(f"blocks compared: {len(ok)} in {ok.folder.nunique()} sessions; frames compared {int(ok.n_compared.sum())}, "
          f"agree {int(ok.n_agree.sum())}, disagree {int(ok.n_disagree.sum())}")
    print(f"blocks with any disagreement: {int((ok.n_disagree > 0).sum())}")
    pd.set_option("display.width", 250)
    print(ok[ok.n_disagree > 0][["folder", "block", "stim", "n_compared", "n_disagree", "disagree_frames", "disagree_shift", "n_landmarks", "validated"]].to_string())
    print("\nstatus counts:", d.status.str.slice(0, 40).value_counts().to_dict())
    lowlm = ok[ok.n_landmarks <= 3]
    print(f"\nblocks with <= 3 landmarks: {len(lowlm)}; disagreeing: {int((lowlm.n_disagree > 0).sum())}")


if __name__ == "__main__":
    main()
