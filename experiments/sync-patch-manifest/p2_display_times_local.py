"""P2 acceptance: npc_stim display times with a patched sync file (npc_stim branch `sync-patch`).

Checks (defined before running):
  B1 unpatched SyncDataset: get_display_times still raises (AssertionError) for files with
     invalid intervals (unchanged behaviour); drop file 840160 still raises (discontinuity).
  B2 patched: every block gets the same source as on the clean-block baseline, and blocks
     that overlap a reconstructed span get source 'photodiode' with n display times == stim
     frames.
  B3 blocks not overlapping any span: display times byte-identical to the unpatched data
     with the fault check bypassed (on_invalid='ignore').
  B4 frame_source marks exactly the frames whose vsync was interpolated, plus frames whose
     display time lies in a reconstructed diode span (+0.1 s).
  B5 display times of diode-toggle frames inside the window vs the NI-DAQ analog diode
     (per-polarity lag from the 3 s before the window): window max error <= pre max + 0.5 ms.

Run: PYTHONPATH="../../../npc_sync/src;../../../npc_stim/src" ../../.venv/Scripts/python.exe p2_display_times_local.py > p2_display_times_local.log 2>&1
"""

from __future__ import annotations

import logging

import numpy as np

import npc_stim
import npc_stim.display_times as DT
import npc_sync
from npc_sync import patch as P

from common import LOCAL7, match_stims_to_blocks, read_stim, sync_start

logging.disable(logging.WARNING)
print("npc_stim", npc_stim.__file__, "| npc_sync", npc_sync.__file__)
OK = {True: "PASS", False: "FAIL"}
results: dict[str, list[bool]] = {}


def check(name, cond, msg):
    results.setdefault(name, []).append(bool(cond))
    print(f"  {name} {OK[bool(cond)]}: {msg}")


for d in sorted(p for p in LOCAL7.iterdir() if p.is_dir()):
    sp = next(d.glob("*T*.h5"))
    stim_paths = sorted(d.glob("*.hdf5"))
    raw = npc_sync.SyncDataset(sp, on_invalid="ignore")
    inv = raw.invalid_intervals
    print(f"\n===== {d.name}: invalid={inv} disc={raw.timebase_discontinuities}")
    try:
        npc_stim.get_display_times(npc_sync.SyncDataset(sp), stim_paths)
        check("B1", False, "unpatched: no error")
    except AssertionError as exc:
        check("B1", True, f"unpatched raises: {str(exc)[:120]}")
    if not inv:
        continue
    blocks = list(raw.vsync_times_in_blocks)
    matches, _ = match_stims_to_blocks(blocks, [read_stim(p, sync_start(raw)) for p in stim_paths])
    stim_blocks = [P.StimBlock(m.block_index, m.stim.frame_intervals, m.stim.diode_box_frame_interval, m.stim.name)
                   for m in matches if m.stim is not None]
    patch = P.build_sync_patch(raw, stim_blocks, builder="p2_display_times_local")
    patched = npc_sync.SyncDataset(sp, patch=patch)
    res = npc_stim.get_display_times(patched, stim_paths)
    # baseline: same file with the fault check bypassed (copies included)
    orig = DT.assert_sync_frame_timing_is_valid
    DT.assert_sync_frame_timing_is_valid = lambda s: None
    try:
        base = npc_stim.get_display_times(npc_sync.SyncDataset(sp, on_invalid="ignore"), stim_paths)
    finally:
        DT.assert_sync_frame_timing_is_valid = orig
    spans = [s.span for s in patch.lines["vsync_stim"]] + [s.span for s in patch.lines["stim_photodiode"]]
    for r, b in zip(res, base):
        v = r.vsync_times
        ov = any(a <= v[-1] + 0.1 and z >= v[0] for a, z in spans)
        n_stim = next((m.stim.n_frames for m in matches if m.block_index == r.block_index and m.stim), None)
        if not ov:
            same = (r.source == b.source) and (
                (r.display_times is None and b.display_times is None)
                or np.array_equal(r.display_times, b.display_times)
            )
            check("B3", same and (r.frame_source == 0).all(), f"block {r.block_index} (no span): source {r.source}, identical to baseline={same}")
            continue
        check("B2", r.source == "photodiode" and len(r.display_times) == n_stim,
              f"block {r.block_index} (patched): source {r.source} {r.reason or ''}; n={None if r.display_times is None else len(r.display_times)} stim={n_stim}; baseline source {b.source} ({(b.reason or '')[:60]})")
        if r.display_times is None:
            continue
        vsrc = patched.get_edge_sources("vsync_stim", "falling")
        allv = patched.get_falling_edges("vsync_stim", units="seconds")
        exp = vsrc[np.searchsorted(allv, v)] == 1
        dspans = [(s.span[0], s.span[1] + 0.1) for s in patch.lines["stim_photodiode"] if s.status == "reconstructed"]
        exp |= P.in_intervals(r.display_times - DT.MONITOR_CENTER_REFRESH_TIME, dspans)
        check("B4", np.array_equal(r.frame_source == 1, exp),
              f"frame_source: {r.n_interpolated} interpolated frames (vsync interpolated: {int((vsrc[np.searchsorted(allv, v)] == 1).sum())})")
        an = LOCAL7.parent / f"analog_{d.name}.npz"
        if not an.exists():
            print("  B5 not run: no analog npz")
            continue
        ta = np.sort(np.load(an)["t_an_b"])
        box = r.diode_box_frame_interval
        tog = r.display_times[::box]
        pol = np.arange(len(tog)) % 2 == 0  # alternating; sign is arbitrary, per-polarity lag fixes it
        for s in patch.lines["stim_photodiode"]:
            if s.status != "reconstructed" or s.method != "vsync+latency":
                continue
            w0, w1 = s.span

            def pairs(lo, hi):
                x = ta[(ta >= lo) & (ta < hi)]
                j = np.clip(np.searchsorted(tog, x), 1, len(tog) - 1)
                j = np.where(np.abs(x - tog[j - 1]) < np.abs(tog[j] - x), j - 1, j)
                return tog[j] - x, pol[j]

            dd, pp = pairs(w0 - 3, w0)
            lag = {True: np.median(dd[pp]), False: np.median(dd[~pp])}

            def err(lo, hi):
                dd, pp = pairs(lo, hi)
                return np.abs(dd - np.where(pp, lag[True], lag[False])) if len(dd) else np.array([np.nan])

            pre, win, post = (err(*x) for x in ((w0 - 3, w0), (w0, w1), (w1, w1 + 2.5)))
            check("B5", np.nanmax(win) <= np.nanmax(pre) + 0.5e-3,
                  f"toggle-frame display times - analog: pre max {np.nanmax(pre) * 1e3:.2f} ms (n={len(pre)}), window max {np.nanmax(win) * 1e3:.2f} ms (n={len(win)}), post max {np.nanmax(post) * 1e3:.2f} ms (n={len(post)})")

print("\n===== summary")
for k in sorted(results):
    print(f"{k}: {sum(results[k])}/{len(results[k])} pass")
