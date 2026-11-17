"""P1 acceptance on the 7 local replay/drop sessions: does the library (`npc_sync.patch`, branch
`sync-patch`) reproduce the prototype (npc_sync/experiments/vsync-frame-count/repair.py)?

Checks (defined before running):
  A1 every block overlapping an invalid interval with a matched stim file reconciles (all gates
     pass); max |anchor residual| <= 0.05 ms (prototype D1: <= 0.05 ms).
  A2 for single-gap blocks, per-frame times equal the prototype `reconcile_replay` (max |diff|
     < 1 us) and the per-frame source counts match.
  A3 camera lines: every span reconstructed, max |gap residual| <= 0.032 ms (prototype
     camera_gap_check2.log), inserted count == copies count where the prototype found equality.
  A4 patched SyncDataset: vsync blocks have exactly the stim frame count for reconciled blocks;
     times equal the reconciliation; edge sources mark the reconstructed frames.
  A5 strict policy: unpatched SyncDataset raises SyncFaultError on edge queries for files with
     invalid intervals; patched one raises for no line, and reports unknown intervals for
     masked lines (lick_sensor, barcode_ephys).
  A6 JSON round trip of the patch is lossless (applied edges identical).
  A7 predicted photodiode edges inside the window vs the NI-DAQ analog diode (sessions with an
     analog npz): window error within the pre-window error + 0.5 ms.
  A8 840160 (drop, no invalid intervals): `timebase_discontinuities` flags it; locate_drop finds
     the 2472.0 s boundary; no patch lines (nothing to patch) and the file is not 'patchable'.

Run: ../../../npc_sync/.venv/Scripts/python.exe acceptance_local7.py > acceptance_local7.log 2>&1
"""

from __future__ import annotations

import logging
import sys

import numpy as np

import npc_sync
from npc_sync import patch as P

from common import LOCAL7, match_stims_to_blocks, read_stim, sync_start

sys.path.insert(0, str(LOCAL7.parent))
import repair  # noqa: E402  prototype

logging.disable(logging.WARNING)
OK = {True: "PASS", False: "FAIL"}
results: dict[str, list[bool]] = {}


def check(name: str, cond: bool, msg: str) -> None:
    results.setdefault(name, []).append(bool(cond))
    print(f"  {name} {OK[bool(cond)]}: {msg}")


for d in sorted(p for p in LOCAL7.iterdir() if p.is_dir()):
    sp = next(d.glob("*T*.h5"))
    raw = npc_sync.SyncDataset(sp, on_invalid="ignore")
    inv = raw.invalid_intervals
    print(f"\n===== {d.name}: invalid={[(round(a, 2), round(z, 2)) for a, z in inv]} "
          f"discontinuities={[(round(a, 2), round(z, 2), round(s, 3)) for a, z, s in raw.timebase_discontinuities]}")
    blocks = list(raw.vsync_times_in_blocks)
    stims = [read_stim(p, sync_start(raw)) for p in sorted(d.glob("*.hdf5"))]
    matches, unmatched = match_stims_to_blocks(blocks, stims)
    for m in matches:
        print(f"  block {m.block_index}: n_vsync={len(blocks[m.block_index])} "
              f"stim={m.stim.name if m.stim else None} n_stim={m.stim.n_frames if m.stim else None} "
              f"offset={m.offset_sec if m.offset_sec is None else round(m.offset_sec, 2)} {m.reason}")
    for m in unmatched:
        print(f"  unmatched stim {m.stim.name}: {m.reason}")

    if not inv:
        # A8
        check("A8", bool(raw.timebase_discontinuities), f"timebase_discontinuities={raw.timebase_discontinuities}")
        for m in matches:
            if m.stim and len(blocks[m.block_index]) < m.stim.n_frames:
                i0, n_missing, drop, cost = P.locate_drop(blocks[m.block_index], m.stim.frame_intervals)
                t = blocks[m.block_index][i0]
                check("A8", np.floor(t * 2) / 2 == 2472.0, f"drop located at vsync {i0} t={t:.4f} s, n_missing={n_missing}, drop={drop} s, cost={cost:.3f} ms")
        # strict policy doesn't apply (no invalid intervals)
        s = npc_sync.SyncDataset(sp)
        s.get_rising_edges("lick_sensor")
        continue

    stim_blocks = [
        P.StimBlock(m.block_index, m.stim.frame_intervals, m.stim.diode_box_frame_interval, m.stim.name)
        for m in matches if m.stim is not None
    ]
    for m in matches:
        if m.stim is None:
            continue
        b = blocks[m.block_index]
        ov = [(a, z) for a, z in inv if a <= b[-1] and z > b[0]]
        if not ov:
            check("A4", len(b) == m.stim.n_frames, f"clean block {m.block_index} count {len(b)} == {m.stim.n_frames}")
            continue
        rec = P.solve_vsync_gaps(b, m.stim.frame_intervals, ov)
        f = rec.failures()
        check("A1", not f and rec.max_abs_residual_sec <= 0.05e-3,
              f"block {m.block_index} gaps={len(rec.gaps)} balance={rec.balance} "
              f"residuals ms={[round(g.residual_sec * 1e3, 4) for g in rec.gaps]} "
              f"second-best ms={[round(g.second_best_sec * 1e3, 1) for g in rec.gaps]} "
              f"n_frames={[g.n_frames for g in rec.gaps]} n_removed={[g.n_removed for g in rec.gaps]} failures={f}")
        if len(ov) == 1:
            proto = repair.reconcile_replay(b, m.stim.frame_intervals, ov)
            diff = np.nanmax(np.abs(proto.times - rec.times))
            same_src = bool(np.array_equal(proto.source, rec.source))
            check("A2", diff < 1e-6 and same_src, f"vs prototype: max |diff|={diff * 1e6:.3f} us, same sources={same_src}")
        last_rec = (m, rec)

    patch = P.build_sync_patch(raw, stim_blocks, builder="acceptance_local7")
    print("  patch summary:", patch.summary())
    for line, spans in patch.lines.items():
        for sp_ in spans:
            if sp_.status == "unknown" and line in ("vsync_stim", "stim_photodiode") or line.endswith("_cam_exposing") and sp_.status == "unknown":
                print(f"    {line} {sp_.span}: unknown: {sp_.reason}")
    # A3 camera lines: exposing lines reproduce camera_gap_check2 (worst printed 0.032 ms, i.e.
    # < 0.0325 ms at its print precision); readout lines weren't in the prototype: reported only
    for k in [k for k in patch.lines if k.endswith(P.CAMERA_LINE_SUFFIXES)]:
        for sp_ in patch.lines[k]:
            dg = sp_.diagnostics
            r = max(abs(dg["rising"]["residual_sec"] or 0), abs(dg["falling"]["residual_sec"] or 0))
            msg = (f"{k:24s} {sp_.span}: {sp_.status} {sp_.method} inserted rising {dg['rising']['n_inserted']} "
                   f"(copies {dg['rising']['n_removed']}), max |resid| {r * 1e3:.4f} ms, "
                   f"frac {max(dg['rising']['fraction'] or 0, dg['falling']['fraction'] or 0):.4f} {sp_.reason}")
            if k.endswith("_cam_exposing"):
                check("A3", sp_.method == "periodic" and r < 0.0325e-3
                      and dg["rising"]["n_inserted"] == dg["rising"]["n_removed"], msg)
            else:
                print(f"  A3-readout (info): {msg}")

    # A5 strict policy
    strict = npc_sync.SyncDataset(sp)
    raised = []
    for line in ("vsync_stim", "lick_sensor", "barcode_ephys", "beh_cam_exposing"):
        try:
            strict.get_rising_edges(line, units="seconds")
            raised.append(False)
        except P.SyncFaultError:
            raised.append(True)
    try:
        strict.vsync_times_in_blocks
        raised.append(False)
    except P.SyncFaultError:
        raised.append(True)
    check("A5", all(raised), f"unpatched raises on vsync/lick/barcode/camera/vsync blocks: {raised}")
    patched = npc_sync.SyncDataset(sp, patch=patch)
    errs = []
    for bit in range(32):
        try:
            patched.get_rising_edges(bit, units="seconds")
            patched.get_falling_edges(bit, units="seconds")
        except Exception as exc:  # noqa: BLE001
            errs.append((bit, repr(exc)[:80]))
    check("A5", not errs, f"patched: no line raises ({errs})")
    def covers_all(spans):
        return all(any(sa <= a and z <= sz for sa, sz in spans) for a, z in inv)
    ul, ub = patched.get_unknown_intervals("lick_sensor"), patched.get_unknown_intervals("barcode_ephys")
    check("A5", covers_all(ul) and covers_all(ub),
          f"lick_sensor and barcode_ephys unknown in every invalid interval: {ul} {ub}")
    masked = npc_sync.SyncDataset(sp, on_invalid="mask")
    ml = masked.get_rising_edges("lick_sensor", units="seconds")
    check("A5", not P.in_intervals(ml, inv).any(), "mask policy: no lick edges inside intervals")

    # A4 patched vsync blocks
    pblocks = patched.vsync_times_in_blocks
    check("A4", len(pblocks) == len(blocks), f"patched blocks {len(pblocks)} == raw blocks {len(blocks)}")
    for m in matches:
        if m.stim is None:
            continue
        b = blocks[m.block_index]
        ov = [(a, z) for a, z in inv if a <= b[-1] and z > b[0]]
        if not ov:
            continue
        rec = P.solve_vsync_gaps(b, m.stim.frame_intervals, ov)
        pb = pblocks[m.block_index]
        same = len(pb) == m.stim.n_frames and np.allclose(pb, rec.times, atol=1e-9)
        src = patched.get_edge_sources("vsync_stim", "falling")
        allf = patched.get_falling_edges("vsync_stim", units="seconds")
        n_interp = int((src[(allf >= pb[0]) & (allf <= pb[-1])] == P.INTERPOLATED).sum())
        check("A4", same and n_interp == int((rec.source == P.INTERPOLATED).sum()),
              f"block {m.block_index}: patched n={len(pb)} (stim {m.stim.n_frames}), equal to reconciliation={same}, interpolated edges={n_interp}")

    # A6 JSON round trip
    rt = P.SyncPatch.from_json(patch.to_json())
    p2 = npc_sync.SyncDataset(sp, patch=rt)
    same = all(
        np.array_equal(p2.get_rising_edges(b, units="seconds"), patched.get_rising_edges(b, units="seconds"))
        and np.array_equal(p2.get_falling_edges(b, units="seconds"), patched.get_falling_edges(b, units="seconds"))
        for b in range(32)
    )
    check("A6", same, f"JSON round trip ({len(patch.to_json()) / 1e3:.0f} kB) gives identical edges on all 32 bits")

    # A7 photodiode vs analog
    an = LOCAL7.parent / f"analog_{d.name}.npz"
    if not an.exists():
        an = LOCAL7.parent / f"analog_{d.name.replace('DRpilot_', '')}.npz"
    diode_spans = [s for s in patch.lines.get("stim_photodiode", []) if s.status == "reconstructed" and s.method == "vsync+latency"]
    for s in patch.lines.get("stim_photodiode", []):
        print(f"    stim_photodiode {tuple(round(x, 2) for x in s.span)}: {s.status} {s.method} "
              f"{ {k: (round(v * 1e3, 3) if isinstance(v, float) else v) for k, v in s.diagnostics.items() if k in ('latency_rising_sec', 'latency_falling_sec', 'baseline_error_sec', 'after_error_sec', 'n_polarity_errors_after', 'n_inserted', 'n_removed')} } {s.reason}")
    if not an.exists():
        print("  A7 not run: no analog npz")
        continue
    a = np.load(an)
    ta = np.sort(a["t_an_b"])
    pr = patched.get_rising_edges("stim_photodiode", units="seconds")
    pf = patched.get_falling_edges("stim_photodiode", units="seconds")
    e = np.r_[pr, pf]
    epol = np.r_[np.ones(len(pr), bool), np.zeros(len(pf), bool)]
    o = np.argsort(e)
    e, epol = e[o], epol[o]
    for s in diode_spans:
        w0, w1 = s.span

        def pairs(lo, hi):
            """(digital - analog, digital polarity) for analog transitions in [lo, hi)."""
            x = ta[(ta >= lo) & (ta < hi)]
            j = np.clip(np.searchsorted(e, x), 1, len(e) - 1)
            j = np.where(np.abs(x - e[j - 1]) < np.abs(e[j] - x), j - 1, j)
            return e[j] - x, epol[j], x

        dpre, ppre, _ = pairs(w0 - 3, w0)
        lag = {True: np.median(dpre[ppre]), False: np.median(dpre[~ppre])}

        def err(lo, hi):
            dd, pp, x = pairs(lo, hi)
            if not len(dd):
                return np.array([np.nan]), x
            return np.abs(dd - np.where(pp, lag[True], lag[False])), x

        (pre_e, _), (win_e, xw), (post_e, _) = (err(*r) for r in ((w0 - 3, w0), (w0, w1), (w1, w1 + 2.5)))
        check("A7", np.nanmax(win_e) <= np.nanmax(pre_e) + 0.5e-3,
              f"digital diode - analog (per-polarity lag {lag[True] * 1e3:.2f}/{lag[False] * 1e3:.2f} ms removed): "
              f"pre max {np.nanmax(pre_e) * 1e3:.2f} ms (n={len(pre_e)}), "
              f"window max {np.nanmax(win_e) * 1e3:.2f} ms (n={len(win_e)}), post max {np.nanmax(post_e) * 1e3:.2f} ms (n={len(post_e)}); "
              f"window errors > pre max at t={[round(float(t), 3) for t in xw[win_e > np.nanmax(pre_e) + 0.5e-3]]}")

print("\n===== summary")
for k in sorted(results):
    v = results[k]
    print(f"{k}: {sum(v)}/{len(v)} pass")
