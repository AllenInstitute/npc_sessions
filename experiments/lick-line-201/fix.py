"""Prototype fix for npc_sessions#201 (not yet in src/): reconstruct_lick_line and
widen_unknown_lick_spans with the line state at each span bound READ FROM THE SYNC DATA WORDS
(every event on every line carries the lick bit) instead of inferred from the nearest measured
lick edges, plus state carried from span to span when no valid event separates them.

Differences from timing.reconstruct_lick_line (npc_sessions 0.0.296, builder 8):

1. `high_at_entry` = lick bit in the last recorded event before the span start `a`
   (t < a: the edges `SyncPatch.apply` keeps), `high_at_exit` = the state just before the first
   recorded event at or after `z` (if that event is a lick edge, the state before it). Gaps
   between consecutive spans are valid data with vsync/camera events every few ms, so each
   span's bounds are observed directly: no reading of edges from before an earlier span.
2. Measured edges used for `last_rising_before` / the median duration are the ones the patched
   view keeps (outside the lick spans' [a, z)), not `_mask_intervals` (which starts 1.5 samples
   early and so dropped the 3241.49999 F edge from the state, but not from the output).
3. If no recorded event lies between two spans, the entry state is the previous span's exit
   state (after its synthesized edges), and the exit state of the earlier one is 'low'.
4. Afterwards: the patched line must alternate, and every synthesized rising edge without a
   `lickFrames` onset (+-1 frame) must be a lick in progress at a span end. Both are counted
   in the diagnostics.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import dataclasses
import logging

import npc_io
import npc_sync
import numpy as np
import numpy.typing as npt

from npc_sessions import timing
from npc_sessions.timing import (
    LICK_FRAME_MAX_ERROR_SEC,
    LICK_LINE,
    _lick_frame_match_fraction,
    _line_state_before,
    _median_lick_duration,
    _patched_view,
    _unknown_spans,
    predict_lick_onsets,
    synthesize_lick_edges,
)

logger = logging.getLogger(__name__)


def line_state_at_bounds(
    raw: npc_sync.SyncDataset, line: str, spans: Sequence[tuple[float, float]]
) -> list[dict[str, Any]]:
    """Per span [a, z): the line's state just before `a` and just after `z`, from the data
    words of the nearest recorded events outside the span (any line), plus whether such an
    event exists between this span and the next/previous one."""
    t = np.asarray(raw.get_all_times("seconds"), dtype=float)
    bit = raw.get_bit(raw._line_to_bit(line)).astype(np.int8)
    changed = np.ediff1d(bit, to_begin=0) != 0
    out = []
    spans = sorted(spans)
    for k, (a, z) in enumerate(spans):
        prev_z = spans[k - 1][1] if k else -np.inf
        next_a = spans[k + 1][0] if k + 1 < len(spans) else np.inf
        i = int(np.searchsorted(t, a, side="left")) - 1  # last event with t < a
        j = int(np.searchsorted(t, z, side="left"))  # first event with t >= z
        before_ok = i >= 0 and t[i] >= prev_z
        after_ok = j < len(t) and t[j] < next_a
        before = bool(bit[i]) if i >= 0 else False
        if j < len(t):
            after = bool(1 - bit[j]) if changed[j] else bool(bit[j])
        else:
            after = False
        out.append(
            dict(
                span=(a, z),
                high_before=before,
                high_after=after,
                before_observed=bool(before_ok),
                after_observed=bool(after_ok),
                t_before=float(t[i]) if i >= 0 else None,
                t_after=float(t[j]) if j < len(t) else None,
            )
        )
    return out


def measured_kept(
    raw: npc_sync.SyncDataset, line: str, spans: Sequence[tuple[float, float]]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Recorded edges on `line` that `SyncPatch.apply` keeps: outside every span [a, z)."""
    r = np.asarray(raw.get_rising_edges(line, units="seconds"), dtype=float)
    f = np.asarray(raw.get_falling_edges(line, units="seconds"), dtype=float)
    return r[~npc_sync.in_intervals(r, spans)], f[~npc_sync.in_intervals(f, spans)]


def alternates(r: npt.NDArray[np.float64], f: npt.NDArray[np.float64]) -> int:
    """Number of adjacent same-kind edges (0 if the line alternates)."""
    t = np.concatenate([r, f])
    k = np.concatenate([np.ones(len(r)), np.zeros(len(f))])[np.argsort(t, kind="stable")]
    return int(np.sum(k[1:] == k[:-1]))


def reconstruct_lick_line_fixed(
    raw: npc_sync.SyncDataset,
    patch: npc_sync.SyncPatch,
    stim_paths: Sequence[npc_io.PathLike],
) -> dict[str, Any]:
    """As `timing.reconstruct_lick_line`, with the fixes in the module docstring."""
    all_spans = sorted(s.span for s in patch.lines.get(LICK_LINE, []))
    unknown = sorted(s.span for s in _unknown_spans(patch, LICK_LINE))
    if not unknown:
        return {}
    rising, falling = measured_kept(raw, LICK_LINE, all_spans)
    duration = _median_lick_duration(rising, falling)
    bounds = {b["span"]: b for b in line_state_at_bounds(raw, LICK_LINE, all_spans)}
    view = _patched_view(raw, patch)
    predictions = []
    for stim_path in stim_paths:
        try:
            onsets, (t0, t1) = predict_lick_onsets(view, stim_path)
        except Exception as exc:
            logger.debug(f"{raw}: no lick onsets from {stim_path!r}: {exc!r}")
            continue
        if np.isfinite(t0):
            predictions.append((npc_io.from_pathlike(stim_path).stem, onsets, t0, t1))
    info: dict[str, Any] = {"median_duration_sec": duration, "spans": []}
    synth_r: list[float] = []
    synth_f: list[float] = []
    prev_exit_high = False
    new: dict[tuple[float, float], npc_sync.SpanPatch] = {}
    for s in sorted(patch.lines[LICK_LINE], key=lambda s: s.span):
        a, z = s.span
        b = bounds[s.span]
        high_at_entry = b["high_before"] if b["before_observed"] else prev_exit_high
        high_at_exit = b["high_after"] if b["after_observed"] else False
        if s.status != "unknown":
            # a span reconstructed by something else: carry its own last state
            r_s, f_s = np.asarray(s.rising), np.asarray(s.falling)
            prev_exit_high = (
                (r_s.max() if len(r_s) else -np.inf) > (f_s.max() if len(f_s) else -np.inf)
                if (len(r_s) or len(f_s))
                else high_at_entry
            )
            continue
        source = next(
            ((n, on) for n, on, t0, t1 in predictions if t0 <= a and z <= t1), None
        )
        if source is None:
            info["spans"].append({"span": [a, z], "status": "unknown"})
            prev_exit_high = high_at_exit
            continue
        name, onsets = source
        cur_r = np.sort(np.r_[rising, synth_r])
        cur_f = np.sort(np.r_[falling, synth_f])
        last_r = _line_state_before(cur_r, cur_f, a)[1]
        r, f = synthesize_lick_edges(
            onsets, (a, z), high_at_entry, last_r, high_at_exit, duration
        )
        synth_r.extend(r.tolist())
        synth_f.extend(f.tolist())
        prev_exit_high = high_at_exit
        n_checked, frac = _lick_frame_match_fraction(onsets, rising, all_spans)
        diag = dict(
            stim=name,
            n_onsets=int(np.sum((onsets >= a) & (onsets < z))),
            high_at_entry=high_at_entry,
            high_at_exit=high_at_exit,
            state_source="data words" if b["before_observed"] and b["after_observed"] else "carried",
            median_duration_sec=duration,
            n_lick_frames_checked=n_checked,
            fraction_lick_frames_within_1_frame=frac,
        )
        info["spans"].append({"span": [a, z], "status": "reconstructed", **diag})
        new[s.span] = npc_sync.SpanPatch(
            span=s.span,
            status="reconstructed",
            rising=r,
            falling=f,
            method="lick_frames",
            error_bound_sec=LICK_FRAME_MAX_ERROR_SEC,
            diagnostics=diag,
            reason=(
                f"onsets from {name} lickFrames (+-1/2 frame); offsets synthetic (onset + "
                "median measured lick duration); line state at each bound read from the sync "
                "data words just outside the span, so the line alternates"
            ),
        )
    patch.lines[LICK_LINE] = [new.get(s.span, s) for s in patch.lines[LICK_LINE]]
    # checks
    all_on = (
        np.sort(np.concatenate([p[1] for p in predictions])) if predictions else np.array([])
    )
    sr = np.asarray(synth_r)
    if len(sr) and len(all_on):
        j = np.clip(np.searchsorted(all_on, sr), 1, max(1, len(all_on) - 1))
        err = np.minimum(np.abs(all_on[j] - sr), np.abs(all_on[j - 1] - sr))
        n_without = int(np.sum(err > 2 * LICK_FRAME_MAX_ERROR_SEC))
    else:
        n_without = int(len(sr))
    n_in_progress = sum(
        1 for d in info["spans"] if d.get("status") == "reconstructed"
        and d["high_at_exit"] and not d["high_at_entry"] and d["n_onsets"] == 0
    )
    info["n_synthetic_rising_without_lick_frame"] = n_without
    info["n_spans_lick_in_progress_at_exit_without_onset"] = n_in_progress
    return info


def widen_unknown_lick_spans_fixed(
    raw: npc_sync.SyncDataset, patch: npc_sync.SyncPatch
) -> None:
    """As `timing.widen_unknown_lick_spans`, with bound states from the data words."""
    unknown = [s.span for s in _unknown_spans(patch, LICK_LINE)]
    if not unknown:
        return
    all_spans = sorted(s.span for s in patch.lines[LICK_LINE])
    rising, falling = measured_kept(raw, LICK_LINE, all_spans)
    bounds = {b["span"]: b for b in line_state_at_bounds(raw, LICK_LINE, all_spans)}
    out = []
    for s in patch.lines[LICK_LINE]:
        if s.status != "unknown":
            out.append(s)
            continue
        a, z = s.span
        b = bounds[s.span]
        if b["high_before"]:
            r = rising[rising < a]
            if len(r):
                a = float(r[-1])
        if b["high_after"]:
            f = falling[falling >= z]
            if len(f):
                z = float(f[0]) + 1 / raw.sample_freq
        if (a, z) == tuple(s.span):
            out.append(s)
            continue
        out.append(
            dataclasses.replace(
                s,
                span=(a, z),
                reason=s.reason + "; widened to where the line is low (state at the bounds "
                "from the sync data words)",
            )
        )
    patch.lines[LICK_LINE] = out
