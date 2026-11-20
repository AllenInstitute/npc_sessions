"""Sync timing for a session: build, validate, persist and apply a `npc_sync.SyncPatch`.

Every session with stim files gets a patch (`get_sync_patch`):

- Photodiode phantom flicker (diodeBoxFrameInterval 1): extra diode edges while a frame is held
  (a dropped frame in the stim's `frameIntervals`), in state-restoring pairs, at most one per
  missed refresh. The patch removes them, and comparator glitches, from the measured line, in
  files with or without replays, after checking that every diode edge of the block is explained;
  a block with unexplained edges is rejected (`SyncPatch.diode_blocks`). npc_stim uses a cleaned
  block as patched and removes nothing itself. Evidence: experiments/phantom-flicker,
  outputs/phantom-flicker-report.md.
- Replayed acquisition buffers (`npc_sync.SyncDataset.invalid_intervals`): the copies overwrite
  real events on every line. Instead of rejecting the session, the patch reconstructs the lines
  that can be reconstructed within known error (vsync from the stim file's `frameIntervals`, the
  digital photodiode from reconstructed vsyncs, camera exposures from their fixed period) and
  marks the rest 'unknown' (see `npc_sync.patch.build_sync_patch`).

This module adds what npc_sync can't know about a session:

- the stim files and their exact one-to-one match to vsync blocks (by start time only, so a
  frame-count mismatch is detected, not hidden)
- the gates: a block that fails one is rejected (no display times, no fallback)
- barcode spans widened so that no partially-copied barcode is decoded
- event lines npc_sync can't reconstruct alone, inside replayed intervals:
  - sound: from the NI-DAQ recording of the sound (npc_samstim), offset to the sync line,
    for ephys sessions (`reconstruct_sound_line`; gate: <= 1 ms against the sync line on
    every trial outside the intervals)
  - licks: onsets from the stim files' `lickFrames` (+-1/2 frame), offsets synthetic
    (`reconstruct_lick_line`): lick durations aren't reliable anyway. A lick in progress at
    either bound gets a synthetic edge, so the line alternates rising/falling everywhere.
    Where no stim file covers the span, it stays unknown, widened so the line is low at
    both bounds (the edge-based lick at a bound is dropped rather than left unpaired)
  - opto (laser lines): from the stim files' opto trigger frames (npc_stim) + the measured
    latency, +-1/2 frame (`reconstruct_opto_lines`). This keeps edge counts and
    index-based trial/stim mapping consistent; trial opto times are NaN there anyway
- persistence: the patch is cached as JSON keyed by session and sync content hash, with the
  builder and the versions of the packages that build it (`PATCH_DEPENDENCIES`); a cached
  patch is rebuilt (and overwritten, if writable) when the builder or the version of any
  `REUSE_DEPENDENCIES` differs. npc_sessions' own version is provenance only (`BUILDER` is
  bumped when its patch code changes); from a git checkout it includes the commit and a dirty
  flag, and a dirty checkout doesn't write to the shared `DEFAULT_PATCH_ROOT`
- reconstructed licks/rewards for the tables, and masks for trial events on lines still
  unknown inside a replayed interval

Policy decisions (outputs/sync-frame-timing-upgrade-plan.md, notes/sync-patching-handoff-2.md):

- files with a timebase discontinuity (dropped or displaced buffers) are not patched: npc_stim
  still refuses them, so they stay rejected
- replayed intervals are NOT `invalid_times` (decided 2026-11-18): frame times are
  reconstructed there, so trials overlapping them are kept
- licks inside replayed intervals are reconstructed from `lickFrames` (+-1/2 frame) on the
  lick line itself, so downstream code needs no special handling. Provenance is in the licks
  table (`timing_source='stim_frame'` where the onset is interpolated; duration NaN where
  either edge is). Decided 2026-11-18, moved into the patch in builder 4
- opto: +-1/2 frame is not usable. Task trials overlapping a replayed interval are kept, with
  `opto_start/stop_time` NaN and `is_opto` still True (it matters that opto happened).
  OptoTagging trials overlapping one are removed (their `trial_index` values are skipped)
- sound onsets on a sound line that is still unknown (no NI-DAQ, or the gate failed) are NaN
  (`mask_unknown_stim_recordings`): npc_samstim would otherwise take the next stimulus's edge
- rewards inside replayed intervals are reconstructed from `rewardFrames` + the session's
  median measured frame->solenoid latency (`timing_source='stim_frame'`, duration NaN)
"""

from __future__ import annotations

import dataclasses
import datetime
import functools
import importlib.metadata
import json
import logging
import os
import pathlib
import subprocess
from collections.abc import Iterable, Sequence
from typing import Any

import npc_io
import npc_stim
import npc_sync
import numpy as np
import numpy.typing as npt
import pandas as pd
import upath

logger = logging.getLogger(__name__)

BUILDER = "npc_sessions.timing 4"
"""Recorded in each patch. Bump when the builder's output would change: cached patches with
another builder are rebuilt.

- 2: phantom-flicker removal and photodiode checks on every box-1 block (any file), box-1
  prediction inside replayed spans from the block alignment
- 3: sound line reconstructed from the NI-DAQ (npc_samstim) and laser lines from opto
  trigger frames (npc_stim) inside replayed intervals
- 4: lick line reconstructed from the stim files' `lickFrames` inside replayed intervals,
  with synthetic falling edges so that the line alternates; lick spans that stay unknown are
  widened so that the line is low at both bounds (`reconstruct_lick_line`)
"""

PATCH_DIR_ENV_VAR = "NPC_SESSIONS_SYNC_PATCH_DIR"
"""Overrides `DEFAULT_PATCH_ROOT` (local dir or s3://). Set to 'none' to disable the cache."""

DEFAULT_PATCH_ROOT = "s3://aind-scratch-data/dynamic-routing/cache/sync_patches"
"""Where patches are cached as `<session>_<hash[:12]>.json` when AWS credentials are available."""

PATCH_DEPENDENCIES = (
    "npc_sessions",
    "npc_sync",
    "npc_stim",
    "npc_samstim",
    "npc_ephys",
)
"""Packages whose code determines a patch's content: their versions are saved with the patch
(`dependency_versions`) as provenance."""

REUSE_DEPENDENCIES = tuple(d for d in PATCH_DEPENDENCIES if d != "npc_sessions")
"""A cached patch is rebuilt if the version of any of these differs. npc_sessions is excluded:
`BUILDER` is bumped when its patch code changes, and its version changes with every release."""

MATCH_OFFSET_MIN_SEC = -1.5
MATCH_OFFSET_MAX_SEC = 60.0
"""Bounds on (first vsync of block) - (stim file startTime on the sync clock). Observed:
experiments/sync-patch-manifest/README.md."""

BARCODE_SPAN_PADDING_SEC = npc_sync.sync.BARCODE_MAX_DURATION_SEC
"""Barcode 'unknown' spans are widened by this on both sides, so a barcode that is only partly
inside a replayed interval is dropped rather than decoded to a wrong value."""

LICK_FRAME_MAX_ERROR_SEC = 1 / 120
"""Lick onsets from `lickFrames`: +-1/2 frame at 60 Hz."""
LICK_DEFAULT_DURATION_SEC = 0.08
"""Synthetic lick duration if the session has no measured licks to take a median from."""
LICK_MAX_DURATION_SEC = 0.5
"""Measured licks longer than this aren't used for the median duration."""
LICK_LINE = "lick_sensor"
REWARD_LATENCY_MAX_SEC = 0.005
"""Measured solenoid onsets more than this after the trigger frame's flip aren't paired."""


class SyncTimingError(ValueError):
    """A sync file or block failed a timing gate."""


# -------------------------------------------------------------------------- stim matching


@dataclasses.dataclass(frozen=True)
class StimMatch:
    block_index: int
    stim_path: upath.UPath
    offset_sec: float
    n_frames: int


def _stim_start_on_sync(stim_data: Any, sync: npc_sync.SyncDataset) -> float:
    start = npc_stim.get_stim_start_time(stim_data)
    return (start - sync.start_time.replace(tzinfo=None)).total_seconds()


def match_stims_to_blocks(
    sync: npc_sync.SyncDataset, stim_paths: Iterable[npc_io.PathLike]
) -> tuple[dict[int, StimMatch], list[tuple[upath.UPath, str]]]:
    """Exact one-to-one matching of stim files to vsync blocks by start time only.

    `sync` should be an unpatched view (`on_invalid='ignore'`): blocks are as recorded.

    Returns (block index -> match, unmatched stims with reasons). A block claimed by more than
    one stim file matches none of them.
    """
    blocks = sync.vsync_times_in_blocks
    onsets = np.array([b[0] for b in blocks])
    claims: dict[int, list[StimMatch]] = {}
    unmatched: list[tuple[upath.UPath, str]] = []
    for p in stim_paths:
        path = npc_io.from_pathlike(p)
        try:
            data = npc_stim.get_h5_stim_data(path)
            n = npc_stim.get_total_stim_frames(data)
            start = _stim_start_on_sync(data, sync)
        except Exception as exc:
            unmatched.append((path, f"unreadable: {exc!r}"))
            continue
        if n == 0:
            continue
        off = onsets - start
        ok = (off >= MATCH_OFFSET_MIN_SEC) & (off <= MATCH_OFFSET_MAX_SEC)
        if not ok.any():
            unmatched.append(
                (path, "no vsync block starts near the stim file's start time")
            )
            continue
        j = int(np.flatnonzero(ok)[np.argmin(np.abs(off[ok]))])
        claims.setdefault(j, []).append(StimMatch(j, path, float(off[j]), n))
    matches = {}
    for j, c in claims.items():
        if len(c) == 1:
            matches[j] = c[0]
        else:
            unmatched.extend(
                (m.stim_path, f"block {j} claimed by {len(c)} stim files") for m in c
            )
    return matches, unmatched


def get_stim_blocks(
    sync: npc_sync.SyncDataset, stim_paths: Iterable[npc_io.PathLike]
) -> list[npc_sync.StimBlock]:
    matches, _ = match_stims_to_blocks(sync, stim_paths)
    out = []
    for j, m in sorted(matches.items()):
        data = npc_stim.get_h5_stim_data(m.stim_path)
        out.append(
            npc_sync.StimBlock(
                block_index=j,
                frame_intervals=np.asarray(data["frameIntervals"][:], dtype=float),
                diode_box_frame_interval=npc_stim.get_diode_box_frame_interval(data),
                name=m.stim_path.stem,
            )
        )
    return out


# -------------------------------------------------------------------------- patch


def build_session_sync_patch(
    sync: npc_sync.SyncPathOrDataset,
    stim_paths: Iterable[npc_io.PathLike],
    ephys_recording_dirs: Iterable[npc_io.PathLike] | None = None,
) -> npc_sync.SyncPatch:
    """Patch for a sync file (see module docstring).

    With `ephys_recording_dirs`, the sound line is reconstructed inside replayed intervals
    from the NI-DAQ recording (`reconstruct_sound_line`). Laser lines are reconstructed
    from the stim files' opto trigger frames where possible (`reconstruct_opto_lines`).
    """
    raw = npc_sync.SyncDataset(npc_sync.get_sync_data(sync), on_invalid="ignore")
    stim_paths = tuple(stim_paths)
    try:
        stim_blocks = get_stim_blocks(raw, stim_paths)
        note = ""
    except (AssertionError, IndexError, ValueError) as exc:
        # sync-level failure (no vsync edges, unbalanced stim_running, ...): the patch can't
        # check or reconstruct anything that needs stim files, but other lines still apply
        stim_blocks, note = (
            [],
            f"stim files could not be matched to vsync blocks: {exc!r}",
        )
        logger.warning(f"{raw}: {note}")
    patch = npc_sync.build_sync_patch(raw, stim_blocks, builder=BUILDER)
    if note:
        patch.notes.append(note)
    if "barcode_ephys" in patch.lines:
        rising = raw.get_rising_edges("barcode_ephys", units="seconds")
        falling = raw.get_falling_edges("barcode_ephys", units="seconds")
        patch.lines["barcode_ephys"] = [
            dataclasses.replace(
                s,
                span=_span_with_line_low_at_bounds(
                    (
                        max(0.0, s.span[0] - BARCODE_SPAN_PADDING_SEC),
                        s.span[1] + BARCODE_SPAN_PADDING_SEC,
                    ),
                    rising,
                    falling,
                ),
                reason=s.reason
                + f"; widened by >= {BARCODE_SPAN_PADDING_SEC} s, to where the line is low"
                " (no partial barcodes, rising/falling edges removed in pairs)",
            )
            for s in patch.lines["barcode_ephys"]
        ]
    if patch.invalid_intervals and stim_paths:
        steps: list[tuple[Any, dict[str, Any]]] = [
            (reconstruct_lick_line, {}),
            (reconstruct_opto_lines, {}),
            (reconstruct_sound_line, {"ephys_recording_dirs": ephys_recording_dirs}),
        ]
        for fn, kwargs in steps:
            try:
                fn(raw, patch, stim_paths, **kwargs)
            except Exception as exc:
                msg = f"{fn.__name__} failed: {exc!r}"
                logger.warning(f"{raw}: {msg}")
                patch.notes.append(msg)
    if patch.invalid_intervals:
        # lick spans still unknown (no stim file covers them, or reconstruction failed)
        widen_unknown_lick_spans(raw, patch)
    return patch


# -------------------------------------------------------------------------- event lines

SOUND_CHECK_WINDOW_SEC = 60.0
"""Sound-line edges within this of a replayed span must all be explained by the stim files'
sound presentations (one-to-one) before the span is reconstructed. Local, because a file can
have a sound pulse no trial accounts for (839146_20260518: 1 pulse after the task's last
trial, 170 s from the span)."""

SOUND_MAX_DEVIATION_SEC = 0.001
"""NI-DAQ sound onsets replace the sync sound line in a replayed interval only if, on every
trial with both, (NI-DAQ - sync) deviates from its median by at most this. Observed max:
0.211 ms (experiments/sync-patch-manifest/p5_sound_nidaq_vs_sync.log)."""

OPTO_MAX_LATENCY_SEC = 0.1
"""A laser line rising edge is paired with an opto trigger frame if it follows it by at most
this."""

OPTO_ERROR_BOUND_SEC = 1 / 120
"""Reconstructed laser edges: +-1/2 frame (the laser follows the trigger frame's flip with a
latency that isn't fixed to the frame). Trials tables don't use them (opto times are NaN
there): they keep edge counts and index-based trial/stim mapping consistent."""


def _patched_view(
    raw: npc_sync.SyncDataset, patch: npc_sync.SyncPatch
) -> npc_sync.SyncDataset:
    return npc_sync.SyncDataset(raw, patch=patch, on_invalid="mask")


def _unknown_spans(patch: npc_sync.SyncPatch, key: str) -> list[npc_sync.SpanPatch]:
    return [s for s in patch.lines.get(key, []) if s.status == "unknown"]


def _replace_unknown_spans(
    patch: npc_sync.SyncPatch,
    key: str,
    rising: npt.NDArray[np.float64],
    falling: npt.NDArray[np.float64],
    method: str,
    error_bound_sec: float,
    diagnostics: dict[str, Any],
    reason: str,
) -> None:
    out = []
    for s in patch.lines[key]:
        if s.status != "unknown":
            out.append(s)
            continue
        a, z = s.span
        out.append(
            npc_sync.SpanPatch(
                span=s.span,
                status="reconstructed",
                rising=np.sort(rising[(rising >= a) & (rising < z)]),
                falling=np.sort(falling[(falling >= a) & (falling < z)]),
                method=method,
                error_bound_sec=error_bound_sec,
                diagnostics=diagnostics,
                reason=reason,
            )
        )
    patch.lines[key] = out


def _check_edges_explained(
    measured: npt.NDArray[np.float64],
    predicted: npt.NDArray[np.float64],
    spans: Sequence[tuple[float, float]],
    tol: float,
    window: float | None = None,
) -> str:
    """Empty if the measured edges outside `spans` (and within `window` of one, if given)
    are one-to-one with the predicted ones (within `tol`), else the reason."""

    def select(t: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        keep = ~npc_sync.in_intervals(t, spans)
        if window is not None:
            keep &= npc_sync.in_intervals(
                t, [(a - window, z + window) for a, z in spans]
            )
        return np.sort(t[keep])

    m, p = select(measured), select(predicted)
    if len(m) != len(p):
        return f"{len(m)} measured edges outside replayed intervals, {len(p)} predicted"
    if len(m) and np.max(np.abs(m - p)) > tol:
        return f"measured edges differ from predicted by up to {np.max(np.abs(m - p)) * 1e3:.3f} ms"
    return ""


def _vsync_times(
    sync: npc_sync.SyncDataset, stim_path: npc_io.PathLike
) -> npt.NDArray[np.float64]:
    t = npc_stim.get_stim_frame_times(stim_path, sync=sync, frame_time_type="vsync")
    v = next(iter(t.values()))
    if isinstance(v, Exception):
        raise v
    return np.asarray(v, dtype=float)


def reconstruct_sound_line(
    raw: npc_sync.SyncDataset,
    patch: npc_sync.SyncPatch,
    stim_paths: Sequence[npc_io.PathLike],
    ephys_recording_dirs: Iterable[npc_io.PathLike] | None = None,
) -> dict[str, Any]:
    """Replace the sound line's 'unknown' spans, in place, with edges from the NI-DAQ
    recording of the sound (npc_samstim).

    Per stim file with sound, onsets come from the sync line (npc_samstim, on the patched
    view) and from the NI-DAQ (`get_stim_latencies_from_nidaq_recording`). Inside the spans,
    the rising edge is the NI-DAQ onset minus the file's median (NI-DAQ - sync) difference.
    The falling edge is onset + waveform duration + the median measured excess pulse width.

    Applied only if (1) in every file, every trial with both onsets is within
    `SOUND_MAX_DEVIATION_SEC` of the median difference, (2) every trial whose sync onset is
    unknown gets a NI-DAQ onset, and (3) the predicted pulses are one-to-one with the
    measured edges within `SOUND_CHECK_WINDOW_SEC` of the spans (nothing else is on the line
    there). Otherwise the spans stay
    unknown, with a note. Returns diagnostics per stim file.
    """
    diagnostics: dict[str, Any] = {}
    try:
        line = raw.get_line_for_stim_onset("sound")
    except ValueError:
        return diagnostics
    key = raw.line_key(line)
    spans = _unknown_spans(patch, key)
    if not spans:
        return diagnostics
    span_bounds = [s.span for s in spans]
    if not ephys_recording_dirs:
        patch.notes.append(
            f"{key} (sound) unknown in replayed intervals: no NI-DAQ recording"
        )
        return diagnostics
    import npc_samstim

    recording_dirs = tuple(ephys_recording_dirs)
    view = _patched_view(raw, patch)
    measured_on = view.get_rising_edges(line, units="seconds")
    measured_off = view.get_falling_edges(line, units="seconds")
    rising: list[float] = []
    falling: list[float] = []
    for stim_path in stim_paths:
        stim = npc_stim.get_h5_stim_data(stim_path)
        try:
            waveforms = npc_samstim.get_waveforms_from_stim_file(stim, "sound")
        except Exception:
            continue
        if not any(w is not None for w in waveforms):
            continue
        name = npc_io.from_pathlike(stim_path).stem
        rs = npc_samstim.get_stim_latencies_from_sync(
            stim_path, view, waveform_type="sound"
        )
        masked = stim_recordings_unknown_mask(rs, view, line, include_offset=False)
        a = np.array(
            [np.nan if r is None else r.onset_time_on_sync for r in rs], dtype=float
        )
        dur = np.array(
            [np.nan if w is None else w.duration for w in waveforms], dtype=float
        )[: len(a)]
        ok = np.isfinite(a) & ~masked
        j = np.searchsorted(measured_off, a[ok])
        valid = j < len(measured_off)
        excess = measured_off[j[valid]] - a[ok][valid] - dur[ok][valid]
        w_med = float(np.median(excess)) if len(excess) else 0.0
        onset = a.copy()
        onset[masked] = np.nan
        diag: dict[str, Any] = dict(
            excess_width_sec=w_med, n_reconstructed=int(masked.sum())
        )
        if masked.any():
            rn = npc_samstim.get_stim_latencies_from_nidaq_recording(
                stim_path, view, recording_dirs, waveform_type="sound"
            )
            b = np.array(
                [np.nan if r is None else r.onset_time_on_sync for r in rn], dtype=float
            )[: len(a)]
            both = np.isfinite(a) & np.isfinite(b) & ~masked
            if both.sum() < 10:
                patch.notes.append(
                    f"{key} (sound) unknown: {name}: {both.sum()} trials with sync and NI-DAQ onsets"
                )
                return diagnostics
            d = b[both] - a[both]
            d_med = float(np.median(d))
            dev = float(np.max(np.abs(d - d_med)))
            diag |= dict(
                n_compared=int(both.sum()),
                median_nidaq_minus_sync_sec=d_med,
                max_deviation_sec=dev,
                reconstructed_onsets=[float(x) for x in b[masked] - d_med],
            )
            diagnostics[name] = diag
            if dev > SOUND_MAX_DEVIATION_SEC:
                patch.notes.append(
                    f"{key} (sound) unknown: {name}: NI-DAQ onsets deviate from sync by up to {dev * 1e3:.3f} ms"
                )
                return diagnostics
            if not np.isfinite(b[masked]).all():
                patch.notes.append(
                    f"{key} (sound) unknown: {name}: no NI-DAQ onset for a trial in a replayed interval"
                )
                return diagnostics
            onset[masked] = b[masked] - d_med
        diagnostics[name] = diag
        f = np.isfinite(onset)
        rising.extend(onset[f])
        falling.extend(onset[f] + dur[f] + w_med)
    r_arr, f_arr = np.asarray(rising, dtype=float), np.asarray(falling, dtype=float)
    for kind, measured, pred in (
        ("rising", measured_on, r_arr),
        ("falling", measured_off, f_arr),
    ):
        why = _check_edges_explained(
            measured,
            pred,
            span_bounds,
            tol=SOUND_MAX_DEVIATION_SEC,
            window=SOUND_CHECK_WINDOW_SEC,
        )
        if why:
            patch.notes.append(
                f"{key} (sound) unknown: {kind} edges not explained by sound stimuli: {why}"
            )
            return diagnostics
    _replace_unknown_spans(
        patch,
        key,
        r_arr,
        f_arr,
        method="nidaq_sound",
        error_bound_sec=SOUND_MAX_DEVIATION_SEC,
        diagnostics=diagnostics,
        reason="sound onsets from the NI-DAQ recording (npc_samstim), offset to the sync "
        "line; offsets = onset + waveform duration + median measured excess width",
    )
    return diagnostics


def _opto_triggers(
    view: npc_sync.SyncDataset, stim_paths: Sequence[npc_io.PathLike]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """(vsync time of every opto trigger frame in the stim files, trialOptoDur), sorted."""
    triggers: list[float] = []
    durations: list[float] = []
    for stim_path in stim_paths:
        stim = npc_stim.get_h5_stim_data(stim_path)
        if "trialOptoOnsetFrame" not in stim:
            continue
        frames = npc_stim.get_stim_trigger_frames(stim, stim_type="opto")
        if not any(f is not None for f in frames):
            continue
        vs = _vsync_times(view, stim_path)
        dur = np.asarray(stim["trialOptoDur"][()], dtype=float)
        dur = dur.reshape(len(dur), -1) if dur.ndim else dur.reshape(1, 1)
        dur = np.array([np.nanmax(r) if np.isfinite(r).any() else np.nan for r in dur])
        for i, fr in enumerate(frames):
            if fr is None or not np.isfinite(fr) or int(fr) >= len(vs):
                continue
            triggers.append(float(vs[int(fr)]))
            durations.append(float(dur[i]) if i < len(dur) else np.nan)
    trig, durs = np.asarray(triggers, dtype=float), np.asarray(durations, dtype=float)
    order = np.argsort(trig)
    return trig[order], durs[order]


def predict_opto_line(
    on: npt.NDArray[np.float64],
    off: npt.NDArray[np.float64],
    trig: npt.NDArray[np.float64],
    durs: npt.NDArray[np.float64],
    spans: Sequence[tuple[float, float]],
) -> dict[str, Any]:
    """Predicted laser edges for all triggers, from the measured edges outside `spans`.

    Returns `status` 'reconstructed' (with `rising`, `falling` and diagnostics) if the
    measured rising edges outside the spans pair one-to-one with all triggers outside the
    spans, else 'unknown' with a `reason`."""
    on = on[~npc_sync.in_intervals(on, spans)]
    off = off[~npc_sync.in_intervals(off, spans)]
    outside = ~npc_sync.in_intervals(trig, spans)
    t_out = trig[outside]
    if not len(on) and not len(trig):
        return dict(
            status="reconstructed",
            rising=np.array([]),
            falling=np.array([]),
            n_paired=0,
            n_triggers_in_spans=0,
        )
    j = np.searchsorted(on, t_out)
    paired = j < len(on)
    lat = np.full(len(t_out), np.inf)
    lat[paired] = on[j[paired]] - t_out[paired]
    paired &= lat <= OPTO_MAX_LATENCY_SEC
    if not len(t_out) or not paired.all() or len(np.unique(j)) != len(on):
        return dict(
            status="unknown",
            reason=(
                f"{len(on)} measured rising edges outside replayed intervals, "
                f"{len(t_out)} opto triggers, {int(paired.sum())} paired one-to-one"
            ),
        )
    lat_med = float(np.median(lat))
    k = np.searchsorted(off, on[j])
    k_ok = k < len(off)
    excess = off[k[k_ok]] - on[j][k_ok] - durs[outside][k_ok]
    w_med = float(np.nanmedian(excess)) if np.isfinite(excess).any() else 0.0
    rise = trig + lat_med
    fall = rise + np.where(np.isfinite(durs), durs, 0.0) + w_med
    return dict(
        status="reconstructed",
        rising=rise,
        falling=fall,
        n_paired=int(len(t_out)),
        median_latency_sec=lat_med,
        latency_range_sec=[float(lat.min()), float(lat.max())],
        excess_width_sec=w_med,
        max_abs_excess_deviation_sec=(
            float(np.nanmax(np.abs(excess - w_med)))
            if np.isfinite(excess).any()
            else None
        ),
        n_triggers_in_spans=int((~outside).sum()),
    )


def reconstruct_opto_lines(
    raw: npc_sync.SyncDataset,
    patch: npc_sync.SyncPatch,
    stim_paths: Sequence[npc_io.PathLike],
) -> dict[str, Any]:
    """Replace laser lines' 'unknown' spans, in place, with edges from the stim files' opto
    trigger frames (npc_stim): rising = trigger frame vsync + the line's median measured
    latency, falling = rising + `trialOptoDur` + the median measured excess width, within
    +-`OPTO_ERROR_BOUND_SEC`.

    The trials tables don't use these times: opto times are NaN in trials overlapping a
    replayed interval. The edges keep counts consistent for index-based trial/stim mapping.
    A line is reconstructed only if its measured rising edges outside the spans pair
    one-to-one with all opto triggers outside the spans (see `predict_opto_line`).
    Returns per-line diagnostics.
    """
    info: dict[str, Any] = {}
    keys = [k for k in ("laser_488", "laser_633") if _unknown_spans(patch, k)]
    if not keys:
        return info
    view = _patched_view(raw, patch)
    trig, durs = _opto_triggers(view, stim_paths)
    for key in keys:
        spans = [s.span for s in _unknown_spans(patch, key)]
        pred = predict_opto_line(
            view.get_rising_edges(key, units="seconds"),
            view.get_falling_edges(key, units="seconds"),
            trig,
            durs,
            spans,
        )
        info[key] = {k: v for k, v in pred.items() if k not in ("rising", "falling")}
        if pred["status"] != "reconstructed":
            patch.notes.append(f"{key} (opto) unknown: {pred['reason']}")
            continue
        _replace_unknown_spans(
            patch,
            key,
            pred["rising"],
            pred["falling"],
            method="opto_trigger_frame+latency",
            error_bound_sec=OPTO_ERROR_BOUND_SEC,
            diagnostics=info[key],
            reason="opto trigger frame vsync (npc_stim) + median measured latency; offsets = "
            "onset + trialOptoDur + median measured excess width. Not used for trial opto "
            "times (NaN in trials overlapping replayed intervals)",
        )
    return info


def _line_state_before(
    rising: npt.NDArray[np.float64], falling: npt.NDArray[np.float64], t: float
) -> tuple[bool, float]:
    """(line is high just before `t`, time of the last rising edge before `t`)."""
    r = rising[rising < t]
    f = falling[falling < t]
    last_r = float(r[-1]) if len(r) else -np.inf
    last_f = float(f[-1]) if len(f) else -np.inf
    return last_r > last_f, last_r


def _first_edge_after(
    rising: npt.NDArray[np.float64], falling: npt.NDArray[np.float64], t: float
) -> tuple[str | None, float]:
    """('rising' / 'falling' / None, time) of the first edge at or after `t`."""
    r = rising[rising >= t]
    f = falling[falling >= t]
    r0 = float(r[0]) if len(r) else np.inf
    f0 = float(f[0]) if len(f) else np.inf
    if not np.isfinite(min(r0, f0)):
        return None, np.inf
    return ("rising", r0) if r0 < f0 else ("falling", f0)


def _measured_outside(
    raw: npc_sync.SyncDataset, line: str, spans: Sequence[tuple[float, float]]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Recorded rising/falling edges on `line` outside `spans` and the file's invalid
    intervals (starting one sample early, as `SyncDataset._mask_intervals`)."""
    masked = [*spans, *raw._mask_intervals]
    r = np.asarray(raw.get_rising_edges(line, units="seconds"), dtype=float)
    f = np.asarray(raw.get_falling_edges(line, units="seconds"), dtype=float)
    return r[~npc_sync.in_intervals(r, masked)], f[~npc_sync.in_intervals(f, masked)]


def predict_lick_onsets(
    sync: npc_sync.SyncDataset, stim_path: npc_io.PathLike
) -> tuple[npt.NDArray[np.float64], tuple[float, float]]:
    """(lick onsets from the stim file's `lickFrames`, (first, last) input read time).

    Outside replayed intervals, ~98% of `lickFrames` have a sync lick onset in the frame
    interval before the frame's input read (median 8.35 ms before it,
    experiments/sync-frame-timing/lick_count_check.log); the midpoint of that interval is used,
    so the error is within +-1/2 frame (+-8.3 ms at 60 Hz).
    """
    data = npc_stim.get_h5_stim_data(stim_path)
    if "lickFrames" not in data:
        return np.array([]), (np.nan, np.nan)
    frames = np.asarray(data["lickFrames"][()], dtype=float).ravel()
    frames = frames[np.isfinite(frames)].astype(int)
    read = np.asarray(npc_stim.get_input_data_times(stim_path, sync=sync), dtype=float)
    if np.isfinite(read).sum() < 2:
        return np.array([]), (np.nan, np.nan)
    frames = frames[(frames >= 2) & (frames < len(read))]
    t = read[frames] - 0.5 * (read[frames] - read[frames - 1])
    return np.sort(t[np.isfinite(t)]), (float(np.nanmin(read)), float(np.nanmax(read)))


def synthesize_lick_edges(
    onsets: npt.ArrayLike,
    span: tuple[float, float],
    high_at_entry: bool,
    last_rising_before: float,
    high_at_exit: bool,
    duration: float,
    min_gap: float = 1e-5,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Rising and falling edges inside `span` = [a, z) for the lick `onsets` there, so that
    the line alternates together with the measured edges outside. Each onset is a rising
    edge, with a falling edge `duration` later (shortened to half the gap to the next onset,
    or to the end of the span), plus:

    - `high_at_entry` (a lick in progress at `a`): a falling edge at
      `last_rising_before + duration`, no earlier than `a` and before the first onset
    - `high_at_exit` (the first measured edge after `z` is falling): the line is left high.
      The last onset's falling edge is omitted, or with no onsets a rising edge is added
      `duration` before `z` (unless the line is high at entry: then it stays high throughout)

    >>> r, f = synthesize_lick_edges([2.0, 2.2], (1.0, 3.0), True, 0.95, False, 0.1)
    >>> r.tolist(), [round(x, 6) for x in f]
    ([2.0, 2.2], [1.05, 2.1, 2.3])
    >>> r, f = synthesize_lick_edges([2.0], (1.0, 3.0), False, -1, True, 0.1)
    >>> r.tolist(), f.tolist()
    ([2.0], [])
    >>> [x.size for x in synthesize_lick_edges([], (1.0, 3.0), True, 0.9, True, 0.1)]
    [0, 0]
    >>> r, f = synthesize_lick_edges([], (1.0, 3.0), False, 0.5, True, 0.1)
    >>> r.tolist(), f.tolist()
    ([2.9], [])
    """
    a, z = span
    on = np.asarray(onsets, dtype=float).ravel()
    on = np.sort(on[(on >= a) & (on < z)])
    rising = [float(x) for x in on]
    falling: list[float] = []
    for i, r in enumerate(on):
        nxt = on[i + 1] if i + 1 < len(on) else z
        falling.append(float(r + min(duration, 0.5 * (nxt - r))))
    if high_at_exit:
        if len(on):
            falling.pop()
        elif not high_at_entry:
            rising.append(max(a, z - duration))
    if high_at_entry and (len(on) or not high_at_exit):
        first = float(on[0]) if len(on) else z
        f0 = max(a, last_rising_before + duration)
        if f0 > first - min_gap:
            f0 = 0.5 * (a + first)
        falling.insert(0, f0)
    return np.asarray(rising, dtype=float), np.sort(np.asarray(falling, dtype=float))


def _median_lick_duration(
    rising: npt.NDArray[np.float64], falling: npt.NDArray[np.float64]
) -> float:
    r, f = pair_rising_falling_edges(rising, falling)
    d = f - r
    d = d[(d > 0) & (d <= LICK_MAX_DURATION_SEC)]
    return float(np.median(d)) if len(d) else LICK_DEFAULT_DURATION_SEC


def _lick_frame_match_fraction(
    onsets: npt.NDArray[np.float64],
    measured: npt.NDArray[np.float64],
    excluded: Sequence[tuple[float, float]],
) -> tuple[int, float | None]:
    """(n, fraction of `lickFrames` onsets outside `excluded` with a measured onset within
    one frame): a diagnostic of the reconstruction, not a gate (~98% expected)."""
    t = onsets[~npc_sync.in_intervals(onsets, excluded)]
    if len(measured) < 2:
        return 0, None
    t = t[(t >= measured[0]) & (t <= measured[-1])]
    if not len(t):
        return 0, None
    j = np.clip(np.searchsorted(measured, t), 1, len(measured) - 1)
    err = np.minimum(np.abs(measured[j] - t), np.abs(measured[j - 1] - t))
    return int(len(t)), float(np.mean(err <= 2 * LICK_FRAME_MAX_ERROR_SEC))


def reconstruct_lick_line(
    raw: npc_sync.SyncDataset,
    patch: npc_sync.SyncPatch,
    stim_paths: Sequence[npc_io.PathLike],
) -> dict[str, Any]:
    """Replace the lick line's 'unknown' spans, in place, with onsets from the `lickFrames`
    (`predict_lick_onsets`, +-`LICK_FRAME_MAX_ERROR_SEC`) of a stim file whose frames cover
    the span, and synthetic offsets from the session's median measured lick duration
    (`synthesize_lick_edges`). The line then alternates rising/falling across the span
    bounds, so downstream code pairs edges as usual. Lick durations on the line aren't
    reliable anyway, and the licks table marks interpolated edges.

    Spans that no stim file covers stay unknown (see `widen_unknown_lick_spans`). Returns
    diagnostics.
    """
    spans = [s.span for s in _unknown_spans(patch, LICK_LINE)]
    if not spans:
        return {}
    rising, falling = _measured_outside(raw, LICK_LINE, spans)
    duration = _median_lick_duration(rising, falling)
    view = _patched_view(raw, patch)
    predictions = []
    for stim_path in stim_paths:
        try:
            onsets, (t0, t1) = predict_lick_onsets(view, stim_path)
        except Exception as exc:
            logger.debug(f"{raw}: no lick onsets from {stim_path}: {exc!r}")
            continue
        if np.isfinite(t0):
            predictions.append((npc_io.from_pathlike(stim_path).stem, onsets, t0, t1))
    info: dict[str, Any] = {"median_duration_sec": duration, "spans": []}
    out = []
    for s in patch.lines[LICK_LINE]:
        if s.status != "unknown":
            out.append(s)
            continue
        a, z = s.span
        source = next(
            ((n, on) for n, on, t0, t1 in predictions if t0 <= a and z <= t1), None
        )
        if source is None:
            info["spans"].append({"span": [a, z], "status": "unknown"})
            out.append(s)
            continue
        name, onsets = source
        high_at_entry, last_r = _line_state_before(rising, falling, a)
        high_at_exit = _first_edge_after(rising, falling, z)[0] == "falling"
        r, f = synthesize_lick_edges(
            onsets, (a, z), high_at_entry, last_r, high_at_exit, duration
        )
        n_checked, frac = _lick_frame_match_fraction(
            onsets, rising, [*spans, *raw._mask_intervals]
        )
        diag = dict(
            stim=name,
            n_onsets=int(np.sum((onsets >= a) & (onsets < z))),
            high_at_entry=high_at_entry,
            high_at_exit=high_at_exit,
            median_duration_sec=duration,
            n_lick_frames_checked=n_checked,
            fraction_lick_frames_within_1_frame=frac,
        )
        info["spans"].append({"span": [a, z], "status": "reconstructed", **diag})
        out.append(
            npc_sync.SpanPatch(
                span=s.span,
                status="reconstructed",
                rising=r,
                falling=f,
                method="lick_frames",
                error_bound_sec=LICK_FRAME_MAX_ERROR_SEC,
                diagnostics=diag,
                reason=(
                    f"onsets from {name} lickFrames (midpoint of the frame interval before "
                    "the input read, +-1/2 frame); offsets synthetic (onset + median measured "
                    "lick duration, shortened to fit), plus a synthetic edge for a lick in "
                    "progress at either bound, so the line alternates"
                ),
            )
        )
    patch.lines[LICK_LINE] = out
    return info


def widen_unknown_lick_spans(
    raw: npc_sync.SyncDataset, patch: npc_sync.SyncPatch
) -> None:
    """Widen the lick line's remaining 'unknown' spans, in place, so the line is low at both
    bounds. A lick in progress at the start (rising edge measured, falling edge in the span)
    or at the end (falling edge is the first measured edge after the span) is included
    whole, so the measured edges outside still alternate."""
    unknown = [s.span for s in _unknown_spans(patch, LICK_LINE)]
    if not unknown:
        return
    rising, falling = _measured_outside(raw, LICK_LINE, unknown)
    out = []
    for s in patch.lines[LICK_LINE]:
        if s.status != "unknown":
            out.append(s)
            continue
        a, z = s.span
        high, last_r = _line_state_before(rising, falling, a)
        if high:
            a = last_r
        kind, t = _first_edge_after(rising, falling, z)
        if kind == "falling":
            z = t + 1 / raw.sample_freq
        if (a, z) == tuple(s.span):
            out.append(s)
            continue
        out.append(
            dataclasses.replace(
                s,
                span=(a, z),
                reason=s.reason
                + "; widened to where the line is low (a lick in progress at a bound is "
                "included whole, so the measured edges outside alternate)",
            )
        )
    patch.lines[LICK_LINE] = out


def _span_with_line_low_at_bounds(
    span: tuple[float, float],
    rising: npt.NDArray[np.float64],
    falling: npt.NDArray[np.float64],
) -> tuple[float, float]:
    """Move span bounds outward so the line is low at both: a pulse in progress at a bound is
    included whole, so the edges outside stay paired (npc_ephys barcode decoding requires
    equal rising/falling counts)."""
    a, z = span

    def high_at(t: float) -> tuple[bool, float, float]:
        r = rising[rising < t]
        f = falling[falling < t]
        last_r = r[-1] if len(r) else -np.inf
        last_f = f[-1] if len(f) else -np.inf
        return bool(last_r > last_f), float(last_r), float(last_f)

    is_high, last_r, _ = high_at(a)
    if is_high:
        a = last_r  # include the rising edge of the pulse in progress
    is_high, _, _ = high_at(z)
    if is_high:
        nxt = falling[falling >= z]
        z = float(nxt[0]) + 1e-9 if len(nxt) else z
    return (a, z)


@functools.cache
def get_npc_sessions_git_state() -> tuple[str, bool] | None:
    """(commit, is_dirty) if npc_sessions is imported from a git checkout (e.g. an editable
    install), else None. Dirty means uncommitted changes under the package's source dir."""
    pkg_dir = pathlib.Path(__file__).resolve().parent

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=pkg_dir,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()

    try:
        top = pathlib.Path(git("rev-parse", "--show-toplevel")).resolve()
        if "site-packages" in pkg_dir.relative_to(top).parts:
            return None  # installed dist inside some other repo (e.g. a venv in a repo)
        commit = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain", "--", str(pkg_dir)))
    except Exception:
        return None
    return commit, dirty


def get_npc_sessions_version() -> str:
    """Installed version, with a local suffix identifying the code when imported from a git
    checkout: `<version>+g<commit[:12]>`, plus `.dirty` with uncommitted changes."""
    try:
        version = importlib.metadata.version("npc_sessions")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    state = get_npc_sessions_git_state()
    if state is None:
        return version
    commit, dirty = state
    return f"{version}+g{commit[:12]}" + (".dirty" if dirty else "")


def get_dependency_versions() -> dict[str, str]:
    """Installed versions of `PATCH_DEPENDENCIES` ('unknown' if not installed as a dist).
    npc_sessions' includes the git commit when run from a checkout
    (`get_npc_sessions_version`)."""
    out = {}
    for name in PATCH_DEPENDENCIES:
        if name == "npc_sessions":
            out[name] = get_npc_sessions_version()
            continue
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = "unknown"
    return out


def is_reusable_versions(saved: dict[str, str], current: dict[str, str]) -> bool:
    """Whether a patch saved with `saved` dependency versions can be reused now: only
    `REUSE_DEPENDENCIES` are compared."""
    return all(saved.get(name) == current.get(name) for name in REUSE_DEPENDENCIES)


def is_shared_patch_root(path: npc_io.PathLike) -> bool:
    """Whether `path` is in the shared `DEFAULT_PATCH_ROOT`."""
    root = npc_io.from_pathlike(DEFAULT_PATCH_ROOT).as_posix().rstrip("/")
    return npc_io.from_pathlike(path).as_posix().startswith(root + "/")


def has_aws_credentials() -> bool:
    try:
        import botocore.session

        return botocore.session.get_session().get_credentials() is not None
    except Exception:
        return False


def get_patch_root(root: npc_io.PathLike | None = None) -> upath.UPath | None:
    """`root`, else `$NPC_SESSIONS_SYNC_PATCH_DIR`, else `DEFAULT_PATCH_ROOT` if AWS
    credentials are available, else None (no cache)."""
    root = root or os.environ.get(PATCH_DIR_ENV_VAR)
    if root is not None and str(root).lower() in ("", "none", "0", "false"):
        return None
    if not root:
        if not has_aws_credentials():
            return None
        root = DEFAULT_PATCH_ROOT
    return npc_io.from_pathlike(root)


def get_patch_path(
    session_id: str, content_hash: str, root: npc_io.PathLike | None = None
) -> upath.UPath | None:
    r = get_patch_root(root)
    if r is None:
        return None
    return r / f"{session_id}_{content_hash[:12]}.json"


def save_patch(patch: npc_sync.SyncPatch, path: npc_io.PathLike) -> None:
    """Save as `SyncPatch` JSON plus `dependency_versions` (ignored by `SyncPatch.load`)."""
    d = patch.to_dict() | {"dependency_versions": get_dependency_versions()}
    npc_io.from_pathlike(path).write_text(json.dumps(d, indent=1))


def load_patch(
    path: npc_io.PathLike,
) -> tuple[npc_sync.SyncPatch, dict[str, str]]:
    """(patch, dependency versions it was built with: empty for patches saved without)."""
    d = json.loads(npc_io.from_pathlike(path).read_text())
    return npc_sync.SyncPatch.from_dict(d), dict(d.get("dependency_versions") or {})


def get_sync_patch(
    session_id: str,
    sync: npc_sync.SyncPathOrDataset,
    stim_paths: Iterable[npc_io.PathLike],
    root: npc_io.PathLike | None = None,
    write: bool = True,
    ephys_recording_dirs: Iterable[npc_io.PathLike] | None = None,
) -> npc_sync.SyncPatch | None:
    """The session's patch (every session: photodiode phantom-flicker removal applies to files
    without replays too).

    Loaded from the cache (`get_patch_root`: `root`, `$NPC_SESSIONS_SYNC_PATCH_DIR`, or
    `DEFAULT_PATCH_ROOT` on S3 when AWS credentials are available) if a patch exists there
    for the same sync content hash, builder and `REUSE_DEPENDENCIES` versions. Otherwise
    built, and saved there (overwriting a stale one) if `write` - except to the shared
    `DEFAULT_PATCH_ROOT` from a git checkout of npc_sessions with uncommitted changes. Cache
    read/write failures are logged and the patch is built in memory.
    """
    raw = npc_sync.SyncDataset(npc_sync.get_sync_data(sync), on_invalid="ignore")
    path = get_patch_path(session_id, raw.content_hash, root)
    versions = get_dependency_versions()
    if path is not None:
        try:
            if path.exists():
                patch, saved_versions = load_patch(path)
                if (
                    patch.sync_content_hash == raw.content_hash
                    and patch.builder == BUILDER
                    and is_reusable_versions(saved_versions, versions)
                ):
                    return patch
                logger.info(
                    f"{session_id}: cached sync patch {path} is stale "
                    f"(builder {patch.builder!r}, versions {saved_versions}; now {BUILDER!r}, "
                    f"{versions}): rebuilding"
                )
        except Exception as exc:
            logger.warning(
                f"{session_id}: could not read cached sync patch {path}: {exc!r}"
            )
    patch = build_session_sync_patch(raw, stim_paths, ephys_recording_dirs)
    if path is not None and write:
        state = get_npc_sessions_git_state()
        if state is not None and state[1] and is_shared_patch_root(path):
            logger.warning(
                f"{session_id}: not saving sync patch to shared cache {path}: npc_sessions "
                f"has uncommitted changes ({versions['npc_sessions']}). Commit them, or set "
                f"${PATCH_DIR_ENV_VAR} to another location"
            )
            return patch
        try:
            save_patch(patch, path)
            logger.info(f"{session_id}: saved sync patch to {path}")
        except Exception as exc:
            logger.warning(
                f"{session_id}: could not save sync patch to {path}: {exc!r}"
            )
    return patch


# -------------------------------------------------------------------------- reports


def get_block_report(
    sync: npc_sync.SyncDataset,
    stim_paths: Iterable[npc_io.PathLike],
    display_times: Sequence[npc_stim.BlockDisplayTimes] | None = None,
) -> pd.DataFrame:
    """One row per vsync block: matched stim, status ('clean' / 'patched' / 'rejected'),
    reason, frame counts and the gate values from the patch."""
    raw = npc_sync.SyncDataset(sync, on_invalid="ignore")
    matches, _ = match_stims_to_blocks(raw, stim_paths)
    patch = sync.patch
    rows = []
    for j, b in enumerate(raw.vsync_times_in_blocks):
        m = matches.get(j)
        row: dict[str, Any] = dict(
            block_index=j,
            start_time=float(b[0]),
            stop_time=float(b[-1]),
            stim=m.stim_path.stem if m else None,
            n_stim_frames=m.n_frames if m else None,
            n_vsyncs_recorded=len(b),
        )
        spans = []
        if patch is not None:
            spans = [
                s
                for line in ("vsync_stim", "stim_photodiode")
                for s in patch.lines.get(line, [])
                if s.span[0] <= b[-1] and s.span[1] >= b[0]
            ]
        unknown = [s for s in spans if s.status == "unknown"]
        row["n_patched_spans"] = len({s.span for s in spans})
        diode = (
            patch.diode_block(float(b[0]), float(b[-1])) if patch is not None else None
        )
        row["diode_status"] = diode.status if diode is not None else None
        row["diode_reason"] = diode.reason if diode is not None else None
        row["n_phantom_flicker_removed"] = (
            diode.n_phantom_flicker if diode and diode.status == "cleaned" else 0
        )
        row["n_glitch_removed"] = (
            diode.n_glitch if diode and diode.status == "cleaned" else 0
        )
        row["anchor_residuals_ms"] = [
            round(s.diagnostics["anchor_residual_sec"] * 1e3, 4)
            for s in spans
            if s.method == "frame_intervals"
        ]
        if display_times is not None and j < len(display_times):
            dt = display_times[j]
            row["display_time_source"] = dt.source
            row["n_interpolated_frames"] = dt.n_interpolated
        if m is None:
            row["status"], row["reason"] = "rejected", "no stim file matched to block"
        elif unknown:
            row["status"] = "rejected"
            row["reason"] = "; ".join(sorted({s.reason for s in unknown}))
        elif diode is not None and diode.status == "rejected":
            row["status"] = "rejected"
            row["reason"] = f"photodiode edges not explained: {diode.reason}"
        elif spans:
            row["status"], row["reason"] = "patched", ""
        elif len(b) != m.n_frames:
            row["status"] = "rejected"
            row["reason"] = (
                f"{len(b)} vsyncs, {m.n_frames} stim frames, no fault explains it"
            )
        else:
            row["status"], row["reason"] = "clean", ""
        if (
            display_times is not None
            and j < len(display_times)
            and row["status"] != "rejected"
            and display_times[j].source != "photodiode"
        ):
            row["status"] = "rejected"
            row["reason"] = f"display times: {display_times[j].reason}"
        rows.append(row)
    return pd.DataFrame(rows)


def get_replayed_intervals(sync: npc_sync.SyncDataset) -> list[dict[str, Any]]:
    """One entry per replayed interval of a patched file: `start_time`, `stop_time` and a
    `reason` listing the lines whose events are unknown there and those reconstructed.

    Not `invalid_times` (see module docstring): for notes, and for the trials tables, which
    NaN the affected events (`unknown_overlap_mask`)."""
    patch = sync.patch
    if patch is None or not sync.invalid_intervals:
        return []
    out = []
    for a, z in sync.invalid_intervals:
        unknown, reconstructed = [], []
        for line, spans in patch.lines.items():
            for s in spans:
                if s.span[0] <= a + 1e-6 and s.span[1] >= z - 1e-6:
                    if s.status == "unknown":
                        unknown.append(line)
                    elif s.status == "reconstructed" and s.method != "no_edges":
                        reconstructed.append(line)

        def fmt(lines: list[str]) -> str:
            return ", ".join(
                sorted(x for x in lines if not x.startswith("bit"))
                + sorted(x for x in lines if x.startswith("bit"))
            )

        out.append(
            dict(
                start_time=float(a),
                stop_time=float(z),
                reason=(
                    "replayed sync data (copied acquisition buffers)"
                    + (f": events unknown on {fmt(unknown)}" if unknown else "")
                    + (
                        f"; reconstructed: {fmt(reconstructed)}"
                        if reconstructed
                        else ""
                    )
                ),
            )
        )
    return out


# -------------------------------------------------------------------------- masks


def get_unknown_intervals(
    sync: npc_sync.SyncDataset | None, line: str | int
) -> tuple[tuple[float, float], ...]:
    """`sync.get_unknown_intervals(line)`, or () without sync or if the line isn't on the
    file."""
    if sync is None:
        return ()
    try:
        return sync.get_unknown_intervals(line)
    except (ValueError, IndexError, KeyError):  # line label not on this file
        return ()


def unknown_overlap_mask(
    start: npt.ArrayLike,
    stop: npt.ArrayLike,
    unknown: Iterable[tuple[float, float]],
) -> npt.NDArray[np.bool_]:
    """True where the closed interval [start, stop] (either order) overlaps an unknown
    interval. NaN bounds are ignored (a NaN stop checks `start` only).

    >>> unknown_overlap_mask([0, 2, 5], [1, 3, np.nan], [(1.5, 2.5), (4.9, 5.1)]).tolist()
    [False, True, True]
    """
    a = np.atleast_1d(np.asarray(start, dtype=float))
    z = np.atleast_1d(np.asarray(stop, dtype=float))
    z = np.where(np.isnan(z), a, z)
    a = np.where(np.isnan(a), z, a)
    lo, hi = np.minimum(a, z), np.maximum(a, z)
    mask = np.zeros(lo.shape, dtype=bool)
    for u0, u1 in unknown:
        mask |= (lo < u1) & (hi >= u0)
    return mask


def pair_edge_indices(
    rising: npt.ArrayLike, falling: npt.ArrayLike
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """Indices into sorted `rising`, `falling` of the pairs in `pair_rising_falling_edges`
    (without the unknown-interval filter).

    >>> [x.tolist() for x in pair_edge_indices([1, 3, 5], [0.5, 2, 6, 7])]
    [[0, 2], [1, 2]]
    """
    r = np.asarray(rising, dtype=float).ravel()
    f = np.asarray(falling, dtype=float).ravel()
    if r.size == 0 or f.size == 0:
        return np.array([], dtype=np.intp), np.array([], dtype=np.intp)
    idx = np.searchsorted(f, r, side="right")  # first falling edge after each rising edge
    next_r = np.r_[r[1:], np.inf]
    has_f = idx < f.size
    f_next = np.where(has_f, f[np.minimum(idx, f.size - 1)], np.inf)
    keep = has_f & (f_next <= next_r)
    return np.flatnonzero(keep), idx[keep]


def pair_rising_falling_edges(
    rising: npt.ArrayLike,
    falling: npt.ArrayLike,
    unknown: Iterable[tuple[float, float]] = (),
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Pair each rising edge with the first falling edge after it and before the next rising
    edge, without assuming the edges alternate. A rising edge with no such falling edge (e.g.
    the falling edge was cut off by a replayed interval) is dropped, as is a falling edge with
    no rising edge. Pairs whose [rising, falling] span overlaps an `unknown` interval are
    dropped too: events there come from another source (e.g. `lickFrames`).

    >>> r, f = pair_rising_falling_edges([1, 3, 5, 9], [0.5, 2, 6, 10], [(8.5, 9.5)])
    >>> r.tolist(), f.tolist()
    ([1.0, 5.0], [2.0, 6.0])
    """
    r = np.sort(np.asarray(rising, dtype=float).ravel())
    f = np.sort(np.asarray(falling, dtype=float).ravel())
    ri, fi = pair_edge_indices(r, f)
    r, f_next = r[ri], f[fi]
    unknown = tuple(unknown)
    if unknown and r.size:
        ok = ~unknown_overlap_mask(r, f_next, unknown)
        r, f_next = r[ok], f_next[ok]
    return r, f_next


def replayed_overlap_mask(
    sync: npc_sync.SyncDataset | None,
    start: npt.ArrayLike,
    stop: npt.ArrayLike,
) -> npt.NDArray[np.bool_]:
    """True where [start, stop] overlaps a replayed (invalid) interval of the sync file."""
    intervals = tuple(getattr(sync, "invalid_intervals", ()) or ())
    return unknown_overlap_mask(start, stop, intervals)


def stim_recordings_unknown_mask(
    recordings: Sequence[Any],
    sync: npc_sync.SyncDataset | None,
    line: str | int,
    include_offset: bool = True,
) -> npt.NDArray[np.bool_]:
    """True for npc_samstim recordings (sound/opto onsets found on a sync line) whose
    trigger-to-onset range (to the offset, if `include_offset`: opto offsets are falling edges
    on the line; sound offsets are onset + duration) overlaps an interval in which `line` is
    unknown: the edge may be missing there, and the edge found would then belong to another
    stimulus. False for None entries."""
    mask = np.zeros(len(recordings), dtype=bool)
    unknown = get_unknown_intervals(sync, line)
    if not unknown:
        return mask
    for i, rec in enumerate(recordings):
        if rec is None:
            continue
        trigger = getattr(
            getattr(rec, "presentation", None), "trigger_time_on_sync", None
        )
        times = [
            float(t)
            for t in (
                trigger,
                rec.onset_time_on_sync,
                rec.offset_time_on_sync if include_offset else None,
            )
            if t is not None and np.isfinite(t)
        ]
        if times and unknown_overlap_mask(min(times), max(times), unknown)[0]:
            mask[i] = True
    if mask.any():
        logger.info(
            f"{mask.sum()} stim recordings on sync line {line!r} masked: line unknown in a replayed interval"
        )
    return mask


def mask_unknown_stim_recordings(
    recordings: Sequence[Any] | None,
    sync: npc_sync.SyncDataset | None,
    line: str | int,
    include_offset: bool = True,
) -> tuple[Any, ...] | None:
    """`recordings` with None where `stim_recordings_unknown_mask` is True."""
    if recordings is None:
        return None
    mask = stim_recordings_unknown_mask(recordings, sync, line, include_offset)
    return tuple(None if m else r for r, m in zip(recordings, mask))


# -------------------------------------------------------------------------- tables


def get_reconstructed_lick_times(
    sync: npc_sync.SyncDataset, task_path: npc_io.PathLike
) -> npt.NDArray[np.float64]:
    """Lick onsets inside the lick line's unknown spans, from the task's `lickFrames`
    (`predict_lick_onsets`). Empty when the patch reconstructs the lick line: the onsets are
    on the line then."""
    unknown = sync.get_unknown_intervals(LICK_LINE)
    if not unknown:
        return np.array([])
    t, _ = predict_lick_onsets(sync, task_path)
    return t[npc_sync.in_intervals(t, unknown)]


def get_reconstructed_reward_times(
    sync: npc_sync.SyncDataset,
    stim_path: npc_io.PathLike,
    measured: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_], float]:
    """Reward (solenoid) onsets inside the reward line's unknown spans, from the stim file's
    `rewardFrames`/`manualRewardFrames`: flip time of the frame + the session's median
    measured latency (pairs: first measured onset within `REWARD_LATENCY_MAX_SEC` after a
    trigger frame's flip; 0.80 ms median, 0.69-1.35 ms, in event_latency_check.log).

    Returns (times, is_manual, latency used). Raises `SyncTimingError` if fewer than 5 pairs
    exist to measure the latency.
    """
    unknown = sync.get_unknown_intervals("reward_1_actuator")
    if not unknown:
        return np.array([]), np.array([], dtype=bool), np.nan
    data = npc_stim.get_h5_stim_data(stim_path)
    frames, manual = [], []
    for key in ("rewardFrames", "manualRewardFrames"):
        if key in data:
            v = np.asarray(data[key][()], dtype=float).ravel()
            v = v[np.isfinite(v)].astype(int)
            frames.extend(v)
            manual.extend([key == "manualRewardFrames"] * len(v))
    if not frames:
        return np.array([]), np.array([], dtype=bool), np.nan
    flips = npc_stim.get_flip_times(stim_path, sync=sync)
    f = np.asarray(frames)
    ok = (f >= 0) & (f < len(flips))
    t_trig, is_manual = flips[f[ok]], np.asarray(manual)[ok]
    m = np.sort(np.asarray(measured, dtype=float))
    j = np.searchsorted(m, t_trig)
    lat = np.where(j < len(m), m[np.clip(j, None, len(m) - 1)] - t_trig, np.inf)
    paired = (lat >= 0) & (lat <= REWARD_LATENCY_MAX_SEC)
    paired &= ~npc_sync.in_intervals(t_trig, unknown)
    if paired.sum() < 5:
        raise SyncTimingError(
            f"only {paired.sum()} measured rewards to estimate the frame->solenoid latency"
        )
    latency = float(np.median(lat[paired]))
    t = t_trig + latency
    inside = npc_sync.in_intervals(t, unknown)
    return t[inside], is_manual[inside], latency


def utc_now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
