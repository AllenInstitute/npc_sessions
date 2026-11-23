"""Sync timing for a session: build, validate, persist and apply a `npc_sync.SyncPatch`.

Every session with stim files gets a patch (`get_sync_patch`):

- Photodiode phantom flicker (diodeBoxFrameInterval 1): extra diode edges while a frame is held
  (a dropped frame in the stim's `frameIntervals`), in state-restoring pairs, at most one per
  missed refresh. The patch removes them, and comparator glitches, from the measured line, in
  files with or without replays, after checking that every diode edge of the block is explained;
  a block with unexplained edges is rejected (`SyncPatch.diode_blocks`). npc_stim uses a cleaned
  block as patched and removes nothing itself. Evidence: experiments/phantom-flicker,
  outputs/phantom-flicker-report.md.
- Dropped and replayed acquisition buffers (`npc_sync.timebase`): the sync program drops
  0.5 s buffers without leaving a gap in sample numbers, so later data is recorded early, and
  writes runs of identical buffers that make up for buffers dropped up to ~20 s earlier. The
  patch carries the file's timebase (dropped buffers located on the camera exposing lines):
  recorded data is moved to its true time, copies are removed, and the intervals that no
  recorded buffer holds are `npc_sync.SyncDataset.invalid_intervals`. There, instead of
  rejecting the session, the patch reconstructs the lines that can be reconstructed within
  known error (vsync from the stim file's `frameIntervals`, the digital photodiode from
  reconstructed vsyncs, camera exposures from their fixed period) and marks the rest
  'unknown' (see `npc_sync.patch.build_sync_patch`). Every time in a patch, and every time
  from `DynamicRoutingSession.sync_data`, is on that true clock. "Replayed interval" below
  means such an interval, whether its data was lost to a drop with or without copies.
  Evidence: experiments/sync-timebase-drops, outputs/sync-dropped-chunks-report.md.

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
  - photodiode: npc_sync predicts the digital photodiode edges there from reconstructed
    vsyncs + the measured latency (in effect a constant lag). For ephys sessions whose
    NI-DAQ records the analog photodiode (from 2025-01), they are replaced by the edges
    the comparator would have produced from the analog trace
    (`reconstruct_photodiode_line`; gates: <= 1 ms against every measured edge around the
    span, and one-to-one with npc_sync's prediction). Otherwise npc_sync's edges are kept
- persistence: the patch is cached as JSON keyed by session and sync content hash, with the
  builder and the versions of the packages that build it (`PATCH_DEPENDENCIES`); a cached
  patch is rebuilt (and overwritten, if writable) when the builder or the version of any
  `REUSE_DEPENDENCIES` differs. npc_sessions' own version is provenance only (`BUILDER` is
  bumped when its patch code changes); from a git checkout it includes the commit and a dirty
  flag, and a dirty checkout doesn't write to the shared `DEFAULT_PATCH_ROOT`
- reconstructed licks/rewards for the tables, and masks for trial events on lines still
  unknown inside a replayed interval

Policy decisions (outputs/sync-frame-timing-upgrade-plan.md, notes/sync-patching-handoff-2.md):

- dropped buffers are corrected by the patch's timebase (builder 5). A file whose barcode
  onset intervals are still irregular on that timebase (`timebase_discontinuities`: a drop
  the cameras couldn't locate) is not corrected: npc_stim still refuses it
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
import re
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import npc_io
import npc_stim
import npc_sync
import numpy as np
import numpy.typing as npt
import pandas as pd
import upath

logger = logging.getLogger(__name__)

BUILDER = "npc_sessions.timing 10"
"""Recorded in each patch. Bump when the builder's output would change: cached patches with
another builder are rebuilt.

- 2: phantom-flicker removal and photodiode checks on every box-1 block (any file), box-1
  prediction inside replayed spans from the block alignment
- 3: sound line reconstructed from the NI-DAQ (npc_samstim) and laser lines from opto
  trigger frames (npc_stim) inside replayed intervals
- 4: lick line reconstructed from the stim files' `lickFrames` inside replayed intervals,
  with synthetic falling edges so that the line alternates; lick spans that stay unknown are
  widened so that the line is low at both bounds (`reconstruct_lick_line`)
- 5: patches are built on the file's timebase (`npc_sync.SyncDataset.find_timebase`):
  dropped buffers located on camera lines, data recorded between a drop and the replayed
  run that makes up for it moved to its true time, and the dropped intervals reconstructed
  like replayed ones were
- 6, 7: barcode spans take in whole barcodes only (`barcode_spans`). Builder 5 still padded
  every dropped interval by 2 s and stopped where the line was low, which removed good
  barcodes and could stop between two pulses of one (703880_2024-04-18: 5 barcodes removed,
  one cut to 2 pulses, onset 765 ms late). 7: also removes the tail of a barcode whose onset
  fell inside a dropped interval (708020_2024-05-17, 714748_2024-06-28: 3 tails that 6 kept),
  and a barcode that may continue into one
- 8: the 2 s padding is back (6 and 7 dropped it on a timebase, which broke npc_ephys: see
  `BARCODE_SPAN_PADDING_SEC`), now with whole barcodes: the padded span takes in every
  barcode it touches
- 9: lick line state at each span bound read from the sync data words just outside the span
  (`_line_states_at_bounds`), not inferred from the nearest measured lick edges: in runs of
  spans with no lick edge between them, builders 4-8 gave every span the same synthetic edge
  (fake licks: 82 in 714748_2024-06-28, 5 in 708020_2024-05-17; #201). The patched lick
  line alternates by construction: no event between two spans means no lick edge, so one
  state across the gap; synthesized edges have distinct times (onsets deduplicated, an onset
  at the start of a span the line is already high in is the lick in progress). Asserted
  (`LickLineReconstructionError`, not caught)
- 10: photodiode edges inside replayed intervals from the NI-DAQ analog photodiode where it
  is recorded and passes its gates (`reconstruct_photodiode_line`). Patches of sessions
  without replayed intervals, without ephys, or without an analog photodiode channel are
  the same as with builder 9
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
"""Barcode 'unknown' spans are widened by this on both sides before whole barcodes are taken
in (`barcode_spans`). It must not be smaller than npc_ephys'
`SYNC_INVALID_INTERVAL_PADDING_S` (2 s): npc_ephys drops barcode edges, one by one, within
that distance of every invalid interval, and fails if that leaves rising and falling counts
unequal ('on and off times should be same length'), which it does if a barcode is left
there (builders 6 and 7 did, on a timebase: ephys timing failed for 703880_2024-04-18,
708020_2024-05-17 and 715709_2024-08-08). So barcodes within 2 s of a dropped interval are
removed even where they are complete."""

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

    `sync` should be an unpatched view (`on_invalid='ignore'`) on the file's timebase
    (`true_clock_view`): blocks are as measured, at their true times.

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


def true_clock_view(sync: npc_sync.SyncPathOrDataset) -> npc_sync.SyncDataset:
    """Unpatched view (`on_invalid='ignore'`) of a sync file on its timebase
    (`npc_sync.SyncDataset.find_timebase`, from the recorded data): measured events at
    their true times, without replayed copies. The view patches are built on."""
    recorded = npc_sync.SyncDataset(
        npc_sync.get_sync_data(sync), on_invalid="ignore", timebase=False
    )
    try:
        timebase: npc_sync.Timebase | None = recorded.find_timebase()
    except NotImplementedError:  # not a 32-bit counter file: as recorded
        timebase = None
    return npc_sync.SyncDataset(
        recorded, on_invalid="ignore", timebase=timebase if timebase else False
    )


def build_session_sync_patch(
    sync: npc_sync.SyncPathOrDataset,
    stim_paths: Iterable[npc_io.PathLike],
    ephys_recording_dirs: Iterable[npc_io.PathLike] | None = None,
) -> npc_sync.SyncPatch:
    """Patch for a sync file (see module docstring).

    With `ephys_recording_dirs`, the sound line is reconstructed inside replayed intervals
    from the NI-DAQ recording (`reconstruct_sound_line`), and the photodiode line from the
    NI-DAQ analog photodiode (`reconstruct_photodiode_line`). Laser lines are reconstructed
    from the stim files' opto trigger frames where possible (`reconstruct_opto_lines`).
    """
    raw = true_clock_view(sync)
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
        padding = BARCODE_SPAN_PADDING_SEC
        old_spans = patch.lines["barcode_ephys"]
        patch.lines["barcode_ephys"] = [
            dataclasses.replace(
                s,
                span=span,
                reason=s.reason
                + "; widened over every barcode that may be incomplete"
                + (f" (after {padding} s padding)" if padding else "")
                + ": no partial barcodes, rising/falling edges removed in pairs",
            )
            for s, span in zip(
                old_spans,
                barcode_spans([s.span for s in old_spans], rising, falling, padding),
                strict=True,
            )
        ]
    if patch.invalid_intervals and stim_paths:
        steps: list[tuple[Any, dict[str, Any]]] = [
            (reconstruct_lick_line, {}),
            (reconstruct_opto_lines, {}),
            (reconstruct_sound_line, {"ephys_recording_dirs": ephys_recording_dirs}),
            (
                reconstruct_photodiode_line,
                {"ephys_recording_dirs": ephys_recording_dirs},
            ),
        ]
        for fn, kwargs in steps:
            try:
                fn(raw, patch, stim_paths, **kwargs)
            except LickLineReconstructionError:
                raise
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


# ---------------------------------------------------------------- analog photodiode

DIODE_LINE = "stim_photodiode"
ANALOG_DIODE_METHOD = "nidaq_analog_photodiode"
"""`SpanPatch.method` of photodiode spans whose edges come from the NI-DAQ analog photodiode."""

ANALOG_DIODE_WINDOW_SEC = 10.0
"""Measured photodiode edges up to this long before and after a replayed span calibrate the
comparator model (thresholds, offsets) and gate it."""

ANALOG_DIODE_MIN_EDGES = 10
"""Measured edges needed per polarity on each side of the span."""

ANALOG_DIODE_MAX_DEVIATION_SEC = 0.001
"""Gate: every measured photodiode edge within `ANALOG_DIODE_WINDOW_SEC` of the span must have
an analog-derived edge of the same polarity within this, and vice versa (observed values:
hub/experiments/analog-photodiode-patch/README.md)."""

ANALOG_DIODE_SMOOTH_SEC = 0.0003
"""Boxcar applied to the analog trace before thresholding (9 samples at 30 kHz): the falling
phase is slow (~40 counts/ms at the comparator's threshold, with ~10 counts of noise)."""

ANALOG_DIODE_MIN_CORRELATION = 0.5
"""The photodiode channel is the NI-DAQ channel that correlates best with the digital line's
state around the span, if at least this (observed: 0.80-0.95 on AI0, |r| < 0.01 on the
others), ..."""
ANALOG_DIODE_MIN_CORRELATION_MARGIN = 0.2
"""... and at least this much above every other channel."""


def _diode_state_at(
    t: npt.NDArray[np.float64],
    rising: npt.NDArray[np.float64],
    falling: npt.NDArray[np.float64],
) -> npt.NDArray[np.bool_]:
    """Line state at times `t` from its edges (low before the first edge if it is rising)."""
    e = np.concatenate([rising, falling])
    pol = np.concatenate([np.ones(len(rising), bool), np.zeros(len(falling), bool)])
    order = np.argsort(e, kind="stable")
    e, pol = e[order], pol[order]
    if not len(e):
        return np.zeros(len(t), dtype=bool)
    j = np.searchsorted(e, t, side="right") - 1
    return np.where(j >= 0, pol[np.clip(j, 0, None)], ~pol[0])


def _unobserved_state_mask(
    t: npt.NDArray[np.float64],
    rising: npt.NDArray[np.float64],
    falling: npt.NDArray[np.float64],
    spans: Iterable[tuple[float, float]],
) -> npt.NDArray[np.bool_]:
    """True where the line state can't be read from the measured edges: inside each span,
    and after it until the first measured edge."""
    e = np.sort(np.concatenate([rising, falling]))
    mask = np.zeros(len(t), dtype=bool)
    for a, z in spans:
        j = int(np.searchsorted(e, z))
        mask |= (t >= a) & (t < (e[j] if j < len(e) else np.inf))
    return mask


def analog_diode_channel_correlations(
    t: npt.NDArray[np.float64],
    x: npt.NDArray[Any],
    rising: npt.ArrayLike,
    falling: npt.ArrayLike,
    spans: Iterable[tuple[float, float]] = (),
) -> list[float | None]:
    """Per channel (column of `x`, sampled at sync times `t`): correlation with the state of
    the digital photodiode line (measured `rising`/`falling` edges), outside `spans` (where
    the state is unknown). None for a constant channel (e.g. at the rail), or if the digital
    line doesn't change."""
    r = np.sort(np.asarray(rising, dtype=float))
    f = np.sort(np.asarray(falling, dtype=float))
    keep = ~_unobserved_state_mask(t, r, f, spans)
    state = _diode_state_at(t, r, f)[keep].astype(float)
    out: list[float | None] = []
    for ch in range(x.shape[1]):
        y = np.asarray(x[:, ch], dtype=float)[keep]
        if not len(y) or y.std() == 0 or state.std() == 0:
            out.append(None)
        else:
            out.append(float(np.corrcoef(y, state)[0, 1]))
    return out


def select_analog_diode_channel(
    correlations: Sequence[float | None],
    min_correlation: float = ANALOG_DIODE_MIN_CORRELATION,
    min_margin: float = ANALOG_DIODE_MIN_CORRELATION_MARGIN,
) -> int | None:
    """Index of the analog photodiode channel, or None if no channel stands out.

    >>> select_analog_diode_channel([0.92, -0.01, None, 0.0])
    0
    >>> select_analog_diode_channel([None, 0.9, 0.02])
    1
    >>> select_analog_diode_channel([0.3, 0.0]), select_analog_diode_channel([0.9, 0.8])
    (None, None)
    """
    c = np.array([-np.inf if v is None else v for v in correlations], dtype=float)
    if not len(c):
        return None
    best = int(np.argmax(c))
    others = np.delete(c, best)
    if c[best] < min_correlation:
        return None
    if len(others) and c[best] - others.max() < min_margin:
        return None
    return best


def schmitt_trigger_edges(
    y: npt.NDArray[np.float64], rise_level: float, fall_level: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """(rising, falling) edge positions, in fractional samples, of a comparator with
    hysteresis on `y`: it goes high when `y` rises through `rise_level` and low when `y`
    falls through `fall_level`.

    The state changes when `y` passes the outer of the two levels (so noise at one level
    can't toggle it), and each edge is timed at the last crossing of its own level before
    that, linearly interpolated.

    >>> y = np.array([0, 0, 4, 8, 8, 8, 6, 2, 0, 0, 8, 8], dtype=float)
    >>> r, f = schmitt_trigger_edges(y, rise_level=6, fall_level=3)
    >>> r.tolist(), f.tolist()
    ([2.5, 9.75], [6.75])
    """
    hi, lo = max(rise_level, fall_level), min(rise_level, fall_level)
    state = np.full(len(y), -1, dtype=np.int8)
    state[y > hi] = 1
    state[y < lo] = 0
    known = state >= 0
    if not known.any():
        return np.array([]), np.array([])
    idx = np.where(known, np.arange(len(y)), 0)
    np.maximum.accumulate(idx, out=idx)
    st = state[idx]
    first = int(np.argmax(known))
    change = np.flatnonzero(np.diff(st[first:]) != 0) + first + 1
    below_rise = np.flatnonzero(y < rise_level)
    above_fall = np.flatnonzero(y > fall_level)
    rising: list[float] = []
    falling: list[float] = []
    for i in change:
        up = st[i] == 1
        level = rise_level if up else fall_level
        other = below_rise if up else above_fall
        k = int(np.searchsorted(other, i)) - 1
        if k < 0:
            continue
        j = int(other[k])  # last sample on the far side of the edge's own level
        d = y[j + 1] - y[j]
        pos = j + ((level - y[j]) / d if d != 0 else 0.5)
        (rising if up else falling).append(float(pos))
    return np.asarray(rising, dtype=float), np.asarray(falling, dtype=float)


def _nearest_diff(
    x: npt.NDArray[np.float64], ref: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """x - nearest value in sorted `ref`, per x (inf if `ref` is empty)."""
    if not len(ref):
        return np.full(len(x), np.inf)
    j = np.searchsorted(ref, x)
    d0 = x - ref[np.clip(j - 1, 0, len(ref) - 1)]
    d1 = x - ref[np.clip(j, 0, len(ref) - 1)]
    return np.where(np.abs(d0) <= np.abs(d1), d0, d1)


def reconstruct_diode_edges_from_analog(
    t: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    rising: npt.ArrayLike,
    falling: npt.ArrayLike,
    span: tuple[float, float],
    predicted_rising: npt.ArrayLike,
    predicted_falling: npt.ArrayLike,
    other_spans: Iterable[tuple[float, float]] = (),
    window_sec: float = ANALOG_DIODE_WINDOW_SEC,
    max_deviation_sec: float = ANALOG_DIODE_MAX_DEVIATION_SEC,
    min_edges: int = ANALOG_DIODE_MIN_EDGES,
    smooth_sec: float = ANALOG_DIODE_SMOOTH_SEC,
) -> dict[str, Any]:
    """Digital photodiode edges inside `span` from the analog photodiode trace `y` (sampled
    at sync times `t`, evenly), as the comparator that makes the sync line would have
    produced them.

    The comparator is modelled as a threshold with hysteresis, measured from the data: the
    rising (falling) level is the median of the smoothed trace at the measured rising
    (falling) edges within `window_sec` of the span, and a per-polarity offset (median
    measured - analog-derived edge time) takes up what is left (the comparator's delay, the
    local error of the NI-DAQ -> sync time map).

    Returns a dict with `status`:

    - 'reconstructed', with `rising` and `falling` (sync seconds) to replace the predicted
      edges, if all gates pass:
      1. at least `min_edges` measured edges of each polarity on each side of the span
      2. around the span (the same windows), every measured edge has an analog-derived edge
         of its polarity within `max_deviation_sec`, and every analog-derived edge a
         measured one: the model reproduces the sync line there, edge for edge
      3. inside the span, the analog-derived edges are one-to-one with `predicted_rising`
         / `predicted_falling` (the patch's vsync-based reconstruction): same count per
         polarity, each less than half the shortest interval between predicted edges from
         its own predicted edge, and alternating with the measured edges on both sides. So
         the frame <-> edge assignment downstream is unchanged, only the edge times are
      4. no edge more than `max_deviation_sec` outside the span (edges less than that
         outside are put at its bound: the line was recorded there, without them)
    - 'failed', with `reason`, otherwise.

    Other keys are diagnostics (JSON-serialisable). `other_spans`: other replayed spans,
    where the line has no measured edges (not used for calibration or the gates).
    """
    a, z = span
    r = np.sort(np.asarray(rising, dtype=float))
    f = np.sort(np.asarray(falling, dtype=float))
    pred = {
        True: np.sort(np.asarray(predicted_rising, dtype=float)),
        False: np.sort(np.asarray(predicted_falling, dtype=float)),
    }
    info: dict[str, Any] = {"status": "failed"}

    def failed(reason: str) -> dict[str, Any]:
        info["reason"] = reason
        return info

    if len(t) < 3:
        return failed("no analog samples")
    dt = float((t[-1] - t[0]) / (len(t) - 1))
    k = max(int(round(smooth_sec / dt)), 1)
    k += (k + 1) % 2  # odd: no shift
    ys = np.convolve(np.asarray(y, dtype=float), np.ones(k) / k, mode="same")
    edge_pad = (k + 2) * dt
    t_lo, t_hi = float(t[0]) + edge_pad, float(t[-1]) - edge_pad
    other_spans = list(other_spans)
    excluded = [(a, z), *other_spans]
    all_e = np.sort(np.concatenate([r, f]))
    before = all_e[all_e < a]
    after = all_e[all_e >= z]
    if not len(before) or not len(after):
        return failed("no measured photodiode edges on one side of the span")
    last_before, first_after = float(before[-1]), float(after[0])
    # calibration/gate windows: up to `window_sec` on each side, inside the analog segment
    windows = [
        (max(a - window_sec, t_lo), a),
        (z, min(z + window_sec, t_hi)),
    ]

    def in_windows(x: npt.NDArray[np.float64], shrink: float = 0.0) -> list[Any]:
        # `shrink`: away from the window bounds and the spans, where an edge's counterpart
        # can be on the other side
        out = npc_sync.in_intervals(x, [(u - shrink, v + shrink) for u, v in excluded])
        return [(x >= w0 + shrink) & (x < w1 - shrink) & ~out for w0, w1 in windows]

    measured = {True: r, False: f}
    sides = {p: in_windows(measured[p]) for p in (True, False)}
    counts = [int(m.sum()) for p in (True, False) for m in sides[p]]
    info["n_calibration_edges"] = counts  # rising before/after, falling before/after
    if min(counts) < min_edges:
        return failed(
            f"too few measured photodiode edges around the span (rising before/after, "
            f"falling before/after: {counts}; {min_edges} needed)"
        )
    cal = {p: measured[p][sides[p][0] | sides[p][1]] for p in (True, False)}
    level = {p: float(np.median(np.interp(cal[p], t, ys))) for p in (True, False)}
    lo_level, hi_level = (float(v) for v in np.percentile(ys, [1, 99]))
    info.update(
        rise_level=level[True],
        fall_level=level[False],
        black_level=lo_level,
        white_level=hi_level,
        smooth_samples=k,
    )
    if not (lo_level < min(level.values()) and max(level.values()) < hi_level):
        return failed("comparator levels are outside the range of the analog trace")
    pos_r, pos_f = schmitt_trigger_edges(ys, level[True], level[False])
    analog = {
        True: np.interp(pos_r, np.arange(len(t)), t),
        False: np.interp(pos_f, np.arange(len(t)), t),
    }
    # offsets, then gate 2: the model reproduces the measured line around the span
    worst = 0.0
    for p, name in ((True, "rising"), (False, "falling")):
        if not len(analog[p]):
            return failed(f"no {name} edges in the analog trace")
        d = _nearest_diff(cal[p], analog[p])
        offset = float(np.median(d))
        analog[p] = analog[p] + offset
        resid = d - offset
        extra = analog[p][np.logical_or(*in_windows(analog[p], 2 * max_deviation_sec))]
        resid_extra = _nearest_diff(extra, cal[p])
        info[f"{name}_offset_sec"] = offset
        info[f"{name}_residual_ms"] = {
            "n": int(len(resid)),
            "median_abs": float(np.median(np.abs(resid)) * 1e3),
            "p99_abs": float(np.percentile(np.abs(resid), 99) * 1e3),
            "max_abs": float(np.max(np.abs(resid)) * 1e3),
        }
        n_bad = int((np.abs(resid) > max_deviation_sec).sum())
        n_extra = int((np.abs(resid_extra) > max_deviation_sec).sum())
        if n_bad or n_extra:
            return failed(
                f"analog photodiode doesn't reproduce the measured {name} edges around the "
                f"span: {n_bad} of {len(resid)} measured edges without an analog edge within "
                f"{max_deviation_sec * 1e3:.1f} ms (max {np.max(np.abs(resid)) * 1e3:.2f} "
                f"ms), {n_extra} analog edges without a measured one"
            )
        worst = max(worst, float(np.max(np.abs(resid))))
    info["max_abs_residual_sec"] = worst
    # gate 3: inside the span, one-to-one with the predicted edges
    margin = 2 * max_deviation_sec
    new = {
        p: analog[p][
            (analog[p] > last_before + margin) & (analog[p] < first_after - margin)
        ]
        for p in (True, False)
    }
    info["n_edges"] = [int(len(new[True])), int(len(new[False]))]
    info["n_predicted"] = [int(len(pred[True])), int(len(pred[False]))]
    if info["n_edges"] != info["n_predicted"]:
        return failed(
            f"{info['n_edges']} (rising, falling) analog photodiode edges in the span, "
            f"{info['n_predicted']} predicted from vsyncs"
        )
    all_pred = np.sort(np.concatenate([pred[True], pred[False]]))
    tol = 0.5 * float(np.min(np.diff(all_pred))) if len(all_pred) > 1 else 1 / 120
    dev = np.concatenate([new[p] - pred[p] for p in (True, False)])
    if len(dev):
        info["analog_minus_predicted_ms"] = {
            "min": float(dev.min() * 1e3),
            "median": float(np.median(dev) * 1e3),
            "max": float(dev.max() * 1e3),
        }
        if np.max(np.abs(dev)) >= tol:
            return failed(
                f"analog photodiode edges are up to {np.max(np.abs(dev)) * 1e3:.1f} ms from "
                f"the edges predicted from vsyncs (>= {tol * 1e3:.1f} ms: assignment "
                "ambiguous)"
            )
    kept_r = r[(r <= last_before) | (r >= first_after)]
    kept_f = f[(f <= last_before) | (f >= first_after)]

    def local(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.sort(x[(x >= a - 1.0) & (x < z + 1.0)])

    if _alternation_errors(
        local(np.r_[kept_r, new[True]]), local(np.r_[kept_f, new[False]]), other_spans
    ) > _alternation_errors(
        local(np.r_[kept_r, pred[True]]), local(np.r_[kept_f, pred[False]]), other_spans
    ):
        return failed(
            "analog photodiode edges don't alternate with the measured edges around the span"
        )
    # the line was recorded outside [a, z) and has no such edge there: an analog-derived edge
    # just outside (by less than the model's error) is put at the bound. This also keeps it
    # inside the span for consumers that go by time (npc_stim's `frame_source`)
    clamped = {p: np.clip(new[p], a, np.nextafter(z, -np.inf)) for p in (True, False)}
    moved = np.concatenate([np.abs(clamped[p] - new[p]) for p in (True, False)])
    info["n_moved_to_span_bounds"] = int((moved > 0).sum())
    info["max_moved_to_span_bounds_sec"] = float(moved.max()) if len(moved) else 0.0
    if len(moved) and moved.max() > max_deviation_sec:
        return failed(
            f"an analog photodiode edge is {moved.max() * 1e3:.2f} ms outside the span, "
            "where the line was recorded without it"
        )
    info.update(status="reconstructed", rising=clamped[True], falling=clamped[False])
    return info


def reconstruct_photodiode_line(
    raw: npc_sync.SyncDataset,
    patch: npc_sync.SyncPatch,
    stim_paths: Sequence[npc_io.PathLike] = (),
    ephys_recording_dirs: Iterable[npc_io.PathLike] | None = None,
) -> dict[str, Any]:
    """Replace, in place, the photodiode edges that npc_sync predicted inside replayed
    intervals (reconstructed vsync + measured latency: `npc_sync.patch.predict_diode_edges`
    / `predict_box1_diode_edges`) with edges from the analog photodiode recorded on the
    ephys NI-DAQ, so that frames there get measured display times.

    The NI-DAQ runs on its own clock and isn't affected by the sync program's dropped
    buffers. Its samples are mapped to sync time by npc_ephys' fit to the barcodes of the
    whole recording (those within 2 s of a replayed interval excluded), so the map doesn't
    depend on the lost sync data; its local error is taken up by the offsets measured
    around each span (`reconstruct_diode_edges_from_analog`).

    Per span with predicted edges: the channel is found by correlation with the digital
    line around the span (`select_analog_diode_channel`: AI0 in every session checked, but
    not assumed; if no channel stands out around the first span, the recording has no
    analog photodiode and the other spans aren't read), and the span's edges are replaced
    only if every gate of
    `reconstruct_diode_edges_from_analog` passes; the new `SpanPatch` has method
    `ANALOG_DIODE_METHOD`, the model's parameters and residuals in `diagnostics`, and the
    edges it replaces in `diagnostics['replaced']`. Otherwise the span is left as it was,
    with a note. Nothing is read, and nothing changes, without `ephys_recording_dirs` or
    without such spans. Sessions before 2025-01, whose AI0 is at the rail, keep npc_sync's
    prediction (no channel correlates), as do spans npc_sync left 'unknown'.

    `stim_paths` isn't used (same signature as the other steps). Returns diagnostics per
    span.
    """
    out: dict[str, Any] = {}
    spans = patch.lines.get(DIODE_LINE, [])
    targets = [
        s
        for s in spans
        if s.status == "reconstructed"
        and s.method not in ("no_edges", ANALOG_DIODE_METHOD)
        and (len(s.rising) or len(s.falling))
    ]
    if not targets or not ephys_recording_dirs:
        return out
    import npc_ephys

    recording_dirs = tuple(ephys_recording_dirs)
    view = _patched_view(raw, patch)
    device = npc_ephys.get_pxi_nidaq_info(recording_dirs)
    nidaq = npc_ephys.get_ephys_timing_on_sync(
        sync=view, recording_dirs=recording_dirs, devices=(device,)
    )[0]
    if not nidaq.is_valid or nidaq.drops:
        patch.notes.append(
            f"{DIODE_LINE}: analog photodiode not used: NI-DAQ timing "
            + (f"failed ({nidaq.reason})" if not nidaq.is_valid else "has lost samples")
        )
        return out
    data = npc_ephys.get_pxi_nidaq_data(*recording_dirs, device_name=device.device.name)
    n_samples = int(data.shape[0])
    measured = {}
    for kind in ("rising", "falling"):
        e = np.asarray(view.get_edges(kind, DIODE_LINE, units="seconds"), dtype=float)
        measured[kind] = e[view.get_edge_sources(DIODE_LINE, kind) == npc_sync.MEASURED]
    all_spans = [s.span for s in spans]
    new_spans = []
    no_channel = False
    for s in spans:
        if no_channel or not any(s is x for x in targets):
            new_spans.append(s)
            continue
        a, z = s.span
        key = f"{a:.3f}-{z:.3f} s"
        pad = ANALOG_DIODE_WINDOW_SEC + 0.5
        k0 = max(int((a - pad - nidaq.start_time) * nidaq.sampling_rate), 0)
        k1 = min(int((z + pad - nidaq.start_time) * nidaq.sampling_rate), n_samples)
        result: dict[str, Any]
        channel = None
        if k1 - k0 < 2:
            result = {"status": "failed", "reason": "span outside the NI-DAQ recording"}
        else:
            x = np.asarray(data[k0:k1])
            t = np.asarray(nidaq.to_sync_time(np.arange(k0, k1)), dtype=float)
            correlations = analog_diode_channel_correlations(
                t, x, measured["rising"], measured["falling"], all_spans
            )
            channel = select_analog_diode_channel(correlations)
            if channel is None:
                no_channel = s is targets[0]
                result = {
                    "status": "failed",
                    "reason": "no NI-DAQ channel follows the digital photodiode line "
                    f"(correlations: {[None if c is None else round(c, 3) for c in correlations]})",
                }
            else:
                result = reconstruct_diode_edges_from_analog(
                    t,
                    x[:, channel].astype(float),
                    measured["rising"],
                    measured["falling"],
                    (a, z),
                    s.rising,
                    s.falling,
                    other_spans=[sp for sp in all_spans if sp != s.span],
                )
            result["channel_correlations"] = correlations
        diag = {k: v for k, v in result.items() if k not in ("rising", "falling")}
        diag.update(
            nidaq_device=device.device.name,
            nidaq_channel=channel,
            nidaq_sampling_rate=nidaq.sampling_rate,
            nidaq_start_time=nidaq.start_time,
        )
        out[key] = diag
        if result["status"] != "reconstructed":
            patch.notes.append(
                f"{DIODE_LINE} {key}: analog photodiode not used ({s.method} kept): "
                f"{result['reason']}"
            )
            new_spans.append(s)
            continue
        diag["replaced"] = {
            "method": s.method,
            "error_bound_sec": s.error_bound_sec,
            "rising": [float(v) for v in s.rising],
            "falling": [float(v) for v in s.falling],
            "diagnostics": s.diagnostics,
        }
        new_spans.append(
            npc_sync.SpanPatch(
                span=s.span,
                status="reconstructed",
                rising=np.asarray(result["rising"], dtype=float),
                falling=np.asarray(result["falling"], dtype=float),
                method=ANALOG_DIODE_METHOD,
                error_bound_sec=ANALOG_DIODE_MAX_DEVIATION_SEC,
                diagnostics=diag,
                reason="edges of the NI-DAQ analog photodiode (comparator thresholds and "
                "offsets measured from the edges around the span), replacing the edges "
                f"predicted from vsyncs ({s.method})",
            )
        )
    patch.lines[DIODE_LINE] = new_spans
    return out


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


def _measured_kept(
    raw: npc_sync.SyncDataset, line: str, spans: Sequence[tuple[float, float]]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Recorded rising/falling edges on `line` that the patched view keeps: outside every
    span [a, z) (`npc_sync.SyncPatch.apply`). Edges at the first sample of a span (a drop or
    copy boundary, which can carry the state of the data after it) are removed there, so the
    line state at the bounds is taken from `_line_states_at_bounds`, not from them."""
    r = np.asarray(raw.get_rising_edges(line, units="seconds"), dtype=float)
    f = np.asarray(raw.get_falling_edges(line, units="seconds"), dtype=float)
    return r[~npc_sync.in_intervals(r, spans)], f[~npc_sync.in_intervals(f, spans)]


def _line_states_at_bounds(
    raw: npc_sync.SyncDataset, line: str, spans: Sequence[tuple[float, float]]
) -> dict[tuple[float, float], tuple[bool | None, bool | None]]:
    """Per span [a, z): (line high just before `a`, line high just after `z`), read from the
    data words of the recorded events nearest the bounds outside the span. Every event, on any
    line, carries the state of every line, so this is observed even where `line` has no edge
    for a long time (vsync and camera events are a few ms apart: <= 6.2 ms from the bounds in
    708020_2024-05-17 and 714748_2024-06-28). None where no event lies between the span and
    its neighbour (or the file's start/end): the state isn't observed there (#201).

    Before `a`: the state after the last event with t < a. After `z`: the state before the
    first event with t >= z. Where that event is an edge on `line` (as the patched view keeps
    them, `_measured_kept`), its kind gives the state, so the result agrees with the edges
    around the span even where the recorded data inside it (replayed copies) doesn't."""
    t = np.asarray(raw.get_all_times("seconds"), dtype=float)
    bit = np.asarray(raw.get_bit(raw._line_to_bit(line)), dtype=np.int8)
    rising, falling = _measured_kept(raw, line, spans)
    ordered = sorted(spans)
    out: dict[tuple[float, float], tuple[bool | None, bool | None]] = {}
    for k, (a, z) in enumerate(ordered):
        prev_z = ordered[k - 1][1] if k else -np.inf
        next_a = ordered[k + 1][0] if k + 1 < len(ordered) else np.inf
        i = int(np.searchsorted(t, a, side="left")) - 1
        j = int(np.searchsorted(t, z, side="left"))
        before: bool | None = None
        if i >= 0 and t[i] >= prev_z:
            high, last_r = _line_state_before(rising, falling, a)
            last_f = falling[falling < a]
            last_edge = max(last_r, float(last_f[-1]) if len(last_f) else -np.inf)
            before = high if last_edge >= t[i] else bool(bit[i])
        after: bool | None = None
        if j < len(t) and t[j] < next_a:
            r, f = rising[rising >= z], falling[falling >= z]
            r0 = float(r[0]) if len(r) else np.inf
            f0 = float(f[0]) if len(f) else np.inf
            after = (f0 < r0) if min(r0, f0) <= t[j] else bool(bit[j])
        out[(a, z)] = (before, after)
    return out


def _alternation_errors(
    rising: npt.NDArray[np.float64],
    falling: npt.NDArray[np.float64],
    unknown: Sequence[tuple[float, float]] = (),
) -> int:
    """Number of adjacent same-kind edges (0 if the line alternates). Pairs with an
    `unknown` span between them aren't counted: their edges there are missing.

    >>> _alternation_errors(np.array([1.0, 3.0]), np.array([2.0, 4.0]))
    0
    >>> _alternation_errors(np.array([1.0, 2.0]), np.array([3.0]))
    1
    >>> _alternation_errors(np.array([1.0, 2.0]), np.array([3.0]), [(1.2, 1.5)])
    0
    """
    t = np.concatenate([rising, falling])
    order = np.argsort(t, kind="stable")
    t = t[order]
    k = np.r_[np.ones(len(rising)), np.zeros(len(falling))][order]
    same = np.flatnonzero(k[1:] == k[:-1])
    return sum(
        1 for i in same if not any(t[i] < z and a < t[i + 1] for a, z in unknown)
    )


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
    - onsets are deduplicated (closer than `min_gap`: one lick), and with `high_at_entry` an
      onset within `2 * min_gap` of `a` is the lick in progress, not a new one, so every
      edge time is distinct and the edges alternate from the entry state to the exit state
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
    on = np.unique(on[(on >= a) & (on < z)])  # sorted, no duplicates
    if len(on) > 1:  # onsets closer than `min_gap` are one lick
        on = on[np.r_[True, np.diff(on) >= min_gap]]
    if high_at_entry and len(on) and on[0] < a + 2 * min_gap:
        # the line is already high at `a`, with no time to fall before this onset: it's the
        # lick in progress, not a new one
        on = on[1:]
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


class LickLineReconstructionError(RuntimeError):
    """The patched lick line would not alternate: a bug in `reconstruct_lick_line` (the
    recorded line always alternates). Not caught by `build_session_sync_patch`."""


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

    The line state at each bound is read from the sync data words just outside the span
    (`_line_states_at_bounds`), not inferred from the nearest measured lick edges: in a run of
    spans with no lick edge between them, that inference gave every span the same synthetic
    edge (fake licks, #201). Where no event of any line separates two spans, there is no lick
    edge between them either, so the line holds one state across that gap: it is taken to be
    the state at entry to the first span of such a run (the last span takes the line to the
    observed state after the run).

    The recorded line always alternates, and the edges synthesized here go from the state at
    each span's entry to the state at its exit, so the patched line alternates too. If it
    doesn't, the reconstruction is wrong (a bug, not a property of the data):
    `LickLineReconstructionError` is raised and the patch isn't built. Synthetic onsets without
    a `lickFrames` onset (a lick in progress at a span's end) are counted in the diagnostics.

    Spans that no stim file covers stay unknown (see `widen_unknown_lick_spans`). Returns
    diagnostics.
    """
    unknown = [s.span for s in _unknown_spans(patch, LICK_LINE)]
    if not unknown:
        return {}
    all_spans = [s.span for s in patch.lines[LICK_LINE]]
    rising, falling = _measured_kept(raw, LICK_LINE, all_spans)
    duration = _median_lick_duration(rising, falling)
    bounds = _line_states_at_bounds(raw, LICK_LINE, all_spans)
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
    new: dict[tuple[float, float], npc_sync.SpanPatch] = {}
    n_without_onset = 0
    exit_high = False  # state at the end of the previous span
    for s in sorted(patch.lines[LICK_LINE], key=lambda s: s.span):
        a, z = s.span
        before, after = bounds[s.span]
        high_at_entry = exit_high if before is None else before
        # no event (so no lick edge) between this span and the next: the line holds its
        # state across the gap, taken as the state at entry to this span
        high_at_exit = high_at_entry if after is None else after
        exit_high = high_at_exit
        if s.status != "unknown":
            r_s, f_s = np.asarray(s.rising), np.asarray(s.falling)
            if after is None and (len(r_s) or len(f_s)):
                exit_high = bool(
                    (r_s.max() if len(r_s) else -np.inf)
                    > (f_s.max() if len(f_s) else -np.inf)
                )
            continue
        source = next(
            ((n, on) for n, on, t0, t1 in predictions if t0 <= a and z <= t1), None
        )
        if source is None:
            info["spans"].append({"span": [a, z], "status": "unknown"})
            continue
        name, onsets = source
        last_r = _line_state_before(
            np.sort(np.r_[rising, synth_r]), np.sort(np.r_[falling, synth_f]), a
        )[1]
        r, f = synthesize_lick_edges(
            onsets, (a, z), high_at_entry, last_r, high_at_exit, duration
        )
        synth_r.extend(r.tolist())
        synth_f.extend(f.tolist())
        n_onsets = int(np.sum((onsets >= a) & (onsets < z)))
        n_without_onset += len(r) - n_onsets
        n_checked, frac = _lick_frame_match_fraction(
            onsets, rising, [*all_spans, *raw._mask_intervals]
        )
        diag = dict(
            stim=name,
            n_onsets=n_onsets,
            high_at_entry=high_at_entry,
            high_at_exit=high_at_exit,
            state_at_bounds=(
                "observed" if before is not None and after is not None else "carried"
            ),
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
                f"onsets from {name} lickFrames (midpoint of the frame interval before "
                "the input read, +-1/2 frame); offsets synthetic (onset + median measured "
                "lick duration, shortened to fit), plus a synthetic edge for a lick in "
                "progress at either bound (line state at the bounds from the sync data "
                "words), so the line alternates"
            ),
        )
    still_unknown = [sp for sp in unknown if sp not in new]
    errors = _alternation_errors(
        np.sort(np.r_[rising, synth_r]), np.sort(np.r_[falling, synth_f]), still_unknown
    )
    info["n_synthetic_onsets_without_lick_frame"] = n_without_onset
    info["alternation_errors"] = errors
    if errors:
        raise LickLineReconstructionError(
            f"{raw}: the reconstructed lick line would not alternate ({errors} adjacent "
            "same-kind edges): reconstruct_lick_line is inconsistent with the recorded line"
        )
    patch.lines[LICK_LINE] = [new.get(s.span, s) for s in patch.lines[LICK_LINE]]
    return info


def widen_unknown_lick_spans(
    raw: npc_sync.SyncDataset, patch: npc_sync.SyncPatch
) -> None:
    """Widen the lick line's remaining 'unknown' spans, in place, so the line is low at both
    bounds. A lick in progress at the start (line high before it: from the last measured
    rising edge) or at the end (line high after it: to the first measured falling edge) is
    included whole, so the measured edges outside still alternate. The state at the bounds is
    read from the sync data words (`_line_states_at_bounds`)."""
    unknown = [s.span for s in _unknown_spans(patch, LICK_LINE)]
    if not unknown:
        return
    all_spans = [s.span for s in patch.lines[LICK_LINE]]
    rising, falling = _measured_kept(raw, LICK_LINE, all_spans)
    bounds = _line_states_at_bounds(raw, LICK_LINE, all_spans)
    out = []
    for s in patch.lines[LICK_LINE]:
        if s.status != "unknown":
            out.append(s)
            continue
        a, z = s.span
        before, after = bounds[s.span]
        r = rising[rising < a]
        if before and len(r):
            a = float(r[-1])
        f = falling[falling >= z]
        if after and len(f):
            z = float(f[0]) + 1 / raw.sample_freq
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


def barcode_spans(
    spans: Sequence[tuple[float, float]],
    rising: npt.ArrayLike,
    falling: npt.ArrayLike,
    padding: float = 0.0,
) -> list[tuple[float, float]]:
    """Each span (an interval without real data on the barcode line), widened to take in,
    whole, every barcode that isn't certainly complete; other barcodes are left alone.

    A barcode is a group of pulses whose rising edges are less than
    `npc_sync.sync.BARCODE_MIN_GAP_SEC` apart and that lasts at most
    `BARCODE_MAX_DURATION_SEC` (longer groups aren't barcodes: only a pulse in progress at a
    bound is taken in). A barcode is removed if

    - it overlaps the span, or
    - it starts less than the maximum barcode duration before the span: it may continue
      into it, or
    - it starts less than that after the span and its onset is off the barcode period
      (`BARCODE_INTERVAL_TOLERANCE_SEC`), measured from the nearest barcode that is far from
      every span: it is the tail of a barcode that started inside the span. If no such
      reference exists, it is removed.

    >>> r = np.concatenate([t + 0.1 * np.arange(8) for t in (10, 41, 72, 103)])
    >>> f = r + 0.04
    >>> [(a, round(z, 6)) for a, z in barcode_spans([(10.35, 10.5)], r, f)]  # inside the first
    [(10.0, 10.74)]
    >>> barcode_spans([(40.2, 40.7)], r, f)  # ends 0.3 s before the second, on time: kept
    [(40.2, 40.7)]
    >>> tail = r[~((r >= 41) & (r < 41.25))]  # the second's onset was lost in the span
    >>> [(a, round(z, 6)) for a, z in barcode_spans([(40.75, 41.25)], tail, tail + 0.04)]
    [(40.75, 41.74)]
    >>> [(a, round(z, 6)) for a, z in barcode_spans([(42.0, 42.5)], r, f)]  # may continue into it
    [(41.0, 42.5)]
    """
    r = np.sort(np.asarray(rising, dtype=float))
    f = np.sort(np.asarray(falling, dtype=float))
    max_dur = npc_sync.sync.BARCODE_MAX_DURATION_SEC
    padded = [(max(0.0, a - padding), z + padding) for a, z in spans]
    onsets = np.array([])
    ends = np.array([])
    if r.size:
        first = np.flatnonzero(
            np.r_[True, np.diff(r) > npc_sync.sync.BARCODE_MIN_GAP_SEC]
        )
        last_rising = r[np.r_[first[1:] - 1, r.size - 1]]
        j = np.searchsorted(f, last_rising)
        end = np.where(j < f.size, f[np.clip(j, 0, max(f.size - 1, 0))], last_rising)
        is_barcode = end - r[first] <= max_dur
        onsets, ends = r[first][is_barcode], end[is_barcode]
    far = np.ones(onsets.size, dtype=bool)
    for a, z in padded:
        far &= (onsets + max_dur <= a) | (onsets >= z + max_dur)
    ref = onsets[far]
    # period from all barcodes: most intervals are one period, whatever is wrong with a few
    period = np.nan
    if onsets.size >= 3 and ref.size:
        d = np.diff(onsets)
        single = np.abs(d / np.median(d) - 1) < 0.05
        if single.any():
            period = float(np.median(d[single]))

    def on_period(t: float) -> bool:
        if not np.isfinite(period):
            return False
        dt = t - ref[np.argmin(np.abs(ref - t))]
        dev = dt - round(dt / period) * period
        return bool(abs(dev) <= npc_sync.sync.BARCODE_INTERVAL_TOLERANCE_SEC)

    out = []
    for (a, z), (a0, z0) in zip(padded, spans, strict=True):
        lo, hi = a, z
        for b0, b1 in zip(onsets, ends, strict=True):
            remove = (
                (b1 >= a and b0 < z)  # touches the padded span
                or (b0 < a0 < b0 + max_dur)  # may continue into the span
                or (z0 <= b0 < z0 + max_dur and not on_period(float(b0)))  # a tail
            )
            if remove:
                lo, hi = min(lo, float(b0)), max(hi, float(b1) + 1e-9)
        out.append(_span_with_line_low_at_bounds((lo, hi), r, f))
    return out


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
    install), else None. Dirty means uncommitted changes under the package's source dir.
    """
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
    display_time_errors: Mapping[str, Exception] | None = None,
) -> pd.DataFrame:
    """One row per vsync block: matched stim, status ('clean' / 'patched' / 'rejected'),
    reason, frame counts and the gate values from the patch.

    With `display_times`: their source ('photodiode' / 'constant_lag' / 'unavailable'), the
    number of interpolated and constant-lag frames, and `npc_stim.BlockDisplayTimes.notes`.
    A block whose photodiode wasn't used is 'rejected' (the reason says why) even if it has
    constant-lag display times.

    `display_time_errors` (stim file stem -> Exception, as returned for stims without
    display times) gives the reason for a matched block whose `display_times` entry has
    no stim file."""
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
        # how the photodiode edges inside the block's replayed spans were obtained
        # (`ANALOG_DIODE_METHOD`: measured on the NI-DAQ analog photodiode; the frames are
        # still counted in `n_interpolated_frames`: their vsyncs are reconstructed)
        row["photodiode_span_methods"] = (
            [
                s.method
                for s in patch.lines.get(DIODE_LINE, [])
                if s.span[0] <= b[-1] and s.span[1] >= b[0]
            ]
            if patch is not None
            else []
        )
        if display_times is not None and j < len(display_times):
            dt = display_times[j]
            row["display_time_source"] = dt.source
            row["n_interpolated_frames"] = dt.n_interpolated
            row["n_constant_lag_frames"] = dt.n_constant_lag_frames
            row["display_time_notes"] = dt.notes
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
            reason = display_times[j].reason
            if (
                display_times[j].stim_path is None
                and display_time_errors
                and m is not None
                and m.stim_path.stem in display_time_errors
            ):
                reason = _display_time_error_reason(
                    display_time_errors[m.stim_path.stem]
                )
            row["reason"] = f"display times: {reason}"
        rows.append(row)
    return pd.DataFrame(rows)


def _display_time_error_reason(exc: Exception) -> str:
    """The reason in an npc_stim 'No frame display times for <stim> (sync block n): <reason>'
    error, else the error's repr."""
    match = re.match(
        r"No frame display times for .*?\(sync block \d+\): (.*)", str(exc), re.S
    )
    return match.group(1) if match else repr(exc)


def get_replayed_intervals(sync: npc_sync.SyncDataset) -> list[dict[str, Any]]:
    """One entry per interval of a patched file whose sync data was lost to dropped
    acquisition buffers (`sync.invalid_intervals`): `start_time`, `stop_time` and a `reason`
    listing the lines whose events are unknown there and those reconstructed.

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
                    "sync data not recorded (dropped acquisition buffers)"
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
    idx = np.searchsorted(
        f, r, side="right"
    )  # first falling edge after each rising edge
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
