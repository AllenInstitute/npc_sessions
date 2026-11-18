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
- persistence: the patch is cached as JSON keyed by session and sync content hash, with the
  builder and the versions of the packages that build it (`PATCH_DEPENDENCIES`); a cached
  patch is rebuilt (and overwritten, if writable) when any of them differ
- masks for trial events whose sync line is unknown inside a replayed interval, and
  reconstructed licks/rewards with a `timing_source` for the tables

Policy decisions (outputs/sync-frame-timing-upgrade-plan.md, notes/sync-patching-handoff-2.md):

- files with a timebase discontinuity (dropped or displaced buffers) are not patched: npc_stim
  still refuses them, so they stay rejected
- replayed intervals are NOT `invalid_times` (decided 2026-11-18): frame times are
  reconstructed there, so trials overlapping them are kept (`timing_source='patched'`).
  Events on lines that are unknown there are NaN in the trials tables instead: `response_time`
  (lick line), sound and opto onsets/offsets (their sync lines). See `unknown_overlap_mask`.
- licks inside replayed intervals are reconstructed from `lickFrames` for the licks table only
  (`timing_source='stim_frame'`, duration NaN); `response_time` is NaN there
- rewards inside replayed intervals are reconstructed from `rewardFrames` + the session's
  median measured frame->solenoid latency (`timing_source='stim_frame'`, duration NaN)
"""

from __future__ import annotations

import dataclasses
import datetime
import importlib.metadata
import json
import logging
import os
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

BUILDER = "npc_sessions.timing 2"
"""2: phantom-flicker removal and photodiode checks on every box-1 block (any file), box-1
prediction inside replayed spans from the block alignment."""
"""Recorded in each patch. Bump when the builder's output would change: cached patches with
another builder are rebuilt."""

PATCH_DIR_ENV_VAR = "NPC_SESSIONS_SYNC_PATCH_DIR"
"""Overrides `DEFAULT_PATCH_ROOT` (local dir or s3://). Set to 'none' to disable the cache."""

DEFAULT_PATCH_ROOT = "s3://aind-scratch-data/dynamic-routing/cache/sync_patches"
"""Where patches are cached as `<session>_<hash[:12]>.json` when AWS credentials are available."""

PATCH_DEPENDENCIES = ("npc_sessions", "npc_sync", "npc_stim")
"""Packages whose code determines a patch's content: their versions are saved with the patch
(`dependency_versions`), and a cached patch with different versions is rebuilt."""

MATCH_OFFSET_MIN_SEC = -1.5
MATCH_OFFSET_MAX_SEC = 60.0
"""Bounds on (first vsync of block) - (stim file startTime on the sync clock). Observed:
experiments/sync-patch-manifest/README.md."""

BARCODE_SPAN_PADDING_SEC = npc_sync.sync.BARCODE_MAX_DURATION_SEC
"""Barcode 'unknown' spans are widened by this on both sides, so a barcode that is only partly
inside a replayed interval is dropped rather than decoded to a wrong value."""

LICK_FRAME_MAX_ERROR_SEC = 1 / 120
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
            unmatched.append((path, "no vsync block starts near the stim file's start time"))
            continue
        j = int(np.flatnonzero(ok)[np.argmin(np.abs(off[ok]))])
        claims.setdefault(j, []).append(StimMatch(j, path, float(off[j]), n))
    matches = {}
    for j, c in claims.items():
        if len(c) == 1:
            matches[j] = c[0]
        else:
            unmatched.extend((m.stim_path, f"block {j} claimed by {len(c)} stim files") for m in c)
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
    sync: npc_sync.SyncPathOrDataset, stim_paths: Iterable[npc_io.PathLike]
) -> npc_sync.SyncPatch:
    """Patch for a sync file (see module docstring)."""
    raw = npc_sync.SyncDataset(npc_sync.get_sync_data(sync), on_invalid="ignore")
    stim_paths = tuple(stim_paths)
    try:
        stim_blocks = get_stim_blocks(raw, stim_paths)
        note = ""
    except (AssertionError, IndexError, ValueError) as exc:
        # sync-level failure (no vsync edges, unbalanced stim_running, ...): the patch can't
        # check or reconstruct anything that needs stim files, but other lines still apply
        stim_blocks, note = [], f"stim files could not be matched to vsync blocks: {exc!r}"
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
    return patch


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


def get_dependency_versions() -> dict[str, str]:
    """Installed versions of `PATCH_DEPENDENCIES` ('unknown' if not installed as a dist)."""
    out = {}
    for name in PATCH_DEPENDENCIES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = "unknown"
    return out


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
) -> npc_sync.SyncPatch | None:
    """The session's patch (every session: photodiode phantom-flicker removal applies to files
    without replays too).

    Loaded from the cache (`get_patch_root`: `root`, `$NPC_SESSIONS_SYNC_PATCH_DIR`, or
    `DEFAULT_PATCH_ROOT` on S3 when AWS credentials are available) if a patch exists there
    for the same sync content hash, builder and `PATCH_DEPENDENCIES` versions. Otherwise
    built, and saved there (overwriting a stale one) if `write`. Cache read/write failures
    are logged and the patch is built in memory.
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
                    and saved_versions == versions
                ):
                    return patch
                logger.info(
                    f"{session_id}: cached sync patch {path} is stale "
                    f"(builder {patch.builder!r}, versions {saved_versions}; now {BUILDER!r}, "
                    f"{versions}): rebuilding"
                )
        except Exception as exc:
            logger.warning(f"{session_id}: could not read cached sync patch {path}: {exc!r}")
    patch = build_session_sync_patch(raw, stim_paths)
    if path is not None and write:
        try:
            save_patch(patch, path)
            logger.info(f"{session_id}: saved sync patch to {path}")
        except Exception as exc:
            logger.warning(f"{session_id}: could not save sync patch to {path}: {exc!r}")
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
        diode = patch.diode_block(float(b[0]), float(b[-1])) if patch is not None else None
        row["diode_status"] = diode.status if diode is not None else None
        row["diode_reason"] = diode.reason if diode is not None else None
        row["n_phantom_flicker_removed"] = diode.n_phantom_flicker if diode and diode.status == "cleaned" else 0
        row["n_glitch_removed"] = diode.n_glitch if diode and diode.status == "cleaned" else 0
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
            row["reason"] = f"{len(b)} vsyncs, {m.n_frames} stim frames, no fault explains it"
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
                    + (f"; reconstructed: {fmt(reconstructed)}" if reconstructed else "")
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
        trigger = getattr(getattr(rec, "presentation", None), "trigger_time_on_sync", None)
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
    """Lick onsets inside the lick line's unknown spans, from the task's `lickFrames`.

    Outside replayed intervals, ~98% of `lickFrames` have a sync lick onset in the frame
    interval before the frame's input read (median 8.35 ms before it,
    experiments/sync-frame-timing/lick_count_check.log); the midpoint of that interval is used,
    so the error is within +-1/2 frame (+-8.3 ms at 60 Hz).
    """
    unknown = sync.get_unknown_intervals("lick_sensor")
    if not unknown:
        return np.array([])
    data = npc_stim.get_h5_stim_data(task_path)
    if "lickFrames" not in data:
        return np.array([])
    frames = np.asarray(data["lickFrames"][()], dtype=float).ravel()
    frames = frames[np.isfinite(frames)].astype(int)
    read = npc_stim.get_input_data_times(task_path, sync=sync)
    frames = frames[(frames >= 2) & (frames < len(read))]
    t = read[frames] - 0.5 * (read[frames] - read[frames - 1])
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
