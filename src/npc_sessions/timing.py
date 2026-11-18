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
- persistence: the patch is cached as JSON keyed by session, sync content hash and builder
- what goes in `invalid_times` (time ranges where trial timing can't be trusted), and
  reconstructed licks/rewards with a `timing_source` for the tables

Policy decisions (outputs/sync-frame-timing-upgrade-plan.md):

- files with a timebase discontinuity (dropped or displaced buffers) are not patched: npc_stim
  still refuses them, so they stay rejected
- trials overlapping a replayed interval are excluded via `invalid_times`: licks, sound and
  opto edges are unknown there. OPEN DECISION: keeping them would need reconstructed licks
  (+-8.3 ms) in the trials table, and sound/opto onsets from a source other than sync
- licks inside replayed intervals are reconstructed from `lickFrames` for the licks table only
  (`timing_source='stim_frame'`, duration NaN)
- rewards inside replayed intervals are reconstructed from `rewardFrames` + the session's
  median measured frame->solenoid latency (`timing_source='stim_frame'`, duration NaN)
"""

from __future__ import annotations

import dataclasses
import datetime
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
"""Directory (local or s3://) where patches are cached as `<session>_<hash[:12]>.json`. If
unset, patches are built in memory and not saved. TODO: `npc_lims.get_sync_patch_path`."""

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
        except Exception as exc:  # noqa: BLE001
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


def get_patch_path(
    session_id: str, content_hash: str, root: npc_io.PathLike | None = None
) -> upath.UPath | None:
    root = root or os.environ.get(PATCH_DIR_ENV_VAR)
    if not root:
        return None
    return npc_io.from_pathlike(root) / f"{session_id}_{content_hash[:12]}.json"


def get_sync_patch(
    session_id: str,
    sync: npc_sync.SyncPathOrDataset,
    stim_paths: Iterable[npc_io.PathLike],
    root: npc_io.PathLike | None = None,
    write: bool = True,
) -> npc_sync.SyncPatch | None:
    """The session's patch (every session: photodiode phantom-flicker removal applies to files
    without replays too).

    Loaded from `root` (default: `$NPC_SESSIONS_SYNC_PATCH_DIR`) if a patch for the same
    sync content hash and builder exists; otherwise built, and saved there if `write`.
    """
    raw = npc_sync.SyncDataset(npc_sync.get_sync_data(sync), on_invalid="ignore")
    path = get_patch_path(session_id, raw.content_hash, root)
    if path is not None and path.exists():
        patch = npc_sync.SyncPatch.load(path)
        if patch.sync_content_hash == raw.content_hash and patch.builder == BUILDER:
            return patch
        logger.info(f"{session_id}: cached sync patch {path} is stale: rebuilding")
    patch = build_session_sync_patch(raw, stim_paths)
    if path is not None and write:
        patch.save(path)
        logger.info(f"{session_id}: saved sync patch to {path}")
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


def get_invalid_times(sync: npc_sync.SyncDataset) -> list[dict[str, Any]]:
    """`invalid_times` entries for replayed intervals in which events on any line are
    unknown (in practice every replayed interval: licks, sound and barcodes are never
    reconstructed on sync)."""
    patch = sync.patch
    if patch is None or not sync.invalid_intervals:
        return []
    out = []
    for a, z in sync.invalid_intervals:
        unknown, reconstructed = [], []
        start, stop = a, z
        for line, spans in patch.lines.items():
            for s in spans:
                if s.span[0] <= a + 1e-6 and s.span[1] >= z - 1e-6:
                    if s.status == "unknown" and line != "barcode_ephys":
                        unknown.append(line)
                        start, stop = min(start, s.span[0]), max(stop, s.span[1])
                    elif s.status == "reconstructed" and s.method != "no_edges":
                        reconstructed.append(line)
        if not unknown:
            continue
        labelled = sorted(x for x in unknown if not x.startswith("bit"))
        out.append(
            dict(
                start_time=float(start),
                stop_time=float(stop),
                reason=(
                    "replayed sync data (copied acquisition buffers): events unknown on "
                    + ", ".join(labelled + sorted(x for x in unknown if x.startswith("bit")))
                    + (
                        f"; reconstructed: {', '.join(sorted(reconstructed))}"
                        if reconstructed
                        else ""
                    )
                ),
            )
        )
    return out


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
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
