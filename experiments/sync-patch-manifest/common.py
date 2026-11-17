"""Shared helpers for the sync-patch manifest and acceptance scripts.

Run with the npc_sync dev venv (editable npc_sync on branch `sync-patch`):
    ../../../npc_sync/.venv/Scripts/python.exe <script>.py
"""

from __future__ import annotations

import dataclasses
import datetime
import pathlib
from collections.abc import Iterable

import h5py
import numpy as np

import npc_sync

SHARE = pathlib.Path("//allen/programs/mindscope/workgroups/dynamicrouting/PilotEphys/Task 2 Pilot")
LOCAL7 = pathlib.Path(__file__).resolve().parents[3] / "npc_sync" / "experiments" / "vsync-frame-count" / "data"
CACHE = pathlib.Path("C:/data/t2p_cache")

# stim start (wall clock, 1 s resolution) -> first vsync of its block, seconds. Bounds checked
# on the population in match_offsets.log (see README).
MATCH_OFFSET_MIN_SEC = -1.5
MATCH_OFFSET_MAX_SEC = 60.0


@dataclasses.dataclass
class Stim:
    path: pathlib.Path
    name: str
    start_on_sync: float | None  # seconds from sync start (None: different day / unreadable)
    frame_intervals: np.ndarray
    diode_box_frame_interval: int | None
    error: str = ""
    events: dict = dataclasses.field(default_factory=dict)
    """stim-file event frame indices by key (EVENT_KEYS present in the file)"""

    @property
    def n_frames(self) -> int:
        return 0 if len(self.frame_intervals) == 0 else len(self.frame_intervals) + 1


EVENT_KEYS = ("lickFrames", "rewardFrames", "manualRewardFrames", "trialStimStartFrame", "trialOptoOnsetFrame", "stimStartFrame")


def read_stim(path, sync_start: datetime.datetime, name: str | None = None) -> Stim:
    name = name or pathlib.Path(path).stem
    events = {}
    try:
        with h5py.File(path, "r") as f:
            fi = f["frameIntervals"][:] if "frameIntervals" in f else np.array([])
            start = f["startTime"][()].decode() if "startTime" in f else None
            box = int(f["diodeBoxFrameInterval"][()]) if "diodeBoxFrameInterval" in f else None
            for k in EVENT_KEYS:
                if k in f:
                    try:
                        v = np.asarray(f[k][()], dtype=float).ravel()
                    except (TypeError, ValueError):
                        continue
                    events[k] = v[np.isfinite(v) & (v >= 0)]
    except Exception as exc:  # noqa: BLE001
        return Stim(path, name, None, np.array([]), None, error=f"{type(exc).__name__}: {exc}")
    if start is None:
        return Stim(path, name, None, fi, box, error="no startTime", events=events)
    dt = datetime.datetime.strptime(start, "%Y%m%d_%H%M%S")
    off = (dt - sync_start).total_seconds()
    return Stim(path, name, off, np.asarray(fi, float), box, events=events)


@dataclasses.dataclass
class Match:
    block_index: int | None
    stim: Stim | None
    offset_sec: float | None = None  # block onset - stim start
    reason: str = ""


def match_stims_to_blocks(blocks: list[np.ndarray], stims: Iterable[Stim]) -> tuple[list[Match], list[Match]]:
    """Exact one-to-one matching by start time only (frame counts are NOT used, so that a
    count mismatch can be detected rather than hidden).

    Each stim with frames is assigned to the block whose first vsync is closest to its start,
    if that offset is within [MATCH_OFFSET_MIN_SEC, MATCH_OFFSET_MAX_SEC]. A block claimed by
    more than one stim is ambiguous: none of them is matched.

    Returns (matches per block, unmatched stims).
    """
    onsets = np.array([b[0] for b in blocks]) if blocks else np.array([])
    claims: dict[int, list[tuple[Stim, float]]] = {}
    unmatched: list[Match] = []
    for s in stims:
        if s.n_frames == 0 or s.start_on_sync is None:
            continue
        if not len(onsets):
            unmatched.append(Match(None, s, reason="no vsync blocks"))
            continue
        off = onsets - s.start_on_sync
        ok = (off >= MATCH_OFFSET_MIN_SEC) & (off <= MATCH_OFFSET_MAX_SEC)
        if not ok.any():
            j = int(np.argmin(np.abs(off)))
            unmatched.append(Match(None, s, float(off[j]), reason="no block onset within bounds of stim start"))
            continue
        j = int(np.flatnonzero(ok)[np.argmin(np.abs(off[ok]))])
        claims.setdefault(j, []).append((s, float(off[j])))
    out = []
    for bi in range(len(blocks)):
        c = claims.get(bi, [])
        if len(c) == 1:
            out.append(Match(bi, c[0][0], c[0][1]))
        elif len(c) > 1:
            out.append(Match(bi, None, reason="claimed by " + ", ".join(x[0].name for x in c)))
            unmatched.extend(Match(None, x[0], x[1], reason=f"block {bi} ambiguous") for x in c)
        else:
            out.append(Match(bi, None, reason="no stim file starts near block onset"))
    return out, unmatched


def sync_start(sync: npc_sync.SyncDataset) -> datetime.datetime:
    return sync.start_time.replace(tzinfo=None)
