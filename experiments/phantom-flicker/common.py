"""Shared helpers for the phantom-flicker analysis. Read-only on the cache and the share.

Uses the exact one-to-one stim <-> block matcher of ../sync-patch-manifest/common.py (start time
only, so a frame-count mismatch is detected, not absorbed).
"""

from __future__ import annotations

import dataclasses
import functools
import importlib.util
import pathlib
import re
import sys

import numpy as np

import npc_sync

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "manifest_common", HERE.parent / "sync-patch-manifest" / "common.py"
)
manifest_common = importlib.util.module_from_spec(_spec)
sys.modules["manifest_common"] = manifest_common
_spec.loader.exec_module(manifest_common)

CACHE = pathlib.Path("C:/data/t2p_cache")


def sync_path(folder: str) -> pathlib.Path:
    """Largest sync file in the cached session folder (same choice as earlier QC)."""
    f = CACHE / folder
    syncs = sorted(
        (p for p in f.iterdir() if re.match(r"^\d{8}T\d{6}\.h5$", p.name)),
        key=lambda p: p.stat().st_size,
    )
    return syncs[-1]


def stim_paths(folder: str) -> list[pathlib.Path]:
    return sorted(p for p in (CACHE / folder).iterdir() if p.suffix == ".hdf5")


@functools.lru_cache(maxsize=4)
def raw_sync(folder: str) -> npc_sync.SyncDataset:
    return npc_sync.SyncDataset(sync_path(folder), on_invalid="ignore")


@functools.lru_cache(maxsize=4)
def matches(folder: str):
    s = raw_sync(folder)
    blocks = list(s.vsync_times_in_blocks)
    start = manifest_common.sync_start(s)
    stims = [manifest_common.read_stim(p, start) for p in stim_paths(folder)]
    return blocks, manifest_common.match_stims_to_blocks(blocks, stims)[0]


@dataclasses.dataclass
class Block:
    folder: str
    block_index: int
    stim_name: str
    stim_path: pathlib.Path
    vsyncs: np.ndarray
    frame_intervals: np.ndarray
    box: int | None
    """diodeBoxFrameInterval from the file (None: key absent, = 1 in TaskControl)."""
    edges: np.ndarray
    """Diode edges (seconds) from the block's first vsync to its last vsync + 0.5 s."""
    rising: np.ndarray
    all_edges: np.ndarray
    all_rising: np.ndarray
    vsync_rising: np.ndarray

    @property
    def n_stim_frames(self) -> int:
        return len(self.frame_intervals) + 1

    @property
    def refresh(self) -> float:
        """Median frame interval (s): one refresh."""
        return float(np.median(self.frame_intervals))

    @property
    def vsync_rise_per_frame(self) -> np.ndarray:
        """Frame-signal rising edge before each frame's falling edge (NaN if none)."""
        j = np.searchsorted(self.vsync_rising, self.vsyncs) - 1
        return np.where(j >= 0, self.vsync_rising[np.clip(j, 0, None)], np.nan)


def diode_edges(s: npc_sync.SyncDataset) -> tuple[np.ndarray, np.ndarray]:
    r = s.get_rising_edges("stim_photodiode", units="seconds")
    f = s.get_falling_edges("stim_photodiode", units="seconds")
    e = np.concatenate([r, f])
    pol = np.concatenate([np.ones(len(r), bool), np.zeros(len(f), bool)])
    o = np.argsort(e, kind="stable")
    return e[o], pol[o]


def block_data(folder: str, bi: int, sync: npc_sync.SyncDataset | None = None) -> Block:
    s = sync or raw_sync(folder)
    blocks, m = matches(folder)
    mt = m[bi]
    if mt.stim is None:
        raise ValueError(f"{folder} block {bi}: {mt.reason}")
    v = blocks[bi]
    e, pol = diode_edges(s)
    sel = (e >= v[0]) & (e < v[-1] + 0.5)
    return Block(
        folder=folder,
        block_index=bi,
        stim_name=mt.stim.name,
        stim_path=mt.stim.path,
        vsyncs=v,
        frame_intervals=mt.stim.frame_intervals,
        box=mt.stim.diode_box_frame_interval,
        edges=e[sel],
        rising=pol[sel],
        all_edges=e,
        all_rising=pol,
        vsync_rising=s.get_rising_edges("vsync_stim", units="seconds"),
    )
