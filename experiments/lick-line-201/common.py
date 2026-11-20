"""Shared loaders for the #201 investigation (lick-line reconstruction across runs of spans)."""
from __future__ import annotations
import copy, functools, os, pickle, pathlib
os.environ.setdefault("NPC_SESSIONS_SYNC_PATCH_DIR", "none")
import numpy as np
import npc_lims, npc_sync, npc_sessions
from npc_sessions import timing

HERE = pathlib.Path(__file__).parent
CACHE = HERE / "cache"; CACHE.mkdir(exist_ok=True)
LICK = timing.LICK_LINE
SESSIONS = ["714748_2024-06-28", "708020_2024-05-17", "703880_2024-04-18", "715709_2024-08-08", "840160_2026-04-28"]

@functools.cache
def load(sid: str):
    """(raw true-clock view, stim paths, base patch with lick spans still 'unknown')."""
    try:
        sync_path = npc_lims.get_h5_sync_from_s3(sid)
    except ValueError:  # pilot sessions (no raw asset): network share
        sync_path = npc_sessions.DynamicRoutingSession(sid).sync_path
    raw = timing.true_clock_view(sync_path)
    f = CACHE / f"{sid}_base.pkl"
    if f.exists():
        stim_paths, base = pickle.loads(f.read_bytes())
    else:
        s = npc_sessions.DynamicRoutingSession(sid)
        stim_paths = tuple(s.stim_paths)
        blocks = timing.get_stim_blocks(raw, stim_paths)
        base = npc_sync.build_sync_patch(raw, blocks, builder=timing.BUILDER)
        f.write_bytes(pickle.dumps((stim_paths, base)))
    return raw, stim_paths, base

def edges(sd, line=LICK):
    return (np.asarray(sd.get_rising_edges(line, units="seconds"), float),
            np.asarray(sd.get_falling_edges(line, units="seconds"), float))

def alternation_errors(r, f):
    """Number of adjacent same-kind edges in the merged sequence (0 = line alternates)."""
    t = np.concatenate([r, f]); k = np.concatenate([np.ones(len(r)), np.zeros(len(f))])
    k = k[np.argsort(t, kind="stable")]
    return int(np.sum(k[1:] == k[:-1]))

def lick_onsets(sd, stim_paths):
    out = []
    for p in stim_paths:
        try:
            on, (t0, t1) = timing.predict_lick_onsets(sd, p)
        except Exception:
            continue
        if np.isfinite(t0):
            out.append(on)
    return np.sort(np.concatenate(out)) if out else np.array([])
