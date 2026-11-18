"""npc_sessions.timing on synthetic sync + stim files with a replayed interval.

Needs npc_sync with `SyncPatch` and npc_stim with patch-aware display times (branches
`sync-patch`); skipped otherwise."""

from __future__ import annotations

import pathlib

import h5py
import numpy as np
import pytest

npc_sync = pytest.importorskip("npc_sync")
if not hasattr(npc_sync, "SyncPatch"):
    pytest.skip("npc_sync without SyncPatch", allow_module_level=True)

import npc_stim  # noqa: E402

from npc_sessions import timing  # noqa: E402

FS = 100_000
BUFFER = 50_000
LABELS = [""] * 32
LABELS[0], LABELS[2], LABELS[4], LABELS[5] = "barcode_ephys", "vsync_stim", "stim_photodiode", "stim_running"
LABELS[15], LABELS[27], LABELS[31] = "reward_1_actuator", "beh_cam_exposing", "lick_sensor"


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("timing")
    rng = np.random.default_rng(1)
    n = 3000
    fi = np.full(n - 1, 1 / 60) + rng.normal(0, 2e-5, n - 1)
    for i in (700, 1500, 2400):
        fi[i] = 3 / 60
    frames = 5.0 + np.r_[0, np.cumsum(fi)]
    edges = []
    for t in frames:
        edges += [(t - 0.002, 2, 1), (t, 2, 0)]
    pol = 1
    for f in range(0, n, 3):
        edges.append((frames[f] + (0.020 if pol else 0.024), 4, pol))
        pol ^= 1
    edges += [(4.95, 5, 1), (frames[-1] + 0.05, 5, 0)]
    for t in np.arange(1.0, 59.0, 1 / 60.02):
        edges += [(t, 27, 1), (t + 0.003, 27, 0)]
    for t in np.arange(0.9, 59.0, 0.5):  # barcode-line pulses that straddle the widened span bounds
        edges += [(t, 0, 1), (t + 0.2, 0, 0)]
    # licks: sensor onset in the frame interval before the read of frame f (read = vsync[f-1])
    lick_frames = np.sort(rng.choice(np.arange(10, n - 10), 120, replace=False))
    lick_frames = lick_frames[np.r_[True, np.diff(lick_frames) > 6]]
    lick_onsets = frames[lick_frames - 2] + rng.uniform(0.001, 0.015, len(lick_frames))
    for t in lick_onsets:
        edges += [(t, 31, 1), (t + 0.03, 31, 0)]
    # rewards: solenoid opens 0.8 ms after the flip of the reward frame
    reward_frames = np.sort(rng.choice(np.arange(10, n - 10), 40, replace=False))
    reward_on = frames[reward_frames] + 0.0008
    for t in reward_on:
        edges += [(t, 15, 1), (t + 0.05, 15, 0)]
    items = sorted((int(round(t * FS)), b, v) for t, b, v in edges)
    times, states, state = [0], [0], 0
    for smp, b, v in items:
        state = (state | (1 << b)) if v else (state & ~(1 << b))
        if times[-1] == smp:
            states[-1] = state
        else:
            times.append(smp)
            states.append(state)
    t, s = np.asarray(times, np.int64), np.asarray(states, np.int64)
    g0, first = 40 * BUFFER, 36 * BUFFER  # 18-20 s are copies of 20-20.5 s
    g = (t >= g0) & (t < g0 + BUFFER)
    before, after = t < first, t >= g0
    t, s = (
        np.r_[t[before], *[first + c * BUFFER + t[g] - g0 for c in range(4)], t[after]],
        np.r_[s[before], *[s[g] for _ in range(4)], s[after]],
    )
    sync = tmp / "20260101T100000.h5"
    meta = {
        "ni_daq": {"sample_freq": float(FS), "counter_bits": 32, "event_bits": 32},
        "start_time": "2026-01-01 10:00:00.000000",
        "stop_time": "2026-01-01 10:01:00.000000",
        "line_labels": LABELS,
        "timeouts": [],
        "version": "synthetic",
        "total_samples": 60 * FS,
    }
    with h5py.File(sync, "w") as f:
        f["data"] = np.c_[t.astype(np.uint32), s.astype(np.uint32)]
        f["meta"] = str(meta)
    stim = tmp / "DynamicRouting1_000000_20260101_100004.hdf5"
    with h5py.File(stim, "w") as f:
        f["frameIntervals"] = fi
        f["startTime"] = b"20260101_100004"
        f["diodeBoxFrameInterval"] = 3
        f["lickFrames"] = lick_frames
        f["rewardFrames"] = reward_frames
    return dict(tmp=tmp, sync=sync, stim=stim, frames=frames, lick_onsets=lick_onsets,
                reward_on=reward_on, interval=(18.0, 20.0))


def test_match_is_by_start_time_only(files) -> None:
    raw = npc_sync.SyncDataset(files["sync"], on_invalid="ignore")
    matches, unmatched = timing.match_stims_to_blocks(raw, [files["stim"]])
    assert list(matches) == [0] and not unmatched
    # no dropped frame inside the copies: the count matches although 2 s of edges are copies
    # (periodic, so they are also close in time: count and timing can't reveal a replay)
    assert len(raw.vsync_times_in_blocks[0]) == matches[0].n_frames
    assert raw.invalid_intervals == (files["interval"],)


def test_patch_is_built_saved_and_reused(files, tmp_path) -> None:
    p1 = timing.get_sync_patch("s", files["sync"], [files["stim"]], root=tmp_path)
    (saved,) = tmp_path.glob("s_*.json")
    p2 = timing.get_sync_patch("s", files["sync"], [files["stim"]], root=tmp_path)
    assert p2.created == p1.created  # loaded, not rebuilt
    assert p1.builder == timing.BUILDER
    # barcode spans widened by >= 2 s, to bounds where the line is low: edges stay paired
    (bc,) = p1.lines["barcode_ephys"]
    assert bc.span[0] <= 18.0 - 2 and bc.span[1] >= 20.0 + 2
    s = npc_sync.SyncDataset(files["sync"], patch=p1)
    on = s.get_rising_edges("barcode_ephys", units="seconds")
    off = s.get_falling_edges("barcode_ephys", units="seconds")
    assert len(on) == len(off) and np.all(off - on > 0) and np.all(off - on < 0.3)


def test_report_gates_and_invalid_times(files) -> None:
    patch = timing.get_sync_patch("s", files["sync"], [files["stim"]])
    sync = npc_sync.SyncDataset(files["sync"], patch=patch)
    display = npc_stim.get_display_times(sync, [files["stim"]])
    report = timing.get_block_report(sync, [files["stim"]], display)
    (row,) = report.to_dict("records")
    assert row["status"] == "patched" and row["display_time_source"] == "photodiode"
    assert row["n_interpolated_frames"] > 0
    assert all(abs(r) < 0.5 for r in row["anchor_residuals_ms"])
    (inv,) = timing.get_invalid_times(sync)
    assert inv["start_time"] <= 18.0 and inv["stop_time"] >= 20.0
    assert "lick_sensor" in inv["reason"] and "vsync_stim" in inv["reason"].split("reconstructed:")[1]
    # without a stim file the vsync span is unknown -> block rejected
    p0 = timing.build_session_sync_patch(files["sync"], [])
    s0 = npc_sync.SyncDataset(files["sync"], patch=p0)
    (row0,) = timing.get_block_report(s0, [], npc_stim.get_display_times(s0, {})).to_dict("records")
    assert row0["status"] == "rejected"


def test_reconstructed_licks_and_rewards(files) -> None:
    patch = timing.get_sync_patch("s", files["sync"], [files["stim"]])
    sync = npc_sync.SyncDataset(files["sync"], patch=patch)
    licks = timing.get_reconstructed_lick_times(sync, files["stim"])
    true = files["lick_onsets"][npc_sync.in_intervals(files["lick_onsets"], sync.get_unknown_intervals("lick_sensor"))]
    assert len(licks) == len(true) > 0
    assert np.max(np.abs(licks - true)) <= 1 / 120 + 1e-4
    measured = sync.get_rising_edges("reward_1_actuator", units="seconds")
    t, manual, latency = timing.get_reconstructed_reward_times(sync, files["stim"], measured)
    true = files["reward_on"][npc_sync.in_intervals(files["reward_on"], sync.get_unknown_intervals(15))]
    assert len(t) == len(true)
    assert latency == pytest.approx(0.0008, abs=2e-5)
    if len(t):
        assert np.max(np.abs(t - true)) < 3 / FS


@pytest.fixture(scope="module")
def flicker_files(tmp_path_factory):
    """A box-1 session without replays: phantom flicker (one-refresh flashes of the other
    buffer) in 4 held frames."""
    tmp = tmp_path_factory.mktemp("flicker")
    R = 1 / 60
    n = 4000
    holds = {700: 3, 1500: 2, 1520: 4, 2400: 11, 3000: 3, 3300: 3, 3600: 2}
    pairs = {700: [1], 2400: [1, 3, 5, 7], 3000: [1], 3300: [1]}
    n_ref = np.ones(n - 1, int)
    for k, h in holds.items():
        n_ref[k] = h
    display = 5.0 + np.r_[0, np.cumsum(n_ref)] * R
    # jitter: exactly periodic data would look like replayed (identical) buffers
    fall = display + 0.0002 + np.random.default_rng(2).normal(0, 2e-5, n)
    edges = []
    for t in fall:
        edges += [(t - 0.015, 2, 1), (t, 2, 0)]
    pol = 1
    for k in range(n):
        edges.append((display[k] + (0.020 if pol else 0.024), 4, pol))
        for r in pairs.get(k, []):
            edges.append((display[k] + r * R + (0.020 if 1 - pol else 0.024) + 0.003, 4, 1 - pol))
            edges.append((display[k] + (r + 1) * R + (0.020 if pol else 0.024), 4, pol))
        pol ^= 1
    edges += [(4.95, 5, 1), (fall[-1] + 0.05, 5, 0)]
    items = sorted((int(round(t * FS)), b, v) for t, b, v in edges)
    times, states, state = [0], [0], 0
    for smp, b, v in items:
        state = (state | (1 << b)) if v else (state & ~(1 << b))
        if times[-1] == smp:
            states[-1] = state
        else:
            times.append(smp)
            states.append(state)
    sync = tmp / "20260101T100000.h5"
    meta = {
        "ni_daq": {"sample_freq": float(FS), "counter_bits": 32, "event_bits": 32},
        "start_time": "2026-01-01 10:00:00.000000",
        "stop_time": "2026-01-01 10:01:40.000000",
        "line_labels": LABELS,
        "timeouts": [],
        "version": "synthetic",
        "total_samples": 100 * FS,
    }
    with h5py.File(sync, "w") as f:
        f["data"] = np.c_[np.asarray(times, np.uint32), np.asarray(states, np.uint32)]
        f["meta"] = str(meta)
    stim = tmp / "DynamicRouting1_000000_20260101_100004.hdf5"
    with h5py.File(stim, "w") as f:
        f["frameIntervals"] = np.diff(fall)
        f["startTime"] = b"20260101_100004"
        f["diodeBoxFrameInterval"] = 1
    return dict(sync=sync, stim=stim, display=display, n_phantom=2 * sum(len(v) for v in pairs.values()))


@pytest.mark.skipif(not hasattr(npc_sync, "align_diode_toggles"), reason="npc_sync without phantom-flicker removal")
def test_phantom_flicker_patch_for_a_session_without_replays(flicker_files) -> None:
    patch = timing.get_sync_patch("s", flicker_files["sync"], [flicker_files["stim"]])
    assert patch is not None and patch.invalid_intervals == ()
    assert patch.diode_summary()["phantom_flicker_removed"] == flicker_files["n_phantom"]
    sync = npc_sync.SyncDataset(flicker_files["sync"], patch=patch)
    display = npc_stim.get_display_times(sync, [flicker_files["stim"]])
    (row,) = timing.get_block_report(sync, [flicker_files["stim"]], display).to_dict("records")
    assert row["status"] == "clean" and row["diode_status"] == "cleaned"
    assert row["n_phantom_flicker_removed"] == flicker_files["n_phantom"]
    assert row["display_time_source"] == "photodiode"
    assert timing.get_invalid_times(sync) == []
    # unpatched, npc_stim rejects the block (flip count)
    (unpatched,) = npc_stim.get_display_times(npc_sync.SyncDataset(flicker_files["sync"]), [flicker_files["stim"]])
    assert unpatched.source == "unavailable"
