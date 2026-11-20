"""npc_sessions.timing on synthetic sync + stim files with a replayed interval.

Needs npc_sync with `SyncPatch` and npc_stim with patch-aware display times (branches
`sync-patch`); skipped otherwise."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

npc_sync = pytest.importorskip("npc_sync")
if not hasattr(npc_sync, "SyncPatch"):
    pytest.skip("npc_sync without SyncPatch", allow_module_level=True)

import npc_io  # noqa: E402
import npc_stim  # noqa: E402

from npc_sessions import timing  # noqa: E402


@pytest.fixture(autouse=True)
def _no_shared_patch_cache(monkeypatch):
    """Never read or write the shared S3 patch cache from tests (`root=` still works)."""
    monkeypatch.setenv(timing.PATCH_DIR_ENV_VAR, "none")


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
    # barcode spans widened by >= 2 s (npc_ephys drops edges that close to an invalid
    # interval, one by one), to bounds where the line is low: edges stay paired
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
    (rep,) = timing.get_replayed_intervals(sync)
    assert (rep["start_time"], rep["stop_time"]) == files["interval"]
    unknown, reconstructed = rep["reason"].split("reconstructed:")
    assert "reward_1_actuator" in unknown and "vsync_stim" in reconstructed
    assert "lick_sensor" in reconstructed
    # without a stim file the vsync span is unknown -> block rejected
    p0 = timing.build_session_sync_patch(files["sync"], [])
    s0 = npc_sync.SyncDataset(files["sync"], patch=p0)
    (row0,) = timing.get_block_report(s0, [], npc_stim.get_display_times(s0, {})).to_dict("records")
    assert row0["status"] == "rejected"


def test_patch_cache_records_dependency_versions(files, tmp_path, monkeypatch) -> None:
    p1 = timing.get_sync_patch("s", files["sync"], [files["stim"]], root=tmp_path)
    (saved,) = tmp_path.glob("s_*.json")
    loaded, versions = timing.load_patch(saved)
    assert versions == timing.get_dependency_versions()
    assert set(versions) == set(timing.PATCH_DEPENDENCIES)
    assert npc_sync.SyncPatch.load(saved).created == p1.created  # still a valid SyncPatch file
    # a changed dependency version -> rebuilt and overwritten
    monkeypatch.setattr(
        timing, "get_dependency_versions", lambda: {**versions, "npc_stim": "0.0.0"}
    )
    p2 = timing.get_sync_patch("s", files["sync"], [files["stim"]], root=tmp_path)
    assert p2 is not p1 and p2.to_dict()["lines"] == p1.to_dict()["lines"]
    assert timing.load_patch(saved)[1]["npc_stim"] == "0.0.0"
    assert timing.get_sync_patch("s", files["sync"], [files["stim"]], root=tmp_path).created == p2.created
    # a changed npc_sessions version alone -> reused (BUILDER covers its patch code)
    monkeypatch.setattr(
        timing,
        "get_dependency_versions",
        lambda: {**versions, "npc_stim": "0.0.0", "npc_sessions": "9.9.9"},
    )
    assert timing.get_sync_patch("s", files["sync"], [files["stim"]], root=tmp_path).created == p2.created


def test_npc_sessions_version_records_git_state(monkeypatch) -> None:
    monkeypatch.setattr(timing, "get_npc_sessions_git_state", lambda: ("a" * 40, True))
    v = timing.get_dependency_versions()["npc_sessions"]
    assert v.endswith("+g" + "a" * 12 + ".dirty")
    monkeypatch.setattr(timing, "get_npc_sessions_git_state", lambda: ("a" * 40, False))
    assert timing.get_npc_sessions_version().endswith("+g" + "a" * 12)
    monkeypatch.setattr(timing, "get_npc_sessions_git_state", lambda: None)
    assert "+" not in timing.get_npc_sessions_version()


def test_dirty_checkout_does_not_write_shared_cache(files, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(timing, "get_npc_sessions_git_state", lambda: ("a" * 40, True))
    monkeypatch.setattr(timing, "DEFAULT_PATCH_ROOT", tmp_path.as_posix())
    assert timing.is_shared_patch_root(tmp_path / "x.json")
    assert not timing.is_shared_patch_root(tmp_path.parent / "other" / "x.json")
    assert timing.get_sync_patch("s", files["sync"], [files["stim"]], root=tmp_path) is not None
    assert not list(tmp_path.glob("*.json"))
    monkeypatch.setattr(timing, "get_npc_sessions_git_state", lambda: ("a" * 40, False))
    timing.get_sync_patch("s", files["sync"], [files["stim"]], root=tmp_path)
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_pair_rising_falling_edges() -> None:
    # lick straddling the start of a replayed interval: its falling edge was masked out
    rising = [1.0, 2.0, 4680.48616, 4737.8633]
    falling = [1.05, 2.05, 4737.9]
    r, f = timing.pair_rising_falling_edges(rising, falling, [(4680.49999, 4681.5)])
    assert r.tolist() == [1.0, 2.0, 4737.8633] and f.tolist() == [1.05, 2.05, 4737.9]
    # leading falling / trailing rising edges are dropped
    r, f = timing.pair_rising_falling_edges([1, 3, 5], [0.5, 2, 4])
    assert r.tolist() == [1, 3] and f.tolist() == [2, 4]
    # a pair spanning an unknown interval is dropped
    r, f = timing.pair_rising_falling_edges([1, 3], [2, 6], [(4, 5)])
    assert r.tolist() == [1] and f.tolist() == [2]
    r, f = timing.pair_rising_falling_edges([], [1])
    assert r.size == 0 and f.size == 0


def test_patch_root(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv(timing.PATCH_DIR_ENV_VAR, raising=False)
    monkeypatch.setattr(timing, "has_aws_credentials", lambda: False)
    assert timing.get_patch_root() is None
    monkeypatch.setattr(timing, "has_aws_credentials", lambda: True)
    assert timing.get_patch_root().as_posix() == timing.DEFAULT_PATCH_ROOT
    monkeypatch.setenv(timing.PATCH_DIR_ENV_VAR, "none")
    assert timing.get_patch_root() is None
    monkeypatch.setenv(timing.PATCH_DIR_ENV_VAR, str(tmp_path))
    assert timing.get_patch_root() == npc_io.from_pathlike(tmp_path)
    assert timing.get_patch_root(tmp_path / "x") == npc_io.from_pathlike(tmp_path / "x")


def test_unknown_masks(files) -> None:
    patch = timing.get_sync_patch("s", files["sync"], [files["stim"]])
    sync = npc_sync.SyncDataset(files["sync"], patch=patch)
    (u,) = timing.get_unknown_intervals(sync, "reward_1_actuator")
    assert u[0] <= 18.0 and u[1] >= 20.0
    assert timing.get_unknown_intervals(sync, "no_such_line") == ()
    assert timing.get_unknown_intervals(None, "reward_1_actuator") == ()
    m = timing.unknown_overlap_mask([10, 17.5, 19, 21, 17], [11, 18.5, np.nan, 22, 23], [u])
    assert m.tolist() == [False, True, True, False, True]

    class Rec:
        def __init__(self, trigger, onset, offset):
            self.presentation = type("P", (), {"trigger_time_on_sync": trigger})()
            self.onset_time_on_sync, self.offset_time_on_sync = onset, offset

    recs = [Rec(10, 10.01, 10.5), Rec(17.9, 20.1, 20.6), Rec(17.0, 17.01, 18.5), None]
    assert timing.stim_recordings_unknown_mask(recs, sync, "reward_1_actuator").tolist() == [
        False, True, True, False
    ]
    # sound: offset is onset + duration, not an edge on the line
    assert timing.stim_recordings_unknown_mask(
        recs, sync, "reward_1_actuator", include_offset=False
    ).tolist() == [False, True, False, False]
    masked = timing.mask_unknown_stim_recordings(recs, sync, "reward_1_actuator")
    assert masked[0] is recs[0] and masked[1] is None and masked[2] is None


def test_reconstructed_licks_and_rewards(files) -> None:
    patch = timing.get_sync_patch("s", files["sync"], [files["stim"]])
    sync = npc_sync.SyncDataset(files["sync"], patch=patch)
    # licks: reconstructed on the line itself, alternating with synthetic offsets
    assert not sync.get_unknown_intervals("lick_sensor")
    assert timing.get_reconstructed_lick_times(sync, files["stim"]).size == 0
    on = sync.get_rising_edges("lick_sensor", units="seconds")
    off = sync.get_falling_edges("lick_sensor", units="seconds")
    assert len(on) == len(off) and np.all(on < off) and np.all(off[:-1] < on[1:])
    src = sync.get_edge_sources("lick_sensor", "rising")
    (span,) = [s.span for s in patch.lines["lick_sensor"]]
    true = files["lick_onsets"][npc_sync.in_intervals(files["lick_onsets"], [span])]
    licks = on[src == npc_sync.patch.INTERPOLATED]
    assert len(licks) == len(true) > 0
    assert np.max(np.abs(licks - true)) <= 1 / 120 + 1e-4
    (rec,) = patch.lines["lick_sensor"]
    assert rec.status == "reconstructed" and rec.method == "lick_frames"
    assert rec.diagnostics["fraction_lick_frames_within_1_frame"] > 0.95
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
    assert timing.get_replayed_intervals(sync) == []
    # unpatched, npc_stim rejects the block (flip count)
    (unpatched,) = npc_stim.get_display_times(npc_sync.SyncDataset(flicker_files["sync"]), [flicker_files["stim"]])
    assert unpatched.source == "unavailable"


def test_predict_opto_line_from_trigger_frames() -> None:
    rng = np.random.default_rng(3)
    trig = np.sort(rng.uniform(10, 1000, 200))
    durs = rng.choice([0.01, 0.1, 0.2], len(trig))
    lat = 0.0042 + rng.uniform(-0.004, 0.004, len(trig))  # jitter within a frame
    on = trig + lat
    off = on + durs + 0.002
    span = [(500.0, 502.0)]
    # a trigger inside the span: its edges are missing from the line
    inside = npc_sync.in_intervals(on, span)
    trig = np.r_[trig, 501.0]
    durs = np.r_[durs, 0.1]
    order = np.argsort(trig)
    trig, durs = trig[order], durs[order]
    pred = timing.predict_opto_line(on[~inside], off[~inside], trig, durs, span)
    assert pred["status"] == "reconstructed" and pred["n_triggers_in_spans"] >= 1
    k = np.searchsorted(trig, 501.0)
    assert abs(pred["rising"][k] - (501.0 + np.median(lat))) < 1e-3
    assert abs(pred["falling"][k] - pred["rising"][k] - 0.102) < 1e-3
    # outside the span predictions are within +-1/2 frame of the measured edges
    m = ~npc_sync.in_intervals(trig, span)
    assert np.max(np.abs(pred["rising"][m] - np.sort(on[~inside]))) <= timing.OPTO_ERROR_BOUND_SEC
    # an extra edge not explained by a trigger: the line stays unknown
    bad = timing.predict_opto_line(np.sort(np.r_[on[~inside], 5.0]), off[~inside], trig, durs, span)
    assert bad["status"] == "unknown"
    # no triggers and no edges: reconstructed, nothing inside
    none = timing.predict_opto_line(np.array([]), np.array([]), np.array([]), np.array([]), span)
    assert none["status"] == "reconstructed" and len(none["rising"]) == 0


def test_edges_explained_and_span_replacement(files) -> None:
    span = [(18.0, 20.0)]
    m = np.array([1.0, 5.0, 25.0])
    assert timing._check_edges_explained(m, np.array([1.0002, 5.0, 19.0, 25.0]), span, 1e-3) == ""
    assert "differ" in timing._check_edges_explained(m, np.array([1.01, 5.0, 25.0]), span, 1e-3)
    assert "2 predicted" in timing._check_edges_explained(m, np.array([1.0, 25.0]), span, 1e-3)
    patch = timing.build_session_sync_patch(files["sync"], [files["stim"]])
    key = "reward_1_actuator"
    assert patch.unknown_intervals(key)
    timing._replace_unknown_spans(
        patch, key, np.array([18.5, 30.0]), np.array([18.6]), "test", 0.001, {}, "test"
    )
    assert not patch.unknown_intervals(key)
    s = npc_sync.SyncDataset(files["sync"], patch=patch)
    on = s.get_rising_edges(key, units="seconds")
    assert np.any(np.isclose(on, 18.5, atol=2 / FS)) and not s.get_unknown_intervals(key)


class _FakeRaw:
    """Minimal stand-in for a SyncDataset with one replayed interval on the lick line."""

    sample_freq = FS

    def __init__(self, rising, falling, interval):
        self.r, self.f = np.asarray(rising, float), np.asarray(falling, float)
        self.invalid_intervals = (interval,)
        self._mask_intervals = ((interval[0] - 1 / FS, interval[1]),)

    def get_rising_edges(self, line, units="seconds"):
        return self.r

    def get_falling_edges(self, line, units="seconds"):
        return self.f


def _lick_patch(interval, status="unknown"):
    a = interval[0] - 1 / FS
    return npc_sync.SyncPatch(
        sync_content_hash="x",
        builder=timing.BUILDER,
        invalid_intervals=(interval,),
        lines={"lick_sensor": [npc_sync.SpanPatch(span=(a, interval[1]), status=status)]},
    )


def _alternates(rising, falling) -> bool:
    e = sorted([(t, 1) for t in rising] + [(t, 0) for t in falling])
    return all(e[i][1] != e[i + 1][1] for i in range(len(e) - 1))


@pytest.mark.parametrize("onsets", [[], [10.3, 10.6]])
def test_lick_line_alternates_across_both_bounds(monkeypatch, onsets) -> None:
    interval = (10.0, 11.0)
    # lick in progress at entry (falling edge masked at the sample before the copy) and at
    # exit (rising edge inside the copy, falling edge the first measured edge after it)
    rising = [8.0, 9.98, 10.5, 11.02]
    falling = [8.05, 10.0 - 1 / FS, 10.55, 11.01, 11.08]
    raw = _FakeRaw(rising, falling, interval)
    patch = _lick_patch(interval)
    monkeypatch.setattr(timing, "_patched_view", lambda raw, patch: raw)
    monkeypatch.setattr(
        timing, "predict_lick_onsets", lambda sync, path: (np.asarray(onsets, float), (5.0, 20.0))
    )
    info = timing.reconstruct_lick_line(raw, patch, ["DynamicRouting1_x.hdf5"])
    (sp,) = patch.lines["lick_sensor"]
    assert sp.status == "reconstructed" and info["spans"][0]["high_at_entry"]
    assert info["spans"][0]["high_at_exit"]
    a, z = sp.span
    outside_r = [t for t in rising if not a <= t < z]
    outside_f = [t for t in falling if not a <= t < z]
    r = np.sort(np.r_[outside_r, sp.rising])
    f = np.sort(np.r_[outside_f, sp.falling])
    assert _alternates(r, f), (r, f)
    assert np.all((sp.rising >= a) & (sp.rising < z)) and np.all((sp.falling >= a) & (sp.falling < z))
    assert np.isin(onsets, sp.rising).all()


def test_unknown_lick_span_widened_to_low_bounds() -> None:
    interval = (10.0, 11.0)
    rising = [8.0, 9.98, 10.5, 11.02]
    falling = [8.05, 10.0 - 1 / FS, 10.55, 11.01, 11.08]
    raw = _FakeRaw(rising, falling, interval)
    patch = _lick_patch(interval)
    timing.widen_unknown_lick_spans(raw, patch)
    (sp,) = patch.lines["lick_sensor"]
    a, z = sp.span
    assert sp.status == "unknown" and a == 9.98 and z > 11.01
    r = [t for t in rising if not a <= t < z]
    f = [t for t in falling if not a <= t < z]
    assert r == [8.0, 11.02] and f == [8.05, 11.08]


# ------------------------------------------------------------------ dropped buffers (timebase)


@pytest.fixture(scope="module")
def dropped_files(tmp_path_factory):
    """One block; buffers written out of step with time, as the sync program does:
    - true 30.0-31.0 s dropped (2 buffers), nothing makes up for it: later data is 1 s early
    - true 40.0-40.5 s dropped, made up for by a copy recorded 3 s later: the data in
      between is recorded 0.5 s early
    Two cameras, so that the drops can be located on their exposing lines."""
    tmp = tmp_path_factory.mktemp("timing_dropped")
    rng = np.random.default_rng(2)
    labels = list(LABELS)
    labels[29] = "eye_cam_exposing"
    n = 3300
    fi = np.full(n - 1, 1 / 59.95) + rng.normal(0, 2e-5, n - 1)
    for i in (700, 1500, 2400):
        fi[i] = 3 / 59.95
    frames = 5.0 + np.r_[0, np.cumsum(fi)]
    edges = []
    for t in frames:
        edges += [(t - 0.002, 2, 1), (t, 2, 0)]
    pol = 1
    for f in range(0, n, 3):
        edges.append((frames[f] + (0.020 if pol else 0.024), 4, pol))
        pol ^= 1
    edges += [(4.95, 5, 1), (frames[-1] + 0.05, 5, 0)]
    for bit, hz in ((27, 60.0027), (29, 60.0012)):
        for t in np.arange(1.0 + bit * 1e-3, 64.0, 1 / hz):
            edges += [(t, bit, 1), (t + 0.003, bit, 0)]
    lick_frames = np.sort(rng.choice(np.arange(10, n - 10), 300, replace=False))
    lick_frames = lick_frames[np.r_[True, np.diff(lick_frames) > 6]]
    lick_onsets = frames[lick_frames - 2] + rng.uniform(0.001, 0.015, len(lick_frames))
    for t in lick_onsets:
        edges += [(t, 31, 1), (t + 0.03, 31, 0)]
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
    # recorded buffer p <- true buffer source[p]
    source = [c for c in range(130) if c not in (60, 61, 80)]  # 30-31 s and 40-40.5 s dropped
    p_copy = source.index(87)  # true 43.5 s: 3 s after the second drop
    source.insert(p_copy, 87)  # a copy of the buffer that follows it
    out_t, out_s, last = [], [], 0
    for p, c in enumerate(source):
        i, j = np.searchsorted(t, [c * BUFFER, (c + 1) * BUFFER])
        start_state = int(s[np.searchsorted(t, c * BUFFER, side="right") - 1])
        if (i == j or t[i] != c * BUFFER) and start_state != last:
            out_t.append(p * BUFFER)
            out_s.append(start_state)
        out_t += (t[i:j] - c * BUFFER + p * BUFFER).tolist()
        out_s += s[i:j].tolist()
        last = int(s[j - 1]) if j > i else start_state
    sync = tmp / "20260101T100000.h5"
    meta = {
        "ni_daq": {"sample_freq": float(FS), "counter_bits": 32, "event_bits": 32},
        "start_time": "2026-01-01 10:00:00.000000",
        "stop_time": "2026-01-01 10:01:05.000000",
        "line_labels": labels,
        "timeouts": [],
        "version": "synthetic",
        "total_samples": len(source) * BUFFER,
    }
    with h5py.File(sync, "w") as f:
        f["data"] = np.c_[np.asarray(out_t, np.uint32), np.asarray(out_s, np.uint32)]
        f["meta"] = str(meta)
    stim = tmp / "DynamicRouting1_000000_20260101_100004.hdf5"
    with h5py.File(stim, "w") as f:
        f["frameIntervals"] = fi
        f["startTime"] = b"20260101_100004"
        f["diodeBoxFrameInterval"] = 3
        f["lickFrames"] = lick_frames
    return dict(sync=sync, stim=stim, frames=frames, lick_onsets=lick_onsets)


def test_dropped_buffers_are_located_and_reconstructed(dropped_files) -> None:
    f = dropped_files
    recorded = npc_sync.SyncDataset(f["sync"], on_invalid="ignore")
    assert recorded.invalid_intervals == ((42.0, 42.5),)  # the copy, on the recorded clock
    raw = timing.true_clock_view(f["sync"])
    assert raw.invalid_intervals == ((30.0, 31.0), (40.0, 40.5))
    matches, unmatched = timing.match_stims_to_blocks(raw, [f["stim"]])
    assert list(matches) == [0] and not unmatched
    patch = timing.get_sync_patch("s", f["sync"], [f["stim"]])
    assert patch.timebase == raw.timebase
    assert [(j.n_chunks, j.source) for j in patch.timebase.drops] == [
        (2, "periodic_lines"),
        (1, "periodic_lines"),
    ]
    sync = npc_sync.SyncDataset(recorded, patch=patch, on_invalid="raise")
    assert sync.invalid_intervals == raw.invalid_intervals and sync.total_seconds == 65.0
    report = timing.get_block_report(sync, [f["stim"]])
    assert report.status.tolist() == ["patched"]
    # frames: measured ones at their true times (those recorded after the first drop are
    # 1 s late on the recorded clock, 1.5 s between the second drop and its copy), the
    # rest reconstructed
    v = sync.get_falling_edges("vsync_stim", units="seconds")
    assert len(v) == len(f["frames"]) and np.max(np.abs(v - f["frames"])) < 2 / FS
    (display,) = npc_stim.get_display_times(sync, [f["stim"]])
    assert display.source == "photodiode" and len(display.display_times) == len(f["frames"])
    # licks: measured at their true times outside the dropped intervals, from lickFrames
    # (+- 1/2 frame) inside
    licks = sync.get_rising_edges("lick_sensor", units="seconds")
    assert not sync.get_unknown_intervals("lick_sensor")
    truth = f["lick_onsets"]
    inside = npc_sync.in_intervals(truth, sync.invalid_intervals)
    assert inside.sum() >= 3
    err = np.array([np.min(np.abs(licks - t)) for t in truth])
    assert np.max(err[~inside]) < 1.5 / FS and np.max(err[inside]) < 0.02
    assert len(licks) == len(truth)
    replayed = timing.get_replayed_intervals(sync)
    assert [(r["start_time"], r["stop_time"]) for r in replayed] == [(30.0, 31.0), (40.0, 40.5)]
    assert "dropped acquisition buffers" in replayed[0]["reason"]


def test_barcode_spans_take_whole_barcodes_only() -> None:
    """703880_2024-04-18: with 2 s of padding to where the line is low, a span ended between
    two pulses of a barcode that started 1.3 s after a dropped interval (2 of 11 pulses left,
    onset 765 ms late), and barcodes that didn't touch a dropped interval were removed.
    708020_2024-05-17: the tail of a barcode whose onset fell inside a dropped interval."""
    onsets = 5.0 + 31.0 * np.arange(6)
    rising = np.concatenate([t + 0.09 * np.arange(11) for t in onsets])
    falling = rising + 0.03

    def n_in(spans, x=rising) -> int:
        return int(npc_sync.in_intervals(x, spans).sum())

    # dropped interval ending 1.3 s before the second barcode, which is on time: kept
    span = (34.2, 34.7)
    assert timing.barcode_spans([span], rising, falling) == [span]
    old = timing._span_with_line_low_at_bounds((span[0] - 2, span[1] + 2), rising, falling)
    kept = rising[~npc_sync.in_intervals(rising, [old])]
    assert 0 < ((kept >= 36.0) & (kept < 38.0)).sum() < 11  # what builder 5 left: a fragment
    # dropped interval inside the third barcode: all of it goes, the others stay
    got = timing.barcode_spans([(67.3, 67.8)], rising, falling)
    assert n_in(got) == 11 and n_in(got, falling) == 11 and n_in([(67.0, 68.0)]) == 11
    # the fourth barcode's first 0.5 s is lost in a dropped interval: its tail goes too
    lost = (97.9, 98.5)
    r2 = rising[~npc_sync.in_intervals(rising, [lost])]
    f2 = falling[~npc_sync.in_intervals(falling, [lost])]
    n_tail = n_in([(98.5, 100.0)], r2)
    assert 0 < n_tail < 11
    got = timing.barcode_spans([lost], r2, f2)
    assert n_in(got, r2) == n_tail and n_in(got, f2) == n_in([(98.5, 100.0)], f2)
    # a barcode that starts 1 s before a dropped interval may continue into it: removed
    got = timing.barcode_spans([(130.0, 130.5)], rising, falling)
    assert n_in(got) == 11
    # several spans at once: only the incomplete barcodes go
    got = timing.barcode_spans([span, (67.3, 67.8), (130.0, 130.5)], rising, falling)
    assert n_in(got) == 22


def test_barcode_spans_cover_npc_ephys_padding() -> None:
    """npc_ephys removes barcode edges within 2 s of an invalid interval, edge by edge, and
    requires equal rising and falling counts afterwards: with the builder's padding, no edge
    is left for it to remove, whatever the phase of the barcodes."""
    pad = timing.BARCODE_SPAN_PADDING_SEC
    rng = np.random.default_rng(0)
    onsets = 5.0 + 31.0 * np.arange(40)
    rising = np.concatenate([t + np.sort(rng.uniform(0, 1.7, 12)) for t in onsets])
    falling = rising + 0.01
    intervals = [(float(a), float(a) + 0.5) for a in np.sort(rng.uniform(40, 1200, 60))]
    spans = timing.barcode_spans(intervals, rising, falling, pad)
    on = rising[~npc_sync.in_intervals(rising, spans)]
    off = falling[~npc_sync.in_intervals(falling, spans)]
    assert len(on) == len(off) and len(on) % 12 == 0  # whole barcodes only
    ephys_cut = [(a - pad, z + pad) for a, z in intervals]
    assert not npc_sync.in_intervals(on, ephys_cut).any()
    assert not npc_sync.in_intervals(off, ephys_cut).any()
