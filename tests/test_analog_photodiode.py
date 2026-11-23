"""npc_sessions.timing: photodiode edges inside replayed sync intervals from the NI-DAQ analog
photodiode (`reconstruct_photodiode_line`), on synthetic data."""

from __future__ import annotations

import copy
import types

import numpy as np
import pytest

npc_sync = pytest.importorskip("npc_sync")
if not hasattr(npc_sync, "SyncPatch"):
    pytest.skip("npc_sync without SyncPatch", allow_module_level=True)

from npc_sessions import timing  # noqa: E402

FS = 30_000.0
RISE_LEVEL, FALL_LEVEL = 30.0, -15.0
BLACK, WHITE = -140.0, 1000.0
SPAN = (20.0, 21.5)
START = 3.0  # sync time of the first NI-DAQ sample


def _trace(toggles: np.ndarray, first_rising: bool, t: np.ndarray, seed: int = 0):
    """Analog photodiode (fast rise, slow fall, noise) for display toggles at `toggles`, and
    the edges a comparator with hysteresis makes from it (from the noiseless trace)."""
    rng = np.random.default_rng(seed)
    y = np.full(len(t), BLACK if first_rising else WHITE)
    rising, falling = [], []
    pol = first_rising
    for i, t0 in enumerate(toggles):
        t1 = toggles[i + 1] if i + 1 < len(toggles) else np.inf
        m = (t >= t0) & (t < t1)
        y0 = y[np.searchsorted(t, t0) - 1]
        target, tau = (WHITE, 0.0015) if pol else (BLACK, 0.004)
        y[m] = target + (y0 - target) * np.exp(-(t[m] - t0) / tau)
        level = RISE_LEVEL if pol else FALL_LEVEL
        cross = t0 + tau * np.log((y0 - target) / (level - target))
        (rising if pol else falling).append(cross)
        pol = not pol
    return y + rng.normal(0, 4, len(t)), np.array(rising), np.array(falling)


@pytest.fixture(scope="module")
def data():
    """Box interval 3 at 60 Hz (a toggle every 50 ms), with the display 3 ms late for 0.5 s
    inside the span: the vsync-based prediction can't know."""
    t = START + np.arange(int(40 * FS)) / FS
    toggles = 5.0 + 0.05 * np.arange(600) + 0.0234
    late = (toggles > 20.4) & (toggles < 20.9)
    true_toggles = toggles + np.where(late, 0.003, 0.0)
    y, rising, falling = _trace(true_toggles, True, t)
    _, pred_r, pred_f = _trace(toggles, True, t)  # what vsync + latency predicts
    in_span = lambda x: (x >= SPAN[0]) & (x < SPAN[1])  # noqa: E731
    return types.SimpleNamespace(
        t=t,
        y=y,
        rising=rising,
        falling=falling,
        measured_r=rising[~in_span(rising)],
        measured_f=falling[~in_span(falling)],
        true_r=rising[in_span(rising)],
        true_f=falling[in_span(falling)],
        pred_r=pred_r[in_span(pred_r)],
        pred_f=pred_f[in_span(pred_f)],
    )


def _reconstruct(d, y=None, **kwargs):
    args = dict(
        rising=d.measured_r,
        falling=d.measured_f,
        span=SPAN,
        predicted_rising=d.pred_r,
        predicted_falling=d.pred_f,
    )
    args.update(kwargs)
    return timing.reconstruct_diode_edges_from_analog(
        d.t, d.y if y is None else y, **args
    )


def test_analog_edges_match_the_comparator(data) -> None:
    res = _reconstruct(data)
    assert res["status"] == "reconstructed", res.get("reason")
    assert len(res["rising"]) == len(data.true_r) == 15
    assert len(res["falling"]) == len(data.true_f) == 15
    # sub-0.1 ms against the comparator's edges, including the frames displayed 3 ms late
    assert np.max(np.abs(res["rising"] - data.true_r)) < 1e-4
    assert np.max(np.abs(res["falling"] - data.true_f)) < 2e-4
    assert np.max(np.abs(data.pred_r - data.true_r)) > 2.9e-3  # the prediction was off there
    assert abs(res["rise_level"] - RISE_LEVEL) < 10
    assert abs(res["fall_level"] - FALL_LEVEL) < 10
    assert res["max_abs_residual_sec"] < 2e-4
    assert res["analog_minus_predicted_ms"]["max"] == pytest.approx(3.0, abs=0.2)


def test_time_map_error_is_taken_up_by_the_offsets(data) -> None:
    """A constant error of the NI-DAQ -> sync map of 0.1 ms (observed: <= 0.07 ms) doesn't
    move the edges: the levels are read from the trace at the measured edges. A larger one
    puts the rising level in the noise of the black level: the gate fails, or holds the
    edges within its tolerance."""
    for shift in (0.0001, -0.0001):
        res = _reconstruct(data, y=np.interp(data.t - shift, data.t, data.y))
        assert res["status"] == "reconstructed", res.get("reason")
        assert np.max(np.abs(res["rising"] - data.true_r)) < 1e-4
        assert np.max(np.abs(res["falling"] - data.true_f)) < 2e-4
    for shift in (0.0004, 0.001, 0.003):
        res = _reconstruct(data, y=np.interp(data.t - shift, data.t, data.y))
        if res["status"] == "reconstructed":
            err = np.r_[res["rising"] - data.true_r, res["falling"] - data.true_f]
            assert np.max(np.abs(err)) < 2 * timing.ANALOG_DIODE_MAX_DEVIATION_SEC


def test_gate_analog_must_reproduce_measured_edges(data) -> None:
    rng = np.random.default_rng(0)
    # a channel that isn't the photodiode
    res = _reconstruct(data, y=rng.normal(0, 100, len(data.t)))
    assert res["status"] == "failed"
    # the photodiode with jitter >> 1 ms relative to the sync line (e.g. a bad time map)
    jitter = np.interp(data.t, data.t[:: int(FS)], rng.normal(0, 0.004, 40))
    res = _reconstruct(data, y=np.interp(data.t + jitter, data.t, data.y))
    assert res["status"] == "failed" and "doesn't reproduce" in res["reason"]
    # an analog transition near the span that the sync line doesn't have
    y = data.y.copy()
    i = int((18.012 - START) * FS)
    y[i : i + 60] = WHITE
    res = _reconstruct(data, y=y)
    assert res["status"] == "failed" and "without a measured one" in res["reason"]


def test_gate_edges_in_span_must_be_one_to_one_with_prediction(data) -> None:
    # a transition inside the span that the stim file doesn't predict (e.g. phantom flicker)
    y = data.y.copy()
    i = int((20.662 - START) * FS)  # white phase
    y[i : i + 150] = BLACK
    res = _reconstruct(data, y=y)
    assert res["status"] == "failed" and "predicted from vsyncs" in res["reason"]
    # edges half a toggle interval or more from the prediction: assignment ambiguous
    res = _reconstruct(
        data, predicted_rising=data.pred_r + 0.03, predicted_falling=data.pred_f + 0.03
    )
    assert res["status"] == "failed" and "ambiguous" in res["reason"]


def test_edge_just_outside_the_span_is_put_at_its_bound(data) -> None:
    """The sync line was recorded outside the span, without the edge: an analog-derived edge
    that lands just outside (model error) belongs inside."""
    first = float(data.true_r[0])
    span = (first + 2e-5, SPAN[1])  # the comparator's edge is 20 us before the span
    res = _reconstruct(
        data,
        span=span,
        predicted_rising=np.r_[span[0] + 1e-5, data.pred_r[1:]],
    )
    assert res["status"] == "reconstructed", res.get("reason")
    assert res["n_moved_to_span_bounds"] == 1
    assert res["rising"][0] == span[0]
    assert res["max_moved_to_span_bounds_sec"] < 1e-4
    assert np.all((res["rising"] >= span[0]) & (res["rising"] < span[1]))
    assert np.all((res["falling"] >= span[0]) & (res["falling"] < span[1]))


def test_gate_needs_measured_edges_on_both_sides(data) -> None:
    res = _reconstruct(
        data,
        rising=data.measured_r[data.measured_r < SPAN[0]],
        falling=data.measured_f[data.measured_f < SPAN[0]],
    )
    assert res["status"] == "failed"
    res = _reconstruct(
        data,
        rising=data.measured_r[data.measured_r < SPAN[1] + 0.3],
        falling=data.measured_f[data.measured_f < SPAN[1] + 0.3],
    )
    assert res["status"] == "failed" and "too few" in res["reason"]


def test_channel_is_found_by_correlation(data) -> None:
    rng = np.random.default_rng(1)
    sel = slice(int(5 * FS), int(30 * FS))
    x = rng.normal(0, 50, (sel.stop - sel.start, 4))
    x[:, 0] = 32763  # AI0 at the rail, as before 2025-01
    x[:, 1] = data.y[sel]
    c = timing.analog_diode_channel_correlations(
        data.t[sel], x, data.measured_r, data.measured_f, [SPAN]
    )
    assert c[0] is None and c[1] > 0.8 and abs(c[2]) < 0.05
    assert timing.select_analog_diode_channel(c) == 1
    x[:, 1] = rng.normal(0, 50, len(x))
    c = timing.analog_diode_channel_correlations(
        data.t[sel], x, data.measured_r, data.measured_f, [SPAN]
    )
    assert timing.select_analog_diode_channel(c) is None


# ------------------------------------------------------------------ the patch step


class _View:
    def __init__(self, d):
        self._edges = {
            "rising": np.sort(np.r_[d.measured_r, d.pred_r]),
            "falling": np.sort(np.r_[d.measured_f, d.pred_f]),
        }
        self._measured = {"rising": d.measured_r, "falling": d.measured_f}

    def get_edges(self, kind, line, units="seconds"):
        assert line == "stim_photodiode" and units == "seconds"
        return self._edges[kind]

    def get_edge_sources(self, line, kind):
        return (~np.isin(self._edges[kind], self._measured[kind])).astype(np.int8)


def _patch(d, method="vsync+latency", status="reconstructed"):
    other = (30.0, 30.5)  # a second replayed span, with no edges predicted
    return npc_sync.SyncPatch(
        sync_content_hash="x",
        builder="test",
        invalid_intervals=(SPAN, other),
        lines={
            "stim_photodiode": [
                npc_sync.SpanPatch(
                    SPAN,
                    status,
                    rising=d.pred_r if status == "reconstructed" else np.array([]),
                    falling=d.pred_f if status == "reconstructed" else np.array([]),
                    method=method,
                    error_bound_sec=6e-4,
                    diagnostics={"n_inserted": 30},
                ),
                npc_sync.SpanPatch(other, "reconstructed", method="no_edges"),
            ]
        },
    )


@pytest.fixture
def nidaq(data, monkeypatch):
    """npc_ephys stand-ins: 4 channels (one at the rail), the photodiode on `channel`
    (settable; None: not recorded)."""
    npc_ephys = pytest.importorskip("npc_ephys")
    state = types.SimpleNamespace(channel=0, calls=0, valid=True, drops=())
    device = types.SimpleNamespace(device=types.SimpleNamespace(name="NI-DAQmx-1"))

    def get_data(*dirs, device_name=None):
        state.calls += 1
        x = np.random.default_rng(2).normal(0, 30, (len(data.t), 4)).astype(np.int16)
        x[:, 2] = 32763
        if state.channel is not None:
            x[:, state.channel] = np.round(data.y)
        return x

    def get_timing(sync, recording_dirs, devices):
        return (
            types.SimpleNamespace(
                is_valid=state.valid,
                reason="barcodes",
                drops=state.drops,
                start_time=START,
                sampling_rate=FS,
                to_sync_time=lambda k: START + np.asarray(k) / FS,
            ),
        )

    monkeypatch.setattr(npc_ephys, "get_pxi_nidaq_info", lambda dirs: device)
    monkeypatch.setattr(npc_ephys, "get_ephys_timing_on_sync", get_timing)
    monkeypatch.setattr(npc_ephys, "get_pxi_nidaq_data", get_data)
    monkeypatch.setattr(timing, "_patched_view", lambda raw, patch: _View(data))
    return state


@pytest.mark.parametrize("channel", [0, 1])
def test_patch_span_replaced_by_analog_edges(data, nidaq, channel) -> None:
    nidaq.channel = channel
    patch = _patch(data)
    info = timing.reconstruct_photodiode_line(None, patch, (), ("recording",))
    s, other = patch.lines["stim_photodiode"]
    assert s.method == timing.ANALOG_DIODE_METHOD and s.status == "reconstructed"
    assert s.span == SPAN and other.method == "no_edges"
    assert np.max(np.abs(s.rising - data.true_r)) < 1e-4
    assert np.max(np.abs(s.falling - data.true_f)) < 2e-4
    assert s.diagnostics["nidaq_channel"] == channel
    assert s.diagnostics["replaced"]["method"] == "vsync+latency"
    assert s.diagnostics["replaced"]["rising"] == [float(v) for v in data.pred_r]
    assert not patch.notes and len(info) == 1
    # serialisable, and a second pass leaves it alone
    again = npc_sync.SyncPatch.from_json(patch.to_json())
    assert np.array_equal(again.lines["stim_photodiode"][0].rising, s.rising)
    calls = nidaq.calls
    assert timing.reconstruct_photodiode_line(None, patch, (), ("recording",)) == {}
    assert nidaq.calls == calls


def _same(a, b) -> bool:
    return a.to_dict() == b.to_dict()


def test_patch_unchanged_without_nidaq_or_spans(data, nidaq, monkeypatch) -> None:
    patch = _patch(data)
    before = copy.deepcopy(patch)
    assert timing.reconstruct_photodiode_line(None, patch, (), None) == {}
    assert timing.reconstruct_photodiode_line(None, patch, (), ()) == {}
    assert _same(patch, before) and nidaq.calls == 0
    # no predicted edges to replace (unknown span, or a file without replayed intervals)
    for p in (
        _patch(data, status="unknown", method="mask"),
        npc_sync.SyncPatch(sync_content_hash="x", builder="test", invalid_intervals=()),
    ):
        before = copy.deepcopy(p)
        assert timing.reconstruct_photodiode_line(None, p, (), ("recording",)) == {}
        assert _same(p, before) and nidaq.calls == 0


def test_patch_keeps_prediction_when_a_gate_fails(data, nidaq) -> None:
    # no analog photodiode channel (older sessions: AI0 at the rail)
    nidaq.channel = None
    patch = _patch(data)
    before = copy.deepcopy(patch)
    timing.reconstruct_photodiode_line(None, patch, (), ("recording",))
    assert [s.to_dict() for s in patch.lines["stim_photodiode"]] == [
        s.to_dict() for s in before.lines["stim_photodiode"]
    ]
    assert len(patch.notes) == 1 and "no NI-DAQ channel" in patch.notes[0]
    # NI-DAQ timing not usable
    for kwargs in ({"valid": False}, {"drops": ((100.0, 3.0),)}):
        nidaq.channel, nidaq.valid, nidaq.drops = 0, True, ()
        for k, v in kwargs.items():
            setattr(nidaq, k, v)
        patch = _patch(data)
        timing.reconstruct_photodiode_line(None, patch, (), ("recording",))
        assert patch.lines["stim_photodiode"][0].method == "vsync+latency"
        assert len(patch.notes) == 1 and "NI-DAQ timing" in patch.notes[0]


def test_build_runs_the_photodiode_step_only_with_replayed_intervals(
    monkeypatch,
) -> None:
    """`build_session_sync_patch` doesn't reach the NI-DAQ for a file without replayed
    intervals (every session has a patch)."""
    called = []
    monkeypatch.setattr(
        timing, "reconstruct_photodiode_line", lambda *a, **k: called.append(1)
    )
    monkeypatch.setattr(timing, "reconstruct_sound_line", lambda *a, **k: {})
    monkeypatch.setattr(timing, "reconstruct_lick_line", lambda *a, **k: {})
    monkeypatch.setattr(timing, "reconstruct_opto_lines", lambda *a, **k: {})
    monkeypatch.setattr(timing, "widen_unknown_lick_spans", lambda *a, **k: None)
    monkeypatch.setattr(timing, "true_clock_view", lambda sync: sync)
    monkeypatch.setattr(timing, "get_stim_blocks", lambda raw, paths: [])
    for invalid, expected in (((), 0), (((1.0, 1.5),), 1)):
        patch = npc_sync.SyncPatch("x", "test", invalid_intervals=invalid)
        monkeypatch.setattr(
            npc_sync, "build_sync_patch", lambda *a, patch=patch, **k: patch
        )
        called.clear()
        timing.build_session_sync_patch("sync", ("stim",), ("recording",))
        assert len(called) == expected
