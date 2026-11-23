"""npc_sessions.qc: quartile visual-latency check on synthetic spikes."""

from __future__ import annotations

import json
import pathlib
import types

import numpy as np
import pytest

from npc_sessions import qc

N_STIMS = 200
ISI = 1.0
T0 = 10.0


def make_onsets(n: int = N_STIMS) -> np.ndarray:
    return T0 + ISI * np.arange(n)


def make_unit(
    rng: np.random.Generator,
    onsets: np.ndarray,
    latency: float | np.ndarray,
    responds: np.ndarray | None = None,
    baseline_hz: float = 5.0,
    response_hz: float = 80.0,
    duration: float = 0.05,
) -> np.ndarray:
    """Poisson baseline + a burst from `latency` (s, scalar or per stim) after each onset"""
    stop = onsets[-1] + 5
    spikes = [rng.uniform(0, stop, rng.poisson(baseline_hz * stop))]
    latency = np.broadcast_to(latency, onsets.shape)
    for i, (t, lat) in enumerate(zip(onsets, latency)):
        if responds is not None and not responds[i]:
            continue
        n = rng.poisson(response_hz * duration)
        spikes.append(t + lat + rng.uniform(0, duration, n))
    return np.sort(np.concatenate(spikes))


def make_units(
    n_units: int,
    onsets: np.ndarray,
    step: float = 0.0,
    step_from: int | None = None,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    delay = np.zeros(len(onsets))
    if step_from is not None:
        delay[step_from:] = step
    return {
        f"unit_{i}": make_unit(rng, onsets, 0.04 + 0.002 * i + delay)
        for i in range(n_units)
    }


def test_stable_latency_passes() -> None:
    onsets = make_onsets()
    result = qc.quartile_latency_check(make_units(10, onsets), onsets, block="test")
    assert result.verdict == "passed" and result.passed
    assert result.vis.n_units == 10
    assert [q.n_trials for q in result.vis.quartiles] == [50] * 4
    assert result.vis_range_ms <= 3
    assert all(
        q.ci_low_ms <= q.latency_ms <= q.ci_high_ms for q in result.vis.quartiles
    )
    assert 40 < result.vis.quartiles[0].latency_ms < 60
    assert result.frac_shifted < 0.1
    assert result.aud is None and np.isnan(result.aud_range_ms)
    assert result.summary.startswith("visual response latency check passed")
    assert "\n" not in result.summary


def test_latency_step_fails() -> None:
    """Display 20 ms late from 60% of the way through the block"""
    onsets = make_onsets()
    result = qc.quartile_latency_check(
        make_units(10, onsets, step=0.02, step_from=int(0.6 * N_STIMS)), onsets
    )
    assert result.verdict == "failed" and not result.passed
    assert 17 < result.vis_range_ms < 23
    latencies = result.vis.latencies_ms
    assert abs(latencies[1] - latencies[0]) < 3  # before the step
    assert latencies[3] - latencies[0] > 17
    assert result.stim_shifts is not None
    assert result.stim_shifts.frac_shifted_by_quartile[0] < 0.1
    assert result.stim_shifts.frac_shifted_by_quartile[3] > 0.9
    assert 17 <= result.stim_shifts.median_shift_ms_by_quartile[3] <= 23
    assert 0.3 < result.frac_shifted < 0.5
    assert "> 8 ms" in result.reason


def test_too_few_units_untestable() -> None:
    onsets = make_onsets()
    # the latency step isn't a failure without enough units
    units = make_units(4, onsets, step=0.02, step_from=100)
    rng = np.random.default_rng(1)
    for i in range(20):  # unresponsive units don't count
        units[f"flat_{i}"] = np.sort(rng.uniform(0, onsets[-1] + 5, 2000))
    result = qc.quartile_latency_check(units, onsets)
    assert result.verdict == "untestable"
    assert result.vis.n_units == 4 and result.vis.n_units_tested == 24
    assert "4 visually responsive units" in result.reason
    assert result.stim_shifts is None
    relaxed = qc.quartile_latency_check(units, onsets, min_responsive_units=4)
    assert relaxed.verdict == "failed"


def test_no_units_or_stims_untestable() -> None:
    onsets = make_onsets()
    assert qc.quartile_latency_check({}, onsets).verdict == "untestable"
    assert qc.quartile_latency_check([], onsets).reason == "no units"
    units = make_units(10, onsets)
    few = qc.quartile_latency_check(units, onsets[:39])
    assert few.verdict == "untestable" and "39 visual stims" in few.reason
    assert few.vis.quartiles == ()
    nan = qc.quartile_latency_check(units, np.full(100, np.nan))
    assert nan.verdict == "untestable"
    json.dumps(few.to_dict(include_arrays=True), allow_nan=False)


def test_deterministic_and_serialisable() -> None:
    onsets = make_onsets()
    units = make_units(8, onsets, step=0.02, step_from=150)
    aud = onsets + 0.5
    a = qc.quartile_latency_check(units, onsets, aud)
    again = qc.quartile_latency_check(units, onsets, aud)
    text = json.dumps(a.to_dict(include_arrays=True), allow_nan=False)
    assert text == json.dumps(again.to_dict(include_arrays=True), allow_nan=False)
    assert json.loads(text)["verdict"] == a.verdict
    # same result (and CIs: fixed seed) whatever the order of the onsets
    b = qc.quartile_latency_check(units, onsets[::-1].copy(), aud[::-1].copy())
    for qa, qb in zip(a.vis.quartiles, b.vis.quartiles):
        assert qa.latency_ms == qb.latency_ms
        assert (qa.ci_low_ms, qa.ci_high_ms) == (qb.ci_low_ms, qb.ci_high_ms)
    small = a.to_dict()
    assert "psth" not in small["vis"]["quartiles"][0]
    assert "shift_ms" not in small["stim_shifts"]
    # the verdict doesn't depend on the bootstrap
    c = qc.quartile_latency_check(
        units, onsets, n_bootstrap=0, seed=5, stim_shifts=False
    )
    assert c.verdict == a.verdict and c.vis.latencies_ms == a.vis.latencies_ms
    assert np.isnan(c.vis.quartiles[0].ci_low_ms) and c.stim_shifts is None
    d = qc.quartile_latency_check(units, onsets, seed=5)
    assert d.vis.latencies_ms == a.vis.latencies_ms
    assert d.vis.quartiles[0].ci_low_ms != a.vis.quartiles[0].ci_low_ms


def test_auditory_control() -> None:
    """Sound onsets don't move with the display: the control stays flat when vis fails"""
    rng = np.random.default_rng(3)
    vis = make_onsets()
    aud = vis + 0.5
    delay = np.where(np.arange(N_STIMS) >= 100, 0.02, 0.0)
    units = {f"v{i}": make_unit(rng, vis, 0.04 + delay) for i in range(6)}
    units |= {
        f"a{i}": make_unit(rng, aud, 0.012, duration=0.03, response_hz=120)
        for i in range(6)
    }
    result = qc.quartile_latency_check(units, vis, aud)
    assert result.verdict == "failed"
    assert result.aud is not None and result.aud.n_units == 6
    assert result.aud_range_ms < 4
    assert set(result.vis.unit_ids) == {f"v{i}" for i in range(6)}


def test_conditions() -> None:
    """RF mapping: each unit responds to stims at one of 6 positions only"""
    rng = np.random.default_rng(4)
    onsets = make_onsets(960)
    conditions = rng.permutation(np.arange(960) % 6) * 10  # arbitrary labels
    units = {
        f"unit_{i}": make_unit(rng, onsets, 0.045, responds=conditions == 10 * (i % 6))
        for i in range(12)
    }
    pooled = qc.quartile_latency_check(units, onsets, min_rate_increase_hz=20)
    assert pooled.verdict == "untestable"  # responses diluted over positions
    result = qc.quartile_latency_check(units, onsets, vis_conditions=conditions)
    assert result.verdict == "passed" and result.vis.n_units == 12
    assert result.params["n_vis_conditions"] == 6
    late = onsets >= onsets[480]
    shifted = {
        k: make_unit(
            rng, onsets, 0.045 + 0.017 * late, responds=conditions == 10 * (i % 6)
        )
        for i, k in enumerate(units)
    }
    result = qc.quartile_latency_check(shifted, onsets, vis_conditions=conditions)
    assert result.verdict == "failed" and 14 < result.vis_range_ms < 20
    with pytest.raises(ValueError):
        qc.quartile_latency_check(units, onsets, vis_conditions=conditions[:-1])


def test_block_bounds() -> None:
    onsets = make_onsets()
    units = make_units(6, onsets)
    result = qc.quartile_latency_check(
        units, onsets, block_start=onsets[40], block_stop=onsets[159]
    )
    assert result.vis.n_trials == 120
    assert result.vis.quartiles[0].start_time == onsets[40]
    assert result.vis.quartiles[-1].stop_time == onsets[159]


def test_estimate_stim_shifts() -> None:
    onsets = make_onsets()
    units = make_units(10, onsets, step=-0.01, step_from=100)
    template = np.arange(N_STIMS) < 50
    shifts = qc.estimate_stim_shifts(list(units.values()), onsets, template)
    assert abs(np.median(shifts[:100])) <= 0.002
    assert -0.013 <= np.median(shifts[100:]) <= -0.007


def test_session_wrapper_with_explicit_onsets() -> None:
    """With onsets and spike times given, the session is only asked for its stim paths"""
    onsets = make_onsets()
    session = types.SimpleNamespace(
        id="000000_2000-01-01",
        stim_paths=(
            pathlib.Path("DynamicRouting1_000000_20000101_120000.hdf5"),
            pathlib.Path("RFMapping_000000_20000101_110000.hdf5"),
        ),
    )
    result = qc.quartile_visual_latency(
        session,  # type: ignore[arg-type]
        "DynamicRouting1",
        vis_onsets=onsets,
        spike_times=make_units(6, onsets),
    )
    assert result.passed
    assert result.block == "000000_2000-01-01 DynamicRouting1_000000_20000101_120000"
    with pytest.raises(ValueError):
        qc.get_stim_path(session, "Spontaneous")  # type: ignore[arg-type]
    path = qc.get_stim_path(session, "RFMapping_000000_20000101_110000")  # type: ignore[arg-type]
    assert path.stem.startswith("RFMapping")


# criterion (c): whole-block latency vs photodiode blocks of the same stim ------------- #


def make_blocks(
    delay: float = 0.0, n_units: int = 10, seed: int = 0
) -> tuple[dict[str, np.ndarray], np.ndarray, list[np.ndarray]]:
    """Spikes for a block of 100 stims between two reference blocks of 100 and 400; the
    display is `delay` s late throughout the middle block"""
    first, block, last = T0 + np.arange(100), T0 + 200 + np.arange(100), T0 + 400 + np.arange(400)
    onsets = np.concatenate([first, block, last])
    lateness = np.where((onsets >= block[0]) & (onsets <= block[-1]), delay, 0.0)
    rng = np.random.default_rng(seed)
    units = {
        f"unit_{i}": make_unit(rng, onsets, 0.04 + 0.002 * i + lateness)
        for i in range(n_units)
    }
    return units, block, [first, last]


def test_reference_latency_same_as_photodiode_blocks_passes() -> None:
    units, block, references = make_blocks()
    result = qc.reference_latency_check(units, block, references, block="b", reference_blocks=["r0", "r1"])
    assert result.verdict == "passed" and result.passed
    assert abs(result.latency_diff_ms) < 3
    assert result.n_units == 10 and result.block.n_trials == 100
    assert [r.n_trials for r in result.references] == [100, 400]
    assert [r.block for r in result.references] == ["r0", "r1"]
    assert result.reference_latency_ms == pytest.approx(
        np.median([r.latency_ms for r in result.references])
    )
    assert result.block.ci_low_ms <= result.block.latency_ms <= result.block.ci_high_ms
    assert result.summary.startswith("photodiode reference latency check passed")
    assert "\n" not in result.summary


def test_reference_latency_whole_block_late_fails() -> None:
    """A display 20 ms late for the whole block: stable over quartiles, so criterion (b)
    passes, but the latency differs from the photodiode blocks'"""
    units, block, references = make_blocks(delay=0.02)
    assert qc.quartile_latency_check(units, block).verdict == "passed"
    result = qc.reference_latency_check(units, block, references)
    assert result.verdict == "failed" and not result.passed
    assert 17 < result.latency_diff_ms < 23
    assert "more than 8 ms" in result.reason
    # the other way round (a reference block as the one checked) fails too
    assert qc.reference_latency_check(units, references[0], [block]).latency_diff_ms < -17


def test_reference_latency_untestable() -> None:
    units, block, references = make_blocks()
    assert qc.reference_latency_check({}, block, references).reason == "no units"
    none = qc.reference_latency_check(units, block, [])
    assert none.verdict == "untestable" and "no reference block" in none.reason
    small = qc.reference_latency_check(units, block, [references[0][:39]])
    assert small.verdict == "untestable" and "no reference block" in small.reason
    few_stims = qc.reference_latency_check(units, block[:39], references)
    assert few_stims.verdict == "untestable" and "39 visual stims" in few_stims.reason
    few_units, block, references = make_blocks(delay=0.02, n_units=4)
    result = qc.reference_latency_check(few_units, block, references)
    assert result.verdict == "untestable" and "4 visually responsive units" in result.reason
    json.dumps(result.to_dict(include_arrays=True), allow_nan=False)
    # a reference block with too few stims isn't used, the other is
    units, block, references = make_blocks()
    one = qc.reference_latency_check(
        units, block, [references[0][:10], references[1]], min_pooled_reference_stims=400
    )
    small_pool = qc.reference_latency_check(units, block, [references[0][:10], references[1]])
    assert small_pool.verdict == "untestable" and "(< 500)" in small_pool.reason
    assert one.verdict == "passed" and np.isnan(one.references[0].latency_ms)
    assert one.reference_latency_ms == one.references[1].latency_ms


def test_reference_latency_with_conditions() -> None:
    """Units that respond at one of 4 positions (given as x, y per stim, with different
    values present in each block) are compared on the stims at their position"""
    rng = np.random.default_rng(0)
    positions = np.array([[-10.0, 0.0], [10.0, 0.0], [0.0, 10.0], [0.0, -10.0]])
    n = [200, 120, 400]
    onsets = [T0 + 1000 * k + 0.5 * np.arange(n_) for k, n_ in enumerate(n)]
    index = [rng.integers(0, 4, n_) for n_ in n]
    all_onsets, all_index = np.concatenate(onsets), np.concatenate(index)
    for delay, verdict in ((0.0, "passed"), (0.02, "failed")):
        lateness = np.where((all_onsets >= onsets[1][0]) & (all_onsets <= onsets[1][-1]), delay, 0.0)
        units = {
            f"unit_{i}": make_unit(rng, all_onsets, 0.04 + lateness, responds=all_index == i % 4)
            for i in range(12)
        }
        result = qc.reference_latency_check(
            units,
            onsets[1],
            [onsets[0], onsets[2]],
            vis_conditions=positions[index[1]],
            reference_vis_conditions=[positions[index[0]], positions[index[2]]],
        )
        assert result.verdict == verdict, result.reason
        assert result.n_units == 12
    with pytest.raises(ValueError):
        qc.reference_latency_check(units, onsets[1], [onsets[0]], vis_conditions=positions[index[1]])


def test_reference_latency_deterministic() -> None:
    units, block, references = make_blocks(delay=0.005)
    a = qc.reference_latency_check(units, block, references)
    b = qc.reference_latency_check(dict(reversed(list(units.items()))), block[::-1], references)
    assert a.verdict == b.verdict
    assert a.latency_diff_ms == pytest.approx(b.latency_diff_ms, abs=1e-9)
    c = qc.reference_latency_check(units, block, references)
    assert json.dumps(a.to_dict(True)) == json.dumps(c.to_dict(True))
