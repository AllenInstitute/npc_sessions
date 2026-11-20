from types import SimpleNamespace

import numpy as np
import pytest

from npc_sessions.trials.TaskControl.OptoTagging import OptoTagging


def make_trials(
    *,
    onset_times: tuple[float, ...] = (10.0, 11.0),
    offset_times: tuple[float, ...] = (10.01, 11.2),
    inter_trial_interval: float = 1.0,
) -> OptoTagging:
    trials = OptoTagging.__new__(OptoTagging)
    trials.__dict__["_stim_recordings"] = tuple(
        SimpleNamespace(
            onset_time_on_sync=onset_time,
            offset_time_on_sync=offset_time,
        )
        for onset_time, offset_time in zip(
            onset_times, offset_times, strict=True
        )
    )
    trials.__dict__["trial_index"] = np.arange(len(onset_times))
    trials.__dict__["_inter_trial_interval"] = inter_trial_interval
    return trials


def test_analysis_periods_exclude_opto_censor_periods() -> None:
    trials = make_trials()

    np.testing.assert_allclose(trials.start_time, [9.8, 10.8])
    np.testing.assert_allclose(trials.stop_time, [10.21, 11.4])
    np.testing.assert_allclose(trials.baseline_start_time, trials.start_time)
    np.testing.assert_allclose(trials.baseline_stop_time, [9.9985, 10.9985])
    np.testing.assert_allclose(trials.response_start_time, [10.0015, 11.0015])
    np.testing.assert_allclose(trials.response_stop_time, [10.0085, 11.1985])


def test_trial_period_is_clipped_to_half_the_inter_trial_interval() -> None:
    trials = make_trials(inter_trial_interval=0.1)

    assert trials._analysis_period == pytest.approx(0.05)
    np.testing.assert_allclose(trials.start_time, [9.95, 10.95])
    np.testing.assert_allclose(trials.stop_time, [10.06, 11.25])


def make_interleaved_trials() -> OptoTagging:
    """Stim data as saved by OptoTagging.py from samgale/DynamicRoutingTask@55e88fa:
    interleaved lasers, `trialOptoPower` instead of `trialOptoVoltage`, and
    wavelength-suffixed labels (with non-probe control sites)."""
    import h5py

    f = h5py.File("interleaved.hdf5", "w", driver="core", backing_store=False)
    f["trialOptoOnsetFrame"] = np.array([60, 120, 180, 240])
    f["trialOptoDevice"] = np.array(
        [b"['laser_633']", b"['laser_488']", b"['laser_488']", b"['laser_633']"],
        dtype=h5py.string_dtype(),
    )
    f["trialOptoPower"] = np.array([5.0, 10.0, 2.5, 10.0])
    f["trialOptoLabel"] = np.array(
        [b"F_633", b"A_488", b"off_brain_488", b"str_633"], dtype=h5py.string_dtype()
    )
    trials = OptoTagging.__new__(OptoTagging)
    trials.__dict__["_hdf5_data"] = f
    trials.__dict__["trial_index"] = np.arange(4)
    return trials


def test_interleaved_lasers_use_per_trial_device() -> None:
    trials = make_interleaved_trials()
    assert trials._trial_opto_device == (
        "laser_633",
        "laser_488",
        "laser_488",
        "laser_633",
    )
    np.testing.assert_array_equal(trials.wavelength, [633, 488, 488, 633])


def test_power_from_trial_opto_power() -> None:
    trials = make_interleaved_trials()
    np.testing.assert_array_equal(trials.power, [5.0, 10.0, 2.5, 10.0])


def test_location_strips_wavelength_suffix() -> None:
    trials = make_interleaved_trials()
    np.testing.assert_array_equal(
        trials.location, ["probeF", "probeA", "off_brain", "str"]
    )
