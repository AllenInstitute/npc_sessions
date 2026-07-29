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
