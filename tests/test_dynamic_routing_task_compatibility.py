from collections.abc import Iterator

import h5py
import numpy as np
import pytest
from DynamicRoutingTask import TaskUtils

import npc_sessions.trials.sam as sam_utils
from npc_sessions.trials.TaskControl.DynamicRouting1 import DynamicRouting1


@pytest.fixture
def legacy_opto_data() -> Iterator[h5py.File]:
    n_trials = 6
    string = h5py.string_dtype()
    with h5py.File("legacy-opto.h5", "w", driver="core", backing_store=False) as data:
        data.create_dataset("rigName", data="NP0", dtype=string)
        data.create_dataset("taskVersion", data="stage 5 ori opto", dtype=string)
        data.create_dataset("startTime", data="20230101_120000", dtype=string)
        data.create_dataset("frameIntervals", data=np.full(140, 1 / 60))
        data.create_dataset("trialStartFrame", data=np.arange(0, 120, 20))
        data.create_dataset("trialEndFrame", data=np.arange(19, 139, 20))
        data.create_dataset("trialStimStartFrame", data=np.arange(5, 125, 20))
        data.create_dataset("newBlockAutoRewards", data=0)
        data.create_dataset("newBlockGoTrials", data=0)
        data.create_dataset("newBlockNogoTrials", data=0)
        data.create_dataset("newBlockCatchTrials", data=0)
        data.create_dataset("autoRewardOnsetFrame", data=6)
        data.create_dataset("trialRepeat", data=np.zeros(n_trials, dtype=bool))
        data.create_dataset("incorrectTrialRepeats", data=False)
        data.create_dataset("incorrectTimeoutFrames", data=60)
        data.create_dataset("quiescentFrames", data=30)
        data.create_dataset("quiescentViolationFrames", data=np.array([], dtype=int))
        data.create_dataset("responseWindow", data=[6, 60])
        data.create_dataset(
            "trialStim",
            data=["vis1", "vis2", "sound1", "sound2", "catch", "vis1"],
            dtype=string,
        )
        data.create_dataset("trialBlock", data=np.ones(n_trials, dtype=int))
        data.create_dataset("blockStimRewarded", data=["vis1"], dtype=string)
        data.create_dataset("rewardFrames", data=[12, 112])
        data.create_dataset("rewardSize", data=[0.005, 0.005])
        data.create_dataset(
            "trialResponse", data=[True, False, True, False, False, True]
        )
        data.create_dataset("trialResponseFrame", data=[10, 0, 50, 0, 0, 110])
        data.create_dataset(
            "trialRewarded", data=[True, False, False, False, False, True]
        )
        data.create_dataset(
            "trialAutoRewardScheduled", data=np.zeros(n_trials, dtype=bool)
        )
        data.create_dataset("trialAutoRewarded", data=np.zeros(n_trials, dtype=bool))
        data.create_dataset("lickFrames", data=np.array([], dtype=int))
        data.create_dataset("visStimContrast", data=1.0)
        data.create_dataset("trialVisStimContrast", data=np.ones(n_trials))
        data.create_dataset("gratingOri_vis1", data=0.0)
        data.create_dataset("gratingOri_vis2", data=90.0)
        data.create_dataset("trialGratingOri", data=np.zeros(n_trials))
        data.create_dataset("soundVolume", data=0.1)
        data.create_dataset("trialSoundVolume", data=np.full(n_trials, 0.1))

        data.create_dataset(
            "trialOptoOnsetFrame", data=[np.nan, 1, np.nan, 1, np.nan, 1]
        )
        data.create_dataset("trialOptoDur", data=np.full(n_trials, 0.5))
        data.create_dataset("trialOptoVoltage", data=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        data.create_dataset(
            "trialGalvoVoltage", data=np.tile([0.5, 0.5], (n_trials, 1))
        )
        data.create_dataset("optoRegions", data=["V1"], dtype=string)
        data.create_dataset("optoVoltage", data=[1.0])
        data.create_dataset("galvoVoltage", data=[[0.5, 0.5]])

        calibration = data.create_group("bregmaGalvoCalibrationData")
        calibration.create_dataset("bregmaX", data=[0.0, 0.0, 1.0, 1.0])
        calibration.create_dataset("bregmaY", data=[0.0, 1.0, 0.0, 1.0])
        calibration.create_dataset("galvoX", data=[0.0, 0.0, 1.0, 1.0])
        calibration.create_dataset("galvoY", data=[0.0, 1.0, 0.0, 1.0])
        power_calibration = data.create_group("optoPowerCalibrationData")
        power_calibration.create_dataset("poly coefficients", data=[0.0, 1.0, 0.0])
        yield data


def test_legacy_opto_data_loads_with_latest_dynamic_routing_task(
    legacy_opto_data: h5py.File,
) -> None:
    original_calibration_loader = TaskUtils.getBregmaGalvoCalibrationData

    sam = sam_utils.get_sam(legacy_opto_data)

    assert sam.nTrials == 6
    assert TaskUtils.getBregmaGalvoCalibrationData is original_calibration_loader


def test_legacy_variable_length_galvo_voltage_loads_with_latest_dynamic_routing_task(
    legacy_opto_data: h5py.File,
) -> None:
    n_trials = legacy_opto_data["trialEndFrame"].size
    del legacy_opto_data["trialGalvoVoltage"]
    trial_galvo_voltage = legacy_opto_data.create_dataset(
        "trialGalvoVoltage", shape=(n_trials,), dtype=h5py.vlen_dtype(np.float64)
    )
    for trial_index in range(n_trials):
        trial_galvo_voltage[trial_index] = (
            np.array([0.5]) if trial_index % 2 else np.array([np.nan, np.nan])
        )
    trial_opto_voltage = legacy_opto_data["trialOptoVoltage"][()]
    del legacy_opto_data["trialOptoVoltage"]
    legacy_opto_data.create_dataset(
        "trialOptoVoltage", data=trial_opto_voltage[:, None]
    )

    assert sam_utils.is_galvo_opto(legacy_opto_data)

    sam = sam_utils.get_sam(legacy_opto_data)

    expected = np.array([np.nan, 0.5, np.nan, 0.5, np.nan, 0.5])[:, None]
    assert np.array_equal(sam.trialGalvoX, expected, equal_nan=True)
    assert np.array_equal(sam.trialGalvoY, expected, equal_nan=True)


def test_empty_legacy_galvo_voltage_is_not_repaired() -> None:
    with h5py.File(
        "empty-legacy-galvo.h5", "w", driver="core", backing_store=False
    ) as data:
        data.create_dataset(
            "trialGalvoVoltage", shape=(0,), dtype=h5py.vlen_dtype(np.float64)
        )

        with sam_utils._repair_legacy_variable_length_galvo_voltage(data) as load_data:
            assert load_data is data


def test_legacy_galvo_layout_uses_raw_combined_coordinates(
    legacy_opto_data: h5py.File,
) -> None:
    trials = DynamicRouting1.__new__(DynamicRouting1)
    trials._hdf5_data = legacy_opto_data
    trials._len = 6

    assert trials._is_opto
    assert trials._is_galvo_opto
    assert not trials._is_galvo_voltage_xy_separate
    assert trials._galvo_voltage_xy == ((0.5, 0.5),) * 6
    assert np.nanmax(trials.opto_power) == 1.0


def test_modern_galvo_layout_uses_raw_split_coordinates() -> None:
    with h5py.File("modern-opto.h5", "w", driver="core", backing_store=False) as data:
        data.create_dataset("trialOptoOnsetFrame", data=[1.0, np.nan])
        data.create_dataset("trialGalvoX", data=[[0.1, 0.2], [np.nan, np.nan]])
        data.create_dataset("trialGalvoY", data=[[0.3, 0.4], [np.nan, np.nan]])
        trials = DynamicRouting1.__new__(DynamicRouting1)
        trials._hdf5_data = data
        trials._len = 2

        assert trials._is_opto
        assert trials._is_galvo_opto
        assert trials._is_galvo_voltage_xy_separate
        assert np.array_equal(trials._galvo_voltage_x[0], np.array([0.1, 0.2]))
        assert np.array_equal(trials._galvo_voltage_y[0], np.array([0.3, 0.4]))
