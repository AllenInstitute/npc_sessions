"""
getting Optotagging trials table requires:
- one or more hdf5 files with trial/stim data, called 'OptoTagging_*.hdf5'
- frame display times from sync, to use in place of frametimes in hdf5 file
- latency estimate for each stim presentation, to be added to frame
display times to get stim onset times
"""
from __future__ import annotations

import functools

import DynamicRoutingTask.TaskUtils
import numpy as np
import numpy.typing as npt

import npc_sessions.utils as utils
from npc_sessions.trials.TaskControl import TaskControl


class OptoTagging(TaskControl):
    """
    >>> stim = 's3://aind-ephys-data/ecephys_662892_2023-08-21_12-43-45/behavior/OptoTagging_662892_20230821_125915.hdf5'
    >>> sync = 's3://aind-ephys-data/ecephys_662892_2023-08-21_12-43-45/behavior/20230821T124345.h5'
    >>> trials = OptoTagging(stim, sync)
    >>> assert trials
    """

    @functools.cached_property
    def _stim_recordings(self) -> tuple[utils.StimRecording | None, ...] | None:
        if self._sync is not None:
            return utils.get_stim_latencies_from_sync(
                self._hdf5, self._sync, waveform_type="opto"
            )
        return None

    @functools.cached_property
    def trial_index(self) -> npt.NDArray[np.int32]:
        """0-indexed"""
        return np.arange(len(self._hdf5["trialOptoOnsetFrame"]))

    @functools.cached_property
    def start_time(self) -> npt.NDArray[np.float64]:
        if self._sync is None:
            return utils.safe_index(
                self._frame_times, self._hdf5["trialOptoOnsetFrame"][self.trial_index]
            )
        assert self._stim_recordings
        return np.array(
            [rec.onset_time_on_sync if rec else np.nan for rec in self._stim_recordings]
        )

    @functools.cached_property
    def stop_time(self) -> npt.NDArray[np.float64]:
        if self._sync is None:
            return self.start_time + self._hdf5["trialOptoDur"][self.trial_index]
        assert self._stim_recordings
        return np.array(
            [
                rec.offset_time_on_sync if rec else np.nan
                for rec in self._stim_recordings
            ]
        )

    @functools.cached_property
    def _bregma_xy(self) -> tuple[tuple[np.float64, np.float64], ...]:
        calibration_data = dict(self._hdf5["bregmaGalvoCalibrationData"])
        for k in ("bregmaXOffset", "bregmaYOffset"):
            calibration_data[k] = calibration_data[k][()]
        return tuple(
            DynamicRoutingTask.TaskUtils.galvoToBregma(
                calibration_data,
                *voltages,
            )
            for voltages in self._hdf5["trialGalvoVoltage"][self.trial_index]
        )

    @functools.cached_property
    def bregma_x(self) -> npt.NDArray[np.float64]:
        return np.array([bregma[0] for bregma in self._bregma_xy])

    @functools.cached_property
    def bregma_y(self) -> npt.NDArray[np.float64]:
        return np.array([bregma[1] for bregma in self._bregma_xy])

    @functools.cached_property
    def location(self) -> npt.NDArray[np.str_]:
        return np.array(
            self._hdf5["trialOptoLabel"].asstr()[self.trial_index], dtype=str
        )

    @functools.cached_property
    def power(self) -> npt.NDArray[np.float64]:
        calibration_data = self._hdf5["optoPowerCalibrationData"]
        trial_voltages = self._hdf5["trialOptoVoltage"][self.trial_index]
        if "poly coefficients" in calibration_data:
            return DynamicRoutingTask.TaskUtils.voltsToPower(
                calibration_data,
                trial_voltages,
            )
        return np.where(~np.isnan(trial_voltages), self._hdf5["optoPower"], np.nan)
        # return trial_voltages * calibration_data['slope'] + calibration_data['intercept']


if __name__ == "__main__":
    import doctest

    import dotenv

    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
