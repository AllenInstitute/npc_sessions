"""
getting Optotagging trials table requires:
- one or more hdf5 files with trial/stim data, called 'OptoTagging_*.hdf5'
- frame display times from sync, to use in place of frametimes in hdf5 file
- latency estimate for each stim presentation, to be added to frame
display times to get stim onset times
"""
import contextlib
import functools

import DynamicRoutingTask.OptoParams as OptoParams
import numpy as np
import numpy.typing as npt

import npc_sessions.utils as utils
from npc_sessions.trials.TaskControl import TaskControl


class OptoTagging(TaskControl):
    """
    >>> stim = 's3://aind-ephys-data/ecephys_662892_2023-08-21_12-43-45/behavior/OptoTagging_662892_20230821_125915.hdf5'
    >>> sync = 's3://aind-ephys-data/ecephys_662892_2023-08-21_12-43-45/behavior/20230821T124345.h5'
    >>> trials = OptoTagging(stim, sync)
    """

    _stim_onset_times: npt.NDArray[np.float64]
    """`[1 x num trials]` onset time of each opto stim relative to start of
    sync. There should be no nans: the times will be used as trial start_time."""
    _stim_offset_times: npt.NDArray[np.float64]

    @functools.cached_property
    def start_time(self) -> npt.NDArray[np.float64]:
        with contextlib.suppress(AttributeError):
            return self._stim_onset_times
        return utils.safe_index(self._frame_times, self._hdf5["trialOptoOnsetFrame"][:])

    @functools.cached_property
    def stop_time(self) -> npt.NDArray[np.float64]:
        with contextlib.suppress(AttributeError):
            return self._stim_offset_times
        return self.start_time + self._hdf5["trialOptoDur"][:]

    @functools.cached_property
    def trial_index(self) -> npt.NDArray[np.int32]:
        """0-indexed"""
        return np.arange(len(self.start_time))

    @functools.cached_property
    def _bregma(self) -> tuple[tuple[np.float64, np.float64], ...]:
        calibration_data = dict(self._hdf5["bregmaGalvoCalibrationData"])
        for k in ("bregmaXOffset", "bregmaYOffset"):
            calibration_data[k] = calibration_data[k][()]
        return tuple(
            OptoParams.galvoToBregma(
                calibration_data,
                *voltages,
            )
            for voltages in self._hdf5["trialGalvoVoltage"][:]
        )

    @functools.cached_property
    def bregma_x(self) -> npt.NDArray[np.float64]:
        return np.array([bregma[0] for bregma in self._bregma])

    @functools.cached_property
    def bregma_y(self) -> npt.NDArray[np.float64]:
        return np.array([bregma[1] for bregma in self._bregma])

    @functools.cached_property
    def location(self) -> npt.NDArray[np.str_]:
        return self._hdf5["trialOptoLabel"].asstr()[:]

    @functools.cached_property
    def power(self) -> npt.NDArray[np.float64]:
        return OptoParams.voltsToPower(
            self._hdf5["optoPowerCalibrationData"], self._hdf5["trialOptoVoltage"][:]
        )


if __name__ == "__main__":
    import doctest

    import dotenv

    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
