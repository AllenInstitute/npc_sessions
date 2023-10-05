"""
getting Optotagging trials table requires:
- one or more hdf5 files with trial/stim data, called 'OptoTagging_*.hdf5'
- frame display times from sync, to use in place of frametimes in hdf5 file
- latency estimate for each stim presentation, to be added to frame
display times to get stim onset times
"""
from __future__ import annotations

from collections.abc import Iterable

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
    >>> assert not trials._df.is_empty()
    """

    def __init__(
        self,
        hdf5: utils.StimPathOrDataset,
        sync: utils.SyncPathOrDataset,
        ephys_recording_dirs: Iterable[utils.PathLike] | None = None,
        **kwargs,
    ) -> None:
        if sync is None:
            raise ValueError(
                f"sync data is required for {self.__class__.__name__} trials table"
            )
        self._ephys_recording_dirs = ephys_recording_dirs
        super().__init__(
            hdf5, sync, ephys_recording_dirs=ephys_recording_dirs, **kwargs
        )

    @utils.cached_property
    def _stim_recordings(self) -> tuple[utils.StimRecording, ...]:
        recordings = utils.get_stim_latencies_from_sync(
            self._hdf5, self._sync, waveform_type="opto"
        )
        assert (
            None not in recordings
        ), f"{recordings.count(None) = } encountered: expected a recording of stim onset for every trial"
        return tuple(_ for _ in recordings if _ is not None)  # for mypy
        # TODO check this works for all older sessions

    @utils.cached_property
    def trial_index(self) -> npt.NDArray[np.int32]:
        """0-indexed"""
        return np.arange(len(self._hdf5["trialOptoOnsetFrame"]))

    @utils.cached_property
    def _inter_trial_interval(self) -> float:
        return self._hdf5["optoInterval"][()] / self._hdf5["frameRate"][()]

    @utils.cached_property
    def start_time(self) -> npt.NDArray[np.float64]:
        return self.stim_start_time - min(0.2, 0.5 * self._inter_trial_interval)

    @utils.cached_property
    def stop_time(self) -> npt.NDArray[np.float64]:
        return self.stim_stop_time + min(0.2, 0.5 * self._inter_trial_interval)

    @utils.cached_property
    def stim_start_time(self) -> npt.NDArray[np.float64]:
        return np.array([rec.onset_time_on_sync for rec in self._stim_recordings])[
            self.trial_index
        ]

    @utils.cached_property
    def stim_stop_time(self) -> npt.NDArray[np.float64]:
        return np.array([rec.offset_time_on_sync for rec in self._stim_recordings])[
            self.trial_index
        ]

    @utils.cached_property
    def stim_name(self) -> npt.NDArray[np.str_]:
        return np.array([recording.name for recording in self._stim_recordings])[
            self.trial_index
        ]

    @utils.cached_property
    def duration(self) -> npt.NDArray[np.float64]:
        return self._hdf5["trialOptoDur"][self.trial_index]

    @utils.cached_property
    def location(self) -> npt.NDArray[np.str_]:
        if all(str(v).upper() in "ABCDEF" for v in self._location):
            return np.array(
                [f"probe{str(v).upper()}" for v in self._location], dtype=str
            )
        return self._location

    @utils.cached_property
    def _bregma_xy(self) -> tuple[tuple[np.float64, np.float64], ...]:
        bregma = self._hdf5.get("optoBregma", None) or self._hdf5.get("bregmaXY", None)
        galvo = self._hdf5["galvoVoltage"][()]
        trial_voltages = self._hdf5["trialGalvoVoltage"]
        return tuple(tuple(bregma[np.all(galvo == v, axis=1)][0]) for v in trial_voltages)  # type: ignore

    @utils.cached_property
    def bregma_x(self) -> npt.NDArray[np.float64]:
        return np.array([bregma[0] for bregma in self._bregma_xy])[self.trial_index]

    @utils.cached_property
    def bregma_y(self) -> npt.NDArray[np.float64]:
        return np.array([bregma[1] for bregma in self._bregma_xy])[self.trial_index]

    @utils.cached_property
    def _location(self) -> npt.NDArray[np.str_]:
        if trialOptoLabel := self._hdf5.get("trialOptoLabel", None):
            return np.array(trialOptoLabel.asstr()[self.trial_index], dtype=str)
        if optoTaggingLocs := self._hdf5.get("optoTaggingLocs"):
            label = optoTaggingLocs["label"].asstr()[()]
            xy = np.array(list(zip(optoTaggingLocs["X"], optoTaggingLocs["Y"])))
            return np.array(
                [label[np.all(xy == v, axis=1)][0] for v in self._bregma_xy], dtype=str
            )[self.trial_index]
        raise ValueError("No known optotagging location data found")

    @utils.cached_property
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
