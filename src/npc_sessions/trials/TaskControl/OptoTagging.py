"""
getting Optotagging trials table requires:
- one or more hdf5 files with trial/stim data, called 'OptoTagging_*.hdf5'
- frame display times from sync, to use in place of frametimes in hdf5 file
- latency estimate for each stim presentation, to be added to frame
display times to get stim onset times
"""

from __future__ import annotations

import datetime
from collections.abc import Iterable

import DynamicRoutingTask.TaskUtils
import npc_io
import npc_samstim
import npc_session
import npc_stim
import npc_sync
import numpy as np
import numpy.typing as npt

from npc_sessions.trials.TaskControl import TaskControl


class OptoTagging(TaskControl):
    """
    >>> stim = 's3://aind-ephys-data/ecephys_662892_2023-08-21_12-43-45/behavior/OptoTagging_662892_20230821_125915.hdf5'
    >>> sync = 's3://aind-ephys-data/ecephys_662892_2023-08-21_12-43-45/behavior/20230821T124345.h5'
    >>> trials = OptoTagging(stim, sync)
    >>> assert not trials.to_dataframe().empty
    """

    def __init__(
        self,
        hdf5: npc_stim.StimPathOrDataset,
        sync: npc_sync.SyncPathOrDataset,
        ephys_recording_dirs: Iterable[npc_io.PathLike] | None = None,
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

    @npc_io.cached_property
    def _datetime(self) -> datetime.datetime:
        return npc_session.DatetimeRecord(self._hdf5["startTime"].asstr()[()]).dt

    @npc_io.cached_property
    def _device(self) -> str:
        """If multiple devices were used, this should be ignored."""
        if (device := self._hdf5.get("optoDev")) is None:
            # older sessions may lack info:
            assert self._datetime.date() < datetime.date(
                2024, 5, 13
            )  # week of first optotagging with 633 laser
            return "laser_488"
        device = device.asstr()[()]
        return device

    def get_trial_opto_device(self, trial_idx: int) -> str:
        """Currently only one device used, but this method is prepared for multiple devices in the future."""
        if (devices := self._hdf5.get("trialOptoDevice")) is None or devices.size == 0:
            assert self._datetime.date() < datetime.date(
                2023, 8, 1
            )  # older sessions may lack info
            return "laser_488"
        devices = devices.asstr()[trial_idx]
        if devices[0] + devices[-1] != "[]":
            # basic check before we eval code from the web
            raise ValueError(f"Invalid opto devices string: {devices}")
        devices = eval(devices)
        assert len(devices) == 1, f"Expected one opto device, got {len(devices)}"
        return devices[0]

    def assert_is_single_device(self) -> None:
        assert (
            self._hdf5.get("trialOptoDevice") is None
        ), f"Multiple optotagging devices found for {self._datetime} session - update `get_trial_opto_device` method to handle multiple devices"

    @npc_io.cached_property
    def _trial_opto_device(self) -> tuple[str, ...]:
        ## for multiple devices:
        # return tuple(self.get_trial_opto_device(idx) for idx in range(self._len))
        self.assert_is_single_device()
        ## for single device:
        return (self._device,) * self._len

    def get_stim_recordings_from_sync(
        self, line_label: str = "laser_488"
    ) -> tuple[npc_samstim.StimRecording, ...] | None:
        try:
            recordings = npc_samstim.get_stim_latencies_from_sync(
                self._hdf5,
                self._sync,
                waveform_type="opto",
                line_index_or_label=npc_sync.get_sync_line_for_stim_onset(line_label),
            )
        except IndexError:
            return None
        assert (
            None not in recordings
        ), f"{recordings.count(None) = } encountered: expected a recording of stim onset for every trial"
        # TODO check this works for all older sessions
        return tuple(_ for _ in recordings if _ is not None)

    @npc_io.cached_property
    def _stim_recordings_488(self) -> tuple[npc_samstim.StimRecording, ...] | None:
        return self.get_stim_recordings_from_sync("laser_488")

    @npc_io.cached_property
    def _stim_recordings_633(self) -> tuple[npc_samstim.StimRecording, ...] | None:
        return self.get_stim_recordings_from_sync("laser_633")

    @npc_io.cached_property
    def _stim_recordings(self) -> tuple[npc_samstim.StimRecording, ...]:
        rec_488 = self._stim_recordings_488 or ()
        rec_633 = self._stim_recordings_633 or ()
        rec = []
        for i in range(self._len):
            device = self._trial_opto_device[i]
            if "633" in device:
                rec.append(rec_633[i])
            elif "488" in device:
                rec.append(rec_488[i])
            else:
                raise NotImplementedError(f"Unexpected opto device: {device}")
        return tuple(rec)

    @npc_io.cached_property
    def _len(self) -> int:
        """Number of trials"""
        return len(self.trial_index)

    @npc_io.cached_property
    def trial_index(self) -> npt.NDArray[np.int32]:
        """0-indexed"""
        return np.arange(len(self._hdf5["trialOptoOnsetFrame"]))

    @npc_io.cached_property
    def _inter_trial_interval(self) -> float:
        return self._hdf5["optoInterval"][()] / self._hdf5["frameRate"][()]

    @npc_io.cached_property
    def start_time(self) -> npt.NDArray[np.float64]:
        return np.array([rec.onset_time_on_sync for rec in self._stim_recordings])[
            self.trial_index
        ]

    @npc_io.cached_property
    def stop_time(self) -> npt.NDArray[np.float64]:
        return np.array([rec.offset_time_on_sync for rec in self._stim_recordings])[
            self.trial_index
        ]

    @npc_io.cached_property
    def stim_name(self) -> npt.NDArray[np.str_]:
        return np.array([recording.name for recording in self._stim_recordings])[
            self.trial_index
        ]

    @npc_io.cached_property
    def duration(self) -> npt.NDArray[np.float64]:
        return self._hdf5["trialOptoDur"][self.trial_index]

    @npc_io.cached_property
    def location(self) -> npt.NDArray[np.str_]:
        if all(str(v).upper() in "ABCDEF" for v in self._location):
            return np.array(
                [f"probe{str(v).upper()}" for v in self._location], dtype=str
            )
        return self._location

    @npc_io.cached_property
    def _bregma_xy(self) -> tuple[tuple[np.float64, np.float64], ...]:
        bregma = self._hdf5.get("optoBregma", None) or self._hdf5.get("bregmaXY", None)
        galvo = self._hdf5["galvoVoltage"][()]
        trial_voltages = self._hdf5["trialGalvoVoltage"]
        return tuple(
            tuple(bregma[np.all(galvo == v, axis=1)][0]) for v in trial_voltages
        )

    @npc_io.cached_property
    def bregma_x(self) -> npt.NDArray[np.float64]:
        return np.array([bregma[0] for bregma in self._bregma_xy])[self.trial_index]

    @npc_io.cached_property
    def bregma_y(self) -> npt.NDArray[np.float64]:
        return np.array([bregma[1] for bregma in self._bregma_xy])[self.trial_index]

    @npc_io.cached_property
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

    @npc_io.cached_property
    def power(self) -> npt.NDArray[np.float64]:
        calibration_data = self._hdf5["optoPowerCalibrationData"]
        trial_voltages = self._hdf5["trialOptoVoltage"][self.trial_index]
        if "poly coefficients" in calibration_data:
            power = DynamicRoutingTask.TaskUtils.voltsToPower(
                calibration_data,
                trial_voltages,
            )
        else:
            power = np.where(~np.isnan(trial_voltages), self._hdf5["optoPower"], np.nan)
        # round power to 3 decimal places, if safe to do so:
        if np.max(np.abs(np.round(power, 3) - power)) < 1e-3:
            power = np.round(power, 3)
        return power
        # return trial_voltages * calibration_data['slope'] + calibration_data['intercept']

    @npc_io.cached_property
    def wavelength(self) -> npt.NDArray[np.int64]:
        def parse_wavelength(device: str) -> int:
            try:
                value = int(device.split("_")[-1])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid opto device string (expected 'laser_488' format): {device}"
                ) from exc
            else:
                assert (
                    300 < value < 1000
                ), f"Unexpected wavelength parsed from `trialOptoDevice`: {value}"
                return value

        return np.array(
            [parse_wavelength(device) for device in self._trial_opto_device]
        )


if __name__ == "__main__":
    import doctest

    import dotenv

    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
