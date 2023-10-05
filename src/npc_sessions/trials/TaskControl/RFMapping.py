"""
getting Optotagging trials table requires:
- one or more hdf5 files with trial/stim data, called 'OptoTagging_*.hdf5'
- frame display times from sync, to use in place of frametimes in hdf5 file
- latency estimate for each stim presentation, to be added to frame
display times to get stim onset times

>>> stim = utils.get_h5_stim_data('s3://aind-ephys-data/ecephys_662892_2023-08-21_12-43-45/behavior/RFMapping_662892_20230821_124434.hdf5')
>>> sync = utils.get_sync_data('s3://aind-ephys-data/ecephys_662892_2023-08-21_12-43-45/behavior/20230821T124345.h5')
>>> vis_mapping_trials = VisRFMapping(stim, sync)
>>> aud_mapping_trials = AudRFMapping(stim, sync)
>>> assert not vis_mapping_trials._df.is_empty()
>>> assert not aud_mapping_trials._df.is_empty()
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import numpy.typing as npt

import npc_sessions.utils as utils
from npc_sessions.trials.TaskControl import TaskControl


class RFMapping(TaskControl):
    _aud_stim_onset_times: npt.NDArray[np.float64]
    """`[1 x num trials]` onset time of each aud stim relative to start of
    sync. Where values are nan, onset times will be taken from display times
    (ie. vis stim assumed)."""
    _aud_stim_offset_times: npt.NDArray[np.float64]

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

    @property
    def _aud_stim_recordings(self) -> tuple[utils.StimRecording | None, ...] | None:
        self._cached_aud_stim_recordings: tuple[utils.StimRecording | None, ...] | None
        cached = getattr(self, "_cached_aud_stim_recordings", None)
        if cached is not None:
            return cached
        if (
            self._sync
            and self._sync.start_time.date() >= utils.FIRST_SOUND_ON_SYNC_DATE
        ):
            self._cached_aud_stim_recordings = utils.get_stim_latencies_from_sync(
                self._hdf5,
                self._sync,
                waveform_type="sound",
            )
        elif (
            recording_dirs := getattr(self, "_ephys_recording_dirs", None)
        ) is not None:
            assert recording_dirs is not None
            self._cached_aud_stim_recordings = (
                utils.get_stim_latencies_from_nidaq_recording(
                    self._hdf5,
                    sync=self._sync,
                    recording_dirs=recording_dirs,
                    waveform_type="sound",
                )
            )
        else:
            self._cached_aud_stim_recordings = None
        return self._cached_aud_stim_recordings

    @_aud_stim_recordings.setter
    def _aud_stim_recordings(self, value: Iterable[utils.StimRecording | None]) -> None:
        """Can be set on init by passing as kwarg"""
        self._set_aud_stim_recordings = tuple(value)

    def get_trial_aud_onset(
        self, trial: int | npt.NDArray[np.int32]
    ) -> npt.NDArray[np.float64]:
        if self._aud_stim_recordings is not None:
            return np.array(
                [
                    np.nan if rec is None else rec.onset_time_on_sync
                    for rec in self._aud_stim_recordings
                ]
            )[trial]
        if not self._sync or not getattr(self, "_aud_stim_onset_times", None):
            return utils.safe_index(
                self._flip_times, self._hdf5["stimStartFrame"][trial]
            )
        return utils.safe_index(self._aud_stim_onset_times, trial)

    def get_trial_aud_offset(
        self, trial: int | npt.NDArray[np.int32]
    ) -> npt.NDArray[np.float64]:
        if self._aud_stim_recordings is not None:
            return np.array(
                [
                    np.nan if rec is None else rec.offset_time_on_sync
                    for rec in self._aud_stim_recordings
                ]
            )[trial]
        if not self._sync or not getattr(self, "_aud_stim_offset_times", None):
            return self.get_trial_aud_onset(trial) + self._hdf5["stimFrames"][()]
        return utils.safe_index(self._aud_stim_offset_times, trial)

    @utils.cached_property
    def _len_all_trials(self) -> int:
        return len(self._hdf5["stimStartFrame"][()])

    def find(self, key: str) -> npt.NDArray[np.bool_] | None:
        if key in self._hdf5:
            return ~np.isnan(self._hdf5[key][()])
        return None

    @utils.cached_property
    def _idx(self) -> npt.NDArray[np.int32]:
        """Used for extracting a subset of inds throughout all properties.
        Must be constructed from private properties directly from the hdf5 file"""
        return np.arange(self._len_all_trials)

    @utils.cached_property
    def _all_aud_freq(self) -> npt.NDArray[np.float64]:
        # don't use self._idx here
        freq = np.full(self._len_all_trials, np.nan)
        for key in ("trialToneFreq", "trialSoundFreq", "trialAMNoiseFreq"):
            if key in self._hdf5:
                array = self._hdf5[key][()]
                freq[~np.isnan(array)] = array[~np.isnan(array)]
        return freq

    @utils.cached_property
    def _all_aud_idx(self) -> npt.NDArray[np.float64]:
        # don't use self._idx here
        return np.where(
            ~np.isnan(self._all_aud_freq), np.arange(self._len_all_trials), np.nan
        )

    @utils.cached_property
    def _all_vis_idx(self) -> npt.NDArray[np.float64]:
        # don't use self._idx here
        flashes = self.find("trialFullFieldContrast")
        if flashes is None:
            flashes = np.full(self._len_all_trials, False)
        gratings = self.find("trialGratingOri")
        if gratings is None:
            gratings = np.full(self._len_all_trials, False)
        return np.where(gratings ^ flashes, np.arange(self._len_all_trials), np.nan)

    @utils.cached_property
    def start_time(self) -> npt.NDArray[np.float64]:
        """falling edge of first vsync in each trial"""
        return utils.safe_index(
            self._flip_times, self._hdf5["stimStartFrame"][self._idx]
        )

    @utils.cached_property
    def stim_start_time(self) -> npt.NDArray[np.float64]:
        """onset of RF mapping stimulus"""
        return np.where(
            self._is_vis_stim,
            utils.safe_index(
                self._vis_display_times, self._hdf5["stimStartFrame"][self._idx]
            ),
            self.get_trial_aud_onset(self._idx),
        )

    @utils.cached_property
    def stim_stop_time(self) -> npt.NDArray[np.float64]:
        """offset of RF mapping stimulus"""
        frames_per_stim = self._hdf5["stimFrames"][()]
        return np.where(
            self._is_vis_stim,
            utils.safe_index(
                self._vis_display_times,
                self._hdf5["stimStartFrame"][self._idx] + frames_per_stim,
            ),
            self.get_trial_aud_offset(self._idx),
        )

    @utils.cached_property
    def stop_time(self) -> npt.NDArray[np.float64]:
        """falling edge of vsync after stimulus end + inter-stim frames"""
        return np.max(
            [
                utils.safe_index(
                    self._flip_times,
                    self._hdf5["stimStartFrame"][self._idx]
                    + self._hdf5["stimFrames"][()]
                    + self._hdf5["interStimFrames"][()],
                ),
                self.stim_stop_time + self._hdf5["interStimFrames"][()],
            ],
            axis=0,
        )

    @utils.cached_property
    def trial_index(self) -> npt.NDArray[np.int32]:
        return np.arange(self._len_all_trials)[self._idx]

    @utils.cached_property
    def _len(self) -> int:
        return len(self._idx)

    @utils.cached_property
    def _tone_freq(self) -> npt.NDArray[np.float64]:
        for key in ("trialToneFreq", "trialSoundFreq"):
            if key in self._hdf5:
                return self._hdf5[key][self._idx]
        return np.full(self._len, np.nan)

    @utils.cached_property
    def _AM_noise_freq(self) -> npt.NDArray[np.float64]:
        return (
            self._hdf5["trialAMNoiseFreq"][self._idx]
            if "trialAMNoiseFreq" in self._hdf5
            else np.full(self._len, np.nan)
        )

    @utils.cached_property
    def _is_aud_stim(self) -> npt.NDArray[np.bool_]:
        """Includes AM noise and pure tones"""
        return np.where(np.isnan(self._all_aud_idx[self._idx]), False, True)

    @utils.cached_property
    def _is_vis_stim(self) -> npt.NDArray[np.bool_]:
        return np.where(np.isnan(self._all_vis_idx[self._idx]), False, True)

    @utils.cached_property
    def _full_field_contrast(self) -> npt.NDArray[np.float64]:
        return (
            self._hdf5["trialFullFieldContrast"][self._idx]
            if "trialFullFieldContrast" in self._hdf5
            else np.full(self._len, np.nan)
        )


class VisRFMapping(RFMapping):
    @utils.cached_property
    def _idx(self) -> npt.NDArray[np.int32]:
        """Used for extracting a subset of inds throughout all properties.
        Must be constructed from private properties directly from the hdf5 file"""
        return np.array(self._all_vis_idx[~np.isnan(self._all_vis_idx)], dtype=np.int32)

    @utils.cached_property
    def is_small_field_grating(self) -> npt.NDArray[np.bool_]:
        return ~np.isnan(self.grating_orientation)

    @utils.cached_property
    def grating_orientation(self) -> npt.NDArray[np.float64]:
        return self._hdf5["trialGratingOri"][self._idx]

    @utils.cached_property
    def grating_x(self) -> npt.NDArray[np.float64]:
        """position of grating patch center, in pixels from screen center"""
        return np.array([xy[0] for xy in self._hdf5["trialVisXY"][self._idx]])

    @utils.cached_property
    def grating_y(self) -> npt.NDArray[np.float64]:
        """position of grating patch center, in pixels from screen center"""
        return np.array([xy[1] for xy in self._hdf5["trialVisXY"][self._idx]])

    @utils.cached_property
    def is_full_field_flash(self) -> npt.NDArray[np.bool_]:
        return ~np.isnan(self._full_field_contrast)

    @utils.cached_property
    def flash_contrast(self) -> npt.NDArray[np.float64]:
        return self._full_field_contrast


class AudRFMapping(RFMapping):
    @utils.cached_property
    def _idx(self) -> npt.NDArray[np.int32]:
        """Used for extracting a subset of inds throughout all properties.
        Must be constructed from private properties directly from the hdf5 file"""
        return np.array(self._all_aud_idx[~np.isnan(self._all_aud_idx)], dtype=np.int32)

    @utils.cached_property
    def is_AM_noise(self) -> npt.NDArray[np.bool_]:
        return ~np.isnan(self._AM_noise_freq)

    @utils.cached_property
    def is_pure_tone(self) -> npt.NDArray[np.bool_]:
        return ~np.isnan(self._tone_freq)

    @utils.cached_property
    def freq(self) -> npt.NDArray[np.float64]:
        """frequency of pure tone or frequency of modulation for AM noise, in Hz"""
        return self._all_aud_freq[self._idx]


if __name__ == "__main__":
    import doctest

    import dotenv

    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
