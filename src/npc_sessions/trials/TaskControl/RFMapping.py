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
>>> assert dict(vis_mapping_trials)
>>> assert dict(aud_mapping_trials)
"""
import functools

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
        sync: utils.SyncPathOrDataset,  # not optional
        **kwargs,
    ) -> None:
        super().__init__(hdf5, sync, **kwargs)

    def get_trial_aud_onset(
        self, trial: int | npt.NDArray[np.int32]
    ) -> npt.NDArray[np.float64]:
        return self.get_vis_display_time(self._hdf5["stimStartFrame"][trial])
        # TODO remove when accurate timing available
        return utils.safe_index(self._aud_stim_onset_times, trial)

    def get_trial_aud_offset(
        self, trial: int | npt.NDArray[np.int32]
    ) -> npt.NDArray[np.float64]:
        return self.get_vis_display_time(
            self._hdf5["stimStartFrame"][trial] + self._hdf5["stimFrames"][()]
        )
        # TODO remove when accurate timing available
        return utils.safe_index(self._aud_stim_offset_times, trial)

    @functools.cached_property
    def _len_all_trials(self) -> int:
        return len(self._hdf5["stimStartFrame"][()])

    def find(self, key: str) -> npt.NDArray[np.bool_] | None:
        if key in self._hdf5:
            return ~np.isnan(self._hdf5[key][()])
        return None

    @functools.cached_property
    def _idx(self) -> npt.NDArray[np.int32]:
        """Used for extracting a subset of inds throughout all properties.
        Must be constructed from private properties directly from the hdf5 file"""
        return np.arange(self._len_all_trials)

    @functools.cached_property
    def _all_aud_freq(self) -> npt.NDArray[np.float64]:
        # don't use self._idx here
        freq = np.full(self._len_all_trials, np.nan)
        for key in ("trialToneFreq", "trialSoundFreq", "trialAMNoiseFreq"):
            if key in self._hdf5:
                array = self._hdf5[key][()]
                freq[~np.isnan(array)] = array[~np.isnan(array)]
        return freq

    @functools.cached_property
    def _all_aud_idx(self) -> npt.NDArray[np.float64]:
        # don't use self._idx here
        return np.where(
            ~np.isnan(self._all_aud_freq), np.arange(self._len_all_trials), np.nan
        )

    @functools.cached_property
    def _all_vis_idx(self) -> npt.NDArray[np.float64]:
        # don't use self._idx here
        flashes = self.find("trialFullFieldContrast")
        if flashes is None:
            flashes = np.full(self._len_all_trials, False)
        gratings = self.find("trialGratingOri")
        if gratings is None:
            gratings = np.full(self._len_all_trials, False)
        return np.where(gratings ^ flashes, np.arange(self._len_all_trials), np.nan)

    @functools.cached_property
    def start_time(self) -> npt.NDArray[np.float64]:
        """Falling edge of first vsync in each trial"""
        return self.get_script_frame_time(self._hdf5["stimStartFrame"][self._idx])

    @functools.cached_property
    def stim_start_time(self) -> npt.NDArray[np.float64]:
        """Onset of RF mapping stimulus"""
        return np.where(
            self._is_vis_stim,
            self.get_vis_display_time(self._hdf5["stimStartFrame"][self._idx]),
            self.get_trial_aud_onset(self._idx),
        )

    @functools.cached_property
    def stim_stop_time(self) -> npt.NDArray[np.float64]:
        """offset of RF mapping stimulus"""
        frames_per_stim = self._hdf5["stimFrames"][()]
        return np.where(
            self._is_vis_stim,
            self.get_vis_display_time(
                self._hdf5["stimStartFrame"][self._idx] + frames_per_stim
            ),
            self.get_trial_aud_offset(self._idx),
        )

    @functools.cached_property
    def stop_time(self) -> npt.NDArray[np.float64]:
        """Falling edge of last vsync, after inter-stim frames"""
        return self.get_vis_display_time(
            self._hdf5["stimStartFrame"][self._idx]
            + self._hdf5["stimFrames"][()]
            + self._hdf5["interStimFrames"][()]
        )

    @functools.cached_property
    def index(self) -> npt.NDArray[np.int32]:
        return np.arange(self._len_all_trials)[self._idx]

    @functools.cached_property
    def _len(self) -> int:
        return len(self._idx)

    @functools.cached_property
    def _tone_freq(self) -> npt.NDArray[np.float64]:
        for key in ("trialToneFreq", "trialSoundFreq"):
            if key in self._hdf5:
                return self._hdf5[key][self._idx]
        return np.full(self._len, np.nan)

    @functools.cached_property
    def _AM_noise_freq(self) -> npt.NDArray[np.float64]:
        return (
            self._hdf5["trialAMNoiseFreq"][self._idx]
            if "trialAMNoiseFreq" in self._hdf5
            else np.full(self._len, np.nan)
        )

    @functools.cached_property
    def _is_aud_stim(self) -> npt.NDArray[np.bool_]:
        """Includes AM noise and pure tones"""
        return np.where(np.isnan(self._all_aud_idx[self._idx]), False, True)

    @functools.cached_property
    def _is_vis_stim(self) -> npt.NDArray[np.bool_]:
        return np.where(np.isnan(self._all_vis_idx[self._idx]), False, True)

    @functools.cached_property
    def _full_field_contrast(self) -> npt.NDArray[np.float64]:
        return (
            self._hdf5["trialFullFieldContrast"][self._idx]
            if "trialFullFieldContrast" in self._hdf5
            else np.full(self._len, np.nan)
        )


class VisRFMapping(RFMapping):
    @functools.cached_property
    def _idx(self) -> npt.NDArray[np.int32]:
        """Used for extracting a subset of inds throughout all properties.
        Must be constructed from private properties directly from the hdf5 file"""
        return np.array(self._all_vis_idx[~np.isnan(self._all_vis_idx)], dtype=np.int32)

    @functools.cached_property
    def is_small_field_grating(self) -> npt.NDArray[np.bool_]:
        return ~np.isnan(self.grating_orientation)

    @functools.cached_property
    def grating_orientation(self) -> npt.NDArray[np.float64]:
        return self._hdf5["trialGratingOri"][self._idx]

    @functools.cached_property
    def grating_x(self) -> npt.NDArray[np.float64]:
        """position of grating patch center, in pixels from screen center"""
        return np.array([xy[0] for xy in self._hdf5["trialVisXY"][self._idx]])

    @functools.cached_property
    def grating_y(self) -> npt.NDArray[np.float64]:
        """position of grating patch center, in pixels from screen center"""
        return np.array([xy[1] for xy in self._hdf5["trialVisXY"][self._idx]])

    @functools.cached_property
    def is_full_field_flash(self) -> npt.NDArray[np.bool_]:
        return ~np.isnan(self._full_field_contrast)

    @functools.cached_property
    def flash_contrast(self) -> npt.NDArray[np.float64]:
        return self._full_field_contrast


class AudRFMapping(RFMapping):
    @functools.cached_property
    def _idx(self) -> npt.NDArray[np.int32]:
        """Used for extracting a subset of inds throughout all properties.
        Must be constructed from private properties directly from the hdf5 file"""
        return np.array(self._all_aud_idx[~np.isnan(self._all_aud_idx)], dtype=np.int32)

    @functools.cached_property
    def is_AM_noise(self) -> npt.NDArray[np.bool_]:
        return ~np.isnan(self._AM_noise_freq)

    @functools.cached_property
    def is_pure_tone(self) -> npt.NDArray[np.bool_]:
        return ~np.isnan(self._tone_freq)

    @functools.cached_property
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
