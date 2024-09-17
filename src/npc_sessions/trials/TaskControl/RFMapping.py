"""
getting Optotagging trials table requires:
- one or more hdf5 files with trial/stim data, called 'OptoTagging_*.hdf5'
- frame display times from sync, to use in place of frametimes in hdf5 file
- latency estimate for each stim presentation, to be added to frame
display times to get stim onset times

>>> stim = npc_stim.get_h5_stim_data('s3://aind-ephys-data/ecephys_662892_2023-08-21_12-43-45/behavior/RFMapping_662892_20230821_124434.hdf5')
>>> sync = npc_sync.get_sync_data('s3://aind-ephys-data/ecephys_662892_2023-08-21_12-43-45/behavior/20230821T124345.h5')
>>> vis_mapping_trials = VisRFMapping(stim, sync)
>>> aud_mapping_trials = AudRFMapping(stim, sync)
>>> assert not vis_mapping_trials.to_dataframe().empty
>>> assert not aud_mapping_trials.to_dataframe().empty
"""

from __future__ import annotations

from collections.abc import Iterable

import npc_io
import npc_samstim
import npc_stim
import npc_sync
import numpy as np
import numpy.typing as npt

from npc_sessions.trials.TaskControl import TaskControl


class RFMapping(TaskControl):
    _aud_stim_onset_times: npt.NDArray[np.float64]
    """`[1 x num trials]` onset time of each aud stim relative to start of
    sync. Where values are nan, onset times will be taken from display times
    (ie. vis stim assumed)."""
    _aud_stim_offset_times: npt.NDArray[np.float64]

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

    @property
    def _aud_stim_recordings(
        self,
    ) -> tuple[npc_samstim.StimRecording | None, ...] | None:
        self._cached_aud_stim_recordings: (
            tuple[npc_samstim.StimRecording | None, ...] | None
        )
        cached = getattr(self, "_cached_aud_stim_recordings", None)
        if cached is not None:
            return cached
        if (
            self._sync
            and self._sync.start_time.date() >= npc_sync.FIRST_SOUND_ON_SYNC_DATE
        ):
            self._cached_aud_stim_recordings = npc_samstim.get_stim_latencies_from_sync(
                self._hdf5,
                self._sync,
                waveform_type="sound",
            )
        elif (
            recording_dirs := getattr(self, "_ephys_recording_dirs", None)
        ) is not None:
            assert recording_dirs is not None
            self._cached_aud_stim_recordings = (
                npc_samstim.get_stim_latencies_from_nidaq_recording(
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
    def _aud_stim_recordings(
        self, value: Iterable[npc_samstim.StimRecording | None]
    ) -> None:
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
            return npc_stim.safe_index(
                self._flip_times, self._hdf5["stimStartFrame"][trial]
            )
        return npc_stim.safe_index(self._aud_stim_onset_times, trial)

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
        return npc_stim.safe_index(self._aud_stim_offset_times, trial)

    @npc_io.cached_property
    def _len_all_trials(self) -> int:
        return len(self._hdf5["stimStartFrame"][()])

    def find(self, key: str) -> npt.NDArray[np.bool_] | None:
        if key in self._hdf5:
            return ~np.isnan(self._hdf5[key][()])
        return None

    @npc_io.cached_property
    def _idx(self) -> npt.NDArray[np.int32]:
        """Used for extracting a subset of inds throughout all properties.
        Must be constructed from private properties directly from the hdf5 file"""
        return np.arange(self._len_all_trials)

    @npc_io.cached_property
    def _all_aud_freq(self) -> npt.NDArray[np.float64]:
        # don't use self._idx here
        freq = np.full(self._len_all_trials, np.nan)
        for key in (
            "trialToneFreq",
            "trialSoundFreq",
            "trialAMNoiseFreq",
            "trialNoiseFreq",
        ):
            if key in self._hdf5:
                array = self._hdf5[key][()]
                if key == "trialNoiseFreq":
                    idx = ~np.isnan(array).all(axis=1)
                    freq[idx] = np.nanmean(
                        array[idx], axis=1
                    )  #! used to determine if trial is sound, not as a meaningful parameter
                else:
                    freq[~np.isnan(array)] = array[~np.isnan(array)]
        return freq

    @npc_io.cached_property
    def _all_aud_idx(self) -> npt.NDArray[np.float64]:
        # don't use self._idx here
        return np.where(
            ~np.isnan(self._all_aud_freq), np.arange(self._len_all_trials), np.nan
        )

    @npc_io.cached_property
    def _all_vis_idx(self) -> npt.NDArray[np.float64]:
        # don't use self._idx here
        flashes = self.find("trialFullFieldContrast")
        if flashes is None:
            flashes = np.full(self._len_all_trials, False)
        gratings = self.find("trialGratingOri")
        if gratings is None:
            gratings = np.full(self._len_all_trials, False)
        return np.where(gratings ^ flashes, np.arange(self._len_all_trials), np.nan)

    @npc_io.cached_property
    def start_time(self) -> npt.NDArray[np.float64]:
        """falling edge of first vsync in each trial"""
        return npc_stim.safe_index(
            self._flip_times, self._hdf5["stimStartFrame"][self._idx]
        )

    @npc_io.cached_property
    def stim_start_time(self) -> npt.NDArray[np.float64]:
        """onset of RF mapping stimulus"""
        return np.where(
            self._is_vis_stim,
            npc_stim.safe_index(
                self._vis_display_times, self._hdf5["stimStartFrame"][self._idx]
            ),
            self.get_trial_aud_onset(self._idx),
        )

    @npc_io.cached_property
    def stim_stop_time(self) -> npt.NDArray[np.float64]:
        """offset of RF mapping stimulus"""
        frames_per_stim = self._hdf5["stimFrames"][()]
        if all(~self._is_vis_stim):
            return self.get_trial_aud_offset(self._idx)

        frame_idx = self._hdf5["stimStartFrame"][self._idx] + frames_per_stim
        if frame_idx[-1] > len(self._vis_display_times):
            # if we have a mix of vis and aud stim, and the last stim is aud, the
            # actual end time (as recorded on sync/ephys daq) can occur after the last
            # frame in the experiment, so `frame_idx[-1]` won't correspond to an actual
            # frame index. Just to be able to index without raising an error we'll
            # use real indices, but the stop time for the last aud stim will still come
            # from the recording
            vis_frame_idx_before_mod = frame_idx[self._is_vis_stim]
            frame_idx = (
                np.searchsorted(
                    np.arange(len(self._vis_display_times)), frame_idx, "right"
                )
                - 1
            )
            assert np.all(
                vis_frame_idx_before_mod == frame_idx[self._is_vis_stim]
            ), "frame_idx should onlu change for non-visual stims"
        return np.where(
            self._is_vis_stim,
            npc_stim.safe_index(
                self._vis_display_times,
                frame_idx,
            ),
            self.get_trial_aud_offset(self._idx),
        )

    @npc_io.cached_property
    def stop_time(self) -> npt.NDArray[np.float64]:
        """stimulus stop time + inter-stim frames"""
        try:
            return npc_stim.safe_index(
                self._flip_times,
                self._hdf5["stimStartFrame"][self._idx]
                + self._hdf5["stimFrames"][()]
                + self._hdf5["interStimFrames"][()],
            )
        except IndexError:
            return (
                self.stim_stop_time
                + self._hdf5["interStimFrames"][()] * self._hdf5["frameRate"][()]
            )

    @npc_io.cached_property
    def trial_index(self) -> npt.NDArray[np.int32]:
        return np.arange(self._len_all_trials)[self._idx]

    @npc_io.cached_property
    def _len(self) -> int:
        return len(self._idx)

    @npc_io.cached_property
    def _tone_freq(self) -> npt.NDArray[np.float64]:
        for key in ("trialToneFreq", "trialSoundFreq"):
            if key in self._hdf5:
                return self._hdf5[key][self._idx]
        return np.full(self._len, np.nan)

    @npc_io.cached_property
    def _AM_noise_freq(self) -> npt.NDArray[np.float64]:
        return (
            self._hdf5["trialAMNoiseFreq"][self._idx]
            if "trialAMNoiseFreq" in self._hdf5
            else np.full(self._len, np.nan)
        )

    @npc_io.cached_property
    def _white_noise_bandpass_freq(self) -> npt.NDArray[np.float64]:
        """2x trials array of low/high freq for bandpass filter"""
        return (
            self._hdf5["trialNoiseFreq"][self._idx]
            if "trialNoiseFreq" in self._hdf5
            else np.full(self._len, np.nan)
        )

    @npc_io.cached_property
    def _is_aud_stim(self) -> npt.NDArray[np.bool_]:
        """Includes AM noise, bandpass filtered white noise, and pure tones"""
        return np.where(np.isnan(self._all_aud_idx[self._idx]), False, True)

    @npc_io.cached_property
    def _is_vis_stim(self) -> npt.NDArray[np.bool_]:
        return np.where(np.isnan(self._all_vis_idx[self._idx]), False, True)

    @npc_io.cached_property
    def _full_field_contrast(self) -> npt.NDArray[np.float64]:
        return (
            self._hdf5["trialFullFieldContrast"][self._idx]
            if "trialFullFieldContrast" in self._hdf5
            else np.full(self._len, np.nan)
        )


class VisRFMapping(RFMapping):
    @npc_io.cached_property
    def _idx(self) -> npt.NDArray[np.int32]:
        """Used for extracting a subset of inds throughout all properties.
        Must be constructed from private properties directly from the hdf5 file"""
        return np.array(self._all_vis_idx[~np.isnan(self._all_vis_idx)], dtype=np.int32)

    @npc_io.cached_property
    def is_small_field_grating(self) -> npt.NDArray[np.bool_]:
        return ~np.isnan(self.grating_orientation)

    @npc_io.cached_property
    def grating_orientation(self) -> npt.NDArray[np.float64]:
        return self._hdf5["trialGratingOri"][self._idx]

    @npc_io.cached_property
    def grating_x(self) -> npt.NDArray[np.float64]:
        """position of grating patch center, in pixels from screen center"""
        return np.array([xy[0] for xy in self._hdf5["trialVisXY"][self._idx]])

    @npc_io.cached_property
    def grating_y(self) -> npt.NDArray[np.float64]:
        """position of grating patch center, in pixels from screen center"""
        return np.array([xy[1] for xy in self._hdf5["trialVisXY"][self._idx]])

    @npc_io.cached_property
    def is_full_field_flash(self) -> npt.NDArray[np.bool_]:
        return ~np.isnan(self._full_field_contrast)

    @npc_io.cached_property
    def flash_contrast(self) -> npt.NDArray[np.float64]:
        return self._full_field_contrast


class AudRFMapping(RFMapping):
    @npc_io.cached_property
    def _idx(self) -> npt.NDArray[np.int32]:
        """Used for extracting a subset of inds throughout all properties.
        Must be constructed from private properties directly from the hdf5 file"""
        return np.array(self._all_aud_idx[~np.isnan(self._all_aud_idx)], dtype=np.int32)

    @npc_io.cached_property
    def is_AM_noise(self) -> npt.NDArray[np.bool_]:
        return ~np.isnan(self._AM_noise_freq)

    @npc_io.cached_property
    def is_white_noise(self) -> npt.NDArray[np.bool_]:
        return np.array(
            [~np.isnan(freqs).any() for freqs in self._white_noise_bandpass_freq]
        )

    @npc_io.cached_property
    def is_pure_tone(self) -> npt.NDArray[np.bool_]:
        return ~np.isnan(self._tone_freq)

    @npc_io.cached_property
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
