"""
getting DR task trials table requires:
- hdf5 file with task data, called 'DynamicRouting1_*.hdf5'
- Sam's DynRoutData obj, to interpret the hdf5 file
- for visual stimuli and all non-stim events:
    frame display times from sync, to use in place of frametimes in hdf5 file
- for aud, opto, other non-visual stimulus presentations:
    latency estimates from NI-DAQ recordings, to be added to frame vsync times
"""

from __future__ import annotations

import copy
import datetime
import logging
from collections.abc import Iterable
from typing import Any, Literal

import DynamicRoutingTask.TaskUtils
import npc_io
import npc_lims
import npc_samstim
import npc_session
import npc_stim
import npc_sync
import numpy as np
import numpy.typing as npt
from DynamicRoutingTask.Analysis.DynamicRoutingAnalysisUtils import DynRoutData

from npc_sessions.trials.TaskControl import TaskControl

logger = logging.getLogger(__name__)


class DynamicRouting1(TaskControl):
    """All property getters without a leading underscore will be
    considered nwb trials columns. Their docstrings will become the column
    `description`.

    To add trials to a pynwb.NWBFile:

    >>> obj = DynamicRouting1("DRpilot_626791_20220817") # doctest: +SKIP

    >>> for column in obj.to_add_trial_column(): # doctest: +SKIP
    ...    nwb_file.add_trial_column(**column)

    >>> for trial in obj.to_add_trial(): # doctest: +SKIP
    ...    nwb_file.add_trial(**trial)

    >>> trials = DynamicRouting1('s3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15/behavior/DynamicRouting1_670248_20230803_123154.hdf5')
    >>> assert not trials.to_dataframe().empty
    """

    _num_opto_devices: int | None = None

    def __init__(
        self,
        hdf5: npc_stim.StimPathOrDataset,
        sync: npc_sync.SyncPathOrDataset | None = None,
        ephys_recording_dirs: Iterable[npc_io.PathLike] | None = None,
        **kwargs,
    ) -> None:
        if sync is None and ephys_recording_dirs is not None:
            raise ValueError(
                "ephys_recording_dirs was provided: must also provide sync file to get waveform timing from NI-DAQ recordings"
            )
        self._ephys_recording_dirs = ephys_recording_dirs
        super().__init__(
            hdf5, sync, ephys_recording_dirs=ephys_recording_dirs, **kwargs
        )

    def assert_single_opto_device(self) -> None:
        """A temporary measure to check before running code that assumes a single
        opto device"""
        if self._num_opto_devices is not None and self._num_opto_devices > 1:
            raise AssertionError(
                "Multiple opto devices used in session - the following section of code assumes only one was used."
            )

    @property
    def _opto_stim_recordings(
        self,
    ) -> tuple[npc_samstim.StimRecording | None, ...] | None:
        self._cached_opto_stim_recordings: (
            tuple[npc_samstim.StimRecording | None, ...] | None
        )
        cached = getattr(self, "_cached_opto_stim_recordings", None)
        if cached is not None:
            return cached
        if self._sync:
            try:
                self._cached_opto_stim_recordings = (
                    npc_samstim.get_stim_latencies_from_sync(
                        self._hdf5,
                        self._sync,
                        waveform_type="opto",
                    )
                )
            except npc_samstim.MissingSyncLineError:
                if self._ephys_recording_dirs:
                    self._cached_opto_stim_recordings = (
                        npc_samstim.get_stim_latencies_from_nidaq_recording(
                            self._hdf5,
                            sync=self._sync,
                            recording_dirs=self._ephys_recording_dirs,
                            waveform_type="opto",
                        )
                    )
                else:
                    logger.warning(
                        f"No opto stim sync line found and no ephys data provided: stim {npc_stim.get_stim_start_time(self._hdf5)}"
                    )
                    self._cached_opto_stim_recordings = None
        else:
            self._cached_opto_stim_recordings = None
        return self._cached_opto_stim_recordings

    @_opto_stim_recordings.setter
    def _opto_stim_recordings(
        self, value: Iterable[npc_samstim.StimRecording | None]
    ) -> None:
        """Can be set on init by passing as kwarg"""
        self._cached_opto_stim_recordings = tuple(value)

    @npc_io.cached_property
    def _opto_stim_waveforms(self) -> tuple[npc_samstim.Waveform | None, ...]:
        if not self._is_opto:
            return tuple(None for _ in range(self._len))
        return npc_samstim.get_opto_waveforms_from_stim_file(self._hdf5)

    @npc_io.cached_property
    def _unique_opto_waveforms(self) -> tuple[npc_samstim.Waveform, ...]:
        return tuple({w for w in self._opto_stim_waveforms if w is not None})

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

    _aud_stim_onset_times: npt.NDArray[np.float64]
    """`[1 x num trials]` onset time of each aud stim relative to start of
    sync. Where values are nan, onset times will be taken from display times
    (ie. vis stim assumed)."""
    _aud_stim_offset_times: npt.NDArray[np.float64]

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
            logger.debug("Using script frame times for opto stim onsets")
            return npc_stim.safe_index(
                self._flip_times, self._sam.stimStartFrame[trial]
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
            return self.get_trial_aud_onset(trial) + self._hdf5["trialSoundDur"][trial]
        return npc_stim.safe_index(self._aud_stim_offset_times, trial)

    def get_trial_opto_onset(
        self, line: Literal["laser_488", "laser_688", "galvo"] = "laser_488"
    ) -> npt.NDArray[np.float64]:
        self.assert_single_opto_device()
        if self._opto_stim_recordings is not None:
            if "488" not in line:
                logger.warning(
                    f"Interpreting analog waveforms for opto onset times is only implemented for a single 488 laser (early DR/Templeton experiments): arg {line =} will be ignored"
                )
            return np.array(
                [
                    np.nan if rec is None else rec.onset_time_on_sync
                    for rec in self._opto_stim_recordings
                ]
            )[: self._len]

        if (script_onset_frames := self._sam.trialOptoOnsetFrame).ndim == 2:
            script_onset_frames = script_onset_frames.squeeze()
        # note: this is different to OptoTagging, where onset frame is abs frame idx
        onset_times_based_on_script = npc_stim.safe_index(
            self._flip_times,
            self._sam.stimStartFrame + script_onset_frames[: self._len],
        )
        if not self._sync:
            logger.debug("Using script frame times for opto stim onsets")
            return onset_times_based_on_script
        try:
            line_number = npc_sync.get_sync_line_for_stim_onset(
                waveform_type=line, date=self._datetime
            )
        except (
            ValueError
        ):  # raised if requested sync line not available, based on date line was added
            logger.debug("Using script frame times for opto stim onsets")
            return onset_times_based_on_script
        assert self._sync is not None
        opto_line_rising_edges = self._sync.get_rising_edges(
            line_number, units="seconds"
        )
        if len(opto_line_rising_edges) < len(onset_times_based_on_script):
            logger.debug("Using script frame times for opto stim onsets")
            return onset_times_based_on_script
        onset_times_based_on_sync = opto_line_rising_edges[
            np.searchsorted(
                opto_line_rising_edges,
                onset_times_based_on_script,
                side="right",
            )
            - 1
        ]
        opto_onset_latency = onset_times_based_on_sync - onset_times_based_on_script
        logger.debug(
            f"Opto onset latencies (sync line - script flip): {np.median(opto_onset_latency)=} {np.max(opto_onset_latency)=}"
        )
        assert np.all(
            opto_onset_latency >= 0
        ), f"Negative opto latencies found (sync line - script flip): {opto_onset_latency = }"
        assert np.all(
            onset_times_based_on_sync < 0.1
        ), f"Large opto onset times found: {onset_times_based_on_sync = }"
        return onset_times_based_on_sync

    def get_trial_opto_offset(
        self, line: Literal["laser_488", "laser_688", "galvo"] = "laser_488"
    ) -> npt.NDArray[np.float64]:
        self.assert_single_opto_device()
        if self._opto_stim_recordings is not None:
            if "488" not in line:
                logger.warning(
                    f"Interpreting analog waveforms for opto offset times is only implemented for a single 488 laser (early DR/Templeton experiments): arg {line =} will be ignored"
                )
            return np.array(
                [
                    np.nan if rec is None else rec.offset_time_on_sync
                    for rec in self._opto_stim_recordings
                ]
            )[: self._len]
        self.assert_single_opto_device()
        offset_times_based_on_nominal_duration = (
            self.get_trial_opto_onset() + self.opto_duration[: self._len]
        )
        if not self._sync:
            return offset_times_based_on_nominal_duration
        try:
            line_number = npc_sync.get_sync_line_for_stim_onset(
                waveform_type=line, date=self._datetime
            )
        except (
            ValueError
        ):  # raised if requested sync line not available, based on date line was added
            logger.debug("Using script frame times for opto stim offsets")
            return offset_times_based_on_nominal_duration
        assert self._sync is not None
        opto_line_falling_edges = self._sync.get_falling_edges(
            line_number, units="seconds"
        )
        if len(opto_line_falling_edges) < len(offset_times_based_on_nominal_duration):
            logger.debug("Using script frame times for opto stim offsets")
            return offset_times_based_on_nominal_duration
        offset_times_based_on_sync = opto_line_falling_edges[
            np.searchsorted(
                opto_line_falling_edges,
                offset_times_based_on_nominal_duration,
                side="right",
            )
            - 1
        ]
        opto_duration = (
            self.get_trial_opto_onset(line=line) - offset_times_based_on_sync
        )
        assert np.all(
            opto_duration >= 0
        ), f"Negative opto durations found (sync offset - onset): {opto_duration = }"
        return offset_times_based_on_sync

    # ---------------------------------------------------------------------- #
    # helper-properties that won't become columns:

    @npc_io.cached_property
    def _is_opto(self) -> bool:
        return npc_samstim.is_opto(self._hdf5)

    @npc_io.cached_property
    def _is_galvo_opto(self) -> bool:
        is_galvo_opto = npc_samstim.is_galvo_opto(self._hdf5)
        if is_galvo_opto and not self._is_opto:
            raise AssertionError(
                f"Conflicting results: {self._is_opto=}, {is_galvo_opto=}"
            )
        return is_galvo_opto

    @npc_io.cached_property
    def _sam(self) -> DynRoutData:
        return npc_samstim.get_sam(self._hdf5)

    @npc_io.cached_property
    def _len(self) -> int:
        """Number of trials"""
        return len(self.start_time)

    @npc_io.cached_property
    def _datetime(self) -> datetime.datetime:
        return npc_session.DatetimeRecord(self._sam.startTime).dt

    @npc_io.cached_property
    def _aud_stims(self) -> npt.NDArray[np.str_]:
        return np.unique([stim for stim in self.stim_name if "sound" in stim.lower()])

    @npc_io.cached_property
    def _vis_stims(self) -> npt.NDArray[np.str_]:
        return np.unique(
            [stim for stim in self._sam.trialStim if "vis" in stim.lower()]
        )

    @npc_io.cached_property
    def _targets(self) -> npt.NDArray[np.str_]:
        return np.unique(list(self._sam.blockStimRewarded))

    @npc_io.cached_property
    def _aud_targets(self) -> npt.NDArray[np.str_]:
        return np.unique([stim for stim in self._aud_stims if stim in self._targets])

    @npc_io.cached_property
    def _vis_targets(self) -> npt.NDArray[np.str_]:
        return np.unique([stim for stim in self._vis_stims if stim in self._targets])

    @npc_io.cached_property
    def _aud_nontargets(self) -> npt.NDArray[np.str_]:
        return np.unique(
            [stim for stim in self._aud_stims if stim not in self._targets]
        )

    @npc_io.cached_property
    def _vis_nontargets(self) -> npt.NDArray[np.str_]:
        return np.unique(
            [stim for stim in self._vis_stims if stim not in self._targets]
        )

    @npc_io.cached_property
    def _trial_rewarded_stim_name(self) -> npt.NDArray[np.str_]:
        return self._sam.blockStimRewarded[self.block_index]

    # ---------------------------------------------------------------------- #
    # times:

    @npc_io.cached_property
    def start_time(self) -> npt.NDArray[np.float64]:
        """earliest time in each trial, before any events occur.

        - currently discards inter-trial period
        - extensions due to quiescent violations are discarded: only the final
          `preStimFramesFixed` before a stim are included
        """
        return self.quiescent_start_time

    @npc_io.cached_property
    def stop_time(self) -> npt.NDArray[np.float64]:
        """latest time in each trial, after all events have occurred"""
        return npc_stim.safe_index(self._flip_times, self._sam.trialEndFrame)

    @npc_io.cached_property
    def quiescent_start_time(self) -> npt.NDArray[np.float64]:
        """start of interval in which the subject should not lick, otherwise the
        trial will start over.

        - only the last quiescent interval (which was not violated) is included
        """
        return npc_stim.safe_index(
            self._flip_times,
            self._sam.stimStartFrame - self._hdf5["quiescentFrames"][()],
        )

    @npc_io.cached_property
    def quiescent_stop_time(self) -> npt.NDArray[np.float64]:
        """end of interval in which the subject should not lick, otherwise the
        trial will start over"""
        return npc_stim.safe_index(
            self._flip_times,
            self._sam.stimStartFrame,
        )

    @npc_io.cached_property
    def stim_start_time(self) -> npt.NDArray[np.float64]:
        """onset of visual or auditory stimulus"""
        starts = np.full(self._len, np.nan)
        for idx in range(self._len):
            if self.is_vis_stim[idx] or self.is_catch[idx]:
                starts[idx] = npc_stim.safe_index(
                    self._vis_display_times, self._sam.stimStartFrame[idx]
                )
            if self.is_aud_stim[idx]:
                starts[idx] = self.get_trial_aud_onset(idx)
        return starts

    @npc_io.cached_property
    def stim_stop_time(self) -> npt.NDArray[np.float64]:
        """offset of visual or auditory stimulus"""
        ends = np.full(self._len, np.nan)
        for idx in range(self._len):
            if self.is_vis_stim[idx] or self.is_catch[idx]:
                ends[idx] = npc_stim.safe_index(
                    self._vis_display_times,
                    self._sam.stimStartFrame[idx]
                    + self._hdf5["trialVisStimFrames"][idx],
                )
            if self.is_aud_stim[idx]:
                ends[idx] = self.get_trial_aud_offset(idx)
        return ends

    @npc_io.cached_property
    def opto_start_time(self) -> npt.NDArray[np.float64]:
        """Onset of optogenetic inactivation"""
        if not self._is_opto:
            return np.full(self._len, np.nan)
        return self.get_trial_opto_onset()

    @npc_io.cached_property
    def opto_stop_time(self) -> npt.NDArray[np.float64]:
        """offset of optogenetic inactivation"""
        if not self._is_opto:
            return np.full(self._len, np.nan)
        return self.get_trial_opto_offset()

    @npc_io.cached_property
    def response_window_start_time(self) -> npt.NDArray[np.float64]:
        """start of interval in which the subject should lick if a GO trial,
        otherwise should not lick"""
        return npc_stim.safe_index(
            self._input_data_times,
            self._sam.stimStartFrame + self._hdf5["responseWindow"][()][0],
        )

    @npc_io.cached_property
    def _response_window_stop_frame(self) -> npt.NDArray[np.float64]:
        return self._sam.stimStartFrame + self._hdf5["responseWindow"][()][1]

    @npc_io.cached_property
    def response_window_stop_time(self) -> npt.NDArray[np.float64]:
        """end of interval in which the subject should lick if a GO trial,
        otherwise should not lick"""
        return npc_stim.safe_index(self._flip_times, self._response_window_stop_frame)

    @npc_io.cached_property
    def task_control_response_time(self) -> npt.NDArray[np.float64]:
        """time of first lick in trial, according to the task control script.

        - a bug present until 2023-10-16 allowed a lick ocurring before
          `response_window_start_time` to be registered as a response, which
          then affected trial outcome
        - values in this column represent the time that the task control script
          registered as a response (regardless of whether the bug was present or not)
        - nan if the task control script did not register a response
        """
        if not self._sync:
            return npc_stim.safe_index(
                self._input_data_times, self._sam.trialResponseFrame
            )
        return npc_stim.safe_index(self._input_data_times, self._sam.trialResponseFrame)

    @npc_io.cached_property
    def response_time(self) -> npt.NDArray[np.float64]:
        """time of first lick within the response window

        - nan if no lick occurred
        - may be nan even if `is_response` is True, for sessions prior to
          2023-10-20 (see `task_control_response_time.description`)
        """
        if not self._sync:
            return self.task_control_response_time

        lick_times = self._sync.get_rising_edges("lick_sensor", units="seconds")
        all_response_times = np.full(self._len, np.nan)
        for idx in self.trial_index:
            trial_response_times = lick_times[
                (lick_times > self.response_window_start_time[idx])
                & (lick_times < self.response_window_stop_time[idx])
            ]
            if not trial_response_times.any():
                if self.is_response[idx]:
                    logger.warning(
                        f"No lick time found within response window on sync for trial {idx}, despite being marked as a response trial."
                    )
                continue
            all_response_times[idx] = trial_response_times[0]
        return all_response_times

    @npc_io.cached_property
    def reward_time(self) -> npt.NDArray[np.floating]:
        """delivery time of water reward, for contingent and non-contingent rewards"""
        all_reward_times = npc_stim.safe_index(self._flip_times, self._sam.rewardFrames)
        all_reward_times = all_reward_times[all_reward_times <= self.stop_time[-1]]
        all_reward_trials = (
            np.searchsorted(
                self.start_time,
                all_reward_times,
                side="right",
            )
            - 1
        )
        reward_time = np.full(self._len, np.nan)
        if np.all(np.where(self.is_rewarded)[0] == all_reward_trials):
            # expected single reward per trial
            reward_time[all_reward_trials] = all_reward_times
        else:
            # mismatch between reward times and trials that are marked as having rewards
            for trial_idx in np.where(self.is_rewarded)[0]:
                reward_times = all_reward_times[
                    (all_reward_times > self.start_time[trial_idx])
                    & (all_reward_times < self.stop_time[trial_idx])
                ]
                if len(reward_times) != 1:
                    logger.warning(
                        f"Multiple reward times found for trial {trial_idx}. Assigning the first: {reward_times} s"
                    )
                reward_time[trial_idx] = reward_times[0]
        return reward_time

    @npc_io.cached_property
    def _timeout_start_frame(self) -> npt.NDArray[np.float64]:
        starts = np.nan * np.ones_like(self.start_time)
        for idx in range(0, self._len - 1):
            if self.is_repeat[idx + 1]:
                starts[idx] = (
                    self._sam.stimStartFrame[idx]
                    + self._hdf5["responseWindow"][()][1]
                    + 1
                )
        return starts

    @npc_io.cached_property
    def timeout_start_time(self) -> npt.NDArray[np.float64]:
        """start of extended inter-trial interval added due to a false alarm"""
        return npc_stim.safe_index(self._input_data_times, self._timeout_start_frame)

    @npc_io.cached_property
    def _timeout_stop_frame(self) -> npt.NDArray[np.float64]:
        return self._timeout_start_frame + self._sam.incorrectTimeoutFrames

    @npc_io.cached_property
    def timeout_stop_time(self) -> npt.NDArray[np.float64]:
        """end of extended inter-trial interval"""
        return npc_stim.safe_index(self._vis_display_times, self._timeout_stop_frame)

    @npc_io.cached_property
    def _post_response_window_start_frame(self) -> npt.NDArray[np.float64]:
        return np.where(
            np.isnan(self._timeout_stop_frame),
            self._response_window_stop_frame,
            self._timeout_stop_frame,
        )

    @npc_io.cached_property
    def post_response_window_start_time(self) -> npt.NDArray[np.float64]:
        """start of null interval in which the subject awaits a new trial;
        may receive a non-contingent reward if scheduled"""
        return npc_stim.safe_index(
            self._vis_display_times, self._post_response_window_start_frame
        )

    @npc_io.cached_property
    def _post_response_window_stop_frame(self) -> npt.NDArray[np.float64]:
        return (
            self._post_response_window_start_frame
            + self._hdf5["postResponseWindowFrames"][()]
        )

    @npc_io.cached_property
    def post_response_window_stop_time(self) -> npt.NDArray[np.float64]:
        """end of null interval"""
        return npc_stim.safe_index(
            self._vis_display_times, self._post_response_window_stop_frame
        )

    '''
    @npc_io.cached_property
    def _time(self) -> npt.NDArray[np.float64]:
        """TODO"""
        return np.nan * np.zeros(self._len)
    '''

    # ---------------------------------------------------------------------- #
    # parameters:

    @npc_io.cached_property
    def stim_name(self) -> npt.NDArray[np.str_]:
        """the stimulus presented; corresponds to a unique stimulus definition,
        randomized over trials
        """
        return self._sam.trialStim

    @npc_io.cached_property
    def block_index(self) -> npt.NDArray[np.int32]:
        """0-indexed block number, increments with each block"""
        assert min(self._sam.trialBlock) == 1
        return self._sam.trialBlock - 1

    @npc_io.cached_property
    def context_name(self) -> npt.NDArray[np.str_]:
        """indicates the rewarded modality in each block"""

        def context(stim: str) -> str:
            if "vis" in stim:
                return "vis"
            if "sound" in stim or "aud" in stim:
                return "aud"
            return "".join(i for i in stim if not i.isdigit())

        return np.array([context(name) for name in self._trial_rewarded_stim_name])

    @npc_io.cached_property
    def trial_index(self) -> npt.NDArray[np.int32]:
        """0-indexed trial number"""
        return np.arange(self._len)

    @npc_io.cached_property
    def trial_index_in_block(self) -> npt.NDArray[np.int32]:
        index_in_block = np.full(self._len, np.nan)
        for i in range(self._len):
            if (
                self.trial_index[i] == 0
                or self.block_index[i] != self.block_index[i - 1]
            ):
                count = 0
            else:
                count += 1
            index_in_block[i] = count
        return index_in_block

    @npc_io.cached_property
    def _regular_trial_index(self) -> npt.NDArray:
        """
        0-indexed trial number for regular trials (with stimuli), increments over
        time.

        - nan for catch trials
        - nan for repeats
        """
        regular_trial_index = np.nan * np.zeros(self._len)
        counter = -1
        for idx in range(self._len):
            if self.is_catch[idx] or self.is_repeat[idx]:
                continue
            counter += 1
            regular_trial_index[idx] = int(counter)
        return regular_trial_index

    @npc_io.cached_property
    def _regular_trial_index_in_block(self) -> npt.NDArray[np.int32]:
        """0-indexed trial number within a block, increments over the block.

        - nan for catch trials
        - nan for repeats
        """
        index_in_block = np.full(self._len, np.nan)
        for i in range(self._len):
            if np.isnan(self._regular_trial_index[i]):
                continue
            elif (
                self.trial_index[i] == 0
                or self.block_index[i] != self.block_index[i - 1]
            ):
                count = 0
            else:
                count += 1
            index_in_block[i] = count
        return index_in_block

    @npc_io.cached_property
    def _scheduled_reward_index_in_block(self) -> npt.NDArray[np.float64]:
        """0-indexed trial number within a block for trials in which a
        non-contingent reward was scheduled.

         - see `is_reward_scheduled`
        """
        return np.where(
            self.is_reward_scheduled is True,
            self.trial_index_in_block,
            np.full(self._len, np.nan),
        )

    def get_trial_opto_devices(self, trial_idx: int) -> tuple[str, ...]:
        if not self._is_opto:
            raise ValueError("No opto devices in non-opto session")
        if (devices := self._hdf5.get("trialOptoDevice")) is None or devices.size == 0:
            assert self._datetime.date() < datetime.date(
                2023, 8, 1
            )  # older sessions may lack info
            return ("laser_488",)
        devices = devices.asstr()[trial_idx]
        if devices[0] + devices[-1] != "[]":
            # basic check before we eval code from the web
            raise ValueError(f"Invalid opto devices string: {devices}")
        return tuple(eval(devices))

    @npc_io.cached_property
    def _trial_opto_devices(self) -> tuple[tuple[str, ...], ...]:
        return tuple(self.get_trial_opto_devices(idx) for idx in range(self._len))

    @npc_io.cached_property
    def opto_wavelength(self) -> tuple[tuple[Any, ...]] | npt.NDArray[np.floating]:
        if not self._is_opto:
            return np.full(self._len, np.nan)

        def parse_wavelengths(
            devices: tuple[str, ...] | str
        ) -> tuple[int | np.floating, ...]:
            if isinstance(devices, str):
                if not devices:
                    return (np.nan,)  # type: ignore
                if devices in ("led_1", "led_2"):
                    return (470,)  # behavior box cannulae test experiments
                try:
                    value = int(devices.split("_")[-1])
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid opto devices string (expected 'laser_488' format): {devices}"
                    ) from exc
                else:
                    assert (
                        300 < value < 1000
                    ), f"Unexpected wavelength parsed from `trialOptoDevice`: {value}"
                    return (value,)
            result: tuple[int | np.floating, ...] = ()
            for device in devices:
                result = result + parse_wavelengths(device)
            return result

        return self._normalize_opto_data(
            [parse_wavelengths(v) for v in self._trial_opto_devices]
        )

    @npc_io.cached_property
    def _opto_params_index(self) -> npt.NDArray[np.float64] | None:
        if not self._is_opto:
            return np.full(self._len, np.nan)
        if found := self._hdf5.get("trialOptoParamsIndex"):
            return found[()][self.trial_index]
        return None

    @property
    def _rig(self) -> str:
        return self._hdf5["rigName"].asstr()[()]

    @staticmethod
    def txtToDict(txt: str) -> dict[str, list[float]]:
        """From Sam's code, modified to use text directly rather than a file"""
        cols = zip(*[line.strip("\n").split("\t") for line in txt.split("\n")])
        return {d[0]: [float(s) for s in d[1:]] for d in cols}

    def getBregmaGalvoCalibrationData(self) -> dict[str, float | list[float]]:
        """From Sam's code, modified to use synced copies on s3"""
        root = npc_lims.DR_DATA_REPO.parent / "OptoGui" / self._rig
        bregmaGalvoFile = root / f"{self._rig}_bregma_galvo.txt"
        bregmaOffsetFile = root / f"{self._rig}_bregma_offset.txt"
        voltages = self.txtToDict(bregmaGalvoFile.read_text())
        offsets = {
            k: v[0] for k, v in self.txtToDict(bregmaOffsetFile.read_text()).items()
        }
        return voltages | offsets

    def getOptoPowerCalibrationData(self, opto_device_name: str):
        """From Sam's code, modified to use synced copies on s3"""
        root = npc_lims.DR_DATA_REPO.parent / "OptoGui" / self._rig
        powerCalibrationFile = root / f"{self._rig}_{opto_device_name}_power.txt"
        d = self.txtToDict(powerCalibrationFile.read_text())
        p = np.polyfit(d["input (V)"], d["power (mW)"], 2)
        d["poly coefficients"] = p.tolist()  # tolist() to quiet mypy
        d["offsetV"] = min(np.roots(p))
        return d

    @npc_io.cached_property
    def _is_galvo_voltage_xy_separate(self) -> bool:
        """Whether galvo voltage is stored as separate x and y values

        - sam separated x and y values for opto in the task on 2024-03-29
        - `trialGalvoVoltage` -> `trialGalvoX` and `trialGalvoY`
        """
        if hasattr(self._sam, "trialGalvoX") and hasattr(self._sam, "trialGalvoY"):
            return True
        if hasattr(self._sam, "trialGalvoVoltage"):
            return False
        return False

    @npc_io.cached_property
    def _galvo_voltage_x(self):
        if self._is_galvo_voltage_xy_separate:
            return tuple(self._sam.trialGalvoX)
        else:
            return tuple([(v[0],) for v in self._galvo_voltage_xy])

    @npc_io.cached_property
    def _galvo_voltage_y(self):
        if self._is_galvo_voltage_xy_separate:
            return tuple(self._sam.trialGalvoY)
        else:
            return tuple([(v[1],) for v in self._galvo_voltage_xy])

    @npc_io.cached_property
    def _galvo_voltage_xy(self) -> tuple[tuple[np.float64, np.float64], ...]:
        """only used to provide separate x and y attrs for old data, pre-2024-03-29"""
        if self._is_galvo_voltage_xy_separate:
            raise AttributeError(
                "This property should not be called when galvo voltage is stored as separate x and y values"
            )
        elif len(self._sam.trialGalvoVoltage.shape) < 3:
            if not all(len(v) == 2 for v in self._sam.trialGalvoVoltage):
                # a set of experiments with 670248 had a bug where galvo
                # voltage was a single value.
                # Fortunately, there was also only one possible galvo voltage
                # available
                if (
                    len(g := self._hdf5["galvoVoltage"][()]) == 1
                    and len(xy := g[0]) == 2
                ):
                    return tuple((xy[0], xy[1]) for _ in range(self._len))
                raise IndexError("trialGalvoVoltage has elements with len != 2")
            result = tuple(tuple(v) for v in self._sam.trialGalvoVoltage)
        else:
            self.assert_single_opto_device()
            result = tuple(tuple(v) for v in self._sam.trialGalvoVoltage[:, 0, :])
        return result
        # return tuple((np.nan, np.nan) if np.isnan(params_idx) else tuple(self._sam.trialGalvoVoltage[idx, int(params_idx), :]) for idx, params_idx in enumerate(self._opto_params_index))

    @npc_io.cached_property
    def _bregma_galvo_calibration_data(self) -> dict[str, float | list[float]]:
        return self.getBregmaGalvoCalibrationData()

    @npc_io.cached_property
    def _opto_location_bregma_x(self) -> tuple[tuple[np.float64], ...]:
        if not self._is_galvo_voltage_xy_separate:
            return tuple((v[0],) for v in self._galvo_voltage_xy)
        else:
            result = copy.deepcopy(self._galvo_voltage_x)
            for trial_idx, x_values in enumerate(self._galvo_voltage_x):
                for location_idx, x in enumerate(x_values):
                    if np.isnan(x):
                        continue
                    else:
                        print(trial_idx, location_idx, x, x_values)
                        value = self.bregma_to_galvo(trial_idx, location_idx)[0]
                    result[trial_idx][location_idx] = value
            return tuple(result)

    def bregma_to_galvo(self, trial_idx, location_idx):
        opto_params = self._hdf5["optoParams"]
        i = opto_params["label"] == self.opto_label[trial_idx][location_idx]
        x, y = (
            opto_params[f"bregma{coord}"][i] + opto_params[f"bregma offset {coord}"][i]
            for coord in "XY"
        )
        return x, y

    @npc_io.cached_property
    def _opto_location_bregma_y(self) -> tuple[tuple[np.float64], ...]:
        if not self._is_galvo_voltage_xy_separate:
            return tuple((v[1],) for v in self._galvo_voltage_xy)
        else:
            result = copy.deepcopy(self._galvo_voltage_y)
            for trial_idx, y_values in enumerate(self._galvo_voltage_y):
                for location_idx, y in enumerate(y_values):
                    if np.isnan(y):
                        continue
                    else:
                        value = self.bregma_to_galvo(trial_idx, location_idx)[1]
                    result[trial_idx][location_idx] = value
            return tuple(result)

    @npc_io.cached_property
    def _opto_location_bregma_xy(self) -> tuple[tuple[np.float64, np.float64], ...]:
        if self._is_galvo_voltage_xy_separate:
            raise AttributeError(
                "This property should not be called when galvo voltage is stored as separate x and y values"
            )
        # bregma xy may be stored in the hdf5 file directly:
        bregma = self._hdf5.get("optoBregma") or self._hdf5.get("bregmaXY")
        if bregma is not None:
            galvo = self._hdf5["galvoVoltage"][()]
            return tuple(
                tuple(bregma[np.all(galvo == v, axis=1)][0])
                for v in self._galvo_voltage_xy
            )

        # otherwise, we need to calculate it from galvo voltages
        old_params = ("bregmaXOffset", "bregmaYOffset")  # not used after 2024-03-29
        if (calibration_data := self._hdf5.get("bregmaGalvoCalibrationData")) is None:
            calibration_data = self.getBregmaGalvoCalibrationData()
        else:
            calibration_data = dict(calibration_data.items())  # prevent writing to hdf5
            for param in old_params:
                if param in calibration_data:
                    calibration_data[param] = calibration_data[param][()]
        bregma_coords = tuple(
            DynamicRoutingTask.TaskUtils.galvoToBregma(
                calibration_data,
                *voltages,
            )
            for voltages in self._galvo_voltage_xy
        )[: self._len]
        if all(param in calibration_data for param in old_params):
            bregma_coords = tuple(
                (
                    coords[0] + calibration_data["bregmaXOffset"],
                    coords[1] + calibration_data["bregmaYOffset"],
                )
                for coords in bregma_coords
            )
        return bregma_coords

    @npc_io.cached_property
    def opto_location_bregma_x(
        self,
    ) -> npt.NDArray[np.floating] | tuple[tuple[Any, ...]]:
        if not self._is_galvo_opto:
            return np.full(self._len, np.nan)
        return self._normalize_opto_data(self._opto_location_bregma_x)

    @npc_io.cached_property
    def opto_location_bregma_y(
        self,
    ) -> npt.NDArray[np.floating] | tuple[tuple[Any, ...]]:
        if not self._is_galvo_opto:
            return np.full(self._len, np.nan)
        return self._normalize_opto_data(self._opto_location_bregma_y)

    @npc_io.cached_property
    def _opto_label(self) -> npt.NDArray[np.floating] | tuple[tuple[Any, ...]]:
        """target location for optogenetic inactivation during the trial"""
        if not self._is_galvo_opto:
            return np.full(self._len, np.nan)
        if trialOptoLabel := self._hdf5.get("trialOptoLabel", None):
            labels = trialOptoLabel.asstr()[()]
            result = np.where(
                labels != "no opto",
                labels,
                np.nan,
            )[: self._len]
        elif optoLocs := self._hdf5.get("optoLocs"):
            label = optoLocs["label"].asstr()[()]
            xy = np.array(list(zip(optoLocs["X"], optoLocs["Y"])))
            result = np.array(
                [
                    label[np.all(xy == v, axis=1)][0]
                    for v in self._opto_location_bregma_xy
                ],
                dtype=str,
            )[: self._len]
        else:
            logger.warning("No known opto location data found")
            u = tuple(set(self._galvo_voltage_xy))
            result = np.asarray(
                [f"unlabeled{u.index(xy)}" for xy in self._galvo_voltage_xy], dtype=str
            )
        return self._normalize_opto_data(result)

    def _normalize_opto_data(self, data: Iterable[Any]) -> tuple[tuple[Any, ...]]:
        """After Sam made changes to how opto params are stored in March '24, we
        need to make sure all opto data is stored in the same format"""
        data = list(data)
        for trial_idx, voltage_data in enumerate(self._galvo_voltage_x):
            trial_data = data[trial_idx]
            if isinstance(trial_data, str) or not isinstance(trial_data, Iterable):
                copy = [trial_data]
            else:
                copy = list(trial_data)
            # we need all opto data to have the same length as the galvo Voltage data
            # (which may specify multiple locations for a single trial)
            if len(copy) != (L := len(voltage_data)):
                assert (
                    len(copy) == 1
                ), f"Trial {trial_idx} has mismatched opto params - expected all params to have len={L} (to match number of locations specified by galvo voltages) or len=1 (for params that are the same for all locations) - got: {trial_data}"
                copy = copy * len(voltage_data)
            data[trial_idx] = tuple(copy)
        return tuple(data)

    @npc_io.cached_property
    def opto_duration(self) -> npt.NDArray[np.floating]:
        if not self._is_opto:
            return np.full(self._len, np.nan)
        return self._sam.trialOptoDur[: self._len].squeeze()

    @npc_io.cached_property
    def opto_label(self) -> npt.NDArray[np.floating] | tuple[tuple[Any, ...], ...]:
        if not self._is_opto:
            return np.full(self._len, np.nan)
        if all(str(v).upper() in "ABCDEF" for v in self._opto_label):
            result = np.array(
                [f"probe{str(v).upper()}" for v in self._opto_label], dtype=str
            )
            return self._normalize_opto_data(result)
        else:
            return self._opto_label

    @npc_io.cached_property
    def _opto_voltage(self) -> npt.NDArray[np.float64]:
        voltages = np.full(self._len, np.nan)
        if not self._is_opto:
            return voltages
        for idx in range(self._len):
            if self._opto_params_index is None:
                voltages[idx] = self._hdf5["trialOptoVoltage"][idx]
                continue
            v = self._hdf5["trialOptoVoltage"][self._opto_params_index][idx]
            if v.shape:
                voltages[idx] = npc_stim.safe_index(v, self._opto_params_index[idx])
                continue
            voltages[idx] = v
        return voltages

    @npc_io.cached_property
    def opto_power(self) -> npt.NDArray[np.floating] | tuple[tuple[Any, ...], ...]:
        if not self._is_opto:
            return np.full(self._len, np.nan)
        if (voltage := self._hdf5["trialOptoVoltage"]).ndim == 1:
            voltages = voltage[: self._len]
        else:
            self.assert_single_opto_device()
            voltages = voltage[: self._len, 0]

        if (
            data := self._hdf5.get("optoPowerCalibrationData")
        ) is None or "poly coefficients" not in data:
            powers = []
            for voltage, devices in zip(voltages, self._trial_opto_devices):
                if not devices:
                    powers.append(np.nan)
                    continue
                if isinstance(devices, str):
                    devices = (devices,)
                if len(devices) > 1:
                    raise ValueError(
                        f"Not ready to handle multiple opto devices: {devices}"
                    )
                if not devices[0]:
                    powers.append(np.nan)
                    continue
                powers.append(
                    DynamicRoutingTask.TaskUtils.voltsToPower(
                        self.getOptoPowerCalibrationData(devices[0]),
                        voltage,
                    ).item()
                )
            result = np.array(powers)
        else:
            result = np.where(
                np.isnan(voltages),
                np.nan,
                DynamicRoutingTask.TaskUtils.voltsToPower(
                    self._hdf5["optoPowerCalibrationData"],
                    voltages,
                ),
            )
        return self._normalize_opto_data(result)

    @npc_io.cached_property
    def opto_stim_name(self) -> npt.NDArray[np.str_] | npt.NDArray[np.floating]:
        """stimulus presented during optogenetic inactivation, corresponding to
        keys in `stimulus` dict.

        - not for comparison across sessions: the same stimulus may have different
         names
        """
        if not self._is_opto:
            return np.full(self._len, np.nan)
        return np.array(
            [
                (
                    f"opto{self._unique_opto_waveforms.index(w)}_{loc}"
                    if w is not None
                    else ""
                )
                for w, loc in zip(self._opto_stim_waveforms, self.opto_label)
            ],
            dtype=str,
        )

    @npc_io.cached_property
    def repeat_index(self) -> npt.NDArray[np.float64]:
        """number of times the trial has already been presented in immediately
        preceding trials.

        - counts repeats due to misses
        - nan for catch trials
        """
        repeats = np.where(
            ~np.isnan(self.trial_index), np.zeros(self._len), np.full(self._len, np.nan)
        )
        counter = 0
        for idx in np.where(repeats == 0)[0]:
            if self.is_repeat[idx]:
                counter += 1
            else:
                counter = 0
            repeats[idx] = int(counter)
        return repeats

    # ---------------------------------------------------------------------- #
    # bools:

    @npc_io.cached_property
    def is_response(self) -> npt.NDArray[np.bool_]:
        """the subject licked one or more times during the response window"""
        return self._sam.trialResponse

    @npc_io.cached_property
    def is_correct(self) -> npt.NDArray[np.bool_]:
        """the subject acted correctly in the response window, according to its
        training.

        - includes correct reject for catch trials
        """
        return (
            self.is_hit | self.is_correct_reject | (self.is_catch & ~self.is_response)
        )

    @npc_io.cached_property
    def is_incorrect(self) -> npt.NDArray[np.bool_]:
        """the subject acted incorrectly in the response window, according to its
        training.

        - includes false alarm for catch trials
        """
        return self.is_miss | self.is_false_alarm | (self.is_catch & self.is_response)

    @npc_io.cached_property
    def is_hit(self) -> npt.NDArray[np.bool_]:
        """the subject responded in a GO trial"""
        return self.is_response & self.is_go

    @npc_io.cached_property
    def is_false_alarm(self) -> npt.NDArray[np.bool_]:
        """the subject responded in a NOGO trial

        - excludes catch trials
        """
        return self.is_response & self.is_nogo

    @npc_io.cached_property
    def is_correct_reject(self) -> npt.NDArray[np.bool_]:
        """the subject did not respond in a NOGO trial

        - excludes catch trials
        """
        return ~self.is_response & self.is_nogo

    @npc_io.cached_property
    def is_miss(self) -> npt.NDArray[np.bool_]:
        """the subject did not respond in a GO trial"""
        return ~self.is_response & self.is_go

    @npc_io.cached_property
    def is_go(self) -> npt.NDArray[np.bool_]:
        """condition in which the subject should respond.

        - target stim presented in rewarded context block
        """
        return self._sam.trialStim == self._sam.rewardedStim

    @npc_io.cached_property
    def is_nogo(self) -> npt.NDArray[np.bool_]:
        """condition in which the subject should not respond.

        - non-target stim presented in any context block
        - target stim presented in non-rewarded context block
        - excludes catch trials
        """
        return self._sam.nogoTrials

    @npc_io.cached_property
    def is_rewarded(self) -> npt.NDArray[np.bool_]:
        """the subject received a reward.

        - includes non-contingent rewards
        """
        return self._sam.trialRewarded

    @npc_io.cached_property
    def is_noncontingent_reward(self) -> npt.NDArray[np.bool_]:
        """the subject received a reward that did not depend on its response"""
        return self._sam.autoRewarded

    @npc_io.cached_property
    def is_contingent_reward(self) -> npt.NDArray[np.bool_]:
        """the subject received a reward for a correct response"""
        return self._sam.rewardEarned

    @npc_io.cached_property
    def is_reward_scheduled(self) -> npt.NDArray[np.bool_]:
        """a non-contingent reward was scheduled to occur, regardless of
        whether it was received.

        - subject may have responded correctly and received contingent reward
          instead
        """
        return self._sam.autoRewardScheduled

    @npc_io.cached_property
    def is_aud_stim(self) -> npt.NDArray[np.bool_]:
        """an auditory stimulus was presented.

        - includes target and non-target stimuli
        - includes rewarded and non-rewarded contexts
        - excludes catch trials (no stimulus)
        """
        return np.isin(self._sam.trialStim, self._aud_stims)

    @npc_io.cached_property
    def is_vis_stim(self) -> npt.NDArray[np.bool_]:
        """a visual stimulus was presented.

        - includes target and non-target stimuli
        - includes rewarded and non-rewarded contexts
        - excludes catch trials (no stimulus)
        """
        return np.isin(self._sam.trialStim, self._vis_stims)

    @npc_io.cached_property
    def is_catch(self) -> npt.NDArray[np.bool_]:
        """no stimulus was presented"""
        return np.isin(self._sam.trialStim, "catch")

    @npc_io.cached_property
    def is_target(self) -> npt.NDArray[np.bool_]:
        """a stimulus was presented that the subject should respond
        to only in a specific context"""
        return np.isin(self._sam.trialStim, self._targets)

    @npc_io.cached_property
    def is_aud_target(self) -> npt.NDArray[np.bool_]:
        """an auditory stimulus was presented that the subject should respond
        to only in a specific context"""
        return np.isin(self._sam.trialStim, self._aud_targets)

    @npc_io.cached_property
    def is_vis_target(self) -> npt.NDArray[np.bool_]:
        """a visual stimulus was presented that the subject should respond to
        only in a specific context"""
        return np.isin(self._sam.trialStim, self._vis_targets)

    @npc_io.cached_property
    def is_nontarget(self) -> npt.NDArray[np.bool_]:
        """a stimulus was presented that the subject should never respond to"""
        return self.is_aud_nontarget | self.is_vis_nontarget

    @npc_io.cached_property
    def is_aud_nontarget(self) -> npt.NDArray[np.bool_]:
        """an auditory stimulus was presented that the subject should never respond to"""
        return np.isin(self._sam.trialStim, self._aud_nontargets)

    @npc_io.cached_property
    def is_vis_nontarget(self) -> npt.NDArray[np.bool_]:
        """a visual stimulus was presented that the subject should never respond to"""
        return np.isin(self._sam.trialStim, self._vis_nontargets)

    @npc_io.cached_property
    def is_vis_context(self) -> npt.NDArray[np.bool_]:
        """visual target stimuli are rewarded"""
        return np.isin(self._trial_rewarded_stim_name, self._vis_stims)

    @npc_io.cached_property
    def is_aud_context(self) -> npt.NDArray[np.bool_]:
        """auditory target stimuli are rewarded"""
        return np.isin(self._trial_rewarded_stim_name, self._aud_stims)

    @npc_io.cached_property
    def is_context_switch(self) -> npt.NDArray[np.bool_]:
        """the first trial with a stimulus after a change in context"""
        return np.isin(self.trial_index_in_block, 0) & ~np.isin(self.block_index, 0)

    @npc_io.cached_property
    def is_repeat(self) -> npt.NDArray[np.bool_]:
        """the trial is a repetition of the previous trial, due to a
        miss"""
        return self._sam.trialRepeat

    @npc_io.cached_property
    def is_opto(self) -> npt.NDArray[np.bool_]:
        """optogenetic inactivation was applied during the trial"""
        return ~np.isnan(self.opto_start_time)

    @npc_io.cached_property
    def is_single_opto_location(self) -> npt.NDArray[np.bool_]:
        """only one optogenetic inactivation was applied during the trial"""
        if not self._is_galvo_opto:
            return np.full(self._len, np.nan)
        self.assert_single_opto_device()
        return np.array(
            [len(locations) > 1 for locations in self._opto_location_bregma_x]
        )

    """
    @npc_io.cached_property
    def is_(self) -> npt.NDArray[np.bool_]:
        """ """
        return
    """


if __name__ == "__main__":
    import doctest

    import dotenv

    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
