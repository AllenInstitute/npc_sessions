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

import functools
import logging
from collections.abc import Iterable

import DynamicRoutingTask.TaskUtils
import numpy as np
import numpy.typing as npt
from DynamicRoutingTask.Analysis.DynamicRoutingAnalysisUtils import DynRoutData

import npc_sessions.utils as utils
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
    >>> assert not trials._df.is_empty()
    """

    def __init__(
        self,
        hdf5: utils.StimPathOrDataset,
        sync: utils.SyncPathOrDataset | None = None,
        ephys_recording_dirs: Iterable[utils.PathLike] | None = None,
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

    @property
    def _opto_stim_recordings(self) -> tuple[utils.StimRecording | None, ...] | None:
        self._cached_opto_stim_recordings: tuple[utils.StimRecording | None, ...] | None
        cached = getattr(self, "_cached_opto_stim_recordings", None)
        if cached is not None:
            return cached
        if self._sync:
            self._cached_opto_stim_recordings = utils.get_stim_latencies_from_sync(
                self._hdf5,
                self._sync,
                waveform_type="opto",
            )
        else:
            self._cached_opto_stim_recordings = None
        return self._cached_opto_stim_recordings

    @_opto_stim_recordings.setter
    def _opto_stim_recordings(
        self, value: Iterable[utils.StimRecording | None]
    ) -> None:
        """Can be set on init by passing as kwarg"""
        self._cached_opto_stim_recordings = tuple(value)

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
            return self.get_script_frame_time(self._hdf5["trialStimStartFrame"][trial])
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
            return self.get_trial_aud_onset(trial) + self._hdf5["trialSoundDur"][trial]
        return utils.safe_index(self._aud_stim_offset_times, trial)

    # ---------------------------------------------------------------------- #
    # helper-properties that won't become columns:

    @functools.cached_property
    def _has_opto(self) -> bool:
        return hasattr(self._sam, "optoVoltage") and any(self._sam.optoVoltage)

    @functools.cached_property
    def _sam(self) -> DynRoutData:
        obj = DynRoutData()
        obj.loadBehavData(filePath="dummy_366122_", h5pyFile=self._hdf5)
        return obj

    @functools.cached_property
    def _len(self) -> int:
        """Number of trials"""
        return len(self.start_time)

    @functools.cached_property
    def _aud_stims(self) -> npt.NDArray[np.str_]:
        return np.unique([stim for stim in self.stim_name if "sound" in stim.lower()])

    @functools.cached_property
    def _vis_stims(self) -> npt.NDArray[np.str_]:
        return np.unique(
            [stim for stim in self._sam.trialStim if "vis" in stim.lower()]
        )

    @functools.cached_property
    def _targets(self) -> npt.NDArray[np.str_]:
        return np.unique(list(self._sam.blockStimRewarded))

    @functools.cached_property
    def _aud_targets(self) -> npt.NDArray[np.str_]:
        return np.unique([stim for stim in self._aud_stims if stim in self._targets])

    @functools.cached_property
    def _vis_targets(self) -> npt.NDArray[np.str_]:
        return np.unique([stim for stim in self._vis_stims if stim in self._targets])

    @functools.cached_property
    def _aud_nontargets(self) -> npt.NDArray[np.str_]:
        return np.unique(
            [stim for stim in self._aud_stims if stim not in self._targets]
        )

    @functools.cached_property
    def _vis_nontargets(self) -> npt.NDArray[np.str_]:
        return np.unique(
            [stim for stim in self._vis_stims if stim not in self._targets]
        )

    @functools.cached_property
    def _trial_rewarded_stim_name(self) -> npt.NDArray[np.str_]:
        return self._sam.blockStimRewarded[self.block_index]

    # ---------------------------------------------------------------------- #
    # times:

    @functools.cached_property
    def start_time(self) -> npt.NDArray[np.float64]:
        """earliest time in each trial, before any events occur.

        - currently discards inter-trial period
        - extensions due to quiescent violations are discarded: only the final
          `preStimFramesFixed` before a stim are included
        """
        return self.get_vis_display_time(
            self._sam.stimStartFrame - self._hdf5["preStimFramesFixed"][()]
        )

    @functools.cached_property
    def stop_time(self) -> npt.NDArray[np.float64]:
        """latest time in each trial, after all events have occurred"""
        return self.get_vis_display_time(self._sam.trialEndFrame)

    @functools.cached_property
    def quiescent_start_time(self) -> npt.NDArray[np.float64]:
        """start of interval in which the subject should not lick, otherwise the
        trial will start over.

        - only the last quiescent interval (which was not violated) is included
        """
        return self.get_vis_display_time(
            self._sam.stimStartFrame - self._hdf5["quiescentFrames"][()]
        )

    @functools.cached_property
    def quiescent_stop_time(self) -> npt.NDArray[np.float64]:
        """end of interval in which the subject should not lick, otherwise the
        trial will start over"""
        return self.stim_start_time

    @functools.cached_property
    def stim_start_time(self) -> npt.NDArray[np.float64]:
        """onset of visual or auditory stimulus"""
        starts = np.nan * np.ones(self._len)
        for idx in range(self._len):
            if self.is_vis_stim[idx] or self.is_catch[idx]:
                starts[idx] = self.get_vis_display_time(self._sam.stimStartFrame[idx])
            if self.is_aud_stim[idx]:
                starts[idx] = self.get_trial_aud_onset(idx)
        return starts

    @functools.cached_property
    def stim_stop_time(self) -> npt.NDArray[np.float64]:
        """offset of visual or auditory stimulus"""
        ends = np.nan * np.ones(self._len)
        for idx in range(self._len):
            if self.is_vis_stim[idx] or self.is_catch[idx]:
                ends[idx] = self.get_vis_display_time(
                    self._sam.stimStartFrame[idx]
                    + self._hdf5["trialVisStimFrames"][idx]
                )
            if self.is_aud_stim[idx]:
                ends[idx] = self.get_trial_aud_offset(idx)
        return ends

    @functools.cached_property
    def opto_start_time(self) -> npt.NDArray[np.float64]:
        """Onset of optogenetic inactivation"""
        if not self._has_opto:
            return np.nan * np.ones(self._len)
        return np.where(
            ~np.isnan(self._sam.trialOptoOnsetFrame),
            self.get_vis_display_time(
                self._sam.stimStartFrame + self._sam.trialOptoOnsetFrame
            ),
            np.nan * np.ones(self._len),
        )

    @functools.cached_property
    def opto_stop_time(self) -> npt.NDArray[np.float64]:
        """offset of optogenetic inactivation"""
        if not self._has_opto:
            return np.nan * np.ones(self._len)
        return self.opto_start_time + self._sam.trialOptoDur

    @functools.cached_property
    def response_window_start_time(self) -> npt.NDArray[np.float64]:
        """start of interval in which the subject should lick if a GO trial,
        otherwise should not lick"""
        return self.get_vis_display_time(
            self._sam.stimStartFrame + self._hdf5["responseWindow"][()][0]
        )

    @functools.cached_property
    def _response_window_stop_frame(self) -> npt.NDArray[np.float64]:
        return self._sam.stimStartFrame + self._hdf5["responseWindow"][()][1]

    @functools.cached_property
    def response_window_stop_time(self) -> npt.NDArray[np.float64]:
        """end of interval in which the subject should lick if a GO trial,
        otherwise should not lick"""
        return self.get_vis_display_time(self._response_window_stop_frame)

    @functools.cached_property
    def response_time(self) -> npt.NDArray[np.float64]:
        """time of first lick within the response window

        - nan if no lick occurred"""
        return self.get_vis_display_time(self._sam.trialResponseFrame)

    @functools.cached_property
    def _timeout_start_frame(self) -> npt.NDArray[np.float64]:
        starts = np.nan * np.ones_like(self.start_time)
        for idx in range(0, self._len - 1):
            if self.is_repeat[idx + 1]:
                starts[idx] = self.get_vis_display_time(
                    self._sam.stimStartFrame[idx]
                    + self._hdf5["responseWindow"][()][1]
                    + 1
                )
        return starts

    @functools.cached_property
    def timeout_start_time(self) -> npt.NDArray[np.float64]:
        """start of extended inter-trial interval added due to a false alarm"""
        return self.get_vis_display_time(self._timeout_start_frame)

    @functools.cached_property
    def _timeout_stop_frame(self) -> npt.NDArray[np.float64]:
        return self._timeout_start_frame + self._sam.incorrectTimeoutFrames

    @functools.cached_property
    def timeout_stop_time(self) -> npt.NDArray[np.float64]:
        """end of extended inter-trial interval"""
        return self.get_vis_display_time(self._timeout_stop_frame)

    @functools.cached_property
    def _post_response_window_start_frame(self) -> npt.NDArray[np.float64]:
        return np.where(
            np.isnan(self._timeout_stop_frame),
            self._response_window_stop_frame,
            self._timeout_stop_frame,
        )

    @functools.cached_property
    def post_response_window_start_time(self) -> npt.NDArray[np.float64]:
        """start of null interval in which the subject awaits a new trial;
        may receive a noncontingent reward if scheduled"""
        return self.get_vis_display_time(self._post_response_window_start_frame)

    @functools.cached_property
    def _post_response_window_stop_frame(self) -> npt.NDArray[np.float64]:
        return (
            self._post_response_window_start_frame
            + self._hdf5["postResponseWindowFrames"][()]
        )

    @functools.cached_property
    def post_response_window_stop_time(self) -> npt.NDArray[np.float64]:
        """end of null interval"""
        return self.get_vis_display_time(self._post_response_window_stop_frame)

    '''
    @functools.cached_property
    def _time(self) -> npt.NDArray[np.float64]:
        """TODO"""
        return np.nan * np.zeros(self._len)
    '''

    # ---------------------------------------------------------------------- #
    # parameters:

    @functools.cached_property
    def stim_name(self) -> npt.NDArray[np.str_]:
        """the stimulus presented; corresponds to a unique stimulus definition,
        randomized over trials
        """
        return self._sam.trialStim

    @functools.cached_property
    def block_index(self) -> npt.NDArray[np.int32]:
        """0-indexed block number, increments with each block"""
        assert min(self._sam.trialBlock) == 1
        return self._sam.trialBlock - 1

    @functools.cached_property
    def context_name(self) -> npt.NDArray[np.str_]:
        """indicates the rewarded modality in each block"""
        return np.array([name[:-1] for name in self._trial_rewarded_stim_name])

    @functools.cached_property
    def trial_index(self) -> npt.NDArray[np.int32]:
        """0-indexed trial number"""
        return np.arange(self._len)

    @functools.cached_property
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

    @functools.cached_property
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

    @functools.cached_property
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

    @functools.cached_property
    def _scheduled_reward_index_in_block(self) -> npt.NDArray[np.float64]:
        """0-indexed trial number within a block for trials in which a
        non-contingent reward was scheduled.

         - see `is_reward_scheduled`
        """
        return np.where(
            self.is_reward_scheduled is True,
            self.trial_index_in_block,
            np.nan * np.ones(self._len),
        )

    def get_trial_opto_devices(self, trial_idx: int) -> tuple[str, ...]:
        devices = self._hdf5["trialOptoDevice"].asstr()[trial_idx]
        if not devices:
            return ()
        if devices[0] + devices[-1] != "[]":
            # basic check before we eval code from the web
            raise ValueError(f"Invalid opto devices string: {devices}")
        return tuple(eval(devices))

    @functools.cached_property
    def _opto_params_index(self) -> npt.NDArray[np.float64]:
        if not any(self.is_opto):
            return np.full(self._len, np.nan)
        return self._hdf5["trialOptoParamsIndex"][()]

    @functools.cached_property
    def _opto_location_bregma_xy(self) -> tuple[tuple[np.float64, np.float64], ...]:
        bregma = self._hdf5.get("optoBregma", None) or self._hdf5.get("bregmaXY", None)
        galvo = self._hdf5["galvoVoltage"][()]
        return tuple(tuple(bregma[np.all(galvo == v, axis=1)][0]) for v in self._hdf5["trialGalvoVoltage"])  # type: ignore

    @functools.cached_property
    def opto_location_bregma_x(self) -> npt.NDArray[np.float64]:
        if not any(self.is_opto):
            return np.full(self._len, np.nan)
        return np.array([bregma[0] for bregma in self._opto_location_bregma_xy])

    @functools.cached_property
    def opto_location_bregma_y(self) -> npt.NDArray[np.float64]:
        if not any(self.is_opto):
            return np.full(self._len, np.nan)
        return np.array([bregma[1] for bregma in self._opto_location_bregma_xy])

    @functools.cached_property
    def _opto_location_name(self) -> npt.NDArray[np.str_]:
        """target location for optogenetic inactivation during the trial"""
        if not any(self.is_opto):
            return np.full(self._len, np.nan)
        if trialOptoLabel := self._hdf5.get("trialOptoLabel", None):
            labels = trialOptoLabel.asstr()[()]
            return np.where(
                labels != "no opto",
                labels,
                np.nan,
            )
        if optoLocs := self._hdf5.get("optoLocs"):
            label = optoLocs["label"].asstr()[()]
            xy = np.array([(x, y) for x, y in zip(optoLocs["X"], optoLocs["Y"])])
            return np.array(
                [
                    label[np.all(xy == v, axis=1)][0]
                    for v in self._opto_location_bregma_xy
                ],
                dtype=str,
            )
        raise ValueError("No known opto location data found (try `optoParams`)")

    @functools.cached_property
    def opto_location_name(self) -> npt.NDArray[np.str_]:
        if all(str(v).upper() in "ABCDEF" for v in self._opto_location_name):
            return np.array(
                [f"probe{str(v).upper()}" for v in self._opto_location_name], dtype=str
            )
        return self._opto_location_name

    @functools.cached_property
    def _opto_location_index(self) -> npt.NDArray[np.int32]:
        """0-indexed target location for optogenetic inactivation during the trial"""
        # TODO
        if not any(self.is_opto):
            return np.full(self._len, np.nan)
        return np.full(self._len, np.nan)

    @functools.cached_property
    def _opto_voltage(self) -> npt.NDArray[np.float64]:
        voltages = np.full(self._len, np.nan)
        if not any(self.is_opto):
            return voltages
        for idx in range(self._len):
            v = self._sam.trialOptoVoltage[idx]
            if v.shape:
                voltages[idx] = utils.safe_index(v, self._opto_params_index[idx])
                continue
            voltages[idx] = v
        return voltages

    @functools.cached_property
    def opto_power(self) -> npt.NDArray[np.float64]:
        if not any(self.is_opto):
            return np.full(self._len, np.nan)
        voltages = self._sam.trialOptoVoltage[self._opto_params_index]
        return np.where(
            np.isnan(voltages),
            np.nan,
            DynamicRoutingTask.TaskUtils.voltsToPower(
                self._hdf5["optoPowerCalibrationData"],
                voltages,
            ),
        )

    @functools.cached_property
    def _galvo_voltage_x(self) -> npt.NDArray[np.float64]:
        if not any(self.is_opto):
            return np.full(self._len, np.nan)
        return np.array([voltage[0] for voltage in self._sam.trialGalvoVoltage])

    @functools.cached_property
    def _galvo_voltage_y(self) -> npt.NDArray[np.float64]:
        if not any(self.is_opto):
            return np.full(self._len, np.nan)
        return np.array([voltage[1] for voltage in self._sam.trialGalvoVoltage])

    @functools.cached_property
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

    @functools.cached_property
    def is_response(self) -> npt.NDArray[np.bool_]:
        """the subject licked one or more times during the response window"""
        return self._sam.trialResponse

    @functools.cached_property
    def is_correct(self) -> npt.NDArray[np.bool_]:
        """the subject acted correctly in the response window, according to its
        training.

        - includes correct reject for catch trials
        """
        return (
            self.is_hit | self.is_correct_reject | (self.is_catch & ~self.is_response)
        )

    @functools.cached_property
    def is_incorrect(self) -> npt.NDArray[np.bool_]:
        """the subject acted incorrectly in the response window, according to its
        training.

        - includes false alarm for catch trials
        """
        return self.is_miss | self.is_false_alarm | (self.is_catch & self.is_response)

    @functools.cached_property
    def is_hit(self) -> npt.NDArray[np.bool_]:
        """the subject responded in a GO trial"""
        return self.is_response & self.is_go

    @functools.cached_property
    def is_false_alarm(self) -> npt.NDArray[np.bool_]:
        """the subject responded in a NOGO trial

        - excludes catch trials
        """
        return self.is_response & self.is_nogo

    @functools.cached_property
    def is_correct_reject(self) -> npt.NDArray[np.bool_]:
        """the subject did not respond in a NOGO trial

        - excludes catch trials
        """
        return ~self.is_response & self.is_nogo

    @functools.cached_property
    def is_miss(self) -> npt.NDArray[np.bool_]:
        """the subject did not respond in a GO trial"""
        return ~self.is_response & self.is_go

    @functools.cached_property
    def is_go(self) -> npt.NDArray[np.bool_]:
        """condition in which the subject should respond.

        - target stim presented in rewarded context block
        """
        return self._sam.goTrials

    @functools.cached_property
    def is_nogo(self) -> npt.NDArray[np.bool_]:
        """condition in which the subject should not respond.

        - non-target stim presented in any context block
        - target stim presented in non-rewarded context block
        - excludes catch trials
        """
        return self._sam.nogoTrials

    @functools.cached_property
    def is_rewarded(self) -> npt.NDArray[np.bool_]:
        """the subject received a reward.

        - includes non-contingent rewards
        """
        return self._sam.trialRewarded

    @functools.cached_property
    def is_noncontingent_reward(self) -> npt.NDArray[np.bool_]:
        """the subject received a reward that did not depend on its response"""
        return self._sam.autoRewarded

    @functools.cached_property
    def is_contingent_reward(self) -> npt.NDArray[np.bool_]:
        """the subject received a reward for a correct response in a GO trial"""
        return self.is_rewarded & self.is_hit

    @functools.cached_property
    def is_reward_scheduled(self) -> npt.NDArray[np.bool_]:
        """a non-contingent reward was scheduled to occur, regardless of
        whether it was received.

        - subject may have responded correctly and received contingent reward
          instead
        """
        return self._sam.autoRewardScheduled

    @functools.cached_property
    def is_aud_stim(self) -> npt.NDArray[np.bool_]:
        """an auditory stimulus was presented.

        - includes target and non-target stimuli
        - includes rewarded and non-rewarded contexts
        - excludes catch trials (no stimulus)
        """
        return np.isin(self._sam.trialStim, self._aud_stims)

    @functools.cached_property
    def is_vis_stim(self) -> npt.NDArray[np.bool_]:
        """a visual stimulus was presented.

        - includes target and non-target stimuli
        - includes rewarded and non-rewarded contexts
        - excludes catch trials (no stimulus)
        """
        return np.isin(self._sam.trialStim, self._vis_stims)

    @functools.cached_property
    def is_catch(self) -> npt.NDArray[np.bool_]:
        """no stimuli were presented"""
        return np.isin(self._sam.trialStim, "catch")

    @functools.cached_property
    def is_aud_target(self) -> npt.NDArray[np.bool_]:
        """an auditory stimulus was presented that the subject should respond
        to only in a specific context"""
        return np.isin(self._sam.trialStim, self._aud_targets)

    @functools.cached_property
    def is_vis_target(self) -> npt.NDArray[np.bool_]:
        """a visual stimulus was presented that the subject should respond to
        only in a specific context"""
        return np.isin(self._sam.trialStim, self._vis_targets)

    @functools.cached_property
    def is_aud_nontarget(self) -> npt.NDArray[np.bool_]:
        """an auditory stimulus was presented that the subject should never respond to"""
        return np.isin(self._sam.trialStim, self._aud_nontargets)

    @functools.cached_property
    def is_vis_nontarget(self) -> npt.NDArray[np.bool_]:
        """a visual stimulus was presented that the subject should never respond to"""
        return np.isin(self._sam.trialStim, self._vis_nontargets)

    @functools.cached_property
    def is_vis_context(self) -> npt.NDArray[np.bool_]:
        """visual target stimuli are rewarded"""
        return np.isin(self._trial_rewarded_stim_name, self._vis_stims)

    @functools.cached_property
    def is_aud_context(self) -> npt.NDArray[np.bool_]:
        """auditory target stimuli are rewarded"""
        return np.isin(self._trial_rewarded_stim_name, self._aud_stims)

    @functools.cached_property
    def is_context_switch(self) -> npt.NDArray[np.bool_]:
        """the first trial with a stimulus after a change in context"""
        return np.isin(self.trial_index_in_block, 0) & ~np.isin(self.block_index, 0)

    @functools.cached_property
    def is_repeat(self) -> npt.NDArray[np.bool_]:
        """the trial is a repetition of the previous trial, due to a
        miss"""
        return self._sam.trialRepeat

    @functools.cached_property
    def is_opto(self) -> npt.NDArray[np.bool_]:
        """optogenetic inactivation was applied during the trial"""
        return ~np.isnan(self.opto_start_time)

    """
    @functools.cached_property
    def is_(self) -> npt.NDArray[np.bool_]:
        """ """
        return
    """
