from __future__ import annotations

import datetime
import io
import logging
import pickle
from collections.abc import Iterable, Mapping
from typing import Any, Callable, Literal, NamedTuple, Union

import h5py
import npc_session
import numba
import numpy as np
import numpy.typing as npt
import upath
from DynamicRoutingTask.TaskUtils import getOptoPulseWaveform, makeSoundArray
from typing_extensions import TypeAlias

import npc_sessions.utils as utils

StimPathOrDataset: TypeAlias = Union[utils.PathLike, h5py.File, Mapping]

logger = logging.getLogger(__name__)


def get_stim_data(stim_path: StimPathOrDataset, **kwargs) -> h5py.File | dict:
    if isinstance(stim_path, h5py.File):
        return stim_path
    path = utils.from_pathlike(stim_path)
    if path.suffix in (".hdf5", ".h5"):
        return get_h5_stim_data(path, **kwargs)
    if path.suffix == ".pkl":
        return get_pkl_stim_data(path, **kwargs)
    raise ValueError(f"Unknown stim file type: {path}")


def get_h5_stim_data(stim_path: StimPathOrDataset, **kwargs) -> h5py.File:
    if isinstance(stim_path, h5py.File):
        return stim_path
    kwargs.setdefault("mode", "r")
    return h5py.File(io.BytesIO(utils.from_pathlike(stim_path).read_bytes()), **kwargs)


def get_pkl_stim_data(stim_path: StimPathOrDataset, **kwargs) -> dict:
    if isinstance(stim_path, Mapping):
        return dict(stim_path)
    kwargs.setdefault("encoding", "latin1")
    return pickle.loads(utils.from_pathlike(stim_path).read_bytes())


class Waveform(NamedTuple):
    samples: npt.NDArray[np.float64]
    sampling_rate: float


class StimPresentation(NamedTuple):
    trial_idx: int
    waveform: Waveform
    onset_sample_on_nidaq: int
    offset_sample_on_nidaq: int
    trigger_time_on_sync: float

    @property
    def duration(self) -> float:
        return len(self.waveform.samples) / self.waveform.sampling_rate


class StimRecording(NamedTuple):
    presentation: StimPresentation
    latency: float

    @property
    def onset_time_on_sync(self) -> float:
        return self.presentation.trigger_time_on_sync + self.latency

    @property
    def offset_time_on_sync(self) -> float:
        return self.onset_time_on_sync + self.presentation.duration


def get_audio_waveforms_from_stim_file(
    stim_file_or_dataset: StimPathOrDataset,
) -> tuple[Waveform, ...]:
    stim_data = get_h5_stim_data(stim_file_or_dataset)

    nTrials = get_num_trials(stim_data)
    soundSampleRate: int = stim_data["soundSampleRate"][()]

    if len(stim_data["trialSoundArray"][:]) > 0:
        trialSoundArray: list[npt.NDArray] = stim_data["trialSoundArray"][:nTrials]
        waveforms = []
        for trialnum in range(0, len(trialSoundArray)):
            waveforms.append(
                Waveform(
                    samples=trialSoundArray[trialnum], sampling_rate=soundSampleRate
                )
            )
        return tuple(waveforms)

    print("trialSoundArray empty; regenerating sound arrays")
    return regenerate_sound_array(stim_data)


def regenerate_sound_array(
    stim_file_or_dataset: StimPathOrDataset,
) -> tuple[Waveform, ...]:
    stim_data = get_h5_stim_data(stim_file_or_dataset)

    waveforms = []
    nTrials = get_num_trials(stim_data)
    trialSoundDur = stim_data["trialSoundDur"][:nTrials]
    trialSoundFreq = stim_data["trialSoundFreq"][:nTrials]
    trialSoundSeed = stim_data["trialSoundSeed"][:nTrials]
    trialSoundType = stim_data["trialSoundType"][:nTrials]
    trialSoundVolume = stim_data["trialSoundVolume"][:nTrials]
    trialSoundAM = stim_data["trialSoundAM"][:nTrials]
    soundSampleRate = stim_data["soundSampleRate"][()]
    soundHanningDur = stim_data["soundHanningDur"][()]

    if len(trialSoundDur) == 0:
        raise IndexError(
            f"trialSoundDur is empty - no sound waveforms to generate from {stim_file_or_dataset}"
        )

    for trialnum in range(0, nTrials):
        if trialSoundType[trialnum].decode() == "":
            soundArray = np.array([])
        else:
            if trialSoundType[trialnum].decode() == "tone":
                # accounts for a quirk of how the trial sound frequencies are saved
                freq = trialSoundFreq[trialnum][0]
            else:
                freq = trialSoundFreq[trialnum]

            soundArray = makeSoundArray(
                soundType=trialSoundType[trialnum].decode(),
                sampleRate=soundSampleRate,
                dur=trialSoundDur[trialnum],
                hanningDur=soundHanningDur,
                vol=trialSoundVolume[trialnum],
                freq=freq,
                AM=trialSoundAM[trialnum],
                seed=trialSoundSeed[trialnum],
            )

        waveform = Waveform(samples=soundArray, sampling_rate=soundSampleRate)

        waveforms.append(waveform)

    return tuple(waveforms)


def generate_opto_waveforms_from_stim_file(
    stim_file_or_dataset: StimPathOrDataset,
) -> tuple[Waveform, ...]:
    stim_data = get_h5_stim_data(stim_file_or_dataset)

    waveforms = []
    nTrials = get_num_trials(stim_data)
    trialOptoDelay = stim_data["trialOptoDelay"][:]
    trialOptoDur = stim_data["trialOptoDur"][:]
    trialOptoOffRamp = stim_data["trialOptoOffRamp"][:]
    trialOptoOnRamp = stim_data["trialOptoOnRamp"][:]
    trialOptoSinFreq = stim_data["trialOptoSinFreq"][:]
    trialOptoVoltage = stim_data["trialOptoVoltage"][:]
    optoOffsetVoltage = stim_data["optoOffsetVoltage"]["laser_488"][()]
    if "optoSampleRate" in stim_data.keys():
        optoSampleRate = stim_data["optoSampleRate"][()]
    else:
        optoSampleRate = 2000

    if len(trialOptoDur) == 0:
        raise IndexError(
            f"trialOptoDur is empty - no opto waveforms to generate from {stim_file_or_dataset}"
        )

    for trialnum in range(0, nTrials):
        if np.isnan(trialOptoDur[trialnum]) is True:
            optoArray = np.array([])
        else:
            optoArray = getOptoPulseWaveform(
                sampleRate=optoSampleRate,
                amp=trialOptoVoltage[trialnum],
                dur=trialOptoDur[trialnum],
                delay=trialOptoDelay[trialnum],
                freq=trialOptoSinFreq[trialnum],
                onRamp=trialOptoOnRamp[trialnum],
                offRamp=trialOptoOffRamp[trialnum],
                offset=optoOffsetVoltage,
            )
        waveform = Waveform(samples=optoArray, sampling_rate=optoSampleRate)
        waveforms.append(waveform)

    return tuple(waveforms)


def get_waveforms_from_stim_file(
    stim_file_or_dataset: StimPathOrDataset,
    waveform_type: Literal["sound", "audio", "opto"],
) -> tuple[Waveform, ...]:
    if any(s in waveform_type for s in ("sound", "audio")):
        return get_audio_waveforms_from_stim_file(stim_file_or_dataset)
    if "opto" in waveform_type:
        return generate_opto_waveforms_from_stim_file(stim_file_or_dataset)
    raise ValueError(
        f"Unexpected value: {waveform_type = }. Should be 'sound' or 'opto'."
    )


@numba.njit(parallel=True)
def _xcorr(v, w, t) -> float:
    c = np.correlate(v, w)
    return t[np.argmax(c)]


def xcorr(
    nidaq_data: npt.NDArray[np.int16],
    nidaq_timing: utils.EphysTimingInfoOnSync,
    nidaq_channel: int,
    presentations: Iterable[StimPresentation],
    padding_sec: float = 0.15,
    **kwargs,
) -> tuple[StimRecording, ...]:
    recordings: list[StimRecording] = []
    padding_samples = int(padding_sec * nidaq_timing.sampling_rate)
    for pres_idx, presentation in enumerate(presentations):
        print(f"{pres_idx+1}/{len(tuple(presentations))}\r", flush=True)
        nidaq_times = (
            np.arange(
                (presentation.offset_sample_on_nidaq + padding_samples)
                - (presentation.onset_sample_on_nidaq - padding_samples)
            )
            / (nidaq_timing.sampling_rate)
            - padding_sec
        )
        nidaq_samples = nidaq_data[
            presentation.onset_sample_on_nidaq
            - padding_samples : presentation.offset_sample_on_nidaq
            + padding_samples,
            nidaq_channel,
        ]
        waveform_times = np.arange(
            0, presentation.duration, 1 / presentation.waveform.sampling_rate
        )
        interp_waveform_times = np.arange(
            0,
            presentation.duration,
            1 / nidaq_timing.sampling_rate,
        )
        interp_waveform_samples = np.interp(
            interp_waveform_times, waveform_times, presentation.waveform.samples
        )

        recordings.append(
            StimRecording(
                presentation=presentation,
                latency=_xcorr(nidaq_samples, interp_waveform_samples, nidaq_times),
            )
        )

        # to verify:
        """
        import matplotlib.pyplot as plt
        norm_nidaq_samples = (nidaq_samples - np.mean(nidaq_samples)) / max(abs((nidaq_samples - np.mean(nidaq_samples))))
        norm_waveform_samples = (interp_waveform_samples - np.mean(interp_waveform_samples)) / max(abs((interp_waveform_samples - np.mean(interp_waveform_samples))))
        plt.plot(nidaq_times, norm_nidaq_samples)
        plt.plot(interp_waveform_times + recordings[-1].latency, norm_waveform_samples / max(abs(norm_waveform_samples)))
        plt.title(f"{recordings[-1].latency = }")
        """

    return tuple(recordings)


def get_stim_latencies_from_nidaq_recording(
    stim_file_or_dataset: StimPathOrDataset,
    sync: utils.SyncPathOrDataset,
    recording_dirs: Iterable[upath.UPath],
    waveform_type: Literal["sound", "audio", "opto"],
    nidaq_device_name: str | None = None,
    correlation_method: Callable[
        [
            npt.NDArray[np.int16],
            utils.EphysTimingInfoOnSync,
            int,
            Iterable[StimPresentation],
        ],
        tuple[StimRecording, ...],
    ] = xcorr,
    correlation_method_kwargs: dict[str, Any] | None = None,
) -> tuple[StimRecording, ...]:
    if not nidaq_device_name:
        nidaq_device = utils.get_pxi_nidaq_device(recording_dirs)
    else:
        nidaq_device = next(
            utils.get_ephys_timing_on_pxi(
                recording_dirs=recording_dirs, only_devices_including=nidaq_device_name
            )
        )

    nidaq_timing: utils.EphysTimingInfoOnSync = next(
        utils.get_ephys_timing_on_sync(
            sync=sync,
            recording_dirs=recording_dirs,
            devices=(nidaq_device,),
        )
    )

    nidaq_data = utils.get_pxi_nidaq_data(
        *recording_dirs,
        device_name=nidaq_device.name,
    )

    if (waveform_type == "audio") | (waveform_type == "sound"):
        nidaq_channel = 1
    stim = get_h5_stim_data(stim_file_or_dataset)

    vsyncs = assert_stim_times(
        get_stim_frame_times(stim, sync=sync, frame_time_type="vsync")[stim]
    )

    num_trials = get_num_trials(stim)

    trigger_frames: npt.NDArray[np.int16] = (
        stim.get("trialStimStartFrame") or stim.get("stimStartFrame")
    )[:num_trials]

    presentations = []

    waveforms = get_waveforms_from_stim_file(stim, waveform_type)
    for idx, waveform in enumerate(waveforms):
        if not any(waveform.samples):
            continue
        trigger_time_on_sync: float = vsyncs[trigger_frames[idx]]
        trigger_time_on_pxi_nidaq = trigger_time_on_sync - nidaq_timing.start_time
        duration = len(waveform.samples) / waveform.sampling_rate
        onset_sample_on_pxi_nidaq = round(
            trigger_time_on_pxi_nidaq * nidaq_timing.sampling_rate
        )
        offset_sample_on_pxi_nidaq = round(
            (trigger_time_on_pxi_nidaq + duration) * nidaq_timing.sampling_rate
        )
        # padding should be done by correlation method, when reading data

        presentations.append(
            StimPresentation(
                trial_idx=idx,
                waveform=waveform,
                onset_sample_on_nidaq=onset_sample_on_pxi_nidaq,
                offset_sample_on_nidaq=offset_sample_on_pxi_nidaq,
                trigger_time_on_sync=trigger_time_on_sync,
            )
        )

    # run the correlation of presentations with nidaq data
    recordings = correlation_method(
        nidaq_data,
        nidaq_timing,
        nidaq_channel,
        presentations,
        **(correlation_method_kwargs or {}),
    )

    return recordings


def assert_stim_times(result: Exception | npt.NDArray) -> npt.NDArray:
    """Raise exception if result is an exception, otherwise return result"""
    if isinstance(result, Exception):
        raise result from None
    return result


def get_stim_frame_times(
    *stim_paths: utils.StimPathOrDataset,
    sync: utils.SyncPathOrDataset,
    frame_time_type: Literal["display_time", "vsync"] = "display_time",
) -> dict[utils.StimPathOrDataset, Exception | npt.NDArray[np.float64]]:
    """
    - keys are the stim paths provided as inputs

    >>> bad_stim = 's3://aind-ephys-data/ecephys_670248_2023-08-02_11-30-53/behavior/DynamicRouting1_670248_20230802_120703.hdf5'
    >>> good_stim_1 = 's3://aind-ephys-data/ecephys_670248_2023-08-02_11-30-53/behavior/Spontaneous_670248_20230802_114611.hdf5'
    >>> good_stim_2 = 's3://aind-ephys-data/ecephys_670248_2023-08-02_11-30-53/behavior/SpontaneousRewards_670248_20230802_130736.hdf5'
    >>> sync = 's3://aind-ephys-data/ecephys_670248_2023-08-02_11-30-53/behavior/20230802T113053.h5'

    Returns the frame times for each stim file based on start time and number
    of frames - can be provided in any order:
    >>> frame_times = get_stim_frame_times(good_stim_2, good_stim_1, sync=sync)
    >>> len(frame_times[good_stim_1])
    36000

    Returns Exception if the stim file can't be opened, or it has no frames.
    Should be used with `assert_stim_times` to raise a possible exception:
    >>> frame_times = get_stim_frame_times(bad_stim, sync=sync)
    >>> assert_stim_times(frame_times[bad_stim])
    Traceback (most recent call last):
    ...
    OSError: Unable to open file (bad object header version number)
    """

    # load sync file once
    sync_data = utils.get_sync_data(sync)
    # get vsync_times_in_blocks
    if "vsync" in frame_time_type:
        frame_times_in_blocks = sync_data.vsync_times_in_blocks
    # get frame_display_time_blocks
    elif "display" in frame_time_type:
        frame_times_in_blocks = sync_data.frame_display_time_blocks
    else:
        raise ValueError(f"Unexpected value: {frame_time_type = }")
    # get num frames in each block
    n_frames_per_block = np.asarray([len(x) for x in frame_times_in_blocks])
    # get first frame time in each block
    first_frame_per_block = np.asarray([x[0] for x in frame_times_in_blocks])

    stim_frame_times: dict[
        utils.StimPathOrDataset, Exception | npt.NDArray[np.float64]
    ] = {}

    exception: Exception | None = None
    # loop through stim files
    for stim_path in stim_paths:
        # load each stim file once - may fail if file wasn't saved correctly
        try:
            stim_data = get_h5_stim_data(stim_path)
        except OSError as exc:
            exception = exc
            stim_frame_times[stim_path] = exception
            continue

        # get number of frames
        n_stim_frames = get_total_stim_frames(stim_data)
        if n_stim_frames == 0:
            exception = ValueError(f"No frames found in {stim_path = }")
            stim_frame_times[stim_path] = exception
            continue

        # get first stimulus frame relative to sync start time
        stim_start_time: datetime.datetime = get_stim_start_time(stim_data)
        if abs((stim_start_time - sync_data.start_time).days > 0):
            logger.error(
                f"Skipping {stim_path =}, sync data is from a different day: {stim_start_time = }, {sync_data.start_time = }"
            )
            continue

        # try to match to vsyncs by start time
        stim_start_time_on_sync = (stim_start_time - sync_data.start_time).seconds
        matching_block = np.argmin(abs(first_frame_per_block - stim_start_time_on_sync))
        num_frames_match: bool = n_stim_frames == n_frames_per_block[matching_block]
        if not num_frames_match:
            frame_diff = n_stim_frames - n_frames_per_block[matching_block]
            exception = IndexError(
                f"Closest match with {stim_path} has a mismatch of {frame_diff = }"
            )
            stim_frame_times[stim_path] = exception
            continue

        stim_frame_times[stim_path] = frame_times_in_blocks[matching_block]
    sorted_keys = sorted(stim_frame_times.keys(), key=lambda x: 0 if isinstance(stim_frame_times[x], Exception) else stim_frame_times[x][0])  # type: ignore[index]
    return {k: stim_frame_times[k] for k in sorted_keys}


def get_num_trials(
    stim_path_or_data: utils.PathLike | h5py.File,
) -> int:
    stim_data = get_h5_stim_data(stim_path_or_data)
    return len((stim_data.get("trialEndFrame") or stim_data.get("trialSoundArray"))[:])


def get_stim_start_time(
    stim_path_or_data: utils.PathLike | h5py.File,
) -> datetime.datetime:
    """Absolute datetime of the first frame, according to the stim file"""
    # TODO make compatible with pkl files
    stim_data = get_h5_stim_data(stim_path_or_data)
    # get stim start time & convert to datetime
    return npc_session.DatetimeRecord(stim_data["startTime"][()].decode()).dt


def get_total_stim_frames(stim_path_or_data: utils.PathLike | h5py.File) -> int:
    # TODO make compatible with pkl files
    stim_data = get_h5_stim_data(stim_path_or_data)
    frame_intervals = stim_data["frameIntervals"][:]
    if len(frame_intervals) == 0:
        return 0
    return len(frame_intervals) + 1


def get_stim_duration(stim_path_or_data: utils.PathLike | h5py.File) -> float:
    # TODO make compatible with pkl files
    stim_data = get_h5_stim_data(stim_path_or_data)
    return np.sum(stim_data["frameIntervals"][:])


if __name__ == "__main__":
    import doctest

    import dotenv

    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
