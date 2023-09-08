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

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sampling_rate

    @property
    def timestamps(self) -> npt.NDArray[np.float64]:
        return np.arange(0, self.duration, 1 / self.sampling_rate)


class StimPresentation(NamedTuple):
    trial_idx: int
    waveform: Waveform
    trigger_time_on_sync: float

    @property
    def duration(self) -> float:
        return self.waveform.duration


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
) -> tuple[Waveform | None, ...]:
    stim_data = get_h5_stim_data(stim_file_or_dataset)

    nTrials = get_num_trials(stim_data)
    soundSampleRate: int = stim_data["soundSampleRate"][()]
    trialSoundArray: list[npt.NDArray] | None = stim_data.get("trialSoundArray", None)

    if trialSoundArray and len(trialSoundArray) > 0:
        waveforms: list[Waveform | None] = [None] * nTrials
        for trialnum in range(nTrials):
            if any(trialSoundArray[trialnum]):
                waveforms[trialnum] = Waveform(
                    samples=trialSoundArray[trialnum], sampling_rate=soundSampleRate
                )
        return tuple(waveforms)

    print("trialSoundArray empty; regenerating sound arrays")
    return regenerate_sound_array(stim_data)


def regenerate_sound_array(
    stim_file_or_dataset: StimPathOrDataset,
) -> tuple[Waveform | None, ...]:
    stim_data = get_h5_stim_data(stim_file_or_dataset)

    nTrials = get_num_trials(stim_data)
    waveforms: list[Waveform | None] = [None] * nTrials
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
            waveforms[trialnum] = None
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

            waveforms[trialnum] = Waveform(
                samples=soundArray, sampling_rate=soundSampleRate
            )

    return tuple(waveforms)


def generate_opto_waveforms_from_stim_file(
    stim_file_or_dataset: StimPathOrDataset,
) -> tuple[Waveform | None, ...]:
    stim_data = get_h5_stim_data(stim_file_or_dataset)

    nTrials = get_num_trials(stim_data)
    if "trialOptoDelay" in stim_data:
        trialOptoDelay = stim_data["trialOptoDelay"][:nTrials]
    elif "optoDelay" in stim_data:
        trialOptoDelay = np.ones(nTrials) * stim_data["optoDelay"][()]
    else:
        trialOptoDelay = np.zeros(nTrials)

    if "trialOptoOffRamp" in stim_data:
        trialOptoOffRamp = stim_data["trialOptoOffRamp"][:nTrials]
    elif "optoOffRamp" in stim_data:
        trialOptoOffRamp = np.ones(nTrials) * stim_data["optoOffRamp"]
    else:
        trialOptoOffRamp = np.zeros(nTrials)

    if "trialOptoOnRamp" in stim_data:
        trialOptoOnRamp = stim_data["trialOptoOnRamp"][:nTrials]
    elif "optoOnRamp" in stim_data:
        trialOptoOnRamp = np.ones(nTrials) * stim_data["optoOnRamp"]
    else:
        trialOptoOnRamp = np.zeros(nTrials)

    if "trialOptoSinFreq" in stim_data:
        trialOptoSinFreq = stim_data["trialOptoSinFreq"][:nTrials]
    elif "optoSinFreq" in stim_data:
        trialOptoSinFreq = np.ones(nTrials) * stim_data["optoSinFreq"]
    else:
        trialOptoSinFreq = np.zeros(nTrials)

    trialOptoDur = stim_data["trialOptoDur"][:nTrials]
    trialOptoVoltage = stim_data["trialOptoVoltage"][:nTrials]

    if "optoSampleRate" in stim_data.keys():
        optoSampleRate = stim_data["optoSampleRate"][()]
    else:
        optoSampleRate = 2000

    if len(trialOptoDur) == 0:
        raise IndexError(
            f"trialOptoDur is empty - no opto waveforms to generate from {stim_file_or_dataset}"
        )

    waveforms: list[Waveform | None] = [None] * nTrials
    for trialnum in range(0, nTrials):
        if np.isnan(trialOptoDur[trialnum]) is True:
            continue
        optoArray = getOptoPulseWaveform(
            sampleRate=optoSampleRate,
            amp=trialOptoVoltage[trialnum],
            dur=trialOptoDur[trialnum],
            delay=trialOptoDelay[trialnum],
            freq=trialOptoSinFreq[trialnum],
            onRamp=trialOptoOnRamp[trialnum],
            offRamp=trialOptoOffRamp[trialnum],
        )
        waveforms[trialnum] = Waveform(samples=optoArray, sampling_rate=optoSampleRate)

    return tuple(waveforms)


def get_waveforms_from_stim_file(
    stim_file_or_dataset: StimPathOrDataset,
    waveform_type: Literal["sound", "audio", "opto"],
) -> tuple[Waveform | None, ...]:
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
    presentations: Iterable[StimPresentation | None],
    padding_sec: float = 0.15,
    **kwargs,
) -> tuple[StimRecording | None, ...]:
    num_presentations = len(tuple(presentations))
    recordings: list[StimRecording | None] = [None] * num_presentations
    padding_samples = int(padding_sec * nidaq_timing.sampling_rate)
    for idx, presentation in enumerate(presentations):
        print(f"{idx+1}/{num_presentations}\r", flush=True)
        if presentation is None:
            continue
        trigger_time_on_nidaq = (
            presentation.trigger_time_on_sync - nidaq_timing.start_time
        )
        onset_sample_on_nidaq = round(
            trigger_time_on_nidaq * nidaq_timing.sampling_rate
        )
        offset_sample_on_nidaq = round(
            (trigger_time_on_nidaq + presentation.duration) * nidaq_timing.sampling_rate
        )
        nidaq_times = (
            np.arange(
                (offset_sample_on_nidaq + padding_samples)
                - (onset_sample_on_nidaq - padding_samples)
            )
            / (nidaq_timing.sampling_rate)
            - padding_sec
        )
        nidaq_samples = nidaq_data[
            onset_sample_on_nidaq
            - padding_samples : offset_sample_on_nidaq
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
            Iterable[StimPresentation | None],
        ],
        tuple[StimRecording | None, ...],
    ] = xcorr,
    correlation_method_kwargs: dict[str, Any] | None = None,
) -> tuple[StimRecording | None, ...]:
    sync = utils.get_sync_data(sync)
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

    nidaq_channel = get_nidaq_channel_for_stim_onset(
        waveform_type, date=sync.start_time.date()
    )

    stim = get_h5_stim_data(stim_file_or_dataset)

    vsyncs = assert_stim_times(
        get_stim_frame_times(stim, sync=sync, frame_time_type="vsync")[stim]
    )

    num_trials = get_num_trials(stim)
    trigger_frames = get_stim_trigger_frames(stim)

    presentations: list[StimPresentation | None] = [None] * num_trials
    waveforms = get_waveforms_from_stim_file(stim, waveform_type)
    for idx, waveform in enumerate(waveforms):
        if waveform is None:
            continue
        # padding should be done by correlation method, when reading data
        presentations[idx] = StimPresentation(
            trial_idx=idx,
            waveform=waveform,
            trigger_time_on_sync=float(vsyncs[trigger_frames[idx]]),
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


def get_stim_latencies_from_sync(
    stim_file_or_dataset: StimPathOrDataset,
    sync: utils.SyncPathOrDataset,
    waveform_type: Literal["sound", "audio", "opto"],
    line_index_or_label: int | str | None = None,
) -> tuple[StimRecording | None, ...]:
    """
    >>> stim = 's3://aind-ephys-data/ecephys_668755_2023-08-31_12-33-31/behavior/DynamicRouting1_668755_20230831_131418.hdf5'
    >>> sync = 's3://aind-ephys-data/ecephys_668755_2023-08-31_12-33-31/behavior/20230831T123331.h5'
    >>> latencies = get_stim_latencies_from_sync(stim, sync, waveform_type='sound')
    >>> assert 0 < next(_.latency for _ in latencies if _ is not None) < 0.1

    """
    stim = get_h5_stim_data(stim_file_or_dataset)
    sync = utils.get_sync_data(sync)
    if not line_index_or_label:
        line_index_or_label = get_sync_line_for_stim_onset(
            waveform_type=waveform_type, date=sync.start_time.date()
        )
    vsyncs = assert_stim_times(
        get_stim_frame_times(stim, sync=sync, frame_time_type="vsync")[stim]
    )
    trigger_times = tuple(
        vsyncs[idx] if idx else None
        for idx in get_stim_trigger_frames(stim, stim_type=waveform_type)
    )
    stim_onsets = sync.get_rising_edges(line_index_or_label, units="seconds")
    recordings: list[StimRecording | None] = [None] * len(trigger_times)
    for idx, (trigger_time, waveform) in enumerate(
        zip(trigger_times, get_waveforms_from_stim_file(stim, waveform_type))
    ):
        if waveform is None:
            continue
        assert trigger_time
        onset_following_trigger = stim_onsets[
            np.searchsorted(stim_onsets, trigger_time, side="right")
        ]
        recordings[idx] = StimRecording(
            presentation=StimPresentation(
                trial_idx=idx,
                waveform=waveform,
                trigger_time_on_sync=float(trigger_time),
            ),
            latency=onset_following_trigger - trigger_time,
        )
    return tuple(recordings)


def get_sync_line_for_stim_onset(
    waveform_type: str | Literal["sound", "audio", "opto"],
    date: datetime.date | None = None,
) -> int:
    FIRST_SOUND_ON_SYNC_DATE = datetime.date(2023, 8, 31)
    if any(label in waveform_type for label in ("aud", "sound")):
        if date and date < FIRST_SOUND_ON_SYNC_DATE:
            raise ValueError(
                f"Sound only recorded on sync since {FIRST_SOUND_ON_SYNC_DATE.isoformat()}: {date = }"
            )
        return 1
    elif "opto" in waveform_type:
        return 11
    else:
        raise ValueError(f"Unexpected value: {waveform_type = }")


def get_nidaq_channel_for_stim_onset(
    waveform_type: str | Literal["sound", "audio", "opto"],
    date: datetime.date | None = None,
) -> int:
    if any(label in waveform_type for label in ("aud", "sound")):
        return 1
    elif "opto" in waveform_type:
        return 5
    else:
        raise ValueError(f"Unexpected value: {waveform_type = }")


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
    """
    >>> get_num_trials('s3://aind-ephys-data/ecephys_668755_2023-08-31_12-33-31/behavior/DynamicRouting1_668755_20230831_131418.hdf5')
    524
    """
    stim_data = get_h5_stim_data(stim_path_or_data)
    return len(stim_data.get("trialEndFrame") or stim_data.get("trialOptoOnsetFrame"))


def get_stim_start_time(
    stim_path_or_data: utils.PathLike | h5py.File,
) -> datetime.datetime:
    """Absolute datetime of the first frame, according to the stim file
    >>> get_stim_start_time('s3://aind-ephys-data/ecephys_668755_2023-08-31_12-33-31/behavior/DynamicRouting1_668755_20230831_131418.hdf5')
    datetime.datetime(2023, 8, 31, 13, 14, 18)
    """
    # TODO make compatible with pkl files
    stim_data = get_h5_stim_data(stim_path_or_data)
    # get stim start time & convert to datetime
    return npc_session.DatetimeRecord(stim_data["startTime"][()].decode()).dt


def get_total_stim_frames(stim_path_or_data: utils.PathLike | h5py.File) -> int:
    """
    >>> get_total_stim_frames('s3://aind-ephys-data/ecephys_668755_2023-08-31_12-33-31/behavior/DynamicRouting1_668755_20230831_131418.hdf5')
    217261
    """
    # TODO make compatible with pkl files
    stim_data = get_h5_stim_data(stim_path_or_data)
    frame_intervals = stim_data["frameIntervals"][:]
    if len(frame_intervals) == 0:
        return 0
    return len(frame_intervals) + 1


def get_stim_duration(stim_path_or_data: utils.PathLike | h5py.File) -> float:
    """
    >>> get_stim_duration('s3://aind-ephys-data/ecephys_668755_2023-08-31_12-33-31/behavior/DynamicRouting1_668755_20230831_131418.hdf5')
    3647.0994503999827
    """
    # TODO make compatible with pkl files
    stim_data = get_h5_stim_data(stim_path_or_data)
    return np.sum(stim_data["frameIntervals"][:])


def get_stim_trigger_frames(
    stim_path_or_data: utils.PathLike | h5py.File,
    stim_type: str | Literal["opto"] = "stim",
) -> tuple[int | None, ...]:
    """Frame index of stim command being sent. len() == num trials.

    >>> path = 's3://aind-ephys-data/ecephys_668755_2023-08-31_12-33-31/behavior/DynamicRouting1_668755_20230831_131418.hdf5'
    >>> frames = get_stim_trigger_frames(path)
    >>> len(frames)
    524

    >>> frames = get_stim_trigger_frames(path, stim_type='opto')
    >>> len(frames)
    0
    """
    stim_data = get_h5_stim_data(stim_path_or_data)
    start_frames = (
        (stim_data.get("trialStimStartFrame") or stim_data.get("stimStartFrame"))
        if stim_type != "opto"
        else stim_data.get("trialOptoOnsetFrame")
    )[: get_num_trials(stim_data)]
    return tuple(
        int(v) if ~np.isnan(v) else None
        for v in utils.safe_index(start_frames, np.arange(len(start_frames)))
    )


if __name__ == "__main__":
    import doctest

    import dotenv

    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
