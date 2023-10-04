from __future__ import annotations

import dataclasses
import datetime
import enum
import functools
import io
import logging
import pickle
from collections.abc import Iterable, Mapping
from typing import Any, Callable, Literal, Protocol, Union

import DynamicRoutingTask.TaskUtils
import h5py
import npc_session
import numba
import numpy as np
import numpy.typing as npt
from typing_extensions import TypeAlias

import npc_sessions.utils as utils

StimPathOrDataset: TypeAlias = Union[utils.PathLike, h5py.File, Mapping]

logger = logging.getLogger(__name__)

FIRST_SOUND_ON_SYNC_DATE = datetime.date(2023, 8, 31)
"""Prior to this date, there's no sync line with "sound running" signal: need to
use NI-DAQ analog recording on OpenEphys PXI"""


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


class WaveformModality(enum.Enum):
    SOUND = enum.auto()
    OPTO = enum.auto()

    def __str__(self) -> str:
        return self.name.lower()

    @classmethod
    def from_factory(cls, s: Any) -> WaveformModality:
        if isinstance(s, WaveformModality):
            return s
        s = str(s)
        if any(
            label in s.lower()
            for label in ("sound", "audio", "tone", "noise", "acoustic")
        ):
            return cls.SOUND
        if any(label in s.lower() for label in ("opto", "optic")):
            return cls.OPTO
        raise ValueError(f"Could not determine modality from {s!r}")


class Waveform(Protocol):
    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def modality(self) -> WaveformModality:
        raise NotImplementedError

    @property
    def samples(self) -> npt.NDArray[np.float64]:
        raise NotImplementedError

    @property
    def sampling_rate(self) -> float:
        raise NotImplementedError

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sampling_rate

    @property
    def timestamps(self) -> npt.NDArray[np.float64]:
        return np.arange(0, self.duration, 1 / self.sampling_rate)

    def __eq__(self, other) -> bool:
        try:
            return (
                np.array_equal(self.samples, other.samples)
                and self.sampling_rate == other.sampling_rate
            )
        except (AttributeError, TypeError):
            return NotImplemented

    def __hash__(self) -> int:
        return hash((self.samples.tobytes(), self.sampling_rate))


class SimpleWaveform(Waveform):
    """
    >>> waveform = SimpleWaveform(name='test', modality='opto',sampling_rate=1, samples=np.array([1, 2, 3]))
    >>> waveform.duration
    3.0
    >>> waveform.timestamps
    array([0., 1., 2.])
    """

    def __init__(
        self,
        name: str,
        modality: str | WaveformModality,
        sampling_rate: float,
        samples: npt.NDArray[np.float64],
    ) -> None:
        self._samples = samples
        self._sampling_rate = sampling_rate
        self._name = name.replace(" ", "_")
        self._modality = WaveformModality.from_factory(modality)

    @property
    def name(self) -> str:
        return self._name

    @property
    def modality(self) -> WaveformModality:
        return self._modality

    @property
    def samples(self) -> npt.NDArray[np.float64]:
        return self._samples

    @property
    def sampling_rate(self) -> float:
        return self._sampling_rate


class LazyWaveform(Waveform):
    """Pass a function with args and kwargs used to generate the waveform
    on-demand, to avoid carrying round arrays in memory

    If the function is wrapped with functools.cache or similar, then we
    waveforms available immediately and stored only once for each unique
    parameter set.

    >>> waveform = LazyWaveform(name='test', modality='opto', sampling_rate=1, fn=lambda dtype: np.array([1, 2, 3], dtype=dtype), dtype=np.float64)
    >>> waveform.samples
    array([1., 2., 3.])
    >>> waveform.duration
    3.0
    >>> waveform.timestamps
    array([0., 1., 2.])
    """

    def __init__(
        self,
        name: str,
        modality: str | WaveformModality,
        sampling_rate: float,
        fn: Callable[..., npt.NDArray[np.float64]],
        **kwargs,
    ) -> None:
        self._name = name.replace(" ", "_")
        self._modality = WaveformModality.from_factory(modality)
        self._sampling_rate = sampling_rate
        self._fn = fn
        self._kwargs = kwargs
        for k, v in kwargs.items():
            if isinstance(v, np.ndarray):
                # convert to tuple to make hashable (for caching)
                self._kwargs[k] = tuple(v)

    @property
    def name(self) -> str:
        return self._name

    @property
    def modality(self) -> WaveformModality:
        return self._modality

    @property
    def sampling_rate(self) -> float:
        return self._sampling_rate

    @property
    def samples(self) -> npt.NDArray[np.float64]:
        return self._fn(**self._kwargs)


@dataclasses.dataclass(frozen=True, eq=True)
class StimPresentation:
    """
    Info about a waveform-stimulus when it was triggered: its sample values (ideal, not actual), the time it was
    sent, the expected duration, etc.

    >>> presentation = StimPresentation(
    ...     trial_idx=0,
    ...     waveform=SimpleWaveform(name="test", modality="sound", sampling_rate=1, samples=np.array([1, 2, 3])),
    ...     trigger_time_on_sync=0,
    ...     )
    >>> presentation.duration
    3.0
    """

    trial_idx: int
    waveform: Waveform
    trigger_time_on_sync: float

    @property
    def duration(self) -> float:
        return self.waveform.duration


class StimRecording(Protocol):
    """Timing information about a waveform-stimulus as recorded."""

    @property
    def name(self) -> str:
        """Descriptive name - will be used as key in `nwb.stimuli` dict"""
        raise NotImplementedError

    @property
    def modality(self) -> WaveformModality:
        raise NotImplementedError

    @property
    def onset_time_on_sync(self) -> float:
        raise NotImplementedError

    @property
    def offset_time_on_sync(self) -> float:
        raise NotImplementedError

    @property
    def latency(self) -> float | None:
        raise NotImplementedError


class FlexStimRecording(StimRecording):
    """Information about an actual recording of a waveform-stimulus, mainly for
    obtaining onset and offset of the stimulus.

    >>> presentation = StimPresentation(
    ...     trial_idx=0,
    ...     waveform=SimpleWaveform(
    ...         name="test",
    ...         modality="sound",
    ...         sampling_rate=1,
    ...         samples=np.array([1, 2, 3]),
    ...     ),
    ...     trigger_time_on_sync=0,
    ... )
    >>> recording = FlexStimRecording(presentation=presentation, latency=0.1)
    >>> recording.onset_time_on_sync
    0.1
    >>> recording.offset_time_on_sync
    3.1

    >>> recorded_waveform = SimpleWaveform(
    ...     name="test",
    ...     modality="sound",
    ...     sampling_rate=1,
    ...     samples=np.array([1, 2, 3]),
    ... )
    >>> recording = FlexStimRecording(waveform=recorded_waveform, onset_time_on_sync=0.1)
    >>> recording.offset_time_on_sync
    3.1
    """

    def __init__(
        self,
        name: str | None = None,
        modality: str | WaveformModality | None = None,
        presentation: StimPresentation | None = None,
        waveform: Waveform | None = None,
        trigger_time_on_sync: float | None = None,
        latency: float | None = None,
        onset_time_on_sync: float | None = None,
        offset_time_on_sync: float | None = None,
    ) -> None:
        if not (name or presentation or waveform):
            raise ValueError(
                "At least one of `name`, `presentation`, `waveform` must be provided"
            )
        if not (presentation or waveform):
            raise ValueError(
                "At least one of `presentation`, `waveform` must be provided"
            )
        if not (presentation or waveform):
            raise ValueError(
                "At least one of `presentation` or `waveform` must be provided"
            )
        if latency is None and onset_time_on_sync is None:
            raise ValueError(
                "At least one of `latency` or `onset_time_on_sync` must be provided"
            )
        if not (presentation or waveform) and offset_time_on_sync is None:
            raise ValueError(
                "At least one of `presentation`, `waveform`, `offset_time_on_sync` must be provided"
            )
        # minimum attrs:
        self.presentation = presentation
        self.waveform = waveform
        self.trigger_time_on_sync = trigger_time_on_sync
        self._name = None if name is None else name.replace(" ", "_")
        self._modality = (
            None if modality is None else WaveformModality.from_factory(modality)
        )

        # attrs that can potentially be derived from other attrs:
        self._latency = latency
        self._onset_time_on_sync = onset_time_on_sync
        self._offset_time_on_sync = offset_time_on_sync

    @property
    def name(self) -> str:
        if self._name is not None:
            return self._name
        if self.waveform is not None:
            return self.waveform.name
        assert self.presentation is not None
        return self.presentation.waveform.name

    @property
    def modality(self) -> WaveformModality:
        if self._modality is not None:
            return self._modality
        if self.waveform is not None:
            return self.waveform.modality
        assert self.presentation is not None
        return self.presentation.waveform.modality

    @property
    def latency(self) -> float | None:
        if self._latency is not None:
            return self._latency
        assert self._onset_time_on_sync is not None
        if self.presentation is not None:
            return self.onset_time_on_sync - self.presentation.trigger_time_on_sync
        if self.trigger_time_on_sync is not None:
            return self.onset_time_on_sync - self.trigger_time_on_sync
        logger.warning("No trigger time available - cannot calculate latency")
        return None

    @property
    def onset_time_on_sync(self) -> float:
        if self._onset_time_on_sync is not None:
            return self._onset_time_on_sync
        assert self.latency is not None
        if self.presentation is not None:
            return self.presentation.trigger_time_on_sync + self.latency
        assert self.trigger_time_on_sync is not None
        return self.trigger_time_on_sync + self.latency

    @property
    def offset_time_on_sync(self) -> float:
        if self._offset_time_on_sync is not None:
            return self._offset_time_on_sync
        if self.waveform is not None:
            return self.onset_time_on_sync + self.duration
        assert self.presentation is not None
        return self.onset_time_on_sync + self.presentation.duration

    @property
    def duration(self) -> float:
        if self.waveform is not None:
            return self.waveform.duration
        return self.offset_time_on_sync - self.onset_time_on_sync


def get_input_data_times(
    stim: utils.StimPathOrDataset,
    sync: utils.SyncPathOrDataset | None = None,
) -> npt.NDArray[np.float64]:
    """Best-estimate time of `getInputData()` in psychopy event loop, in seconds, from start
    of experiment. Uses preceding frame's vsync time if sync provided"""
    stim = get_stim_data(stim)
    assert isinstance(stim, h5py.File), "Only hdf5 stim files supported for now"
    if not sync:
        return np.concatenate(([0], np.cumsum(stim["frameIntervals"][:])))
    return np.concatenate(
        [
            [np.nan],
            assert_stim_times(
                get_stim_frame_times(stim, sync=sync, frame_time_type="vsync")[stim]
            )[:-1],
        ]
    )


def get_flip_times(
    stim: utils.StimPathOrDataset,
    sync: utils.SyncPathOrDataset | None = None,
) -> npt.NDArray[np.float64]:
    """Best-estimate time of `flip()` at end of psychopy event loop, in seconds, from start
    of experiment. Uses frame's vsync time sync provided."""
    stim = get_stim_data(stim)
    assert isinstance(stim, h5py.File), "Only hdf5 stim files supported for now"
    if not sync:
        return np.concatenate((np.cumsum(stim["frameIntervals"][:]), [np.nan]))
    return assert_stim_times(
        get_stim_frame_times(stim, sync=sync, frame_time_type="vsync")[stim]
    )


def get_vis_display_times(
    stim: utils.StimPathOrDataset,
    sync: utils.SyncPathOrDataset | None = None,
) -> npt.NDArray[np.float64]:
    """Best-estimate time of monitor update. Uses photodiode if sync provided. Without sync, this equals frame times."""
    stim = get_stim_data(stim)
    assert isinstance(stim, h5py.File), "Only hdf5 stim files supported for now "
    if not sync:
        return get_flip_times(stim)
    return assert_stim_times(
        get_stim_frame_times(stim, sync=sync, frame_time_type="display_time")[stim]
    )


def get_waveforms_from_stim_file(
    stim_file_or_dataset: StimPathOrDataset,
    waveform_type: Literal["sound", "audio", "opto"],
) -> tuple[Waveform | None, ...]:
    if any(s in waveform_type for s in ("sound", "audio")):
        return get_audio_waveforms_from_stim_file(stim_file_or_dataset)
    if "opto" in waveform_type:
        return get_opto_waveforms_from_stim_file(stim_file_or_dataset)
    raise ValueError(
        f"Unexpected value: {waveform_type = }. Should be 'sound' or 'opto'."
    )


def get_audio_waveforms_from_stim_file(
    stim_file_or_dataset: StimPathOrDataset,
) -> tuple[Waveform | None, ...]:
    """
    >>> path = 's3://aind-ephys-data/ecephys_662892_2023-08-21_12-43-45/behavior/RFMapping_662892_20230821_124434.hdf5'
    >>> waveforms = get_audio_waveforms_from_stim_file(path)
    >>> next(w for w in waveforms if w is not None).duration
    0.25
    """
    stim_data = get_h5_stim_data(stim_file_or_dataset)

    trialSoundArray: list[npt.NDArray] | None = stim_data.get("trialSoundArray", None)
    if (
        trialSoundArray is None
        or len(trialSoundArray) == 0
        or all(a.size == 0 for a in trialSoundArray)
    ):
        print("trialSoundArray empty; regenerating sound arrays")
        return generate_sound_waveforms(stim_data)

    # extract saved waveforms
    waveforms: list[Waveform | None] = [None] * get_num_trials(stim_data)
    for idx in range(len(waveforms)):
        if any(trialSoundArray[idx]):
            if sound_type := stim_data.get("trialSoundType"):
                name = sound_type[idx].decode()
            elif (noise := stim_data.get("trialAMNoiseFreq")) and ~np.isnan(noise[idx]):
                name = "AM_noise"
            elif (tone := stim_data.get("trialToneFreq")) and ~np.isnan(tone[idx]):
                name = "tone"
            else:
                raise ValueError(
                    f"Could not determine sound type for trial {idx} in {stim_file_or_dataset}"
                )
            waveforms[idx] = SimpleWaveform(
                name=name,
                modality=WaveformModality.SOUND,
                sampling_rate=stim_data["soundSampleRate"][()],
                samples=trialSoundArray[idx],
            )
    return tuple(waveforms)


def get_opto_waveforms_from_stim_file(
    stim_file_or_dataset: StimPathOrDataset,
) -> tuple[Waveform | None, ...]:
    stim_data = get_h5_stim_data(stim_file_or_dataset)
    if "trialOptoDur" not in stim_data or len(stim_data["trialOptoDur"]) == 0:
        raise ValueError(
            f"trialOptoDur is empty - no opto waveforms to generate from {stim_file_or_dataset}"
        )
    return generate_opto_waveforms(stim_data)


@functools.wraps(DynamicRoutingTask.TaskUtils.makeSoundArray)
@functools.cache
def get_cached_sound_waveform(*args, **kwargs) -> npt.NDArray[np.float64]:
    # any unhashable args/kwargs (incl np.ndarray) will raise TypeError
    return DynamicRoutingTask.TaskUtils.makeSoundArray(*args, **kwargs)


@functools.wraps(DynamicRoutingTask.TaskUtils.getOptoPulseWaveform)
@functools.cache
def get_cached_opto_pulse_waveform(*args, **kwargs) -> npt.NDArray[np.float64]:
    # any unhashable args/kwargs (incl np.ndarray) will raise TypeError
    return DynamicRoutingTask.TaskUtils.getOptoPulseWaveform(*args, **kwargs)


def generate_sound_waveforms(
    stim_file_or_dataset: StimPathOrDataset,
) -> tuple[Waveform | None, ...]:
    """
    >>> path = 's3://aind-ephys-data/ecephys_668755_2023-08-31_12-33-31/behavior/DynamicRouting1_668755_20230831_131418.hdf5'
    >>> waveforms = generate_sound_waveforms(path)
    >>> next(w for w in waveforms if w is not None).duration
    0.5
    """
    stim_data = get_h5_stim_data(stim_file_or_dataset)

    nTrials = get_num_trials(stim_data)
    trialSoundDur = stim_data["trialSoundDur"][:nTrials]
    trialSoundFreq = stim_data["trialSoundFreq"][:nTrials]
    if "trialSoundSeed" in stim_data:
        trialSoundSeed = stim_data["trialSoundSeed"][:nTrials]
    else:
        logger.debug(
            "trialSoundSeed not found; likely older (2022) recording; setting to None"
        )
        trialSoundSeed = [None] * nTrials
    trialSoundType = stim_data["trialSoundType"][:nTrials]
    trialSoundVolume = stim_data["trialSoundVolume"][:nTrials]
    trialSoundAM = stim_data["trialSoundAM"][:nTrials]
    soundSampleRate = stim_data["soundSampleRate"][()]
    soundHanningDur = stim_data["soundHanningDur"][()]

    waveforms: list[Waveform | None] = [None] * nTrials
    for idx in range(len(waveforms)):
        if trialSoundType[idx].decode() == "":
            continue
        if trialSoundType[idx].decode() == "tone":
            # accounts for a quirk of how the trial sound frequencies are saved
            freq = trialSoundFreq[idx][0]
        else:
            freq = trialSoundFreq[idx]

        waveforms[idx] = LazyWaveform(
            name=trialSoundType[idx].decode(),
            modality=WaveformModality.SOUND,
            sampling_rate=soundSampleRate,
            fn=get_cached_sound_waveform,
            soundType=trialSoundType[idx].decode(),
            sampleRate=soundSampleRate,
            dur=trialSoundDur[idx],
            hanningDur=soundHanningDur,
            vol=trialSoundVolume[idx],
            freq=freq,
            AM=trialSoundAM[idx],
            seed=trialSoundSeed[idx],
        )

    return tuple(waveforms)


def generate_opto_waveforms(
    stim_file_or_dataset: StimPathOrDataset, device_index: int | None = None
) -> tuple[Waveform | None, ...]:
    """
    >>> path = 's3://aind-ephys-data/ecephys_662892_2023-08-21_12-43-45/behavior/OptoTagging_662892_20230821_125915.hdf5'
    >>> waveforms = generate_opto_waveforms(path)
    >>> next(w for w in waveforms if w is not None).duration
    0.2025
    """
    stim_data = get_h5_stim_data(stim_file_or_dataset)

    nTrials = get_num_trials(stim_data)

    trialOptoDur = stim_data["trialOptoDur"][:nTrials].squeeze()
    trialOptoVoltage = stim_data["trialOptoVoltage"][:nTrials].squeeze()

    # TODO update `trialOptoDelay` to accommodate multiple devices (task only)
    # Sam says: there is a trialOptoDelay value for each device (because the
    # analog output has to start synchronously for each laser but you might
    # want one laser to actually turn on later than the other one)
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

    if "optoSampleRate" in stim_data.keys():
        optoSampleRate = stim_data["optoSampleRate"][()]
    else:
        optoSampleRate = 2000

    def device(array: npt.NDArray) -> npt.NDArray:
        if array.ndim > 1:
            return array[:, device_index or 0]
        return array

    waveforms: list[Waveform | None] = [None] * nTrials
    for trialnum in range(0, nTrials):
        if any(
            np.isnan(v[trialnum]) or v[trialnum] == 0
            for v in (trialOptoDur, trialOptoVoltage)
        ):
            continue

        if trialOptoSinFreq[trialnum] != 0:
            name = "sine"
        else:
            name = "square"

        waveform = LazyWaveform(
            name=name,
            modality=WaveformModality.OPTO,
            sampling_rate=optoSampleRate,
            fn=get_cached_opto_pulse_waveform,
            sampleRate=optoSampleRate,
            amp=device(trialOptoVoltage)[trialnum],
            dur=device(trialOptoDur)[trialnum],
            delay=device(trialOptoDelay)[trialnum],
            freq=device(trialOptoSinFreq)[trialnum],
            onRamp=device(trialOptoOnRamp)[trialnum],
            offRamp=device(trialOptoOffRamp)[trialnum],
        )
        assert waveform is not None and waveform.samples.any()
        waveforms[trialnum] = waveform

    return tuple(waveforms)


def find_envelope(s, t, dmin=1, dmax=1):
    # locals min
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1

    # global min of dmin-chunks of locals min
    lmin = lmin[
        [i + np.argmin(s[lmin[i : i + dmin]]) for i in range(0, len(lmin), dmin)]
    ]
    # global max of dmax-chunks of locals max
    lmax = lmax[
        [i + np.argmax(s[lmax[i : i + dmax]]) for i in range(0, len(lmax), dmax)]
    ]

    # upsample envelope to original sampling rate
    s_min = np.interp(t, t[lmin], s[lmin])
    s_max = np.interp(t, t[lmax], s[lmax])

    return s_min, s_max


@numba.njit(parallel=True)
def _xcorr(v, w, t) -> tuple[float, float]:
    c = np.correlate(v, w)
    return t[np.argmax(c)], np.max(c)


def xcorr(
    nidaq_data: npt.NDArray[np.int16],
    nidaq_timing: utils.EphysTimingInfoOnSync,
    nidaq_channel: int,
    presentations: Iterable[StimPresentation | None],
    use_envelope: bool = False,
    padding_sec: float = 0.15,
    **kwargs,
) -> tuple[StimRecording | None, ...]:
    num_presentations = len(tuple(presentations))
    recordings: list[StimRecording | None] = [None] * num_presentations
    padding_samples = int(padding_sec * nidaq_timing.sampling_rate)
    xcorr_values = []
    for idx, presentation in enumerate(presentations):
        # print(f"{idx+1}/{num_presentations}\r", end=' ', flush=True)
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

        interp_waveform_times = np.arange(
            0,
            presentation.duration,
            1 / nidaq_timing.sampling_rate,
        )
        interp_waveform_samples = np.interp(
            interp_waveform_times,
            presentation.waveform.timestamps,
            presentation.waveform.samples,
        )

        if use_envelope is False:
            lag, xcorr = _xcorr(nidaq_samples, interp_waveform_samples, nidaq_times)

        elif use_envelope is True:
            _, nidaq_samples_max = find_envelope(nidaq_samples, nidaq_times)
            _, interp_waveform_samples_max = find_envelope(
                interp_waveform_samples, interp_waveform_times
            )

            lag, xcorr = _xcorr(
                nidaq_samples_max, interp_waveform_samples_max, nidaq_times
            )

        # TODO: upsample option
        # interp_nidaq_times = np.arange(
        #     nidaq_times[0],
        #     nidaq_times[-1],
        #     1 / presentation.waveform.sampling_rate,
        # )
        # interp_nidaq_samples = np.interp(
        #     interp_nidaq_times,
        #     nidaq_times,
        #     nidaq_samples,
        # )

        # _,interp_nidaq_samples_max=find_envelope(interp_nidaq_samples,interp_nidaq_times)
        # _,waveform_samples_max=find_envelope(presentation.waveform.samples,presentation.waveform.timestamps)

        # lag, xcorr = _xcorr(interp_nidaq_samples_max, waveform_samples_max, interp_nidaq_times)

        recordings[idx] = FlexStimRecording(
            presentation=presentation,
            latency=lag,
        )

        xcorr_values.append(xcorr)
        # to verify:
        """
        import matplotlib.pyplot as plt
        norm_nidaq_samples = (nidaq_samples - np.mean(nidaq_samples)) / max(abs((nidaq_samples - np.mean(nidaq_samples))))
        norm_waveform_samples = (interp_waveform_samples - np.mean(interp_waveform_samples)) / max(abs((interp_waveform_samples - np.mean(interp_waveform_samples))))
        plt.plot(nidaq_times, norm_nidaq_samples)
        plt.plot(interp_waveform_times + recordings[-1].latency, norm_waveform_samples / max(abs(norm_waveform_samples)))
        plt.title(f"{recordings[-1].latency = }")
        """
    logger.info(
        f"Cross-correlation values: {max(xcorr_values)=}, {min(xcorr_values)=}, {np.mean(xcorr_values)=}"
    )
    return tuple(recordings)


def get_stim_latencies_from_nidaq_recording(
    stim_file_or_dataset: StimPathOrDataset,
    sync: utils.SyncPathOrDataset,
    recording_dirs: Iterable[utils.PathLike],
    waveform_type: Literal["sound", "audio", "opto"],
    nidaq_device_name: str | None = None,
    correlation_method: Callable[
        [
            npt.NDArray[np.int16],
            utils.EphysTimingInfoOnSync,
            int,
            Iterable[StimPresentation | None],
            bool,
        ],
        tuple[StimRecording | None, ...],
    ] = xcorr,
    correlation_method_kwargs: dict[str, Any] | None = None,
    use_envelope: bool = False,
) -> tuple[StimRecording | None, ...]:
    """
    >>> stim = 's3://aind-ephys-data/ecephys_668755_2023-08-31_12-33-31/behavior/DynamicRouting1_668755_20230831_131418.hdf5'
    >>> sync = 's3://aind-ephys-data/ecephys_668755_2023-08-31_12-33-31/behavior/20230831T123331.h5'
    >>> recording_dirs = (
    ...     's3://aind-ephys-data/ecephys_668755_2023-08-31_12-33-31/ecephys_clipped/Record Node 102/experiment2/recording1',
    ...     's3://aind-ephys-data/ecephys_668755_2023-08-31_12-33-31/ecephys_clipped/Record Node 103/experiment2/recording1',
    ... )
    >>> recordings = get_stim_latencies_from_nidaq_recording(stim, sync, recording_dirs, waveform_type='sound') # doctest:+ELLIPSIS
    >>> latency = next(_ for _ in recordings if _ is not None).latency
    >>> assert 0 < latency < 0.1
    """
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
        use_envelope,
        **(correlation_method_kwargs or {}),
    )

    return recordings


def assert_stim_times(result: Exception | npt.NDArray) -> npt.NDArray:
    """Raise exception if result is an exception, otherwise return result"""
    if isinstance(result, Exception):
        raise result from None
    return result


class MissingSyncLineError(IndexError):
    pass


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
    if not sync.get_rising_edges(line_index_or_label).any():
        raise MissingSyncLineError(
            f"No edges found for {line_index_or_label = } in {sync = }"
        )
    vsyncs = assert_stim_times(
        get_stim_frame_times(stim, sync=sync, frame_time_type="vsync")[stim]
    )
    trigger_times = tuple(
        vsyncs[idx] if idx is not None else None
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
        recordings[idx] = FlexStimRecording(
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
    FileNotFoundError: aind-ephys-data/ecephys_670248_2023-08-02_11-30-53/behavior/DynamicRouting1_670248_20230802_120703.hdf5
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
        matching_block_idx_by_start_time = np.argmin(
            abs(first_frame_per_block - stim_start_time_on_sync)
        )
        matching_block_idx_by_len = np.argmin(abs(n_frames_per_block - n_stim_frames))
        start_and_len_match_disagree: bool = (
            (matching_block_idx_by_start_time != matching_block_idx_by_len)
            and (
                len(
                    [
                        same_len_stims
                        for same_len_stims in n_frames_per_block
                        if same_len_stims
                        == n_frames_per_block[matching_block_idx_by_len]
                    ]
                )
                == 1
            )
            # if multiple blocks have the same number of frames, then we can't
            # use the number of frames to disambiguate
        )
        num_frames_match: bool = (
            n_stim_frames == n_frames_per_block[matching_block_idx_by_start_time]
        )
        # use first frame time for actual matching
        if not num_frames_match and not start_and_len_match_disagree:
            frame_diff = (
                n_stim_frames - n_frames_per_block[matching_block_idx_by_start_time]
            )
            exception = IndexError(
                f"Closest match with {stim_path} has a mismatch of {frame_diff} frames."
            )
            stim_frame_times[stim_path] = exception
            continue
        elif start_and_len_match_disagree:
            # if frame len gets the right match, and there's only one stim with that
            # number of frames (checked earlier), then we take it as the
            # correct match - however it indicates a problem with time info on
            # sync or in the stim files that we should log
            msg = f"failed to match frame times using {stim_start_time = } with {sync_data.start_time = }, expected {stim_start_time_on_sync = }. Sync or stim file may have the wrong start-time info."
            if n_stim_frames == n_frames_per_block[matching_block_idx_by_len]:
                logger.warning(
                    f"{stim_path = } matched to sync block using {n_stim_frames = }, but {msg}"
                )
                stim_frame_times[stim_path] = frame_times_in_blocks[
                    matching_block_idx_by_len
                ]
                continue
            # otherwise, we have a mismatch that we can't resolve
            time_diff_len = (
                stim_start_time_on_sync
                - first_frame_per_block[matching_block_idx_by_len]
            )
            time_diff_start = (
                stim_start_time_on_sync
                - first_frame_per_block[matching_block_idx_by_start_time]
            )
            exception = IndexError(
                f"{matching_block_idx_by_start_time=} != {matching_block_idx_by_len=} for {stim_path}: {msg} Closest match by start time has a mismatch of {time_diff_start:.1f} seconds. Closest match by number of frames has a mismatch of {time_diff_len:.1f} seconds."
            )
            stim_frame_times[stim_path] = exception
            continue
        stim_frame_times[stim_path] = frame_times_in_blocks[
            matching_block_idx_by_start_time
        ]
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
    return len(
        stim_data.get("trialEndFrame")
        or stim_data.get("trialOptoOnsetFrame")
        or stim_data.get("stimStartFrame")
    )


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

    - for DynamicRouting1 files, use `stim_type='opto'` to get the trigger frames for opto

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
        else (opto := stim_data.get("trialOptoOnsetFrame"))
    )

    if start_frames is None and opto is not None:
        # optoTagging experiments use "trialOptoOnsetFrame" instead of
        # "trialStimStartFrame" - should be safe to switch.. the stim_type
        # parameter just wasn't set to 'opto' when the function was called
        start_frames = opto
        if stim_data.get("optoTaggingLocs") is None:
            logger.warning(
                'Using "trialOptoOnsetFrame" instead of "trialStimStartFrame" - this is likely an old optoTagging experiment, and `stim_type` was specified as `stim` instead of `opto`.'
            )

    start_frames = start_frames[: get_num_trials(stim_data)].squeeze()
    monotonic_increase = np.all(
        (without_nans := start_frames[~np.isnan(start_frames)])[1:] > without_nans[:-1]
    )
    if not monotonic_increase:
        # behavior files with opto report the onset frame of opto relative to stim onset for
        # each trial. OptoTagging files specify absolute frame index
        start_frames += stim_data.get("trialStimStartFrame")[
            : get_num_trials(stim_data)
        ].squeeze()

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
