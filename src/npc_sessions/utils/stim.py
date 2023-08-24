from __future__ import annotations

import io
import logging
import pickle
from collections.abc import Iterable
from typing import Callable, NamedTuple, TypeAlias, Mapping, Any, Optional

import numba
import numpy as np
import numpy.typing as npt
import upath
import h5py

import npc_sessions.utils as utils
import npc_sessions.utils.sync as sync

StimPathOrDataset: TypeAlias = utils.PathLike | h5py.File | Mapping

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


class StimPresentation(NamedTuple):
    trial_idx: int
    waveform: npt.NDArray[np.float64]
    sampling_rate: float
    onset_sample_on_nidaq: int
    offset_sample_on_nidaq: int
    trigger_time_on_sync: float

    @property
    def duration(self) -> float:
        return len(self.waveform) / self.sampling_rate


class StimRecording(NamedTuple):
    presentation: StimPresentation
    latency: float

    @property
    def onset_time_on_sync(self) -> float:
        return self.presentation.trigger_time_on_sync + self.latency

    @property
    def offset_time_on_sync(self) -> float:
        return self.onset_time_on_sync + self.presentation.duration


def get_frame_display_times(
    stim_path: StimPathOrDataset, sync_file_or_dataset: sync.SyncPathOrDataset
) -> npt.NDArray[np.float64]:
    return np.array([])
    # TODO ethan working on it


@numba.njit(parallel=True)
def _xcorr(v, w, t) -> float:
    c = np.correlate(v, w)
    return t[np.argmax(c)]


def xcorr(
    nidaq_data: npt.NDArray[np.int16],
    presentations: Iterable[StimPresentation],
    padding_sec: float = 0.15,
    **kwargs,
) -> tuple[StimRecording, ...]:
    recordings: list[StimRecording] = []
    for presentation in presentations:
        print(f"{presentation.trial_idx}/{len(tuple(presentations))}\r", flush=True)

        times = np.arange(
            presentation.offset_sample_on_nidaq - presentation.onset_sample_on_nidaq
        ) / (presentation.sampling_rate - padding_sec)
        values = nidaq_data[
            presentation.onset_sample_on_nidaq : presentation.offset_sample_on_nidaq
        ]
        interp_times = np.arange(
            -padding_sec,
            presentation.duration + padding_sec,
            1 / presentation.sampling_rate,
        )
        interp_values = np.interp(interp_times, times, values)

        recordings.append(
            StimRecording(
                presentation=presentation,
                latency=_xcorr(interp_values, presentation.waveform, interp_times),
            )
        )
        # long padding slows down np.corr: could change dynamically
        # padding_sec = 2 * recordings[-1].latency

        # to verify:
        """
        import matplotlib.pyplot as plt
        norm_values = (interp_values - np.mean(interp_values))/max(interp_values)
        waveform_times = np.arange(0, presentation.duration, 1 / presentation.sampling_rate)
        plt.plot(waveform_times + recordings[-1].latency, presentation.waveform)
        plt.plot(interp_times, norm_values)
        """

    return tuple(recordings)


def get_stim_latencies_from_nidaq_recording(
    *stim_files_or_datasets: StimPathOrDataset,
    sync_file_or_dataset: sync.SyncPathOrDataset,
    recording_dirs: Iterable[upath.UPath],
    nidaq_device_name: str | None = None,
    correlation_method: Callable[
        [npt.NDArray[np.int16], Iterable[StimPresentation]], tuple[StimRecording, ...]
    ] = xcorr,
    correlation_method_kwargs: Optional[dict[str, Any]] = None,
) -> dict[StimPathOrDataset, tuple[StimRecording, ...]]:
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
            sync_path_or_dataset=sync_file_or_dataset,
            recording_dirs=recording_dirs,
            devices=(nidaq_device,),
        )
    )

    nidaq_data = utils.get_pxi_nidaq_data(
        recording_dirs=recording_dirs,
        device_name=nidaq_device_name,
    )

    output = {}
    for stim_file in stim_files_or_datasets:
        stim = get_h5_stim_data(stim_file)

        vsyncs = get_frame_display_times(stim_file, sync_file_or_dataset)

        num_trials = len((stim.get("trialEndFrame") or stim.get("trialSoundArray"))[:])

        trigger_frames: npt.NDArray[np.int16] = (
            stim.get("trialStimStartFrame") or stim.get("stimStartFrame")
        )[:num_trials]
        waveform_rate = float(stim["soundSampleRate"][()])
        waveforms = np.array(stim["trialSoundArray"][:num_trials])

        presentations = []

        for idx, waveform in enumerate(waveforms):
            if not any(waveform):
                continue

            trigger_time_on_sync = float(vsyncs[trigger_frames[idx]])
            trigger_time_on_pxi_nidaq = trigger_time_on_sync - nidaq_timing.start_time
            duration = len(waveform) / waveform_rate
            onset_sample_on_pxi_nidaq = int(
                trigger_time_on_pxi_nidaq * nidaq_timing.sampling_rate
            )
            offset_sample_on_pxi_nidaq = int(
                (trigger_time_on_pxi_nidaq + duration) * nidaq_timing.sampling_rate
            )
            # padding should be done by correlation method, when reading data

            presentations.append(
                StimPresentation(
                    trial_idx=idx,
                    waveform=waveform,
                    sampling_rate=waveform_rate,
                    onset_sample_on_nidaq=onset_sample_on_pxi_nidaq,
                    offset_sample_on_nidaq=offset_sample_on_pxi_nidaq,
                    trigger_time_on_sync=trigger_time_on_sync,
                )
            )

        # run the correlation of presentations with nidaq data
        recordings = correlation_method(
            nidaq_data, presentations, **(correlation_method_kwargs or {})
        )

        output[stim_file] = recordings

    return output


if __name__ == "__main__":
    import doctest

    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
