from __future__ import annotations

import logging
from typing import Callable, Literal

import numpy as np
import numpy.typing as npt
import scipy.signal

import npc_sessions.utils as utils

logger = logging.getLogger(__name__)

FRAMERATE = 60
"""Visual stim f.p.s - assumed to equal running wheel sampling rate. i.e. one
running wheel sample per camstim vsync"""

RUNNING_SPEED_UNITS: Literal["cm/s", "m/s"] = "cm/s"
"""How to report in NWB - NWB expects SI, SDK might have previously reported cm/s"""

RUNNING_LOWPASS_FILTER_HZ = 4
"""Frequency for filtering running speed - filtered data stored in NWB `processing`, unfiltered
in `acquisition`"""


def get_running_speed_from_stim_files(
    *stim_path_or_dataset: utils.StimPathOrDataset,
    sync: utils.SyncPathOrDataset | None = None,
    filt: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] | None = None,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Pools running speeds across files. Returns arrays of running speed and
    corresponding timestamps."""
    if not sync and len(stim_path_or_dataset) > 1:
        raise ValueError(
            "Must pass sync file to coordinate data from multiple stim files."
        )

    running_speed = np.array([])
    timestamps = np.array([])

    def _append(values, times):
        if len(times) + 1 == len(values):
            values = values[1:]

        if len(times) == len(values):
            times = times[1:]
            values = values[1:]
            times = times + 0.5 * np.median(np.diff(times))
        else:
            raise ValueError(
                f"Length mismatch between running speed ({len(values)}) and timestamps ({len(times)})"
            )
        nonlocal running_speed, timestamps
        # we need to filter before pooling discontiguous blocks of samples
        running_speed = np.concatenate(
            [running_speed, filt(values) if filt else values]
        )
        timestamps = np.concatenate([timestamps, times])

    if sync is None:
        _append(
            get_running_speed_from_hdf5(*stim_path_or_dataset),
            get_frame_times_from_stim_file(*stim_path_or_dataset),
        )
    else:
        # we need timestamps for each frame's nidaq-read time (wheel encoder is read before frame's
        # flip time)
        # there may be multiple h5 files with encoder
        # data per sync file: vsyncs are in blocks with a separating gap
        for hdf5 in stim_path_or_dataset:
            read_times = utils.get_input_data_times(hdf5, sync)
            h5_data = get_running_speed_from_hdf5(hdf5)
            if h5_data is None:
                continue
            _append(h5_data, read_times)

    assert len(running_speed) == len(timestamps)
    return running_speed, timestamps


def get_frame_times_from_stim_file(
    stim_path_or_dataset: utils.StimPathOrDataset,
) -> npt.NDArray:
    return np.concatenate(
        ([0], np.cumsum(utils.get_stim_data(stim_path_or_dataset)["frameIntervals"][:]))
    )


def get_running_speed_from_hdf5(
    stim_path_or_dataset: utils.StimPathOrDataset,
) -> npt.NDArray | None:
    """
    Running speed in m/s or cm/s (see `UNITS`).


    To align with timestamps, remove timestamp[0] and sample[0] and shift
    timestamps by half a frame (speed is estimated from difference between
    timestamps)

    See https://github.com/samgale/DynamicRoutingTask/blob/main/Analysis/DynamicRoutingAnalysisUtils.py
    """
    d = utils.get_stim_data(stim_path_or_dataset)
    if (
        "rotaryEncoder" in d
        and isinstance(d["rotaryEncoder"][()], bytes)
        and d["rotaryEncoder"].asstr()[()] == "digital"
    ):
        if "frameRate" in d:
            assert d["frameRate"][()] == FRAMERATE
        wheel_revolutions = (
            d["rotaryEncoderCount"][:] / d["rotaryEncoderCountsPerRev"][()]
        )
        if not any(wheel_revolutions):
            return None
        wheel_radius_cm = d["wheelRadius"][()]
        if RUNNING_SPEED_UNITS == "m/s":
            running_disk_radius = wheel_radius_cm / 100
        elif RUNNING_SPEED_UNITS == "cm/s":
            running_disk_radius = wheel_radius_cm
        else:
            raise ValueError(
                f"Unexpected units for running speed: {RUNNING_SPEED_UNITS}"
            )
        speed = np.diff(wheel_revolutions * 2 * np.pi * running_disk_radius * FRAMERATE)
        # we lost one sample due to diff: pad with nan to keep same number of samples
        return np.concatenate([[np.nan], speed])
    return None


def lowpass_filter(running_speed: npt.NDArray) -> npt.NDArray:
    """
    Careful not to filter discontiguous blocks of samples.
    See
    https://github.com/AllenInstitute/AllenSDK/blob/36e784d007aed079e3cad2b255ca83cdbbeb1330/allensdk/brain_observatory/behavior/data_objects/running_speed/running_processing.py
    """
    b, a = scipy.signal.butter(
        3, Wn=RUNNING_LOWPASS_FILTER_HZ, fs=FRAMERATE, btype="lowpass"
    )
    return scipy.signal.filtfilt(b, a, np.nan_to_num(running_speed))
