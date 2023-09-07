"""
Created on Wed Aug 22 16:40:36 2018

@author: svc_ccg

from the ecephys repo (https://github.com/AllenInstitute/ecephys_pipeline)
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt


def extract_barcodes_from_times(
    on_times: np.ndarray,
    off_times: np.ndarray,
    inter_barcode_interval: float = 29,
    bar_duration: float = 0.015,
    barcode_duration_ceiling: float = 2,
    nbits: int = 32,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    # from ecephys repo
    """Read barcodes from timestamped rising and falling edges.
    Parameters
    ----------
    on_times : numpy.ndarray
        Timestamps of rising edges on the barcode line
    off_times : numpy.ndarray
        Timestamps of falling edges on the barcode line
    inter_barcode_interval : numeric, optional
        Minimun duration of time between barcodes.
    bar_duration : numeric, optional
        A value slightly shorter than the expected duration of each bar
    barcode_duration_ceiling : numeric, optional
        The maximum duration of a single barcode
    nbits : int, optional
        The bit-depth of each barcode
    Returns
    -------
    barcode_start_times : list of numeric
        For each detected barcode, the time at which that barcode started
    barcodes : list of int
        For each detected barcode, the value of that barcode as an integer.
    Notes
    -----
    ignores first on pulse (intended - this is needed to identify that a barcode is starting)
    """
    if len(on_times) > len(off_times):
        on_times = on_times[:-1]

    start_indices = np.diff(on_times)
    a = np.where(start_indices > inter_barcode_interval)[0]
    if on_times[0] > barcode_duration_ceiling:
        a = np.insert(a, 0, -1)  # to add back first barcode
    barcode_start_times = on_times[a + 1]

    barcodes = []

    for _i, t in enumerate(barcode_start_times):
        oncode = on_times[
            np.where(
                np.logical_and(on_times > t, on_times < t + barcode_duration_ceiling)
            )[0]
        ]
        offcode = off_times[
            np.where(
                np.logical_and(off_times > t, off_times < t + barcode_duration_ceiling)
            )[0]
        ]

        currTime = offcode[0]

        bits = np.zeros((nbits,))

        for bit in range(0, nbits):
            nextOn = np.where(oncode > currTime)[0]
            nextOff = np.where(offcode > currTime)[0]

            if nextOn.size > 0:
                nextOn = oncode[nextOn[0]]
            else:
                nextOn = t + inter_barcode_interval

            if nextOff.size > 0:
                nextOff = offcode[nextOff[0]]
            else:
                nextOff = t + inter_barcode_interval

            if nextOn < nextOff:
                bits[bit] = 1

            currTime += bar_duration

        barcode = 0

        # least sig left
        for bit in range(0, nbits):
            barcode += bits[bit] * pow(2, bit)

        barcodes.append(barcode)

    return barcode_start_times, np.array(barcodes, dtype=np.int64)


def find_matching_index(
    master_barcodes: np.ndarray,
    probe_barcodes: np.ndarray,
    alignment_type: Literal["start", "end"] = "start",
) -> tuple[int, int] | tuple[None, None]:
    """Given a set of barcodes for the master clock and the probe clock, find the
    indices of a matching set, either starting from the beginning or the end
    of the list.
    Parameters
    ----------
    master_barcodes : np.ndarray
        barcode values on the master line. One per barcode
    probe_barcodes : np.ndarray
        barcode values on the probe line. One per barcode
    alignment_type : string
        'start' or 'end'
    Returns
    -------
    master_barcode_index : int
        matching index for master barcodes (None if not found)
    probe_barcode_index : int
        matching index for probe barcodes (None if not found)


    >>> find_matching_index(np.array([1, 2, 3]), np.array([1, 2, 3]), alignment_type="start")
    (0, 0)
    >>> find_matching_index(np.array([1, 2, 3]), np.array([1, 2, 3]), alignment_type="end")
    (2, -1)
    >>> find_matching_index(np.array([1, 2, 3]), np.array([4, 5, 6]), alignment_type="start")
    (None, None)
    """

    foundMatch = False
    master_barcode_index = None

    if alignment_type == "start":
        probe_barcode_index = 0
        direction = 1
    else:
        probe_barcode_index = -1
        direction = -1

    while not foundMatch and abs(probe_barcode_index) < len(probe_barcodes):
        master_barcode_index = np.where(
            master_barcodes == probe_barcodes[probe_barcode_index]
        )[0]

        assert len(master_barcode_index) < 2

        if len(master_barcode_index) == 1:
            foundMatch = True
        else:
            probe_barcode_index += direction

    if foundMatch and master_barcode_index is not None:
        return master_barcode_index[0], probe_barcode_index
    else:
        return None, None


def match_barcodes(
    master_times: np.ndarray,
    master_barcodes: np.ndarray,
    probe_times: np.ndarray,
    probe_barcodes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Given sequences of barcode values and (local) times on a probe line and a master
    line, find the time points on each clock corresponding to the first and last shared
    barcode.
    If there's only one probe barcode, only the first matching timepoint is returned.
    Parameters
    ----------
    master_times : np.ndarray
        start times of barcodes (according to the master clock) on the master line.
        One per barcode.
    master_barcodes : np.ndarray
        barcode values on the master line. One per barcode
    probe_times : np.ndarray
        start times (according to the probe clock) of barcodes on the probe line.
        One per barcode
    probe_barcodes : np.ndarray
        barcode values on the probe_line. One per barcode
    Returns
    -------
    probe_interval : np.ndarray
        Start and end times of the matched interval according to the probe_clock.
    master_interval : np.ndarray
        Start and end times of the matched interval according to the master clock
    """

    master_start_index, probe_start_index = find_matching_index(
        master_barcodes, probe_barcodes, alignment_type="start"
    )

    if master_start_index is not None:
        t_m_start = master_times[master_start_index]
        t_p_start = probe_times[probe_start_index]
    else:
        t_m_start, t_p_start = None, None

    # print(master_barcodes)
    # print(probe_barcodes)

    # print("Master start index: " + str(master_start_index))
    if len(probe_barcodes) > 2:
        master_end_index, probe_end_index = find_matching_index(
            master_barcodes, probe_barcodes, alignment_type="end"
        )

        if probe_end_index is not None:
            # print("Probe end index: " + str(probe_end_index))
            t_m_end = master_times[master_end_index]
            t_p_end = probe_times[probe_end_index]
        else:
            t_m_end = None
            t_p_end = None
    else:
        t_m_end, t_p_end = None, None

    return np.array([t_p_start, t_p_end]), np.array([t_m_start, t_m_end])


def linear_transform_from_intervals(
    master: np.ndarray | Sequence[float], probe: np.ndarray | Sequence[float]
) -> tuple[float, float]:
    # from ecephys repo
    """Find a scale and translation which aligns two 1d segments
    Parameters
    ----------
    master : iterable
        Pair of floats defining the master interval. Order is [start, end].
    probe : iterable
        Pair of floats defining the probe interval. Order is [start, end].
    Returns
    -------
    scale : float
        Scale factor. If > 1.0, the probe clock is running fast compared to the
        master clock. If < 1.0, the probe clock is running slow.
    translation : float
        If > 0, the probe clock started before the master clock. If > 0, after.
    Notes
    -----
    solves
        (master + translation) * scale = probe
    for scale and translation
    """

    scale = (probe[1] - probe[0]) / (master[1] - master[0])
    translation = probe[0] / scale - master[0]

    return float(scale), float(translation)


def get_probe_time_offset(
    master_times: np.ndarray,
    master_barcodes: np.ndarray,
    probe_times: np.ndarray,
    probe_barcodes: np.ndarray,
    acq_start_index: int,
    local_probe_rate: float,
) -> tuple[float, float, np.ndarray]:
    # from ecephys repo
    """Time offset between master clock and recording probes. For converting probe time to master clock.

    Parameters
    ----------
    master_times : np.ndarray
        start times of barcodes (according to the master clock) on the master line.
        One per barcode.
    master_barcodes : np.ndarray
        barcode values on the master line. One per barcode
    probe_times : np.ndarray
        start times (according to the probe clock) of barcodes on the probe line.
        One per barcode
    probe_barcodes : np.ndarray
        barcode values on the probe_line. One per barcode
    acq_start_index : int
        sample index of probe acquisition start time
    local_probe_rate : float
        the probe's apparent sampling rate

    Returns
    -------
    total_time_shift : float
        Time at which the probe started acquisition, assessed on
        the master clock. If < 0, the probe started earlier than the master line.
    probe_rate : float
        The probe's sampling rate, assessed on the master clock
    master_endpoints : iterable
        Defines the start and end times of the sync interval on the master clock

    """

    probe_endpoints, master_endpoints = match_barcodes(
        master_times, master_barcodes, probe_times, probe_barcodes
    )
    rate_scale, time_offset = linear_transform_from_intervals(
        master_endpoints, probe_endpoints
    )

    probe_rate = local_probe_rate * rate_scale
    acq_start_time = acq_start_index / probe_rate

    total_time_shift = time_offset - acq_start_time

    return total_time_shift, probe_rate, master_endpoints


if __name__ == "__main__":
    import doctest

    import dotenv

    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
