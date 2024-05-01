from __future__ import annotations

import typing
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal, Union

import hdmf.common
import numpy as np
import numpy.typing as npt
import pandas as pd
import rich.progress
from typing_extensions import TypeAlias

from npc_sessions.utils.misc import get_taskcontrol_intervals_table_name

Interval: TypeAlias = Union[
    Sequence[float],
    Mapping[Literal["start_time", "stop_time"], float],
    pd.Series,
    pd.DataFrame,
]
UnitSelection: TypeAlias = Union[int, str, Iterable[int], pd.DataFrame, pd.Series]


def parse_intervals(
    intervals: Interval | Iterable[Interval],
) -> tuple[tuple[float, float], ...]:
    intervals = tuple(intervals)
    is_single_interval = len(intervals) == 2 and isinstance(intervals[0], (int, float))
    intervals = [intervals] if is_single_interval else intervals

    _intervals: list[tuple[float, float]] = []
    for i in intervals:
        times: tuple[float, float]
        try:
            times = (i.get("start_time"), i.get("stop_time"))  # type: ignore [union-attr]
        except AttributeError:
            times = tuple(i)
        if all(isinstance(t, hdmf.common.table.VectorData) for t in times):
            t = typing.cast(
                tuple[hdmf.common.table.VectorData, hdmf.common.table.VectorData], times
            )
            times = (t[0].data, t[1].data)
        if all(isinstance(t, Iterable) for t in times):
            t = typing.cast(tuple[Iterable[float], Iterable[float]], times)
            _intervals.extend(parse_intervals(zip(*t)))
            continue
        if None in times or len(times) != 2 or sorted(times) != list(times):
            raise ValueError(
                f"Invalid interval, expected [start_time, stop_time]: {times!r}"
            )
        else:
            _intervals.append(times)
    return tuple(_intervals)


def is_overlap_in_intervals(*intervals: Interval) -> bool:
    """For two or more intervals, check if any overlap.

    >>> is_overlap_in_intervals((0, 1), (1, 2))
    False
    >>> is_overlap_in_intervals((0, 1), (1, 2), (2, 3))
    False
    >>> is_overlap_in_intervals(dict(start_time=0, stop_time=1), dict(start_time=1, stop_time=2))
    False
    >>> is_overlap_in_intervals((0, 1), (0.5, 2))
    True
    >>> is_overlap_in_intervals((0, 1), (0.5, 0.6))
    True
    >>> is_overlap_in_intervals((0, 1), (0.5, 1.5))
    True
    >>> is_overlap_in_intervals((0, 1), (-0.5, 0.5))
    True
    """
    _intervals = list(parse_intervals(intervals))
    if len(_intervals) < 2:
        raise ValueError(f"Expected two or more intervals, got {_intervals!r}")
    for idx in range(len(_intervals) - 1):
        i = _intervals.pop(idx)
        is_overlap = any(
            (i[0] < other[1] and other[0] < i[1])
            or (other[0] < i[1] and i[0] < other[1])
            for other in _intervals
        )
        if is_overlap:
            return True
    return False


def is_within_intervals(interval: Interval, *other_intervals: Interval) -> bool:
    """Check if inteval is completely within any of the other intervals.

    >>> is_within_intervals((0, 1), (1, 2))
    False
    >>> is_within_intervals((0, 1), (1, 2), (2, 3))
    False
    >>> is_within_intervals((0, 1), (0, 2))
    True
    >>> is_within_intervals((0, 1), (0, 2), (2, 3))
    True
    >>> is_within_intervals((0, 1), (0, 0.5))
    False
    >>> is_within_intervals((0, 1), (-0.5, 0.5))
    False
    """
    _intervals = list(parse_intervals([interval, *other_intervals]))
    interval = _intervals.pop(0)
    for other in _intervals:
        is_within = other[0] <= interval[0] and interval[1] <= other[1]
        if is_within:
            return True
    return False


def is_valid_interval(session, interval: Interval) -> bool:
    """Check if time interval is valid, based on `invalid_times`"""
    if session.invalid_times is not None:
        if any(
            is_overlap_in_intervals(interval, invalid_interval)
            for invalid_interval in parse_intervals(session.invalid_times)
        ):
            return False
    return True


def parse_units(session, unit_selection: UnitSelection = None) -> pd.DataFrame:
    units: pd.DataFrame
    if hasattr(unit_selection, "spike_times"):
        return unit_selection
    if isinstance(unit_selection, (pd.DataFrame, pd.Series)):
        units = session.units[:].query(f"unit_id in {unit_selection.unit_id.array!r}")
    elif isinstance(unit_selection, str):
        units = session.units[:].query(f"unit_id == {unit_selection!r}")
    elif isinstance(
        unit_selection, (int, Iterable)
    ):  # beware order: str/DataFrame are Iterable
        units = session.units[unit_selection]
    elif unit_selection is None:
        units = session.units[:]
    else:
        raise TypeError(
            f"Expected unit_selection to be {UnitSelection=} - got {unit_selection!r}"
        )
    if unit_selection is not None and len(units) == 0:
        raise ValueError(f"No units found for unit_selection: {unit_selection!r}")
    return units


import numba


@numba.njit(nogil=True)
def get_unit_spike_times(
    unit_spike_times: npt.NDArray[np.floating],
    start_time: float,
    stop_time: float,
) -> npt.NDArray[np.floating]:
    """Get spike times within interval"""
    return unit_spike_times[
        # bisect.bisect_left(unit_spike_times, start_time): bisect.bisect_right(unit_spike_times, stop_time)
        (unit_spike_times >= start_time)
        & (unit_spike_times <= stop_time)
    ]


@numba.njit(nogil=True, parallel=True, fastmath=True)
def get_spike_counts(
    spike_times: npt.NDArray[np.floating],
    intervals: npt.NDArray[np.floating],
) -> npt.NDArray[np.intp]:
    counts = np.diff(
        np.searchsorted(spike_times, intervals)
    )  # axis not supported in np.diff for numba
    return counts


def apply_invalid_intervals(
    session,
    intervals: Interval | Iterable[Interval],
    units_by_intervals: npt.NDArray | Iterable[npt.NDArray],
    unit_selection: UnitSelection | None = None,
) -> npt.NDArray[np.float64]:
    _intervals = parse_intervals(intervals)
    units = parse_units(session, unit_selection)

    # convert to float64 to allow NaNs
    units_by_intervals = np.array(units_by_intervals, dtype=np.float64, copy=True)
    assert units_by_intervals.shape[0:2] == (len(units), len(_intervals))

    is_global_obs_interval = all(
        i == units.iloc[0].obs_intervals for i in units.obs_intervals.array
    )
    # deal with unobserved intervals on a per-unit basis
    for idx, unit in enumerate(units):
        if not is_global_obs_interval:
            units_by_intervals[idx, ...] = np.where(
                [is_within_intervals(i, *unit.obs_intervals) for i in _intervals],
                units_by_intervals[idx, ...],
                np.nan,
            )
    # deal with global unobserved intervals (ie same for all units) and global invalid
    # intervals
    for idx, interval in enumerate(_intervals):
        if not is_valid_interval(session, interval) or (
            is_global_obs_interval
            and not is_within_intervals(interval, *units.iloc[0].obs_intervals)
        ):
            units_by_intervals[:, idx] = np.nan
    return units_by_intervals


def get_spike_counts_in_intervals(
    session,
    intervals: Interval | Iterable[Interval],
    unit_selection: UnitSelection | None = None,
    as_spikes_per_second: bool = False,
) -> npt.NDArray[np.floating]:
    """Get spike counts in interval(s) for unit(s).

    - returns [units x intervals] array of spike counts (as floats)
    - if no spiking data is available within an interval for a unit, its
      count is `np.nan`
    """
    _intervals = parse_intervals(intervals)
    units = parse_units(session, unit_selection)

    _spikes_per_interval_per_unit: list[npt.NDArray[np.floating]] = []
    for unit in rich.progress.track(
        units.itertuples(),
        description="getting spike counts for units",
        total=len(units),
    ):
        spikes_per_interval = get_spike_counts(
            np.array(unit.spike_times), np.array(_intervals)
        )
        if as_spikes_per_second:
            spikes_per_interval / np.diff(_intervals)
        _spikes_per_interval_per_unit.append(spikes_per_interval)

    spikes_per_interval_per_unit = apply_invalid_intervals(
        session, _intervals, _spikes_per_interval_per_unit, unit_selection
    )
    return spikes_per_interval_per_unit


def get_response_in_intervals(
    session,
    response_intervals: Interval | Iterable[Interval],
    baseline_intervals: Interval | Iterable[Interval],
    unit_selection: UnitSelection | None = None,
    as_spikes_per_second: bool = True,
    as_normalized_ratio: bool = False,
) -> npt.NDArray[np.floating]:
    if not as_spikes_per_second:
        assert np.diff(parse_intervals(response_intervals)) == np.diff(
            parse_intervals(baseline_intervals)
        ), "response and baseline intervals must have same duration to express response as spike counts"
    _response = get_spike_counts_in_intervals(
        session, response_intervals, unit_selection, as_spikes_per_second
    )
    _baseline = get_spike_counts_in_intervals(
        session, baseline_intervals, unit_selection, as_spikes_per_second
    )

    output = _response - _baseline
    if as_normalized_ratio:
        output /= _baseline
        # 0/0 is NaN, make it 0
        output[(_baseline == 0) & (_response == 0)] = 0
    return output


VIS_MIN_RESP_LATENCY = 0.025
AUD_MIN_RESP_LATENCY = 0.01
OPTO_MIN_RESP_LATENCY = 0.01

VIS_RESP_WINDOW = 0.2
AUD_RESP_WINDOW = 0.2
OPTO_RESP_WINDOW = 0.1


def add_opto_response_metric(session) -> None:
    """Adds `opto_response` metric to `session.units`"""
    trials = session.intervals.get(get_taskcontrol_intervals_table_name("OptoTagging"))
    if trials is None:
        return


def add_vis_response_metric(session) -> None:
    """Adds `vis_response` metric to `session.units`"""
    trials = session.intervals[
        get_taskcontrol_intervals_table_name("VisRFMapping")
    ].to_dataframe()
    if trials is None:
        return
    units = session.units[:]  # slight speed-up by creating df only once
    get_response_in_intervals(
        session,
        response_intervals=zip(
            (start := trials.stim_start_time[:] + VIS_MIN_RESP_LATENCY),
            start + VIS_RESP_WINDOW,
        ),
        baseline_intervals=zip(trials.start_time[:], trials.stim_start_time[:]),
        as_spikes_per_second=True,
        as_normalized_ratio=False,
        unit_selection=units,
    )
    trials = trials.query("is_small_field_grating")
    stim_by_unit_responses = []
    for grating_x in trials.grating_x.unique():
        for grating_y in trials.grating_y.unique():
            t = trials.query(f"grating_x == {grating_x} and grating_y == {grating_y}")
            if len(t) == 0:
                continue
            # get mean response for each unit to this stim location
            stim_by_unit_responses.append(
                np.nanmean(
                    get_response_in_intervals(
                        session,
                        response_intervals=zip(
                            (start := t.stim_start_time[:] + VIS_MIN_RESP_LATENCY),
                            start + VIS_RESP_WINDOW,
                        ),
                        baseline_intervals=zip(t.start_time[:], t.stim_start_time[:]),
                        as_spikes_per_second=True,
                        as_normalized_ratio=False,
                        unit_selection=session.units[
                            :
                        ],  # slight speed-up by creating df only once
                    ),
                    axis=1,
                )
            )
    session.units.add_column(
        name="vis_response",
        description="mean change in firing rate in response to small-field grating stimulus at most responsive location on screen",
        data=np.nanmax(stim_by_unit_responses, axis=0),
    )


def add_aud_response_metric(session) -> None:
    """Adds `aud_response` metric to `session.units`"""
    trials = session.intervals["AudRFMapping"]
    if trials is None:
        return


if __name__ == "__main__":
    import doctest

    import dotenv

    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
