
from __future__ import annotations

from typing import Iterable, Literal, Mapping, Sequence, Union
import typing

import numpy as np
import numpy.typing as npt
import pandas as pd
from typing_extensions import TypeAlias

Interval: TypeAlias = Union[Sequence[float], Mapping[Literal['start_time', 'stop_time'], float]]

def parse_intervals(
    *intervals: Interval
    ) -> tuple[tuple[float, float], ...]:
    if len(intervals) < 2:
        raise ValueError(f'Expected two or more intervals: {intervals!r}')
    _intervals: list[tuple[float, float]] = []
    for i in intervals:
        times: tuple[float, float]
        try:
            times = (i.get("start_time"), i.get('stop_time')) # type: ignore [assignment, union-attr]
        except AttributeError:
            times = tuple(i) # type: ignore [assignment]
        if None in times or sorted(times) != list(times) or len(times) != 2:
            raise ValueError(f'Invalid interval, expected [start_time, stop_time]: {times!r}')
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
    _intervals = list(parse_intervals(*intervals))
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
    _intervals = list(parse_intervals(interval, *other_intervals))
    interval = _intervals.pop(0)
    for other in _intervals:
        is_within = other[0] <= interval[0] and interval[1] <= other[1]
        if is_within:
            return True
    return False


def is_valid_interval(session, start_time: float, stop_time: float) -> bool:
    """Check if time interval is valid, based on `invalid_times`"""
    if session.invalid_times is not None:
        for _, invalid_interval in session.invalid_times[:].iterrows():
            if is_overlap_in_intervals((start_time, stop_time), invalid_interval):
                return False
    return True

@typing.overload
def get_spike_counts_in_intervals(
    session, 
    start_times: float, 
    stop_times: float,
    unit_idx_or_name=None,
    ) -> tuple[int | np.nan, ...]:
    ...
    
@typing.overload
def get_spike_counts_in_intervals(
    session, 
    start_times: Iterable[float], 
    stop_times: Iterable[float],
    unit_idx_or_name=None,
    ) -> tuple[tuple[int | np.nan, ...], ...]:
    ...
    
@typing.overload
def get_spike_counts_in_intervals(
    session, 
    start_times, 
    stop_times,
    unit_idx_or_name: int | str,
    ) -> int | np.nan | tuple[int | np.nan, ...]:
    ...
    
def get_spike_counts_in_intervals(
    session, 
    start_times: float | Iterable[float], 
    stop_times: float | Iterable[float],
    unit_idx_or_name: int | str | None = None,
    ) ->  int | np.nan | tuple[int | np.nan | tuple[int | np.nan, ...], ...]:
    """
    
    
    """
    is_single_interval = False
    if not isinstance(start_times, Iterable):
        start_times = [start_times]
        is_single_interval = True
    if not isinstance(stop_times, Iterable):
        stop_times = [stop_times]
        is_single_interval = True
    
    if unit_idx_or_name and isinstance(unit_idx_or_name, str):
        units: pd.DataFrame = session.units[:].query(f"unit_id == {unit_idx_or_name!r}")
    elif unit_idx_or_name and isinstance(unit_idx_or_name, int):
        units = session.units[unit_idx_or_name]
    else:
        units = session.units[:]
    assert len(units) > 0
    
    invalid_intervals_idx = [
        idx
        for idx, (start_time, stop_time) in enumerate(zip(start_times, stop_times))
        if not is_valid_interval(session, start_time, stop_time)
    ]
    spike_counts: list[tuple[int | np.nan, ...]] = []
    for _, unit in units.iterrows():
        unit_spike_times = np.array(unit["spike_times"])
        assert unit_spike_times.ndim == 1
        unit_spike_counts_in_intervals: list[int | np.nan] = []
        for idx, (start_time, stop_time) in enumerate(zip(start_times, stop_times)):
            if idx in invalid_intervals_idx:
                unit_spike_counts_in_intervals.append(np.nan)
                continue
            if not is_within_intervals((start_time, stop_time), *unit["obs_intervals"]):
                unit_spike_counts_in_intervals.append(np.nan)
                continue
            unit_spike_counts_in_intervals.append(
                unit_spike_times[(unit_spike_times >= start_time) & (unit_spike_times <= stop_time)].size
                )
        spike_counts.append(tuple(unit_spike_counts_in_intervals))
     
    if is_single_interval: 
        counts = tuple(c[0] for c in spike_counts) 
    else:
        counts = tuple(spike_counts) # type: ignore [arg-type]
        
    if unit_idx_or_name:
        return counts[0]
    return counts

if __name__ == "__main__":
    
    import doctest

    import dotenv

    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )