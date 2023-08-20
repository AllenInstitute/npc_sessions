from __future__ import annotations

import collections.abc
import pathlib
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, Literal

import npc_session
import numpy as np
import upath


def is_stim_file(
    path: str | upath.UPath | pathlib.Path,
    subject_spec: str | int | npc_session.SubjectRecord | None = None,
    date_spec: str | npc_session.DateRecord | None = None,
    time_spec: str | npc_session.TimeRecord | None = None,
) -> bool:
    """Does the string or path match a known stimulus file pattern.

    Optional arguments can be used to check the subject, date, and time.
    >>> is_stim_file("366122_20230101_120000.hdf5")
    True
    >>> is_stim_file("366122_20230101_120000.hdf5", date_spec="20230101")
    True
    >>> is_stim_file("366122.stim.pkl")
    True
    >>> is_stim_file("366123.stim.pkl", subject_spec="366122")
    False
    """
    path = pathlib.Path(path)
    subject_from_path = npc_session.extract_subject(path.stem)
    date_from_path = npc_session.extract_isoformat_date(path.stem)
    time_from_path = npc_session.extract_isoformat_time(path.stem)

    is_stim = False
    is_stim |= path.suffix in (".pkl",) and any(
        label in path.stem for label in ("stim", "mapping", "opto", "behavior")
    )
    is_stim |= (
        path.suffix in (".hdf5",) and bool(subject_from_path) and bool(date_from_path)
    )
    is_correct = True
    if subject_spec:
        is_correct &= subject_from_path == npc_session.SubjectRecord(str(subject_spec))
    if date_spec:
        is_correct &= date_from_path == npc_session.DateRecord(date_spec)
    if time_spec:
        is_correct &= time_from_path == npc_session.TimeRecord(time_spec)
    return is_stim and is_correct


def extract_video_file_name(path: str) -> Literal["eye", "face", "behavior"]:
    names: dict[str, Literal["eye", "face", "behavior"]] = {
        "eye": "eye",
        "face": "face",
        "beh": "behavior",
    }
    return names[next(n for n in names if n in str(path).lower())]


def check_array_indices(
    indices: int | float | Iterable[int | float],
) -> Iterable[int | float]:
    """Check indices can be safely converted from float to int (i.e. all
    are integer values). Makes a single int/float index iterable.

    >>> check_array_indices(1)
    (1,)
    >>> check_array_indices([1, 2, 3.0])
    [1, 2, 3.0]
    >>> check_array_indices([1, 2, 3.1])
    Traceback (most recent call last):
    ...
    TypeError: Non-integer `float` used as an index
    """
    if not isinstance(indices, Iterable):
        indices = (indices,)

    for idx in indices:
        if (
            isinstance(idx, (float, np.floating))
            and not np.isnan(idx)
            and int(idx) != idx
        ):
            raise TypeError("Non-integer `float` used as an index")
    return indices


def reshape_into_blocks(
    indices: Sequence[float],
    min_gap: int | float | None = None,
) -> tuple[Sequence[float], ...]:
    """
    Find the large gaps in indices and split at each gap.

    For example, if two blocks of stimuli were recorded in a single sync
    file, there will be one larger-than normal gap in frame timestamps.

    - default min gap threshold: median + 3 * std (won't work well for short seqs)

    >>> reshape_into_blocks([0, 1, 2, 103, 104, 105], min_gap=100)
    ([0, 1, 2], [103, 104, 105])

    >>> reshape_into_blocks([0, 1, 2, 3])
    ([0, 1, 2, 3],)
    """
    intervals = np.diff(indices)
    long_interval_threshold = (
        min_gap
        if min_gap is not None
        else (np.median(intervals) + 3 * np.std(intervals))
    )

    gaps_between_blocks = []
    for interval_index, interval in zip(
        intervals.argsort()[::-1], sorted(intervals)[::-1]
    ):
        if interval > long_interval_threshold:
            # large interval found
            gaps_between_blocks.append(interval_index + 1)
        else:
            break

    if not gaps_between_blocks:
        # a single block of timestamps
        return (indices,)

    # create blocks as intervals [start:end]
    gaps_between_blocks.sort()
    blocks = []
    start = 0
    for end in gaps_between_blocks:
        blocks.append(indices[start:end])
        start = end
    # add end of last block
    blocks.append(indices[start:])

    # filter out blocks with a single sample (not a block)
    blocks = [block for block in blocks if len(block) > 1]

    # filter out blocks with long avg timstamp interval (a few, widely-spaced timestamps)
    blocks = [
        block for block in blocks if np.median(np.diff(block)) < long_interval_threshold
    ]

    return tuple(blocks)


class LazyDict(collections.abc.Mapping):
    """Dict for postponed evaluation of functions and caching of results.

    Assign values as a tuple of (callable, *args). The callable will be
    evaluated when the key is first accessed. The result will be cached and
    returned directly on subsequent access.

    Effectively immutable after initialization.

    Initialize with a dict:
    >>> d = LazyDict({'a': (lambda x: x + 1, 1)})
    >>> d['a']
    2

    or with keyword arguments:
    >>> d = LazyDict(b=(min, 1, 2))
    >>> d['b']
    1
    """

    def __init__(self, *args, **kwargs) -> None:
        self._raw_dict = dict(*args, **kwargs)

    def __getitem__(self, key) -> Any:
        func, *args = self._raw_dict.__getitem__(key)
        try:
            self._raw_dict.__setitem__(key, func(*args))
        finally:
            return self._raw_dict.__getitem__(key)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._raw_dict)

    def __len__(self) -> int:
        return len(self._raw_dict)


if __name__ == "__main__":
    import doctest

    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
