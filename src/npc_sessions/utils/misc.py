from __future__ import annotations

import collections.abc
import pathlib
from collections.abc import Iterable, Iterator
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


def extract_camera_name(path: str) -> Literal["eye", "face", "behavior"]:
    names: dict[str, Literal["eye", "face", "behavior"]] = {
        "eye": "eye",
        "face": "face",
        "beh": "behavior",
    }
    try:
        return names[next(n for n in names if n in str(path).lower())]
    except StopIteration as exc:
        raise ValueError(f"Could not extract camera name from {path}") from exc


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
