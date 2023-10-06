from __future__ import annotations

import collections.abc
import contextlib
import pathlib
from collections.abc import Iterable, Iterator
from typing import Literal, SupportsFloat, TypeVar

import npc_lims
import npc_session
import numpy as np
import numpy.typing as npt
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


def safe_index(
    array: npt.ArrayLike, indices: SupportsFloat | Iterable[SupportsFloat]
) -> npt.NDArray:
    """Checks `indices` can be safely used as array indices (i.e. all
    numerical float values are integers), then indexes into `array` using `np.where`.

    - returns nans where `indices` is nan
    - returns a scalar if `indices` is a scalar #TODO current type annotation is insufficient

    >>> safe_index([1, 2], 0)
    1
    >>> safe_index([1., 2.], 0)
    1.0
    >>> safe_index([1., 2.], np.nan)
    nan
    >>> safe_index([1., 2., 3.1], [0, np.nan, 2.0])
    array([1. , nan, 3.1])

    Type of array is preserved, if possible:
    >>> safe_index([1, 2, 3], [0., 1., 2.])
    array([1, 2, 3])

    Type of array can't be preserved if any indices are nan:
    >>> safe_index([1, 2, 3], [0, np.nan, 2.0])
    array([ 1., nan,  3.])
    """
    idx: npt.NDArray = np.array(indices)  # copy
    if not all(idx[~np.isnan(idx)] == idx[~np.isnan(idx)].astype(np.int32)):
        raise TypeError(
            f"Non-integer numerical values cannot be used as indices: {idx[np.isnan(idx)][0]}"
        )
    array = np.array(array)  # copy/make sure array can be fancy-indexed
    int_idx = np.where(np.isnan(idx), -1, idx)
    result = np.where(np.isnan(idx), np.nan, array[int_idx.astype(np.int32)])
    # np.where casts indexed array to floats just because of the
    # possibility of nans being in result, even if they aren't:
    # cast back if appropriate
    if not np.isnan(result).any():
        result = result.astype(array.dtype)
    # if indices was a scalar, return a scalar instead of a 0d array
    if not isinstance(indices, Iterable):
        assert result.size == 1
        return result.item()
    return result


K = TypeVar("K")
V = TypeVar("V")


class LazyDict(collections.abc.Mapping[K, V]):
    """Dict for postponed evaluation of functions and caching of results.

    Assign values as a tuple of (callable, args, kwargs). The callable will be
    evaluated when the key is first accessed. The result will be cached and
    returned directly on subsequent access.

    Effectively immutable after initialization.

    Initialize with a dict:
    >>> d = LazyDict({'a': (lambda x: x + 1, (1,), {})})
    >>> d['a']
    2

    or with keyword arguments:
    >>> d = LazyDict(b=(min, (1, 2), {}))
    >>> d['b']
    1
    """

    def __init__(self, *args, **kwargs) -> None:
        self._raw_dict = dict(*args, **kwargs)

    def __getitem__(self, key) -> V:
        with contextlib.suppress(TypeError):
            func, args, *kwargs = self._raw_dict.__getitem__(key)
            self._raw_dict.__setitem__(key, func(*args, **kwargs[0]))
        return self._raw_dict.__getitem__(key)

    def __iter__(self) -> Iterator[K]:
        return iter(self._raw_dict)

    def __len__(self) -> int:
        return len(self._raw_dict)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(keys={list(self._raw_dict.keys())})"


def assert_s3_write_credentials() -> None:
    test = npc_lims.DR_DATA_REPO / "test.txt"
    test.touch()
    test.unlink()


if __name__ == "__main__":
    import doctest

    import dotenv

    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
