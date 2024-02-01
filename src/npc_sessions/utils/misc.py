from __future__ import annotations

import datetime
import zoneinfo
from collections.abc import Iterable
from typing import SupportsFloat

import npc_io
import npc_lims
import npc_session
import numpy as np
import numpy.typing as npt


def is_stim_file(
    path: npc_io.PathLike,
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
    >>> is_stim_file("366122/test.stim.pkl", subject_spec="366122")
    True
    """
    path = npc_io.from_pathlike(path)
    subject_from_path = npc_session.extract_subject(path.as_posix())
    date_from_path = npc_session.extract_isoformat_date(path.as_posix())
    time_from_path = npc_session.extract_isoformat_time(path.as_posix())

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


def assert_s3_write_credentials() -> None:
    test = npc_lims.DR_DATA_REPO / "test.txt"
    test.touch()
    test.unlink()


def get_aware_dt(dt: str | datetime.datetime) -> datetime.datetime:
    """Add Seattle timezone info to a datetime string or object."""
    if isinstance(dt, datetime.datetime):
        dt = dt.isoformat()
    return npc_session.DatetimeRecord(dt).dt.replace(
        tzinfo=zoneinfo.ZoneInfo("America/Los_Angeles")
    )


if __name__ == "__main__":
    import doctest

    import dotenv

    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
