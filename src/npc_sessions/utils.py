from __future__ import annotations

import contextlib
import datetime
import itertools
import operator
import re
from typing import Sequence, TypeVar


def extract_subject(filename: str) -> int:
    """Extract subject ID from filename.

    >>> extract_subject('Name_366122_2021-06-01_10-00-00_1.hdf5')
    366122
    """
    for sub in filename.split("_"):
        if sub.isnumeric() and len(sub) in (6, 7):
            return int(sub)
    raise ValueError(f"Could not find subject ID in {filename}")


def cast_to_dt(
    v: str | int | float | datetime.datetime | datetime.date,
    allow_missing_time: bool = True,
) -> datetime.datetime:
    """Try to create a datetime object from the input that is sensible as a
    timestamp for a session.

    >>> a = cast_to_dt(20220425150237)
    >>> b = cast_to_dt('2022-04-25T15:02:37')
    >>> c = cast_to_dt(1650924157.0)
    >>> d = cast_to_dt(datetime.datetime(2022, 4, 25, 15, 2, 37))
    >>> e = cast_to_dt('366122_20220425_150237')
    >>> f = cast_to_dt('366122_2022-04-25T15:02:37')
    >>> g = cast_to_dt('366122_2022-04-25T15:02:37_366122')
    >>> a == b == c == d == e == f == g
    True

    >>> cast_to_dt(20220425, allow_missing_time=True)
    datetime.datetime(2022, 4, 25, 0, 0)
    >>> cast_to_dt(20220425, allow_missing_time=False) # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    ValueError: Cannot convert 20220425 to datetime.datetime when allow_missing_time=False
    """

    if not v:
        raise ValueError(f"Null type not an accepted input: {type(v) = }")
    if isinstance(v, datetime.datetime):
        return validate_dt(v)
    elif isinstance(v, datetime.date):
        if not allow_missing_time:
            raise ValueError(
                "Cannot convert datetime.date to datetime.datetime when ALLOW_TIMELESS_DT=False"
            )
        return validate_dt(datetime.datetime.combine(v, datetime.time()))
    elif isinstance(v, float) or (isinstance(v, int) and len(str(v)) == 10):
        # int len=10 timestamp (from time.time()) corresponds to date in year range 2001 - 2286
        return validate_dt(datetime.datetime.fromtimestamp(v))

    try:
        s = str(v)
    except TypeError as exc:
        raise ValueError(
            f"Input must be a datetime.datetime, float, int or str. Got {type(v)}"
        ) from exc

    substrings = (
        sorted(
            subslices(re.sub("[^0-9_]", "", "".join(s)).split("_")),
            key=lambda x: len(x),
            reverse=True,
        )
        if "_" in s
        else (s,)
    )
    # convention is date then time (20220425_150237), but if we pass the unsorted
    # substrings when `allow_missing_time` is True, the date with default time will
    # be returned
    if len(substrings) > 1:
        for strings in substrings:
            with contextlib.suppress(ValueError):
                return validate_dt(
                    cast_to_dt(
                        re.sub("[^0-9]", "", "".join(strings)),
                        allow_missing_time,
                    )
                )
    with contextlib.suppress(ValueError):
        dt = datetime.datetime.fromisoformat(s)
        if dt.time() == datetime.time() and not allow_missing_time:
            raise ValueError
        return dt
    with contextlib.suppress(ValueError):
        return datetime.datetime.fromisoformat(f"{s[:8]}T{s[8:14]}")
    with contextlib.suppress(ValueError):
        return datetime.datetime.fromisoformat(f"{s[:8]}T{s[8:12]}")
    with contextlib.suppress(ValueError):
        if allow_missing_time:
            return datetime.datetime.fromisoformat(f"{s[:8]}")
    raise ValueError(
        f"Cannot convert {v!r} to datetime.datetime: possibly because {allow_missing_time=}"
    )


def validate_dt(dt: datetime.datetime) -> datetime.datetime:
    """Raise ValueError if the date doesn't make sense for a previous
    Neuropixels session at the Allen.

    >>> validate_dt(datetime.datetime(2022, 4, 25, 15, 2, 37))
    datetime.datetime(2022, 4, 25, 15, 2, 37)
    >>> validate_dt(datetime.datetime(2010, 4, 25, 15, 2, 37)) # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: Invalid year: 2010
    >>> validate_dt(datetime.datetime(2050, 4, 25, 15, 2, 37)) # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: Invalid year: 2050
    """
    if not 2015 < dt.year <= datetime.datetime.now().year:
        raise ValueError(f"Invalid year: {dt.year}")
    return dt


T = TypeVar("T")


def subslices(seq: Sequence[T]) -> tuple[Sequence[T], ...]:
    """Return all contiguous non-empty subslices of a sequence.
    >>> subslices('ABCD')
    ('A', 'AB', 'ABC', 'ABCD', 'B', 'BC', 'BCD', 'C', 'CD', 'D')
    >>> subslices([1, 2])
    ([1], [1, 2], [2])
    """
    slices = itertools.starmap(slice, itertools.combinations(range(len(seq) + 1), 2))
    return tuple(map(operator.getitem, itertools.repeat(seq), slices))


if __name__ == "__main__":
    import doctest

    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
