from __future__ import annotations

import collections.abc
from collections.abc import Iterator
from typing import Any, Literal


def extract_video_file_name(path: str) -> Literal["eye", "face", "behavior"]:
    names: dict[str, Literal["eye", "face", "behavior"]] = {
        "eye": "eye",
        "face": "face",
        "beh": "behavior",
    }
    return names[next(n for n in names if n in str(path).lower())]


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
