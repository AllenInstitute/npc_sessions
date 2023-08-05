from __future__ import annotations

import contextlib
import datetime
import itertools
import operator
import re
from collections.abc import Sequence
from typing import TypeVar

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
