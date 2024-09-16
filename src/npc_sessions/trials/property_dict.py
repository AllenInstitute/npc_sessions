from __future__ import annotations

import collections.abc
import datetime
import doctest
import functools
import logging
import reprlib
from collections.abc import Generator, Iterator, Sequence
from typing import Any, Literal

import npc_io
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EXCLUDE_IF_ONLY_NANS = True
"""If True, exclude columns with all NaN values from the resulting dict. This is
the behavior expected from nwb validation. For some columns that are not always
available (such as `opto_location_name`) we return `np.full(np.nan)`: if those
columns aren't dropped, then dataframe schemas will become inconsistent (str vs double)"""

EXPLODE_LISTS = True
"""If True, any with values in the resulting dataframe that are list-like (list, tuple, set, np.ndarray), will be
expanded into separate rows, e.g. 

this dataframe:
   start_time  stop_time  trial_idx opto_bregma_x opto_bregma_y                opto_target
0        12.3       13.3          3     [-10, 10]       [2, -2]  [upper left, lower right]
1        15.7       16.7          4       [-5, 5]       [-4, 4]  [lower left, upper right]

becomes:
   start_time  stop_time  trial_idx opto_bregma_x opto_bregma_y  opto_target
0        12.3       13.3          3           -10             2   upper left
1        12.3       13.3          3            10            -2  lower right
2        15.7       16.7          4            -5            -4   lower left
3        15.7       16.7          4             5             4  upper right

via `df.explode(['opto_bregma_x', 'opto_bregma_y', 'opto_target'], ignore_index=True)`

The major assumption here is that all list-like values in the dataframe are of the
same length, and correspond to the same parameter.

This can make it easier to deal with trials with multiple values for a single
location parameter, such as bilateral opto stimulation targets
"""


class PropertyDict(collections.abc.Mapping):
    """Dict type, where the keys are the class's properties (regular attributes
    and property getters) which don't have a leading underscore, and values are
    corresponding property values.

    Methods can be added and used as normal, they're just not visible in the
    dict.
    """

    @property
    def _properties(self) -> tuple[str, ...]:
        """Names of properties without leading underscores. No methods.
        Excludes properties with all NaN values."""
        self_and_super_attrs = tuple(
            attr
            for cls in reversed(self.__class__.__mro__)
            for attr in cls.__dict__.keys()
        )
        dict_attrs = collections.abc.Mapping.__dict__.keys()
        no_dict_attrs = tuple(
            attr for attr in self_and_super_attrs if attr not in dict_attrs
        )
        no_leading_underscore = tuple(attr for attr in no_dict_attrs if attr[0] != "_")
        no_functions = tuple(
            attr for attr in no_leading_underscore if not callable(getattr(self, attr))
        )

        def all_nan(attr):
            return (
                isinstance(a := getattr(self, attr), np.ndarray)
                and a.dtype == np.floating
                and all(np.isnan(a))
            )

        if EXCLUDE_IF_ONLY_NANS:
            return tuple(attr for attr in no_functions if not all_nan(attr))
        return no_functions

    @property
    def _dict(self) -> dict[str, Any]:
        return {attr: getattr(self, attr) for attr in self._properties}

    def __getitem__(self, key) -> Any:
        return self._dict[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._dict)

    def __len__(self) -> int:
        return len(self._dict)

    def __repr__(self) -> str:
        return reprlib.repr(self._dict)

    def to_add_trial_column(
        self,
    ) -> Generator[dict[Literal["name", "description"], str], None, None]:
        """Name and description for each trial column.

        Does not include `start_time` and `stop_time`: those columns
        always exist in an nwb trials table, trying to add them again raises
        an error.

        Iterate over result and unpack each dict:

        >>> for column in obj.to_add_trial_column(): # doctest: +SKIP
        ...    nwb_file.add_trial_column(**column)

        """
        attrs = self._properties
        descriptions = self._docstrings
        missing = tuple(column for column in attrs if column not in descriptions)
        if any(missing):
            logger.warning(
                f"These properties do not have descriptions (add docstrings to their property getters): {missing}"
            )
        descriptions.update(dict(zip(missing, ("" for _ in missing))))
        descriptions.pop("start_time", None)
        descriptions.pop("stop_time", None)
        return (
            {"name": name, "description": description}
            for name, description in descriptions.items()
        )

    def to_add_trial(
        self,
    ) -> Generator[dict[str, int | float | str | datetime.datetime], None, None]:
        """Column name and value for each trial.

        Iterate over result and unpack each dict:

        >>> for trial in obj.to_add_trial(): # doctest: +SKIP
        ...    nwb_file.add_trial(**trial)

        """
        mandatory_columns = ("start_time", "stop_time")
        if any(mandatory not in self._properties for mandatory in mandatory_columns):
            raise AttributeError(
                f"{self} is missing one of {mandatory_columns = } required for nwb trials table"
            )

        for trial in self.to_dataframe().iterrows():
            yield dict(trial[1])

    def to_dataframe(self) -> pd.DataFrame:
        """pandas dataframe"""
        df = pd.DataFrame(
            data={
                k: pd.Series(list(v) if isinstance(v, np.ndarray) else v)
                for k, v in self.items()
            }
        )
        if EXPLODE_LISTS:
            explode_cols = list(
                col
                for col in df.columns
                if isinstance(df[col][0], (list, tuple, set, np.ndarray))
            )
            if explode_cols:
                # all explode_cols should have values with the same dimensions -
                # pandas will raise an error if they don't
                df = df.explode(explode_cols, ignore_index=True)
        return df

    @property
    def _docstrings(self) -> dict[str, str]:
        """Docstrings of property getter methods that have no leading
        underscore in their name.

        - getting the docstring of a regular property/attribute isn't easy:
        if we query its docstring in the same way as a property getter/method,
        we'll just receive the docstring for its value's type.
        """

        def cls_attr(attr):
            return getattr(self.__class__, attr)

        {
            attr: ""
            for attr in self._properties
            if not isinstance(
                cls_attr(attr),
                (property, functools.cached_property, npc_io.cached_property),
            )
        }
        property_getters = {
            attr: cls_attr(attr).__doc__ or ""
            for attr in self._properties
            if isinstance(
                cls_attr(attr),
                (property, functools.cached_property, npc_io.cached_property),
            )
        }

        def fmt(docstring):
            return (
                docstring.replace("\n", " ")
                .replace("  ", "")
                .replace(".- ", "; ")
                .replace("- ", "; ")
                .replace(" ; ", "; ")
                .strip(". ")
            )

        return {
            attr: (
                fmt(cls_attr(attr).__doc__) if cls_attr(attr).__doc__ else ""
            )  # if no docstring present, __doc__ is None
            for attr in property_getters
        }


class TestPropertyDict(PropertyDict):
    """
    >>> obj = TestPropertyDict()
    >>> obj
    {'no_docstring': True, 'visible_property': True, 'visible_property_getter': True}

    >>> obj.invisible_method()
    True

    >>> obj._docstrings
    {'visible_property_getter': 'Docstring available', 'no_docstring': ''}
    """

    visible_property = True

    @property
    def visible_property_getter(self):
        """Docstring available"""
        return True

    @property
    def no_docstring(self):
        return True

    _invisible_property: Any = None

    def invisible_method(self):
        """Docstring not available"""
        return True


class TestPropertyDictInheritance(TestPropertyDict):
    """
    >>> obj = TestPropertyDictInheritance()
    >>> obj
    {'no_docstring': True, 'visible_property': True, 'visible_property_getter': True}
    """

    pass


class TestPropertyDictExports(PropertyDict):
    """
    >>> obj = TestPropertyDictExports()

    >>> for kwargs in obj.to_add_trial_column():
    ...     print(kwargs)
    {'name': 'test_start_time', 'description': 'Start of trial'}
    {'name': 'test_stop_time', 'description': 'End of trial'}

    note: `start_time` and `stop_time` cannot be added as columns (they always
    exist)

    >>> for kwargs in obj.to_add_trial():
    ...     print(kwargs)
    {'start_time': 0.0, 'stop_time': 1.0, 'test_start_time': 1.0, 'test_stop_time': 1.5}
    {'start_time': 1.0, 'stop_time': 2.0, 'test_start_time': 2.0, 'test_stop_time': 2.5}
    """

    start_time = [0, 1]
    stop_time = [1, 2]

    @property
    def test_start_time(self) -> Sequence[float]:
        "Start of trial"
        return [1.0, 2.0]

    @property
    def test_stop_time(self) -> Sequence[float]:
        "End of trial"
        return [start + 0.5 for start in self.test_start_time]


if __name__ == "__main__":
    doctest.testmod()
