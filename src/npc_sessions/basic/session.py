from __future__ import annotations
import contextlib

import datetime
from typing import Any, Iterable, Union

import pydantic


class Metadata(pydantic.BaseModel):
    """A base class for all metadata objects, defining a unique `id`.

    Provides magic methods for str, equality, hashing and ordering:

    >>> a = Metadata(id=1)
    >>> a.id
    1
    >>> str(a)
    '1'
    >>> b = Metadata(id=1)
    >>> c = Metadata(id=2)
    >>> a == b
    True
    >>> a == c
    False
    >>> sorted([c, b])
    [Metadata(id=1), Metadata(id=2)]
    """

    id: Union[int, str]
    """A unique identifier for this object."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id})"

    def __str__(self) -> str:
        return str(self.id)

    def __hash__(self) -> int:
        return hash(self.id) ^ hash(self.__class__.__name__)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.id == other.id

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return str(self.id) < str(other.id)


class Project(Metadata):
    """A project has collections of subjects and sessions."""

    @property
    def subject_ids(self) -> Iterable[Union[int, str]]:
        """Return a sequence of all subject_ids in this project."""
        return NotImplemented

    @property
    def session_ids(self) -> Iterable[Union[int, str]]:
        """Return a sequence of all session_ids in this project."""
        return NotImplemented


class Subject(Metadata):
    """A subject has a project and a collection of sessions."""

    project_id: Union[int, str]

    @property
    def session_ids(self) -> Iterable[Union[int, str]]:
        """Return a sequence of all session_ids in this project."""
        return NotImplemented


class Session(Metadata):
    """A session comprises, at the very least, a subject, a datetime and a project.

    >>> s = Session(id=1, subject_id=1, project_id=1, dt=20220425150237)
    >>> s.date
    datetime.date(2022, 4, 25)
    """

    subject_id: Union[int, str]
    project_id: Union[int, str]
    dt: datetime.datetime
    """will be cast as datetime.datetime objects"""

    @pydantic.field_validator("dt", mode="before")
    def _validate_dt(
        cls, v: Any
    ) -> datetime.datetime:  # pylint: disable=no-self-argument
        return cast_to_dt(v)

    @property
    def date(self) -> datetime.date:
        return self.dt.date()

    @property
    def time(self) -> datetime.time:
        return self.dt.time()


def cast_to_dt(v: Union[int, float, datetime.datetime]) -> datetime.datetime:
    """Try to create a sensible datetime object from the input."""
    if isinstance(v, datetime.datetime):
        return v
    elif isinstance(v, float) or (isinstance(v, int) and len(str(v)) == 10):
        # int len=10 corresponds to year range 2001 - 2286
        return datetime.datetime.fromtimestamp(v)
    elif not isinstance(v, str):
        raise ValueError(
            f"Input must be a datetime.datetime, float, int or str. Got {type(v)}"
        )

    with contextlib.suppress(Exception):
        return datetime.datetime.fromisoformat(v)
    s = str(v)
    with contextlib.suppress(Exception):
        return datetime.datetime.fromisoformat(
            f"{s[:4]}-{s[4:6]}-{s[6:8]}T{s[8:10]}:{s[10:12]}:{s[12:14]}"
        )
    with contextlib.suppress(Exception):
        return datetime.datetime.fromisoformat(
            f"{s[:4]}-{s[4:6]}-{s[6:8]}T{s[8:10]}:{s[10:12]}"
        )
    raise ValueError(f"Could not convert {v} to datetime.datetime")


if __name__ == "__main__":
    import doctest

    doctest.testmod()
