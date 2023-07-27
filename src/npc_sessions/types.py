from __future__ import annotations

import contextlib
import datetime
import functools
from typing import Any, Iterable, Optional, Union

from typing_extensions import Self

import pydantic

ALLOW_TIMELESS_DT: bool = True
"""Whether to allow a date without time to define a session (given another
required arguments, such as subject id). Should be False if subjects ever
undergo multiple sessions per day."""

class MetadataRecord:
    """A base class for minimal info to define a record, using a unique `id`
    (unique for the type, not necessarily globally).
    Intended use is for looking-up and referring-to metadata components
    in databases, regardless of the backend.

    Provides magic methods for str, equality, hashing and ordering:

    >>> a = MetadataKey(id=1)
    >>> a.id
    1
    >>> str(a)
    '1'
    >>> a == 1 and a == '1'
    True
    >>> b = MetadataKey(id=1)
    >>> c = MetadataKey(id=2)
    >>> a == b
    True
    >>> a == c
    False
    >>> sorted([c, b])
    [MetadataKey(id=1), MetadataKey(id=2)]
    """

    def __init__(self, id: int | str | Self) -> None:
        if isinstance(id, MetadataRecord):
            id = id.id
        if not isinstance(id, (int, str)):
            raise TypeError(f"{__class__.__name__}.id must be int or str, not {type(id)}")
        self.id = id

    @property
    def id(self) -> int | str:
        """A unique identifier for the object.
        Read-only, and type at assignment is preserved.
        """
        return self._id
    
    @id.setter
    def id(self, value: int | str | Self)-> None:
        """A unique identifier for the object.
        Read-only, and type at assignment is preserved.
        """
        if hasattr(self, "_id"):
            raise AttributeError(f"{__class__.__name__}.id is read-only")
        if isinstance(value, MetadataRecord):
            value = value.id
        self._id = value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={repr(self.id)})"

    def __str__(self) -> str:
        return str(self.id)

    def __hash__(self) -> int:
        return hash(self.id) ^ hash(self.__class__.__name__)

    def __eq__(self, other: Any) -> bool:
        try:
            # allow comparison of obj with str or int, via obj.id
            return str(self) == str(other)
        except TypeError:
            return NotImplemented

    def __lt__(self, other: Any) -> bool:
        try:
            return str(self) < str(other)
        except TypeError:
            return NotImplemented


class ProjectRecord(MetadataRecord):
    """To uniquely define a project we need:
    - `id`: a lims code or descriptive string name

    >>> p = ProjectRecord(id='DR')
    >>> p
    ProjectRecord(id='DR')
    >>> p == 'DR'
    True
    """
    id: str | int


class SubjectRecord(MetadataRecord):
    """To uniquely define a subject we need:
    - `id`: labtracks MID, etc.

    >>> sub = SubjectRecord(id=366122)
    >>> sub
    SubjectRecord(id=366122)
    >>> sub == 366122 and sub == '366122'
    True
    >>> sub.project = 1
    >>> sub = SubjectRecordRecord(id=366122, project=1)
    >>> sub.project
    """
    id: str | int


class SessionRecord:
    """A session comprises some combination of:
    - an id
    - a date or datetime
    - a subject
    - a project
    
    >>> a = SessionRecord(id=1, subject=1, project=1, dt=20220425150237)
    >>> b = SessionRecord(id=1, subject=1, project=1, dt='2022-04-25T15:02:37')
    >>> c = SessionRecord(id=1, subject=1, project=1, dt=1650924157.0)
    >>> a == b == c
    True
    >>> a.date
    datetime.date(2022, 4, 25)

    Sorting is based on id, not dt:
    >>> d = SessionRecord(id=2, subject=1, project=1, dt=0650924157.0)
    >>> sorted([d, c])
    [SessionRecord(id=1), SessionRecord(id=2)]
    """
    
    _required_kwarg_groups: Iterable[Iterable[str]] = (
        ('id', ),
        ('dt', ),
        ('subject', 'dt', ),
        # ('date', 'subject', ) if ALLOW_TIMELESS_DT else ('date', 'subject', 'time', ),
    )
    """At least one of these sets of kwargs must be provided to disambiguate a
    session"""
    
    def __init__(
        self,
        id: Optional[int | str | MetadataRecord] = None,
        dt: Optional[int | float | str | datetime.datetime] = None,
        subject: Optional[int | str | MetadataRecord] = None,
        project: Optional[int | str | MetadataRecord] = None,
        # date: Optional[int | str | datetime.date] = None,
        # time: Optional[int | str | datetime.time] = None,
    ) -> None:
        kwargs = {k: v for k, v in locals().items() if k in self._kwarg_names}
        if not any(kwargs.values()):
            raise ValueError(f"At least one argument to {self.__class__} must be provided: {tuple(kwargs.keys())}")
        for required_kwargs in self._required_kwarg_groups:
            if all(k in kwargs for k in required_kwargs):
                break
        else:
            raise ValueError(f"Not enough arguments to disambiguate session. Provide all of the arguments in one of these groups:\n{self._required_kwarg_groups}") 
        
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def _kwarg_names(self) -> tuple[str, ...]:
        return type(self).__init__.__code__.co_varnames[1:]
    
    @property
    def _available_attrs(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self._kwarg_names if getattr(self, k) is not None}
    
    def _raise_if_assigned(self, attr: str) -> None:
        if hasattr(self, attr):
            raise AttributeError(f"{__class__.__name__}.{attr} is already assigned. Create a new instance.")

    def __repr__(self) -> str:
        attrs = ", ".join(f"{k}={repr(v)}" for k, v in self._available_attrs.items())
        return f"{self.__class__.__name__}({attrs})"

    def __str__(self) -> str:
        if self.id is not None:
            return str(self.id)
        return repr(self)

    def __hash__(self) -> int:
        if self.id is not None:
            return hash(self.id) ^ hash(self.__class__.__name__)
        h = 0
        for value in self._available_attrs.values():
            h ^= hash(value)
        return h

    def __eq__(self, other: Any) -> bool:
        if self.id is not None:
            return MetadataRecord(self.id) == other
        if (
            self.dt is not None 
            and isinstance(other, datetime.datetime)
        ):
            return self.dt == other
        for group in self._required_kwarg_groups:
            if all(getattr(self, k) == getattr(self, k) for k in group):
                break
        else:
            return False
        return True

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, (int, str)) and self.id is not None:
            return str(self.id) < str(other)
        if isinstance(other, datetime.datetime) and self.dt is not None:
            return self.dt < other
        if isinstance(other, SessionRecord):
            for attr in self._kwarg_names:
                self_val = getattr(self, attr)
                if self_val is None:
                    continue
                other_val = getattr(other, attr)
                if self_val != other_val:
                    return str(self_val) < str(other_val)
            return False
        return NotImplemented
        
    @property
    def id(self) -> int | str | None:
        """A unique identifier for the object.
        Write once. Type at assignment is preserved.
        """
        return self._id
    
    @id.setter
    def id(self, value: int | str | MetadataRecord) -> None:
        self._raise_if_assigned("_id")
        if isinstance(value, MetadataRecord):
            value = value.id
        self._id = value if value is not None else None
    
    @property
    def dt(self) -> datetime.datetime | None:
        """A datetime object for the session.
        Write once. Type at assignment is preserved.
        """
        return self._dt
    
    @dt.setter
    def dt(self, value: int | float | str | datetime.datetime) -> None:
        self._raise_if_assigned("_dt")
        self._dt = cast_to_dt(value)
        
    @property
    def project(self) -> ProjectRecord | None:
        return self._project
    
    @project.setter
    def project(self, value: int | str | ProjectRecord) -> None:
        self._raise_if_assigned("_project")
        self._project = ProjectRecord(value) if value is not None else None
        
    @property
    def subject(self) -> SubjectRecord | None:
        return self._subject
    
    @subject.setter
    def subject(self, value: int | str | SubjectRecord) -> None:
        self._raise_if_assigned("_subject")
        self._subject = SubjectRecord(value) if value is not None else None
    
    @property
    def date(self) -> datetime.date | None:
        return self.dt.date() if self.dt is not None else None

    @property
    def time(self) -> datetime.time | None:
        return self.dt.time() if self.dt is not None else None


def cast_to_dt(v: str | int | float | datetime.datetime | datetime.datetime) -> datetime.datetime:
    """Try to create a sensible datetime object from the input."""
    if isinstance(v, datetime.datetime):
        return v
    elif isinstance(v, datetime.date):
        if not ALLOW_TIMELESS_DT:
            raise ValueError("Cannot convert datetime.date to datetime.datetime when ALLOW_TIMELESS_DT=False")
        return datetime.datetime.combine(v, datetime.time())
    elif isinstance(v, float) or (isinstance(v, int) and len(str(v)) == 10):
        # int len=10 timestamp (from `time.time()``)corresponds to date in year range 2001 - 2286
        return datetime.datetime.fromtimestamp(v)
    try:
        s = str(v)
    except TypeError as exc:
        raise ValueError(
            f"Input must be a datetime.datetime, float, int or str. Got {type(v)}"
        ) from exc
    with contextlib.suppress(Exception):
        return datetime.datetime.fromisoformat(v)
    with contextlib.suppress(Exception):
        return datetime.datetime.fromisoformat(
            f"{s[:4]}-{s[4:6]}-{s[6:8]}T{s[8:10]}:{s[10:12]}:{s[12:14]}"
        )
    with contextlib.suppress(Exception):
        return datetime.datetime.fromisoformat(
            f"{s[:4]}-{s[4:6]}-{s[6:8]}T{s[8:10]}:{s[10:12]}"
        )
    with contextlib.suppress(Exception):
        if ALLOW_TIMELESS_DT:
            return datetime.datetime.fromisoformat(f"{s[:4]}-{s[4:6]}-{s[6:8]}")
    raise ValueError(f"Cannot convert {v!r} to datetime.datetime")


if __name__ == "__main__":
    import doctest

    doctest.testmod()
