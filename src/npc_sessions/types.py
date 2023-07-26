from __future__ import annotations
import contextlib

import dataclasses
import datetime
import functools
from typing import Any, Collection, Iterable, Mapping, NamedTuple, Optional, Sequence, TypeAlias, Union

import pydantic


class MetadataKey(pydantic.BaseModel):
    """A base class for all metadata primary keys, defining a unique `id`.
    Intended use is for looking-up and referring-to metadata components 
    in databases, regardless of the backend.

    Provides magic methods for str, equality, hashing and ordering:

    >>> a = MetadataKey(id=1)
    >>> a.id
    1
    >>> str(a)
    '1'
    >>> a == 1 == '1'
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

    id: Union[int, str]
    """A unique identifier for this object. Immutable, and type at assignment is preserved."""
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(id={self.id})'

    def __str__(self) -> str:
        return str(self.id)
    
    def __hash__(self) -> int:
        return hash(self.id) ^ hash(self.__class__.__name__)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, self.__class__):
            return self.id == other.id
        try:
            # allow comparison of obj with str or int via obj.id
            return str(self) < str(other)
        except TypeError:
            return NotImplemented

    def __lt__(self, other: Any) -> bool:
        try:
            return str(self) < str(other)
        except TypeError:            
            return NotImplemented


class Project(MetadataKey):
    """A project has collections of subjects and sessions.
    
    >>> p = Project(id='DR')
    >>> p
    Project(id='DR')
    >>> p == 'DR'
    True
    """

    @functools.cached_property
    def subjects(self) -> Iterable[Union[int, str]]:
        """Return a sequence of all subjects in this project."""
        raise NotImplementedError

    @functools.cached_property
    def sessions(self) -> Iterable[Union[int, str]]:
        """Return a sequence of all sessions in this project."""
        raise NotImplementedError



class Subject(MetadataKey):
    """A subject has a project and a collection of sessions.
    
    >>> sub = Subject(id=366122)
    >>> sub
    Subject(id=366122)
    >>> sub == 366122 == '366122'
    True
    """

    @pydantic.computed_field(alias='project_id', repr=False)
    @functools.cached_property
    def project(self) -> Project: 
        raise NotImplementedError

    @functools.cached_property
    def sessions(self) -> Iterable[Union[int, str]]:
        """All sessions in this project."""
        raise NotImplementedError


class Session(MetadataKey):
    """A session comprises, at the very least, a subject, a datetime and a project.
    
    >>> a = Session(id=1, subject=1, project=1, dt=20220425150237)
    >>> b = Session(id=1, subject=1, project=1, dt='2022-04-25T15:02:37')
    >>> c = Session(id=1, subject=1, project=1, dt=1650924157.0)
    >>> a == b == c
    True
    >>> a.date
    datetime.date(2022, 4, 25)

    Sorting is based on id, not dt:
    >>> d = Session(id=2, subject=1, project=1, dt=0650924157.0)
    >>> sort([d, c])
    [Session(id=1), Session(id=2)]
    """

    subject: Union[int, str]

    @pydantic.computed_field(alias='project_id', repr=False)
    @functools.cached_property    
    def project(self) -> Project: 
        raise NotImplementedError

    dt: datetime.datetime
    """will be cast as datetime.datetime objects"""

    @pydantic.field_validator('dt', mode='before')
    def _validate_dt(cls, v: Any) -> datetime.datetime: # pylint: disable=no-self-argument
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
        raise ValueError(f'Input must be a datetime.datetime, float, int or str. Got {type(v)}')
    
    with contextlib.suppress(Exception):
        return datetime.datetime.fromisoformat(v)
    s = str(v)
    with contextlib.suppress(Exception):
        return datetime.datetime.fromisoformat(
            f'{s[:4]}-{s[4:6]}-{s[6:8]}T{s[8:10]}:{s[10:12]}:{s[12:14]}'
            )
    with contextlib.suppress(Exception):
        return datetime.datetime.fromisoformat(
            f'{s[:4]}-{s[4:6]}-{s[6:8]}T{s[8:10]}:{s[10:12]}'
            )
    raise ValueError(f'Could not convert {v} to datetime.datetime')

if __name__ == '__main__':
    import doctest
    doctest.testmod()