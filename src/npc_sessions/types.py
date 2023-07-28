from __future__ import annotations

import collections.abc
import contextlib
import datetime
import functools
import inspect
import itertools
import operator
import re
from typing import Any, ClassVar, Collection, Container, Iterable, Mapping, Optional, Protocol, Sequence, Set, Sized, TypeAlias, TypeVar, Union

from typing_extensions import Self


class SupportsID(Protocol):
    id: int | str
    
class MetadataRecord:
    """A base class for minimal info to define a record, using a unique `id`
    (unique for the type, not necessarily globally).
    Intended use is for looking-up and referring-to metadata components
    in databases, regardless of the backend.

    Provides magic methods for str, equality, hashing and ordering:

    >>> a = MetadataRecord(id=1)
    >>> a.id
    1
    >>> str(a)
    '1'
    >>> a == 1 and a == '1'
    True
    >>> b = MetadataRecord(id=1)
    >>> c = MetadataRecord(id=2)
    >>> a == b
    True
    >>> a == c
    False
    >>> sorted([c, b])
    [MetadataRecord(id=1), MetadataRecord(id=2)]
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
    """
    id: str | int


class SessionSpec:
    """Used to specify individual sessions.

    A session comprises some combination of:
    - `id`: a descriptive id or lims/db primary key
    - `dt`: a date or date+time
    - `subject`: labtracks id or similar
    - `project`: descriptive id or lims code
    
    Usage:
    
    >>> SessionSpec(id=1)
    SessionSpec(id=1)
    
    Variables not assigned at init won't be available:
    >>> SessionSpec(id=1).project is None
    True

    If too few properties are specified, we can't identify a session:
    >>> SessionSpec(project='DR')           # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    TypeError: Not enough arguments to disambiguate session. 
    Provide all of the arguments in one of these groups:
    (('id',), ('dt',), ('subject', 'dt'))

    `dt` can be provided in many ways, with validation:
    >>> a = SessionSpec(dt=20220425150237)
    >>> b = SessionSpec(dt='2022-04-25T15:02:37')
    >>> c = SessionSpec(dt=1650924157.0)
    >>> d = SessionSpec(dt=datetime.datetime(2022, 4, 25, 15, 2, 37))
    >>> a == b == c == d
    True

    `date` and `time` follow from `dt`:
    >>> s = SessionSpec(dt=20220425150237)
    >>> s.date, s.time
    (datetime.date(2022, 4, 25), datetime.time(15, 2, 37))

    ...except in the case of a placeholder time being used, explicitly or implicitly:
    >>> s = SessionSpec(subject=366122, dt=20220425)
    >>> s.dt, s.time
    (datetime.datetime(2022, 4, 25, 0, 0), None)

    If `id` contains a datetime, it will be used to populate `dt`:
    >>> s = SessionSpec('366122_20220425_150237')
    >>> s.id, s.dt
    ('366122_20220425_150237', datetime.datetime(2022, 4, 25, 15, 2, 37))

    Sorting is based on id if available, then (subject, dt):
    >>> sorted([SessionSpec(id=2), SessionSpec(id=1)])
    [SessionSpec(id=1), SessionSpec(id=2)]
    >>> sorted([SessionSpec(subject=2, dt=20221231), SessionSpec(subject=2, dt=20220101), SessionSpec(id=1, subject=1)])
    [SessionSpec(id=1, subject=1), SessionSpec(dt=2022-01-01 00:00:00, subject=2), SessionSpec(dt=2022-12-31 00:00:00, subject=2)]

    Properties can only be written on init, then are read-only (for hashability):
    >>> SessionSpec(id=1).project = 'DR'    # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    AttributeError: SessionSpec.id is already assigned. Create a new instance.
    >>> SessionSpec(id=1).id = 2            # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    AttributeError: SessionSpec.id is already assigned. Create a new instance.
    
    Equality uses only the common properties - useful for searching with partial info:
    >>> sessions = (SessionSpec(id=1, subject=1), )
    >>> SessionSpec(id=1) in sessions
    True

    ..and like other record types, is equal to the value of its id:
    >>> 1 == SessionSpec(id=1) and '1' == SessionSpec(id=1)
    True
    >>> 1 in (SessionSpec(id=1, subject=1), )
    True
    """
    
    _required_kwarg_groups: ClassVar[Iterable[Iterable[str]]] = (
        ('id', ),
        ('dt', ), # note that a 
        ('subject', 'dt'),
    )
    """At least one of these groups of kwargs must be provided to disambiguate a
    session"""
    
    IS_DEFAULT_TIME_ALLOWED: ClassVar[bool] = True
    """If True, an input to `dt` without a time will not be rejected, but will 
    have the default time of 00:00 on that date, as per the behaviour of 
    `datetime.datetime()`. 

    - should be False if subjects undergo multiple subjects per day
    """

    IS_DATE_WITHOUT_TIME_ALLOWED: ClassVar[bool] = False
    """If True, a session can be specified using just a date. If False (default)
    a `ValueError` will be raised on init when a date alone is the only input.

    - should only be True when:
        - there's maximum one session per day (across all subjects, rigs)
    """

    def __init__(
        self,
        id: Optional[int | str | datetime.datetime | SupportsID] = None,
        dt: Optional[int | float | str | datetime.datetime] = None,
        subject: Optional[int | str | SupportsID] = None,
        project: Optional[int | str | SupportsID] = None,
    ) -> None:
        kwargs = {k: v for k, v in locals().items() if k in self._kwarg_names}
        self._validate_kwargs(kwargs)
        kwargs = self._get_dt_from_id_if_missing(kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)        
        self._raise_if_date_only_input(kwargs)

    def _get_dt_from_id_if_missing(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """If `id` contains a valid timestamp it will be assigned to `dt`.
        """
        with contextlib.suppress(TypeError, ValueError):
            self.is_date_from_id = False
            if not kwargs['dt']:
                kwargs['dt'] = cast_to_dt(kwargs['id'], self.IS_DEFAULT_TIME_ALLOWED)
                self.is_date_from_id = True
        return kwargs
    
    def _validate_kwargs(self, kwargs: dict[str, Any]) -> None:    
        if not any(kwargs.values()):
            raise TypeError(f"At least one argument to {self.__class__} must be provided: {tuple(kwargs.keys())}")
        for required_kwargs in self._required_kwarg_groups:
            if all(k in kwargs and kwargs[k] is not None for k in required_kwargs):
                break
        else:
            raise TypeError(
                "Not enough arguments to disambiguate session.\n"
                "Provide all of the arguments in one of these groups:\n"
                f"{self._required_kwarg_groups}",
            )
    
    def _raise_if_date_only_input(self, kwargs: dict[str, Any]) -> None:
        if not self.IS_DATE_WITHOUT_TIME_ALLOWED:
            input_args = tuple(k for k in kwargs if kwargs[k] is not None)
            is_date_only_input = (input_args == ('dt',)) #or (input_args == ('id', 'dt') and self.is_date_from_id)
            is_date_without_time = self.time is None
            if is_date_only_input and is_date_without_time:
                raise TypeError(
                    "Date alone without time is not enough to disambiguate session."
                    "If you know it is enough (only one session per day, across all subjects)"
                    f"you can toggle {self.IS_DATE_WITHOUT_TIME_ALLOWED=}",
                )

    @property
    def _kwarg_names(self) -> tuple[str, ...]:
        return tuple(inspect.signature(type(self).__init__).parameters)[1:]
    
    @property
    def _available_attrs(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self._kwarg_names if getattr(self, k) is not None}
    
    def _raise_if_assigned(self, attr: str) -> None:
        hidden_attr = f"_{attr.lstrip('_')}"
        if hasattr(self, hidden_attr):
            raise AttributeError(f"{self.__class__.__name__}.{attr} is already assigned. Create a new instance.")

    def __repr__(self) -> str:
        attrs = ", ".join(f"{k}={repr(v) if isinstance(v, (str, int)) else v}" for k, v in self._available_attrs.items())
        return f"{self.__class__.__name__}({attrs})"

    def __str__(self) -> str:
        if self.id is not None:
            return str(self.id)
        return repr(self)

    def __hash__(self) -> int:
        if self.id is not None:
            return hash(self.id) ^ hash(self.__class__.__name__)
        return NotImplemented
        # h = 0
        # for value in self._available_attrs.values():
        #     h ^= hash(value)
        # return h

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
        if self.id is not None and isinstance(other, (int, str)):
            return str(self.id) < str(other)
        if self.dt is not None and isinstance(other, datetime.datetime):
            return self.dt < other
            # compare the first common attribute found, in rev order of __init__ signature
        for attr in self._kwarg_names:
            self_val = getattr(self, attr)
            if self_val is None:
                continue
            other_val = getattr(other, attr, None)
            if str(self_val) == str(other_val):
                # sorts in groups of same project, same subject, etc.
                continue
            return str(self_val) < str(other_val)
            # return False
        return NotImplemented
        
    @property
    def id(self) -> int | str | None:
        """A unique identifier for the object.
        Write once. Type at assignment is preserved.
        """
        return self._id
    
    @id.setter
    def id(self, value: int | str | SupportsID) -> None:
        self._raise_if_assigned("id")
        if hasattr(value, "id"):
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
        self._raise_if_assigned("dt")
        self._dt = cast_to_dt(value, self.IS_DEFAULT_TIME_ALLOWED) if value is not None else None
        
    @property
    def project(self) -> ProjectRecord | None:
        return self._project
    
    @project.setter
    def project(self, value: int | str | ProjectRecord) -> None:
        self._raise_if_assigned("project")
        self._project = ProjectRecord(value) if value is not None else None
        
    @property
    def subject(self) -> SubjectRecord | None:
        return self._subject
    
    @subject.setter
    def subject(self, value: int | str | SubjectRecord) -> None:
        self._raise_if_assigned("subject")
        self._subject = SubjectRecord(value) if value is not None else None
    
    @property
    def date(self) -> datetime.date | None:
        return self.dt.date() if self.dt is not None else None

    @property
    def time(self) -> datetime.time | None:
        """Returns `None` if `dt.time()` is default 00:00:00"""
        if self.dt is None or self.dt.time() == datetime.time():
            return None
        return self.dt.time()
    
    ArgsT: TypeAlias = Union[
        int, str, datetime.datetime, SupportsID,
    ]


class SessionMapping(collections.abc.Mapping):
    
    _cache: dict[SessionSpec, Any]

    def __getitem__(self, key: SessionSpec.ArgsT | SessionSpec) -> Any:
        if key is None:
            raise KeyError("Cannot get a session with a null key")
        key = SessionSpec(id=key) if not isinstance(key, SessionSpec) else key
        return self._cache.__getitem__(key)
    
    def __iter__(self):
        ...
    def __len__(self):
        ...

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
    >>> a == b == c == d == e == f
    True

    >>> cast_to_dt(20220425, allow_missing_time=True)
    datetime.datetime(2022, 4, 25, 0, 0)
    >>> cast_to_dt(20220425, allow_missing_time=False) # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):      
    ...
    ValueError: Cannot convert 20220425 to datetime.datetime when allow_missing_time=False

    """
    def validate(dt: datetime.datetime) -> datetime.datetime:
        if not 2015 < dt.year < 2030:
            raise ValueError(f"Invalid year: {dt.year}")
        return dt
    if not v:
        raise ValueError(f'Null type not an accepted input: {type(v) = }')
    if isinstance(v, datetime.datetime):
        return v
    elif isinstance(v, datetime.date):
        if not allow_missing_time:
            raise ValueError("Cannot convert datetime.date to datetime.datetime when ALLOW_TIMELESS_DT=False")
        return datetime.datetime.combine(v, datetime.time())
    elif isinstance(v, float) or (isinstance(v, int) and len(str(v)) == 10):
        # int len=10 timestamp (from time.time()) corresponds to date in year range 2001 - 2286
        return validate(datetime.datetime.fromtimestamp(v))
        
    try:
        s = str(v)
    except TypeError as exc:
        raise ValueError(
            f"Input must be a datetime.datetime, float, int or str. Got {type(v)}"
        ) from exc

    substrings = sorted(subslices(s.split('_')), key=lambda x: len(x), reverse=True)
    # convention is date then time (20220425_150237), but if we pass the unsorted 
    # substrings when `allow_missing_time` is True, the date with default time will
    # be returned
    for strings in substrings:
        with contextlib.suppress(Exception):
            return cast_to_dt(
                re.sub("[^0-9]", "", ''.join(strings)), 
                allow_missing_time,
                )
    with contextlib.suppress(Exception):
        return datetime.datetime.fromisoformat(
            f"{s[:8]}T{s[8:14]}"
        )
    with contextlib.suppress(Exception):
        return datetime.datetime.fromisoformat(
            f"{s[:8]}T{s[8:12]}"
        )
    with contextlib.suppress(Exception):
        if allow_missing_time:
            return validate(datetime.datetime.fromisoformat(f"{s[:8]}"))
    raise ValueError(f"Cannot convert {v!r} to datetime.datetime: possibly because {allow_missing_time=}")

T = TypeVar('T')
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
    # cast_to_dt(20220425, allow_missing_time=True)
    cast_to_dt('2022-04-25T15:02:37')
    import doctest
    doctest.testmod(optionflags=(
        doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE
    ))
    SessionSpec(id=1, subject=1),