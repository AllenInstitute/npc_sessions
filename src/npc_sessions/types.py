from __future__ import annotations

import collections.abc
import contextlib
import datetime
import inspect
import itertools
import operator
import re
from typing import Any, ClassVar, Container, Iterable, Optional, Protocol, Sequence, Type, TypeAlias, TypeVar, Union
import typing

import pydantic
from typing_extensions import Self

import npc_sessions.utils as utils

@typing.runtime_checkable
class SupportsID(Protocol):
    @property
    def id(self) -> T: ...
    
T = TypeVar("T", int, str)

class MetadataRecord:
    """A base class for definitions of unique metadata records.
    (unique amongst that type of metadata, not necessarily globally).
    
    Intended use is for looking-up and referring-to metadata components
    in databases, regardless of the backend.
    
    - normalizes and validates values where logical
    - allows int or str IDs
    - provides magic methods for str, equality, hashing and ordering:

    >>> a = MetadataRecord(1)
    >>> a
    1
    >>> str(a)
    '1'
    >>> a == 1 and a == '1'
    True
    >>> isinstance(a, (int, str)) 
    False
    >>> b = MetadataRecord(1)
    >>> c = MetadataRecord(2)
    >>> a == b
    True
    >>> a == c
    False
    >>> sorted([c, b])
    [1, 2]
    """
    
    id_type: ClassVar[tuple[Type, ...]] = (int, str)
    """Acceptable types for the id, will be validated on assignment with
    isinstance check."""
    
    def __init__(self, id: int | str | SupportsID) -> None:
        self.id = id.id if isinstance(id, SupportsID) else id

    def parse_id(self, id: T) -> T:
        return id
    
    def validate_id(self, id: int | str) -> None:
        """Raise ValueError if id is not valid for this type of record"""
        if isinstance(id, int):
            if id < 0:
                raise ValueError(f"{self.__class__.__name__}.id must be non-negative, not {id}")
        if isinstance(id, str):
            if not re.match(r"^[a-zA-Z0-9_]+$", id):
                raise ValueError(f"{self.__class__.__name__}.id must be alphanumeric, not {id}")
    
    @property
    def id(self) -> int | str:
        """A unique identifier for the object.
        Read-only, and type at assignment is preserved.
        """
        return self._id
    
    @id.setter
    def id(self, value: int | str)-> None:
        """A unique identifier for the object.
        Read-only, and type at assignment is preserved.
        """
        if not isinstance(value, self.id_type):
            raise TypeError(f"{self.__class__.__name__}.id must be [{self.id_type}], not {type(value)}")
        if hasattr(self, "_id"):
            raise AttributeError(f"{self.__class__.__name__}.id is read-only")
        if isinstance(value, MetadataRecord):
            value = value.id
        self.validate_id(value)
        self._id = value

    def __repr__(self) -> str:
        return repr(self.id)

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
    - `id`: a descriptive string or acronym

    >>> project = ProjectRecord('DR')
    >>> project
    'DR'
    >>> project == 'DR'
    True
    >>> isinstance(project, str)
    False
    """
    id_type: ClassVar[tuple[type, ...]] = (str, )


class SubjectRecord(MetadataRecord):
    """To uniquely define a subject we need:
    - `id`: labtracks MID

    >>> subject = SubjectRecord('366122')
    >>> subject = SubjectRecord(366122)
    >>> subject
    366122
    >>> subject == 366122 and subject == '366122'
    True
    >>> isinstance(subject, (int, str)) 
    False
    """
    id_type: ClassVar[tuple[type, ...]] = (int, str)


class SessionRecord(MetadataRecord):
    """To uniquely define a subject we need:
    - `id`: a 2 or 3-part underscore-separated str corresponding to:
    
        <subject_id>_<date>_<index [optional]>
        
        where:
        - index corresponds to the index of the session on that date for that
        subject
        - if the 3rd component is missing, it is implicitly assumed to be the only 
        session on the date for that subject
        - index currently isn't needed, but is included for future-proofing
        
    >>> session = SessionRecord('366122_20220425')
    >>> session
    '366122_20220425'
    >>> session.subject, session.date, session.index
    (366122, '2022-04-25', 0)
    
    >>> SessionRecord('366122_20220425_0') == '366122_20220425'
    True
    >>> SessionRecord('366122_20220425_0') in ['366122_20220425']
    True
    
    subject and date validated on init:
    >>> SessionRecord('1_20220425') 
    Traceback (most recent call last):
    ...
    ValueError: SessionRecord.id must be in format <subject_id>_<date>_<index [optional]>
    
    date must make sense:
    >>> SessionRecord('366122_20221325') 
    Traceback (most recent call last):
    ...
    ValueError: SessionRecord.id must be in format <subject_id>_<date>_<index [optional]>
    """
    id: ClassVar[str]
    id_type: ClassVar = (str, )
    id_regex: ClassVar = re.compile(
        r"^[0-9]{6,7}_20[2-9][0-9](0[1-9])|(11)|(12)[0-3][0-9](_[0-9]+)?$"
        )
    
    def validate_id(self, id: int | str) -> None:
        super().validate_id(id)
        if not isinstance(id, str) or not self.id_regex.match(id):
            raise ValueError(
                f"{self.__class__.__name__}.id must be in format <subject_id>_<date>_<index [optional]>, not {id}"
            )
            
    @property
    def subject(self) -> SubjectRecord:
        return SubjectRecord(int(self.id.split('_')[0]))
    
    @property
    def date(self) -> str:
        date = self.id.split('_')[1]
        return f"{date[:4]}-{date[4:6]}-{date[6:8]}"
    
    @property
    def index(self) -> int:
        try:
            return int(self.id.split('_')[2])
        except IndexError:
            return 0
    
    @property
    def with_index(self) -> SessionRecord:
        """Verbose form with index included"""
        return SessionRecord(f"{'_'.join(self.id.split('_')[:2])}_{self.index}")
    
    def __eq__(self, other: Any) -> bool:
        if self.index == 0:
            try:
                return self.id.split('_')[:2] == str(other).split('_')[:2]
            except IndexError:
                return False
        return super().__eq__(other)
    
    def __hash__(self) -> int:
        return hash(self.with_index) ^ hash(self.__class__.__name__)
    
# class SessionSpec:
#     """Used to specify individual sessions.

#     A session comprises some combination of:
#     - `id`: a descriptive id or lims/db primary key
#     - `dt`: a date or date+time
#     - `subject`: labtracks id or similar
#     - `project`: descriptive id or lims code
    
#     Usage:
    
#     >>> SessionSpec(id=1)
#     SessionSpec(id=1)
    
#     Variables not assigned at init won't be available:
#     >>> SessionSpec(id=1).project is None
#     True

#     If too few properties are specified, we can't identify a session:
#     >>> SessionSpec(project='DR')           # doctest: +IGNORE_EXCEPTION_DETAIL
#     Traceback (most recent call last):
#     ...
#     TypeError: Not enough arguments to disambiguate session. 
#     Provide all of the arguments in one of these groups:
#     (('id',), ('dt',), ('subject', 'dt'))

#     `dt` can be provided in many ways, with validation:
#     >>> a = SessionSpec(dt=20220425150237)
#     >>> b = SessionSpec(dt='2022-04-25T15:02:37')
#     >>> c = SessionSpec(dt=1650924157.0)
#     >>> d = SessionSpec(dt=datetime.datetime(2022, 4, 25, 15, 2, 37))
#     >>> a == b == c == d
#     True

#     `date` and `time` follow from `dt`:
#     >>> s = SessionSpec(dt=20220425150237)
#     >>> s.date, s.time
#     (datetime.date(2022, 4, 25), datetime.time(15, 2, 37))

#     ...except in the case of a placeholder time being used, explicitly or implicitly:
#     >>> s = SessionSpec(subject=366122, dt=20220425)
#     >>> s.dt, s.time
#     (datetime.datetime(2022, 4, 25, 0, 0), None)

#     If `id` contains a datetime, it will be used to populate `dt`:
#     >>> s = SessionSpec('366122_20220425_150237')
#     >>> s.id, s.dt
#     ('366122_20220425_150237', datetime.datetime(2022, 4, 25, 15, 2, 37))

#     Sorting is based on id if available, then (subject, dt):
#     >>> sorted([SessionSpec(id=2), SessionSpec(id=1)])
#     [SessionSpec(id=1), SessionSpec(id=2)]
#     >>> sorted([SessionSpec(subject=2, dt=20221231), SessionSpec(subject=2, dt=20220101), SessionSpec(id=1, subject=1)])
#     [SessionSpec(id=1, subject=1), SessionSpec(dt=2022-01-01 00:00:00, subject=2), SessionSpec(dt=2022-12-31 00:00:00, subject=2)]

#     Properties can only be written on init, then are read-only (for hashability):
#     >>> SessionSpec(id=1).project = 'DR'    # doctest: +IGNORE_EXCEPTION_DETAIL
#     Traceback (most recent call last):
#     ...
#     AttributeError: SessionSpec.id is already assigned. Create a new instance.
#     >>> SessionSpec(id=1).id = 2            # doctest: +IGNORE_EXCEPTION_DETAIL
#     Traceback (most recent call last):
#     ...
#     AttributeError: SessionSpec.id is already assigned. Create a new instance.
    
#     Equality uses only the common properties - useful for searching with partial info:
#     >>> sessions = (SessionSpec(id=1, subject=1), )
#     >>> SessionSpec(id=1) in sessions
#     True

#     ..and like other record types, is equal to the value of its id:
#     >>> 1 == SessionSpec(id=1) and '1' == SessionSpec(id=1)
#     True
#     >>> 1 in (SessionSpec(id=1, subject=1), )
#     True
#     """
    
#     _required_kwarg_groups: ClassVar[Iterable[Iterable[str]]] = (
#         ('id', ),
#         ('dt', ), # date+time ok: date without other args will raise unless IS_DATE_WITHOUT_TIME_ALLOWED
#         ('subject', 'dt'),
#     )
#     """At least one of these groups of kwargs must be provided to disambiguate a
#     session"""
    
#     IS_DEFAULT_TIME_ALLOWED: ClassVar[bool] = True
#     """If True, an input to `dt` without a time will not be rejected, but will 
#     have the default time of 00:00 on that date, as per the behaviour of 
#     `datetime.datetime()`. 

#     - should be False if subjects undergo multiple subjects per day
#     """

#     IS_DATE_WITHOUT_TIME_ALLOWED: ClassVar[bool] = False
#     """If True, a session can be specified using just a date. If False (default)
#     a `ValueError` will be raised on init when a date alone is the only input.

#     - should only be True when:
#         - there's maximum one session per day (across all subjects, rigs)
#     """
    
#     ArgTypes: TypeAlias = Union[
#         int, str, datetime.datetime, SupportsID,
#     ]
    
#     def __init__(
#         self,
#         id: Optional[int | str | datetime.datetime | SupportsID] = None,
#         dt: Optional[int | float | str | datetime.datetime] = None,
#         subject: Optional[int | str | SupportsID] = None,
#         project: Optional[int | str | SupportsID] = None,
#     ) -> None:
#         kwargs = {k: v for k, v in locals().items() if k in self._kwarg_names}
#         self._validate_kwargs(kwargs)
#         kwargs = self._get_dt_from_id_if_missing(kwargs)
#         for k, v in kwargs.items():
#             setattr(self, k, v)        
#         self._raise_if_date_only_input(kwargs)

#     def _get_dt_from_id_if_missing(self, kwargs: dict[str, Any]) -> dict[str, Any]:
#         """If `id` contains a valid timestamp it will be assigned to `dt`.
#         """
#         with contextlib.suppress(TypeError, ValueError):
#             self.is_date_from_id = False
#             if not kwargs['dt']:
#                 kwargs['dt'] = utils.cast_to_dt(kwargs['id'], self.IS_DEFAULT_TIME_ALLOWED)
#                 self.is_date_from_id = True
#         return kwargs
    
#     def _validate_kwargs(self, kwargs: dict[str, Any]) -> None:    
#         if not any(kwargs.values()):
#             raise TypeError(f"At least one argument to {self.__class__} must be provided: {tuple(kwargs.keys())}")
#         for required_kwargs in self._required_kwarg_groups:
#             if all(k in kwargs and kwargs[k] is not None for k in required_kwargs):
#                 break
#         else:
#             raise TypeError(
#                 "Not enough arguments to disambiguate session.\n"
#                 "Provide all of the arguments in one of these groups:\n"
#                 f"{self._required_kwarg_groups}",
#             )
    
#     def _raise_if_date_only_input(self, kwargs: dict[str, Any]) -> None:
#         if not self.IS_DATE_WITHOUT_TIME_ALLOWED:
#             input_args = tuple(k for k in kwargs if kwargs[k] is not None)
#             is_date_only_input = (input_args == ('dt',)) #or (input_args == ('id', 'dt') and self.is_date_from_id)
#             is_date_without_time = self.time is None
#             if is_date_only_input and is_date_without_time:
#                 raise TypeError(
#                     "Date alone without time is not enough to disambiguate session."
#                     "If you know it is enough (only one session per day, across all subjects)"
#                     "for the dataset in question, you can toggle "
#                     f"{self.IS_DATE_WITHOUT_TIME_ALLOWED=}",
#                 )

#     @property
#     def _kwarg_names(self) -> tuple[str, ...]:
#         return tuple(inspect.signature(type(self).__init__).parameters)[1:]
    
#     @property
#     def _available_attrs(self) -> dict[str, Any]:
#         return {k: getattr(self, k) for k in self._kwarg_names if getattr(self, k) is not None}
    
#     def _raise_if_assigned(self, attr: str) -> None:
#         hidden_attr = f"_{attr.lstrip('_')}"
#         if hasattr(self, hidden_attr):
#             raise AttributeError(f"{self.__class__.__name__}.{attr} is already assigned. Create a new instance.")

#     def __repr__(self) -> str:
#         attrs = ", ".join(f"{k}={repr(v) if isinstance(v, (str, int)) else v}" for k, v in self._available_attrs.items())
#         return f"{self.__class__.__name__}({attrs})"

#     def __str__(self) -> str:
#         if self.id is not None:
#             return str(self.id)
#         return repr(self)

#     def __hash__(self) -> int:
#         if self.id is not None:
#             return hash(self.id) ^ hash(self.__class__.__name__)
#         return NotImplemented
#         # h = 0
#         # for value in self._available_attrs.values():
#         #     h ^= hash(value)
#         # return h

#     def __eq__(self, other: Any) -> bool:
#         if self.id is not None:
#             return str(self.id) == str(other)
#         if (
#             self.dt is not None 
#             and isinstance(other, datetime.datetime)
#         ):
#             return self.dt == other
#         for group in self._required_kwarg_groups:
#             if all(getattr(self, k) == getattr(self, k) for k in group):
#                 break
#         else:
#             return False
#         return True

#     def __lt__(self, other: Any) -> bool:
#         if self.id is not None and isinstance(other, (int, str)):
#             return str(self.id) < str(other)
#         if self.dt is not None and isinstance(other, datetime.datetime):
#             return self.dt < other
#             # compare the first common attribute found, in rev order of __init__ signature
#         for attr in self._kwarg_names:
#             self_val = getattr(self, attr)
#             if self_val is None:
#                 continue
#             other_val = getattr(other, attr, None)
#             if str(self_val) == str(other_val):
#                 # sorts in groups of same project, same subject, etc.
#                 continue
#             return str(self_val) < str(other_val)
#             # return False
#         return NotImplemented
        
#     @property
#     def id(self) -> int | str | None:
#         """A unique identifier for the object.
#         Write once. Type at assignment is preserved.
#         """
#         return self._id
    
#     @id.setter
#     def id(self, value: int | str | SupportsID) -> None:
#         self._raise_if_assigned("id")
#         with contextlib.suppress(AttributeError):
#             value = value.id    # type: ignore
#         assert isinstance(value, (int, str)), f"{type(value)=}"
#         self._id = value
    
#     @property
#     def dt(self) -> datetime.datetime | None:
#         """A datetime object for the session.
#         Write once. Type at assignment is preserved.
#         """
#         return self._dt
    
#     @dt.setter
#     def dt(self, value: int | float | str | datetime.datetime) -> None:
#         self._raise_if_assigned("dt")
#         self._dt = utils.cast_to_dt(value, self.IS_DEFAULT_TIME_ALLOWED) if value is not None else None
        
#     @property
#     def project(self) -> ProjectRecord | None:
#         return self._project
    
#     @project.setter
#     def project(self, value: int | str | ProjectRecord) -> None:
#         self._raise_if_assigned("project")
#         self._project = ProjectRecord(value) if value is not None else None
        
#     @property
#     def subject(self) -> SubjectRecord | None:
#         return self._subject
    
#     @subject.setter
#     def subject(self, value: int | str | SubjectRecord) -> None:
#         self._raise_if_assigned("subject")
#         self._subject = SubjectRecord(value) if value is not None else None
    
#     @property
#     def date(self) -> datetime.date | None:
#         return self.dt.date() if self.dt is not None else None

#     @property
#     def time(self) -> datetime.time | None:
#         """Returns `None` if `dt.time()` is default 00:00:00"""
#         if self.dt is None or self.dt.time() == datetime.time():
#             return None
#         return self.dt.time()



# class Sessions(collections.abc.Collection):
    
#     _cache: dict[SessionRecord, Any]

#     def __getitem__(self, key: SessionSpec.ArgTypes | SessionSpec) -> Any:
#         if key is None:
#             raise KeyError("Cannot get a session with a null key")
#         key = SessionSpec(id=key) if not isinstance(key, SessionSpec) else key
#         return self._cache.__getitem__(key)
    
#     def __iter__(self):
#         ...
#     def __len__(self):
#         ...



if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=(
        doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE
    ))