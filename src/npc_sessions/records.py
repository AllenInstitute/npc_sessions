from __future__ import annotations

import collections.abc
import datetime
import re
import typing
from typing import Any, ClassVar, Protocol, TypeAlias, TypeVar, Union

import npc_sessions.parsing as parsing


@typing.runtime_checkable
class SupportsID(Protocol):
    @property
    def id(self) -> int | str:
        ...


T = TypeVar("T", int, str)


class MetadataRecord:
    """A base class for definitions of unique metadata records
    (unique amongst that type of metadata, not necessarily globally).
    
    Intended use is for looking-up and referring to metadata components
    in databases, regardless of the backend.
    
    Each implementation should:
    - accept a variety of input formats
    - parse, normalize, then validate input
    - give a single consistent representation (`id`)
    - store `id` as int or str

    Instances are effectively the same as their stored `id` attribute, but with extra
    properties, including magic methods for str, equality, hashing and ordering:

    >>> a = MetadataRecord(1)
    >>> a
    MetadataRecord(1)
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
    [MetadataRecord(1), MetadataRecord(2)]
    """

    id_input: TypeAlias = Union[int, str]
    """Acceptable input types for `id`. Will be parsed and, 
    if not already, cast to `id_type`. Not strict: for static analysis only."""

    id_type: TypeAlias = Union[int, str]
    """Type of `id` stored after parsing and used for records. Strict: will be
    validated with isinstance check on assignment. May be a union of types."""

    valid_id_regex: ClassVar[str] = r"[0-9-_: ]+"

    def __init__(self, id: id_input) -> None:
        self.id = id

    def parse_id(self, value: id_input) -> id_type:
        """Pre-validation. Handle any parsing or casting to get to the stored type."""
        if isinstance(value, (SupportsID,)):
            value = value.id
        # at this point, the type of the id value should match the specified
        # type for this record class
        id_type = typing.get_args(self.__class__.id_type) or self.__class__.id_type
        if not isinstance(id_type, tuple):
            id_type = tuple([id_type])
        if not isinstance(value, id_type):
            raise TypeError(
                f"{self.__class__.__name__}.id must be in {typing.get_args(self.id_type) or self.id_type}, not {type(value)}"
            )
        return value

    def validate_id(self, value: id_type) -> None:
        """Post-parsing. Raise ValueError if not a valid value for this type of metadata record."""
        if isinstance(value, int):
            if value < 0:
                raise ValueError(
                    f"{self.__class__.__name__}.id must be non-negative, not {value!r}"
                )
        if isinstance(value, str):
            if not re.match(self.valid_id_regex, value):
                raise ValueError(
                    f"{self.__class__.__name__}.id must match {self.valid_id_regex}, not {value!r}"
                )

    @property
    def id(self) -> int | str:
        """A unique identifier for the object.
        Read-only, and type at assignment is preserved.
        """
        return self._id

    @id.setter
    def id(self, value: id_input) -> None:
        """A unique identifier for the object.
        Write-once, then read-only. Type at assignment is preserved.
        """
        if hasattr(self, "_id"):
            raise AttributeError(f"{self.__class__.__name__}.id is read-only")
        value = self.parse_id(value)
        self.validate_id(value)
        self._id = value

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.id!r})'

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
    ProjectRecord('DR')
    >>> project == 'DR'
    True
    >>> isinstance(project, str)
    False
    """

    id_input: TypeAlias = str
    id_type: TypeAlias = str
    valid_id_regex = r"[a-zA-Z0-9-_.]+"


class SubjectRecord(MetadataRecord):
    """To uniquely define a subject we need:
    - `id`: labtracks MID

    >>> subject = SubjectRecord('366122')
    >>> subject = SubjectRecord(366122)
    >>> subject
    SubjectRecord(366122)
    >>> subject == 366122 and subject == '366122'
    True
    >>> isinstance(subject, (int, str))
    False
    """

    id_input: TypeAlias = Union[int, str]
    id_type: TypeAlias = int

    def parse_id(self, value: id_input) -> id_type:
        return int(super().parse_id(int(str(value))))


class DatetimeRecord(MetadataRecord):
    """Datetime records are stored in isoformat with a resolution of seconds,
    and space separator between date/time, hypen between date components, colon
    between time components.

    >>> dt = DatetimeRecord('2022-04-25 15:02:37')
    >>> dt = DatetimeRecord('366122_20220425_150237_2')
    >>> dt
    DatetimeRecord('2022-04-25 15:02:37')
    >>> str(dt)
    '2022-04-25 15:02:37'
    >>> datetime.datetime.fromisoformat(dt.id)
    datetime.datetime(2022, 4, 25, 15, 2, 37)

    Components of datetime are also made available:
    >>> dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second
    (2022, 4, 25, 15, 2, 37)
    """

    id_input: TypeAlias = Union[int, str, datetime.datetime]
    id_type: TypeAlias = str

    valid_id_regex: ClassVar[str] = parsing.VALID_DATETIME
    """A valid datetime this century, format YYYY-MM-DD HH:MM:SS"""

    dt: datetime.datetime

    def parse_id(self, value: id_input) -> id_type:
        self.date = parsing.extract_isoformat_date(str(value))
        self.time = parsing.extract_isoformat_time(str(value))
        if self.date is None or self.time is None:
            raise ValueError(
                f"{self.__class__.__name__} requires a date YYYY-MM-DD HH:MM:SS with optional separators, not {value!r}"
            )
        dt = f"{self.date} {self.time}"
        self.dt = datetime.datetime.fromisoformat(dt)
        return str(super().parse_id(dt))

    def __getattribute__(self, __name: str) -> Any:
        if __name in ("month", "year", "day", "hour", "minute", "second", "resolution"):
            return self.dt.__getattribute__(__name)
        return super().__getattribute__(__name)


class DateRecord(MetadataRecord):
    """Date records are stored in isoformat with hyphen seperators.

    >>> DateRecord('2022-04-25')
    DateRecord('2022-04-25')
    >>> DateRecord('20220425')
    DateRecord('2022-04-25')
    >>> date = DateRecord('20220425')
    >>> datetime.date.fromisoformat(str(date))
    datetime.date(2022, 4, 25)

    Components of date are also made available:
    >>> date.year, date.month, date.day
    (2022, 4, 25)
    """
    id_input: TypeAlias = Union[int, str, datetime.date]
    id_type: TypeAlias = str
    valid_id_regex: ClassVar[str] = parsing.VALID_DATE
    """A valid date this century, format YYYY-MM-DD."""

    dt: datetime.date

    def parse_id(self, value: id_input) -> id_type:
        date = parsing.extract_isoformat_date(str(value))
        if date is None:
            raise ValueError(
                f"{self.__class__.__name__} requires a date YYYY-MM-DD with optional separators, not {value!r}"
            )
        self.dt = datetime.date.fromisoformat(date)
        return str(super().parse_id(date))

    def __getattribute__(self, __name: str) -> Any:
        if __name in ("month", "year", "day", "resolution"):
            return self.dt.__getattribute__(__name)
        return super().__getattribute__(__name)


class SessionRecord(MetadataRecord):
    """To uniquely define a subject we need:
    - `id`: a multipart underscore-separated str corresponding to:

        <subject_id>_<date>_<idx [optional]>

        where:
        - `date` is in format YYYY-MM-DD, with optional separators (hyphen, slash)
        - `idx` corresponds to the index of the session on that date for that
        subject
        - `idx` must be appended as the last component, separated by an underscore
        - if the last component is missing, it is implicitly assumed to be the only
        session on the date for that subject
        - `idx` currently isn't needed, but is included for future-proofing
        - varname `idx` is used to avoid conflicts with `index` methods in
        possible base classes
        
    Record provides:
    - normalized id
    - subject record
    - date record
    - idx (0 if not specified)

    We can extract these components from typical session identifiers, such as
    folder names, regardless of component order:
    >>> a = SessionRecord('DRPilot_366122_20220425')
    >>> b = SessionRecord('0123456789_366122_20220425')
    >>> c = SessionRecord('366122_2022-04-25')
    >>> d = SessionRecord('2022-04-25_12:00:00_366122')
    >>> a
    SessionRecord('366122_2022-04-25')
    >>> a == b == c == d
    True

    Components are also available for use:
    >>> a.subject, a.date, a.idx
    (SubjectRecord(366122), DateRecord('2022-04-25'), 0)
    >>> a.date.year, a.date.month, a.date.day
    (2022, 4, 25)

    Missing index is synonymous with idx=0:
    >>> SessionRecord('366122_2022-04-25').idx
    0
    >>> SessionRecord('366122_2022-04-25_1').idx
    1

    To change index (which is read-only), use `with_idx` to create a new
    record instance:
    >>> a = SessionRecord('366122_2022-04-25_1')
    >>> b = a.with_idx(2)
    >>> b
    SessionRecord('366122_2022-04-25_2')
    >>> a != b and a is not b
    True

    Subject and date are validated on init:
    - subject must be a recent or near-future labtracks MID:
    >>> SessionRecord('1_2022-04-25')
    Traceback (most recent call last):
    ...
    ValueError: SessionRecord.id must be in format <subject_id>_<date>_<idx [optional]>

    - date must make be a valid recent or near-future date:
    >>> SessionRecord('366122_2022-13-25')
    Traceback (most recent call last):
    ...
    ValueError: SessionRecord.id must be in format <subject_id>_<date>_<idx
    [optional]>

    Comparisons are based on the session's normalized id:
    >>> assert SessionRecord('366122_2022-04-25_0') == '366122_2022-04-25'
    >>> assert SessionRecord('366122_2022-04-25') == '366122_2022-04-25_0'
    >>> assert SessionRecord('366122_2022-04-25_0') in ['366122_2022-04-25']
    >>> assert SessionRecord('366122_2022-04-25') in ['366122_20220425_0']
    >>> assert SessionRecord('366122_2022-04-25_0') in ['366122_2022-04-25_12:00:00']
    >>> assert SessionRecord('366122_2022-04-25_0') in ['366122_2022-04-25_12:00:00_0']
    >>> assert SessionRecord('366122_2022-04-25_0') not in ['366122_2022-04-25_12:00:00_1']
    >>> assert SessionRecord('366122_2022-04-25_12:00:00').idx == 0
    """

    id: str
    id_input: TypeAlias = str
    id_type: TypeAlias = str
    valid_id_regex: ClassVar = parsing.VALID_SESSION_ID

    display_null_idx = False
    """Whether to append `_0` to `id`` when a session index is 0. Default is
    False as long as each subject has at most one session per day."""

    def parse_id(self, value: id_input) -> id_type:
        value = parsing.extract_session_id(str(value), self.display_null_idx)
        value = str(super().parse_id(value))
        self._subject = SubjectRecord(value.split('_')[0])
        self._date = DateRecord(value.split('_')[1])
        self._idx: int = parsing.extract_session_index(value) or 0
        return value

    def validate_id(self, id: int | str) -> None:
        super().validate_id(id)
        if not isinstance(id, str) or not re.match(self.valid_id_regex, id):
            raise ValueError(
                f"{self.__class__.__name__}.id must be in format <subject_id>_<date>_<idx [optional]>, not {id}"
            )

    @property
    def subject(self) -> SubjectRecord:
        return self._subject

    @property
    def date(self) -> DateRecord:
        return self._date

    @property
    def idx(self) -> int:
        return self._idx

    def with_idx(self, idx: int) -> SessionRecord:
        """New instance form with an updated idx included"""
        return SessionRecord(f"{self.subject}_{self.date}_{idx}")

    def __eq__(self, other: Any) -> bool:
        if self.idx == 0:
            try:
                return parsing.extract_session_id(
                    str(self)
                ) == parsing.extract_session_id(str(other))
            except ValueError:
                return False
        return super().__eq__(other)

    def __hash__(self) -> int:
        return hash(self.with_idx) ^ hash(self.__class__.__name__)


# class SessionSpec:
#     """Used to get the record corresponding to individual sessions.

#     A session comprises some combination of:
#     - `id`: all or part of an existing record id
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


class Sessions(collections.abc.Mapping):
    _cache: dict[SessionRecord, Any]

    def __getitem__(self, key: str) -> Any:
        if key is None:
            raise KeyError("Cannot get a session with a null key")
        key = SessionRecord(id=key) if not isinstance(key, SessionSpec) else key
        return self._cache.__getitem__(key)

    def __iter__(self):
        ...

    def __len__(self):
        ...


if __name__ == "__main__":
    DatetimeRecord('2022-04-25 15:02:37')

    import doctest
    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
