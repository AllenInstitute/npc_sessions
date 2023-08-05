from __future__ import annotations

import datetime
import re
import typing
from typing import Any, ClassVar, Optional, Protocol, TypeAlias, TypeVar, Union

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
    - store `id` as an int or str, "immutable" after init

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
    >>> a == MetadataRecord(a)
    True
    >>> MetadataRecord(1).id = 2
    Traceback (most recent call last):
    ...
    AttributeError: MetadataRecord.id is read-only
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
            id_type = (id_type,)
        if not isinstance(value, id_type): # pragma: no cover
            raise TypeError(
                f"{self.__class__.__name__}.id must be in {typing.get_args(self.id_type) or self.id_type}, not {type(value)}"
            )
        return value

    @classmethod
    def validate_id(cls, value: id_type) -> None: # pragma: no cover
        """Post-parsing. Raise ValueError if not a valid value for this type of metadata record."""
        if isinstance(value, int):
            if value < 0:
                raise ValueError(
                    f"{cls.__name__}.id must be non-negative, not {value!r}"
                )
        if isinstance(value, str):
            if not re.match(cls.valid_id_regex, value):
                raise ValueError(
                    f"{cls.__name__}.id must match {cls.valid_id_regex}, not {value!r}"
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
        return f"{self.__class__.__name__}({self.id!r})"

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

    @classmethod
    def validate_id(cls, value: id_type) -> None:
        """
        >>> DatetimeRecord.validate_id('2002-04-25 15:02:37')
        Traceback (most recent call last):
        ...
        ValueError: Invalid year: 2002
        """
        super().validate_id(value)
        dt = datetime.datetime.fromisoformat(value)
        if not 2015 < dt.year <= datetime.datetime.now().year:
            raise ValueError(f"Invalid year: {dt.year}")
        
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
    
    @classmethod
    def validate_id(cls, value: id_type) -> None:
        """
        >>> DateRecord.validate_id('2002-04-25')
        Traceback (most recent call last):
        ...
        ValueError: Invalid year: 2002
        """
        super().validate_id(value)
        dt = datetime.date.fromisoformat(value)
        if not 2015 < dt.year <= datetime.datetime.now().year:
            raise ValueError(f"Invalid year: {dt.year}")
        
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

    ...default is to hide a null index, since they're not currently used.
    To modify this, either set the class attribute `display_null_idx` to True,
    or pass `display_null_idx=True` to the constructor:
    >>> SessionRecord('366122_2022-04-25', display_null_idx=True)
    SessionRecord('366122_2022-04-25_0')
    
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

    - date must be a valid recent or near-future date:
    >>> SessionRecord('366122_2022-13-25')
    Traceback (most recent call last):
    ...
    ValueError: SessionRecord.id must be in format <subject_id>_<date>_<idx
    [optional]>

    Comparisons are based on the session's normalized id:
    >>> assert SessionRecord('366122_2022-04-25_0') == '366122_2022-04-25'
    >>> assert SessionRecord('366122_2022-04-25') == '366122_2022-04-25_0'
    >>> a = {SessionRecord('366122_2022-04-25_0')}
    
    Validator can also be used to check if a string is a valid session id:
    >>> SessionRecord.validate_id('366122_2022-04-25_0') == None
    True
    >>> SessionRecord.validate_id('366122_2022-99-25_0') == None
    Traceback (most recent call last):
    ...
    ValueError: SessionRecord.id must match SessionRecord.valid_id_regex
    """

    id: str
    id_input: TypeAlias = str
    id_type: TypeAlias = str
    valid_id_regex: ClassVar = parsing.VALID_SESSION_ID

    display_null_idx = False
    """Whether to append `_0` to `id`` when a session index is 0. Default is
    False as long as each subject has at most one session per day."""

    def __init__(self, id: id_input, display_null_idx = False) -> None:
        self.display_null_idx = display_null_idx
        super().__init__(id)
        
    def parse_id(self, value: id_input) -> id_type:
        value = parsing.extract_session_id(str(value), self.display_null_idx)
        value = str(super().parse_id(value))
        self._subject = SubjectRecord(value.split("_")[0])
        self._date = DateRecord(value.split("_")[1])
        self._idx: int = parsing.extract_session_index(value) or 0
        return value

    @property
    def subject(self) -> SubjectRecord:
        return self._subject

    @property
    def date(self) -> DateRecord:
        return self._date

    @property
    def idx(self) -> int:
        return self._idx

    def with_idx(self, idx: Optional[int] = None) -> SessionRecord:
        """New instance with an updated idx included and visible.
        
        >>> SessionRecord('366122_2022-04-25_0').with_idx(1)
        SessionRecord('366122_2022-04-25_1')
        >>> SessionRecord('366122_2022-04-25').with_idx(1)
        SessionRecord('366122_2022-04-25_1')
        
        If `idx` isn't supplied, the current idx is used with
        `display_null_idx` set to True: 
        >>> SessionRecord('366122_2022-04-25').with_idx()
        SessionRecord('366122_2022-04-25_0')
        >>> SessionRecord('366122_2022-04-25_1').with_idx() # no change
        SessionRecord('366122_2022-04-25_1')    
        """
        return SessionRecord(f"{self.subject}_{self.date}_{idx or self.idx}", display_null_idx=True)

    def __eq__(self, other: Any) -> bool:
        """
        >>> assert SessionRecord('366122_2022-04-25') == SessionRecord('366122_2022-04-25')
        >>> assert SessionRecord('366122_2022-04-25_0') == SessionRecord('366122_2022-04-25')
        >>> assert SessionRecord('366122_2022-04-25_1') == SessionRecord('366122_2022-04-25_1')
        
        >>> assert SessionRecord('366122_2022-04-25_0') != '366122_2022-04-25_1'
        
        Missing index is the same as _0:
        >>> assert SessionRecord('366122_2022-04-25_0') == '366122_2022-04-25'
        
        Comparison possible with stringable types:
        >>> assert SessionRecord('366122_2022-04-25') == MetadataRecord('366122_2022-04-25')
        
        False for comparison with non-stringable types:
        >>> assert SessionRecord('366122_2022-04-25') != 366122
        
        >>> assert SessionRecord('366122_2022-04-25_0')  == '366122_2022-04-25'
        >>> assert SessionRecord('366122_2022-04-25_0') == '366122_2022-04-25_12:00:00'
        >>> assert SessionRecord('366122_2022-04-25_0') == '2022-04-25_12:00:00_366122'
        >>> assert SessionRecord('366122_2022-04-25_0') != '366122_2022-04-25_12:00:00_1'
        """
        if self.idx == 0:
            try:
                return parsing.extract_session_id(
                    str(self)
                ) == parsing.extract_session_id(str(other))
            except ValueError:
                return False
        return super().__eq__(other)

    def __hash__(self) -> int:
        """
        >>> assert SessionRecord('366122_2022-04-25') in {SessionRecord('366122_2022-04-25')}
        """
        return hash(self.subject) ^ hash(self.date) ^ hash(self.idx) ^ hash(self.__class__.__name__)


if __name__ == "__main__":
    import doctest

    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
