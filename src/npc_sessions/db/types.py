"""
Protocols (interfaces) that define the required methods for jobs and
job-queues, and types for static analysis (mypy).
"""
from __future__ import annotations

import abc
import collections.abc
import contextlib
import dataclasses
import datetime
import pathlib
import time
import typing
from typing import (Any, Mapping, NamedTuple, Optional, Protocol, Type,
                    TypeVar, Union)

from typing_extensions import Self, TypeAlias

from npc_sessions.types import SessionSpec

SessionArgs: TypeAlias = Union[SessionSpec.ArgTypes, SessionSpec]
"""Any valid input to create a `SessionSpec` instance."""""

JobArgs: TypeAlias = Union[str, int, float, datetime.datetime, None]
"""Types of value that can be stored - mainly constrained by sqlite3 types."""
JobKwargs: TypeAlias = Mapping[str, JobArgs]
"""Key:value pairs of job attributes.
For a job stored in sqlite, these would correspond to column-name:value pairs.
"""
JobT = TypeVar('JobT', bound='Job')
"""TypeVar with upper-bound `Job`."""
JobQueueT = TypeVar('JobQueueT', bound='JobQueue')
"""TypeVar with upper-bound `JobQueue`."""


@typing.runtime_checkable
class Job(Protocol):
    """Base class for jobs. The only required attribute is `session`, to
    match a job with a session. All other fields can be set to None."""
    
    @property
    @abc.abstractmethod
    def id(self) -> int | str:
        """Session folder name, from which we can make an `np_session.Session` instance.
        
        - each job must have a Session
        - each queue can only have one job per session (session is unique)
        """
    
    @property
    @abc.abstractmethod
    def priority(self) -> None | int:
        """
        Priority level for this job.
        Processed in descending order (then ordered by `added`).
        """
        
    @property
    @abc.abstractmethod
    def added(self) -> None | int | float | datetime.datetime:
        """
        When the job was added to the queue.
        Jobs processed in ascending order (after ordering by `priority`).
        """
    
    @property
    @abc.abstractmethod
    def started(self) -> None | int | float | datetime.datetime:
        """Whether the job has started (can also represent time)."""
        
    @property
    @abc.abstractmethod
    def hostname(self) -> None | str:
        """The hostname of the machine that is currently processing this
        session.
        
        Can also be set to choose a specific machine to process the job.
        """
        
    @property
    @abc.abstractmethod
    def finished(self) -> None | int | float | datetime.datetime:
        """Whether the session has been verified as finished."""
    
    @property
    @abc.abstractmethod
    def error(self) -> None | str:
        """Error message, if the job errored."""
    
    
@typing.runtime_checkable
class JobQueue(Protocol):
    """Base class for job queues."""
    
    @abc.abstractmethod
    def __setitem__(self, session_or_job: SessionArgs | Job, value: Job) -> None:
        """Add a job to the queue."""
    
    @abc.abstractmethod
    def __getitem__(self, session_or_job: SessionArgs | Job) -> Job:
        """Get a job from the queue."""
    
    @abc.abstractmethod
    def __delitem__(self, session_or_job: SessionArgs | Job) -> None:
        """Remove a job from the queue."""
    
    @abc.abstractmethod
    def __contains__(self, session_or_job: SessionArgs | Job) -> bool:
        """Whether the session or job is in the queue."""
    
    @abc.abstractmethod
    def __len__(self) -> int:
        """Number of jobs in the queue."""
    
    @abc.abstractmethod
    def __iter__(self) -> collections.abc.Iterator[Job]:
        """Iterate over the jobs in the queue.   
        Sorted by priority (desc), then date added (asc).
        """
        
    @abc.abstractmethod
    def add_or_update(self, session_or_job: SessionArgs | Job, **kwargs: JobArgs) -> None:
        """Add an entry to the queue or update the existing entry."""
        
    @abc.abstractmethod
    def next(self) -> Job | None:
        """
        Get the next job to process.
        Sorted by priority (desc), then date added (asc).
        """
    
    @abc.abstractmethod
    def set_finished(self, session_or_job: SessionArgs | Job) -> None:
        """Mark a job as finished. May be irreversible, so be sure."""
        
    @abc.abstractmethod
    def set_started(self, session_or_job: SessionArgs | Job) -> None:
        """Mark a job as being processed. Reversible"""

    @abc.abstractmethod
    def set_queued(self, session_or_job: SessionArgs | Job) -> None:
        """Mark a job as requiring processing, undoing `set_started`."""
    
    @abc.abstractmethod
    def set_errored(self, session_or_job: SessionArgs | Job, error: str | Exception) -> None:
        """Mark a job as having errored."""

    @abc.abstractmethod
    def is_started(self, session_or_job: SessionArgs | Job) -> bool:
        """Whether the job has started processing, but not yet finished."""
