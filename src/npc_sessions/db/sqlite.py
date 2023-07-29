"""
Base class for sqlite job queues.

- file must be accessible on //allen (has implications for sqlite, e.g.
  incompat with 'wal' mode)

>>> with SqliteIsilonJobQueue(table_name='test').cursor() as c:
...   _ = c.execute('DROP TABLE IF EXISTS test')
>>> q = SqliteIsilonJobQueue(table_name='test')
>>> q['123456789_366122_20230422'] = get_job('123456789_366122_20230422')
>>> assert q.next().session == np_session.Session('123456789_366122_20230422')
>>> q.add_or_update('123456789_366122_20230422', priority=99)
>>> import datetime; assert datetime.datetime.fromtimestamp(q['123456789_366122_20230422'].added)
>>> q.update('123456789_366122_20230422', finished=0)
>>> assert q['123456789_366122_20230422'].priority == 99
>>> q.set_started('123456789_366122_20230422')
>>> assert q.is_started('123456789_366122_20230422')
>>> q.set_finished('123456789_366122_20230422')
>>> assert q['123456789_366122_20230422'].finished == 1
>>> q.set_queued('123456789_366122_20230422')
>>> assert not q['123456789_366122_20230422'].finished
>>> assert not q.is_started('123456789_366122_20230422')
>>> del q['123456789_366122_20230422']
>>> assert '123456789_366122_20230422' not in q
"""
from __future__ import annotations

import collections.abc
import contextlib
import dataclasses
import datetime
import pathlib
import socket
import sqlite3
import time
from typing import Any, Generator, Iterator, Optional, Type

import pydantic

from npc_sessions.db.types import Job, JobArgs, JobT, SessionArgs
from npc_sessions.types import SessionSpec, SupportsID
import npc_sessions.utils as utils

DEFAULT_DB_PATH = "./job_queue.sqlite"

JOB_ARGS_TO_SQL_DEFINITIONS: dict[str, str] = {
    'id': 'TEXT PRIMARY KEY NOT NULL UNIQUE',
    'added': 'DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL',  # YYYY-MM-DD HH:MM:SS
    'priority': 'INTEGER DEFAULT 0',
    'started': 'DATETIME DEFAULT NULL',
    'hostname': 'TEXT DEFAULT NULL',
    'finished': 'DATETIME DEFAULT NULL',  # [None] 0 or 1
    'error': 'TEXT DEFAULT NULL',
}
"""Mapping of job attribute names (keys in db) to sqlite3 column definitions."""
    

@pydantic.dataclasses.dataclass
class JobDataclass:
    """Dataclass with only session required.
    
    >>> job = JobDataclass('123456789_366122_20230422')
    >>> assert isinstance(job, Job)
    """
    id: str
    added: datetime.datetime = dataclasses.field(default_factory=datetime.datetime.now)
    priority: int = 0
    started: Optional[datetime.datetime] = None
    hostname: Optional[str] = None
    finished: Optional[datetime.datetime] = None
    error: Optional[str] = None

def get_hostname() -> str:
    return socket.gethostname()
    
def get_session(*args, **kwargs: SessionArgs | Job) -> SessionSpec:
    """Parse a session argument into a Neuropixels Session.
    
    >>> get_session('123456789_366122_20230422')
    SessionSpec(id='123456789_366122_20230422', dt=2023-04-22 00:00:00)
    >>> assert _ == get_session(SessionSpec('123456789_366122_20230422'))
    >>> get_session('DRpilot_644866_20230207')
    SessionSpec(id='DRpilot_644866_20230207', dt=2023-02-07 00:00:00)
    """
    if isinstance(args, SessionSpec):
        return args
    if isinstance(args, Job):
        return get_session(**args.__dict__)
    return SessionSpec(*args, **{k:v for k, v in kwargs.items() if k in SessionSpec._kwarg_names})

    
def get_job(*args, job_type: Type[JobT] = JobDataclass, **kwargs: SessionArgs | Job, ) -> JobT:
    """Get a job with default values and just the `session` attr filled in.
    
    >>> job = get_job('123456789_366122_20230422')
    >>> assert isinstance(job, Job)
    >>> assert job == get_job(job)
    """
    if isinstance(args, job_type):
        return args
    return job_type(
        get_session(*args, **kwargs).id,
        )


def sql_table_columns(column_name_to_definition_mapping: dict[str, str]) -> str:
    """
    Define table in sqlite3.
    
    >>> sql_table({'col1': 'TEXT PRIMARY KEY NOT NULL', 'col2': 'INTEGER'})
    '(col1 TEXT PRIMARY KEY NOT NULL, col2 INTEGER)'
    """
    return (
        '('
        + ', '.join(
            [
                '{} {}'.format(col, defn)
                for col, defn in column_name_to_definition_mapping.items()
            ]
        )
        + ')'
    )

class SqliteTable:
    
    db: sqlite3.Connection
    """sqlite3 db connection to shared file on Isilon"""
    
    sqlite_db_path: str | pathlib.Path = DEFAULT_DB_PATH
    table_name: str = 'table'
    
    column_definitions: dict[str, str] = JOB_ARGS_TO_SQL_DEFINITIONS
    """Mapping of column names to sqlite3 column definitions."""
    
    is_wal_disabled: bool = False
    """Should be disabled if the file is on Isilon or S3."""
    
    def __init__(
        self,
        **kwargs,
        ) -> None:
        """
        Pass in any attributes as kwargs to assign to instance.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.create_db()
        self.setup_db_connection()
        self.create()
        
    def create_db(self) -> None:
        if not pathlib.Path(self.sqlite_db_path).exists():
            pathlib.Path(self.sqlite_db_path).parent.mkdir(parents=True, exist_ok=True)
    
    def validate_attributes(self) -> None: ...
        
    def setup_db_connection(self) -> None:
        self.db = sqlite3.connect(str(self.sqlite_db_path), timeout=1)
        self.db.isolation_level = None  # autocommit mode
        if self.is_wal_disabled:
            self.db.execute('pragma journal_mode="delete"')
            self.db.execute('pragma synchronous=2')
            
    def create(self) -> None:
        """
        Create table with `self.table_name` if it doesn't exist.    
        
        >>> s = SqliteIsilonJobQueue(table_name='test')
        >>> s.setup_table()
        >>> with s.cursor() as c:
        ...   result = c.execute('SELECT count(*) FROM sqlite_schema WHERE type="table" AND name="test"').fetchall()[0][0]
        >>> assert result == 1, f'Test result returned {result}: expected 1 (True)'
        """
        with self.cursor() as c:
            c.execute(
                f'CREATE TABLE IF NOT EXISTS {self.table_name} '
                + sql_table_columns(self.column_definitions),
            )
            
    def drop(self) -> None:
        with self.cursor() as c:
            c.execute(f'DROP TABLE IF EXISTS {self.table_name}')
            
    @contextlib.contextmanager
    def cursor(self) -> Generator[sqlite3.Cursor, None, None]:
        """
        >>> with SqliteIsilonJobQueue(table_name='test').cursor() as c:
        ...    assert isinstance(c, sqlite3.Cursor)
        ...    _ = c.execute('SELECT 1').fetchall()
        """
        cursor = self.db.cursor()
        try:
            cursor.execute('begin exclusive')
            yield cursor
        except Exception:
            self.db.rollback()
            raise
        else:
            self.db.commit()
        finally:
            cursor.close()
    
    
class SqliteQueue(collections.abc.MutableMapping, SqliteTable):

    table_name: str = 'job_queue'
    
    column_definitions: dict[str, str] = JOB_ARGS_TO_SQL_DEFINITIONS
    job_type: Type[Job] = JobDataclass
    """Job class to use for the queue - see `np_jobs.types.Job` protocol for required attributes"""
    
    def __init__(
        self,
        **kwargs,
        ) -> None:
        """
        Pass in any attributes as kwargs to assign to instance.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.validate_attributes()
        super().__init__(**kwargs)
        
    def validate_attributes(self) -> None:
        assert all(hasattr(self.job_type('test'), attr) for attr in self.column_definitions.keys()), (
            '`self.job_type` must have all attributes exactly matching keys in `self.column_definitions`.',
            f'{self.job_type("test")=} {self.column_definitions.keys()=}',
        )
        assert isinstance(self.job_type('test'), Job)
    
    def from_job(self, job: Job) -> tuple[JobArgs, ...]:
        """Convert a job to a tuple of args for inserting into sqlite."""
        job_args = []
        for attr in self.column_definitions.keys():
            value = getattr(job, attr)
            job_args.append(value)    
        return tuple(job_args)
    
    def to_job(self, *args: JobArgs, **kwargs: JobArgs) -> JobT:
        """Convert args or kwargs into a job.
        
        If args are provided, the assumption is they came from sqlite in the
        order specified by `self.column_definitions`.
        """
        if args and kwargs:
            raise ValueError(f'Cannot pass both args and kwargs: {args=}, {kwargs=}')
        if args:
            kwargs = dict(zip(self.column_definitions.keys(), args))
        return self.job_type(**kwargs)

    def __getitem__(self, session_or_job: SessionArgs | Job) -> JobT:
        """Get a job from the queue, matching on session."""
        session = get_session(session_or_job)
        with self.cursor() as c:
            hits = c.execute(
                f'SELECT * FROM {self.table_name} WHERE id = ?',
                (session.id,),
            ).fetchall()
        if not hits:
            raise KeyError(session)
        if len(hits) > 1:
            raise ValueError(f'Found multiple jobs for {session=}. Expected `session` to be unique.')
        return self.to_job(*hits[0])
        
    def __setitem__(self, session_or_job: SessionArgs | Job, job: Job) -> None:
        """Add a job to the queue or update the existing entry."""
        session = get_session(session_or_job)
        if session != job.id:
            raise ValueError(f'`session` values don"t match {session_or_job=}, {job=}')
        with self.cursor() as c:
            c.execute(
                (
                    f'INSERT OR REPLACE INTO {self.table_name} (' +
                    ', '.join(self.column_definitions.keys()) + ') VALUES (' +
                    ', '.join('?'* len(self.column_definitions)) + ')'
                ),
                (
                    *self.from_job(job),
                ),
            )
            
    def __delitem__(self, session_or_job: SessionArgs | Job) -> None:
        """Remove a job from the queue."""
        session = get_session(session_or_job)
        with self.cursor() as c:
            c.execute(
                f'DELETE FROM {self.table_name} WHERE id = ?',
                (session.id,),
            )
            
    def __contains__(self, session_or_job: SessionArgs | Job) -> bool:
        """Whether the session or job is in the queue."""
        session = get_session(session_or_job)
        with self.cursor() as c:
            hits = c.execute(
                f'SELECT * FROM {self.table_name} WHERE id = ?',
                (session.id,),
            ).fetchall()
        return bool(hits)
        
    def __len__(self) -> int:
        """Number of jobs in the queue."""
        with self.cursor() as c:
            return c.execute(
                f'SELECT count(*) FROM {self.table_name}',
                (),
            ).fetchall()[0][0]
    
    def __iter__(self) -> Iterator[JobT]:
        """Iterate over the jobs in the queue.   
        Sorted by priority (desc), then date added (asc).
        """
        with self.cursor() as c:
            hits = c.execute(
                f'SELECT * FROM {self.table_name} ORDER BY error ASC NULLS FIRST, priority DESC, added ASC',
                (),
            ).fetchall()
        return iter(self.to_job(*hit) for hit in hits)
    
    def add_or_update(self, session_or_job: SessionArgs | Job, **kwargs: JobArgs) -> None:
        """Add an entry to the queue or update the existing entry.
        - any kwargs provided will be updated on the job
        - job will be re-queued
        """
        self.update(session_or_job, **kwargs)
        self.set_queued(session_or_job)
        
    def update(self, session_or_job: SessionArgs | Job, **kwargs: JobArgs) -> None:
        """Update an existing entry in the queue.
        Any kwargs provided will be updated on the job.
        """
        job = self.setdefault(session_or_job, get_job(session_or_job, job_type=self.job_type)) 
        for key, value in kwargs.items():
            setattr(job, key, value)
        super().update({job.id: job})
    
        
    def __next__(self):
        jobs = self.outstanding_jobs()
        yield from jobs
        
    def outstanding_jobs(self) -> Generator[JobT, Any, None]:
        """
        Get the next job to process.
        Sorted by priority (desc), then date added (asc).
        """
        for job in iter(self):
            if (
                not self.is_started(job)
                and not job.finished
                and not job.error
            ):
                yield job
    
    def set_finished(self, session_or_job: SessionArgs | Job) -> None:
        """Mark a job as finished. May be irreversible, so be sure."""
        self.update(session_or_job, finished=datetime.datetime.now())
        
    def set_started(self, session_or_job: SessionArgs | Job) -> None:
        """Mark a job as being processed. Reversible"""
        self.update(session_or_job, started=datetime.datetime.now(), hostname=get_hostname(), finished=None)
        
    def set_queued(self, session_or_job: SessionArgs | Job) -> None:
        """Mark a job as requiring processing, undoing `set_started`."""
        self.update(session_or_job, started=None, hostname=None, finished=None, errored=None)
    
    def set_errored(self, session_or_job: SessionArgs | Job, error: str | Exception) -> None:
        self.update(session_or_job, error=str(error))
    
    def is_started(self, session_or_job: SessionArgs | Job) -> bool:
        """Whether the job has started processing, but not yet finished."""
        job = self[session_or_job]
        return (
            job.started
            and not job.finished
            and not job.error
        )



if __name__ == '__main__':
    # import doctest

    # doctest.testmod(verbose=False, raise_on_error=False)
    x = SqliteQueue()
    x.add_or_update('123456789_366122_20230422', priority=99)