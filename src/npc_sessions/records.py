from __future__ import annotations
import datetime

from typing import NamedTuple, Optional


class SessionRecord(NamedTuple):
    dt: datetime.datetime | datetime.date
    subject: int | str | SubjectRecord
    project: Optional[int | str | ProjectRecord] = None
    
class SubjectRecord(NamedTuple):
    id: int | str

class ProjectRecord(NamedTuple):
    id: int | str

x = SessionRecord(dt=datetime.datetime(2022, 4, 25, 15, 2, 37), subject=SubjectRecord(id=1), project=ProjectRecord(id=1))
y = SessionRecord(dt=datetime.datetime(2022, 4, 25, 15, 2, 37), subject=SubjectRecord(id=1), project=)
x
