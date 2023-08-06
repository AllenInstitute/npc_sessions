import json
import pathlib
from collections.abc import MutableSequence
from typing import Literal, NamedTuple, TypeAlias

import npc_session  # type: ignore
import yaml


class SessionInfo(NamedTuple):
    session: npc_session.SessionRecord
    subject: npc_session.SubjectRecord
    date: npc_session.DateRecord
    idx: int
    project: npc_session.ProjectRecord
    is_ephys: bool
    is_sync: bool
    """The session has sync data, implying more than a behavior-box."""


def get_session_info() -> tuple[SessionInfo, ...]:
    """Quickly get a sequence of all tracked sessions.

    Each object in the sequence has info about one session:
    >>> sessions = get_session_info()
    >>> sessions[0].__class__.__name__
    'SessionInfo'
    >>> sessions[0].is_ephys
    True
    >>> any(s for s in sessions if s.date.year < 2021)
    False
    """
    return _get_session_info_from_local_file()


_LOCAL_FILE = pathlib.Path(__file__).parent / "tracked_sessions.yaml"
FileContents: TypeAlias = dict[
    Literal["ephys", "behavior_with_sync", "behavior"], dict[str, str]
]


def _get_session_info_from_local_file() -> tuple[SessionInfo, ...]:
    """Load yaml and parse sessions.
    - currently assumes all sessions include behavior data
    """
    f = _session_info_from_file_contents
    if _LOCAL_FILE.suffix == ".json":
        return f(json.loads(_LOCAL_FILE.read_text()))
    if _LOCAL_FILE.suffix == ".yaml":
        return f(yaml.load(_LOCAL_FILE.read_bytes(), yaml.FullLoader))
    raise ValueError(f"Add loader for {_LOCAL_FILE.suffix}")  # pragma: no cover


def _session_info_from_file_contents(contents: FileContents) -> tuple[SessionInfo, ...]:
    sessions: MutableSequence[SessionInfo] = []
    for session_type, projects in contents.items():
        if not projects:
            continue
        sync = any(tag in session_type for tag in ("sync", "ephys"))
        ephys = "ephys" in session_type
        for project_name, session_ids in projects.items():
            if not session_ids:
                continue
            for session_id in session_ids:
                s = npc_session.SessionRecord(session_id)
                sessions.append(
                    SessionInfo(
                        *(s, s.subject, s.date, s.idx),
                        project=npc_session.ProjectRecord(project_name),
                        is_ephys=ephys,
                        is_sync=sync,
                    )
                )
    return tuple(sessions)


if __name__ == "__main__":
    import doctest

    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
