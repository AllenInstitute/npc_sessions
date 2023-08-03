import json
import pathlib
from typing import Literal, MutableSequence, NamedTuple, TypeAlias

import yaml

from npc_sessions import records


class SessionInfo(NamedTuple):
    session: records.SessionRecord
    subject: records.SubjectRecord
    date: records.DateRecord
    idx: int
    project: records.ProjectRecord
    is_ephys: bool
    is_behavior: bool = True


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
    return _get_session_info_from_local_yaml()

_LOCAL_FILE = pathlib.Path(__file__).parent / "tracked_sessions.yaml"
FileContents: TypeAlias = dict[Literal['ephys', 'behavior_with_sync', 'behavior'], dict[str, str]]

def _get_session_info_from_local_yaml() -> tuple[SessionInfo, ...]:
    """Load yaml and parse sessions. 
    - currently assumes all sessions include behavior data
    """
    sessions_from_file: FileContents = (
        yaml.load(_LOCAL_FILE.with_suffix('.yaml').read_bytes(),
                  Loader=yaml.FullLoader
                  )
    )
    return _session_info_from_file_contents(sessions_from_file)

def _get_session_info_from_local_json() -> tuple[SessionInfo, ...]:
    """Load json and parse sessions. 
    - currently assumes all sessions include behavior data
    """
    sessions_from_file: FileContents = (
        json.loads(_LOCAL_FILE.with_suffix('.json').read_text())
    )
    return _session_info_from_file_contents(sessions_from_file)

def _session_info_from_file_contents(contents: FileContents) -> tuple[SessionInfo, ...]:
    sessions: MutableSequence[SessionInfo] = []
    for session_type, projects in contents.items():
        if not projects:
            continue
        for project_name, session_ids in projects.items():
            if not session_ids:
                continue
            for session_id in session_ids:
                s = records.SessionRecord(session_id)
                sessions.append(
                    SessionInfo(
                        *(s, s.subject, s.date, s.idx),
                        project=records.ProjectRecord(project_name),
                        is_ephys="ephys" in session_type,
                        is_behavior=True,
                    )
            )
    return tuple(sessions)

if __name__ == "__main__":
    import doctest
    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )