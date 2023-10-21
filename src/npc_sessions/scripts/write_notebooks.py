from __future__ import annotations

import logging
import sys

import npc_lims
import npc_session
import upath

import npc_sessions

logger = logging.getLogger(__name__)

QC_REPO = npc_lims.DR_DATA_REPO.parent.parent / "qc"


def move_to_s3(file: npc_sessions.PathLike) -> None:
    """copy to s3, remove local copy"""
    file = npc_sessions.from_pathlike(file)
    (s3 := QC_REPO / file.name).write_bytes(file.read_bytes())
    file.unlink()
    logger.info(f"moved {file.name} to {s3.parent}")


def main() -> None:
    npc_sessions.assert_s3_write_credentials()
    sessions = sys.argv[1:] or (s.id for s in npc_lims.get_session_info() if s.is_ephys)
    for session in sessions:
        path = npc_sessions.write_qc_notebook(
            session,
            upath.UPath.cwd() / f"{npc_session.SessionRecord(session)}_qc.ipynb",
        )
        move_to_s3(path)


if __name__ == "__main__":
    main()
