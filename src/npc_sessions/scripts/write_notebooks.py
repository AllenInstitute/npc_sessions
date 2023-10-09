from __future__ import annotations

import importlib.resources
import logging
import os
import subprocess
import sys

import npc_lims
import npc_session
import upath

import npc_sessions

logger = logging.getLogger(__name__)

QC_REPO = npc_lims.DR_DATA_REPO.parent.parent / "qc"

QC_NOTEBOOK = npc_sessions.from_pathlike(
    importlib.resources.files("npc_sessions") / "notebooks" / "dynamic_routing_qc.ipynb"
)


def write_qc_notebook(
    session: str | npc_session.SessionRecord | npc_sessions.DynamicRoutingSession,
    **session_kwargs,
) -> upath.UPath:
    """Execute and save (as .ipynb) the DR QC notebook for a given session."""
    if not isinstance(session, npc_sessions.DynamicRoutingSession):
        session = npc_sessions.DynamicRoutingSession(session, **session_kwargs)

    # pass session to run in notebook via env var
    if session.info:
        var = session.info.allen_path.as_posix()
    else:
        var = str(session.id)
    os.environ["NPC_SESSION_ID"] = var

    new_name = session.id if not session.info else session.info.allen_path.stem
    logger.info(f"running {QC_NOTEBOOK.name} for {session.id}")
    subprocess.run(
        f"jupyter nbconvert --to notebook --execute {QC_NOTEBOOK} --allow-errors --output {new_name}",
        shell=True,
        check=True,
        capture_output=False,
        env=os.environ,
    )
    return QC_NOTEBOOK.with_name(f"{new_name}.ipynb")


def move_to_s3(file: npc_sessions.PathLike) -> None:
    """copy to s3, remove local copy"""
    file = npc_sessions.from_pathlike(file)
    (s3 := QC_REPO / file.name).write_bytes(file.read_bytes())
    file.unlink()
    logger.info(f"moved {file.name} to {s3.parent}")


def main() -> None:
    npc_sessions.assert_s3_write_credentials()
    sessions = sys.argv[1:] or (s.id for s in npc_lims.get_session_info())
    for session in sessions:
        path = write_qc_notebook(session)
        move_to_s3(path)


if __name__ == "__main__":
    main()
