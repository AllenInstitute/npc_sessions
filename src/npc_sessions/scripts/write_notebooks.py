from __future__ import annotations

import concurrent.futures
import logging
import sys

import npc_lims
import npc_session
import tqdm
import upath

import npc_sessions

logger = logging.getLogger(__name__)

QC_REPO = npc_lims.DR_DATA_REPO.parent.parent / "qc"


def get_qc_path(session_id: str | npc_session.SessionRecord) -> upath.UPath:
    """get path to notebook for session"""
    return QC_REPO / f"{npc_session.SessionRecord(session_id)}_qc.ipynb"


def move(src: npc_sessions.PathLike, dest: npc_sessions.PathLike) -> None:
    """copy to dest, remove local copy"""
    src = npc_sessions.from_pathlike(src)
    dest = npc_sessions.from_pathlike(dest)
    dest.write_bytes(src.read_bytes())
    src.unlink()
    logger.info(f"moved {src.name} to {dest}")


def helper(session: str | npc_session.SessionRecord) -> None:
    dest_path = get_qc_path(session)
    local_path = npc_sessions.write_qc_notebook(
        session,
        save_path=upath.UPath.cwd() / dest_path.name,
    )
    move(local_path, dest_path)
    logger.info(f"done with {session}")


def write_notebooks(parallel: bool = True) -> None:
    npc_sessions.assert_s3_write_credentials()
    sessions = sys.argv[1:] or tuple(
        s.id for s in npc_lims.get_session_info(is_ephys=True)
    )
    if len(sessions) < 5:
        parallel = False
    if parallel:
        future_to_session = {}
        pool = concurrent.futures.ProcessPoolExecutor()
        for session in tqdm.tqdm(sessions, desc="Submitting jobs"):
            future_to_session[pool.submit(helper, session)] = session.id
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(future_to_session.keys()),
            desc="Processing jobs",
            total=len(future_to_session),
        ):
            print(f"{future_to_session[future]} done")
            continue
    else:
        for session in sessions:
            helper(session)
    logger.info("done")


def main() -> None:
    write_notebooks()


if __name__ == "__main__":
    main()
