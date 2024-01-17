from __future__ import annotations

import concurrent.futures
import datetime
import logging
import time
from typing import Literal

import npc_lims
import npc_session
import tqdm
import upath

import npc_sessions
import npc_sessions.scripts.utils as utils

logger = logging.getLogger()


QC_REPO = npc_lims.S3_SCRATCH_ROOT / "qc"


def get_qc_path(
    session_id: str | npc_session.SessionRecord,
    version: str | None = None,
) -> upath.UPath:
    """get path to notebook for session"""
    version = version or npc_lims.get_current_cache_version()
    return QC_REPO / version / f"{npc_session.SessionRecord(session_id)}_qc.ipynb"


def move(src: npc_sessions.PathLike, dest: npc_sessions.PathLike) -> None:
    """copy to dest, remove local copy"""
    src = npc_sessions.from_pathlike(src)
    dest = npc_sessions.from_pathlike(dest)
    dest.write_bytes(src.read_bytes())
    src.unlink()
    logger.info(f"moved {src.name} to {dest}")


def helper(
    session: str | npc_session.SessionRecord,
    version: str | None = None,
    skip_existing: bool = True,
) -> None:
    dest_path = get_qc_path(session_id=session, version=version)
    if skip_existing and dest_path.exists():
        logger.info(f"skipping {session} - {dest_path} already exists")
        return
    local_path = npc_sessions.write_qc_notebook(
        session,
        save_path=upath.UPath.cwd() / dest_path.name,
    )
    move(local_path, dest_path)
    logger.info(f"Finished {session}")


def write_notebooks(
    session_type: Literal["training", "ephys", "all"] = "all",
    skip_existing: bool = True,
    version: str | None = None,
    parallel: bool = True,
    max_workers: int | None = None,
) -> None:
    t0 = time.time()
    session_infos = utils.get_session_infos(session_type=session_type)

    helper_opts = {
        "version": version,
        "skip_existing": skip_existing,
    }
    if len(session_infos) == 1:
        parallel = False
    if parallel:
        future_to_session = {}
        pool = concurrent.futures.ProcessPoolExecutor(
            max_workers or utils.get_max_workers()
        )
        for session in tqdm.tqdm(session_infos, desc="Submitting jobs"):
            future_to_session[
                pool.submit(
                    helper,
                    session.id,
                    **helper_opts,  # type: ignore[arg-type]
                )
            ] = session.id
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(future_to_session.keys()),
            desc="Processing jobs",
            total=len(future_to_session),
        ):
            logger.info(f"{future_to_session[future]} done")
            continue
    else:
        for session in tqdm.tqdm(session_infos, desc="Processing jobs"):
            helper(
                session.id,
                **helper_opts,  # type: ignore[arg-type]
            )
            logger.info(f"{session.id} done")
    logger.info(f"Time elapsed: {datetime.timedelta(seconds=time.time() - t0)}")


def main() -> None:
    npc_sessions.assert_s3_write_credentials()
    kwargs = utils.setup()
    write_notebooks(**kwargs)  # type: ignore[misc, arg-type]


if __name__ == "__main__":
    main()
