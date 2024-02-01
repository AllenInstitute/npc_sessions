from __future__ import annotations

import concurrent.futures
import datetime
import logging
import logging.config
import multiprocessing
import time
from typing import Literal

import npc_lims
import npc_session
import tqdm

import npc_sessions
import npc_sessions.scripts.utils as utils

logger = logging.getLogger()


def helper(
    session_id: str | npc_session.SessionRecord | npc_lims.SessionInfo,
    **write_all_components_kwargs,
) -> None:
    session = npc_sessions.DynamicRoutingSession(session_id)
    logger.info(f"Processing {session.id}")
    npc_sessions.write_all_components_to_cache(
        session,
        **write_all_components_kwargs,
    )
    del session


def write_sessions_to_cache(
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
            max_workers or utils.get_max_workers(session_type),
            mp_context=multiprocessing.get_context("spawn"),
        )
        for info in tqdm.tqdm(session_infos, desc="Submitting jobs"):
            future_to_session[
                pool.submit(
                    helper,
                    session_id=info,
                    **helper_opts,  # type: ignore[arg-type]
                )
            ] = info.id
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(future_to_session.keys()),
            desc="Processing jobs",
            total=len(future_to_session),
        ):
            logger.info(f"{future_to_session[future]} done")
            continue
    else:
        for info in tqdm.tqdm(session_infos, desc="Processing jobs"):
            helper(
                session_id=info,
                **helper_opts,  # type: ignore[arg-type]
            )
            logger.info(f"{info.id} done")
    npc_sessions.consolidate_all_caches()
    logger.info(f"Time elapsed: {datetime.timedelta(seconds=time.time() - t0)}")


def main() -> None:
    kwargs = utils.setup()
    write_sessions_to_cache(**kwargs)  # type: ignore[misc, arg-type]


if __name__ == "__main__":
    main()
