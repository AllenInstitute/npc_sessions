from __future__ import annotations

import concurrent.futures
import datetime
import time
from typing import Literal

import npc_lims
import npc_session
import tqdm

import npc_sessions


def helper(
    session_id: str | npc_session.SessionRecord | npc_lims.SessionInfo,
    is_ephys: bool | None,
    **write_all_components_kwargs,
) -> None:
    if is_ephys is None:
        session = npc_sessions.DynamicRoutingSession(session_id)
    else:
        session = npc_sessions.DynamicRoutingSession(session_id, is_ephys=is_ephys)
    npc_sessions.write_all_components_to_cache(
        session,
        **write_all_components_kwargs,
    )


def write_sessions_to_cache(
    session_type: Literal["training", "ephys", "all"] = "all",
    skip_existing: bool = True,
    version: str | None = None,
    parallel: bool = True,
) -> None:
    t0 = time.time()
    if session_type not in ("training", "ephys", "all"):
        raise ValueError(f"{session_type=} must be one of 'training', 'ephys', 'all'")

    is_ephys: bool | None = None
    if session_type == "all":
        is_ephys = None
        session_infos = npc_lims.get_session_info()
    else:
        is_ephys = session_type == "ephys"
        session_infos = tuple(
            s for s in npc_lims.get_session_info() if s.is_ephys == is_ephys
        )

    if parallel:
        future_to_session = {}
        pool = concurrent.futures.ProcessPoolExecutor()
        for info in tqdm.tqdm(session_infos, desc="Submitting jobs"):
            future_to_session[
                pool.submit(
                    helper,
                    session_id=info,
                    is_ephys=is_ephys,
                    skip_existing=skip_existing,
                    version=version,
                )
            ] = info.id
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(future_to_session.keys()),
            desc="Processing jobs",
            total=len(future_to_session),
        ):
            print(f"{future_to_session[future]} done")
            continue
    else:
        for info in tqdm.tqdm(session_infos, desc="Processing jobs"):
            helper(
                session_id=info,
                is_ephys=is_ephys,
                skip_existing=skip_existing,
                version=version,
            )
            print(f"{info.id} done")
    print(f"Time elapsed: {datetime.timedelta(seconds=time.time() - t0)}")


def write_ephys_sessions_to_cache() -> None:
    write_sessions_to_cache(session_type="ephys")


def write_training_sessions_to_cache() -> None:
    write_sessions_to_cache(session_type="training")


def main() -> None:
    write_sessions_to_cache()


if __name__ == "__main__":
    main()
