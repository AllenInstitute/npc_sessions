import concurrent.futures
import datetime
import functools
import time

import npc_lims
import tqdm

import npc_sessions


def write_training_sessions_to_cache(
    skip_existing: bool = False,
) -> None:
    t0 = time.time()
    future_to_session = {}
    pool = concurrent.futures.ProcessPoolExecutor()
    for session_info in tqdm.tqdm(
        npc_lims.get_session_info(is_ephys=False), desc="Submitting jobs"
    ):
        future_to_session[
            pool.submit(
                functools.partial(
                    npc_sessions.write_all_components_to_cache,
                    skip_existing=skip_existing,
                ),
                npc_sessions.DynamicRoutingSession(session_info),
            )
        ] = session_info.id
    for future in tqdm.tqdm(
        concurrent.futures.as_completed(future_to_session.keys()),
        desc="Processing jobs",
        total=len(future_to_session),
    ):
        print(f"{future_to_session[future]} done")
        continue
    print(f"Time elapsed: {datetime.timedelta(seconds=time.time() - t0)}")


def write_ephys_sessions_to_cache(
    skip_existing: bool = False,
) -> None:
    t0 = time.time()
    future_to_session = {}
    pool = concurrent.futures.ProcessPoolExecutor()
    for session in npc_sessions.get_sessions(is_ephys=True):
        future_to_session[
            pool.submit(
                functools.partial(
                    npc_sessions.write_all_components_to_cache,
                    skip_existing=skip_existing,
                ),
                session,
            )
        ] = session.id
    for future in tqdm.tqdm(
        concurrent.futures.as_completed(future_to_session.keys()),
        desc="Processing jobs",
        total=len(future_to_session),
    ):
        print(f"{future_to_session[future]} done")
        continue
    print(f"Time elapsed: {datetime.timedelta(seconds=time.time() - t0)}")


def main() -> None:
    write_training_sessions_to_cache(skip_existing=False)
    write_ephys_sessions_to_cache(skip_existing=False)


if __name__ == "__main__":
    main()
