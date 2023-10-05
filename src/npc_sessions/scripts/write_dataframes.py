from __future__ import annotations

import concurrent.futures
import contextlib
import pathlib
from collections.abc import Iterable

import npc_session
import pandas as pd

import npc_sessions


def get_session_dfs(
    session: str | npc_session.SessionRecord | npc_sessions.DynamicRoutingSession,
    attrs: Iterable[str],
    **session_kwargs,
) -> dict[str, pd.DataFrame]:
    if not isinstance(session, npc_sessions.DynamicRoutingSession):
        session = npc_sessions.DynamicRoutingSession(session, **session_kwargs)
    session_dfs = dict.fromkeys(attrs, pd.DataFrame())

    for attr in attrs:
        container = getattr(session, attr, session.intervals.get(attr, None))
        if container is None:
            continue

        if hasattr(container, "to_dataframe"):
            df = container.to_dataframe()

            # drop columns containing linked pynwb objs
            if attr == "units":
                if "electrodes" in df.columns:
                    df = df.drop(columns=["electrodes"])
                if "electrode_group" in df.columns:
                    df = df.drop(columns=["electrode_group"])
            if attr == "electrodes" and "group" in df.columns:
                df = df.drop(columns=["group"])

        else:
            container = container._AbstractContainer__field_values
            if attr == "metadata" and "subject" in container:
                del container["subject"]
            df = pd.DataFrame.from_dict(container, orient="index").T

        df["session_id"] = [str(session.id)] * len(df)
        df.set_index("session_id", inplace=True)

        session_dfs[attr] = pd.concat((session_dfs[attr], df))
    return session_dfs


def get_all_ephys_session_dfs(**session_kwargs) -> dict[str, pd.DataFrame]:
    attrs = (
        "units",
        "electrodes",
        "OptoTagging",
        "DynamicRouting1",
        "VisRFMapping",
        "AudRFMapping",
        "performance",
        "epochs",
        "subject",
        "metadata",
    )
    all_session_dfs = dict.fromkeys(attrs, pd.DataFrame())
    session_to_future: dict[npc_session.SessionRecord, concurrent.futures.Future] = {}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for _idx, session in enumerate(
            npc_sessions.get_sessions()
        ):  # only tracked-ephys sessions
            if not (session.is_sorted and session.is_annotated):
                continue
            session_to_future[session.id] = executor.submit(
                get_session_dfs, session.id, attrs, **session_kwargs
            )
    for future in concurrent.futures.as_completed(session_to_future.values()):
        with contextlib.suppress(Exception):
            session_dfs = future.result()
            for attr in attrs:
                all_session_dfs[attr] = pd.concat(
                    (all_session_dfs[attr], session_dfs[attr])
                )
    return all_session_dfs


def main() -> None:
    attr_to_df = get_all_ephys_session_dfs()
    path = pathlib.Path("results")
    path.mkdir(exist_ok=True)
    for attr, df in attr_to_df.items():
        if not any(df):
            continue
        df.to_pickle(path / f"{attr}.pkl")


if __name__ == "__main__":
    main()

    ## single session test:
    # for session in npc_sessions.get_sessions():
    #     if not (session.is_sorted and session.is_annotated):
    #         continue
    #     break
    # print(session)
    # attr_to_df = get_session_dfs(
    #     session.id,
    #     attrs = (
    #         "units",
    #         "metadata",
    #         "electrodes",
    #         "epochs",
    #         "OptoTagging",
    #         "DynamicRouting1",
    #         "VisRFMapping",
    #         "AudRFMapping",
    #         "performance",
    #         "subject",
    #     ),
    # )
