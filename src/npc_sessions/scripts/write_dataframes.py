from __future__ import annotations

import concurrent.futures
from collections.abc import Iterable

import loky
import npc_lims
import npc_session
import pandas as pd

import npc_sessions

S3_DATAFRAME_REPO = npc_lims.NWB_REPO.parent / "dataframes"


def get_session_dfs(
    session: str | npc_session.SessionRecord | npc_sessions.DynamicRoutingSession,
    attrs: Iterable[str],
    **session_kwargs,
) -> dict[str, pd.DataFrame]:
    if not isinstance(session, npc_sessions.DynamicRoutingSession):
        session = npc_sessions.DynamicRoutingSession(session, **session_kwargs)
    attr_to_df = dict.fromkeys(attrs, pd.DataFrame())

    for attr in attrs:
        container = getattr(session, attr, session.intervals.get(attr, None))
        if container is None:
            continue

        if hasattr(container, "to_dataframe"):
            df = container.to_dataframe()

            # drop columns containing linked pynwb objs
            if attr == "units":
                for col in (
                    "electrodes",
                    "electrode_group",
                    "waveform_sd",
                    "waveform_mean",
                ):
                    if col in df.columns:
                        df = df.drop(columns=[col])
            if attr == "electrodes" and "group" in df.columns:
                df = df.drop(columns=["group"])

        else:
            container = container._AbstractContainer__field_values
            if attr == "metadata" and "subject" in container:
                del container["subject"]
            df = pd.DataFrame.from_dict(container, orient="index").T

        df["session_id"] = [str(session.id)] * len(df)
        df.set_index("session_id", inplace=True)

        attr_to_df[attr] = pd.concat((attr_to_df[attr], df))
    return attr_to_df


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
    future_to_session_id: dict[
        concurrent.futures.Future, npc_session.SessionRecord
    ] = {}
    attr_to_df: dict[str, pd.DataFrame] = dict.fromkeys(attrs, pd.DataFrame())

    for idx, session in enumerate(npc_sessions.get_sessions()):
        if not (session.is_sorted and session.is_annotated):
            continue
        print(idx)
        future = loky.get_reusable_executor(max_workers=4).submit(
            get_session_dfs, session.id, attrs, **session_kwargs
        )
        future_to_session_id[future] = session.id
        print(f"submitted {session.id}")
        if idx > 1:
            break

    while future_to_session_id:
        # for future in concurrent.futures.as_completed(future_to_session_id):
        # with contextlib.suppress(Exception):
        session_dfs: dict[str, pd.DataFrame] = future.result()
        for attr in attrs:
            if not (df := session_dfs[attr]).empty:
                attr_to_df[attr] = pd.concat((attr_to_df[attr], df))
                print(f"added {future_to_session_id[future]} {attr} df")
        del future_to_session_id[future]
    if all(df.empty for df in attr_to_df.values()):
        raise RuntimeError("No dataframes were created")
    return attr_to_df


def write_df(
    path: npc_sessions.PathLike, df: pd.DataFrame, append: bool = False
) -> None:
    path = npc_sessions.from_pathlike(path)
    for parent in path.parents:
        if parent.exists():
            break
        parent.mkdir(exist_ok=True)
    if append and path.is_file():
        df = pd.concat((pd.read_pickle(path), df))
    df.to_pickle(path)


def main() -> None:
    npc_sessions.assert_s3_write_credentials()  # before we do a load of work, make sure we can write to s3
    for attr, df in get_all_ephys_session_dfs().items():
        write_df(S3_DATAFRAME_REPO / f"{attr}.pkl", df, append=False)


if __name__ == "__main__":
    main()
