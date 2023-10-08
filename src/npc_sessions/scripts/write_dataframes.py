from __future__ import annotations

from collections.abc import Iterable

import npc_lims
import npc_session
import pandas as pd

import npc_sessions

S3_DATAFRAME_REPO = npc_lims.NWB_REPO.parent / "dataframes"

ATTR_TO_DIR = {
    "units": "units",
    "electrodes": "metadata",
    "OptoTagging": "intervals",
    "DynamicRouting1": "intervals",
    "VisRFMapping": "intervals",
    "AudRFMapping": "intervals",
    "performance": "intervals",
    "epochs": "intervals",
    "subject": "metadata",
    "metadata": "metadata",
}


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


def write_all_ephys_session_dfs(**session_kwargs) -> None:
    attrs = ATTR_TO_DIR.keys()
    attr_to_df: dict[str, pd.DataFrame] = dict.fromkeys(attrs, pd.DataFrame())
    for file in S3_DATAFRAME_REPO.rglob("*.pkl"):
        file.unlink()
    for idx, session in enumerate(npc_sessions.get_sessions()):
        if not (session.is_sorted and session.is_annotated):
            continue
        print(f"{idx}: {session.id}")

        try:
            session_dfs: dict[str, pd.DataFrame] = get_session_dfs(
                session.id, attrs, **session_kwargs
            )
        except Exception as exc:
            print(f"{session.id} errored: {exc!r}")
        else:
            session_id = str(session.id)
            for attr in attrs:
                df = session_dfs[attr]
                if df.empty:
                    continue
                if attr == "units":
                    write_df(
                        S3_DATAFRAME_REPO / ATTR_TO_DIR[attr] / f"{session_id}.pkl",
                        df.query("default_qc"),
                        append=False,
                    )
                    print(f"wrote {session_id} {attr} df")
                    continue

                attr_to_df[attr] = pd.concat((attr_to_df[attr], df))
                write_df(
                    S3_DATAFRAME_REPO / ATTR_TO_DIR[attr] / f"{attr}.pkl",
                    df,
                    append=True,
                )
                print(f"added {session_id} {attr} df")


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
    write_all_ephys_session_dfs()


if __name__ == "__main__":
    main()
