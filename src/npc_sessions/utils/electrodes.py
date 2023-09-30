from __future__ import annotations

import functools

import npc_lims
import npc_session
import numpy as np
import pandas as pd
import upath

S3_ELECTRODE_PATH = upath.UPath(
    "s3://aind-scratch-data/arjun.sridhar/tissuecyte_cloud_processed"
)

TISSUECYTE_MICRONS_PER_PIXEL = 25


@functools.cache
def get_tissuecyte_annotation_files_from_s3(
    session: str | npc_session.SessionRecord,
) -> tuple[upath.UPath, ...]:
    """For each probe inserted, get a csv file containing CCF coordinates for each
    electrode (channel) on the probe.

    >>> electrode_files = get_tissuecyte_annotation_files_from_s3('626791_2022-08-16')
    >>> assert len(electrode_files) > 0
    >>> electrode_files[0].name
    'Probe_A2_channels_626791_warped_processed.csv'
    """
    session = npc_session.SessionRecord(session)
    day = npc_lims.get_session_info(session).day
    subject_electrode_network_path = S3_ELECTRODE_PATH / str(session.subject.id)

    if not subject_electrode_network_path.exists():
        raise FileNotFoundError(
            f"CCF annotations for {session} have not been uploaded to s3"
        )

    electrode_files = tuple(
        subject_electrode_network_path.glob(
            f"Probe_*{day}_channels_{str(session.subject.id)}_warped_processed.csv"
        )
    )
    if not electrode_files:
        raise FileNotFoundError(
            f"{subject_electrode_network_path} exists, but no CCF annotation files found matching {day} and {session.subject.id} - check session day"
        )

    return electrode_files


@functools.cache
def get_tissuecyte_electrodes_table(
    session: str | npc_session.SessionRecord,
) -> pd.DataFrame:
    """Get annotation data for each electrode (channel) on each probe inserted in
    a session. Column names are ready for insertion into nwb ElectrodeTable.

    >>> df = get_tissuecyte_electrodes_table('626791_2022-08-16')
    >>> df.columns
    Index(['group_name', 'channel', 'location', 'structure', 'x', 'y', 'z'], dtype='object')
    """
    electrode_files = get_tissuecyte_annotation_files_from_s3(session)

    session_electrodes = pd.DataFrame()

    for electrode_file in electrode_files:
        probe_electrodes = pd.read_csv(electrode_file)

        probe_name = npc_session.ProbeRecord(electrode_file.stem)

        probe_electrodes["group_name"] = [str(probe_name)] * len(probe_electrodes)

        session_electrodes = pd.concat([session_electrodes, probe_electrodes])

    session_electrodes.rename(
        columns={
            "AP": "x",
            "DV": "y",
            "ML": "z",
            "region": "location",
            "region_stripped": "structure",
        },
        inplace=True,
    )
    for column in ("x", "y", "z"):
        # -1 is code for "not inserted": make this NaN
        session_electrodes[column] = session_electrodes[column].replace(-1, np.nan)
        session_electrodes[column] *= TISSUECYTE_MICRONS_PER_PIXEL
    session_electrodes = session_electrodes[
        ["group_name", "channel", "location", "structure", "x", "y", "z"]
    ]
    return session_electrodes


if __name__ == "__main__":
    import doctest

    import dotenv

    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
