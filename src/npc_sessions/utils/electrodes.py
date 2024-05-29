from __future__ import annotations

import functools

import npc_lims
import npc_session
import numpy as np
import pandas as pd

TISSUECYTE_MICRONS_PER_PIXEL = 25


@functools.cache
def get_tissuecyte_electrodes_table(
    session: str | npc_session.SessionRecord,
) -> pd.DataFrame:
    """Get annotation data for each electrode (channel) on each probe inserted in
    a session. Column names are ready for insertion into nwb ElectrodeTable.

    >>> df = get_tissuecyte_electrodes_table('626791_2022-08-16')
    >>> df.columns
    Index(['group_name', 'channel', 'location', 'structure', 'x', 'y', 'z',
           'raw_structure'],
          dtype='object')
    """
    electrode_files = npc_lims.get_tissuecyte_annotation_files_from_s3(session)

    session_electrodes = pd.DataFrame()

    for electrode_file in electrode_files:
        probe_electrodes = pd.read_csv(electrode_file)

        probe_name = npc_session.ProbeRecord(electrode_file.stem).name

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
        [
            "group_name",
            "channel",
            "location",
            "structure",
            "x",
            "y",
            "z",
        ]
        + (
            [
                "raw_structure",
            ]
            if "raw_structure" in session_electrodes.columns
            else []
        )
    ]
    return session_electrodes


if __name__ == "__main__":
    import doctest

    import dotenv

    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
