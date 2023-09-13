from __future__ import annotations

import functools

import npc_lims
import npc_session
import pandas as pd
import upath

S3_ELECTRODE_PATH = upath.UPath(
    "s3://aind-scratch-data/arjun.sridhar/tissuecyte_cloud_processed"
)


@functools.cache
def get_electrode_files_from_s3(
    session: str | npc_session.SessionRecord,
) -> tuple[upath.UPath, ...]:
    """
    >>> electrode_files = get_electrode_files_from_s3('626791_2022-08-16')
    >>> assert len(electrode_files) > 0
    """
    session = npc_session.SessionRecord(session)
    day = npc_lims.get_day(session)
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

    return electrode_files


def get_horizontal_vertical_positions(
    probe_electrodes: pd.DataFrame,
) -> tuple[list[int], list[int]]:
    vertical_position = 20
    horizontal_position_even = [43, 59]
    horizontal_position = horizontal_position_even[0]
    horizontal_position_even_index = 0
    horizontal_position_odd = [11, 27]
    horizontal_position_odd_index = 0

    vertical_positions: list[int] = []
    horizontal_positions: list[int] = []

    for index, _row in probe_electrodes.iterrows():
        if index != 0 and index % 2 == 0:
            vertical_position += 20

        if index == 0:
            horizontal_position = horizontal_position_even[0]
        elif index == 1:
            horizontal_position = horizontal_position_odd[0]
        elif index != 0 and index % 2 == 0:
            if horizontal_position_even_index == 0:
                horizontal_position_even_index = 1
                horizontal_position = horizontal_position_even[
                    horizontal_position_even_index
                ]
            else:
                horizontal_position_even_index = 0
                horizontal_position = horizontal_position_even[
                    horizontal_position_even_index
                ]
        elif index != 1 and index % 1 == 0:
            if horizontal_position_odd_index == 0:
                horizontal_position_odd_index = 1
                horizontal_position = horizontal_position_odd[
                    horizontal_position_odd_index
                ]
            else:
                horizontal_position_odd_index = 0
                horizontal_position = horizontal_position_odd[
                    horizontal_position_odd_index
                ]

        horizontal_positions.append(horizontal_position)
        vertical_positions.append(vertical_position)

    return horizontal_positions, vertical_positions


@functools.cache
def create_tissuecyte_electrodes_table(
    session: str | npc_session.SessionRecord,
) -> pd.DataFrame:
    electrode_files = get_electrode_files_from_s3(session)

    session_electrodes = None

    for electrode_file in electrode_files:
        string_file = str(electrode_file.stem)
        probe = string_file[
            string_file.index("_") + 1 : string_file.index("_") + 2
        ].upper()

        probe_electrodes = pd.read_csv(electrode_file)

        probe_electrodes["device_name"] = [
            f"Probe{probe}" for i in range(len(probe_electrodes))
        ]

        horizontal_positions, vertical_positions = get_horizontal_vertical_positions(
            probe_electrodes
        )
        probe_electrodes["probe_vertical_position"] = vertical_positions
        probe_electrodes["probe_horizontal_position"] = horizontal_positions

        if session_electrodes is None:
            session_electrodes = probe_electrodes
        else:
            session_electrodes = pd.concat([session_electrodes, probe_electrodes])

    if session_electrodes is not None:
        session_electrodes.rename(
            columns={
                "AP": "anterior_posterior_ccf_coordinate",
                "DV": "dorsal_ventral_ccf_coordinate",
                "ML": "left_right_ccf_coordinate",
                "region": "structure_layer",
                "region_stripped": "structure_acronym",
            },
            inplace=True,
        )

        session_electrodes["anterior_posterior_ccf_coordinate"] *= 25
        session_electrodes["dorsal_ventral_ccf_coordinate"] *= 25
        session_electrodes["left_right_ccf_coordinate"] *= 25

    return session_electrodes


if __name__ == "__main__":
    import doctest

    import dotenv

    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
