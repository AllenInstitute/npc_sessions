from __future__ import annotations

import pickle
import warnings

import npc_lims
import npc_session
import numpy as np
import numpy.typing as npt
import pandas as pd
import SimpleITK as sitk
import upath

NETWORK_ELECTRODE_PATH = upath.UPath(
    "//allen/programs/mindscope/workgroups/np-behavior/tissuecyte"
)


def get_acronym_map() -> dict[str, int]:
    # TODO get from allen brain map
    return pickle.loads(
        upath.UPath(
            "//allen/programs/mindscope/workgroups/np-behavior/tissuecyte/field_reference/acrnm_map.pkl"
        ).read_bytes()
    )


def get_annotation_volume() -> npt.NDArray[np.int64]:
    # TODO get from somewhere in the cloud
    return sitk.GetArrayFromImage(
        sitk.ReadImage(
            upath.UPath(
                "//allen/programs/mindscope/workgroups/np-behavior/tissuecyte/field_reference/ccf_ano.mhd"
            )
        )
    )


def get_structure_acronym(
    acronym_map: dict[str, int],
    annotation_volume: npt.NDArray[np.int64],
    point: tuple[int, int, int],
) -> str:
    if point[1] < 0:
        return "root"

    structure_ids = tuple(acronym_map.values())
    labels = tuple(acronym_map.keys())

    structure_id = annotation_volume[point[0], point[1], point[2]]
    if structure_id in structure_ids:
        index = structure_ids.index(structure_id)
        label = labels[index]
    else:
        label = "root"

    return label


def get_electrode_files_from_network(
    session: str | npc_session.SessionRecord,
) -> tuple[upath.UPath, ...] | None:
    session = npc_session.SessionRecord(session)
    all_subject_sessions = npc_lims.get_subject_data_assets(session.subject)
    raw_subject_assets = sorted(
        asset["name"]
        for asset in all_subject_sessions
        if npc_lims.is_raw_data_asset(asset)
    )
    day = tuple(
        raw_subject_assets.index(asset) + 1
        for asset in raw_subject_assets
        if f"{session.subject}_{session.date}" in asset
    )[0]

    subject_electrode_network_path = NETWORK_ELECTRODE_PATH / str(session.subject.id)

    if not subject_electrode_network_path.exists():
        warnings.warn(f"Session {session} does not have any annotations", stacklevel=2)
        return None

    electrode_files = tuple(
        subject_electrode_network_path.glob(
            f"Probe_*{day}_channels_{str(session.subject.id)}_warped.csv"
        )
    )
    if not electrode_files:
        warnings.warn(f"Session {session.id} channels have not been aligned")
        return None

    return electrode_files


def get_electrodes_from_network(
    session: str | npc_session.SessionRecord,
) -> pd.DataFrame | None:
    electrode_files = get_electrode_files_from_network(session)

    if not electrode_files:
        return None

    session_electrodes = None
    volume = get_annotation_volume()
    map = get_acronym_map()

    for electrode_file in electrode_files:
        string_file = str(electrode_file)
        probe = string_file[string_file.index("_") + 1 : string_file.index("_") + 2]

        probe_electrodes = pd.read_csv(electrode_file)

        vertical_position = 20
        horizontal_position_even = [43, 59]
        horizontal_position = horizontal_position_even[0]
        horizontal_position_even_index = 0
        horizontal_position_odd = [11, 27]
        horizontal_position_odd_index = 0

        vertical_positions: list[int] = []
        horizontal_positions: list[int] = []

        for index, row in probe_electrodes.iterrows():
            if pd.isna(row.region):
                label = get_structure_acronym(map, volume, (row.AP, row.DV, row.ML))

                probe_electrodes.loc[index, "region"] = label

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

        probe_electrodes["device_name"] = [
            f"Probe{probe}" for i in range(len(probe_electrodes))
        ]
        probe_electrodes["probe_vertical_position"] = vertical_positions
        probe_electrodes["probe_horizontal_position"] = horizontal_positions

        if session_electrodes is None:
            session_electrodes = probe_electrodes
        else:
            session_electrodes = pd.concat([session_electrodes, probe_electrodes])

    return session_electrodes


if __name__ == "__main__":
    import doctest

    import dotenv

    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
