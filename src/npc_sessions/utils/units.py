import os

import npc_lims
import pandas as pd


def run_capsules_for_units_kilosort_codeocean(session_id: str) -> None:
    raw_data_asset = npc_lims.get_session_raw_data_asset(session_id)
    if raw_data_asset is None:
        raise ValueError(
            f"Session {session_id} has no raw data asset."
        )  # TODO move to function in npc_lims
    sorted_data_asset = npc_lims.get_session_sorted_data_asset(session_id)
    if sorted_data_asset is None:
        raise ValueError(
            f"Session {session_id} has no sorted data asset."
        )  # TODO move to function in npc_lims
    # capsule_results_units = npc_lims.run_capsule_and_get_results('980c5218-abef-41d8-99ed-24798d42313b',
    # (raw_data_asset, sorted_data_asset))
    # npc_lims.register_session_data_asset(self.id, capsule_results_units)

    units_no_peak_channel_asset = npc_lims.get_session_units_data_asset(session_id)
    if units_no_peak_channel_asset is None:
        raise ValueError(
            f"Session {session_id} has no peak_channel data asset."
        )  # TODO move to function in npc_lims

    capsule_result_units_peak_channels = npc_lims.run_capsule_and_get_results(
        "d1a5c3a8-8fb2-4cb0-8e9e-96e6e1d03ff1",
        (raw_data_asset, sorted_data_asset, units_no_peak_channel_asset),
    )
    npc_lims.register_session_data_asset(session_id, capsule_result_units_peak_channels)


def get_units(session_id: str, method: str = "kilosort_codeocean") -> tuple:
    records = npc_lims.NWBSqliteDBHub().get_records(
        npc_lims.Units, session_id=session_id
    )
    if not records:
        units_s3_path = npc_lims.get_units_file_from_s3(session_id)

        if units_s3_path is None:
            # TODO: DO SOMETHING TO GET UNITS - Right now, run spike nwb capsule, and register units as data asset
            if method == "kilosort_codeocean":
                run_capsules_for_units_kilosort_codeocean(session_id)
                units_s3_path = npc_lims.get_units_file_from_s3(session_id)

        if units_s3_path is not None:  # TODO: change to polars
            pd.read_csv(
                units_s3_path,
                storage_options={
                    "key": os.getenv("AWS_ACCESS_KEY_ID"),
                    "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
                },
            )
            print()

    return ()
