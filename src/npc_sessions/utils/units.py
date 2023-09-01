from __future__ import annotations

import io
import warnings

import npc_lims
import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl
import upath


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

    capsule_results_units = npc_lims.run_capsule_and_get_results(
        "980c5218-abef-41d8-99ed-24798d42313b", (raw_data_asset, sorted_data_asset)
    )
    npc_lims.register_session_data_asset(session_id, capsule_results_units)

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


def get_units_spike_paths_from_kilosort_codeocean_output(
    units_s3_path: tuple[upath.UPath, ...]
) -> tuple[upath.UPath, ...]:
    units_path = next(path for path in units_s3_path if "csv" in str(path))
    spike_times_path = next(path for path in units_s3_path if "spike" in str(path))
    mean_waveforms_path = next(path for path in units_s3_path if "mean" in str(path))
    sd_waveforms_path = next(path for path in units_s3_path if "sd" in str(path))

    return units_path, spike_times_path, mean_waveforms_path, sd_waveforms_path


def update_permissions_for_data_asset(session_id):
    units_data_asset = (
        npc_lims.codeocean.get_session_units_with_peak_channels_data_asset(session_id)
    )

    if not units_data_asset:
        warnings.warn(f"No units found for session {session_id}", stacklevel=2)
    else:
        npc_lims.codeocean_client.update_permissions(
            units_data_asset["id"], everyone="viewer"
        )


def get_units_spike_paths(
    session_id: str, method: str = "kilosort_codeocean"
) -> tuple[upath.UPath, ...]:
    # records = npc_lims.NWBSqliteDBHub().get_records(
    #   npc_lims.Units, session_id=session_id
    # )
    records = ()
    if not records:
        units_s3_path = npc_lims.get_units_file_from_s3(session_id)

        if units_s3_path is None:
            # TODO: DO SOMETHING TO GET UNITS - Right now, run spike nwb capsule, and register units as data asset
            if method == "kilosort_codeocean":
                run_capsules_for_units_kilosort_codeocean(session_id)
                update_permissions_for_data_asset(session_id)
                units_s3_path = npc_lims.get_units_file_from_s3(session_id)

        if units_s3_path is not None:
            if method == "kilosort_codeocean":
                return get_units_spike_paths_from_kilosort_codeocean_output(
                    units_s3_path
                )

    return records


def get_spike_times(
    units: pd.DataFrame, spike_times_path: upath.UPath
) -> dict[str, npt.NDArray[np.float64]]:
    spike_times_dict = {}
    units_names = units["unit_name"]
    with io.BytesIO(spike_times_path.read_bytes()) as f:
        spike_times = np.load(f, allow_pickle=True)

    for i in range(len(units_names)):
        spike_times_dict[units_names[i]] = spike_times[i]

    return spike_times_dict


def get_units_spike_times(
    units_spike_paths: tuple[upath.UPath, ...]
) -> tuple[pd.DataFrame, dict[str, npt.NDArray[np.float64]]]:
    units_path = next(path for path in units_spike_paths if "csv" in str(path))
    spike_times_path = next(path for path in units_spike_paths if "spike" in str(path))

    units = pl.read_csv(units_path.read_bytes()).to_pandas()
    spike_times = get_spike_times(units, spike_times_path)

    return units, spike_times
