from __future__ import annotations

import functools
import io
import os
from typing import Iterable

import npc_lims
import npc_session
import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl
import upath

import npc_sessions.utils as utils


@functools.cache
def get_units_electrodes(
    session: str, sorting_method="codeocean_ks25", annotation_method="tissuecyte"
) -> pd.DataFrame:
    if sorting_method == "codeocean_ks25":
        units_path = npc_lims.get_units_codeoean_kilosort_path_from_s3(session)
        units = pd.read_csv(
            units_path,
            storage_options={
                "key": os.getenv("AWS_ACCESS_KEY_ID"),
                "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
            },
        )
    if annotation_method == "tissuecyte":
        units = merge_units_electrodes(session, units)

    units.drop(columns=["electrodes"], inplace=True)
    return units


def merge_units_electrodes(session: str, units: pd.DataFrame) -> pd.DataFrame:
    try:
        electrodes = utils.get_tissuecyte_electrodes_table(session)
        units = units.merge(
            electrodes,
            left_on=["group_name", "peak_channel"],
            right_on=["device_name", "channel"],
        )
        units.drop(columns=["channel"], inplace=True)
    except FileNotFoundError as e:
        print(str(e) + ". Returning units without electrodes")

    return units


def bin_spike_times(
    spike_times: npt.NDArray[np.float64], bin_time: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    spike_times = np.concatenate(spike_times, axis=0)  # flatten array
    return np.histogram(spike_times, bins=np.arange(0, np.max(spike_times), bin_time))


@functools.cache
def get_unit_spike_times_dict(
    session: str, unit_ids: tuple[str, ...], sorting_method="codeocean_ks25"
) -> dict[str, npt.NDArray[np.float64]]:
    # change function call depending on sorting_method
    spike_times_dict = {}
    if sorting_method == "codeocean_ks25":
        spike_times_path = npc_lims.get_spike_times_codeocean_kilosort_path_from_s3(
            session
        )
        with io.BytesIO(spike_times_path.read_bytes()) as f:
            spike_times = np.load(f, allow_pickle=True)

        for i in range(len(unit_ids)):
            spike_times_dict[unit_ids[i]] = spike_times[i]

    return spike_times_dict


def get_units_electrodes_spike_times(session: str, *args, **kwargs) -> pl.DataFrame:
    units_df = get_units_electrodes(session, *args, **kwargs)
    unit_ids = units_df["unit_name"].to_list()
    spike_times_dict = get_unit_spike_times_dict(
        session, tuple(unit_ids), *args, **kwargs
    )
    spike_times_df = pl.DataFrame(
        {
            "unit_name": tuple(spike_times_dict.keys()),
            "spike_times": tuple(t for t in spike_times_dict.values()),
        }
    )
    units = pl.DataFrame(units_df).join(spike_times_df, on="unit_name")
    return (
        units.with_columns(
            pl.col("device_name").str.replace("P", "p").alias("device_name"),
            pl.concat_str(
                pl.col("device_name").str.replace("Probe", f"{session}_"),
                pl.col("ks_unit_id").cast(pl.Int64),
                separator="_",
            ).alias("unit_id"),
        ).select(
            pl.exclude("unit_name", "id"),
        )
        # .filter(
        #    "default_qc",
        # )
    )


def get_aligned_spike_times(
    spike_times: npt.NDArray[np.floating],
    device_timing_on_sync: utils.EphysTimingInfoOnSync,
) -> npt.NDArray[np.float64]:
    return (
        spike_times / device_timing_on_sync.sampling_rate
    ) + device_timing_on_sync.start_time


def get_closest_channel(
    unit_location: npt.NDArray[np.float64], channel_positions: npt.NDArray[np.int64]
) -> int:
    distances = np.sum((channel_positions - unit_location) ** 2, axis=1)
    return int(np.argmin(distances))


def get_peak_channels(
    units_locations: npt.NDArray[np.float64],
    electrode_positions: tuple[dict[str, tuple[int, int]], ...],
) -> list[int]:
    peak_channels: list[int] = []
    channel_positions = np.array(list(electrode_positions[0].values()))

    for location in units_locations:
        unit_location = np.array([location[0], location[1]])
        peak_channels.append(get_closest_channel(unit_location, channel_positions))

    return peak_channels


def make_units_metrics_device_table_ks25(
    quality_metrics_device_df: pd.DataFrame,
    template_metrics_device_df: pd.DataFrame,
    units_locations: npt.NDArray[np.floating],
    device_name: str,
    electrode_positions: tuple[dict[str, tuple[int, int]], ...],
) -> pd.DataFrame:
    df_quality_template_metrics = quality_metrics_device_df.merge(
        template_metrics_device_df, left_index=True, right_index=True
    )

    peak_channels = get_peak_channels(units_locations, electrode_positions)
    df_quality_template_metrics["peak_channel"] = peak_channels
    df_quality_template_metrics["device_name"] = [
        device_name for i in range(len(df_quality_template_metrics))
    ]

    return df_quality_template_metrics


def get_amplitudes_mean_waveforms_ks25(
    templates: npt.NDArray[np.floating], ks_unit_ids: npt.NDArray[np.int64]
) -> tuple[list[float], list[npt.NDArray[np.float64]]]:
    unit_amplitudes: list[float] = []
    templates_mean: list[npt.NDArray[np.float64]] = []

    """
    @property
    def nbefore(self) -> int:
        nbefore = int(self._params["ms_before"] * self.sampling_frequency / 1000.0)
        return nbefore
    """
    before = 90  # from spike interface, ms_before = 3, TODO: look at this further
    for index, _ks_unit_id in enumerate(ks_unit_ids):
        template = templates[index, :, :]
        values = -template[before, :]
        unit_amplitudes.append(np.max(values))
        templates_mean.append(template)

    return unit_amplitudes, templates_mean


def get_units_spike_times_ks25(
    sorting_cached: dict[str, npt.NDArray],
    spike_times_aligned: npt.NDArray[np.float64],
    ks_unit_ids: npt.NDArray[np.int64],
) -> list[npt.NDArray[np.float64]]:
    units_spike_times: list[npt.NDArray[np.float64]] = []

    spike_labels = sorting_cached['spike_labels_seg0']

    for ks_unit_id in ks_unit_ids:
        label_indices = np.argwhere(spike_labels == ks_unit_id)
        unit_spike_time = spike_times_aligned[label_indices].flatten()
        units_spike_times.append(unit_spike_time)

    return units_spike_times


def make_units_table_from_spike_interface_ks25(
    session_or_spikeinterface_data_or_path: str | npc_session.SessionRecord | utils.PathLike | utils.SpikeInterfaceKS25Data,
    settings_xml_data_or_path: utils.PathLike | utils.SettingsXmlInfo,
    devices_timing: Iterable[utils.EphysTimingInfoOnSync],
) -> pd.DataFrame:
    """
    >>> devices_timing = utils.get_ephys_timing_on_sync(npc_lims.get_h5_sync_from_s3('662892_20230821'), npc_lims.get_recording_dirs_experiment_path_from_s3('662892_20230821'))
    >>> settings_xml_data = utils.get_settings_xml_data(npc_lims.get_settings_xml_path_from_s3('662892_20230821'))
    >>> units = make_units_table_from_spike_interface_ks25('662892_20230821', settings_xml_data, devices_timing)
    CCF annotations for 662892_2023-08-21 have not been uploaded to s3. Returning units without electrodes
    >>> len(units[units['device_name'] == 'probeA'])
    237
    """
    units = pd.DataFrame()
    settings_xml_info = utils.get_settings_xml_data(settings_xml_data_or_path)
    device_names = settings_xml_info.probe_letters
    electrode_positions = settings_xml_info.channel_pos_xy
    device_names_probe = tuple(f"probe{name}" for name in device_names)
    spike_interface_data = utils.get_spikeinterface_data(session_or_spikeinterface_data_or_path)

    for device_name in device_names_probe:
        device_timing_on_sync = next(
            timing for timing in devices_timing if device_name[0].upper()+device_name[1:] in timing.name
        )

        df_device_metrics = make_units_metrics_device_table_ks25(
            spike_interface_data.quality_metrics_df(device_name),
            spike_interface_data.template_metrics_df(device_name),
            spike_interface_data.unit_locations(device_name),
            device_name,
            electrode_positions,
        )

        amplitudes, mean_waveforms = get_amplitudes_mean_waveforms_ks25(
            spike_interface_data.templates_average(device_name), df_device_metrics.index.values
        )
        spike_times_aligned = get_aligned_spike_times(
            spike_interface_data.sorting_cached(device_name)["spike_indexes_seg0"], device_timing_on_sync
        )
        unit_spike_times = get_units_spike_times_ks25(
            spike_interface_data.sorting_cached(device_name),
            spike_times_aligned,
            df_device_metrics.index.values,
        )

        df_device_metrics["default_qc"] = spike_interface_data.default_qc(device_name)
        df_device_metrics["amplitude"] = amplitudes
        df_device_metrics["waveform_mean"] = mean_waveforms
        df_device_metrics["spike_times"] = unit_spike_times
        df_device_metrics['ks_unit_id'] = df_device_metrics.index.to_list()

        units = pd.concat([units, df_device_metrics])

    return units

def format_unit_ids(units: pd.DataFrame, session: str | npc_session.SessionRecord) -> pd.DataFrame:
    """Add session and probe letter"""
    units["unit_id"] = [
        f"{session}_{row.device_name.replace('probe', '')}-{row.unit_id}" 
        if session not in row.unit_id
        else row.unit_id
        for _, row in units.iterrows()
        ]
    return units

if __name__ == "__main__":
    import doctest

    import dotenv

    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
