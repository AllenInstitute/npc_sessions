from __future__ import annotations

import concurrent.futures
import functools
import io
import logging
import os
from collections.abc import Iterable

import npc_lims
import npc_session
import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl

import npc_sessions.utils as utils

logger = logging.getLogger(__name__)


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
        units = add_tissuecyte_annotations(units, session)

    units.drop(columns=["electrodes"], inplace=True)
    return units


def bin_spike_times(
    spike_times: npt.NDArray[np.float64], bin_interval: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    spike_times = np.concatenate(spike_times, axis=0)  # flatten array
    return np.histogram(
        spike_times,
        bins=np.arange(
            np.floor(np.min(spike_times)), np.ceil(np.max(spike_times)), bin_interval
        ),
    )


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
    unit_location: npt.NDArray[np.float64], channel_positions: npt.NDArray[np.floating]
) -> int:
    distances = np.sum((channel_positions - unit_location) ** 2, axis=1)
    return int(np.argmin(distances))


def get_peak_channels(
    units_locations: npt.NDArray[np.float64],
    electrode_positions: npt.NDArray[np.floating],
) -> list[int]:
    peak_channels: list[int] = []

    for location in units_locations:
        unit_location = np.array([location[0], location[1]])
        peak_channels.append(get_closest_channel(unit_location, electrode_positions))

    return peak_channels


def get_amplitudes_mean_waveforms_peak_channels_ks25(
    templates: npt.NDArray[np.floating],
    ks_unit_ids: npt.NDArray[np.int64],
    sampling_rate: float,
    post_processed_params: dict,
) -> tuple[list[float], list[npt.NDArray[np.floating]], list[np.int64]]:
    unit_amplitudes: list[float] = []
    templates_mean: list[npt.NDArray[np.floating]] = []
    peak_channels = []

    """
    @property
    def nbefore(self) -> int:
        nbefore = int(self._params["ms_before"] * self.sampling_frequency / 1000.0)
        return nbefore
    """
    # https://github.com/SpikeInterface/spikeinterface/blob/777a07d3a538394d52a18a05662831a403ee35f9/src/spikeinterface/core/template_tools.py#L8
    before = int(
        post_processed_params["ms_before"] * sampling_rate / 1000.0
    )  # from spike interface, ms_before = 3, # TODO: #38 @arjunsridhar12345 look at this further
    for index, _ks_unit_id in enumerate(ks_unit_ids):
        template = templates[index, :, :]
        values = -template[before, :]
        # emailed Josh to see how he was getting peak channel - using waveforms, peak channel might be part of metrics in future
        peak_channel = np.argmax(values)
        unit_amplitudes.append(values[peak_channel])
        templates_mean.append(template)
        peak_channels.append(peak_channel)

    return unit_amplitudes, templates_mean, peak_channels


def get_waveform_sd_ks25(
    templates_std: npt.NDArray[np.floating], ks_unit_ids: list[int]
) -> list[npt.NDArray[np.floating]]:
    unit_templates_std: list[npt.NDArray[np.floating]] = []
    for index, _ks_unit_id in enumerate(ks_unit_ids):
        template = templates_std[index, :, :]
        unit_templates_std.append(template)

    return unit_templates_std


def get_units_spike_times_ks25(
    sorting_cached: dict[str, npt.NDArray],
    spike_times_aligned: npt.NDArray[np.float64],
    ks_unit_ids: npt.NDArray[np.int64],
) -> list[npt.NDArray[np.float64]]:
    units_spike_times: list[npt.NDArray[np.float64]] = []

    spike_labels = sorting_cached["spike_labels_seg0"]

    for ks_unit_id in ks_unit_ids:
        label_indices = np.argwhere(spike_labels == ks_unit_id)
        unit_spike_time = spike_times_aligned[label_indices].flatten()
        units_spike_times.append(unit_spike_time)

    return units_spike_times


def _device_helper(
    device_timing_on_sync: utils.EphysTimingInfoOnSync,
    spike_interface_data: utils.SpikeInterfaceKS25Data,
    include_waveform_arrays: bool,
) -> pd.DataFrame:
    electrode_group_name = npc_session.ProbeRecord(
        device_timing_on_sync.device.name
    ).name
    spike_interface_data.electrode_locations_xy(electrode_group_name)

    df_device_metrics = spike_interface_data.quality_metrics_df(
        electrode_group_name
    ).merge(
        spike_interface_data.template_metrics_df(electrode_group_name),
        left_index=True,
        right_index=True,
    )

    df_device_metrics["electrode_group_name"] = [str(electrode_group_name)] * len(
        df_device_metrics
    )

    (
        amplitudes,
        mean_waveforms,
        peak_channels,
    ) = get_amplitudes_mean_waveforms_peak_channels_ks25(
        spike_interface_data.templates_average(electrode_group_name),
        df_device_metrics.index.values,
        device_timing_on_sync.sampling_rate,
        spike_interface_data.postprocessed_params_dict(electrode_group_name),
    )

    df_device_metrics["peak_channel"] = peak_channels

    spike_times_aligned = get_aligned_spike_times(
        spike_interface_data.sorting_cached(electrode_group_name)["spike_indexes_seg0"],
        device_timing_on_sync,
    )
    unit_spike_times = get_units_spike_times_ks25(
        spike_interface_data.sorting_cached(electrode_group_name),
        spike_times_aligned,
        df_device_metrics.index.values,  # TODO #37 @arjunsridhar12345 is this safe?
    )

    df_device_metrics["default_qc"] = spike_interface_data.default_qc(
        electrode_group_name
    )
    df_device_metrics["amplitude"] = amplitudes
    if include_waveform_arrays:
        df_device_metrics["waveform_mean"] = mean_waveforms
        df_device_metrics["waveform_sd"] = get_waveform_sd_ks25(
            spike_interface_data.templates_std(electrode_group_name),
            df_device_metrics.index.to_list(),
        )
    df_device_metrics["spike_times"] = unit_spike_times
    df_device_metrics["unit_id"] = df_device_metrics.index.to_list()

    return df_device_metrics


def make_units_table_from_spike_interface_ks25(
    session_or_spikeinterface_data_or_path: str
    | npc_session.SessionRecord
    | utils.PathLike
    | utils.SpikeInterfaceKS25Data,
    device_timing_on_sync: Iterable[utils.EphysTimingInfoOnSync],
    include_waveform_arrays: bool = False,
) -> pd.DataFrame:
    """
    >>> device_timing_on_sync = utils.get_ephys_timing_on_sync(npc_lims.get_h5_sync_from_s3('662892_20230821'), npc_lims.get_recording_dirs_experiment_path_from_s3('662892_20230821'))
    >>> units = make_units_table_from_spike_interface_ks25('662892_20230821', device_timing_on_sync)
    >>> len(units[units['electrode_group_name'] == 'probeA'])
    237
    """
    spike_interface_data = utils.get_spikeinterface_data(
        session_or_spikeinterface_data_or_path
    )

    devices_timing = tuple(
        timing for timing in device_timing_on_sync if timing.device.name.endswith("-AP")
    )

    device_to_future: dict[str, concurrent.futures.Future] = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for device_timing in devices_timing:
            device_to_future[device_timing.device.name] = executor.submit(
                _device_helper,
                device_timing,
                spike_interface_data,
                include_waveform_arrays,
            )

    return pd.concat(
        device_to_future[device].result() for device in sorted(device_to_future.keys())
    )


def add_tissuecyte_annotations(
    units: pd.DataFrame, session: str | npc_session.SessionRecord
) -> pd.DataFrame:
    """Join units table with tissuecyte electrode locations table and drop redundant columns."""
    try:
        electrodes = utils.get_tissuecyte_electrodes_table(session)
    except FileNotFoundError as e:
        logger.warning(f"{e}: returning units without locations.")
        return units
    units = units.merge(
        electrodes,
        left_on=["electrode_group_name", "peak_channel"],
        right_on=["group_name", "channel"],
    )
    units.drop(columns=["channel"], inplace=True)

    return units


def add_global_unit_ids(
    units: pd.DataFrame,
    session: str | npc_session.SessionRecord,
    unit_id_column: str = "unit_id",
) -> pd.DataFrame:
    """Add session and probe letter"""
    units["unit_id"] = [
        f"{session}_{row.electrode_group_name.replace('probe', '')}-{row[unit_id_column]}"
        if session not in str(row[unit_id_column])  # in case we aready ran this fn
        else row[unit_id_column]
        for _, row in units.iterrows()
    ]
    if unit_id_column != "unit_id":
        units.drop(columns=[unit_id_column], inplace=True)
    return units


if __name__ == "__main__":
    import doctest

    import dotenv

    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
