from __future__ import annotations

import functools
import io
import os

import npc_lims
import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl
import upath

import npc_sessions


@functools.cache
def get_units_electrodes(
    session: str, units_method="codeocean_kilosort", electrode_method="tissuecyte"
) -> pd.DataFrame:
    if units_method == "codeocean_kilosort":
        units_path = npc_lims.get_units_codeoean_kilosort_path_from_s3(session)
        units = pd.read_csv(
            units_path,
            storage_options={
                "key": os.getenv("AWS_ACCESS_KEY_ID"),
                "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
            },
        )
    if electrode_method == "tissuecyte":
        try:
            electrodes = npc_sessions.create_tissuecyte_electrodes_table(session)
            units = units.merge(
                electrodes,
                left_on=["device_name", "peak_channel"],
                right_on=["device_name", "channel"],
            )
            units.drop(columns=["channel"], inplace=True)
        except FileNotFoundError as e:
            print(str(e) + ". Returning units without electrodes")

    units.drop(columns=["electrodes"], inplace=True)
    return units


def bin_spike_times(
    spike_times: npt.NDArray[np.float64], bin_time: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    spike_times = np.concatenate(spike_times, axis=0)  # flatten array
    return np.histogram(spike_times, bins=np.arange(0, np.max(spike_times), bin_time))


@functools.cache
def get_unit_spike_times_dict(
    session: str, unit_ids: tuple[str, ...], method="codeocean_kilosort"
) -> dict[str, npt.NDArray[np.float64]]:
    # change function call depending on method
    spike_times_dict = {}
    if method == "codeocean_kilosort":
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
        )
        .select(
            pl.exclude("unit_name", "id"),
        )
        .filter(
            "default_qc",
        )
    )


@functools.cache
def get_mean_waveforms(session: str) -> npt.NDArray[np.float64]:
    mean_waveforms_path = npc_lims.get_mean_waveform_codeocean_kilosort_path_from_s3(
        session
    )
    with io.BytesIO(mean_waveforms_path.read_bytes()) as f:
        mean_waveforms = np.load(f, allow_pickle=True)

    return mean_waveforms


@functools.cache
def get_sd_waveforms(session: str) -> npt.NDArray[np.float64]:
    sd_waveforms_path = npc_lims.get_sd_waveform_codeocean_kilosort_path_from_s3(
        session
    )
    with io.BytesIO(sd_waveforms_path.read_bytes()) as f:
        sd_waveforms = np.load(f, allow_pickle=True)

    return sd_waveforms


def align_device_kilosort_spike_times(
    session: str,
    device_name: str,
    device_timing_on_sync: npc_sessions.EphysTimingInfoOnSync,
) -> npt.NDArray[np.float64]:
    sorting_cached_file = npc_lims.get_spike_sorting_device_path_from_s3(
        session, device_name
    )
    with io.BytesIO(sorting_cached_file.read_bytes()) as f:
        sorting_cached = np.load(f, allow_pickle=True)
        spike_times_unaligned = sorting_cached["spike_indexes_seg0"]

    # directly getting spike times from sorted output, don't need sample start it seems
    # spike_times_unaligned = spike_times_unaligned - device_timing_on_sync.device.start_sample
    return (
        spike_times_unaligned / device_timing_on_sync.sampling_rate
    ) + device_timing_on_sync.start_time


def get_aligned_device_kilosort_spike_times(
    session: str,
) -> dict[str, npt.NDArray[np.float64]]:
    """
    Returns the algined spike times for each device, prior to making the units table
    """
    device_spike_times = {}
    recording_dirs_experiment = npc_lims.get_recording_dirs_experiment_path_from_s3(
        session
    )
    sync_path = npc_lims.get_h5_sync_from_s3(session)
    settings_xml_path = npc_lims.get_settings_xml_path_from_s3(session)

    device_names = npc_sessions.get_settings_xml_info(settings_xml_path).probe_letters
    device_names_probe = tuple(f"Probe{name}" for name in device_names)

    devices = npc_sessions.get_ephys_timing_on_pxi(recording_dirs_experiment)
    devices_timing_on_sync = npc_sessions.get_ephys_timing_on_sync(
        sync_path, devices=devices
    )

    for device_name in device_names_probe:
        device_timing_on_sync = next(
            timing for timing in devices_timing_on_sync if device_name in timing.name
        )
        # TODO: Save to s3?
        device_spike_times[device_name] = align_device_kilosort_spike_times(
            session, device_name, device_timing_on_sync
        )

    return device_spike_times


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


def make_units_metrics_device_table(
    quality_metrics_path: upath.UPath,
    template_metrics_path: upath.UPath,
    units_locations_path: upath.UPath,
    sorting_precurated_path: upath.UPath,
    device_name: str,
    electrode_positions: tuple[dict[str, tuple[int, int]], ...],
) -> pd.DataFrame:
    df_quality_metrics = pd.read_csv(
        quality_metrics_path,
        storage_options={
            "key": os.getenv("AWS_ACCESS_KEY_ID"),
            "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
        },
    )
    df_quality_metrics.rename(columns={"Unnamed: 0": "ks_unit_id"}, inplace=True)

    df_template_metrics = pd.read_csv(
        template_metrics_path,
        storage_options={
            "key": os.getenv("AWS_ACCESS_KEY_ID"),
            "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
        },
    )
    df_template_metrics.rename(columns={"Unnamed: 0": "ks_unit_id"}, inplace=True)

    df_quality_template_metrics = df_quality_metrics.merge(
        df_template_metrics, on="ks_unit_id"
    )

    with io.BytesIO(units_locations_path.read_bytes()) as f:
        units_locations = np.load(f, allow_pickle=True)

    peak_channels = get_peak_channels(units_locations, electrode_positions)
    df_quality_template_metrics["peak_channel"] = peak_channels
    df_quality_metrics["device_name"] = [
        device_name for i in range(len(df_quality_metrics))
    ]

    return df_quality_metrics


def make_units_table(session: str) -> pd.DataFrame:
    recording_dirs_experiment = npc_lims.get_recording_dirs_experiment_path_from_s3(
        session
    )
    sync_path = npc_lims.get_h5_sync_from_s3(session)
    settings_xml_path = npc_lims.get_settings_xml_path_from_s3(session)
    settings_xml_info = npc_sessions.get_settings_xml_info(settings_xml_path)
    device_names = settings_xml_info.probe_letters
    electrode_positions = settings_xml_info.channel_pos_xy

    device_names_probe = tuple(f"Probe{name}" for name in device_names)

    devices = npc_sessions.get_ephys_timing_on_pxi(recording_dirs_experiment)
    npc_sessions.get_ephys_timing_on_sync(sync_path, devices=devices)

    quality_metrics_paths = sorted(npc_lims.get_quality_metrics_paths_from_s3(session))
    template_metrics_paths = sorted(
        npc_lims.get_template_metrics_paths_from_s3(session)
    )
    unit_locations_paths = sorted(npc_lims.get_unit_locations_paths_from_s3(session))
    sorting_precurated_paths = sorted(
        npc_lims.get_sorted_precurated_paths_from_s3(session)
    )

    for i in range(len(quality_metrics_paths)):
        quality_metrics_path = quality_metrics_paths[i]
        template_metrics_path = template_metrics_paths[i]
        units_locations_path = unit_locations_paths[i]
        sorting_precurated_path = sorting_precurated_paths[i]
        device_name = next(
            device_name_probe
            for device_name_probe in device_names_probe
            if device_name_probe in str(quality_metrics_path)
        )

        df_device_metrics = make_units_metrics_device_table(
            quality_metrics_path,
            template_metrics_path,
            units_locations_path,
            sorting_precurated_path,
            device_name,
            electrode_positions,
        )
        df_device_metrics["unit_id"] = [
            f"{session}_{ks_id}" for ks_id in df_device_metrics["ks_unit_id"]
        ]

        # TODO: add amplitudes, default qc, spike times for units, and clean up code, code below fails
        """
        sorting_precurated = si.load_extractor(sorting_precurated_path.read_text())
        we = si.load_waveforms(quality_metrics_path.parent.parent, sorting=sorting_precurated, with_recording=False)
        amplitudes = list(si.get_template_extremum_amplitude(we).values())
        """


if __name__ == "__main__":
    import doctest

    import dotenv

    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
