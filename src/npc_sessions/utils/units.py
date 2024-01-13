from __future__ import annotations

import concurrent.futures
import logging
from collections.abc import Iterable
from typing import NamedTuple

import npc_session
import numpy as np
import numpy.typing as npt
import pandas as pd
import pynwb
import tqdm

import npc_sessions.utils as utils

logger = logging.getLogger(__name__)


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


def get_aligned_spike_times(
    spike_times: npt.NDArray[np.floating],
    device_timing_on_sync: utils.EphysTimingInfo,
) -> npt.NDArray[np.float64]:
    return (
        spike_times / device_timing_on_sync.sampling_rate
    ) + device_timing_on_sync.start_time


class AmplitudesWaveformsChannels(NamedTuple):
    """Data class, each entry a sequence with len == N units"""

    amplitudes: tuple[np.floating, ...]
    templates_mean: tuple[npt.NDArray[np.floating], ...]
    templates_sd: tuple[npt.NDArray[np.floating], ...]
    peak_channels: tuple[np.intp, ...]
    channels: tuple[tuple[np.intp, ...], ...]


def get_amplitudes_waveforms_channels_ks25(
    spike_interface_data: utils.SpikeInterfaceKS25Data,
    electrode_group_name: str,
    sampling_rate: float,
) -> AmplitudesWaveformsChannels:
    unit_amplitudes: list[np.floating] = []
    templates_mean: list[npt.NDArray[np.floating]] = []
    templates_sd: list[npt.NDArray[np.floating]] = []
    peak_channels: list[np.intp] = []
    channels: list[tuple[np.intp, ...]] = []

    # nbefore sets the sample index (in each waveform timeseries) at which to
    # extract values -> the timeseries with the highest value becomes the peak channel
    """
    @property
    def nbefore(self) -> int:
        nbefore = int(self._params["ms_before"] * self.sampling_frequency / 1000.0)
        return nbefore
    """
    sparse_channel_indices = spike_interface_data.sparse_channel_indices(
        electrode_group_name
    )
    _templates_mean = spike_interface_data.templates_average(electrode_group_name)
    _templates_sd = spike_interface_data.templates_std(electrode_group_name)

    # https://github.com/SpikeInterface/spikeinterface/blob/777a07d3a538394d52a18a05662831a403ee35f9/src/spikeinterface/core/template_tools.py#L8
    nbefore = int(
        spike_interface_data.postprocessed_params_dict(electrode_group_name)[
            "ms_before"
        ]
        * sampling_rate
        / 1000.0
    )  # from spike interface, ms_before = 3,
    for unit_index in range(_templates_mean.shape[0]):
        # TODO replace this section when annotations are updated
        # - ---------------------------------------------------------------- #
        _mean = _templates_mean[unit_index, :, :]
        values = -_mean[nbefore, :]
        peak_channel = np.argmax(values)
        unit_amplitudes.append(values[peak_channel])
        # emailed Josh to see how he was getting peak channel - using waveforms, peak channel might be part of metrics in future
        _sd = _templates_sd[unit_index, :, :]
        # - ---------------------------------------------------------------- #
        idx = np.where(_mean.any(axis=0))[0]
        very_sparse_channel_indices = np.array(sparse_channel_indices)[idx]
        templates_mean.append(_mean[:, idx])
        templates_sd.append(_sd[:, idx])
        peak_channels.append(peak_channel)
        logger.debug(f"very_sparse_channel_indices: {very_sparse_channel_indices}")
        channels.append(tuple(idx))  # TODO switch to very_sparse_channel_indices

    return AmplitudesWaveformsChannels(
        amplitudes=tuple(unit_amplitudes),
        templates_mean=tuple(templates_mean),
        templates_sd=tuple(templates_sd),
        peak_channels=tuple(peak_channels),
        channels=tuple(channels),
    )


def get_waveform_sd_ks25(
    templates_std: npt.NDArray[np.floating],
) -> list[npt.NDArray[np.floating]]:
    unit_templates_std: list[npt.NDArray[np.floating]] = []
    for unit_index in range(templates_std.shape[0]):
        template = templates_std[unit_index, :, :]
        unit_templates_std.append(template)

    return unit_templates_std


def get_units_x_spike_times(
    spike_times: npt.NDArray[np.float64],
    unit_indexes: npt.NDArray[np.int64],
) -> tuple[npt.NDArray[np.float64], ...]:
    """
    Note: index in tuple != unit_index (may be gaps in unique(unit_indexes)))
    >>> spike_times = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    >>> unit_indexes = np.array([3, 0, 1, 1, 0])    # {0, 1, 3} - must handle gaps
    >>> get_units_x_spike_times(spike_times, unit_indexes)
    (array([0.2, 0.5]), array([0.3, 0.4]), array([0.1]))
    """
    if len(spike_times) != len(unit_indexes):
        raise ValueError(
            "spike_times and unit_indexes must be same length (values should correspond)"
        )
    units = sorted(np.unique(unit_indexes))  # may have gaps
    units_x_spike_times = tuple(spike_times[unit_indexes == unit] for unit in units)
    assert (
        spike_times[0]
        == units_x_spike_times[np.argwhere(units == unit_indexes[0]).item()][0]
    ), "Interpretation of unit_indexes/spike_times is faulty"
    return units_x_spike_times


def _device_helper(
    device_timing_on_sync: utils.EphysTimingInfo,
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

    awc = get_amplitudes_waveforms_channels_ks25(
        spike_interface_data=spike_interface_data,
        electrode_group_name=electrode_group_name,
        sampling_rate=device_timing_on_sync.sampling_rate,
    )

    df_device_metrics["peak_channel"] = awc.peak_channels
    cluster_id = df_device_metrics.index.to_list()
    assert np.array_equal(
        cluster_id, spike_interface_data.original_cluster_id(electrode_group_name)
    ), "cluster-ids from npy file do not match index column in metrics.csv"
    df_device_metrics["cluster_id"] = df_device_metrics.index.to_list()
    df_device_metrics["default_qc"] = spike_interface_data.default_qc(
        electrode_group_name
    )
    df_device_metrics["amplitude"] = awc.amplitudes
    if include_waveform_arrays:
        df_device_metrics["waveform_mean"] = awc.templates_mean
        df_device_metrics["waveform_sd"] = awc.templates_sd
        df_device_metrics["channels"] = awc.channels

    spike_times_aligned = get_aligned_spike_times(
        spike_interface_data.spike_indexes(electrode_group_name),
        device_timing_on_sync,
    )
    unit_indexes = spike_interface_data.unit_indexes(electrode_group_name)
    units_x_spike_times = get_units_x_spike_times(
        spike_times=spike_times_aligned,
        unit_indexes=unit_indexes,
    )
    assert len(units_x_spike_times) == len(
        df_device_metrics
    ), "Mismatch number of units in spike_times and metrics.csv"
    assert np.array_equal(
        df_device_metrics["num_spikes"].array,
        [len(unit) for unit in units_x_spike_times],
    ), "Mismatch between rows in spike_times and metrics.csv"
    df_device_metrics["spike_times"] = units_x_spike_times

    return df_device_metrics


def make_units_table_from_spike_interface_ks25(
    session_or_spikeinterface_data_or_path: str
    | npc_session.SessionRecord
    | utils.PathLike
    | utils.SpikeInterfaceKS25Data,
    device_timing_on_sync: Iterable[utils.EphysTimingInfo],
    include_waveform_arrays: bool = False,
) -> pd.DataFrame:
    """
    >>> import npc_lims
    >>> device_timing_on_sync = utils.get_ephys_timing_on_sync(npc_lims.get_h5_sync_from_s3('662892_20230821'), npc_lims.get_recording_dirs_experiment_path_from_s3('662892_20230821'), only_devices_including='ProbeA')
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

        for future in tqdm.tqdm(
            iterable=concurrent.futures.as_completed(device_to_future.values()),
            desc="fetching units",
            unit="device",
            total=len(device_to_future),
            ncols=80,
            ascii=False,
        ):
            device = next(k for k, v in device_to_future.items() if v == future)
            session = spike_interface_data.session or ""
            try:
                _ = future.result()
            except utils.ProbeNotFoundError:
                logger.warning(
                    f"Path to {session}{' ' if session else ''}{device} sorted data not found: likely skipped by SpikeInterface"
                )
                del device_to_future[device]
            except AssertionError as e:
                logger.error(f"{session}{' ' if session else ''}{device}")
                raise e from None
            except Exception as e:
                logger.error(f"{session}{' ' if session else ''}{device}")
                raise RuntimeError(
                    f"Error fetching units for {session} - see original exception above/below"
                ) from e

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
        electrodes[["group_name", "channel", "x", "y", "z", "structure", "location"]],
        left_on=["electrode_group_name", "peak_channel"],
        right_on=["group_name", "channel"],
    )
    units.drop(columns=["channel"], inplace=True)
    units.rename(
        columns={
            "x": "ccf_ap",
            "y": "ccf_dv",
            "z": "ccf_ml",
        },
        inplace=True,
    )
    return units


def add_global_unit_ids(
    units: pd.DataFrame,
    session: str | npc_session.SessionRecord,
    unit_id_column: str = "unit_id",
) -> pd.DataFrame:
    """Add session and probe letter"""
    units[unit_id_column] = [
        f"{session}_{row.electrode_group_name.replace('probe', '')}-{row['cluster_id']}"
        for _, row in units.iterrows()
    ]
    return units


def good_units(
    units: pynwb.misc.Units | pd.DataFrame, qc_column: str = "default_qc"
) -> pd.DataFrame:
    units = units[:]
    if units[qc_column].dtype != bool:
        raise NotImplementedError(
            f"currently qc_column {qc_column} must be boolean - either don't use this function or add a fix"
        )
    return units[units[qc_column]]


if __name__ == "__main__":
    import doctest

    import dotenv

    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
