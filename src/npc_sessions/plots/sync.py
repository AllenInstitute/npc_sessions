from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import rich

if TYPE_CHECKING:
    import npc_sessions

import npc_sessions.plots.plot_utils as plot_utils
import npc_sessions.utils as utils


def plot_barcode_times(session: "npc_sessions.DynamicRoutingSession"):
    devices = utils.get_ephys_timing_on_pxi(session.ephys_recording_dirs)
    for device in devices:
        (
            ephys_barcode_times,
            ephys_barcode_ids,
        ) = utils.extract_barcodes_from_times(
            on_times=device.ttl_sample_numbers[device.ttl_states > 0]
            / device.sampling_rate,
            off_times=device.ttl_sample_numbers[device.ttl_states < 0]
            / device.sampling_rate,
        )
        plt.plot(np.diff(ephys_barcode_times))


def plot_barcode_intervals(session: "npc_sessions.DynamicRoutingSession"):
    """
    Plot barcode intervals for sync and for each probe after sample rate
    correction
    """
    full_exp_recording_dirs = [
        utils.get_single_oebin_path(directory).parent
        for directory in session.ephys_record_node_dirs
    ]

    barcode_rising = session.sync_data.get_rising_edges(0, "seconds")
    barcode_falling = session.sync_data.get_falling_edges(0, "seconds")
    barcode_times, barcodes = utils.extract_barcodes_from_times(
        barcode_rising, barcode_falling
    )

    devices_pxi = utils.get_ephys_timing_on_pxi(full_exp_recording_dirs)
    devices_sync = tuple(
        utils.get_ephys_timing_on_sync(session.sync_path, session.ephys_recording_dirs)
    )
    device_barcode_dict = {}
    for device in devices_pxi:
        if "NI-DAQmx" in device.name or "LFP" in device.name:
            continue

        device_sync = [d for d in devices_sync if d.name == device.name][0]

        (
            ephys_barcode_times,
            ephys_barcode_ids,
        ) = utils.extract_barcodes_from_times(
            on_times=device.ttl_sample_numbers[device.ttl_states > 0]
            / device.sampling_rate,
            off_times=device.ttl_sample_numbers[device.ttl_states < 0]
            / device.sampling_rate,
        )
        raw = ephys_barcode_times
        corrected = ephys_barcode_times * (30000 / device_sync.sampling_rate)
        intervals = np.diff(corrected)
        max_deviation = np.max(np.abs(intervals - np.median(intervals)))

        device_barcode_dict[device.name] = {
            "barcode_times_raw": raw,
            "barcode_times_corrected": corrected,
            "max_deviation_from_median_interval": max_deviation,
        }

    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches([8, 4])
    sync_intervals = np.diff(barcode_times)
    sync_max_deviation_from_median_interval = np.max(
        np.abs(sync_intervals - np.median(sync_intervals))
    )
    sync_max_deviation_string = plot_utils.add_valence_to_string(
        f"Sync deviation: {sync_max_deviation_from_median_interval}",
        sync_max_deviation_from_median_interval,
        sync_max_deviation_from_median_interval < 0.001,
        sync_max_deviation_from_median_interval > 0.001,
    )
    rich.print(sync_max_deviation_string)

    ax[0].plot(sync_intervals)
    legend = []
    for device_name, device_data in device_barcode_dict.items():
        ax[1].plot(np.diff(device_data["barcode_times_raw"]))
        ax[2].plot(np.diff(device_data["barcode_times_corrected"]))
        legend.append(device_name.split("Probe")[1])
        max_deviation = device_data["max_deviation_from_median_interval"]
        max_deviation_string = plot_utils.add_valence_to_string(
            f"{device_name}: {max_deviation}",
            max_deviation,
            max_deviation < 0.001,
            max_deviation > 0.001,
        )

        rich.print(max_deviation_string)

    ax[2].plot(sync_intervals, "k")
    ax[2].legend(legend + ["sync"])
    ax[0].set_title("Sync Barcode Intervals")
    ax[1].set_title("Probe Barcode Intervals")
    ax[2].set_title("Probe Barcode Intervals Corrected")

    plt.tight_layout()
