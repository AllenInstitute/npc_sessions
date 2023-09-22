from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    import pandas as pd
    import pynwb.NWBFile

    import npc_sessions


def plot_unit_quality_metrics_per_probe(session: npc_sessions.DynamicRoutingSession):
    units: pd.DataFrame = session.units[:]

    metrics = [
        "drift_ptp",
        "isi_violations_ratio",
        "amplitude",
        "amplitude_cutoff",
        "presence_ratio",
    ]
    probes = units["device_name"].unique()

    x_labels = {
        "presence_ratio": "fraction of session",
        "isi_violations_ratio": "violation rate",
        "drift_ptp": "microns",
        "amplitude": "uV",
        "amplitude_cutoff": "frequency",
    }

    for metric in metrics:
        fig, ax = plt.subplots(1, len(probes))
        probe_index = 0
        fig.suptitle(f"{metric}")
        for probe in probes:
            units_probe_metric = units[units["device_name"] == probe][metric]
            ax[probe_index].hist(units_probe_metric, bins=20)
            ax[probe_index].set_title(f"{probe}")
            ax[probe_index].set_xlabel(x_labels[metric])
            probe_index += 1

        fig.set_size_inches([16, 6])
    plt.tight_layout()


def plot_all_unit_spike_histograms(session: npc_sessions.DynamicRoutingSession):
    units: pd.DataFrame = session.units[:]

    probes = units["device_name"].unique()

    for probe in probes:
        fig, ax = plt.subplots()
        unit_spike_times = units[units["device_name"] == probe][
            "spike_times"
        ].to_numpy()

        hist, bins = npc_sessions.bin_spike_times(unit_spike_times, bin_time=1)
        ax.plot(hist)
        ax.set_title(f"{probe} Spike Histogram")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Spike Count per 1 second bin")


def plot_unit_spikes_channels(
    session: npc_sessions.DynamicRoutingSession,
    lower_channel: int,
    upper_channel: int,
):
    units: pd.DataFrame = session.units[:]

    probes = units["device_name"].unique()
    for probe in probes:
        fig, ax = plt.subplots()
        unit_spike_times = units[units["device_name"] == probe]
        unit_spike_times_channel = unit_spike_times[
            (unit_spike_times["peak_channel"] >= lower_channel)
            & (unit_spike_times["peak_channel"] <= upper_channel)
        ]["spike_times"].to_numpy()
        hist, bins = npc_sessions.bin_spike_times(unit_spike_times_channel, bin_time=1)

        ax.plot(hist)
        ax.set_title(
            f"{probe} spike hist for channel range {lower_channel} to {upper_channel}"
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Spike Count per 1 second bin")


def plot_drift_maps(
    session: npc_sessions.DynamicRoutingSession | pynwb.NWBFile,
) -> tuple[plt.Figure, ...]:
    figs = []
    for _k, v in session.analysis["drift_maps"]:
        fig, ax = plt.subplots()
        ax.imshow(v)
        ax.set_title(f"{session.session_id}")
        ax.axis("off")
        fig.set_size_inches([8, 8])
        fig.tight_layout()
        figs.append(fig)
    return tuple(figs)
