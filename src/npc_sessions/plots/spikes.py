from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    import pandas as pd
    import pynwb

    import npc_sessions

import npc_sessions.utils as utils


def plot_unit_quality_metrics_per_probe(session: npc_sessions.DynamicRoutingSession):
    units: pd.DataFrame = session.units[:]

    metrics = [
        "drift_ptp",
        "isi_violations_ratio",
        "amplitude",
        "amplitude_cutoff",
        "presence_ratio",
    ]
    probes = units["electrode_group_name"].unique()

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
            units_probe_metric = units[units["electrode_group_name"] == probe][metric]
            ax[probe_index].hist(units_probe_metric, bins=20)
            ax[probe_index].set_title(f"{probe}")
            ax[probe_index].set_xlabel(x_labels[metric])
            probe_index += 1

        fig.set_size_inches([10, 6])
    plt.tight_layout()


def plot_all_unit_spike_histograms(session: npc_sessions.DynamicRoutingSession):
    units: pd.DataFrame = session.units[:].query('default_qc')
    
    for probe,obj in session.all_spike_histograms.items():
        fig, ax = plt.subplots()
        
        for row,epoch in session.epochs[:].iterrows():
            name = next((k for k in epoch.tags if k in epoch_color_map), None)
            color = epoch_color_map[name] if name else 'black'
            ax.axvspan(epoch.start_time,epoch.stop_time,alpha=0.1,color=color)
            ax.text((epoch.stop_time+epoch.start_time)/2,0,epoch.name,ha='center',va='top',fontsize=8)
        ax.plot(obj.timestamps,obj.data)
        ax.set_title(f"{probe} Spike Histogram")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Spike Count per 1 second bin")

epoch_color_map = {
    'RFMapping': 'blue',
    'DynamicRouting1': 'green',
    'OptoTagging1': 'cyan',
    'Spontaneous': 'red',
    'SpontaneousRewards': 'magenta',
}

def plot_unit_spikes_channels(
    session: npc_sessions.DynamicRoutingSession,
    lower_channel: int = 0,
    upper_channel: int = 384,
):
    units: pd.DataFrame = session.units[:]

    probes = units["electrode_group_name"].unique()
    for probe in probes:
        fig, ax = plt.subplots()
        unit_spike_times = units[units["electrode_group_name"] == probe]
        unit_spike_times_channel = unit_spike_times[
            (unit_spike_times["peak_channel"] >= lower_channel)
            & (unit_spike_times["peak_channel"] <= upper_channel)
        ]["spike_times"].to_numpy()
        hist, bins = utils.bin_spike_times(unit_spike_times_channel, bin_interval=1)

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
    for k, v in session.analysis["drift_maps"].images.items():
        fig, ax = plt.subplots()
        ax.imshow(v)
        fig.suptitle(f"{session.session_id}")
        ax.set_title(k)
        ax.axis("off")
        fig.set_size_inches([8, 8])
        fig.tight_layout()
        figs.append(fig)
    return tuple(figs)
