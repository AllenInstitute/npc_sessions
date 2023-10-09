from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.figure
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    import pandas as pd
    import pynwb

    import npc_sessions

import npc_sessions.plots.plot_utils as plot_utils
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


def plot_all_unit_spike_histograms(
    session: npc_sessions.DynamicRoutingSession,
) -> tuple[matplotlib.figure.Figure, ...]:  # -> tuple:# -> tuple:
    session.units[:].query("default_qc")
    figs: list[matplotlib.figure.Figure] = []
    for obj in session.all_spike_histograms.children:
        fig, ax = plt.subplots()
        ax.plot(obj.timestamps, obj.data, linewidth=0.1, alpha=0.8, color="k")
        plot_utils.add_epoch_color_bars(
            ax, session.epochs[:], y=50, va="bottom", rotation=90
        )
        ax.set_title(obj.description, fontsize=8)
        fig.suptitle(session.session_id, fontsize=10)
        ax.set_xlabel(obj.timestamps_unit)
        ax.set_ylabel(obj.unit)
        ax.set_ylim([0, ax.get_ylim()[1]])
        ax.margins(0)
        ax.set_frame_on(False)
        fig.set_layout_engine("tight")
        fig.set_size_inches(5, 5)
        figs.append(fig)
    return tuple(figs)


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
        ax.set_title(k, fontsize=8)
        ax.margins(0)
        ax.axis("off")
        fig.set_size_inches([5, 5])
        fig.tight_layout()
        figs.append(fig)
    return tuple(figs)
