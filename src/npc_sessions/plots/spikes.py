from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.colors
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    import pynwb

    import npc_sessions

import npc_sessions.plots.plot_utils as plot_utils
import npc_sessions.utils as utils


def plot_unit_quality_metrics_per_probe(session: npc_sessions.DynamicRoutingSession):
    units: pd.DataFrame = utils.good_units(session.units)

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
            ax[probe_index].hist(units_probe_metric, bins=20, density=True)
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
    units: pd.DataFrame = utils.good_units(session.units)

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


def plot_unit_waveform(
    units: pynwb.Units, index_or_id: int | str
) -> matplotlib.figure.Figure:
    """Waveform on peak channel"""
    fig = plt.figure()
    unit = (
        units[:].iloc[index_or_id]
        if isinstance(index_or_id, int)
        else units[:].query("unit_id == @index_or_id").iloc[0]
    )

    mean = unit["waveform_mean"][:, unit["peak_channel"]]
    sd = unit["waveform_sd"][:, unit["peak_channel"]]
    t = np.arange(mean.size) / units.waveform_rate * 1000  # convert to ms
    t -= max(t) / 2  # center around 0

    ax = fig.add_subplot(111)
    # ax.hlines(0, t[0], t[-1], color='grey', linestyle='--')
    m = ax.plot(t, mean, label=f"Unit {unit['unit_id']}")
    ax.fill_between(t, mean + sd, mean - sd, color=m[0].get_color(), alpha=0.25)
    ax.set_xlabel("milliseconds")
    ax.set_ylabel(units.waveform_unit)
    ax.set_xmargin(0)
    # if units.waveform_unit == "microvolts":
    ax.set_aspect(1 / 25)
    ax.grid(True)
    fig.show()
    return fig


def plot_unit_spatiotemporal_waveform(
    units: pynwb.Units,
    electrodes: pynwb.core.DynamicTable,
    index_or_id: int | str,
    vertical_span_microns: int = 300,
) -> matplotlib.figure.Figure:
    """Waveforms across channels around peak channel - currently no interpolation"""

    unit = (
        units[:].iloc[index_or_id]
        if isinstance(index_or_id, int)
        else units[:].query("unit_id == @index_or_id").iloc[0]
    )

    unit["peak_channel"]
    unit["electrode_group_name"]
    electrode_group = (
        electrodes[:]
        .query("group_name == @electrode_group_name")
        .sort_values("channel")
        .reset_index()
    )

    peak_electrode = electrode_group.query("channel == @peak_channel")
    peak_electrode_rel_x, peak_electrode_rel_y = (
        peak_electrode.iloc[0].rel_x,
        peak_electrode.iloc[0].rel_y,
    )
    max(
        peak_electrode_rel_y - vertical_span_microns / 2, electrode_group.rel_y.min()
    )
    min(
        peak_electrode_rel_y + vertical_span_microns / 2, electrode_group.rel_y.max()
    )
    selected_electrodes = electrode_group.query(
        "rel_y >= @min_rel_y and rel_y <= @max_rel_y"
    ).query("rel_x == @peak_electrode_rel_x")

    waveforms = unit["waveform_mean"][:, selected_electrodes.index]
    #! note waveforms.shape[1] != len(selected_electrodes) - some waveforms missing
    # TODO presumably surface channels missing, but need to confirm

    t = np.arange(waveforms.shape[0]) / units.waveform_rate * 1000  # convert to ms
    t -= max(t) / 2  # center around 0
    y = sorted(selected_electrodes.rel_y)

    fig = plt.figure()
    norm = matplotlib.colors.TwoSlopeNorm(
        vmin=waveforms.min(), vcenter=0, vmax=waveforms.max()
    )
    _ = plt.pcolormesh(t, y, waveforms.T, norm=norm, cmap="bwr")
    ax = fig.gca()
    ax.set_xmargin(0)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("milliseconds")
    ax.set_ylabel("distance from tip (microns)")
    ax.set_yticks(y)
    # plt.set_cmap('bwr')
    ax.set_aspect(1 / 100)
    cbar = plt.colorbar()
    cbar.set_label(units.waveform_unit)

    return fig
