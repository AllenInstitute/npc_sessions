from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.colors
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
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
    session: npc_sessions.DynamicRoutingSession | pynwb.NWBFile,
    index_or_id: int | str
) -> matplotlib.figure.Figure:
    """Waveform on peak channel"""
    fig = plt.figure()
    unit = (
        session.units[:].iloc[index_or_id]
        if isinstance(index_or_id, int)
        else session.units[:].query("unit_id == @index_or_id").iloc[0]
    )

    mean = unit["waveform_mean"][:, unit["peak_channel"]]
    sd = unit["waveform_sd"][:, unit["peak_channel"]]
    t = np.arange(mean.size) / session.units.waveform_rate * 1000  # convert to ms
    t -= max(t) / 2  # center around 0

    ax = fig.add_subplot(111)
    # ax.hlines(0, t[0], t[-1], color='grey', linestyle='--')
    m = ax.plot(t, mean, label=f"Unit {unit['unit_id']}")
    ax.fill_between(t, mean + sd, mean - sd, color=m[0].get_color(), alpha=0.25)
    ax.set_xlabel("milliseconds")
    ax.set_ylabel(session.units.waveform_unit)
    ax.set_xmargin(0)
    # if units.waveform_unit == "microvolts":
    ax.set_aspect(1 / 25)
    ax.grid(True)
    fig.show()
    return fig


def plot_unit_spatiotemporal_waveform(
    session: npc_sessions.DynamicRoutingSession | pynwb.NWBFile,
    index_or_id: int | str,
    vertical_span_microns: int = 190,
) -> matplotlib.figure.Figure:
    """Waveforms across channels around peak channel - currently no interpolation"""

    unit = (
        session.units[:].iloc[index_or_id]
        if isinstance(index_or_id, int)
        else session.units[:].query("unit_id == @index_or_id").iloc[0]
    )

    peak_channel = unit["peak_channel"] # fmt: skip
    electrode_group_name = unit["electrode_group_name"] # fmt: skip
    electrode_group = (
        session.electrodes[:]
        .query("group_name == @electrode_group_name")
        .sort_values("channel")
        .reset_index()
    )

    # assemble df of channels whose data we'll plot
    # TODO replace with unit.electrodes (indices in electrodes table) when populated
    peak_electrode = electrode_group.query("channel == @peak_channel")
    peak_electrode_rel_x, peak_electrode_rel_y = (
        peak_electrode.iloc[0].rel_x,
        peak_electrode.iloc[0].rel_y,
    )
    min_rel_y = max(
        peak_electrode_rel_y - vertical_span_microns / 2, electrode_group.rel_y.min()
    ) # fmt: skip
    max_rel_y = min(
        peak_electrode_rel_y + vertical_span_microns / 2, electrode_group.rel_y.max()
    ) # fmt: skip
    selected_electrodes = electrode_group.query(
        "rel_y >= @min_rel_y and rel_y <= @max_rel_y"
    )
    
    # find index of "column" of electrodes the peak electrode is in (but don't assume
    # a column means shared x position: they're staggered in 1.0 probes)
    shared_peak_y = electrode_group.query("rel_y == @peak_electrode_rel_y").sort_values('rel_x').reset_index()
    peak_x_index = shared_peak_y.query('channel == @peak_channel').index[0]
    
    rows = []
    for rel_y in selected_electrodes.rel_y.unique(): # fmt: skip
        # get all channels at this y position
        y = selected_electrodes.query("rel_y == @rel_y").sort_values('rel_x').reset_index()
        rows.append(y.iloc[peak_x_index])
    column_electrodes = pd.DataFrame(rows)
    assert len(column_electrodes) == len(selected_electrodes.rel_y.unique())
    
    waveforms = unit["waveform_mean"][:, column_electrodes['id']]
    #! note waveforms.shape[1] != len(electrodes), some waveforms missing
    # TODO presumably surface channels missing, but need to confirm

    t = np.arange(waveforms.shape[0]) / session.units.waveform_rate * 1000  # convert to ms
    t -= max(t) / 2  # center around 0
    y = sorted(column_electrodes.rel_y) 
    y -= peak_electrode_rel_y # center around peak electrode

    fig = plt.figure()
    norm = matplotlib.colors.TwoSlopeNorm(
        vmin=waveforms.min() or -1.0, vcenter=0, vmax=waveforms.max() or 1.0,
    ) # otherwise, if all waveforms are zeros the vmin/vmax args become invalid
    _ = plt.pcolormesh(t, y, waveforms.T, norm=norm, cmap="bwr")
    ax = fig.gca()
    ax.set_xmargin(0)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("milliseconds")
    ax.set_ylabel("Gmicrons")
    ax.set_yticks(y)
    ax.set_aspect(1 / 100)
    ax.vlines(0, y[0], y[-1], color="grey", linestyle="--")
    cbar = plt.colorbar()
    cbar.set_label(session.units.waveform_unit)

    return fig
