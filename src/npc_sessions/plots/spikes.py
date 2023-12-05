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
    session: npc_sessions.DynamicRoutingSession | pynwb.NWBFile, index_or_id: int | str
) -> matplotlib.figure.Figure:
    """Waveform on peak channel"""
    fig = plt.figure()
    unit = (
        session.units[:].iloc[index_or_id]
        if isinstance(index_or_id, int)
        else session.units[:].query("unit_id == @index_or_id").iloc[0]
    )

    electrodes: list[int] = unit["electrodes"]
    peak_channel_idx = electrodes.index(unit["peak_electrode"])
    mean = unit["waveform_mean"][:, peak_channel_idx]
    sd = unit["waveform_sd"][:, peak_channel_idx]
    t = np.arange(mean.size) / session.units.waveform_rate * 1000  # convert to ms
    t -= max(t) / 2  # center around 0

    ax = fig.add_subplot(111)
    # ax.hlines(0, t[0], t[-1], color='grey', linestyle='--')
    m = ax.plot(t, mean, label=f"Unit {unit['unit_id']}")
    ax.fill_between(t, mean + sd, mean - sd, color=m[0].get_color(), alpha=0.25)
    ax.set_xlabel("milliseconds")
    ax.set_ylabel(session.units.waveform_unit)
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(min(-100, ax.get_ylim()[0]), max(50, ax.get_ylim()[1]))
    ax.set_xmargin(0)
    if session.units.waveform_unit == "microvolts":
        ax.set_aspect(1 / 25)
    ax.grid(True)

    return fig


def plot_unit_spatiotemporal_waveform(
    session: npc_sessions.DynamicRoutingSession | pynwb.NWBFile,
    index_or_id: int | str,
    **pcolormesh_kwargs,
) -> matplotlib.figure.Figure:
    """Waveforms across channels around peak channel - currently no interpolation"""

    unit = (
        session.units[:].iloc[index_or_id]
        if isinstance(index_or_id, int)
        else session.units[:].query("unit_id == @index_or_id").iloc[0]
    )

    # assemble df of channels whose data we'll plot

    # electrodes with waveforms for this unit:
    electrode_group = session.electrodes[:].loc[unit["electrodes"]]

    # get largest signal from each row of electrodes on probe
    electrode_group["amplitudes"] = np.max(unit["waveform_mean"], axis=0) - np.min(
        unit["waveform_mean"], axis=0
    )

    peak_electrode = session.electrodes[:].loc[unit["peak_electrode"]]
    # ^ this is incorrect until annotations have been updated
    peak_electrode = electrode_group.sort_values(by="amplitudes").iloc[-1]

    rows = []
    for _rel_y in electrode_group.rel_y.unique():
        rows.append(
            electrode_group.query(f"rel_y == {_rel_y}")
            .sort_values(by="amplitudes")
            .iloc[-1]
        )
    selected_electrodes = pd.DataFrame(rows)
    assert len(selected_electrodes) == len(electrode_group.rel_y.unique())

    electrode_indices: list[int] = unit["electrodes"]
    waveforms = unit["waveform_mean"][
        :, np.searchsorted(electrode_indices, selected_electrodes.index)
    ]

    t = (
        np.arange(waveforms.shape[0]) / session.units.waveform_rate * 1000
    )  # convert to ms
    t -= max(t) / 2  # center around 0
    absolute_y = sorted(selected_electrodes.rel_y)
    relative_y = absolute_y - peak_electrode.rel_y  # center around peak electrode

    fig = plt.figure()
    norm = matplotlib.colors.TwoSlopeNorm(
        vmin=-150,
        vcenter=0,
        vmax=150,
    )  # otherwise, if all waveforms are zeros the vmin/vmax args become invalid

    pcolormesh_kwargs.setdefault("cmap", "bwr")
    _ = plt.pcolormesh(t, relative_y, waveforms.T, norm=norm, **pcolormesh_kwargs)
    ax = fig.gca()
    ax.set_xmargin(0)
    ax.set_xlim(-1.25, 1.25)
    ax.set_xlabel("milliseconds")
    ax.set_ylabel("microns from peak channel")
    ax.set_yticks(relative_y)
    secax = ax.secondary_yaxis(
        "right",
        functions=(
            lambda y: y + peak_electrode.rel_y,
            lambda y: y - peak_electrode.rel_y,
        ),
    )
    secax.set_ylabel("microns from tip")
    secax.set_yticks(absolute_y)
    ax.set_aspect(1 / 50)
    ax.grid(True, axis="x", lw=0.5, color="grey", alpha=0.5)
    plt.colorbar(
        ax=ax,
        fraction=0.01,
        pad=0.2,
        label=session.units.waveform_unit,
        ticks=[norm.vmin, norm.vcenter, norm.vmax],
    )
    fig.suptitle(
        f"{unit['unit_id']}\n{unit.peak_channel=}\nunit.amplitude={electrode_group['amplitudes'].max():.0f} {session.units.waveform_unit}",
        fontsize=8,
    )
    return fig
