from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.axes
import matplotlib.colors
import matplotlib.figure
import matplotlib.pyplot as plt
import npc_ephys
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import pynwb

    import npc_sessions

import npc_sessions.plots.plot_utils as plot_utils
import npc_sessions.utils as utils

matplotlib.rcParams.update({"font.size": 8})


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
        fig, _ = plt.subplots(1, len(probes))
        probe_index = 0
        fig.suptitle(f"{metric}")
        for probe in probes:
            units_probe_metric = units[units["electrode_group_name"] == probe][metric]
            fig.axes[probe_index].hist(units_probe_metric, bins=20, density=True)
            fig.axes[probe_index].set_title(f"{probe}")
            fig.axes[probe_index].set_xlabel(x_labels[metric])
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
        hist, bins = npc_ephys.bin_spike_times(unit_spike_times_channel, bin_interval=1)

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


def plot_ephys_noise(
    timeseries: pynwb.TimeSeries,
    interval: utils.Interval | None = None,
    median_subtraction: bool = True,
    ax: matplotlib.axes.Axes | None = None,
    **plot_kwargs,
) -> matplotlib.figure.Figure:
    timestamps = timeseries.get_timestamps()
    if interval is None:
        interval = ((t := np.ceil(timestamps)[0]), t + 1)
    t0, t1 = utils.parse_intervals(interval)[0]
    s0, s1 = np.searchsorted(timestamps, (t0, t1))
    if s0 == s1:
        raise ValueError(
            f"{interval=} is out of bounds ({timestamps[0]=}, {timestamps[-1]=})"
        )
    samples = np.arange(s0, s1)
    data = timeseries.data[samples, :] * timeseries.conversion * 1000  # microvolts

    def std(data):
        std = np.nanstd(data, axis=0)
        return std

    if ax is None:
        ax = plt.subplot()

    plot_kwargs.setdefault("lw", 0.5)
    plot_kwargs.setdefault("color", "k")
    ax.plot(
        std(data),
        np.arange(data.shape[1]),
        **plot_kwargs,
    )
    if median_subtraction:
        offset_corrected_data = data - np.nanmedian(data, axis=0)
        median_subtracted_data = (
            offset_corrected_data.T - np.nanmedian(offset_corrected_data, axis=1)
        ).T
        ax.plot(
            std(median_subtracted_data),
            np.arange(median_subtracted_data.shape[1]),
            **plot_kwargs | {"color": "r", "alpha": 0.5},
        )
    ax.set_ymargin(0)
    ax.set_xlabel("SD (microvolts)")
    ax.set_ylabel("channel number")
    ax.set_title(f"noise on {timeseries.electrodes.description}")
    fig = ax.get_figure()
    assert fig is not None
    return fig


def plot_ephys_image(
    timeseries: pynwb.TimeSeries,
    interval: utils.Interval | None = None,
    median_subtraction: bool = True,
    ax: matplotlib.axes.Axes | None = None,
    **imshow_kwargs,
) -> matplotlib.figure.Figure:
    timestamps = timeseries.get_timestamps()
    if interval is None:
        interval = ((t := np.ceil(timestamps)[0]), t + 1)
    t0, t1 = utils.parse_intervals(interval)[0]
    s0, s1 = np.searchsorted(timestamps, (t0, t1))
    if s0 == s1:
        raise ValueError(
            f"{interval=} is out of bounds ({timestamps[0]=}, {timestamps[-1]=})"
        )
    samples = np.arange(s0, s1)
    data = timeseries.data[samples, :] * timeseries.conversion
    if median_subtraction:
        offset_corrected_data = data - np.nanmedian(data, axis=0)
        data = (offset_corrected_data.T - np.nanmedian(offset_corrected_data, axis=1)).T

    if ax is None:
        ax = plt.subplot()

    imshow_kwargs.setdefault("vmin", -(vrange := np.nanstd(data) * 3))
    imshow_kwargs.setdefault("vmax", vrange)
    imshow_kwargs.setdefault("cmap", "bwr")
    imshow_kwargs.setdefault("interpolation", "none")

    ax.imshow(
        data.T,
        aspect=5 / data.shape[1],  # assumes`extent` provided with seconds
        extent=(t0, t1, data.shape[1], 0),
        **imshow_kwargs,
    )
    ax.invert_yaxis()
    ax.set_ylabel("channel number")
    ax.set_xlabel("seconds")
    fig = ax.get_figure()
    assert fig is not None
    return fig


def plot_session_ephys_noise(
    session: npc_sessions.DynamicRoutingSession,
    lfp: bool = False,
    interval: utils.Interval = None,
    median_subtraction: bool = True,
    **plot_kwargs,
) -> matplotlib.figure.Figure:
    if lfp:
        container = session._raw_lfp
    else:
        container = session._raw_ap
    fig, _ = plt.subplots(1, len(container.electrical_series), sharex=True, sharey=True)
    for idx, (label, timeseries) in enumerate(container.electrical_series.items()):
        ax = fig.axes[idx]
        plot_ephys_noise(
            timeseries,
            ax=ax,
            interval=interval,
            median_subtraction=median_subtraction,
            **plot_kwargs,
        )
        ax.set_title(label, fontsize=8)
        if idx > 0:
            ax.yaxis.set_visible(False)

        if idx != round(len(fig.axes) / 2):
            ax.xaxis.set_visible(False)

    fig.suptitle(
        f'noise on channels with {"LFP" if lfp else "AP"} data | {session.session_id}',
        fontsize=10,
    )
    return fig


def plot_session_ephys_images(
    session: npc_sessions.DynamicRoutingSession,
    lfp: bool = False,
    interval: utils.Interval = None,
    median_subtraction: bool = True,
    **imshow_kwargs,
) -> matplotlib.figure.Figure:
    if lfp:
        container = session._raw_lfp
    else:
        container = session._raw_ap

    fig, _ = plt.subplots(1, len(container.electrical_series), sharex=True, sharey=True)
    for idx, (label, timeseries) in enumerate(container.electrical_series.items()):
        ax = fig.axes[idx]
        plot_ephys_image(
            timeseries,
            ax=ax,
            interval=interval,
            median_subtraction=median_subtraction,
            **imshow_kwargs,
        )
        ax.set_title(label, fontsize=8)
        if idx > 0:
            ax.yaxis.set_visible(False)

        if idx != round(len(fig.axes) / 2):
            ax.xaxis.set_visible(False)
    fig.suptitle(
        f'noise on channels with {"LFP" if lfp else "AP"} data | {median_subtraction=} | {session.session_id}',
        fontsize=10,
    )
    return fig


def plot_raw_ap_vs_surface(
    session: npc_sessions.DynamicRoutingSession | pynwb.NWBFile,
) -> tuple[plt.Figure, ...]:
    time_window = 0.5

    figs = []
    for probe in session._raw_ap.electrical_series.keys():
        n_samples = int(time_window * session._raw_ap[probe].rate)
        offset_corrected = session._raw_ap[probe].data[-n_samples:, :] - np.median(
            session._raw_ap[probe].data[-n_samples:, :], axis=0
        )
        car = (offset_corrected.T - np.median(offset_corrected, axis=1)).T

        if (
            session.is_surface_channels
            and probe in session.surface_recording._raw_ap.fields["electrical_series"]
        ):
            n_samples = int(time_window * session.surface_recording._raw_ap[probe].rate)
            offset_corrected_surface = session.surface_recording._raw_ap[probe].data[
                -n_samples:, :
            ] - np.median(
                session.surface_recording._raw_ap[probe].data[-n_samples:, :], axis=0
            )
            car_surface = (
                offset_corrected_surface.T - np.median(offset_corrected_surface, axis=1)
            ).T
            surface_channel_recording = True
        else:
            surface_channel_recording = False

        range = np.nanstd(car.flatten()) * 3

        fig, ax = plt.subplots(1, 2, figsize=(15, 8))

        ax[0].imshow(
            car.T,
            aspect="auto",
            interpolation="none",
            cmap="bwr",
            vmin=-range,
            vmax=range,
        )
        ax[0].invert_yaxis()
        ax[0].set_title(
            "deep channels (last " + str(time_window) + " sec of recording)"
        )
        ax[0].set_ylabel("channel number")
        ax[0].set_xlabel("samples")

        if surface_channel_recording:
            ax[1].imshow(
                car_surface.T,
                aspect="auto",
                interpolation="none",
                cmap="bwr",
                vmin=-range,
                vmax=range,
            )

        ax[1].invert_yaxis()
        ax[1].set_title(
            "surface channels (first " + str(time_window) + " sec of recording)"
        )
        ax[1].set_xlabel("samples")

        fig.suptitle(session.session_id + " " + probe)

        figs.append(fig)

    return tuple(figs)
