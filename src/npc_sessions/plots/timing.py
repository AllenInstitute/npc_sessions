import random
from typing import TYPE_CHECKING

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import rich

if TYPE_CHECKING:
    import npc_sessions

import npc_sessions.utils as utils


def plot_bad_lick_times(
    session: "npc_sessions.DynamicRoutingSession",
) -> tuple[plt.Figure, ...]:
    """A loop making eventplots vsyncs for trials with:
    - licks in script but no lick within response window
    - licks not in script, but lick within response window
    """
    assert session.is_sync and session._trials._sync is not None

    trials_with_lick_outside_response_window = (
        session.trials[:]
        .query("is_response")
        .query(
            "response_window_start_time > response_time or response_window_stop_time < response_time"
        )
    ).index.tolist()

    trials_with_lick_inside_response_window_but_not_recorded = (
        session.trials[:]
        .query("not is_response")
        .query(
            "response_window_start_time <= response_time <= response_window_stop_time"
        )
    ).index

    figs = []
    for idx in (
        *trials_with_lick_outside_response_window,
        *trials_with_lick_inside_response_window_but_not_recorded,
    ):
        figs.append(plot_trial_lick_timing(session, idx))
    return tuple(figs)


def plot_assorted_lick_times(
    session: "npc_sessions.DynamicRoutingSession",
) -> tuple[plt.Figure, ...]:
    sync_time = session._trials.response_time
    script_time = utils.safe_index(
        session._trials._flip_times, session._trials._sam.trialResponseFrame
    )
    intervals = np.abs(sync_time - script_time)
    figs = []
    for idx, trial_idx in enumerate(
        [
            np.nanargmax(intervals),
            np.nanargmin(intervals),
            random.choice(session.trials[:].query("is_response").index),
        ]
    ):
        fig = session.plot_trial_lick_timing(trial_idx)  # type: ignore[attr-defined]
        fig.axes[0].set_title(
            fig.axes[0].get_title()
            + " - "
            + ["longest", "shortest", "random"][idx]
            + " lick-lickFrame interval"
        )
        figs.append(fig)
    return tuple(figs)


def plot_trial_lick_timing(
    session: "npc_sessions.DynamicRoutingSession", trial_idx: int
) -> matplotlib.figure.Figure:
    if not session.is_sync or session._trials._sync is None:
        raise ValueError("session must have sync data")
    start = session._trials.response_window_start_time[trial_idx]
    stop = session._trials.response_window_stop_time[trial_idx]
    vsyncs = session._trials._sync.get_falling_edges("vsync_stim", units="seconds")
    lick_sensor_rising = session._trials._sync.get_rising_edges(
        "lick_sensor", units="seconds"
    )
    lick_sensor_falling = session._trials._sync.get_falling_edges(
        "lick_sensor", units="seconds"
    )

    fig, ax = plt.subplots()
    padding = 0.3
    marker_config = {"linestyles": "-", "linelengths": 0.2}
    line_config = {"linestyles": "-", "linelengths": 1}
    ax.eventplot(
        vsyncs[(vsyncs >= start - padding) & (vsyncs <= stop + padding)],
        **line_config,
        label="vsyncs",
        alpha=0.5,
        color="orange",
    )
    ax.eventplot(
        vsyncs[(vsyncs >= start) & (vsyncs <= stop)],
        **line_config,
        label="vsyncs within response window",
        color="orange",
    )
    ax.eventplot(
        lick_sensor_rising[
            (lick_sensor_rising >= start - padding)
            & (lick_sensor_rising <= stop + padding)
        ],
        **marker_config,
        label="lick sensor rising",
        color="k",
        lineoffsets=1,
    )
    ax.eventplot(
        lick_sensor_falling[
            (lick_sensor_falling >= start - padding)
            & (lick_sensor_falling <= stop + padding)
        ],
        **marker_config,
        label="lick sensor falling",
        color="grey",
        lineoffsets=1,
    )
    ax.eventplot(
        [
            utils.safe_index(
                session._trials._input_data_times,
                session._trials._sam.trialResponseFrame[trial_idx],
            )
        ],
        **marker_config,
        label="responseFrame in TaskControl",
        color="lime",
        lineoffsets=1.6,
    )

    ax.eventplot(
        [
            (
                lick_frames := utils.safe_index(
                    session._trials._input_data_times, session._trials._sam.lickFrames
                )
            )[(lick_frames >= start) & (lick_frames <= stop)]
        ],
        **(marker_config | {"linestyles": ":"}),
        label="lickFrames in TaskControl",
        color="g",
        lineoffsets=1.6,
    )

    ax.eventplot(
        [
            utils.safe_index(
                session._trials._flip_times,
                session._trials._sam.stimStartFrame[trial_idx],
            )
        ],
        **marker_config,
        label="stim start frame in TaskControl",
        color="r",
        lineoffsets=1.6,
    )

    ax.legend(fontsize=8, loc="upper center", fancybox=True, ncol=4)
    ax.set_yticks([])
    ax.set_xlabel("time (s)")
    ax.title.set_text(
        f"{session.id} - sync & script timing - trial {trial_idx} - {session._trials.stim_name[trial_idx]}"
    )
    fig.set_size_inches(12, 4)
    return fig


def plot_lick_times_on_sync_and_script(
    session: "npc_sessions.DynamicRoutingSession",
) -> tuple[plt.Figure, plt.Figure]:
    """-stem plot of lick times on sync relative to lick times in TaskControl
    - histogram showing distribution of same intervals
    """
    sync_time = session._trials.response_time
    script_time = utils.safe_index(
        session._trials._flip_times, session._trials._sam.trialResponseFrame
    )

    intervals = sync_time - script_time
    fig1, ax = plt.subplots()
    markerline, stemline, baseline = plt.stem(
        sync_time,
        intervals,
        bottom=0,
        orientation="horizontal",
    )
    plt.setp(stemline, linewidth=0.5, alpha=0.3)
    plt.setp(markerline, markersize=0.5, alpha=0.8)
    plt.setp(baseline, visible=False)
    ax.set_xlabel("lick time on sync relative to lick time in TaskControl (s)")
    ax.set_ylabel("experiment time (s)")
    ax.set_title(f"{np.nanmean(intervals) = :.3f}s, {np.nanstd(intervals) = :.3f}")

    fig2, ax = plt.subplots()
    ax.hist(intervals, bins=50)
    ax.set_xlabel("lick time on sync relative to lick time in TaskControl (s)")
    ax.set_ylabel("count")
    ax.set_title(f"{np.nanmean(intervals) = :.3f}s, {np.nanstd(intervals) = :.3f}")
    return fig1, fig2


def plot_diode_flip_intervals(
    session: "npc_sessions.DynamicRoutingSession",
) -> matplotlib.figure.Figure:
    fig = session.sync_data.plot_diode_measured_sync_square_flips()
    names = tuple(
        k for k, v in session.stim_frame_times.items() if not isinstance(v, Exception)
    )
    for idx, ax in enumerate(fig.axes):
        if len(names) == len(fig.axes):
            ax.set_title(names[idx].split("_")[0])
    fig.set_size_inches(12, 6)
    return fig


def plot_vsyncs_and_diode_flips_at_ends_of_each_stim(
    session: "npc_sessions.DynamicRoutingSession",
) -> matplotlib.figure.Figure:
    rich.print("[bold] Fraction long frames [/bold]")
    for stim_name, stim_times in session.stim_frame_times.items():
        if isinstance(stim_times, Exception):
            continue
        intervals = np.diff(stim_times)
        fraction_long = np.sum(intervals > 0.02) / len(intervals)
        longest_interval = max(intervals)
        start_tag, end_tag = (
            ("[bold green]", "[/bold green]")
            if fraction_long < 0.01 and longest_interval < 0.5
            else ("[bold magenta]", "[/bold magenta]")
        )
        rich.print(
            start_tag
            + stim_name
            + ": "
            + str(fraction_long)
            + " \t\t longest interval:"
            + str(longest_interval)
            + end_tag
        )

    # TODO switch this to get epoch start/ stop times and plot only for good stimuli
    names = tuple(
        k for k, v in session.stim_frame_times.items() if not isinstance(v, Exception)
    )
    fig = session.sync_data.plot_stim_onsets()
    for idx, ax in enumerate(fig.axes):
        if len(names) == len(fig.axes):
            ax.set_title(names[idx].split("_")[0])
    fig.set_size_inches(10, 5 * len(fig.axes))
    fig.subplots_adjust(hspace=0.3)

    fig = session.sync_data.plot_stim_offsets()
    names = tuple(k for k, v in session.stim_frame_times.items() if v is not None)
    for idx, ax in enumerate(fig.axes):
        if len(names) == len(fig.axes):
            ax.set_title(names[idx].split("_")[0])
    fig.set_size_inches(10, 5 * len(fig.axes))
    fig.subplots_adjust(hspace=0.3)

    return fig


def plot_histogram_of_vsync_intervals(session):
    stim_frame_times = {
        k: v
        for k, v in session.stim_frame_times.items()
        if not isinstance(v, Exception)
    }

    fig_hist, axes_hist = plt.subplots(1, len(stim_frame_times))
    fig_hist.set_size_inches(12, 6)

    for ax, (stim_name, stim_times) in zip(axes_hist, stim_frame_times.items()):
        ax.hist(np.diff(stim_times), bins=np.arange(0, 0.1, 0.001))
        ax.set_yscale("log")
        ax.axvline(1 / 60, c="k", ls="dotted")
        ax.set_title(stim_name.split("_")[0])
        ax.set_xlabel("time (s)")
        ax.set_ylabel("frame interval count")
    plt.tight_layout()


def plot_reward_times(session):
    fig, ax = plt.subplots()
    ax.hist(session.trials[:].reward_time - session.trials[:].response_time)
    ax.xaxis.label.set_text("contingent_reward_time - response_time (s)")
    ax.yaxis.label.set_text("count")
    ax.set_xlim(min(0, ax.get_xlim()[0]), ax.get_xlim()[1])
    ax.vlines(0, 0, ax.get_ylim()[1], color="k", linestyle="dotted")

    return fig


def plot_long_vsyncs_distribution_across_trial(
    session: "npc_sessions.DynamicRoutingSession",
) -> matplotlib.figure.Figure:
    all_vsyncs = np.hstack(session.sync_data.vsync_times_in_blocks)

    interval_threshold = 0.017  # s

    all_long_intervals: list[float] = []
    fig, axes = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(12, 4))
    for cidx, condition in enumerate(("is_vis_stim", "is_aud_stim")):
        plt.sca(axes[cidx])
        for idx, trial in session.trials[:].query(condition).iterrows():
            vsyncs = (
                all_vsyncs[
                    (all_vsyncs >= trial.start_time) & (all_vsyncs <= trial.stop_time)
                ]
                - trial.start_time
            )
            intervals = np.diff(vsyncs)
            long_vsyncs = vsyncs[:-1][intervals > interval_threshold]
            long_intervals = intervals[intervals > interval_threshold]
            plt.scatter(
                long_vsyncs,
                np.ones_like(long_vsyncs) * idx,
                c=long_intervals,
                marker=".",
                facecolor="none",
                cmap="magma_r",
                s=4,
                alpha=1,
            )
            all_long_intervals.extend(long_intervals)

        plt.gca().set_title(condition.split("_")[1])
        top_ax = plt.gca().secondary_xaxis("top")
        top_ax.set_xticks([trial["stim_start_time"] - trial.start_time])
        top_ax.set_xticklabels(
            [f'stim_start_time {" (incl device latency)" if session.is_ephys else ""}']
        )
        c = plt.colorbar()
        c.set_ticks(
            [
                interval_threshold,
                *(unique_intervals := np.unique(np.round(all_long_intervals, 2))),
            ]
        )
        plt.clim([interval_threshold, max(unique_intervals)])
        plt.xlabel("time from trial start (s)")
        if cidx == 0:
            plt.ylabel("trial index")

    plt.suptitle(
        f"{session.id} - vsync intervals > {interval_threshold:.3f} s", fontsize=8
    )
    return fig
