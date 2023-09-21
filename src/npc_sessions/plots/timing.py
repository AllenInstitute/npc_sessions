from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    import npc_sessions


def plot_bad_lick_times(session: "npc_sessions.DynamicRoutingSession") -> None:
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

    for idx in (
        *trials_with_lick_outside_response_window,
        *trials_with_lick_inside_response_window_but_not_recorded,
    ):
        start = session._trials.response_window_start_time[idx]
        stop = session._trials.response_window_stop_time[idx]
        vsyncs = session._trials._sync.get_falling_edges("vsync_stim", units="seconds")
        licks = session._trials._sync.get_rising_edges("lick_sensor", units="seconds")
        padding = 0.3
        marker_config = {"linestyles": "-", "linelengths": 0.2}
        line_config = {"linestyles": "-", "linelengths": 1}
        plt.eventplot(
            vsyncs[(vsyncs >= start - padding) & (vsyncs <= stop + padding)],
            **line_config,
            label="vsyncs",
            alpha=0.5,
            color="orange",
        )
        plt.eventplot(
            vsyncs[(vsyncs >= start) & (vsyncs <= stop)],
            **line_config,
            label="vsyncs within response window",
            color="orange",
        )
        plt.eventplot(
            licks[(licks >= start - padding) & (licks <= stop + padding)],
            **marker_config,
            label="licks",
            color="k",
            lineoffsets=1,
        )
        plt.eventplot(
            [
                session._trials.get_script_frame_time(
                    session._trials._sam.trialResponseFrame[idx]
                )
            ],
            **marker_config,
            label="lick frame in TaskControl",
            color="lime",
            lineoffsets=1.6,
        )
        plt.eventplot(
            [
                session._trials.get_script_frame_time(
                    session._trials._sam.stimStartFrame[idx]
                )
            ],
            **marker_config,
            label="stim start frame in TaskControl",
            color="r",
            lineoffsets=1.6,
        )

        plt.legend(fontsize=8, loc="upper center", fancybox=True, ncol=4)
        plt.gca().set_yticks([])
        plt.gca().set_xlabel("time (s)")
        plt.gcf().set_size_inches(12, 4)
        plt.gca().title.set_text(
            f"sync & script timing - trial {idx} - {session._trials.stim_name[idx]}"
        )
        # plt.show()


def plot_lick_times_on_sync_and_script(
    session: "npc_sessions.DynamicRoutingSession",
) -> None:
    """-stem plot of lick times on sync relative to lick times in TaskControl
    - histogram showing distribution of same intervals
    """
    sync_time = session._trials.response_time
    script_time = session._trials.get_script_frame_time(
        session._trials._sam.trialResponseFrame
    )

    intervals = sync_time - script_time

    markerline, stemline, baseline = plt.stem(
        sync_time,
        intervals,
        bottom=0,
        orientation="horizontal",
    )
    plt.setp(stemline, linewidth=0.5, alpha=0.3)
    plt.setp(markerline, markersize=0.5, alpha=0.8)
    plt.setp(baseline, visible=False)
    plt.gca().set_xlabel("lick time on sync relative to lick time in TaskControl (s)")
    plt.gca().set_ylabel("experiment time (s)")
    plt.gca().set_title(
        f"{np.nanmean(intervals) = :.3f}s, {np.nanstd(intervals) = :.3f}"
    )
    plt.show()

    plt.hist(intervals, bins=50)
    plt.gca().set_xlabel("lick time on sync relative to lick time in TaskControl (s)")
    plt.gca().set_ylabel("count")
    plt.gca().set_title(
        f"{np.nanmean(intervals) = :.3f}s, {np.nanstd(intervals) = :.3f}"
    )
    plt.show()
