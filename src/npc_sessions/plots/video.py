from __future__ import annotations

import datetime
import random
from collections.abc import Iterable
from typing import TYPE_CHECKING

import cv2
import matplotlib.figure
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import rich
import upath

import npc_sessions.utils as utils

if TYPE_CHECKING:
    import npc_sessions

import npc_sessions.plots.plot_utils as plot_utils
import npc_sessions.utils as utils


def plot_video_info(
    session: npc_sessions.DynamicRoutingSession,
) -> None:
    "Not a plot: prints info to stdout"

    augmented_camera_info = utils.get_augmented_camera_info(
        session.sync_data, *session.video_paths
    )

    for camera, info in augmented_camera_info.items():
        rich.print(f"[bold]{camera} camera stats[bold]")

        frame_rate = info["FPS"]
        frame_rate_string = plot_utils.add_valence_to_string(
            f"Frame Rate: {frame_rate} \t",
            frame_rate,
            abs(frame_rate - 60) < 0.01,
            abs(frame_rate - 60) > 0.05,
        )

        lost_frame_percentage = 100 * info["FramesLostCount"] / info["FramesRecorded"]
        lost_frame_string = plot_utils.add_valence_to_string(
            f"Lost frame percentage: {np.round(lost_frame_percentage, 3)} \t",
            lost_frame_percentage,
            lost_frame_percentage < 0.01,
            lost_frame_percentage > 0.05,
        )

        frame_diff_from_expected = info["expected_minus_actual"]
        frame_diff_string = plot_utils.add_valence_to_string(
            f"Frames expected minus actual: {frame_diff_from_expected}",
            frame_diff_from_expected,
            abs(frame_diff_from_expected) < 1,
            abs(frame_diff_from_expected) > 10,
        )

        rich.print(frame_rate_string + lost_frame_string + frame_diff_string)


def plot_camera_frame_grabs_simple(
    session: npc_sessions.DynamicRoutingSession,
    paths: Iterable[upath.UPath] | None = None,
    num_frames_to_grab: int = 5,
) -> matplotlib.figure.Figure:
    """Just plots evenly spaced frames, no concept of epochs.

    video frames across cameras aren't synced .
    """
    if paths is None:
        paths = session.video_paths

    paths = tuple(paths)

    fig = plt.figure(
        figsize=[10, 3 * len(paths)], constrained_layout=True, facecolor="0.5"
    )
    gs = gridspec.GridSpec(len(paths), num_frames_to_grab, figure=fig)
    gs.update(wspace=0.0, hspace=0.0)
    for idx, video_path in enumerate(paths):
        # get frames to plot

        v = utils.get_video_data(video_path)  # TODO open with upath from cloud

        frame_delta = np.ceil(v.get(cv2.CAP_PROP_FRAME_COUNT) / num_frames_to_grab + 1)
        frames_of_interest = np.arange(
            v.get(cv2.CAP_PROP_FPS), v.get(cv2.CAP_PROP_FRAME_COUNT), frame_delta
        )

        for i, f in enumerate(frames_of_interest):
            v.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ret, frame = v.read()
            ax = fig.add_subplot(gs[idx, i])
            ax.imshow(frame)
            # ax.axis('off')
            ax.tick_params(
                top=False,
                bottom=False,
                left=False,
                right=False,
                labelleft=False,
                labelbottom=False,
            )
            ax.set_title(
                datetime.timedelta(seconds=f / v.get(cv2.CAP_PROP_FPS)), fontsize=10
            )
    return fig


def plot_video_frames_with_licks(
    session: npc_sessions.DynamicRoutingSession,
    trial_idx: int | None = None,
    lick_time: float | None = None,
):
    NUM_LICKS = 3 if (trial_idx is None and lick_time is None) else 1
    NUM_CAMERAS = 2  # 1 x face, 1 x body

    FRAMES_EITHER_SIDE_OF_LICK = 2
    """Symmetric around frame closest to lick"""
    FRAMES_PER_ROW = FRAMES_EITHER_SIDE_OF_LICK * 2 + 1

    ROWS_PER_VIDEO = 2  # 1 x camera frames, 1 x eventplots
    ROWS_PER_LICK = ROWS_PER_VIDEO * NUM_CAMERAS

    if lick_time is None:
        response_times: npt.NDArray = (
            (session.trials[:].query("is_response").response_time.to_numpy())
            if trial_idx is None
            else (session.trials[trial_idx].response_time.to_numpy())
        )
        lick_times = sorted(
            random.sample(tuple(response_times[~np.isnan(response_times)]), NUM_LICKS)
        )
    else:
        lick_times = [lick_time]

    fig = plt.figure(figsize=[12, 5 * NUM_LICKS], facecolor="0.5")
    # fig, axes = plt.subplots(NUM_LICKS * ROWS_PER_LICK, FRAMES_PER_ROW,)
    # constrained_layout=True,
    gs = gridspec.GridSpec(
        NUM_LICKS * ROWS_PER_LICK,
        FRAMES_PER_ROW,
        height_ratios=[1, 0.1] * NUM_CAMERAS * NUM_LICKS,
    )
    gs.update(wspace=0.0, hspace=0.0)

    video_frame_times = utils.get_video_frame_times(
        session.sync_data, *session.video_paths
    )
    for vid_idx, (video_path, frame_times) in enumerate(video_frame_times.items()):
        if "eye" in video_path.stem.lower():
            continue
        for lick_idx, lick_time in enumerate(lick_times):
            v = utils.get_video_data(video_path)  # TODO open with upath from cloud

            closest_frame_index = np.nanargmin(np.abs(frame_times - lick_time))  # type: ignore[operator]
            frame_indices = np.arange(
                closest_frame_index - FRAMES_EITHER_SIDE_OF_LICK,
                closest_frame_index + FRAMES_EITHER_SIDE_OF_LICK + 1,
            )

            # video frames around lick time
            for frame_idx, frame_index in enumerate(frame_indices):
                v.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ret, frame = v.read()
                ax = fig.add_subplot(
                    gs[
                        (vid_idx * ROWS_PER_VIDEO) + (lick_idx * ROWS_PER_LICK),
                        frame_idx,
                    ]
                )
                if "beh" in video_path.stem.lower():
                    x = slice(0, frame.shape[1] // 3)
                    y = slice(frame.shape[0] // 5, frame.shape[0] // 2)
                if "face" in video_path.stem.lower():
                    ymid, xmid = frame.shape[0] // 2, frame.shape[1] // 2
                    yspan, xspan = frame.shape[0] // 5, frame.shape[1] // 5
                    x = slice(xmid - xspan, xmid + xspan)
                    y = slice(ymid - yspan, ymid + yspan)
                ax.imshow(frame[y, x])
                ax.axis("off")
                if frame_index == closest_frame_index:
                    trial = (
                        np.searchsorted(  # type: ignore[call-overload]
                            session.trials[:].start_time, lick_time, "right"
                        )
                        - 1
                    )
                    ax.set_title(f"{trial=}, {lick_time=:.1f} s", fontsize=8)

            # markers for frame and lick times
            ax = fig.add_subplot(
                gs[
                    1 + (vid_idx * ROWS_PER_VIDEO) + (lick_idx * ROWS_PER_LICK),
                    :FRAMES_PER_ROW,
                ]
            )
            linekwargs = {"linewidths": 3}
            ax.eventplot(frame_times[frame_indices], label="frame", **linekwargs)
            ax.eventplot([lick_time], color="red", label="lick", **linekwargs)
            ax.axis("off")
            if lick_idx == vid_idx == 0:
                ax.legend(fontsize=8, fancybox=True, ncol=2, loc="upper right")

    plt.tight_layout()
