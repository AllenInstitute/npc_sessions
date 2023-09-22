from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    import npc_sessions


def plot_performance_by_block(
    session: "npc_sessions.DynamicRoutingSession",
) -> plt.Figure:
    task_performance_by_block_df: pd.DataFrame = session._task_performance_by_block[:]

    n_passing_blocks = np.sum(task_performance_by_block_df["cross_modal_dprime"] >= 1.5)
    failed_block_ind = task_performance_by_block_df["cross_modal_dprime"] < 1.5

    # blockwise behavioral performance
    xvect = task_performance_by_block_df.index.values
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(
        xvect,
        task_performance_by_block_df["signed_cross_modal_dprime"],
        "ko-",
        label="cross-modal",
    )
    ax[0].plot(
        xvect[failed_block_ind],
        task_performance_by_block_df["signed_cross_modal_dprime"][failed_block_ind],
        "ro",
        label="failed",
    )
    ax[0].axhline(0, color="k", linestyle="--", linewidth=0.5)
    ax[0].set_title(
        "cross-modal dprime: "
        + str(n_passing_blocks)
        + "/"
        + str(len(task_performance_by_block_df))
        + " blocks passed"
    )
    ax[0].set_ylabel("aud <- dprime -> vis")

    ax[1].plot(
        xvect, task_performance_by_block_df["vis_intra_dprime"], "go-", label="vis"
    )
    ax[1].plot(
        xvect, task_performance_by_block_df["aud_intra_dprime"], "bo-", label="aud"
    )
    ax[1].set_title("intra-modal dprime")
    ax[1].legend(["vis", "aud"])
    ax[1].set_xlabel("block index")
    ax[1].set_ylabel("dprime")

    fig.suptitle(session.id)
    fig.tight_layout()

    return fig


def plot_first_lick_latency_hist(
    session: "npc_sessions.DynamicRoutingSession",
) -> plt.Figure:
    # first lick latency histogram

    trials: pd.DataFrame = session.trials[:]

    xbins = np.arange(0, 1, 0.05)
    fig, ax = plt.subplots(1, 1)
    ax.hist(
        trials.query("is_vis_stim==True")["response_time"]
        - trials.query("is_vis_stim==True")["stim_start_time"],
        bins=xbins,
        alpha=0.5,
    )

    ax.hist(
        trials.query("is_aud_stim==True")["response_time"]
        - trials.query("is_aud_stim==True")["stim_start_time"],
        bins=xbins,
        alpha=0.5,
    )

    ax.legend(["vis stim", "aud stim"])
    ax.set_xlabel("lick latency (s)")
    ax.set_ylabel("trial count")
    ax.set_title("lick latency: " + session.id)

    return fig


def plot_lick_raster(session: "npc_sessions.DynamicRoutingSession") -> plt.Figure:
    timeseries = session.acquisition["lick_spout"]
    trials: pd.DataFrame = session.trials[:]

    fig, ax = plt.subplots(1, 1)
    ax.axvline(0, color="k", linestyle="--", linewidth=0.5)
    for tt, trial in trials.iterrows():
        trial_licks = (
            timeseries.timestamps[
                (timeseries.timestamps > trial["stim_start_time"] - 1)
                & (timeseries.timestamps < trial["stim_start_time"] + 2)
            ]
            - trial["stim_start_time"]
        )

        ax.vlines(trial_licks, tt, tt + 1)

    ax.set_xlim([-1, 2])
    ax.set_xlabel("time rel to stim onset (s)")
    ax.set_ylabel("trial number")
    ax.set_title(timeseries.description, fontsize=8)
    fig.suptitle(session.id, fontsize=10)

    return fig


def plot_running(session: "npc_sessions.DynamicRoutingSession") -> plt.Figure:
    timeseries = session.processing["running"]
    fig, ax = plt.subplots(1, 1)
    plt.plot(timeseries.timestamps, timeseries.data)
    ax.set_ylim(-0.2, 1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel(timeseries.unit)
    ax.set_title(timeseries.description, fontsize=8)
    fig.suptitle(session.id, fontsize=10)
    fig.set_size_inches(12, 3)
    return fig
