from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import matplotlib.axes
    import pandas as pd

bad_tag = "[bold red]"
good_tag = "[bold green]"
cautious_tag = "[bold yellow]"


EPOCH_COLOR_MAP = {
    "RFMapping": "black",
    "DynamicRouting1": "blue",
    "OptoTagging": "cyan",
    "Spontaneous": "red",
    "SpontaneousRewards": "magenta",
}


def add_epoch_color_bars(
    ax: matplotlib.axes.Axes,
    epochs: pd.DataFrame,
    epoch_color_map: dict[str, Any] | None = None,
    **text_kwargs,
) -> None:
    if epoch_color_map is None:
        epoch_color_map = EPOCH_COLOR_MAP
    text_kwargs = {
        "y": 0,
        "ha": "center",
        "rotation": 0,
        "fontsize": 6,
        "zorder": 1,
    } | text_kwargs
    for _, epoch in epochs.iterrows():
        epoch_name = next((k for k in epoch.tags if k in epoch_color_map), "")
        color = epoch_color_map[epoch_name] if epoch_name else "black"
        ax.axvspan(epoch.start_time, epoch.stop_time, alpha=0.1, color=color, zorder=0)
        ax.text(
            x=(epoch.stop_time + epoch.start_time) / 2,
            s=epoch_name,
            **text_kwargs,
        )


def bad_string(base: str) -> str:
    return bad_tag + base + bad_tag


def good_string(base: str) -> str:
    return good_tag + base + good_tag


def cautious_string(base: str) -> str:
    return cautious_tag + base + cautious_tag


def determine_string_valence(
    value: Any, good_criterion: bool, bad_criterion: bool
) -> Callable[[str], str]:
    if good_criterion:
        return good_string

    elif bad_criterion:
        return bad_string

    else:
        return cautious_string


def add_valence_to_string(
    basestring: str, value: float | int, good_criterion: bool, bad_criterion: bool
) -> str:
    valence_func = determine_string_valence(value, good_criterion, bad_criterion)

    return valence_func(basestring)
