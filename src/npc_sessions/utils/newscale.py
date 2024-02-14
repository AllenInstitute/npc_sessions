from __future__ import annotations

import collections
import datetime
import logging
from collections.abc import Iterable

import npc_session
import pandas as pd
import polars as pl

import npc_sessions.utils as utils

logger = logging.getLogger(__name__)

SERIAL_NUM_TO_PROBE_LETTER = (
    {
        "SN32148": "A",
        "SN32142": "B",
        "SN32144": "C",
        "SN32149": "D",
        "SN32135": "E",
        "SN24273": "F",
    }  # NP.0
    | {
        "SN40911": "A",
        "SN40900": "B",
        "SN40912": "C",
        "SN40913": "D",
        "SN40914": "E",
        "SN40910": "F",
    }  # NP.1
    | {
        "SN45356": "A",
        "SN45484": "B",
        "SN45485": "C",
        "SN45359": "D",
        "SN45482": "E",
        "SN45361": "F",
    }  # NP.2
    | {
        "SN40906": "A",
        "SN40908": "B",
        "SN40907": "C",
        "SN41084": "D",
        "SN40903": "E",
        "SN40902": "F",
    }  # NP.3
)
SHORT_TRAVEL_SERIAL_NUMBERS = {
    "SN32148",
    "SN32142",
    "SN32144",
    "SN32149",
    "SN32135",
    "SN24273",
}
SHORT_TRAVEL_RANGE = 6_000
LONG_TRAVEL_RANGE = 15_000
NEWSCALE_LOG_COLUMNS = (
    "last_movement",
    "device",
    "x",
    "y",
    "z",
    "x_virtual",
    "y_virtual",
    "z_virtual",
)


def get_newscale_data(path: utils.PathLike) -> pl.DataFrame:
    """
    >>> df = get_newscale_data('s3://aind-ephys-data/ecephys_686740_2023-10-23_14-11-05/behavior_videos/log.csv')
    """
    return pl.read_csv(
        source=utils.from_pathlike(path).as_posix(),
        new_columns=NEWSCALE_LOG_COLUMNS,
        try_parse_dates=True,
    )


def get_newscale_data_lazy(path: utils.PathLike) -> pl.LazyFrame:
    """
    # >>> df = get_newscale_data_lazy('s3://aind-ephys-data/ecephys_686740_2023-10-23_14-11-05/behavior_videos/log.csv')
    """
    # TODO not working with s3 paths
    return pl.scan_csv(
        source=utils.from_pathlike(path).as_posix(),
        with_column_names=lambda _: list(NEWSCALE_LOG_COLUMNS),
        try_parse_dates=True,
    )


def get_newscale_coordinates(
    newscale_log_path: utils.PathLike,
    recording_start_time: (
        str | datetime.datetime | npc_session.DatetimeRecord | None
    ) = None,
) -> pd.DataFrame:
    """Returns the coordinates of each probe at the given time, by scanning for the most-recent prior movement on each motor.

    - looks up the timestamp of movement preceding `recording_start_time`
    - if not provided, attempt to parse experiment (sync) start time from `newscale_log_path`:
      assumes manipulators were not moved after the start time

    >>> df = get_newscale_coordinates('s3://aind-ephys-data/ecephys_686740_2023-10-23_14-11-05/behavior_videos/log.csv', '2023-10-23 14-11-05')
    >>> list(df['x'])
    [6278.0, 6943.5, 7451.0, 4709.0, 4657.0, 5570.0]
    >>> list(df['z'])
    [3920.0, 6427.0, 8500.0, 6893.0, 6962.0, 5875.0]
    """
    if recording_start_time is None:
        p = utils.from_pathlike(newscale_log_path)
        try:
            start = npc_session.DatetimeRecord(p.as_posix())
        except ValueError as exc:
            raise ValueError(
                f"`recording_start_time` must be provided to indicate start of ephys recording: no time could be parsed from {p.as_posix()}"
            ) from exc
    else:
        start = npc_session.DatetimeRecord(recording_start_time)

    movement = pl.col(NEWSCALE_LOG_COLUMNS[0])
    serial_number = pl.col(NEWSCALE_LOG_COLUMNS[1])
    df: pl.DataFrame
    try:
        df = get_newscale_data_lazy(newscale_log_path)  # type: ignore [assignment]
    except (pl.ComputeError, OSError):
        df = get_newscale_data(newscale_log_path)

    z_df = df.select(["z"])
    if isinstance(df, pl.LazyFrame):
        z_df = z_df.collect()
    z_values = z_df["z"]
    z_inverted: bool = is_z_inverted(z_values)

    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    df = (
        df.filter(movement < start.dt)
        .group_by(serial_number)
        .agg(
            pl.col(NEWSCALE_LOG_COLUMNS[:-3]).sort_by(movement).last()
        )  # get last-moved for each manipulator
        .top_k(6, by=movement)
    )
    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    # serial numbers have an extra leading space
    manipulators = df.get_column(NEWSCALE_LOG_COLUMNS[1]).map_elements(
        lambda _: _.strip()
    )
    df = df.with_columns(manipulators)
    # convert str floats to floats
    for column in NEWSCALE_LOG_COLUMNS[2:8]:
        if column not in df.columns:
            continue
        df = df.with_columns(
            df.get_column(column).map_elements(lambda _: _.strip()).cast(pl.Float64)
        )
    probes = manipulators.map_dict(
        {k: f"probe{v}" for k, v in SERIAL_NUM_TO_PROBE_LETTER.items()}
    ).alias("electrode_group")

    # correct z values
    z = df["z"]
    for idx, device in enumerate(df["device"]):
        if z_inverted:
            z[idx] = get_z_travel(device) - z[idx]
    df = df.with_columns(z)

    return (
        df.insert_column(index=0, column=probes)
        .sort(pl.col("electrode_group"))
        .to_pandas()
    )


def get_z_travel(serial_number: str) -> int:
    """
    >>> get_z_travel('SN32144')
    6000
    >>> get_z_travel('SN40911')
    15000
    """
    if serial_number not in SERIAL_NUM_TO_PROBE_LETTER:
        raise ValueError(
            f"{serial_number=} is not a known serial number: need to update {__file__}"
        )
    if serial_number in SHORT_TRAVEL_SERIAL_NUMBERS:
        return SHORT_TRAVEL_RANGE
    return LONG_TRAVEL_RANGE


def is_z_inverted(z_values: Iterable[float]) -> bool:
    """
    The limits of the z-axis are [0-6000] for NP.0 and [0-15000] for NP.1-3. The
    NewScale software sometimes (but not consistently) inverts the z-axis, so
    retracted probes have a z-coordinate of 6000 or 15000 not 0. This function checks
    the values in the z-column and tries to determine if the z-axis is inverted.

    Assumptions:
    - the manipulators spend more time completely retracted than completely extended

    >>> is_z_inverted([0, 3000, 3000, 0])
    False
    >>> is_z_inverted([15000, 3000, 3000, 15000])
    True
    """
    c = collections.Counter(z_values)
    is_long_travel = bool(c[LONG_TRAVEL_RANGE])
    travel_range = LONG_TRAVEL_RANGE if is_long_travel else SHORT_TRAVEL_RANGE
    return c[0] < c[travel_range]


if __name__ == "__main__":
    import doctest

    import dotenv

    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
