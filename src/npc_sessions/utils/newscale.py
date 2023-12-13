from __future__ import annotations

import datetime
import logging

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
    time: str | datetime.datetime | npc_session.DatetimeRecord,
) -> pd.DataFrame:
    """Returns the coordinates of each probe at the given time.

    - looks up the timestamp of movement preceding `time`

    >>> df = get_newscale_coordinates('s3://aind-ephys-data/ecephys_686740_2023-10-23_14-11-05/behavior_videos/log.csv', '2023-10-23 14-11-05')
    >>> list(df['x'])
    [6278.0, 6943.5, 7451.0, 4709.0, 4657.0, 5570.0]
    """
    start = npc_session.DatetimeRecord(time)
    movement = pl.col(NEWSCALE_LOG_COLUMNS[0])
    serial_number = pl.col(NEWSCALE_LOG_COLUMNS[1])
    df: pl.DataFrame
    try:
        df = get_newscale_data_lazy(newscale_log_path)  # type: ignore [assignment]
    except pl.ComputeError:
        df = get_newscale_data(newscale_log_path)

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
    probes = manipulators.map_dict(
        {k: f"probe{v}" for k, v in SERIAL_NUM_TO_PROBE_LETTER.items()}
    ).alias("electrode_group")
    return df.insert_at_idx(0, probes).sort(pl.col("electrode_group")).to_pandas()


if __name__ == "__main__":
    import doctest

    import dotenv

    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
