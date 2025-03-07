from __future__ import annotations

import gc
import logging
import math
import warnings
from collections.abc import Iterable
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl
import polars._typing
import scipy.stats
import tqdm

logger = logging.getLogger(__name__)

ACTIVITY_DRIFT_THRESHOLD = 0.1


def insert_is_observed(
    intervals_frame: polars._typing.FrameType,
    units_frame: polars._typing.FrameType,
    col_name: str = "is_observed",
    unit_id_col: str = "unit_id",
) -> polars._typing.FrameType:

    if isinstance(intervals_frame, pl.LazyFrame):
        intervals_lf = intervals_frame
    else:
        intervals_lf = intervals_frame.lazy()

    if isinstance(units_frame, pl.LazyFrame):
        units_lf = units_frame
    else:
        units_lf = units_frame.lazy()

    units_schema = units_lf.collect_schema()
    if unit_id_col not in units_schema:
        raise ValueError(
            f"units_frame does not contain {unit_id_col!r} column: can be customized by passing unit_id_col"
        )
    if "obs_intervals" not in units_schema:
        raise ValueError("units_frame must contain 'obs_intervals' column")

    unit_ids = units_lf.select(unit_id_col).collect().get_column(unit_id_col).unique()
    intervals_schema = intervals_lf.collect_schema()
    if unit_id_col not in intervals_schema:
        if len(unit_ids) > 1:
            raise ValueError(
                f"units_frame contains multiple units, but intervals_frame does not contain {unit_id_col!r} column to perform join"
            )
        elif len(unit_ids) == 0:
            raise ValueError(
                f"units_frame contains no unit ids in {unit_id_col=} column"
            )
        else:
            intervals_lf = intervals_lf.with_columns(
                pl.lit(unit_ids[0]).alias(unit_id_col)
            )
    if not all(c in intervals_schema for c in ("start_time", "stop_time")):
        raise ValueError(
            "intervals_frame must contain 'start_time' and 'stop_time' columns"
        )

    if units_schema["obs_intervals"] in (
        pl.List(pl.List(pl.Float64())),
        pl.List(pl.List(pl.Int64())),
        pl.List(pl.List(pl.Null())),
    ):
        logger.info("Converting 'obs_intervals' column to list of lists")
        units_lf = units_lf.explode("obs_intervals")
    assert (type_ := units_lf.collect_schema()["obs_intervals"]) == pl.List(
        pl.Float64
    ), f"Expected exploded obs_intervals to be pl.List(f64), got {type_}"
    intervals_lf = (
        intervals_lf.join(
            units_lf.select(unit_id_col, "obs_intervals"), on=unit_id_col, how="left"
        )
        .with_columns(
            pl.when(
                pl.col("obs_intervals").list.get(0).gt(pl.col("start_time"))
                | pl.col("obs_intervals").list.get(1).lt(pl.col("stop_time")),
            )
            .then(pl.lit(False))
            .otherwise(pl.lit(True))
            .alias(col_name),
        )
        .group_by("unit_id", "start_time")
        .agg(
            pl.all().exclude("obs_intervals", col_name).first(),
            pl.col(col_name).any(),
        )
    )
    if isinstance(intervals_frame, pl.LazyFrame):
        return intervals_lf
    return intervals_lf.collect()


def get_per_trial_spike_times(
    units_df: pl.DataFrame,
    trials_df: pl.DataFrame,
    starts: pl.Expr | Iterable[pl.Expr],
    ends: pl.Expr | Iterable[pl.Expr],
    col_names: str | Iterable[str],
    apply_obs_intervals: bool = True,
    as_counts: bool = False,
) -> pl.DataFrame:
    """"""
    if isinstance(starts, pl.Expr):
        starts = (starts,)
    if isinstance(ends, pl.Expr):
        ends = (ends,)
    if isinstance(col_names, str):
        col_names = (col_names,)
    assert isinstance(col_names, tuple)  # for mypy
    assert isinstance(starts, tuple)
    assert isinstance(ends, tuple)
    if len(set(col_names)) != len(col_names):
        raise ValueError("col_names must be unique")
    if len(starts) != len(ends) != len(col_names):
        raise ValueError("starts, ends, and col_names must have the same length")

    # temp add columns for each interval with type list[float] (start, end)
    temp_col_prefix = "__temp_interval"
    for start, end, col_name in zip(starts, ends, col_names):
        trials_df = trials_df.with_columns(
            pl.concat_list(start, end).alias(f"{temp_col_prefix}_{col_name}"),
        )

    results: dict[str, list] = {
        "unit_id": [],
        "trial_index": trials_df["trial_index"].to_list() * len(units_df),
    }
    for col_name in col_names:
        results[col_name] = []
    for row in units_df.select("unit_id", "spike_times").iter_rows(named=True):
        if row["unit_id"] is None:
            raise ValueError(f"Missing unit_id in {row=}")
        results["unit_id"].extend([row["unit_id"]] * len(trials_df))

        for start, end, col_name in zip(starts, ends, col_names):
            # get spike times with start:end interval for each row of the trials table
            spike_times = row["spike_times"]
            spikes_in_all_intervals: list[list[float | int] | float | int] = []
            for a, b in np.searchsorted(
                spike_times,
                trials_df[f"{temp_col_prefix}_{col_name}"].to_list(),
            ):
                interval_spike_times: npt.NDArray[np.floating] = spike_times[a:b]
                #! spikes coincident with end of interval are not included
                if as_counts:
                    spikes_in_all_intervals.append(len(interval_spike_times))
                else:
                    spikes_in_all_intervals.append(interval_spike_times.tolist())
            results[col_name].extend(spikes_in_all_intervals)

    if apply_obs_intervals:
        results_df = trials_df.drop(pl.selectors.starts_with(temp_col_prefix)).join(
            other=pl.DataFrame(results),
            on="trial_index",
            how="left",
        )
    else:
        results_df = pl.DataFrame(results)

    if apply_obs_intervals:
        results_df = insert_is_observed(
            intervals_frame=results_df,
            units_frame=units_df,
        ).with_columns(
            *[
                pl.when(pl.col("is_observed").not_())
                .then(pl.lit(float("nan")))
                .otherwise(pl.col(col_name))
                .alias(col_name)
                for col_name in col_names
            ]
        )
    return results_df


def get_test_samples(
    unit_df: pl.DataFrame, interval: Literal["baseline", "response", "trial"]
) -> list[list[int]] | None:
    if interval == "trial" and "trial" not in unit_df.columns:
        unit_df = unit_df.with_columns(trial=pl.col("baseline") + pl.col("response"))
    if unit_df[interval].sum() == 0:
        # cannot perform test if all samples are the same (zero spikes is a common case)
        return None
    if unit_df.n_unique("block_index") < 2:
        # for Templeton we chunk spike counts into 3 segments of time
        return [
            s[interval].to_list()
            for s in unit_df.sort("trial_index").iter_slices(
                math.ceil(len(unit_df) / 3)
            )
        ]
    return (
        unit_df.group_by("block_index")
        .agg(pl.col(interval))
        .get_column(interval)
        .to_list()
    )


def add_activity_drift_metric(
    units_df: pd.DataFrame,
    trials_df: pd.DataFrame,
    col_name: str = "activity_drift",
) -> pd.DataFrame:

    units_pl = pl.from_pandas(units_df[["unit_id", "spike_times", "obs_intervals"]])
    trials_pl = pl.from_pandas(trials_df[:])
    trials_with_counts = get_per_trial_spike_times(
        units_df=units_pl,
        trials_df=trials_pl,
        starts=(
            pl.col("stim_start_time") - 1.5,
            pl.col("stim_start_time"),
        ),
        ends=(
            pl.col("stim_start_time"),
            pl.col("stim_start_time") + 2,
        ),
        col_names=(
            "baseline",
            "response",
        ),
        as_counts=True,
    )
    del units_pl
    del trials_pl
    gc.collect()

    class NullResult:
        statistic = np.nan
        pvalue = np.nan

    null_result = NullResult()

    test_results = []
    with np.errstate(over="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for (unit_id, context_name, stim_name, *_), unit_df in tqdm.tqdm(
            iterable=(
                trials_with_counts.drop_nulls(["baseline", "response"])
                .filter(pl.col("stim_name") != "catch")
                .group_by("unit_id", "context_name", "stim_name", maintain_order=True)
            ),
            total=trials_with_counts.n_unique("unit_id"),
            unit="unit",
            desc="calculating activity drift",
            ncols=80,
        ):
            result = dict(
                unit_id=unit_id,
                context_name=context_name,
                stim_name=stim_name,
            )
            for interval in (
                "trial",
            ):  # previous intervals we explored were "baseline" and "response"
                samples = get_test_samples(unit_df, interval)
                if samples is None:
                    stats = null_result
                else:
                    try:
                        stats = scipy.stats.anderson_ksamp(
                            samples,
                            midrank=False,
                            method=None,
                            # even with 10k permutations, majorit of p-values are clipped at 0.0001, which is useless
                            # so use default lookup table method, which is fast, then discard p-value
                        )
                    except RuntimeWarning as e:
                        print(f"Warning for {unit_id}, {interval}: {e!r}")
                        logger.warning(
                            f"Warning encountered calculating AD test for {unit_id}, {interval}: {e!r}"
                        )
                        stats = null_result
                result[f"ad_stat_{interval}"] = stats.statistic
            test_results.append(result)
    max_min_df = (
        pl.DataFrame(test_results)
        .select(
            "unit_id",
            pl.col("ad_stat_trial")
            .max()
            .over("unit_id")
            .truediv(100)
            .clip(0, 1)
            .alias(col_name),
            # very few units have slightly negative scores, so we clip them to 0
        )
        .with_columns(
            pl.col(col_name).lt(ACTIVITY_DRIFT_THRESHOLD).alias("is_not_drift")
        )
        .unique("unit_id")
    )
    return units_df.merge(max_min_df.to_pandas(), on="unit_id", how="left")
