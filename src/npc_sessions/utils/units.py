from __future__ import annotations

import functools
import io
import os

import npc_lims
import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl

import npc_sessions


@functools.cache
def get_units_electrodes(
    session: str, units_method="codeocean_kilosort", electrode_method="tissuecyte"
) -> pd.DataFrame:
    if units_method == "codeocean_kilosort":
        units_path = npc_lims.get_units_codeoean_kilosort_path_from_s3(session)
        units = pd.read_csv(
            units_path,
            storage_options={
                "key": os.getenv("AWS_ACCESS_KEY_ID"),
                "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
            },
        )
    if electrode_method == "tissuecyte":
        try:
            electrodes = npc_sessions.create_tissuecyte_electrodes_table(session)
            units = units.merge(
                electrodes,
                left_on=["device_name", "peak_channel"],
                right_on=["device_name", "channel"],
            )
            units.drop(columns=["channel"], inplace=True)
        except FileNotFoundError as e:
            print(str(e) + ". Returning units without electrodes")

    units.drop(columns=["electrodes"], inplace=True)
    return units


@functools.cache
def get_unit_spike_times_dict(
    session: str, unit_ids: tuple[str, ...], method="codeocean_kilosort"
) -> dict[str, npt.NDArray[np.float64]]:
    # change function call depending on method
    spike_times_dict = {}
    if method == "codeocean_kilosort":
        spike_times_path = npc_lims.get_spike_times_codeocean_kilosort_path_from_s3(
            session
        )
        with io.BytesIO(spike_times_path.read_bytes()) as f:
            spike_times = np.load(f, allow_pickle=True)

        for i in range(len(unit_ids)):
            spike_times_dict[unit_ids[i]] = spike_times[i]

    return spike_times_dict


def get_units_electrodes_spike_times(session: str, *args, **kwargs) -> pl.DataFrame:
    units_df = get_units_electrodes(session, *args, **kwargs)
    unit_ids = units_df["unit_name"].to_list()
    spike_times_dict = get_unit_spike_times_dict(
        session, tuple(unit_ids), *args, **kwargs
    )
    spike_times_df = pl.DataFrame(
        {
            "unit_name": tuple(spike_times_dict.keys()),
            "spike_times": tuple(t for t in spike_times_dict.values()),
        }
    )
    return pl.DataFrame(units_df).join(spike_times_df, on="unit_name")


if __name__ == "__main__":
    import doctest

    import dotenv

    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
