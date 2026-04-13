from __future__ import annotations

import contextlib
import functools
import json
import logging

import npc_lims
import npc_session
import numpy as np
import pandas as pd
import polars as pl
import upath

TISSUECYTE_MICRONS_PER_PIXEL = 25

logger = logging.getLogger(__name__)


class NoElectrodeDataError(ValueError):
    """Raised when no electrode annotation data can be found for a session."""

    pass


@functools.cache
def get_electrodes_table(session: str | npc_session.SessionRecord) -> pd.DataFrame:
    """Get electrode annotation data for a session, from either TissueCyte or IBL
    annotation files. Columns are ready for insertion into nwb ElectrodeTable.

    If both TissueCyte and IBL annotation files are found for a session, IBL
    data will be used.
    """
    with contextlib.suppress(NoElectrodeDataError):
        return get_tissuecyte_electrodes_table(session)
    with contextlib.suppress(NoElectrodeDataError):
        return get_ibl_electrodes_table(session)
    raise NoElectrodeDataError(
        f"No TissueCyte or IBL electrode annotation data found for session {session}"
    )


@functools.cache
def get_structure_tree_df() -> pd.DataFrame:
    return pd.read_csv(
        "https://raw.githubusercontent.com/cortex-lab/allenCCF/refs/heads/master/structure_tree_safe_2017.csv"
    )


def strip_layer_from_area(areastr: str) -> str:
    areastr = areastr.split("-")[0]
    # remove layer stuff
    structure_tree_df = get_structure_tree_df()
    areaname = structure_tree_df[structure_tree_df["acronym"] == areastr]["name"].values
    if len(areaname) == 0:
        return areastr
    else:
        areaname = areaname[0]

    if "layer" in areaname:
        layer = areaname.split("layer")[-1].split(" ")[-1]
        areastr = areastr.replace(layer, "")

    # the above logic fails for some areas
    for prefix in ["ACAd", "ACAv", "MOp", "ECT"]:
        if areastr.startswith(prefix):
            areastr = prefix
            break

    return areastr


def get_tissuecyte_electrodes_table(
    session: str | npc_session.SessionRecord,
) -> pd.DataFrame:
    """Get annotation data for each electrode (channel) on each probe inserted in
    a session. Column names are ready for insertion into nwb ElectrodeTable.

    >>> df = get_tissuecyte_electrodes_table('626791_2022-08-16')
    >>> df.columns
    Index(['group_name', 'channel', 'location', 'structure', 'x', 'y', 'z',
           'raw_structure'],
          dtype='object')
    """
    try:
        electrode_files = npc_lims.get_tissuecyte_annotation_files_from_s3(session)
    except FileNotFoundError:
        raise NoElectrodeDataError(
            f"No TissueCyte electrode annotation files found for session {session}"
        ) from None

    session_electrodes = pd.DataFrame()

    for electrode_file in electrode_files:
        probe_electrodes = pd.read_csv(electrode_file)

        probe_name = npc_session.ProbeRecord(electrode_file.stem).name

        probe_electrodes["group_name"] = [str(probe_name)] * len(probe_electrodes)

        session_electrodes = pd.concat([session_electrodes, probe_electrodes])

    session_electrodes.rename(
        columns={
            "AP": "x",
            "DV": "y",
            "ML": "z",
            "region": "location",
            "region_stripped": "structure",
        },
        inplace=True,
    )
    # an upstream bug leaves ECT structure with layers: they need to be stripped
    session_electrodes["structure"] = session_electrodes["structure"].where(
        ~session_electrodes["structure"].str.startswith("ECT"), other="ECT"
    )
    if "raw_structure" in session_electrodes.columns:
        session_electrodes["raw_structure"] = session_electrodes["raw_structure"].where(
            ~session_electrodes["raw_structure"].str.startswith("ECT"), other="ECT"
        )
    for column in ("x", "y", "z"):
        # -1 is code for "not inserted": make this NaN
        session_electrodes[column] = session_electrodes[column].replace(-1, np.nan)
        session_electrodes[column] *= TISSUECYTE_MICRONS_PER_PIXEL
    session_electrodes = session_electrodes[
        [
            "group_name",
            "channel",
            "location",
            "structure",
            "x",
            "y",
            "z",
        ]
        + (
            [
                "raw_structure",
            ]
            if "raw_structure" in session_electrodes.columns
            else []
        )
    ]
    return session_electrodes


def get_ibl_electrodes_table(
    session: str | npc_session.SessionRecord,
) -> pd.DataFrame:
    """Get annotation data for each electrode (channel) on each probe inserted in
    a session. Column names are ready for insertion into nwb ElectrodeTable.

    >>> df = get_ibl_electrodes_table('752311_2025-01-22')
    >>> df.columns
    Index(['group_name', 'channel', 'location', 'structure', 'x', 'y', 'z'],
          dtype='object')
    """
    try:
        ccf_df = get_ibl_ccf_channel_locations_df(session)
    except FileNotFoundError:
        raise NoElectrodeDataError(
            f"No IBL electrode annotation files found for session {session}"
        ) from None
    return (
        ccf_df.join(
            pl.DataFrame(get_structure_tree_df()[["acronym", "name", "id"]]),
            left_on="brain_region_id",
            right_on="id",
            how="left",
        )
        .drop("x", "y", "z")  # original IBL coordinate values
        .rename(
            {
                "probe": "group_name",
                "acronym": "structure",
                "ccf_ap": "x",
                "ccf_dv": "y",
                "ccf_ml": "z",
            }
        )
        .with_columns(
            pl.col("structure")
            .map_elements(strip_layer_from_area, return_dtype=pl.String)
            .alias("location")
        )
        .with_columns(
            pl.col("structure").fill_null(pl.lit("out of brain")),
            pl.col("location").fill_null(pl.lit("out of brain")),
            pl.col("x").fill_null(pl.lit(float("nan"))),
            pl.col("y").fill_null(pl.lit(float("nan"))),
            pl.col("z").fill_null(pl.lit(float("nan"))),
        )
        .select("group_name", "channel", "location", "structure", "x", "y", "z")
    ).to_pandas()


def get_ibl_ccf_channel_locations_df(
    session: str | npc_session.SessionRecord,
) -> pl.DataFrame:
    """Get a polars DataFrame of CCF channel locations for all probes in a session,
    from IBL annotation `ccf_channel_locations.json` files.

    Columns: probe, channel, x, y, z, axial, lateral, brain_region_id, brain_region,
    ccf_ap, ccf_dv, ccf_ml (all CCF values in µm).

    Examples:
        >>> df = get_ibl_ccf_channel_locations_df('752311_2025-01-22')
        >>> assert len(df) > 0
        >>> assert {'ccf_ap', 'ccf_dv', 'ccf_ml', 'probe', 'channel'}.issubset(df.columns)
    """
    ccf_ap = (pl.col("y") * 1000).alias("ccf_ap")
    ccf_dv = (pl.col("z") * -1000).alias("ccf_dv")
    ccf_ml = (pl.col("x") * -1000).alias("ccf_ml")

    get_ibl_annotation_files_from_s3 = getattr(
        npc_lims, "get_ibl_annotation_files_from_s3", None
    )
    if get_ibl_annotation_files_from_s3 is not None:
        annotation_files = get_ibl_annotation_files_from_s3(session)
    else:
        session = npc_session.SessionRecord(session)
        # dir is organized as <root>/<subject_id>/<session_id>/<probe_name>/<file_name>.json
        annotation_files = tuple(
            upath.UPath("s3://aind-scratch-data/dynamic-routing/ibl-gui-output").glob(
                f"{session.subject}/*{session}*/*/*.json"
            )
        )
    frames: list[pl.DataFrame] = []
    for path in annotation_files:
        probe_name = path.parent.name
        logger.info("Reading %s annotations from %s", probe_name, path)
        data: dict[str, dict] = json.loads(path.read_text())
        rows = [
            {
                "channel": int(key.split("_")[1]),
                **values,
            }
            for key, values in data.items()
        ]
        if probe_name.lower().startswith("probe") and len(probe_name) == 6:
            probe_name = f"probe{probe_name[-1].upper()}"
        df = (
            pl.DataFrame(rows)
            .with_columns(probe=pl.lit(probe_name))
            .with_columns(
                ccf_ap,
                ccf_dv,
                ccf_ml,
            )
        )
        for col in ("ccf_ap", "ccf_dv", "ccf_ml"):
            mean_val = df[col].mean()
            if isinstance(mean_val, (int, float)) and mean_val < 0:
                logger.warning(
                    f"Mean of {col} coordinates in {probe_name} IBL GUI annotations for {session} have likely been updated and no longer need negating. Negation will be reverted automatically."
                )
                df = df.with_columns(pl.col(col) * -1)
        frames.append(df)
    return pl.concat(frames).sort("probe", "channel")


if __name__ == "__main__":
    import doctest

    import dotenv

    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
