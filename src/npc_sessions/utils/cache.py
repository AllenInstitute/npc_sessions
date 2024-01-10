"""

- write units to cache file

"""
from __future__ import annotations

import importlib.metadata
import logging
import typing
from collections.abc import Mapping

import npc_lims
import npc_session
import pandas as pd
import pyarrow
import pyarrow.dataset
import pyarrow.parquet
import pynwb

if typing.TYPE_CHECKING:
    import npc_sessions

logger = logging.getLogger(__name__)

class MissingComponentError(AttributeError):
    pass

def _get_nwb_component(
    session: pynwb.NWBFile | npc_sessions.DynamicRoutingSession,
    component_name: npc_lims.NWBComponentStr,
) -> pynwb.core.NWBContainer | pd.DataFrame | None:
    def _component_metadata_to_single_row_df(component):
        # if value is a list, wrap it in another list so pandas doesn't interpret
        # it as multiple rows
        return pd.DataFrame(
            {
                k: (v if not isinstance(v, list) else [v])
                for k, v in component.fields.items()
                if not isinstance(v, (Mapping, pynwb.core.Container))
            },
            index=[0],
        )

    if component_name == "session":
        if not isinstance(session, pynwb.NWBFile):
            session = session.metadata
        return _component_metadata_to_single_row_df(session).drop(
            columns="file_create_date"
        )
    elif component_name == "subject":
        return _component_metadata_to_single_row_df(session.subject)
    elif component_name in ("vis_rf_mapping", "VisRFMapping"):
        return session.intervals.get("VisRFMapping", None)
    elif component_name in ("aud_rf_mapping", "AudRFMapping"):
        return session.intervals.get("AudRFMapping", None)
    elif component_name in ("optotagging", "OptoTagging"):
        return session.intervals.get("OptoTagging", None)
    elif component_name == "performance":
        if session.analysis:
            return session.analysis.get("performance", None)
        else:
            return None
    else:
        c = getattr(session, component_name, None)
        if c is None:
            raise MissingComponentError(
                f"Unknown NWB component {component_name!r} - available tables include {typing.get_args(npc_lims.NWBComponentStr)}"
            )
        return c


def write_nwb_component_to_cache(
    component: pynwb.core.NWBContainer | pd.DataFrame,
    component_name: npc_lims.NWBComponentStr,
    session_id: str | npc_session.SessionRecord,
    version: str | None = None,
    skip_existing: bool = True,
) -> None:
    """Write units to cache file (e.g. .parquet) after processing columns.

    - ND arrays are not supported, so waveform mean/sd are condensed to 1D arrays
      with data for peak channel only, or dropped entirely
    - links to NWBContainers cannot be stored, so extract necessary 'foreign key'
      to enable joining tables later
    """
    if component_name == "units":
        component = _flatten_units(component)
    df = _remove_pynwb_containers(component)
    df = add_session_metadata(df, session_id)
    _write_to_cache(
        session_id=session_id,
        component_name=component_name,
        df=df,
        version=version,
        skip_existing=skip_existing,
    )


def write_all_components_to_cache(
    session: pynwb.NWBFile | npc_sessions.DynamicRoutingSession,
    skip_existing: bool = True,
    version: str | None = None,
) -> None:
    """Write all components to cache files (e.g. .parquet) after processing columns.

    - ND arrays are not supported, so waveform mean/sd are condensed to 1D arrays
      with data for peak channel only, or dropped entirely
    - links to NWBContainers cannot be stored, so extract necessary 'foreign key'
      to enable joining tables later
    """
    logger.info(f"Writing all components to cache for {session.id}")
    for component_name in typing.get_args(npc_lims.NWBComponentStr):
        # skip before we potentially do a lot of processing to get component
        if (
            skip_existing
            and npc_lims.get_cache_path(
                nwb_component=component_name, session_id=session.id, version=version
            ).exists()
        ):
            logger.info(
                f"Skipping {session.id} {component_name} - already in cache (set skip_existing=False if you want to overwrite)"
            )
            continue
        try:
            component = _get_nwb_component(session, component_name)
            # component may be None
        except ValueError:
            component = None
        if component is None:
            logger.debug(f"{component_name} not available for {session.id}")
            return
        write_nwb_component_to_cache(
            component=component,
            component_name=component_name,
            session_id=session.id,
            version=version,
            skip_existing=skip_existing,
        )


def add_session_metadata(
    df: pd.DataFrame, session_id: str | npc_session.SessionRecord
) -> pd.DataFrame:
    session_id = npc_session.SessionRecord(session_id)
    df = df.copy()
    df["session_idx"] = session_id.idx
    df["date"] = session_id.date.dt
    df["subject_id"] = session_id.subject
    return df


def _write_to_cache(
    session_id: str | npc_session.SessionRecord,
    component_name: npc_lims.NWBComponentStr,
    df: pd.DataFrame,
    version: str | None = None,
    skip_existing: bool = True,
) -> None:
    """Write dataframe to cache file (e.g. .parquet)."""
    version = version or importlib.metadata.version("npc-sessions")
    cache_path = npc_lims.get_cache_path(
        nwb_component=component_name, session_id=session_id, version=version
    )
    if cache_path.suffix != ".parquet":
        raise NotImplementedError(f"{cache_path.suffix=}")
    if skip_existing and cache_path.exists():
        logger.debug(
            f"Skipping write to {cache_path} - already exists and skip_existing=True"
        )
        return
    table = pyarrow.Table.from_pandas(df, preserve_index=False)
    if component_name == "units" and "location" in table.schema.names:
        table = table.sort_by("location")
        # most common access will be units from the same areas, so make sure
        # these rows are stored together
    pyarrow.parquet.write_table(
        table=table,
        where=cache_path.as_posix(),
        row_group_size=20 if "component" == "units" else None,
        # each list in the spike_times column is large - should not really be
        # stored in this format. But we can at least optimize for it by creating
        # smaller row groups, so querying a single unit returns a chunk of rows
        # equal to row_group_size, instead of default 10,000 rows per session
        compression="zstd",
        compression_level=15,
    )
    logger.info(f"Wrote {cache_path}")


def get_dataset(
    nwb_component: npc_lims.NWBComponentStr,
    session_id: str | npc_session.SessionRecord | None = None,
    version: str | None = None,
) -> pyarrow.dataset.Dataset:
    """Get dataset for all sessions, for all components, for the latest version."""
    return pyarrow.dataset.dataset(
        paths=npc_lims.get_cache_path(
            nwb_component=nwb_component,
            session_id=session_id,
            version=version,
        ),
        format=npc_lims.get_cache_file_suffix(nwb_component).lstrip("."),
    )


def _flatten_units(units: pynwb.misc.Units | pd.DataFrame) -> pd.DataFrame:
    units = units[:].copy()
    # deal with links to other NWBContainers
    units["device_name"] = units["electrode_group"].apply(lambda eg: eg.device.name)
    units["electrode_group_name"] = units["electrode_group"].apply(lambda eg: eg.name)
    return _remove_pynwb_containers(units)


def _remove_pynwb_containers(
    table_or_df: pynwb.core.DynamicTable | pd.DataFrame,
) -> pd.DataFrame:
    df = table_or_df[:].copy()
    return df.drop(
        [col for col in df.columns if isinstance(df[col][0], pynwb.core.NWBContainer)]
    )


if __name__ == "__main__":
    import doctest

    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
