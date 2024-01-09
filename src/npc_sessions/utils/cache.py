"""

- write units to cache file

"""
from __future__ import annotations

import importlib.metadata
import logging
import typing
from typing import Literal, Mapping

import npc_session
import packaging.version
import pandas as pd
import polars as pl
import pyarrow
import pyarrow.dataset
import pyarrow.parquet
import pynwb
import upath
from typing_extensions import TypeAlias

import npc_sessions.utils as utils

logger = logging.getLogger(__name__)

S3_SCRATCH_ROOT = upath.UPath("s3://aind-scratch-data/ben.hardcastle")
CACHE_ROOT = S3_SCRATCH_ROOT / "session-caches"

NWBComponentStr: TypeAlias = Literal[
    "session",
    "subject",
    "units",
    "epochs",
    "trials",
    "performance",
    "vis_rf_mapping",
    "aud_rf_mapping",
    "optotagging",
    "invalid_times",
    "electrodes",
    "electrode_groups",
    "devices",
    #TODO licks, pupil area, xy pos, running speed (zarr?)
]

CACHED_FILE_EXTENSIONS: dict[str, str] = dict.fromkeys(
    typing.get_args(NWBComponentStr), ".parquet"
)
"""Mapping of NWB component name to file extension"""

assert CACHED_FILE_EXTENSIONS.keys() == set(
    typing.get_args(NWBComponentStr)
), "CACHED_FILE_EXTENSIONS must have keys for all NWBComponent values"
assert all(
    v.startswith(".") for v in CACHED_FILE_EXTENSIONS.values()
), "CACHED_FILE_EXTENSIONS must have values that start with a period"

def _get_nwb_component(session: pynwb.NWBFile | "npc_sessions.DynamicRoutingSession", component_name: NWBComponentStr) -> pynwb.core.NWBContainer | pd.DataFrame:
    if component_name == "session":
        if not isinstance(session, pynwb.NWBFile):
            session = session.nwb
        return pd.DataFrame(
            {k:v for k,v in session.fields.items() if not isinstance(v, (Mapping, pynwb.core.NWBContainer))}
        )
    elif component_name in ("vis_rf_mapping", "VisRFMapping"):
        return session.intervals["VisRFMapping"]
    elif component_name in ("aud_rf_mapping", "AudRFMapping"):
        return session.intervals["AudRFMapping"]
    elif component_name in ("optotagging", "OptoTagging"):
        return session.intervals["OptoTagging"]
    else:
        return getattr(session, component_name)
    
def write_nwb_component_to_cache(
    component: pynwb.core.NWBContainer | pd.DataFrame,
    component_name: NWBComponentStr,
    session_id: npc_session.SessionRecord,
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
    _write_to_cache(session_id=session_id, component_name=component_name, df=df, skip_existing=skip_existing)

def write_all_components_to_cache(
    session: pynwb.NWBFile | "npc_sessions.DynamicRoutingSession",
    skip_existing: bool = True,
) -> None:
    """Write all components to cache files (e.g. .parquet) after processing columns.

    - ND arrays are not supported, so waveform mean/sd are condensed to 1D arrays
      with data for peak channel only, or dropped entirely
    - links to NWBContainers cannot be stored, so extract necessary 'foreign key'
      to enable joining tables later
    """
    for component_name in typing.get_args(NWBComponentStr):
        # skip before we potentially do a lot of processing to get component
        if (p := get_cache_path(nwb_component=component_name, session_id=session.id)).exists():
            logger.debug(f"Skipping {component_name} for {session.id} - already exists at {p}")
            continue
        component = _get_nwb_component(session, component_name)
        try:
            write_nwb_component_to_cache(
                component=component,
                component_name=component_name,
                session_id=session.id,
                skip_existing=skip_existing,
            )
        except AttributeError:
            logger.warning(f"{component_name} not available for {session.id}")
        
def add_session_metadata(df: pd.DataFrame, session_id: str | npc_session.SessionRecord) -> pd.DataFrame:
    session_id = npc_session.SessionRecord(session_id)
    df = df.copy()
    df['session_idx'] = session_id.idx
    df['date'] = session_id.date.dt
    df['subject_id'] = session_id.subject
    return df

def _write_to_cache(
    session_id: str | npc_session.SessionRecord,
    component_name: NWBComponentStr,
    df: pd.DataFrame,
    skip_existing: bool = True,
) -> None:
    """Write dataframe to cache file (e.g. .parquet)."""
    version = importlib.metadata.version("npc-sessions")
    cache_path = get_cache_path(nwb_component=component_name, session_id=session_id, version=version)
    if cache_path.suffix != '.parquet':
        raise NotImplementedError(f"{cache_path.suffix=}")
    if skip_existing and cache_path.exists():
        logger.debug(f"Skipping write to {cache_path} - already exists and skip_existing=True")    
        return
    table = pyarrow.Table.from_pandas(df)
    if component_name == 'units' and 'location' in table.schema.names:
        table = table.sort_by('location')
        # most common access will be units from the same areas, so make sure
        # these rows are stored together
    pyarrow.parquet.write_table(
        table=table,
        where=cache_path.as_posix(),
        row_group_size=20 if 'component' == 'units' else None,
        # each list in the spike_times column is large - should not really be
        # stored in this format. But we can at least optimize for it by creating
        # smaller row groups, so querying a single unit returns a chunk of rows
        # equal to row_group_size, instead of default 10,000 rows per session
        compression='zstd',
        compression_level=15,
        )
    logger.info(f"Wrote {cache_path}")

def get_dataset(
    nwb_component: NWBComponentStr,
    session_id: str | npc_session.SessionRecord | None = None,
    version: str | None = None,
    ) -> pyarrow.dataset.Dataset:
    """Get dataset for all sessions, for all components, for the latest version."""
    return pyarrow.dataset.dataset(
        paths=get_cache_path(
            nwb_component=nwb_component,
            session_id=session_id,
            version=version,
        ),
        format=get_cache_file_suffix(nwb_component).lstrip('.'),
        )
    
def get_lazy_frame_from_path(
    path: utils.PathLike,
) -> pl.LazyFrame:
    """Read dataframe from cache file (e.g. .parquet)."""
    path = utils.from_pathlike(path)
    if path.suffix != '.parquet':
        raise NotImplementedError(f"{path.suffix=}")
    lf = pl.scan_parquet(path.as_uri())
    return lf

def get_lazy_frame_from_cache(
    component_name: NWBComponentStr,
    session_id: str | npc_session.SessionRecord | None = None,
    version: str | None = None,
) -> pl.LazyFrame:
    """Read dataframe from cache file (e.g. .parquet)."""
    if session_id is not None:
        return get_lazy_frame_from_path(get_cache_path(nwb_component=component_name, session_id=session_id, version=version))
    paths = get_all_cache_paths(component_name, version)
    return pl.concat([get_lazy_frame_from_path(path) for path in paths], how='diagonal')

def _flatten_units(units: pynwb.misc.Units | pd.DataFrame) -> pd.DataFrame:
    units = units[:].copy()
    # deal with links to other NWBContainers
    units["device_name"] = units["electrode_group"].apply(lambda eg: eg.device.name)
    units["electrode_group_name"] = units["electrode_group"].apply(lambda eg: eg.name)
    return _remove_pynwb_containers(units)

def _remove_pynwb_containers(table_or_df: pynwb.core.DynamicTable | pd.DataFrame) -> pd.DataFrame:
    df = table_or_df[:].copy()
    return df.drop([col for col in df.columns if isinstance(df[col][0], pynwb.core.NWBContainer)])

def get_cache_file_suffix(nwb_component: NWBComponentStr) -> str:
    """
    >>> get_cache_ext("session")
    '.parquet'
    """
    if (ext := CACHED_FILE_EXTENSIONS.get(nwb_component, None)) is None:
        raise ValueError(
            f"Unknown NWB component {nwb_component!r} - must be one of {NWBComponentStr}"
        )
    return ext


def get_current_cache_version() -> str:
    """
    >>> (get_cache_path(nwb_component="units", session_id="366122_2023-12-31", version="v0.0.0").parent / 'test.txt').touch()
    >>> v = get_current_cache_version()
    >>> assert v >= 'v0.0.0'
    """
    if not (version_dirs := sorted(CACHE_ROOT.glob("v*"))):
        raise FileNotFoundError(f"No cache versions found in {CACHE_ROOT}")
    return version_dirs[-1].name


def _parse_version(version: str) -> str:
    return f"v{packaging.version.parse(str(version))}"


def _parse_cache_path(
    nwb_component: NWBComponentStr,
    session_id: str | npc_session.SessionRecord | None = None,
    version: str | None = None,
) -> upath.UPath:
    version = _parse_version(version) if version else get_current_cache_version()
    d = (
        CACHE_ROOT
        / version
        / nwb_component
    )
    if session_id is None:
        return d
    return d / f"{npc_session.SessionRecord(session_id)}{get_cache_file_suffix(nwb_component)}"


def get_cache_path(
    nwb_component: NWBComponentStr,
    session_id: str | npc_session.SessionRecord | None = None,
    version: str | None = None,
    check_exists: bool = False,
) -> upath.UPath:
    """
    If version is not specified, the latest version currently in the cache will be
    used, ie. will point to the most recent version of the file.

    >>> get_cache_path(nwb_component="units", version="1.0.0")
    S3Path('s3://aind-scratch-data/ben.hardcastle/session-caches/v1.0.0/units')
    >>> get_cache_path(nwb_component="units", session_id="366122_2023-12-31", version="1.0.0")
    S3Path('s3://aind-scratch-data/ben.hardcastle/session-caches/v1.0.0/units/366122_2023-12-31.parquet')
    """
    path = _parse_cache_path(session_id=session_id, nwb_component=nwb_component, version=version)
    if check_exists and not path.exists():
        raise FileNotFoundError(
            f"Cache file for {session_id} {nwb_component} {version} does not exist"
        )
    return path


def get_all_cache_paths(
    nwb_component: NWBComponentStr,
    version: str | None = None,
) -> tuple[upath.UPath, ...]:
    """
    For a particular NWB component, return cached file paths for all sessions, for
    the latest version (default) or a specific version.

    >>> get_all_cache_paths("units", version="0.0.0")
    ()
    """
    dir_path = get_cache_path(nwb_component=nwb_component, version=version)
    if not dir_path.exists():
        raise FileNotFoundError(
            f"Cache directory for {nwb_component} {version} does not exist"
        )
    return tuple(path for path in dir_path.glob(f"*{get_cache_file_suffix(nwb_component)}"))


if __name__ == "__main__":
    import doctest

    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )

    import npc_sessions
    for s in npc_sessions.get_sessions():
        if not s.is_ephys:
            continue
        # try:
        write_all_components_to_cache(s, skip_existing=True)
        break
        # except Exception as e:
        #     print(s.id, repr(e))
    # lf = get_lazy_frame_from_cache('units')
    # lf
