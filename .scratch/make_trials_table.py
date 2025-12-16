import concurrent.futures as cf

import npc_lims
import npc_session
import pandas as pd
import polars as pl
import pyarrow

import npc_sessions
# import npc_sessions 
# import npc_lims 


_PARQUET_COMPRESSION = "zstd"
_COMPRESSION_LEVEL = 15


current_trials = pl.read_parquet('s3://aind-scratch-data/dynamic-routing/cache/nwb_components/v0.0.272/consolidated/trials.parquet')
print(current_trials)


        
def _write_df_to_cache(
    session_id: str | npc_session.SessionRecord,
    component_name: npc_lims.NWBComponentStr,
    df: pd.DataFrame,
    version: str | None = None,
    skip_existing: bool = True,
) -> None:
    """Write dataframe to cache file (e.g. .parquet)."""
    if df.empty:
        raise ValueError(f"{session_id} {component_name} df is empty")

    cache_path = npc_lims.get_cache_path(
        nwb_component=component_name,
        session_id=session_id,
        version=version or npc_sessions.get_package_version(),
        consolidated=False,
    )

    if skip_existing and cache_path.exists():
        print(
            f"Skipping write to {cache_path} - already exists and skip_existing=True"
        )
        return
    if component_name == "units" and "location" in df.columns:
        # most common access will be units from the same areas, so make sure
        # these rows are stored together
        df = df.sort_values("location")

    if cache_path.suffix == ".parquet":
        pyarrow.parquet.write_table(
            table=pyarrow.Table.from_pandas(df, preserve_index=True),
            where=cache_path,
            # disabled --------------------------------------------------------- #
            ## row_group_size=20 if component_name == "units" else None,
            # - ---------------------------------------------------------------- #
            # each list in the units.spike_times column is large & should not really be
            # stored in this format. But we can at least optimize access.
            # Row groups are indivisible, so querying a single row will download a
            # chunk of row_group_size rows: default is 10,000 rows, so accessing spike_times
            # for a single unit would take the same as accessing 10,000.
            # Per session, each area has 10-200 units per 'location', so we probably
            # want a row_group_size in that range.
            # Tradeoff is row_group_size gives worse compression and worse access speeds
            # across chunks (so querying entire columns will be much slower than default)
            compression=_PARQUET_COMPRESSION,
            compression_level=_COMPRESSION_LEVEL,
        )
    else:
        raise NotImplementedError(f"{cache_path.suffix=}")
    print(f"Wrote {cache_path}")
    
    
def main():
    for session_id in current_trials['session_id'].unique():
        try:
            session = npc_sessions.Session(session_id)
            trials = session.trials     
        