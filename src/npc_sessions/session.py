from __future__ import annotations

import abc
from collections.abc import MutableMapping, Sequence
import functools
from typing import Any, NamedTuple, Protocol

import polars as pl
import upath
import npc_session
import npc_lims
import npc_lims.status.tracked_sessions as tracked_sessions
import npc_sessions.parse_settings_xml as parse_settings_xml
import upath


# TODO

# class SupportsDataFrame(Protocol):
    
#     @abc.abstractmethod
#     def to_df(self) -> pl.DataFrame:
#         pass
# db = npc_lims.get_db(Optional[path])
# db = npc_lims.NWBSqliteDBHub()

class Session:
    
    
    def __init__(self, session: str) -> None:
        self.record = npc_session.SessionRecord(str(session))

    def __getattribute__(self, __name: str) -> Any:
        if __name in ("date", "subject", "idx"):
            return self.record.__getattribute__(__name)
        return super().__getattribute__(__name)

    @functools.cached_property
    def info(self) -> tracked_sessions.SessionInfo | None:  # noqa: F821
        return next(
            (
                info 
                for info in npc_lims.tracked
                if info.session == self.record
            ),
            None,
            )
    
    @property
    def is_sync(self) -> bool:
        if self.info is None:
            return False
        return self.info.is_sync
    
    @property
    def is_ephys(self) -> bool:
        if self.info is None:
            return False
        return self.info.is_ephys
    
    @property
    def raw(self) -> tuple[upath.UPath, ...]:
        return npc_lims.get_raw_data_paths_from_s3(self.record)

    @functools.cached_property
    def sync_file(self) -> upath.UPath | None:
        """
        >>> Session('670248_2023-08-03').sync_file
        S3Path('s3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15/behavior/20230803T120415.h56')
        """
        if not self.is_sync:
            return None
        paths = tuple(
            p for p in self.raw
            if p.suffix in (".h5", ".sync") and 
            p.stem.startswith(f"{self.record.date.replace('-', '')}T")
        )
        if not paths:
            return None
        if len(paths) > 1:
            raise ValueError(f"Expected 1 sync file, found {paths}")
        return paths[0]
    
    @property
    def settings_xml_path(self) -> upath.UPath | None:
        settings_xml_path = npc_lims.get_settings_xml_path_from_s3(self.record)
        if not settings_xml_path:
            return None
        
        return settings_xml_path
    
    @functools.cached_property
    def settings_xml_info(self) -> parse_settings_xml.SettingsXmlInfo | None:
        if self.settings_xml_path is None:
            return None
        
        return parse_settings_xml.settings_xml_info_from_path(self.settings_xml_path)
        
    # paths: Sequence[upath.UPath]
    # trials: pl.DataFrame
    # intervals: Sequence[pl.DataFrame]
    # epochs: pl.DataFrame
    # state: MutableMapping[str | int, Any]
    # subject: MutableMapping[str, Any]
    # session: MutableMapping[str, Any]
    # stimuli: pl.DataFrame
    # units: pl.DataFrame


if __name__ == "__main__":
    import doctest

    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
