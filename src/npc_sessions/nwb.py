from __future__ import annotations

import functools
from collections.abc import Iterable, Iterator
from typing import ClassVar, Protocol

import npc_lims
import pandas as pd
import polars as pl
import pynwb


class SupportsToNWB(Protocol):
    def to_nwb(self, nwb: pynwb.NWBFile) -> None:
        ...


class NWBContainer(SupportsToNWB):
    add_to_nwb_method: ClassVar[str] = NotImplemented

    records: tuple[npc_lims.Record, ...]

    def __contains__(self, key: npc_lims.Record) -> bool:
        return key in self.records

    def __iter__(self) -> Iterator[npc_lims.Record]:
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def __init__(self, records: Iterable[npc_lims.Record], **kwargs) -> None:
        self.records = tuple(records)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_nwb(self, nwb: pynwb.NWBFile) -> None:
        for record in self.records:
            getattr(nwb, self.add_to_nwb_method)(**record.__dict__)


class NWBContainerWithDF(NWBContainer):
    def to_dataframe(self) -> pd.DataFrame:
        return self.df.to_pandas()

    @functools.cached_property
    def df(self) -> pl.DataFrame:
        return pl.from_records(self.records)


class Epochs(NWBContainerWithDF):
    records: tuple[npc_lims.Epoch, ...]
    add_to_nwb_method = "add_epoch"


class Devices(NWBContainerWithDF):
    records: tuple[npc_lims.Device, ...]
    add_to_nwb_method = "add_device"


class ElectrodeGroups(NWBContainerWithDF):
    records: tuple[npc_lims.ElectrodeGroup, ...]
    add_to_nwb_method = "add_electrode_group"


class Electrodes(NWBContainerWithDF):
    records: tuple[npc_lims.Electrode, ...]
    add_to_nwb_method = "add_electrode"


class Units(NWBContainerWithDF):
    records: tuple[npc_lims.Units, ...]

    def add_to_nwb(self, nwb: pynwb.NWBFile) -> None:
        nwb.units = pynwb.misc.Units.from_dataframe(self.to_dataframe(), name="units")


class Intervals(NWBContainerWithDF):
    """Pass `name`, `description` and `column_names_to_descriptions` as kwargs."""

    records: tuple[npc_lims.Epoch, ...]
    name: str
    description: str
    column_names_to_descriptions: dict[str, str] = {}

    def add_to_nwb(self, nwb: pynwb.NWBFile) -> None:
        module = pynwb.epoch.TimeIntervals(
            name=self.name,
            description=self.description,
        )
        for key in self.records[0].__dict__.keys():
            if key in ("start_time", "stop_time"):
                continue
            module.add_column(
                name=key, description=self.column_names_to_descriptions.get(key, "")
            )

        for record in self.records:
            module.add_row(**record.__dict__)

        nwb.add_time_intervals(module)
