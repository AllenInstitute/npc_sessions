from __future__ import annotations

import abc
import functools
from collections.abc import Iterator
from typing import ClassVar

import npc_lims
import pandas as pd
import polars as pl
import pynwb


class NWBContainer(abc.ABC):
    add_to_nwb_method: ClassVar[str] = NotImplemented

    records: tuple[npc_lims.Record, ...]

    def __contains__(self, key: npc_lims.Record) -> bool:
        return key in self.records

    def __iter__(self) -> Iterator[npc_lims.Record]:
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def __init__(self, records: npc_lims.Iterable[npc_lims.Record]) -> None:
        self.records = tuple(records)

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
