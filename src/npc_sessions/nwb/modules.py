from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import ClassVar, Protocol

import ndx_events
import npc_lims
import pandas as pd
import polars as pl
import pynwb

import npc_sessions.utils as utils


def get_behavior(nwb_file: pynwb.NWBFile) -> pynwb.ProcessingModule:
    """Get or create `nwb_file['behavior']`"""
    return nwb_file.processing.get("behavior") or nwb_file.create_processing_module(
        name="behavior",
        description="Processed behavioral data",
    )


def get_ecephys(nwb_file: pynwb.NWBFile) -> pynwb.ProcessingModule:
    """Get or create `nwb_file['ecephys']`"""
    return nwb_file.processing.get("ecephys") or nwb_file.create_processing_module(
        name="ecephys",
        description="Processed ecephys data",
    )


class SupportsToNWB(Protocol):
    def to_nwb(self, nwb: pynwb.NWBFile) -> None:
        ...


class SupportsAsNWB(Protocol):
    def as_nwb(self) -> pynwb.core.NWBContainer:
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
            if hasattr(record, "nwb"):
                getattr(nwb, self.add_to_nwb_method)(**record.nwb)
            else:
                getattr(nwb, self.add_to_nwb_method)(**record)


class NWBContainerWithDF(NWBContainer):
    def to_dataframe(self) -> pd.DataFrame:
        return self.df.to_pandas()

    @utils.cached_property
    def df(self) -> pl.DataFrame:
        if all(isinstance(record, npc_lims.RecordWithNWB) for record in self.records):
            return pl.from_records(tuple(record.nwb for record in self.records))  # type: ignore [attr-defined]
        return pl.from_records(self.records)


class Subject(NWBContainerWithDF):
    records: tuple[npc_lims.Subject, ...]
    add_to_nwb_method = "add_subject"


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

    def to_nwb(self, nwb: pynwb.NWBFile) -> None:
        nwb.units = pynwb.misc.Units.from_dataframe(self.to_dataframe(), name="units")


class LickSpout(SupportsToNWB):
    name = "lick_sensor_rising"
    description = (
        "times at which the subject interacted with a water spout - "
        "putatively licks, but may include other events such as grooming"
    )

    def __init__(self, sync_path_or_dataset: utils.SyncPathOrDataset) -> None:
        self.timestamps = utils.get_sync_data(sync_path_or_dataset).get_rising_edges(
            "lick_sensor", units="seconds"
        )

    def as_nwb(self) -> ndx_events.Events:
        return ndx_events.Events(
            timestamps=self.timestamps,
            name=self.name,
            description=self.description,
        )

    def to_nwb(self, nwb_file: pynwb.NWBFile) -> None:
        nwb_file.add_acquisition(lick_nwb_data=self.as_nwb())


class RunningSpeed(SupportsToNWB):
    name = "running_speed"
    description = (
        "linear forward running speed on a rotating disk, low-pass filtered "
        f"at {utils.RUNNING_LOWPASS_FILTER_HZ} Hz with a 3rd order Butterworth filter"
    )
    unit = "m/s"
    conversion = 100 if utils.RUNNING_SPEED_UNITS == "cm/s" else 1.0
    # comments = f'Assumes mouse runs at `radius = {utils.RUNNING_DISK_RADIUS} {utils.RUNNING_SPEED_UNITS.split("/")[0]}` on disk.'

    def __init__(
        self,
        *stim: utils.StimPathOrDataset,
        sync: utils.SyncPathOrDataset | None = None,
    ) -> None:
        self.data, self.timestamps = utils.get_running_speed_from_stim_files(
            *stim, sync=sync, filt=utils.lowpass_filter
        )

    def as_nwb(self) -> pynwb.TimeSeries:
        return pynwb.TimeSeries(
            name=self.name,
            description=self.description,
            data=self.data,
            timestamps=self.timestamps,
            unit=self.unit,
            conversion=self.conversion,
        )

    def to_nwb(self, nwb_file: pynwb.NWBFile) -> None:
        get_behavior(nwb_file).add(self.as_nwb())


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
