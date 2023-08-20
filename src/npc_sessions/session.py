"""
>>> s = Session('670248_2023-08-03')
>>> s.session_start_time
'2023-08-03 12:04:15'
>>> 'DynamicRouting1' in s.epoch_tags
True
>>> s.sync_path
S3Path('s3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15/behavior/20230803T120415.h5')
>>> s.stim_paths[0]
S3Path('s3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15/behavior/DynamicRouting1_670248_20230803_123154.hdf5')
>>> s.devices.df['device_id'][0] == s.electrode_groups.df['device'][0]
True
"""
from __future__ import annotations

import datetime
import functools
import io
import json
import operator
from collections.abc import Mapping
from typing import Any

import h5py
import npc_lims
import npc_lims.status.tracked_sessions as tracked_sessions
import npc_session
import pynwb
import upath

import npc_sessions.components.parse_settings_xml as parse_settings_xml
import npc_sessions.components.sync_dataset as sync_dataset
import npc_sessions.nwb as nwb
import npc_sessions.utils as utils


class Session:
    trials_interval_name: str = "dynamic_routing_task"

    experimenter: str | None = None
    experiment_description: str | None = None
    identifier: str | None = None
    notes: str | None = None
    source_script: str | None = None

    def __init__(self, session: str, **kwargs) -> None:
        self.id = npc_session.SessionRecord(str(session))
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getattribute__(self, __name: str) -> Any:
        if __name in ("date", "subject", "idx"):
            return self.id.__getattribute__(__name)
        return super().__getattribute__(__name)

    @functools.cached_property
    def info(self) -> tracked_sessions.SessionInfo | None:
        return next(
            (info for info in npc_lims.tracked if info.session == self.id),
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
    def session_start_time(self) -> npc_session.DatetimeRecord:
        if self.is_sync:
            return npc_session.DatetimeRecord(self.sync_path.stem)
        start_time = self.epochs.df["start_time"].min()
        start_time = (
            start_time.decode() if isinstance(start_time, bytes) else start_time
        )
        return npc_session.DatetimeRecord(f"{self.date} {start_time}")

    def get_record(self) -> npc_lims.Session:
        return npc_lims.Session(
            session_id=self.id,
            subject_id=self.subject,
            session_start_time=self.session_start_time,
            stimulus_notes=self.task_version,
            experimenter=self.experimenter,
            experiment_description=self.experiment_description,
            epoch_tags=list(self.epoch_tags),
            source_script=self.source_script,
            identifier=self.identifier,
            notes=self.notes,
        )

    @functools.cached_property
    def record(self) -> npc_lims.Session:
        return self.get_record()

    def to_nwb(self, nwb: pynwb.NWBFile) -> None:
        for attr in self.record.__dict__:
            if attr in nwb.__dict__:
                nwb.__setattr__(attr, self.record.__getattribute__(attr))

    class Subject(npc_lims.Subject):
        def __init__(self, session: Session) -> None:
            self.session = session
            self.record = npc_lims.Subject(session.subject)

        @property
        def age(self) -> str:
            if self.record.date_of_birth is None:
                raise ValueError(f"{self.record} does not have a date of birth")
            dob = npc_session.DatetimeRecord(self.record.date_of_birth)
            return f"P{(self.session.session_start_time.dt - dob.dt).days}D"

        def to_nwb(self, nwb: pynwb.NWBFile) -> None:
            nwb.subject = pynwb.file.Subject(**self.record.__dict__)

    @property
    def epoch_tags(self) -> tuple[str, ...]:
        return tuple(
            set(functools.reduce(operator.add, self.epochs.df["tags"].to_list()))
        )

    @functools.cached_property
    def raw_data_paths(self) -> tuple[upath.UPath, ...]:
        if not self.is_ephys:
            raise ValueError(f"{self.id} is not a session with a raw ephys upload")
        return npc_lims.get_raw_data_paths_from_s3(self.id)

    @functools.cached_property
    def sorted_data_paths(self) -> tuple[upath.UPath, ...]:
        if not self.is_ephys:
            raise ValueError(f"{self.id} is not a session with ephys")
        return npc_lims.get_sorted_data_paths_from_s3(self.id)

    @property
    def sync_path(self) -> upath.UPath:
        if not self.is_sync:
            raise ValueError(f"{self.id} is not a session with sync data")
        paths = tuple(
            p
            for p in self.raw_data_paths
            if p.suffix in (".h5", ".sync")
            and p.stem.startswith(f"{self.id.date.replace('-', '')}T")
        )
        if not len(paths) == 1:
            raise ValueError(f"Expected 1 sync file, found {paths = }")
        return paths[0]

    @functools.cached_property
    def raw_data_asset_id(self) -> str:
        if not self.is_ephys:  # currently only ephys sessions have raw data assets
            raise ValueError(f"{self.id} is not a session with ephys raw data")
        asset_info = npc_lims.get_session_raw_data_asset(self.id)
        if not asset_info:
            raise ValueError(f"{self.id} does not have a raw data asset yet")
        return asset_info["id"]

    @functools.cached_property
    def sync_file_record(self) -> npc_lims.File:
        path = self.sync_path
        return npc_lims.File(
            session_id=self.id,
            name="sync",
            suffix=path.suffix,
            timestamp=npc_session.TimeRecord.parse_id(path.stem),
            size=path.stat()["size"],
            s3_path=path.as_posix(),
            data_asset_id=None if not self.is_ephys else self.raw_data_asset_id,
        )

    @functools.cached_property
    def sync_data(self) -> sync_dataset.SyncDataset:
        return sync_dataset.SyncDataset(io.BytesIO(self.sync_path.read_bytes()))

    @property
    def stim_paths(self) -> tuple[upath.UPath, ...]:
        return tuple(
            p
            for p in self.raw_data_paths
            if (
                p.suffix in (".pkl")
                and any(
                    label in p.stem for label in ("stim", "mapping", "opto", "behavior")
                )
            )
            or (
                p.suffix in (".hdf5")
                and f"{self.subject}_{self.date.replace('-', '')}" in p.stem
            )
        )

    @functools.cached_property
    def stim_file_records(self) -> tuple[npc_lims.File, ...]:
        return tuple(
            npc_lims.File(
                session_id=self.id,
                name=path.stem.split("_")[0],
                suffix=path.suffix,
                timestamp=npc_session.TimeRecord.parse_id(path.stem),
                size=path.stat()["size"],
                s3_path=path.as_posix(),
                data_asset_id=None if not self.is_ephys else self.raw_data_asset_id,
            )
            for path in self.stim_paths
        )

    @functools.cached_property
    def stim_data(self) -> Mapping[str, Any]:
        def h5_dataset(path: upath.UPath) -> h5py.File:
            return h5py.File(io.BytesIO(path.read_bytes()), "r")

        return utils.LazyDict(
            (path.stem, (h5_dataset, path)) for path in self.stim_paths
        )

    @property
    def video_paths(self) -> tuple[upath.UPath, ...]:
        if not self.is_sync:
            raise ValueError(
                f"{self.id} is not a session with sync data (required for video)"
            )
        return tuple(
            p
            for p in self.raw_data_paths
            if p.suffix in (".avi", ".mp4", ".zip")
            and any(label in p.stem.lower() for label in ("eye", "face", "beh"))
        )

    @property
    def video_info_paths(self) -> tuple[upath.UPath, ...]:
        return tuple(
            p.with_suffix(".json").with_stem(
                p.stem.replace(".mp4", "").replace(".avi", "")
            )
            for p in self.video_paths
        )

    @functools.cached_property
    def video_info_data(self) -> Mapping[str, Any]:
        def recording_report(path: upath.UPath) -> dict[str, str | int | float]:
            return json.loads(path.read_bytes())["RecordingReport"]

        return utils.LazyDict(
            (utils.extract_video_file_name(path.stem), (recording_report, path))
            for path in self.video_info_paths
        )

    @functools.cached_property
    def video_file_records(self) -> tuple[npc_lims.File, ...]:
        return tuple(
            npc_lims.File(
                session_id=self.id,
                name=utils.extract_video_file_name(path.stem),
                suffix=path.suffix,
                timestamp=npc_session.TimeRecord.parse_id(
                    self.video_info_data[utils.extract_video_file_name(path.stem)][
                        "TimeStart"
                    ]
                ),
                size=path.stat()["size"],
                s3_path=path.as_posix(),
                data_asset_id=None if not self.is_ephys else self.raw_data_asset_id,
            )
            for path in self.video_paths
        )

    @functools.cached_property
    def video_info_file_records(self) -> tuple[npc_lims.File, ...]:
        return tuple(
            npc_lims.File(
                session_id=self.id,
                name=utils.extract_video_file_name(path.stem),
                suffix=path.suffix,
                timestamp=npc_session.TimeRecord.parse_id(
                    self.video_info_data[path.stem]["TimeStart"]
                ),
                size=path.stat()["size"],
                s3_path=path.as_posix(),
                data_asset_id=None if not self.is_ephys else self.raw_data_asset_id,
            )
            for path in self.video_info_paths
        )

    @property
    def task_data(self) -> h5py.File:
        return next(self.stim_data[k] for k in self.stim_data if "DynamicRouting" in k)

    @property
    def task_version(self) -> str:
        return str(self.task_data["taskVersion"].asstr()[()])

    def get_epoch_record(self, stim_file_name: str) -> npc_lims.Epoch:
        h5 = self.stim_data[stim_file_name]
        tags = [stim_file_name.split("_")[0]]
        if any(label in h5 for label in ("optoRegions", "optoParams")):
            tags.append("opto")
        if any(h5["rewardFrames"][:]):
            tags.append("rewards")
        hdf5_start = npc_session.extract_isoformat_datetime(h5["startTime"].asstr()[()])
        assert hdf5_start is not None
        start = datetime.datetime.fromisoformat(hdf5_start)
        stop = start + datetime.timedelta(seconds=sum(h5["frameIntervals"][:]))
        start_time = start.time().isoformat(timespec="seconds")
        stop_time = stop.time().isoformat(timespec="seconds")
        assert start_time != stop_time
        return npc_lims.Epoch(
            session_id=self.id,
            start_time=start_time,
            stop_time=stop_time,
            tags=tags,
        )

    @functools.cached_property
    def epochs(self) -> nwb.NWBContainerWithDF:
        return nwb.Epochs(self.get_epoch_record(stim) for stim in self.stim_data)

    @property
    def settings_xml_path(self) -> upath.UPath:
        if not self.is_ephys:
            raise ValueError(
                f"{self.id} is not an ephys session (required for settings.xml)"
            )
        settings_xml_path = npc_lims.get_settings_xml_path_from_s3(self.id)
        if not settings_xml_path:
            raise ValueError(
                f"settings.xml not found for {self.id} on s3 - check status of raw upload"
            )
        return settings_xml_path

    @functools.cached_property
    def settings_xml_data(self) -> parse_settings_xml.SettingsXmlInfo:
        return parse_settings_xml.settings_xml_info_from_path(self.settings_xml_path)

    @functools.cached_property
    def settings_xml_file_record(self) -> npc_lims.File:
        return npc_lims.File(
            session_id=self.id,
            name="openephys-settings",
            suffix=".xml",
            timestamp=self.settings_xml_data.start_time.isoformat(timespec="seconds"),
            size=self.settings_xml_path.stat()["size"],
            s3_path=self.settings_xml_path.as_posix(),
            data_asset_id=self.raw_data_asset_id,
        )

    @functools.cached_property
    def devices(self) -> nwb.NWBContainerWithDF:
        return nwb.Devices(
            npc_lims.Device(
                device_id=serial_number,
                description=probe_type,
            )
            for serial_number, probe_type in zip(
                self.settings_xml_data.probe_serial_numbers,
                self.settings_xml_data.probe_types,
            )
        )

    @property
    def electrode_group_description(self) -> str:
        # TODO get correct channels range from settings xml
        return "Neuropixels 1.0 lower channels (1:384)"

    @functools.cached_property
    def probe_insertions(self) -> dict[str, Any] | None:
        path = next(
            (path for path in self.raw_data_paths if "probe_insertions" in path.stem),
            None,
        )
        if not path:
            return None
        return json.loads(path.read_text())["probe_insertions"]

    @property
    def implant(self) -> str:
        if self.probe_insertions is None:
            # TODO get from sharepoint
            return "unknown implant"
        implant: str = self.probe_insertions["implant"]
        return "2002" if "2002" in implant else implant

    @functools.cached_property
    def electrode_groups(self) -> nwb.NWBContainerWithDF:
        locations = (
            {
                v["letter"]: f"{self.implant} {v['hole']}"
                for k, v in self.probe_insertions.items()
                if k.startswith("probe") and "hole" in v and "letter" in v
            }
            if self.probe_insertions
            else {}
        )
        return nwb.ElectrodeGroups(
            npc_lims.ElectrodeGroup(
                session_id=self.id,
                device=serial_number,
                name=f"probe{probe_letter}",  # type: ignore
                description=probe_type,
                location=locations.get(probe_letter, None),
            )
            for serial_number, probe_type, probe_letter in zip(
                self.settings_xml_data.probe_serial_numbers,
                self.settings_xml_data.probe_types,
                self.settings_xml_data.probe_letters,
            )
        )

    def get_intervals(self) -> tuple[nwb.Intervals, ...]:
        return ()

    @functools.cached_property
    def intervals(self) -> dict[str, nwb.NWBContainerWithDF]:
        return {interval.name: interval for interval in self.get_intervals()}

    @property
    def trials(self) -> nwb.NWBContainerWithDF:
        trials = self.intervals.get(self.trials_interval_name)
        if trials is None:
            raise ValueError(
                f"no intervals named {self.trials_interval_name} found for {self.id}"
            )
        return trials

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
