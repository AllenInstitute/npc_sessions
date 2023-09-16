from __future__ import annotations

import contextlib
import datetime
import functools
import io
import itertools
import json
import logging
import re
import uuid
import warnings
from collections.abc import Generator, Iterable, Mapping
from typing import Any, Callable, Literal

import h5py
import ndx_events
import npc_lims
import npc_lims.status.tracked_sessions as tracked_sessions
import npc_session
import numpy as np
import numpy.typing as npt
import PIL.Image
import polars as pl
import pynwb
import upath
from DynamicRoutingTask.Analysis.DynamicRoutingAnalysisUtils import DynRoutData

import npc_sessions.config as config
import npc_sessions.nwb as nwb_internal
import npc_sessions.trials as TaskControl
import npc_sessions.utils as utils

logger = logging.getLogger(__name__)


def get_sessions() -> Generator[DynamicRoutingSession, None, None]:
    """Uploaded sessions, tracked in npc_lims via `tracked_sessions.yaml`, newest
    to oldest.

    - sessions with known issues are excluded
    - session-specific config from `npc_sessions.config.session_kwargs`
      is always re-applied: modify it if you want to pass in parameters to each
      session init
    - returns a generator not because objects take long to create (data is
      loaded lazily) but because attributes are cached, so we want to avoid
      keeping references to sessions that are no longer needed

    ## getting an indexable sequence of sessions
    Just convert the output to a list or tuple, but see the note below if you intend to
    loop over this sequence to process large amounts of data:
    >>> sessions = list(get_sessions())

    ## looping over sessions
    Data is cached in each session object after fetching from the cloud, so
    looping over all sessions can use a lot of memory if all sessions are
    retained.

    ### do this
    loop over the generator so each session object is discarded after use:
    >>> nwbs = []
    >>> for session in get_sessions():           # doctest: +SKIP
    ...     nwbs.append(session.nwb)

    ### avoid this
    `sessions` will end up storing all data for all sessions in memory:
    >>> sessions = list(get_sessions())
    >>> nwbs = []
    >>> for session in sessions:                              # doctest: +SKIP
    ...     nwbs.append(session.nwb)
    """
    for session in sorted(npc_lims.tracked, key=lambda x: x.date, reverse=True):
        if session.is_uploaded and session.id not in config.session_issues:
            yield DynamicRoutingSession(
                session.id, **config.session_kwargs.get(session.id, {})
            )


class DynamicRoutingSession:
    """Class for fetching & processing raw data for a session, making
    NWB modules and an NWBFile instance available as attributes.

    >>> s = DynamicRoutingSession('670248_2023-08-03')

    # paths/raw data processing:
    >>> 'DynamicRouting1' in s.stim_tags
    True
    >>> s.sync_path
    S3Path('s3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15/behavior/20230803T120415.h5')
    >>> s.stim_paths[0]
    S3Path('s3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15/behavior/DynamicRouting1_670248_20230803_123154.hdf5')
    >>> s.ephys_timing_data[0].name, s.ephys_timing_data[0].sampling_rate, s.ephys_timing_data[0].start_time
    ('Neuropix-PXI-100.ProbeA-AP', 30000.070518634246, 20.080209634424037)

    # access nwb modules individually before compiling a whole nwb file:
    >>> s.session_start_time
    datetime.datetime(2023, 8, 3, 12, 4, 15, 854423)
    >>> s.subject.age
    'P166D'
    >>> s.subject.genotype
    'VGAT-ChR2-YFP/wt'
    >>> 'DynamicRouting1' in s.epoch_tags
    True
    """

    suppress_errors = False
    """If True, just compile as much as possible from available stim files,
    ignoring non-critical errors."""

    # pass any of these properties to init to set
    # NWB metadata -------------------------------------------------------------- #
    experimenter: str | None = None
    experiment_description: str = "visual-auditory task-switching behavior session"
    institution: str | None = (
        "Neural Circuits & Behavior | MindScope program | Allen Institute"
    )
    notes: str | None = None

    # --------------------------------------------------------------------------- #

    task_stim_name: str = "DynamicRouting1"

    intervals_descriptions = {
        TaskControl.VisRFMapping: "visual receptive-field mapping trials",
        TaskControl.AudRFMapping: "auditory receptive-field mapping trials",
        TaskControl.DynamicRouting1: "visual-auditory task-switching behavior trials",  # must be "trials" if assigned as main trials table in nwb
        TaskControl.OptoTagging: "opto-tagging trials",
    }

    root_path: upath.UPath | None = None
    """Assigned on init if session_or_path is a pathlike object.
    May also be assigned later when looking for raw data if Code Ocean upload is missing.
    """

    def __init__(self, session_or_path: str | utils.PathLike, **kwargs) -> None:
        self.id = npc_session.SessionRecord(str(session_or_path))
        if (
            any(
                char in (path := utils.from_pathlike(session_or_path)).as_posix()
                for char in "\\/."
            )
            and path.exists()
        ):
            if path.is_dir():
                self.root_path = path
            if path.is_file():
                self.root_path = path.parent
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError:
                setattr(self, f"_{key}", value)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.id!r})"

    def __getattribute__(self, __name: str) -> Any:
        if __name in ("date", "idx"):
            return self.id.__getattribute__(__name)
        return super().__getattribute__(__name)

    def __eq__(self, other: Any) -> bool:
        if other_id := getattr(other, "id", None):
            return self.id == other_id
        return self.id == str(other)

    def __hash__(self) -> int:
        return hash(self.id)

    @property
    def nwb(self) -> pynwb.NWBFile:
        # if self._nwb_hdf5_path:
        #     self.nwb = pynwb.NWBHDF5IO(self._nwb_hdf5_path, "r").read()
        return pynwb.NWBFile(
            session_id=self.id,
            session_description=self.experiment_description,
            identifier=self.identifier,
            session_start_time=self.session_start_time.astimezone(),
            experimenter=self.experimenter,
            lab=self.lab,
            notes=self.notes,
            source_script=self.source_script,
            subject=self.subject,
            keywords=self.keywords,
            epochs=self.epochs,
            epoch_tags=self.epoch_tags,
            stimulus_template=None,  # TODO pass tuple of stimulus templates
            trials=self.trials
            if self.is_task
            else None,  # we have one sessions without trials (670248_2023-08-02)
            intervals=self._intervals,
            acquisition=self._acquisition,
            processing=self._processing,
            analysis=self._analysis,
            devices=self._devices,
            electrode_groups=self._electrode_groups if self.is_ephys else None,
            electrodes=self.electrodes if self.is_ephys else None,
            units=self.units
            if self._is_units
            else None,  # should use is_sorted when processing is done on-demand
        )

    # metadata ------------------------------------------------------------------ #

    @property
    def session_start_time(self) -> datetime.datetime:
        if self.is_sync:
            return self.sync_data.start_time
        start_time = self.epochs[:]["start_time"].min()
        start_time = (
            start_time.decode() if isinstance(start_time, bytes) else start_time
        )
        return npc_session.DatetimeRecord(f"{self.id.date} {start_time}").dt

    @property
    def _session_start_time(self) -> npc_session.DatetimeRecord:
        return npc_session.DatetimeRecord(self.session_start_time.isoformat())

    @property
    def source_script(self) -> str | None:
        if self.is_task and (script := self.task_data.get("githubTaskScript", None)):
            if isinstance(script[()], bytes):
                return script.asstr()[()]
            if isinstance(script[()], np.floating) and not np.isnan(script[()]):
                return str(script[()])
        return None

    @property
    def lab(self) -> str | None:
        with contextlib.suppress(AttributeError):
            return self.rig
        return None

    @property
    def identifier(self) -> str:
        if getattr(self, "_identifier", None) is None:
            self._identifier = str(uuid.uuid4())
        return self._identifier

    @property
    def keywords(self) -> list[str]:
        if getattr(self, "_keywords", None) is None:
            self._keywords: list[str] = []
            self.keywords.append("behavior")
            if self.is_sync:
                self.keywords.append("sync")
            if not self.is_task:
                self.keywords.append("no trials")
            if self.is_video:
                self.keywords.append("video")
            if self.is_ephys:
                self.keywords.append("ephys")
            if (
                not self._is_units
            ):  # TODO switch back to `is_sorted` when processing is done on-demand
                self.keywords.append("no units")
            if self.is_annotated:
                self.keywords.append("CCF")
            if self.is_opto:
                self.keywords.append("opto")
        return self._keywords

    @keywords.setter
    def keywords(self, value: Iterable[str]) -> None:
        keywords = getattr(self, "_keywords", [])
        keywords += list(value)
        self._keywords = list(set(keywords))

    @functools.cached_property
    def subject(self) -> pynwb.file.Subject:
        try:
            metadata = self._subject_aind_metadata
        except FileNotFoundError:
            warnings.warn(
                "Could not find subject.json metadata in raw upload: information will be limited"
            )
            return pynwb.file.Subject(
                subject_id=self.id.subject,
            )
        assert metadata["subject_id"] == self.id.subject
        dob = npc_session.DatetimeRecord(metadata["date_of_birth"])
        return pynwb.file.Subject(
            subject_id=metadata["subject_id"],
            species="Mus musculus",
            sex=metadata["sex"][0].upper(),
            date_of_birth=dob.dt.astimezone(),
            genotype=metadata["genotype"],
            description=None,
            strain=metadata["background_strain"] or metadata["breeding_group"],
            age=f"P{(self.session_start_time - dob.dt).days}D",
        )

    @property
    def epoch_tags(self) -> list[str]:
        if getattr(self, "_epoch_tags", None) is None:
            self._epoch_tags: list[str] = list(set(self.epochs.tags))
        return self._epoch_tags

    @epoch_tags.setter
    def epoch_tags(self, value: Iterable[str]) -> None:
        epoch_tags = getattr(self, "_epoch_tags", [])
        epoch_tags += list(value)
        self._epoch_tags = list(set(epoch_tags))

    # LabelledDicts ------------------------------------------------------------- #

    @property
    def acquisition(
        self,
    ) -> pynwb.core.LabelledDict[
        str, pynwb.core.NWBDataInterface | pynwb.core.DynamicTable
    ]:
        """Raw data, as acquired - filtered data goes in `processing`.

        The property as it appears on an NWBFile"""
        acquisition = pynwb.core.LabelledDict(label="acquisition", key_attr="name")
        for module in self._acquisition:
            acquisition[module.name] = module
        return acquisition

    @property
    def processing(
        self,
    ) -> pynwb.core.LabelledDict[
        str, pynwb.core.NWBDataInterface | pynwb.core.DynamicTable
    ]:
        """Data after processing and filtering - raw data goes in
        `acquisition`.

        The property as it appears on an NWBFile"""
        processing = pynwb.core.LabelledDict(label="processing", key_attr="name")
        for module in self._processing:
            processing[module.name] = module
        return processing

    @property
    def analysis(
        self,
    ) -> pynwb.core.LabelledDict[
        str, pynwb.core.NWBDataInterface | pynwb.core.DynamicTable
    ]:
        """Derived data that would take time to re-compute.

        The property as it appears on an NWBFile"""
        analysis = pynwb.core.LabelledDict(label="analysis", key_attr="name")
        for module in self._analysis:
            analysis[module.name] = module
        return analysis

    @functools.cached_property
    def _acquisition(
        self,
    ) -> tuple[pynwb.core.NWBDataInterface | pynwb.core.DynamicTable, ...]:
        """The version passed to NWBFile.__init__"""
        modules = []
        if self.is_sync:
            modules.append(self._licks.as_nwb())
        if self.is_video:
            modules.extend(self._video_frame_times)
        modules.append(self._rewards)
        return tuple(modules)

    @functools.cached_property
    def _processing(
        self,
    ) -> tuple[pynwb.core.NWBDataInterface | pynwb.core.DynamicTable, ...]:
        """The version passed to NWBFile.__init__"""
        # TODO add LFP
        modules = []
        modules.append(self._running.as_nwb())
        return tuple(modules)

    @functools.cached_property
    def _analysis(
        self,
    ) -> tuple[pynwb.core.NWBDataInterface | pynwb.core.DynamicTable, ...]:
        """The version passed to NWBFile.__init__"""
        # TODO add RF maps
        modules = []
        modules.append(self.drift_maps)
        return tuple(modules)

    # intervals ----------------------------------------------------------------- #

    @property
    def trials(self) -> pynwb.epoch.TimeIntervals | None:
        if (
            cached := getattr(self, "_cached_nwb_trials", None)
        ) is not None and cached != -1:
            return cached
        try:
            _ = self._trials
        except ValueError as exc:
            if self.suppress_errors:
                if (cached := getattr(self, "_cached_nwb_trials", None)) == -1:
                    return None  # avoid repeating the same message
                logger.warning(f"Couldn't create trials table for {self.id}: {exc!r}")
                self._cached_nwb_trials = -1
                return None
            if self.id == "670248_20230802":
                raise ValueError(
                    "DynamicRouting1*.hdf5 was recorded badly for 670248_20230802 and won't open.\nIf you wish to compile an nwb anyway, set `session.suppress_errors = True` for this session and re-run"
                )
            raise exc

        trials = pynwb.epoch.TimeIntervals(
            name="trials",
            description=self.intervals_descriptions[self._trials.__class__],
        )
        for column in self._trials.to_add_trial_column():
            trials.add_column(**column)
        for trial in self._trials.to_add_trial():
            trials.add_interval(**trial)
        self._cached_nwb_trials = trials
        return trials

    @functools.cached_property
    def _trials(self) -> TaskControl.DynamicRouting1:
        """Main behavior task trials"""
        stim_name = next(
            (_ for _ in self.stim_paths if self.task_stim_name in _.stem), None
        )
        if stim_name is None:
            raise ValueError(
                f"no stim named {self.task_stim_name}* found for {self.id}"
            )
        # avoid iterating over values and checking for type, as this will
        # create all intervals in lazydict if they don't exist
        if stim_name.stem not in self._all_trials.keys():
            raise IndexError(
                f"no intervals named {self.task_stim_name}* found for {self.id}"
            )

        trials = self._all_trials[stim_name.stem]
        assert isinstance(trials, TaskControl.DynamicRouting1)  # for mypy
        return trials

    @functools.cached_property
    def intervals(self) -> pynwb.core.LabelledDict:
        """AKA trials tables other than the main behavior task.

        The property as it appears on an NWBFile.
        """
        intervals = pynwb.core.LabelledDict(
            label="intervals",
            key_attr="name",
        )
        for module in self._intervals:
            intervals[module.name] = module
        return intervals

    @functools.cached_property
    def _intervals(self) -> tuple[pynwb.epoch.TimeIntervals, ...]:
        """The version passed to NWBFile.__init__"""
        intervals: list[pynwb.epoch.TimeIntervals] = []
        for k, v in self._all_trials.items():
            # if self._trials_interval_name in k:
            #     continue

            if not any(
                existing := [i for i in intervals if i.name == v.__class__.__name__]
            ):
                nwb_intervals = pynwb.epoch.TimeIntervals(
                    name=v.__class__.__name__,
                    description=self.intervals_descriptions[v.__class__],
                )
                trial_idx_offset = 0
                for column in v.to_add_trial_column():
                    nwb_intervals.add_column(**column)
            else:
                nwb_intervals = existing[0]
                trial_idx_offset = nwb_intervals[:]["trial_index"].max() + 1
                assert (a := set(nwb_intervals.colnames)) == (
                    b := set(v.keys())
                ), f"columns don't match for {k} and existing {nwb_intervals.name} intervals: {a.symmetric_difference(b) = }"
            for trial in v.to_add_trial():
                nwb_intervals.add_interval(**trial)
            if trial_idx_offset == 0:
                intervals.append(nwb_intervals)
        # TODO deal with stimuli across epochs
        #! requires stimulus param hashes or links to stimulus table, to
        # identify unique stims across stim files
        return tuple(intervals)

    @functools.cached_property
    def _all_trials(self) -> utils.LazyDict[str, TaskControl.TaskControl]:
        if self.is_sync:
            sync = self.sync_data
            # get the only stims for which we have times:
            stim_paths = tuple(
                path
                for path in self.stim_paths
                if path.stem in self.stim_frame_times.keys()
            )
        else:
            sync = None
            stim_paths = self.stim_paths

        def get_intervals(
            cls: type[TaskControl.TaskControl], stim_filename: str, *args
        ) -> TaskControl.TaskControl:
            return cls(self.stim_data[stim_filename], sync, *args)

        filename_to_args: dict[
            str, tuple[Callable, type[TaskControl.TaskControl], str]
        ] = {}
        for stim_path in stim_paths:
            assert isinstance(stim_path, upath.UPath)
            stim_filename = stim_path.stem
            if "RFMapping" in stim_filename:
                filename_to_args["Aud" + stim_filename] = (
                    get_intervals,
                    TaskControl.AudRFMapping,
                    stim_filename,
                )
                filename_to_args["Vis" + stim_filename] = (
                    get_intervals,
                    TaskControl.VisRFMapping,
                    stim_filename,
                )
            else:
                try:
                    cls: type[TaskControl.TaskControl] = getattr(
                        TaskControl, stim_filename.split("_")[0]
                    )
                except AttributeError:
                    continue  # some stims (e.g. Spontaneous) have no trials class
                # TODO feed-in ephys recording dirs for alignment
                # for DynamicRouting1 and AudRFMapping
                filename_to_args[stim_filename] = (get_intervals, cls, stim_filename)
        return utils.LazyDict((k, v) for k, v in filename_to_args.items())

    @functools.cached_property
    def epochs(self) -> pynwb.file.TimeIntervals:
        epochs = pynwb.file.TimeIntervals(
            name="epochs",
            description="time intervals corresponding to different phases of the session",
        )
        epochs.add_column(
            "notes",
            description="notes about the experiment or the data collected during the epoch",
        )
        for stim in self.stim_paths:
            if stim.stem not in self.stim_frame_times.keys():
                continue
            epochs.add_interval(
                **self.get_epoch_record(stim).nwb,
            )
        return epochs

    # probes, devices, units ---------------------------------------------------- #

    @functools.cached_property
    def _probes(self) -> tuple[pynwb.device.Device, ...]:
        if not self.is_ephys:
            raise AttributeError(f"{self.id} is not an ephys session")
        return tuple(
            pynwb.device.Device(
                name=str(serial_number),
                description=probe_type,
                manufacturer="imec",
            )
            for serial_number, probe_type in zip(
                self.ephys_settings_xml_data.probe_serial_numbers,
                self.ephys_settings_xml_data.probe_types,
            )
        )

    @property
    def _devices(self) -> tuple[pynwb.device.Device, ...]:
        """The version passed to NWBFile.__init__"""
        return tuple(itertools.chain(self._probes))  # add other devices as we need them

    @functools.cached_property
    def devices(self) -> pynwb.core.LabelledDict[str, pynwb.device.Device]:
        """Currently just probe model + serial number.

        Could include other devices: laser, monitor, etc.

        The property as it appears on an NWBFile"""
        devices = pynwb.core.LabelledDict(label="devices", key_attr="name")
        for module in self._devices:
            devices[module.name] = module
        return devices

    @functools.cached_property
    def electrode_groups(self) -> pynwb.core.LabelledDict[str, pynwb.device.Device]:
        """The group of channels on each inserted probe.

        The property as it appears on an NWBFile"""
        if not self.is_ephys:
            raise AttributeError(f"{self.id} is not an ephys session")
        electrode_groups = pynwb.core.LabelledDict(
            label="electrode_groups", key_attr="name"
        )
        for module in self._electrode_groups:
            electrode_groups[module.name] = module
        return electrode_groups

    @functools.cached_property
    def _electrode_groups(self) -> tuple[pynwb.ecephys.ElectrodeGroup, ...]:
        """The version passed to NWBFile.__init__"""
        if not self.is_ephys:
            raise AttributeError(f"{self.id} is not an ephys session")
        locations = (
            {
                v["letter"]: f"{self.implant} {v['hole']}"
                for k, v in self.probe_insertions.items()
                if k.startswith("probe") and "hole" in v and "letter" in v
            }
            if self.probe_insertions
            else {}
        )  # TODO upload probe insertion records for all sessions
        return tuple(
            pynwb.ecephys.ElectrodeGroup(
                name=f"probe{probe_letter}",
                device=self.devices[str(serial_number)],
                description=probe_type,
                location=locations.get(probe_letter, probe_letter),
            )
            for serial_number, probe_type, probe_letter in zip(
                self.ephys_settings_xml_data.probe_serial_numbers,
                self.ephys_settings_xml_data.probe_types,
                self.ephys_settings_xml_data.probe_letters,
            )
        )

    @functools.cached_property
    def electrodes(self) -> pynwb.core.DynamicTable:
        """Individual channels on an inserted probe, including location, CCF coords."""
        if not self.is_ephys:
            raise AttributeError(f"{self.id} is not an ephys session")
        electrodes = pynwb.file.ElectrodeTable()
        for column in (
            "rel_x",
            "rel_y",
            "channel",  # "id", # "x", "y", "z",
        ):
            electrodes.add_column(name=column, description="")
        for probe_letter, channel_pos_xy in zip(
            self.ephys_settings_xml_data.probe_letters,
            self.ephys_settings_xml_data.channel_pos_xy,
        ):
            group = self.electrode_groups[f"probe{probe_letter}"]
            for channel_label, (x, y) in channel_pos_xy.items():
                channel_idx = int(channel_label.strip("CH"))
                electrodes.add_row(
                    # x=,
                    # y=,
                    # z=,
                    # TODO: get ccf coordinates
                    location="unknown",
                    group=group,
                    group_name=group.name,
                    rel_x=x,
                    rel_y=y,
                    channel=channel_idx,
                )
        return electrodes

    @functools.cached_property
    def _units(self) -> pl.DataFrame:
        if not self.is_sorted:
            raise AttributeError(f"{self.id} hasn't been spike-sorted")
        return utils.get_units_electrodes_spike_times(self.id)

    @functools.cached_property
    def units(self) -> pynwb.misc.Units:
        units = pynwb.misc.Units(
            name="units",
            description="spike-sorted units from Kilosort 2.5",
            waveform_rate=30_000.0,
            waveform_unit="microvolts",
            electrode_table=self.electrodes,
        )
        for column in self._units.columns:
            if column in ("spike_times",):
                continue
            units.add_column(name=column, description="")
        df = self._units.fill_null(np.nan)
        for unit in df.iter_rows(named=True):
            ## for ref:
            # add_unit(spike_times=None, obs_intervals=None, electrodes=None, electrode_group=None, waveform_mean=None, waveform_sd=None, waveforms=None, id=None)
            units.add_unit(
                **unit,  # contains spike_times
                electrodes=[
                    self.electrodes[:]
                    .query(f"channel == {unit['peak_channel']}")
                    .query(f"group_name == {unit['device_name']!r}")
                    .index.item()
                ],
                electrode_group=self.electrode_groups[unit["device_name"]],
            )
        return units

    # images -------------------------------------------------------------------- #

    @functools.cached_property
    def drift_maps(self) -> pynwb.image.Images:
        return pynwb.image.Images(
            name="drift_maps",
            images=tuple(self.img_to_nwb(p) for p in self.drift_map_paths),
            description="activity plots (time x probe depth x firing rate) over the entire ecephys recording, for assessing probe drift",
        )

    @staticmethod
    def img_to_nwb(path: utils.PathLike) -> pynwb.image.Image:
        path = utils.from_pathlike(path)
        img = PIL.Image.open(io.BytesIO(path.read_bytes()))
        mode_to_nwb_cls = {
            "L": pynwb.image.GrayscaleImage,
            "RGB": pynwb.image.RGBImage,
            "RGBA": pynwb.image.RGBAImage,
        }
        return mode_to_nwb_cls[img.mode](
            name=path.stem,
            data=np.array(img.convert(img.mode)),
        )

    # session ------------------------------------------------------------------- #

    @functools.cached_property
    def info(self) -> tracked_sessions.SessionInfo | None:
        return next(
            (info for info in npc_lims.tracked if info.id == self.id),
            None,
        )

    @functools.cached_property
    def is_task(self) -> bool:
        if v := getattr(self, "_is_task", None):
            return v
        with contextlib.suppress(FileNotFoundError, ValueError, StopIteration):
            _ = self.task_data
            return True
        return False

    @functools.cached_property
    def is_sync(self) -> bool:
        if v := getattr(self, "_is_sync", None):
            return v
        if v := getattr(self, "_sync_path", None):
            return True
        if self.info:
            return self.info.is_sync
        with contextlib.suppress(FileNotFoundError, ValueError):
            if self.get_sync_paths():
                return True
        return False

    @functools.cached_property
    def is_video(self) -> bool:
        if v := getattr(self, "_is_video", None):
            return v
        if not self.is_sync:
            return False
        with contextlib.suppress(FileNotFoundError, ValueError):
            if self.video_paths:
                return True
        return False

    @functools.cached_property
    def is_ephys(self) -> bool:
        if v := getattr(self, "_is_ephys", None):
            return v
        if self.info:
            return self.info.is_ephys
        with contextlib.suppress(FileNotFoundError, ValueError):
            if self.ephys_record_node_dirs:
                return True
        return False

    @functools.cached_property
    def is_sorted(self) -> bool:
        if v := getattr(self, "_is_sorted", None):
            return v
        if not self.is_ephys:
            return False
        if self.info:
            return self.info.is_sorted
        with contextlib.suppress(FileNotFoundError, ValueError):
            _ = npc_lims.is_sorted_data_asset(self.id)
            return True
        return False

    @functools.cached_property
    def _is_units(self) -> bool:
        """Temp attr to check if arjun's processed units files are available"""
        if not self.is_sorted:
            return False
        with contextlib.suppress(FileNotFoundError, ValueError):
            _ = npc_lims.get_units_codeoean_kilosort_path_from_s3(self.id)
            return True
        return False

    @functools.cached_property
    def is_annotated(self) -> bool:
        """CCF annotation data accessible"""
        if not self.is_ephys:
            return False
        with contextlib.suppress(FileNotFoundError, ValueError):
            if utils.get_electrode_files_from_s3(self.id):
                return True
        return False

    @functools.cached_property
    def is_opto(self) -> bool:
        """Opto during behavior task && not wt/wt (if genotype info available)"""
        genotype: str | None = (
            self.subject.genotype
        )  # won't exist if subject.json not found
        if self.trials is None:
            return False
        if self._trials._has_opto and (
            genotype is None or "wt/wt" not in genotype.lower()
        ):
            if genotype is None:
                logger.warning(
                    f"Could not find genotype for {self.id}: returning is_opto = True regardless"
                )
            return True
        return False

    def get_record(self) -> npc_lims.Session:
        return npc_lims.Session(
            session_id=self.id,
            subject_id=self.id.subject,
            session_start_time=self._session_start_time,
            stimulus_notes=self.task_version,
            experimenter=self.experimenter,
            experiment_description=self.experiment_description,
            # epoch_tags=list(self.epoch_tags),
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

    @functools.cached_property
    def _raw_upload_metadata_json_paths(self):
        return tuple(
            file
            for file in npc_lims.get_raw_data_root(self.id).iterdir()
            if file.suffix == ".json"
        )

    @functools.cached_property
    def sorting_vis(self) -> dict[str, dict | str]:
        """To open links:
        import webbrowser
        webbrowser.open(_)
        """
        if not self.is_ephys:
            raise AttributeError(f"{self.id} is not an ephys session")
        if not self.is_sorted:
            raise AttributeError(f"{self.id} has not been sorted")
        path = next(
            p for p in self.sorted_data_paths if p.name == "visualization_output.json"
        )
        return json.loads(path.read_text())

    @property
    def _nwb_hdf5_path(self) -> upath.UPath | None:
        return npc_lims.get_nwb_file_from_s3(self.id)

    @functools.cached_property
    def _subject_aind_metadata(self) -> dict[str, Any]:
        try:
            file = next(
                f
                for f in self._raw_upload_metadata_json_paths
                if f.name == "subject.json"
            )
        except StopIteration as exc:
            raise FileNotFoundError(
                f"Could not find subject.json for {self.id} in {self._raw_upload_metadata_json_paths}"
            ) from exc
        return json.loads(file.read_text())

    @property
    def stim_tags(self) -> tuple[str, ...]:
        """Currently assumes TaskControl hdf5 files"""
        return tuple(
            name.split("_")[0]
            for name in sorted(
                [p.name for p in self.stim_paths], key=npc_session.DatetimeRecord
            )
        )

    def get_raw_data_paths_from_root(self) -> tuple[upath.UPath, ...]:
        if not self.root_path:
            raise ValueError(f"{self.id} does not have a local path assigned yet")
        ephys_paths = itertools.chain(
            self.root_path.glob("Record Node *"),
            self.root_path.glob("*/Record Node *"),
        )
        root_level_paths = tuple(p for p in self.root_path.iterdir() if p.is_file())
        return root_level_paths + tuple(set(ephys_paths))

    def get_task_hdf5_from_s3_repo(self) -> upath.UPath:
        try:
            return next(
                (npc_lims.DR_DATA_REPO / str(self.id.subject)).glob(
                    f'{self.task_stim_name}*{self.id.date.replace("-", "")}*.hdf5'
                )
            )
        except StopIteration:
            raise FileNotFoundError(
                f"Could not find file in {npc_lims.DR_DATA_REPO} for {self.id}"
            ) from None

    @functools.cached_property
    def raw_data_paths(self) -> tuple[upath.UPath, ...]:
        if self.root_path:
            return self.get_raw_data_paths_from_root()
        with contextlib.suppress(FileNotFoundError, ValueError):
            return npc_lims.get_raw_data_paths_from_s3(self.id)
        with contextlib.suppress(StopIteration):
            stim = self.get_task_hdf5_from_s3_repo()
            logger.warning(f"Using {stim.name} in {npc_lims.DR_DATA_REPO}")
            self.root_path = stim.parent
            return self.get_raw_data_paths_from_root()
        raise ValueError(
            f"{self.id} is either an untracked ephys session with no Code Ocean upload, or a behavior session with no data in the synced s3 repo {npc_lims.DR_DATA_REPO}"
        )

    @functools.cached_property
    def sorted_data_paths(self) -> tuple[upath.UPath, ...]:
        if not self.is_ephys:
            raise ValueError(f"{self.id} is not a session with ephys")
        return npc_lims.get_sorted_data_paths_from_s3(self.id)

    @functools.cached_property
    def sync_path(self) -> upath.UPath:
        if path := getattr(self, "_sync_path", None):
            return path
        if not self.is_sync:
            raise ValueError(f"{self.id} is not a session with sync data")
        paths = self.get_sync_paths()
        if not len(paths) == 1:
            raise ValueError(f"Expected 1 sync file, found {paths = }")
        self._sync_path = paths[0]
        return self._sync_path

    def get_sync_paths(self) -> tuple[upath.UPath, ...]:
        return tuple(
            p
            for p in self.raw_data_paths
            if p.suffix == ".sync"
            or (
                p.suffix in (".h5",)
                and p.stem.startswith(f"{self.id.date.replace('-', '')}T")
            )
        )

    @functools.cached_property
    def raw_data_asset_id(self) -> str:
        if not self.is_ephys:
            raise ValueError(
                f"{self.id} currently only ephys sessions have raw data assets"
            )
        return npc_lims.get_session_raw_data_asset(self.id)["id"]

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
    def sync_data(self) -> utils.SyncDataset:
        return utils.SyncDataset(io.BytesIO(self.sync_path.read_bytes()))

    @property
    def stim_path_root(self) -> upath.UPath:
        return npc_lims.DR_DATA_REPO / str(self.id.subject)

    @functools.cached_property
    def stim_paths(self) -> tuple[upath.UPath, ...]:
        def is_valid_stim_file(p) -> bool:
            return utils.is_stim_file(
                p, subject_spec=self.id.subject, date_spec=self.id.date
            )

        if self.is_ephys:
            stim_paths = tuple(p for p in self.raw_data_paths if is_valid_stim_file(p))
            if stim_paths:
                return stim_paths
        return tuple(p for p in self.stim_path_root.iterdir() if is_valid_stim_file(p))

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
    def rig(self) -> str:
        for hdf5 in itertools.chain(
            (self.task_data,) if self.is_task else (),
            (v for v in self.stim_data.values()),
        ):
            if rig := hdf5.get("rigName", None):
                return rig.asstr()[()]
        raise AttributeError(f"Could not find rigName for {self.id} in stim files")

    @property
    def sam(self) -> DynRoutData:
        if not hasattr(self, "_sam"):
            obj = DynRoutData()
            obj.loadBehavData(
                self.task_path.as_posix(),
                self.task_data,
            )
            self._sam = obj
        return self._sam

    @property
    def task_path(self) -> upath.UPath:
        return next(path for path in self.stim_paths if "DynamicRouting" in path.stem)

    @property
    def task_data(self) -> h5py.File:
        return next(self.stim_data[k] for k in self.stim_data if "DynamicRouting" in k)

    @property
    def task_version(self) -> str | None:
        return self.sam.taskVersion if isinstance(self.sam.taskVersion, str) else None

    @functools.cached_property
    def stim_data(self) -> utils.LazyDict[str, h5py.File]:
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
        return utils.get_video_file_paths(*self.raw_data_paths)

    @property
    def video_info_paths(self) -> tuple[upath.UPath, ...]:
        return utils.get_video_info_file_paths(*self.raw_data_paths)

    @functools.cached_property
    def video_info_data(self) -> Mapping[str, Any]:
        def recording_report(path: upath.UPath) -> dict[str, str | int | float]:
            return json.loads(path.read_bytes())["RecordingReport"]

        return utils.LazyDict(
            (utils.extract_camera_name(path.stem), (recording_report, path))
            for path in self.video_info_paths
        )

    @functools.cached_property
    def video_file_records(self) -> tuple[npc_lims.File, ...]:
        return tuple(
            npc_lims.File(
                session_id=self.id,
                name=utils.extract_camera_name(path.stem),
                suffix=path.suffix,
                timestamp=npc_session.TimeRecord.parse_id(
                    self.video_info_data[utils.extract_camera_name(path.stem)][
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
                name=utils.extract_camera_name(path.stem),
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

    @functools.cached_property
    def _stim_frame_times(self) -> dict[str, Exception | npt.NDArray[np.float64]]:
        """Frame times dict for all stims, containing time arrays or Exceptions."""
        frame_times = utils.get_stim_frame_times(
            *self.stim_data.values(),  # use cached data
            sync=self.sync_data,
            frame_time_type="display_time",
        )

        def reverse_lookup(d, value) -> str:
            return next(str(k) for k, v in d.items() if v is value)

        return {reverse_lookup(self.stim_data, k): frame_times[k] for k in frame_times}

    @property
    def stim_frame_times(self) -> dict[str, npt.NDArray[np.float64]]:
        """Frame times dict for stims with time arrays, or optionally raising
        exceptions."""
        if self.suppress_errors:
            return {
                path: times
                for path, times in self._stim_frame_times.items()
                if not isinstance(times, Exception)
            }
        asserted_stim_frame_times: dict[str, npt.NDArray[np.float64]] = {}
        for k, v in self._stim_frame_times.items():
            v = utils.assert_stim_times(v)
            asserted_stim_frame_times[k] = v
        assert not any(
            isinstance(v, Exception) for v in asserted_stim_frame_times.values()
        )
        return asserted_stim_frame_times

    def get_epoch_record(
        self, stim_file: utils.PathLike, sync: utils.SyncPathOrDataset | None = None
    ) -> npc_lims.Epoch:
        stim_file = utils.from_pathlike(stim_file)
        h5 = self.stim_data[stim_file.stem]
        tags = []
        tags.append(stim_file.stem.split("_")[0])
        if any(label in h5 for label in ("optoRegions", "optoParams")):
            tags.append("opto")
        if any(h5["rewardFrames"][:]):
            tags.append("rewards")

        if sync:
            sync = utils.get_sync_data(sync)
        elif self.is_sync:
            sync = self.sync_data

        if sync is None:
            start_time = 0.0
            stop_time = utils.get_stim_duration(h5)
        else:
            frame_times = self.stim_frame_times[stim_file.stem]
            start_time = frame_times[0]
            stop_time = frame_times[-1]

        assert start_time != stop_time
        return npc_lims.Epoch(
            session_id=self.id,
            start_time=start_time,
            stop_time=stop_time,
            tags=tags,
        )

    @property
    def ephys_record_node_dirs(self) -> tuple[upath.UPath, ...]:
        return tuple(
            p for p in self.raw_data_paths if re.match(r"^Record Node [0-9]+$", p.name)
        )

    @functools.cached_property
    def ephys_recording_dirs(self) -> tuple[upath.UPath, ...]:
        return tuple(
            p
            for record_node in self.ephys_record_node_dirs
            for p in record_node.glob("experiment*/recording*")
        )

    @functools.cached_property
    def ephys_timing_data(self) -> tuple[utils.EphysTimingInfoOnSync, ...]:
        return tuple(
            itertools.chain(
                utils.get_ephys_timing_on_sync(
                    self.sync_data, self.ephys_recording_dirs
                )
            )
        )

    @functools.cached_property
    def drift_map_paths(self) -> tuple[upath.UPath, ...]:
        return tuple(
            next(d for d in self.sorted_data_paths if d.name == "drift_maps").iterdir()
        )

    @functools.cached_property
    def ephys_sync_messages_path(self) -> upath.UPath:
        return next(
            p
            for p in itertools.chain(
                *(record_node.iterdir() for record_node in self.ephys_recording_dirs)
            )
            if "sync_messages.txt" == p.name
        )

    @functools.cached_property
    def ephys_structure_oebin_paths(self) -> tuple[upath.UPath, ...]:
        return tuple(
            p
            for p in itertools.chain(
                *(record_node.iterdir() for record_node in self.ephys_recording_dirs)
            )
            if "structure.oebin" == p.name
        )

    @functools.cached_property
    def ephys_structure_oebin_data(
        self,
    ) -> dict[Literal["continuous", "events", "spikes"], list[dict[str, Any]]]:
        return utils.get_merged_oebin_file(self.ephys_structure_oebin_paths)

    @functools.cached_property
    def ephys_sync_messages_data(
        self,
    ) -> dict[str, dict[Literal["start", "rate"], int]]:
        return utils.get_sync_messages_data(self.ephys_sync_messages_path)

    @functools.cached_property
    def ephys_experiment_dirs(self) -> tuple[upath.UPath, ...]:
        return tuple(
            p
            for record_node in self.ephys_record_node_dirs
            for p in record_node.glob("experiment*")
        )

    @functools.cached_property
    def ephys_settings_xml_paths(self) -> tuple[upath.UPath, ...]:
        if not self.is_ephys:
            raise ValueError(
                f"{self.id} is not an ephys session (required for settings.xml)"
            )
        return tuple(
            next(record_node.glob("settings*.xml"))
            for record_node in self.ephys_record_node_dirs
        )

    @functools.cached_property
    def ephys_settings_xml_path(self) -> upath.UPath:
        """Single settings.xml path, if applicable"""
        if not self.ephys_settings_xml_paths:
            raise ValueError(
                f"settings.xml not found for {self.id} - check status of raw upload"
            )
        utils.assert_xml_files_match(*self.ephys_settings_xml_paths)
        return self.ephys_settings_xml_paths[0]

    @functools.cached_property
    def ephys_settings_xml_data(self) -> utils.SettingsXmlInfo:
        return utils.get_settings_xml_info(self.ephys_settings_xml_path)

    @functools.cached_property
    def ephys_settings_xml_file_record(self) -> npc_lims.File:
        return npc_lims.File(
            session_id=self.id,
            name="openephys-settings",
            suffix=".xml",
            timestamp=self.ephys_settings_xml_data.start_time.isoformat(
                timespec="seconds"
            ),
            size=self.ephys_settings_xml_path.stat()["size"],
            s3_path=self.ephys_settings_xml_path.as_posix(),
            data_asset_id=self.raw_data_asset_id,
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
    def _licks(self) -> nwb_internal.SupportsAsNWB:
        return nwb_internal.LickSpout(self.sync_data)

    @functools.cached_property
    def _running(self) -> nwb_internal.SupportsAsNWB:
        return nwb_internal.RunningSpeed(
            *self.stim_data.values(), sync=self.sync_data if self.is_sync else None
        )

    @functools.cached_property
    def _video_frame_times(
        self,
    ) -> tuple[pynwb.core.NWBDataInterface | pynwb.core.DynamicTable, ...]:
        # currently doesn't require opening videos
        path_to_timestamps = utils.get_video_frame_times(
            self.sync_data, *self.video_paths
        )
        return tuple(
            ndx_events.Events(
                timestamps=timestamps,
                name=f"{utils.extract_camera_name(path.stem)}_camera",
                description=f"video frame timestamps for {path.stem}",
            )
            for path, timestamps in path_to_timestamps.items()
        )

    @functools.cached_property
    def _rewards(self) -> pynwb.core.NWBDataInterface | pynwb.core.DynamicTable:
        def get_reward_frames(data: h5py.File) -> list[int]:
            r = []
            for key in ("rewardFrames", "manualRewardFrames"):
                if v := data.get(key, None):
                    r.extend(v[:])
            return r

        reward_times: list[npt.NDArray[np.floating]] = []
        if self.is_sync:
            for stim_file, frame_times in self.stim_frame_times.items():
                if any(name in stim_file.lower() for name in ("mapping", "tagging")):
                    continue
                reward_times.extend(
                    frame_times[get_reward_frames(self.stim_data[stim_file])]
                )
        else:
            reward_times.extend(self.stim_data[self.task_path.stem])
        return ndx_events.Events(
            timestamps=np.sort(np.unique(reward_times)),
            name="rewards",
            description="individual water rewards delivered to the subject",
        )


if __name__ == "__main__":
    list(get_sessions())
    import doctest

    import dotenv

    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
