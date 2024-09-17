from __future__ import annotations

import contextlib
import copy
import datetime
import functools
import io
import itertools
import json
import logging
import re
import typing
import uuid
from collections.abc import Iterable, Iterator
from typing import Any, Literal

import aind_data_schema.core.session
import aind_data_schema.models.coordinates
import aind_data_schema.models.devices
import aind_data_schema.models.modalities
import aind_data_schema.models.stimulus
import cv2
import h5py
import hdmf
import hdmf.common
import hdmf_zarr
import ndx_events
import ndx_pose
import npc_ephys
import npc_io
import npc_lims
import npc_mvr
import npc_samstim
import npc_session
import npc_stim
import npc_sync
import numpy as np
import numpy.typing as npt
import pandas as pd
import PIL.Image
import pynwb
import upath
import zarr
from DynamicRoutingTask.Analysis.DynamicRoutingAnalysisUtils import DynRoutData

import npc_sessions.trials as TaskControl
import npc_sessions.utils as utils

logger = logging.getLogger(__name__)


@typing.overload
def get_sessions(
    id_or_ids: None = None,
    **all_session_kwargs,
) -> Iterator[DynamicRoutingSession]: ...


@typing.overload
def get_sessions(
    id_or_ids: str | npc_session.SessionRecord | npc_lims.SessionInfo,
    **all_session_kwargs,
) -> DynamicRoutingSession: ...


@typing.overload
def get_sessions(
    id_or_ids: list | set | tuple,
    **all_session_kwargs,
) -> Iterator[DynamicRoutingSession]: ...


# see overloads above for type hints
def get_sessions(id_or_ids=None, **all_session_kwargs):
    """Uploaded sessions, tracked in npc_lims via `get_session_info()`, newest
    to oldest.

    - if `id_or_ids` is provided as a single string or record, a single session object is returned
    - if `id_or_ids` is an iterable of strings or records, a generator of session
      objects is returned for those sessions
    - if `id_or_ids` is None, a generator over all sessions is returned

    - sessions with known issues are excluded

    - `all_session_kwargs` will be applied on top of any session-specific kwargs
        - session-specific config from `npc_lims.get_session_kwargs`
          is always applied
        - add extra kwargs here if you want to override or append
          parameters passed to every session __init__

    - returns a generator not because objects take long to create (data is
      loaded lazily) but because some large attributes are cached in memory, so
      we want to avoid keeping references to sessions that are no longer needed

    ## looping over sessions
    Data is cached in each session object after fetching from the cloud, so
    looping over all sessions can use a lot of memory if all sessions are
    retained.

    ### do this:
    loop over the generator so each session object is discarded after use:
    >>> nwbs = []
    >>> for session in get_sessions():                          # doctest: +SKIP
    ...     nwbs.append(session.nwb)

    ### avoid this:
    `sessions` will end up storing all data for all sessions in memory:
    >>> sessions = list(get_sessions())                         # doctest: +SKIP
    >>> nwbs = []
    >>> for session in sessions:                                # doctest: +SKIP
    ...     nwbs.append(session.nwb)

    ## using `all_session_kwargs`
    if, for example, you wanted to get trials tables for all sessions without using sync
    information for timing (the default when `session.is_sync == True`), you can set the
    property on all sessions (just for this loop) by passing it in as a kwarg:
    >>> trials_dfs = []
    >>> for session in get_sessions(is_sync=False):             # doctest: +SKIP
    ...     trials_dfs.append(session.trials)
    """
    cls = DynamicRoutingSession  # may dispatch type in future

    is_single_session: bool = id_or_ids is not None and (
        isinstance(id_or_ids, str)  # single str is iterable
        or not isinstance(id_or_ids, Iterable)
    )
    if is_single_session:
        return cls(id_or_ids, **all_session_kwargs)

    session_info_kwargs = {
        k: v for k, v in all_session_kwargs.items() if not k.startswith("_")
    }

    def multi_session_generator() -> Iterator[DynamicRoutingSession]:
        if id_or_ids is None:
            session_infos = sorted(
                npc_lims.get_session_info(**session_info_kwargs),
                key=lambda x: x.date,
                reverse=True,
            )
        else:
            session_infos = [
                npc_lims.get_session_info(id_, **session_info_kwargs)
                for id_ in id_or_ids
            ]
        for session_info in session_infos:
            if session_info.issues:
                logger.warning(
                    f"Skipping session {session_info.id} due to known issues: {session_info.issues}"
                )
                continue
            try:
                yield cls(
                    session_info.id,
                    **all_session_kwargs,
                )
            except Exception as exc:
                logger.warning(
                    f"Error instantiating {cls}({session_info.id!r}): {exc!r}"
                )
                continue

    return multi_session_generator()


class DynamicRoutingSession:
    """Class for fetching & processing raw data for a session, making
    NWB modules and an NWBFile instance available as attributes.

    >>> s = DynamicRoutingSession('670248_2023-08-03')

    # paths/raw data processing:
    >>> 'DynamicRouting1' in s.stim_names
    True
    >>> s.stim_paths[0].name
    'DynamicRouting1_670248_20230803_123154.hdf5'
    >>> s.sync_path
    S3Path('s3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15/behavior/20230803T120415.h5')
    >>> s.ephys_timing_data[0].device.name, s.ephys_timing_data[0].sampling_rate, s.ephys_timing_data[0].start_time
    ('Neuropix-PXI-100.ProbeA-AP', 30000.070472670912, 20.080209602072898)
    >>> s.sam.dprimeSameModal
    [3.5501294698425694]

    # access nwb modules individually before compiling a whole nwb file:
    >>> s.session_start_time
    datetime.datetime(2023, 8, 3, 12, 4, 15, tzinfo=zoneinfo.ZoneInfo(key='America/Los_Angeles'))
    >>> s.subject.age
    'P166D'
    >>> s.subject.genotype
    'VGAT-ChR2-YFP/wt'
    >>> 'task' in s.epoch_tags
    True
    >>> s.probe_insertions['A']
    'A2'
    """

    # pass any of these properties to init to set
    # NWB metadata -------------------------------------------------------------- #
    institution: str | None = (
        "Neural Circuits & Behavior | MindScope program | Allen Institute for Neural Dynamics"
    )

    # --------------------------------------------------------------------------- #

    task_stim_name: str = "DynamicRouting1"
    """Used to distinguish the main behavior task stim file from others"""

    excluded_stim_file_names = ["DynamicRouting1_670248_20230802_120703"]
    """File names (or substrings) that should never be considered as valid stim
    files, for example they are known to be corrupt and cannot be opened"""

    mvr_to_nwb_camera_name = {
        "eye": "eye_camera",
        "face": "front_camera",
        "behavior": "side_camera",
    }

    def __init__(
        self, session_or_path: str | npc_io.PathLike | npc_lims.SessionInfo, **kwargs
    ) -> None:
        if isinstance(session_or_path, npc_lims.SessionInfo):
            session_or_path = session_or_path.id
        self.id = npc_session.SessionRecord(str(session_or_path))

        # if a path was supplied and it exists, set it as the root data path for the session
        if any(
            char in (path := npc_io.from_pathlike(session_or_path)).as_posix()
            for char in "\\/."
        ):
            if path.is_dir():
                self._root_path: upath.UPath | None = path
            elif path.is_file():
                self._root_path = path.parent
            elif not path.exists():
                raise FileNotFoundError(f"{path} does not exist")

        # if available, get session config kwargs from the npc_lims session-tracking yaml file
        if self.info is not None:
            if issues := self.info.issues:
                logger.warning(f"Session {self.id} has known issues: {issues}")
            kwargs = copy.copy(self.info.session_kwargs) | kwargs
            if self.info.is_sync and not (self.info.is_uploaded or self.root_path):
                logger.warning(
                    f"Session {self.id} is marked as `is_sync=True` by `npc_lims`, but raw data has not been uploaded. "
                    "`is_sync` and `is_ephys` will be set to False for this session (disabling all related data)."
                )
                kwargs["is_sync"] = False
                kwargs["is_ephys"] = False

        if kwargs:
            logger.info(f"Applying session kwargs to {self.id}: {kwargs}")
        self.kwargs = kwargs
        for key, value in kwargs.items():
            if isinstance(
                getattr(self.__class__, key, None), functools.cached_property
            ):
                # avoid overwriting cached properties
                setattr(self, f"_{key}", value)
            else:
                try:
                    setattr(self, key, value)
                except AttributeError:
                    setattr(self, f"_{key}", value)

        # as a shortcut, make all plotting functions available as instance methods
        self._add_plots_as_methods()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.id!r})"

    def __eq__(self, other: Any) -> bool:
        if other_id := getattr(other, "id", None):
            return self.id == other_id
        return self.id == str(other)

    def __hash__(self) -> int:
        return hash(self.id)

    @classmethod
    def _add_plots_as_methods(cls) -> None:
        """Add plots as methods to session object, so they can be called
        directly, e.g. `session.plot_drift_maps()`.

        - requires `npc_sessions_cache` to be installed
        - looks for functions starting with `plot_`
        """
        try:
            import npc_sessions_cache.plots as plots
        except ImportError:
            logger.debug(
                "npc_sessions_cache not installed - plotting functions will not be available"
            )
            return
        for attr in (attr for attr in plots.__dict__ if attr.startswith("plot_")):
            if getattr((fn := getattr(plots, attr)), "__call__", None) is not None:
                setattr(cls, attr, fn)

    @property
    def ignore_stim_errors(self) -> bool:
        """If True, just compile as much as possible from available stim files,
        ignoring non-critical errors."""
        return getattr(self, "_suppress_errors", False) or getattr(
            self, "_ignore_stim_errors", False
        )

    @property
    def nwb_from_cache(self) -> pynwb.NWBFile | None:
        if (
            path := npc_lims.get_nwb_path(
                self.id, version=None
            )  # always return the latest
        ).exists():
            if path.suffix == ".zarr":
                return hdmf_zarr.NWBZarrIO(path=path.as_posix(), mode="r").read()
            else:
                return pynwb.NWBHDF5IO(path=path.as_posix(), mode="r").read()
        return None

    @property
    def nwb(self) -> pynwb.NWBFile:
        return pynwb.NWBFile(
            session_id=self.session_id,
            session_description=self.session_description,
            experiment_description=self.experiment_description,
            identifier=self.identifier,
            institution=self.institution,
            session_start_time=self.session_start_time,
            experimenter=self.experimenter,
            lab=self.lab,
            notes=self.notes,
            source_script=self.source_script,
            source_script_file_name=self.source_script_file_name,
            stimulus_notes=self.task_version if self.is_task else None,
            subject=self.subject,
            keywords=self.keywords,
            epochs=self.epochs,
            epoch_tags=self.epoch_tags,
            stimulus_template=None,  # TODO pass tuple of stimulus templates
            invalid_times=self.invalid_times,
            trials=(
                self.trials if self.is_task else None
            ),  # we have one session without trials (670248_2023-08-02)
            intervals=self._intervals,
            acquisition=self._acquisition,
            processing=tuple(self.processing.values()),
            analysis=self._analysis,
            devices=self._devices if self._devices else None,
            electrode_groups=self._electrode_groups if self.is_ephys else None,
            electrodes=self.electrodes if self.is_ephys else None,
            units=self.units if self.is_sorted else None,
        )

    def write_nwb(
        self,
        path: str | npc_io.PathLike | None = None,
        metadata_only: bool = False,
        zarr=True,
        force=False,
    ) -> upath.UPath:
        """Write NWB file to disk - file path is normalized and returned"""
        if path is None:
            path = npc_lims.get_nwb_path(self.id)
        else:
            path = npc_io.from_pathlike(path)
        path = path.with_stem(
            path.name.replace(".hdf5", "").replace(".nwb", "").replace(".zarr", "")
        ).with_suffix(".nwb.zarr" if zarr else ".nwb")
        if not force and path.exists() and (npc_io.get_size(path) // 1024) > 1:
            raise FileExistsError(
                f"{path} already exists - use `force=True` to overwrite"
            )
        elif force and zarr and path.exists():
            logger.warning(
                f"Overwriting zarr directories is not advised: remnants of previous data may remain.\nSuggest deleting {path} first."
            )
        nwb = self.nwb if not metadata_only else self.metadata
        if zarr:
            with hdmf_zarr.NWBZarrIO(path.as_posix(), "w") as io:
                io.write(
                    nwb, link_data=False
                )  # link_data=False so that lazily-opened zarrays are read and copied into nwb (instead of being added as a link, which is currently broken)
        else:
            with pynwb.NWBHDF5IO(path.as_posix(), "w") as io:
                io.write(nwb)
        logger.info(
            f"Saved NWB file to {path}: {npc_io.get_size(path) // 1024 ** 2} MB"
        )
        return path

    # metadata ------------------------------------------------------------------ #

    @npc_io.cached_property
    def metadata(self) -> pynwb.NWBFile:
        """NWB file with session metadata-alone"""
        return pynwb.NWBFile(
            session_id=self.session_id,
            session_description=self.session_description,
            experiment_description=self.experiment_description,
            institution=self.institution,
            identifier=self.identifier,
            session_start_time=self.session_start_time,
            experimenter=self.experimenter,
            lab=self.lab,
            notes=self.notes,
            source_script=self.source_script,
            source_script_file_name=self.source_script_file_name,
            stimulus_notes=self.task_version if self.is_task else None,
            subject=self.subject,
            keywords=self.keywords,
            epoch_tags=self.epoch_tags,
            invalid_times=self.invalid_times,
        )

    @property
    def session_id(self) -> str:
        return str(self.id)

    @property
    def session_start_time(self) -> datetime.datetime:
        if self.is_sync:
            return utils.get_aware_dt(self.sync_data.start_time)
        return utils.get_aware_dt(npc_stim.get_stim_start_time(self.task_data))

    @property
    def notes(self) -> str | None:
        notes = ""
        if self.info:
            notes += "; ".join([self.info.notes] + self.info.issues)
        return notes or None

    @property
    def exp_path(self) -> upath.UPath | None:
        """Dir with record of experiment workflow, environment lock file, logs
        etc. - may not be a dedicated subfolder if contents were moved to behavior
        root.

        - does not exist for some sessions (surface channel recordings, sessions recorded before ipynb workflow was implemented)
        """
        behavior_path = next(
            (p.parent for p in self.raw_data_paths if p.parent.name == "behavior"), None
        )
        if behavior_path is None:
            return None
        exp_path = next(behavior_path.glob("exp"), None)
        if exp_path:
            return exp_path
        if (behavior_path / "debug.log").exists():
            return behavior_path
        logger.debug(
            f"exp path not found for {self.id} - likely a session recorded before ipynb workflow was implemented"
        )
        return None

    @property
    def exp_log_path(self) -> upath.UPath | None:
        """Debug log file from experiment.

        - used to be in a subfolder: `behavior/exp/logs/debug.log`
        - directories may be flattened too: `behavior/debug.log`
        """
        exp_path = self.exp_path
        if exp_path is None:
            return None
        log_path = next(exp_path.rglob("debug.log"), None)
        return log_path

    def get_experimenter_from_experiment_log(self) -> str | None:
        """Returns lims user name, if found in the experiment log file. Otherwise,
        None.

        >>> s = DynamicRoutingSession('DRpilot_662892_20230822')
        >>> s.get_experimenter_from_experiment_log()
        'Hannah Cabasco'
        """
        if (log_path := self.exp_log_path) is None:
            return None
        text = log_path.read_text()
        matches = re.findall(r"User\(\'(.+)\'\)", text)
        if not matches:
            return None

        def _get_name(match: str) -> str | None:
            if "." in match:
                return match.replace(".", " ").title()
            return {
                "samg": "Sam Gale",
                "corbettb": "Corbett Bennett",
            }.get(match)

        return _get_name(
            matches[-1]
        )  # last user, in case it changed at the start of the session

    @property
    def experimenter(self) -> list[str] | None:
        with contextlib.suppress(FileNotFoundError, ValueError):
            if experimenter := self.get_experimenter_from_experiment_log():
                return [experimenter]
        if self.id.date.dt < datetime.date(2023, 8, 8):
            # older DR/Templeton sessions, prior to Hannah C becoming 100% DR
            return ["Jackie Kuyat"]
        elif "NP" in self.rig:
            # older DR/Templeton sessions, prior to Hannah C becoming 100% DR
            return ["Sam Gale"]
            # some sessions Sam ran with no experiment log
        return None

    @property
    def session_description(self) -> str:
        """Uses `is_` bools to construct a text description.
        - won't be correct if testing with datastreams manually disabled
        """
        opto = ", with optogenetic inactivation during task" if self.is_opto else ""
        video = ", with video recording of behavior" if self.is_video else ""
        sync = ", without precise timing information" if not self.is_sync else ""
        if not self.is_task:
            description = "session"
            if self.info and (e := self.info.experiment_day) is not None:
                description += f" (day {e})"
            description += opto
            description += video
            description += sync
            return description
        if not self.is_ephys:
            description = "training session with behavioral task data"
            if self.info and (b := self.info.behavior_day) is not None:
                description += f" (day {b})"
            description += opto
            description += video
            description += sync
            return description
        else:
            description = "ecephys session"
            if self.info and (e := self.info.experiment_day) is not None:
                description += f" (day {e})"
            description += " without sorted units," if not self.is_sorted else ""
            description += (
                " without CCF-annotated units,"
                if self.is_sorted and not self.is_annotated
                else ""
            )
            description += (
                f" {'with' if self.is_task else 'without'} behavioral task data"
            )
            if self.info and (b := self.info.behavior_day) is not None:
                description += f" (day {b})"
            description += opto
            description += video
            description += sync
        return (
            ", with ".join(description.split(", with ")[:-1])
            + " and "
            + description.split(", with ")[-1]
        )

    @property
    def experiment_description(self) -> str:
        """Also used for description of main behavior task intervals table"""
        if (v := getattr(self, "_experiment_description", None)) is not None:
            desc = v
        elif self.is_templeton:
            desc = "sensory discrimination task experiment with task-irrelevant stimuli"
        elif self.epoch_tags == ["LuminanceTest"]:
            desc = "experiment with varying luminance levels of visual stimulus display for assessing pupil size"
        else:
            desc = "visual-auditory task-switching behavior experiment"
        assert (
            "experiment" in desc
        ), "experiment description should contain 'experiment', due to other function which replaces the word"
        return desc

    @npc_io.cached_property
    def source_script(self) -> str:
        """`githubTaskScript` from the task stim file, if available.
        Otherwise, url to Sam's repo on github"""
        if self.is_task and (script := self.task_data.get("githubTaskScript", None)):
            if isinstance(script[()], bytes):
                return script.asstr()[()].replace("Task//", "Task/")
            if isinstance(script[()], np.floating) and not np.isnan(script[()]):
                return str(script[()]).replace("Task//", "Task/")
        return (
            "https://github.com/samgale/DynamicRoutingTask/blob/main/DynamicRouting1.py"
        )

    @property
    def source_script_file_name(self) -> str:
        """url to tagged version of packaging code repo on github"""
        return f"https://github.com/AllenInstitute/npc_sessions/releases/tag/v{utils.get_package_version()}"

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
            if self.info and self.info.issues:
                self.keywords.append("issues")
            if self.is_task:
                self.keywords.append("task")
            if self.is_sync:
                self.keywords.append("sync")
            if self.is_video:
                self.keywords.append("video")
            if self.is_ephys:
                self.keywords.append("ephys")
            if self.is_sorted:
                self.keywords.append("units")
            if self.is_annotated:
                self.keywords.append("ccf")
            if self.is_training:
                self.keywords.append("training")
            if self.is_hab:
                self.keywords.append("hab")
            if self.is_opto:
                self.keywords.append("opto_perturbation")
            elif self.is_opto_control:
                self.keywords.append("opto_control")
            if self.is_templeton:
                self.keywords.append("templeton")
            else:
                self.keywords.append("dynamic_routing")
            if self.is_production:
                self.keywords.append("production")
            else:
                self.keywords.append("development")
            if self.is_injection_perturbation:
                self.keywords.append("injection_perturbation")
            elif self.is_injection_control:
                self.keywords.append("injection_control")
            if self.is_context_naive:
                self.keywords.append("context_naive")
            if self.is_task and self.is_late_autorewards:
                self.keywords.append("late_autorewards")
            elif self.is_task:
                self.keywords.append("early_autorewards")
            for t in self.epoch_tags:
                if t not in self.keywords:
                    self.keywords.append(t)
            # TODO these should be moved to `lab_metadata` when we have an ndx extension:
            if self.info and self.info.experiment_day is not None:
                self.keywords.append(f"experiment_day_{self.info.experiment_day}")
            if self.info and self.info.behavior_day is not None:
                self.keywords.append(f"behavior_day_{self.info.behavior_day}")
            # TODO
            # muscimol, perturbation, context_naive
        return self._keywords

    @keywords.setter
    def keywords(self, value: Iterable[str]) -> None:
        keywords = getattr(self, "_keywords", [])
        keywords += list(value)
        self._keywords = list(set(keywords))

    @npc_io.cached_property
    def subject(self) -> pynwb.file.Subject:
        with contextlib.suppress(FileNotFoundError, ValueError):
            return self.get_subject_from_aind_metadata()
        with contextlib.suppress(KeyError):
            return self.get_subject_from_training_sheet()
        logger.warning(f"Limited Subject information is available for {self.id}")
        return pynwb.file.Subject(
            subject_id=str(self.id.subject),
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

    @npc_io.cached_property
    def _acquisition(
        self,
    ) -> tuple[pynwb.core.NWBDataInterface | pynwb.core.DynamicTable, ...]:
        """The version passed to NWBFile.__init__"""
        modules: list[pynwb.core.NWBDataInterface | pynwb.core.DynamicTable] = []
        if self.is_sync and len(self._all_licks) >= 2:
            modules.append(self._all_licks[1])
        # if self.is_lfp:
        #     modules.append(self._raw_lfp)
        # if self.is_ephys:
        #     modules.append(self._raw_ap)
        if self.is_video:
            modules.extend(self._video_frame_times)
        with contextlib.suppress(AttributeError):
            modules.append(self._manipulator_positions)
        return tuple(modules)

    @property
    def processing(
        self,
    ) -> pynwb.core.LabelledDict[str, pynwb.base.ProcessingModule]:
        """Data after processing and filtering - raw data goes in
        `acquisition`.
        """
        processing = pynwb.core.LabelledDict(label="processing", key_attr="name")
        for module_name in ("behavior",) + (("ecephys",) if self.is_ephys else ()):
            module = getattr(self, f"_{module_name}")
            processing[module_name] = pynwb.base.ProcessingModule(
                name=module_name,
                description=f"processed {module_name} data",
                data_interfaces=module,
            )
        return processing

    @npc_io.cached_property
    def _behavior(
        self,
    ) -> tuple[pynwb.core.NWBDataInterface | pynwb.core.DynamicTable, ...]:
        modules: list[pynwb.core.NWBDataInterface | pynwb.core.DynamicTable] = []
        if self.is_task:
            modules.append(self._quiescent_interval_violations)
        if self._all_licks:
            modules.append(self._all_licks[0])
        modules.append(self._running_speed)
        try:
            modules.append(self._reward_times_with_duration)
        except AttributeError:
            modules.append(self._reward_frame_times)
        if self.is_video:
            if self.info and self.info.is_dlc_eye:
                modules.append(self._eye_tracking)
            modules.extend(self._dlc)
            modules.extend(self._facemap)
            modules.extend(self._LPFaceParts)

        return tuple(modules)

    @npc_io.cached_property
    def _ecephys(
        self,
    ) -> tuple[pynwb.core.NWBDataInterface | pynwb.core.DynamicTable, ...]:
        # TODO add filtered, sub-sampled LFP
        modules: list[pynwb.core.NWBDataInterface | pynwb.core.DynamicTable] = []
        return tuple(modules)

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

    @npc_io.cached_property
    def _analysis(
        self,
    ) -> tuple[pynwb.core.NWBDataInterface | pynwb.core.DynamicTable, ...]:
        """The version passed to NWBFile.__init__"""
        # TODO add RF maps
        modules: list[pynwb.core.NWBDataInterface | pynwb.core.DynamicTable] = []
        if self.is_sorted:
            modules.append(self.all_spike_histograms)
            # modules.append(self.drift_maps) # TODO no longer in sorted output: generate from scratch
        if self.is_task:
            modules.append(self.performance)
        return tuple(modules)

    # intervals ----------------------------------------------------------------- #

    @property
    def intervals_descriptions(self) -> dict[type[TaskControl.TaskControl], str]:
        return {
            TaskControl.VisRFMapping: "visual receptive-field mapping trials",
            TaskControl.AudRFMapping: "auditory receptive-field mapping trials",
            TaskControl.DynamicRouting1: self.experiment_description.replace(
                "experiment", "trials"
            ),  # name will be "trials" if assigned as main trials table in nwb
            TaskControl.OptoTagging: "optotagging trials",
        }

    def is_valid_interval(self, start_time: Any, stop_time: Any) -> bool:
        """Check if time interval is valid, based on `invalid_times`"""
        return utils.is_valid_interval(self, (start_time, stop_time))

    @npc_io.cached_property
    def invalid_times(self) -> pynwb.epoch.TimeIntervals | None:
        """Time intervals when recording was interrupted, stim malfunctioned or
        otherwise invalid.

        - current strategy is to not include intervals (in trials tables) where
          they overlap with entries in `invalid_times`

        A separate attribute can mark invalid times for individual ecephys units:
        see `NWBFile.Units.get_unit_obs_intervals()`
        """
        invalid_times = getattr(self, "_invalid_times", None)
        if invalid_times is None:
            return None
        intervals = pynwb.epoch.TimeIntervals(
            name="invalid_times",
            description="time intervals to be removed from analysis",
        )
        intervals.add_column(
            name="reason",
            description="reason for invalidation",
        )
        if self.info and invalid_times is not None:
            for interval in invalid_times:
                if (
                    stop_time := interval.get("stop_time", None)
                ) is None or stop_time == -1:
                    interval["stop_time"] = (
                        self.sync_data.total_seconds
                        if self.is_sync
                        else self.sam.frameTimes[-1]
                    )
                _ = interval.setdefault("reason", "unknown")
                for time in ("start_time", "stop_time"):
                    interval[time] = float(interval[time])
                intervals.add_interval(**interval)
        return intervals

    @property
    def trials(self) -> pynwb.epoch.TimeIntervals:
        if not self.is_task:
            if self.id == "670248_20230802":
                raise AttributeError(
                    "DynamicRouting1*.hdf5 was recorded badly for 670248_20230802 and won't open. "
                    "If you wish to compile an nwb anyway, set `session.is_task = False` for this session and re-run"
                )
            raise AttributeError(
                f"No trials table available for {self.id}: {self.is_task=}"
            )
        if (cached := getattr(self, "_cached_nwb_trials", None)) is not None:
            return cached
        trials = pynwb.epoch.TimeIntervals(
            name="trials",
            description=self.intervals_descriptions.get(
                self._trials.__class__,
                f"trials table for {self._trials.__class__.__name__}",
            ),
        )
        for column in self._trials.to_add_trial_column():
            trials.add_column(**column)
        for trial in self._trials.to_add_trial():
            if self.is_valid_interval(trial["start_time"], trial["stop_time"]):
                trials.add_interval(**trial)
        self._cached_nwb_trials = trials
        return trials

    @npc_io.cached_property
    def _trials(self) -> TaskControl.DynamicRouting1:
        """Main behavior task trials"""
        try:
            _ = self.task_path
        except StopIteration:
            raise ValueError(
                f"no stim named {self.task_stim_name}* found for {self.id}"
            ) from None
        # avoid iterating over values and checking for type, as this will
        # create all intervals in lazydict if they don't exist
        trials = self._all_trials[self.task_path.stem]
        assert isinstance(trials, TaskControl.DynamicRouting1)  # for mypy
        return trials

    @property
    def performance(self) -> pynwb.epoch.TimeIntervals:
        if not self.is_task:
            raise AttributeError(
                f"No performance table available for {self.id}: {self.is_task=}"
            )
        trials: pd.DataFrame = self.trials[:]
        task_performance_by_block: dict[str, dict[str, float | str]] = {}

        is_first_block_aud = any(
            v in self.sam.blockStimRewarded[0] for v in ("aud", "sound")
        )

        for block_idx in trials.block_index.unique():
            block_performance: dict[str, float | str] = {}
            block_df = trials[trials["block_index"] == block_idx]

            block_performance["block_index"] = block_idx
            block_performance["is_first_block_aud"] = (
                is_first_block_aud  # same for all blocks
            )
            rewarded_modality = block_df["context_name"].unique().item()
            block_performance["rewarded_modality"] = rewarded_modality
            if block_idx == 0 and is_first_block_aud:
                assert rewarded_modality in (
                    "aud",
                    "sound",
                ), f"Mismatch: {is_first_block_aud=} {rewarded_modality=}"

            block_performance["cross_modal_dprime"] = self.sam.dprimeOtherModalGo[
                block_idx
            ]
            block_performance["same_modal_dprime"] = self.sam.dprimeSameModal[block_idx]
            block_performance["nonrewarded_modal_dprime"] = (
                self.sam.dprimeNonrewardedModal[block_idx]
            )

            if rewarded_modality == "vis":
                block_performance["signed_cross_modal_dprime"] = (
                    self.sam.dprimeOtherModalGo[block_idx]
                )
                block_performance["vis_intra_dprime"] = self.sam.dprimeSameModal[
                    block_idx
                ]
                block_performance["aud_intra_dprime"] = self.sam.dprimeNonrewardedModal[
                    block_idx
                ]

            elif rewarded_modality in ("aud", "sound"):
                block_performance["signed_cross_modal_dprime"] = (
                    -self.sam.dprimeOtherModalGo[block_idx]
                )
                block_performance["aud_intra_dprime"] = self.sam.dprimeSameModal[
                    block_idx
                ]
                block_performance["vis_intra_dprime"] = self.sam.dprimeNonrewardedModal[
                    block_idx
                ]

            block_performance["n_trials"] = len(block_df)
            block_performance["n_responses"] = block_df.is_response.sum()
            block_performance["n_hits"] = self.sam.hitCount[block_idx]
            block_performance["n_contingent_rewards"] = block_df[
                "is_contingent_reward"
            ].sum()
            block_performance["hit_rate"] = self.sam.hitRate[block_idx]
            block_performance["false_alarm_rate"] = self.sam.falseAlarmRate[block_idx]
            block_performance["catch_response_rate"] = self.sam.catchResponseRate[
                block_idx
            ]
            for stim, target in itertools.product(
                ("vis", "aud"), ("target", "nontarget")
            ):
                stimulus_trials = block_df.query(
                    f"is_{stim}_{target} & ~is_reward_scheduled"
                )
                n_stimuli = len(stimulus_trials)
                n_responses = stimulus_trials.query(
                    "is_response & ~is_reward_scheduled"
                ).is_response.sum()
                block_performance[f"{stim}_{target}_response_rate"] = (
                    n_responses / n_stimuli
                )

            task_performance_by_block[block_idx] = block_performance

        nwb_intervals = pynwb.epoch.TimeIntervals(
            name="performance",
            description=f"behavioral performance for each context block in task (refers to `trials` or `intervals[{self.task_stim_name!r}])",
        )
        column_name_to_description = {
            "block_index": "presentation position of the block in the task (0-indexed)",
            "n_trials": "the number of trials in the block",
            "n_responses": "the number of responses the subject made in trials in the block",
            "n_hits": "the number of correct responses the subject made in GO trials in the block (excluding trials with scheduled reward)",
            "n_contingent_rewards": "the number of rewards the subject received for correct responses in the block",
            "hit_rate": "the proportion of correct responses the subject made in GO trials in the block (excluding trials with scheduled reward)",
            "false_alarm_rate": "the proportion of incorrect responses the subject made in NOGO trials in the block",
            "catch_response_rate": "the proportion of responses the subject made in catch trials in the block",
            "rewarded_modality": "the modality of the target stimulus that was rewarded in the block: normally `vis` or `aud`",
            "is_first_block_aud": "whether the rewarded modality of the first block in the task was auditory",
            "cross_modal_dprime": "dprime across modalities; hits=response rate to rewarded target stimulus, false alarms=response rate to non-rewarded target stimulus",
            "signed_cross_modal_dprime": "same as cross_modal_dprime, but with negative values for auditory blocks",
            "same_modal_dprime": "dprime within rewarded modality; hits=response rate to rewarded target stimulus, false alarms=response rate to same modality non-target stimulus",
            "nonrewarded_modal_dprime": "dprime within non-rewarded modality; hits=response rate to non-rewarded target stimulus, false alarms=response rate to same modality non-target stimulus",
            "vis_intra_dprime": "dprime within visual modality; hits=response rate to visual target stimulus, false alarms=response rate to visual non-target stimulus",
            "aud_intra_dprime": "dprime within auditory modality; hits=response rate to auditory target stimulus, false alarms=response rate to auditory non-target stimulus",
        } | {
            f"{stim}_{target}_response_rate": f"the proportion of responses the subject made to {'auditory' if stim == 'aud' else 'visual'} {target} stimulus trials in the block{' (excluding trials with scheduled reward)' if target else ''}"
            for stim, target in itertools.product(
                ("vis", "aud"), ("target", "nontarget")
            )
        }
        for name, description in column_name_to_description.items():
            nwb_intervals.add_column(name=name, description=description)
        for block_index in task_performance_by_block:
            start_time = trials[trials["block_index"] == block_index][
                "start_time"
            ].min()
            stop_time = trials[trials["block_index"] == block_index]["stop_time"].max()
            items: dict[str, str | float] = dict.fromkeys(
                column_name_to_description, np.nan
            )
            if self.is_valid_interval(start_time, stop_time):
                items |= task_performance_by_block[block_index]
            nwb_intervals.add_interval(
                start_time=start_time,
                stop_time=stop_time,
                **items,
            )
        return nwb_intervals

    @npc_io.cached_property
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

    @npc_io.cached_property
    def _intervals(self) -> tuple[pynwb.epoch.TimeIntervals, ...]:
        """The version passed to NWBFile.__init__"""
        intervals: list[pynwb.epoch.TimeIntervals] = []
        for k, v in self._all_trials.items():
            if self.task_stim_name in k and self.is_task:
                # intervals.append(self.trials)
                intervals.append(self.performance)
            intervals_table_name = utils.get_taskcontrol_intervals_table_name(
                v.__class__.__name__
            )
            if not any(
                existing := [i for i in intervals if i.name == intervals_table_name]
            ):
                nwb_intervals = pynwb.epoch.TimeIntervals(
                    name=intervals_table_name,
                    description=self.intervals_descriptions.get(
                        v.__class__,
                        f"{intervals_table_name.replace('_', ' ')} table",
                    ),
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
                if self.is_valid_interval(trial["start_time"], trial["stop_time"]):
                    nwb_intervals.add_interval(**trial)
            if trial_idx_offset == 0:
                intervals.append(nwb_intervals)
        # TODO deal with stimuli across epochs
        #! requires stimulus param hashes or links to stimulus table, to
        # identify unique stims across stim files
        return tuple(intervals)

    @npc_io.cached_property
    def _all_trials(self) -> npc_io.LazyDict[str, TaskControl.TaskControl]:
        if self.is_sync:
            # get the only stims for which we have times:
            stim_paths = tuple(
                path
                for path in self.stim_paths
                if path.stem in self.stim_data_without_timing_issues.keys()
            )
        else:
            stim_paths = self.stim_paths

        def get_intervals(
            cls: type[TaskControl.TaskControl], stim_filename: str, **kwargs
        ) -> TaskControl.TaskControl:
            logger.info(f"Generating intervals: {cls.__name__}({stim_filename!r})")
            return cls(self.stim_data[stim_filename], **kwargs)

        lazy_dict_items: dict[str, tuple] = {}  # tuple of (func, args, kwargs)

        def set_lazy_eval(
            key: str,
            cls: type[TaskControl.TaskControl],
            stim_filename: str,
            taskcontrol_kwargs: dict[str, Any],
        ) -> None:
            lazy_dict_items[key] = (
                get_intervals,
                (cls, stim_filename),
                taskcontrol_kwargs,
            )

        for stim_path in stim_paths:
            assert isinstance(stim_path, upath.UPath)
            stim_filename = stim_path.stem
            kwargs: dict[str, Any] = {}
            if self.is_sync:
                kwargs |= {"sync": self.sync_data}
            if self.is_ephys and self.is_sync:
                kwargs |= {"ephys_recording_dirs": self.ephys_recording_dirs}

            # set items in LazyDict for postponed evaluation
            if "RFMapping" in stim_filename:
                # create two separate trials tables
                set_lazy_eval(
                    key=f"Aud{stim_filename}",
                    cls=TaskControl.AudRFMapping,
                    stim_filename=stim_filename,
                    taskcontrol_kwargs=kwargs,
                )
                set_lazy_eval(
                    key=f"Vis{stim_filename}",
                    cls=TaskControl.VisRFMapping,
                    stim_filename=stim_filename,
                    taskcontrol_kwargs={
                        k: v for k, v in kwargs.items() if k != "ephys_recording_dirs"
                    },
                    # passing ephys_recordings would run sound alignment unnecessarily
                )
            else:
                try:
                    cls: type[TaskControl.TaskControl] = getattr(
                        TaskControl, stim_filename.split("_")[0]
                    )
                except AttributeError:
                    # some stims (e.g. Spontaneous) have no trials class
                    continue
                set_lazy_eval(
                    key=stim_filename,
                    cls=cls,
                    stim_filename=stim_filename,
                    taskcontrol_kwargs=kwargs,
                )
        return npc_io.LazyDict((k, v) for k, v in lazy_dict_items.items())

    @npc_io.cached_property
    def epochs(self) -> pynwb.file.TimeIntervals:
        epochs = pynwb.file.TimeIntervals(
            name="epochs",
            description="time intervals corresponding to different phases of the session; each epoch corresponds to one TaskControl subclass that controlled stimulus presentation during the epoch, which corresponds to one .hdf5 stimulus file",
        )
        epochs.add_column(
            "stim_name",
            description="the name of the TaskControl subclass that controlled stimulus presentation during the epoch",
        )
        epochs.add_column(
            "notes",
            description="notes about the experiment or the data collected during the epoch",
        )
        epochs.add_column(
            "interval_names",
            description="names of other intervals tables that contain trial data from the epoch",
            index=True,
        )

        def get_epoch_record(stim_file: npc_io.PathLike) -> dict[str, Any]:
            stim_file = npc_io.from_pathlike(stim_file)
            h5 = self.stim_data[stim_file.stem]
            stim_name = stim_file.stem.split("_")[0]

            tags = []
            if self.task_stim_name in stim_file.name:
                tags.append("task")
                if npc_samstim.is_opto(h5) or npc_samstim.is_galvo_opto(h5):
                    if self.is_opto:
                        tags.append("opto_perturbation")
                    else:
                        tags.append("opto_control")
            if (rewards := h5.get("rewardFrames", None)) is not None and any(
                rewards[:]
            ):
                tags.append("rewards")
            if stim_file.stem not in self.stim_data_without_timing_issues:
                tags.append("timing_issues")
            for tag in ("spontaneous", "mapping"):
                if tag in stim_name.lower():
                    tags.append(tag)
            if "optotagging" in stim_file.stem.lower():
                if self.is_wildtype:
                    tags.append("optotagging_control")
                else:
                    tags.append("optotagging")

            interval_names = []
            if "RFMapping" in stim_name:
                interval_names.extend(
                    [
                        utils.get_taskcontrol_intervals_table_name(n)
                        for n in ("VisRFMapping", "AudRFMapping")
                    ]
                )
            else:
                interval_names.append(
                    utils.get_taskcontrol_intervals_table_name(stim_name)
                )
                if self.task_stim_name in stim_file.name and self.is_task:
                    interval_names.append("performance")
            interval_names = list(dict.fromkeys(interval_names).keys())

            invalid_times_notes: list[str] = []
            if not self.is_sync:
                # only one stim, so we use its frame times as recorded on stim computer
                start_time = 0.0
                stop_time = npc_stim.get_stim_duration(h5)
            else:
                frame_times = self.stim_frame_times[stim_file.stem]
                start_time = frame_times[0]
                stop_time = frame_times[-1]
            assert start_time != stop_time
            if self.invalid_times is not None:
                for _, invalid_interval in self.invalid_times[:].iterrows():
                    if any(
                        start_time <= invalid_interval[time] <= stop_time
                        for time in ("start_time", "stop_time")
                    ):
                        invalid_times_notes.append(invalid_interval["reason"])
                        if "invalid_times" not in tags:
                            tags.append("invalid_times")

            return {
                "start_time": start_time,
                "stop_time": stop_time,
                "stim_name": stim_name,
                "tags": tags,
                "interval_names": (
                    interval_names
                    if stim_file.stem in self.stim_data_without_timing_issues
                    else []
                ),
                "notes": (
                    ""
                    if not invalid_times_notes
                    else f"includes invalid times: {'; '.join(invalid_times_notes)}"
                ),
            }

        records = []
        for stim in self.stim_paths:
            if self.is_sync and stim.stem not in self.stim_frame_times.keys():
                continue
            records.append(get_epoch_record(stim))

        for record in sorted(records, key=lambda _: _["start_time"]):
            epochs.add_interval(**record)
        return epochs

    # probes, devices, units ---------------------------------------------------- #

    @npc_io.cached_property
    def _probes(self) -> tuple[pynwb.device.Device, ...]:
        if not self.is_ephys:
            raise AttributeError(f"{self.id} is not an ephys session")
        return tuple(
            pynwb.device.Device(
                name=str(serial_number),
                description=probe_type,
                manufacturer="imec",
            )
            for serial_number, probe_type, probe_letter in zip(
                self.ephys_settings_xml_data.probe_serial_numbers,
                self.ephys_settings_xml_data.probe_types,
                self.ephys_settings_xml_data.probe_letters,
            )
            if probe_letter in self.probe_letters_to_use
        )

    @npc_io.cached_property
    def _manipulators(self) -> tuple[pynwb.device.Device, ...]:
        if not self.is_ephys:
            raise AttributeError(f"{self.id} is not an ephys session")
        try:
            _ = self._manipulator_positions
        except AttributeError as exc:
            raise AttributeError(
                f"{self.id} is an ephys session, but no manipulator info available"
            ) from exc
        return tuple(
            pynwb.device.Device(
                name=row["device_name"],
                description=f"motorized 3-axis micromanipulator for positioning and inserting {probe.name}",
                manufacturer="NewScale",
            )
            for _, row in self._manipulator_positions[:].iterrows()
            if (probe := npc_session.ProbeRecord(row["electrode_group_name"]))
            in self.probe_letters_to_use
        )

    @property
    def _devices(self) -> tuple[pynwb.device.Device, ...]:
        """The version passed to NWBFile.__init__"""
        devices: list[pynwb.device.Device] = []
        if self.is_ephys:
            devices.extend(self._probes)
            with contextlib.suppress(AttributeError):
                devices.extend(self._manipulators)
        return tuple(devices)  # add other devices as we need them

    @npc_io.cached_property
    def devices(self) -> pynwb.core.LabelledDict[str, pynwb.device.Device]:
        """Currently just probe model + serial number.

        Could include other devices: laser, monitor, etc.

        The property as it appears on an NWBFile"""
        devices = pynwb.core.LabelledDict(label="devices", key_attr="name")
        for module in self._devices:
            devices[module.name] = module
        return devices

    @npc_io.cached_property
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

    @npc_io.cached_property
    def _electrode_groups(self) -> tuple[pynwb.ecephys.ElectrodeGroup, ...]:
        """The version passed to NWBFile.__init__"""
        if not self.is_ephys:
            raise AttributeError(f"{self.id} is not an ephys session")
        locations: dict[str, str | None] = {}
        if self.probe_insertions and self.implant:
            for probe_letter in self.probe_insertions:
                locations[probe_letter] = (
                    f"{self.implant} {self.probe_insertions[probe_letter]}"
                )
        return tuple(
            pynwb.ecephys.ElectrodeGroup(
                name=f"probe{probe_letter}",
                device=next(p for p in self._probes if p.name == str(serial_number)),
                description=probe_type,
                location=locations.get(probe_letter, "unknown"),
            )
            for serial_number, probe_type, probe_letter in zip(
                self.ephys_settings_xml_data.probe_serial_numbers,
                self.ephys_settings_xml_data.probe_types,
                self.ephys_settings_xml_data.probe_letters,
            )
            if probe_letter in self.probe_letters_to_use
        )

    @npc_io.cached_property
    def electrodes(self) -> pynwb.core.DynamicTable:
        """Individual channels on an inserted probe, including location, CCF
        coords.

        Currently assumes Neuropixels 1.0 probes.
        """
        if not self.is_ephys:
            raise AttributeError(f"{self.id} is not an ephys session")

        electrodes = pynwb.file.ElectrodeTable()

        column_names: tuple[str, ...] = (
            "channel",
            "rel_x",
            "rel_y",
            "reference",
            "imp",
        )
        if self.is_annotated:
            column_names = (
                "structure",
                "x",
                "y",
                "z",
            ) + column_names
            ccf_df = utils.get_tissuecyte_electrodes_table(self.id)
            if "raw_structure" in ccf_df:
                column_names = column_names + ("raw_structure",)
        column_description = {
            "structure": "acronym for the Allen CCF structure that the electrode recorded from - less-specific than `location`",
            "raw_structure": "same as `structure`, except white matter areas (lowercase names) have not been reassigned",
            "x": "x coordinate in the Allen CCF, +x is posterior",
            "y": "y coordinate in the Allen CCF, +y is inferior",
            "z": "z coordinate in the Allen CCF, +z is right",
            "channel": "channel index on the probe, as used in OpenEphys (0 at probe tip)",
            "rel_x": "position on the short-axis of the probe surface, in microns",
            "rel_y": "position on the long-axis of the probe surface, relative to the tip, in microns",
            "reference": "the reference electrode or referencing scheme used",
            "imp": "impedance, in ohms",
        }
        for column in column_names:
            electrodes.add_column(
                name=column,
                description=column_description[column],
            )

        for probe_letter, channel_pos_xy in zip(
            self.ephys_settings_xml_data.probe_letters,
            self.ephys_settings_xml_data.channel_pos_xy,
        ):
            if probe_letter not in self.probe_letters_to_use:
                continue
            group = self.electrode_groups[f"probe{probe_letter}"]
            for channel_label, (x, y) in channel_pos_xy.items():
                channel_idx = int(channel_label.strip("CH"))
                row_kwargs: dict[str, str | float] = {
                    "group": group,
                    "group_name": group.name,
                    "rel_x": x,
                    "rel_y": y,
                    "channel": channel_idx,
                    "imp": 150e3,  # https://www.neuropixels.org/_files/ugd/832f20_4a14406ba1204e60ae8534b09e201b49.pdf
                    "reference": "tip",
                    "location": "unannotated",
                }
                if self.is_annotated and any(
                    (
                        annotated_probes := ccf_df.query(
                            f"group_name == {group.name!r}"
                        )
                    ).any()
                ):
                    row_kwargs |= (
                        annotated_probes.query(f"channel == {channel_idx}")
                        .iloc[0]
                        .to_dict()
                    )
                electrodes.add_row(
                    **row_kwargs,
                )

        return electrodes

    @npc_io.cached_property
    def is_waveforms(self) -> bool:
        """Whether to include waveform arrays in the units table"""
        if (v := getattr(self, "_is_waveforms", None)) is not None:
            return v
        return False

    @npc_io.cached_property
    def _units(self) -> pd.DataFrame:
        if not self.is_sorted:
            raise AttributeError(f"{self.id} hasn't been spike-sorted")
        units = npc_ephys.add_global_unit_ids(
            units=npc_ephys.make_units_table_from_spike_interface_ks25(
                self.sorted_data,
                self.ephys_timing_data,
                include_waveform_arrays=self.is_waveforms,
            ),
            session=self.id,
        )
        # remove units from probes that weren't inserted
        units = units[units["electrode_group_name"].isin(self.probes_inserted)]
        if self.is_annotated:
            units = npc_ephys.add_electrode_annotations_to_units(
                units=units,
                annotated_electrodes=utils.get_tissuecyte_electrodes_table(self.id),
            )
        return units

    def get_obs_intervals(
        self, probe: str | npc_session.ProbeRecord
    ) -> tuple[tuple[float, float], ...]:
        """[[start stop], ...] for intervals in which ephys data is available for
        a given probe.

        - times are on sync clock, relative to start
        - one interval for most sessions: start and stop of ephys recording
        - multiple intervals when recording is split into parts # TODO not supported yet
        - if sync stopped before ephys for any reason, stop time is sync stop time
        """
        timing_data = next(
            (
                t
                for t in self.ephys_timing_data
                if npc_session.extract_probe_letter(t.device.name)
                == npc_session.ProbeRecord(probe)
            ),
            None,
        )
        if timing_data is None:
            raise ValueError(f"no ephys timing data for {self.id} {probe}")
        stop_time = min(timing_data.stop_time, self.sync_data.total_seconds)
        return ((timing_data.start_time, stop_time),)

    @npc_io.cached_property
    def sorted_channel_indices(self) -> dict[npc_session.ProbeRecord, tuple[int, ...]]:
        """SpikeInterface stores channels as 1-indexed integers: "AP1", ...,
        "AP384". This method returns the 0-indexed *integers* for each probe
        recorded, for use in indexing into the electrode table.
        """
        return {
            probe: self.sorted_data.sparse_channel_indices(probe)
            for probe in self.probe_letters_to_use
        }

    @npc_io.cached_property
    def units(self) -> pynwb.misc.Units:
        if not self.is_ephys:
            raise AttributeError(f"{self.id} is not an ephys session")
        units = pynwb.misc.Units(
            name="units",
            description="spike-sorted units from Kilosort 2.5",
            waveform_rate=30_000.0,
            waveform_unit="microvolts",
            electrode_table=self.electrodes,
        )
        for column in self._units.columns:
            if column in (
                "spike_times",
                "waveform_mean",
                "waveform_sd",
                "electrode_group",
            ):
                continue
            units.add_column(name=column, description="")  # TODO add descriptions
        units.add_column(
            name="peak_electrode",
            description="index in `electrodes` table of channel with largest amplitude waveform",
        )
        # TODO add back when annotations are correct
        # units.add_column(
        #     name="peak_waveform_index",
        #     description="index in `waveform_mean` and `waveform_sd` arrays for channel with largest amplitude waveform",
        # )
        electrodes = self.electrodes[:]
        for _, row in self._units.iterrows():
            group_query = f"group_name == {row['electrode_group_name']!r}"
            if self.is_waveforms:
                row["electrodes"] = electrodes.query(
                    f"{group_query} & channel in {row['channels']}"
                ).index.to_list()
            ## for ref:
            # add_unit(spike_times=None, obs_intervals=None, electrodes=None, electrode_group=None, waveform_mean=None, waveform_sd=None, waveforms=None, id=None)
            units.add_unit(
                **row,  # contains spike_times
                electrode_group=self.electrode_groups[row["electrode_group_name"]],
                peak_electrode=(
                    peak_electrode := electrodes.query(
                        f"{group_query} & channel == {row['peak_channel']}"
                    ).index.item()
                ),
                # TODO incorrect for some units: add back when annotations are correct
                # peak_waveform_index=row["electrodes"].index(peak_electrode),
                obs_intervals=self.get_obs_intervals(row["electrode_group_name"]),
            )
        if "waveform_mean" in units:
            assert all(
                len(unit.electrodes) == unit.waveform_mean.shape[1]
                for _, unit in units[:].iterrows()
            )
        if "waveform_sd" in units:
            assert all(
                len(unit.electrodes) == unit.waveform_sd.shape[1]
                for _, unit in units[:].iterrows()
            )
        return units

    class AP(pynwb.core.MultiContainerInterface):
        """
        AP-band ephys data from one or more channels. The electrode map in each published ElectricalSeries will
        identify which channels are providing AP data. Filter properties should be noted in the
        ElectricalSeries description or comments field.
        """

        __clsconf__ = [
            {
                "attr": "electrical_series",
                "type": pynwb.ecephys.ElectricalSeries,
                "add": "add_electrical_series",
                "get": "get_electrical_series",
                "create": "create_electrical_series",
            }
        ]

    @npc_io.cached_property
    def _raw_ap(self) -> pynwb.core.MultiContainerInterface:
        ap = self.AP()
        #! this will likely not write to disk as the class is not registered with 'CORE_NAMESPACE'
        # there's currently no appropriate ephys MultiContainerInterface
        # but `pynwb.ecephys.FilteredEphys()` would work if otherwise unused
        band: str = "0.3-10 kHz"
        for probe in self.electrode_groups.values():
            timing_info = next(
                d
                for d in self.ephys_timing_data
                if d.device.name.endswith("AP")
                and probe.name.lower() in d.device.name.lower()
            )

            electrode_table_region = hdmf.common.DynamicTableRegion(
                name="electrodes",  # pynwb requires this not be renamed
                description=f"channels with AP data on {probe.name}",
                data=tuple(range(0, 384)),  # TODO get correct channel indices
                table=self.electrodes,
            )

            # as long as we don't index into the data array (ie to take a subset), it
            # will be instantly inserted into the electrical series container for lazy access
            if timing_info.device.compressed:
                data = zarr.open(timing_info.device.compressed, mode="r")["traces_seg0"]
            else:
                data = np.memmap(
                    timing_info.device.continuous / "continuous.dat",
                    dtype=np.int16,
                    mode="r",
                ).reshape(-1, 384)

            ap.create_electrical_series(
                name=probe.name,
                data=data,
                electrodes=electrode_table_region,
                starting_time=float(timing_info.start_time),
                rate=float(timing_info.sampling_rate),
                channel_conversion=None,
                filtering=band,
                conversion=0.195e-6,  # bit/microVolt from open-ephys
                comments="",
                resolution=0.195e-6,
                description=f"action potential-band voltage timeseries ({band}) from electrodes on {probe.name}",
                # units=microvolts, # doesn't work - electrical series must be in volts
            )
        return ap

    @npc_io.cached_property
    def _raw_lfp(self) -> pynwb.ecephys.LFP:
        lfp = pynwb.ecephys.LFP()
        band: str = "0.5-500 Hz"

        for probe in self.electrode_groups.values():
            timing_info = next(
                d
                for d in self.ephys_timing_data
                if d.device.name.endswith("LFP")
                and probe.name.lower() in d.device.name.lower()
            )

            electrode_table_region = hdmf.common.DynamicTableRegion(
                name="electrodes",  # pynwb requires this not be renamed
                description=f"channels with LFP data on {probe.name}",
                data=tuple(range(0, 384)),  # TODO get correct channel indices
                table=self.electrodes,
            )

            # as long as we don't index into the data array (ie to take a subset), it
            # will be instantly inserted into the electrical series container for lazy access
            if timing_info.device.compressed:
                data = zarr.open(timing_info.device.compressed, mode="r")["traces_seg0"]
            else:
                data = np.memmap(
                    timing_info.device.continuous / "continuous.dat",
                    dtype=np.int16,
                    mode="r",
                ).reshape(-1, 384)

            lfp.create_electrical_series(
                name=probe.name,
                data=data,
                electrodes=electrode_table_region,
                starting_time=float(timing_info.start_time),
                rate=float(timing_info.sampling_rate),
                channel_conversion=None,
                filtering=band,
                conversion=0.195e-6,  # bit/microVolt from open-ephys
                comments="",
                resolution=0.195e-6,
                description=f"local field potential-band voltage timeseries ({band}) from electrodes on {probe.name}",
                # units=microvolts, # doesn't work - electrical series must be in volts
            )
        return lfp

    # images -------------------------------------------------------------------- #

    @npc_io.cached_property
    def drift_maps(self) -> pynwb.image.Images:
        return pynwb.image.Images(
            name="drift_maps",
            images=tuple(
                self.img_to_nwb(p)
                for p in self.drift_map_paths
                if npc_session.ProbeRecord(p.as_posix()) in self.probe_letters_to_use
            ),
            description="activity plots (time x probe depth x firing rate) over the entire ecephys recording, for assessing probe drift",
        )

    @staticmethod
    def img_to_nwb(path: npc_io.PathLike) -> pynwb.image.Image:
        path = npc_io.from_pathlike(path)
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

    @npc_io.cached_property
    def info(self) -> npc_lims.SessionInfo | None:
        with contextlib.suppress(ValueError):
            return npc_lims.get_session_info(self.id)
        return None

    @npc_io.cached_property
    def is_task(self) -> bool:
        if (v := getattr(self, "_is_task", None)) is not None:
            return v
        with contextlib.suppress(
            FileNotFoundError, ValueError, StopIteration, npc_lims.MissingCredentials
        ):
            _ = self.task_data
            return True
        return False

    @npc_io.cached_property
    def is_sync(self) -> bool:
        if (v := getattr(self, "_is_sync", None)) is not None:
            return v
        if (v := getattr(self, "_sync_path", None)) is not None:
            return True
        with contextlib.suppress(
            FileNotFoundError, ValueError, npc_lims.MissingCredentials
        ):
            if self.get_sync_paths():
                return True
        return False

    @npc_io.cached_property
    def is_video(self) -> bool:
        if (v := getattr(self, "_is_video", None)) is not None:
            return v
        if not self.is_sync:
            return False
        with contextlib.suppress(
            FileNotFoundError, ValueError, npc_lims.MissingCredentials
        ):
            if self.video_paths:
                return True
        return False

    @npc_io.cached_property
    def is_ephys(self) -> bool:
        if (v := getattr(self, "_is_ephys", None)) is not None:
            return v
        if self.info:
            return self.info.is_ephys
        with contextlib.suppress(
            FileNotFoundError, ValueError, npc_lims.MissingCredentials
        ):
            if self.ephys_record_node_dirs:
                return True
        return False

    @npc_io.cached_property
    def is_training(self) -> bool:
        if (v := getattr(self, "_is_training", None)) is not None:
            return v
        return (
            self.is_task
            and not self.is_ephys
            and not any(name in self.rig for name in ("NP", "OG"))
        )

    @npc_io.cached_property
    def is_hab(self) -> bool:
        if (v := getattr(self, "_is_hab", None)) is not None:
            return v
        return (
            self.is_task
            and not self.is_ephys
            and "NP" in self.rig
            and not (self.is_opto or self.is_opto_control)
        )

    @npc_io.cached_property
    def is_sorted(self) -> bool:
        if (v := getattr(self, "_is_sorted", None)) is not None:
            return v
        if not self.is_ephys:
            return False
        if self.info:
            return self.info.is_sorted
        with contextlib.suppress(
            FileNotFoundError, ValueError, npc_lims.MissingCredentials
        ):
            _ = npc_lims.get_session_sorted_data_asset(self.id)
            return True
        return False

    @npc_io.cached_property
    def is_annotated(self) -> bool:
        """CCF annotation data accessible"""
        if not self.is_ephys:
            return False
        if self.info:
            return self.info.is_annotated
        with contextlib.suppress(
            FileNotFoundError, ValueError, npc_lims.MissingCredentials
        ):
            if npc_lims.get_tissuecyte_annotation_files_from_s3(self.id):
                return True
        return False

    @npc_io.cached_property
    def is_surface_channels(self) -> bool:
        if self.info and self.info.is_surface_channels:
            return True
        with contextlib.suppress(FileNotFoundError, ValueError):
            return bool(self.surface_root_path)
        return False

    @npc_io.cached_property
    def is_lfp(self) -> bool:
        if (v := getattr(self, "_is_lfp", None)) is not None:
            return v
        return self.is_ephys

    @npc_io.cached_property
    def is_wildtype(self) -> bool:
        if self.subject.genotype is None:  # won't exist if subject.json not found
            logger.warning(
                f"Could not find genotype for {self.id}: returning is_wildtype = True regardless"
            )
            return True
        return "wt/wt" in self.subject.genotype.lower()

    @npc_io.cached_property
    def is_opto(self) -> bool:
        """Opto during behavior task && not wt/wt (if genotype info available)"""
        if (
            self.is_task
            and npc_samstim.is_opto(self.task_data)
            and not self.is_wildtype
        ):
            return True
        return False

    @npc_io.cached_property
    def is_opto_control(self) -> bool:
        """Opto during behavior task && wt/wt"""
        if self.is_task and npc_samstim.is_opto(self.task_data) and self.is_wildtype:
            return True
        return False

    @property
    def is_production(self) -> bool:
        if (v := getattr(self, "_is_production", None)) is not None:
            return v
        return True

    @property
    def is_injection_perturbation(self) -> bool:
        if (v := getattr(self, "_is_injection_perturbation", None)) is not None:
            return v
        return False

    @property
    def is_injection_control(self) -> bool:
        if (v := getattr(self, "_is_injection_control", None)) is not None:
            return v
        return False

    @property
    def is_context_naive(self) -> bool:
        if (v := getattr(self, "_is_context_naive", None)) is not None:
            return v
        return False

    @property
    def is_late_autorewards(self) -> bool:
        if not self.is_task:
            raise AttributeError(f"{self.id} is not a session with behavior task")
        return self.sam.autoRewardOnsetFrame == 60

    @property
    def is_templeton(self) -> bool:
        if (v := getattr(self, "_is_templeton", None)) is not None:
            return v
        if self.info is not None:
            return self.info.is_templeton
        if self.is_task and self.task_version is not None:
            return bool("templeton" in self.task_version)
        raise NotImplementedError(
            "Not enough information to tell if this is a Templeton session"
        )

    # helper properties -------------------------------------------------------- #

    @npc_io.cached_property
    def _raw_upload_metadata_json_paths(self) -> tuple[upath.UPath, ...]:
        return tuple(
            file
            for file in npc_lims.get_raw_data_root(self.id).iterdir()
            if file.suffix == ".json"
        )

    @npc_io.cached_property
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

    @npc_io.cached_property
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

    @npc_io.cached_property
    def _subject_training_sheet_metadata(self) -> dict[str, Any]:
        with contextlib.suppress(KeyError):
            return npc_lims.get_subjects_from_training_db()[self.id.subject]
        with contextlib.suppress(KeyError):
            return npc_lims.get_subjects_from_training_db(nsb=True)[self.id.subject]
        raise KeyError(
            f"Could not find {self.id.subject} in training sheets (NSB & non-NSB)"
        )

    def get_subject_from_training_sheet(self) -> pynwb.file.Subject:
        metadata = self._subject_training_sheet_metadata
        assert metadata["mouse_id"] == self.id.subject
        dob = utils.get_aware_dt(metadata["birthdate"])
        return pynwb.file.Subject(
            subject_id=metadata["mouse_id"],
            species="Mus musculus",
            sex=metadata["sex"][0].upper(),
            date_of_birth=dob,
            genotype=metadata["genotype"],
            description=None,
            age=f"P{(self.session_start_time - dob).days}D",
        )

    def get_subject_from_aind_metadata(self) -> pynwb.file.Subject:
        metadata = self._subject_aind_metadata
        assert metadata["subject_id"] == self.id.subject
        dob = utils.get_aware_dt(metadata["date_of_birth"])

        strain = metadata["background_strain"]
        if strain is None:
            strain = metadata.get("breeding_group", None)
        if strain is None:
            breeding_info = metadata.get("breeding_info", None) or {}
            with contextlib.suppress(KeyError):
                strain = breeding_info.get("breeding_group", None)

        return pynwb.file.Subject(
            subject_id=metadata["subject_id"],
            species="Mus musculus",
            sex=metadata["sex"][0].upper(),
            date_of_birth=dob,
            genotype=metadata["genotype"],
            description=None,
            strain=strain,
            age=f"P{(self.session_start_time - dob).days}D",
        )

    @property
    def stim_names(self) -> tuple[str, ...]:
        """Currently assumes TaskControl hdf5 files"""
        return tuple(
            name.split("_")[0]
            for name in sorted(
                [p.name for p in self.stim_paths], key=npc_session.DatetimeRecord
            )
        )

    @property
    def root_path(self) -> upath.UPath | None:
        """Assigned on init if session_or_path is a pathlike object.
        May also be assigned later when looking for raw data if Code Ocean upload is missing.
        """
        if (v := getattr(self, "_root_path", None)) is not None:
            return v
        self._root_path = None
        if self.info is not None and not self.info.is_uploaded:
            for path in (self.info.cloud_path, self.info.allen_path):
                if path is not None and path.exists():
                    self._root_path = path
                    break
        return self._root_path

    def get_raw_data_paths_from_root(
        self, root: upath.UPath | None = None
    ) -> tuple[upath.UPath, ...]:
        root = root or self.root_path
        if root is None:
            raise ValueError(f"{self.id} does not have a local root_path assigned yet")
        if root.is_file():
            return (root,)
        ephys_paths = itertools.chain(
            root.glob("Record Node *"),
            root.glob("*/Record Node *"),
        )
        root_level_paths = tuple(p for p in root.iterdir() if p.is_file())
        return root_level_paths + tuple(set(ephys_paths))

    def get_task_hdf5_from_s3_repo(self) -> upath.UPath:
        try:
            return next(
                file.path
                for file in npc_lims.get_hdf5_stim_files_from_s3(self.id)
                if self.task_stim_name in file.path.stem
            )
        except StopIteration:
            raise FileNotFoundError(
                f"Could not find file in {npc_lims.DR_DATA_REPO} for {self.id}"
            ) from None

    @npc_io.cached_property
    def newscale_log_path(self) -> upath.UPath:
        if not self.is_ephys:
            raise AttributeError(
                f"{self.id} is not an ephys session: no NewScale logs available"
            )
        p = tuple(
            p
            for p in self.raw_data_paths
            if p.suffix == ".csv"
            and any(v in p.stem for v in ("log", "motor-locs", "motor_locs"))
        )
        if not p:
            raise FileNotFoundError("Cannot find .csv with motor locs")
        if len(p) > 1:
            raise ValueError(f"Multiple NewScale log files found: {p}")
        return p[0]

    @npc_io.cached_property
    def _manipulator_positions(self) -> pynwb.core.DynamicTable:
        if not self.is_ephys:
            raise AttributeError(
                f"{self.id} is not an ephys session: no manipulator coords available"
            )
        start_of_np3_logging = datetime.date(2023, 1, 30)
        if self.id.date.dt < start_of_np3_logging:
            raise AttributeError(
                f"{self.id} is an ephys session, but no NewScale log file available"
            ) from None

        try:
            _ = self.newscale_log_path
        except FileNotFoundError as exc:
            raise AttributeError(
                f"{self.id} has no log.csv file to get manipulator coordinates"
            ) from exc

        df = npc_ephys.get_newscale_coordinates(
            self.newscale_log_path,
            f"{self.id.date}_{self.ephys_settings_xml_data.start_time.isoformat()}",
        )
        df = df.drop(columns="last_movement_dt")
        t = pynwb.core.DynamicTable(
            name="manipulator_positions",
            description="nominal positions of the motorized stages on each probe's manipulator assembly at the time of ecephys recording",
        )
        colnames = {
            "electrode_group_name": "name of probe mounted on the manipulator",
            "device_name": "serial number of NewScale device",
            "last_movement_time": "time of last movement of the manipulator, in seconds, relative to `session_start_time`; should always be negative: manipulators do not move after recording starts; value includes at least 15 min for probe to settle before recording",
            "x": "horizontal position in microns (direction of axis varies)",
            "y": "horizontal position in microns (direction of axis varies)",
            "z": "vertical position in microns (+z is inferior, 0 is fully retracted)",
        }
        for name, description in colnames.items():
            t.add_column(name=name, description=description)
        for _, row in df.iterrows():
            t.add_row(data=dict(row))
        return t

    @npc_io.cached_property
    def raw_data_paths(self) -> tuple[upath.UPath, ...]:
        def _filter(paths: tuple[upath.UPath, ...]) -> tuple[upath.UPath, ...]:
            return tuple(
                p
                for p in paths
                if not any(e in p.as_posix() for e in self.excluded_stim_file_names)
            )

        if self.root_path:
            return _filter(self.get_raw_data_paths_from_root())
        with contextlib.suppress(FileNotFoundError, ValueError):
            return _filter(npc_lims.get_raw_data_paths_from_s3(self.id))
        if (
            getattr(self, "_is_task", None) is not False
        ):  # using regular version will cause infinite recursion
            with contextlib.suppress(StopIteration):
                if stim_files := npc_lims.get_hdf5_stim_files_from_s3(self.id):
                    self._root_path = stim_files[0].path.parent
                    logger.warning(
                        f"Using {self._root_path} as root path for {self.id}"
                    )
                    return _filter(self.get_raw_data_paths_from_root())
        raise ValueError(
            f"{self.id} is either an ephys session with no Code Ocean upload, or a behavior session with no data in the synced s3 repo {npc_lims.DR_DATA_REPO}"
        )

    @npc_io.cached_property
    def sorted_data_asset_id(self) -> str | None:
        return getattr(self, "_sorted_data_asset_id", None)

    @npc_io.cached_property
    def sorted_data(self) -> npc_ephys.SpikeInterfaceKS25Data:
        return npc_ephys.SpikeInterfaceKS25Data(
            self.id,
            self.sorted_data_paths[0].parent,
        )

    @npc_io.cached_property
    def sorted_data_paths(self) -> tuple[upath.UPath, ...]:
        if not self.is_ephys:
            raise ValueError(f"{self.id} is not a session with ephys")
        return npc_lims.get_sorted_data_paths_from_s3(
            self.id, self.sorted_data_asset_id
        )

    @npc_io.cached_property
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

    @npc_io.cached_property
    def raw_data_asset_id(self) -> str:
        if not self.is_ephys:
            raise ValueError(
                f"{self.id} currently only ephys sessions have raw data assets"
            )
        return npc_lims.get_session_raw_data_asset(self.id)["id"]

    @npc_io.cached_property
    def sync_data(self) -> npc_sync.SyncDataset:
        return npc_sync.SyncDataset(io.BytesIO(self.sync_path.read_bytes()))

    @property
    def stim_path_root(self) -> upath.UPath:
        return npc_lims.DR_DATA_REPO / str(self.id.subject)

    @npc_io.cached_property
    def stim_paths(self) -> tuple[upath.UPath, ...]:
        def is_valid_stim_file(p) -> bool:
            if not utils.is_stim_file(
                p, subject_spec=self.id.subject, date_spec=self.id.date
            ):
                return False
            if any(e in p.as_posix() for e in self.excluded_stim_file_names):
                return False
            if not self.is_sync:
                if self.task_stim_name not in p.stem:
                    return (
                        False  # only analyse the task stim file if we have no sync data
                    )
                return True
            if (
                dt := npc_session.DatetimeRecord(p.stem).dt
            ) < self.sync_data.start_time:
                return False
            if (
                dt > self.sync_data.stop_time  # this check first can save opening file
                or dt + datetime.timedelta(seconds=npc_stim.get_stim_duration(p))
                > self.sync_data.stop_time
            ):
                return False
            return True

        # look for stim files in raw data folder, then in stim file repo if none
        # are found. Rationale is that raw data folder may have curated set,
        # with some unwanted stim files removed
        for raw_data_paths in (self.raw_data_paths, self.stim_path_root.iterdir()):
            stim_paths = [p for p in raw_data_paths if is_valid_stim_file(p)]
            if stim_paths:
                break
        if not stim_paths:
            raise FileNotFoundError(
                f"Could not find stim files for {self.id} in raw data paths or {self.stim_path_root}"
            )
        if (
            len(
                tasks := sorted(
                    [p for p in stim_paths if self.task_stim_name in p.stem]
                )
            )
            > 1
        ):
            # ensure only one task file (e.g. 676909_2023-11-09 has two)
            logger.warning(
                f"{self.id} has multiple {self.task_stim_name} stim files. Only the first will be used."
            )
            for extra_task in tasks[1:]:
                stim_paths.remove(extra_task)
        return tuple(stim_paths)

    @npc_io.cached_property
    def rig(self) -> str:
        add_period = False  # sam's h5 files store "NP3" and "BEH.E"

        # sam probably relies on this format, but we might want to convert
        # to "NP.3" format at some point
        def _format(rig: str) -> str:
            if add_period and rig.startswith("NP") and rig[2] != ".":
                rig = ".".join(rig.split("NP"))
            return rig

        if (v := getattr(self, "_rig", None)) is not None:
            return _format(v)
        for hdf5 in itertools.chain(
            (self.task_data,) if self.is_task else (),
            (v for v in self.stim_data.values()),
        ):
            if rigName := hdf5.get("rigName", None):
                rig: str = rigName.asstr()[()]
                return _format(rig)
        if self.is_ephys:
            return _format(
                {
                    "W10DT713842": "NP0",
                    "W10DT713843": "NP1",
                    "W10DT713844": "NP2",
                    "W10DTM714205": "NP3",
                    "W10DT05516": "NP3",
                }[self.ephys_settings_xml_data.hostname.upper()]
            )
        raise AttributeError(
            f"Could not find rigName for {self.id} in stim files or ephys files"
        )

    @property
    def sam(self) -> DynRoutData:
        if getattr(self, "_sam", None) is None:
            self._sam = npc_samstim.get_sam(self.task_data)
        return self._sam

    @property
    def task_path(self) -> upath.UPath:
        return next(
            path for path in self.stim_paths if self.task_stim_name in path.stem
        )

    @property
    def task_data(self) -> h5py.File:
        # don't check if self.is_task here, as this is used to determine that!
        return self.stim_data[
            next(
                k
                for k in self.stim_data_without_timing_issues
                if self.task_stim_name in k
            )
        ]

    @property
    def task_version(self) -> str | None:
        return self.sam.taskVersion if isinstance(self.sam.taskVersion, str) else None

    @npc_io.cached_property
    def stim_data(self) -> npc_io.LazyDict[str, h5py.File]:
        def h5_dataset(path: upath.UPath) -> h5py.File:
            return h5py.File(io.BytesIO(path.read_bytes()), "r")

        return npc_io.LazyDict(
            (path.stem, (h5_dataset, (path,), {})) for path in self.stim_paths
        )

    @npc_io.cached_property
    def stim_data_without_timing_issues(
        self,
    ) -> dict[str, h5py.File] | npc_io.LazyDict[str, h5py.File]:
        """Checks stim files against sync (if available) and uses only those that have confirmed matches in number of vsyncs and order within the experiment"""
        if not self.is_sync:
            return self.stim_data
        return {
            k: v
            for k, v in self.stim_data.items()
            if not isinstance(self._stim_frame_times[k], Exception)
        }

    @property
    def video_paths(self) -> tuple[upath.UPath, ...]:
        if not self.is_sync:
            raise ValueError(
                f"{self.id} is not a session with sync data (required for video)"
            )
        return npc_mvr.get_video_file_paths(*self.raw_data_paths)

    @npc_io.cached_property
    def video_data(self) -> npc_io.LazyDict[str, cv2.VideoCapture]:
        return npc_io.LazyDict(
            (path.stem, (npc_mvr.get_video_data, (path,), {}))
            for path in self.video_paths
        )

    @property
    def video_info_paths(self) -> tuple[upath.UPath, ...]:
        return npc_mvr.get_video_info_file_paths(*self.raw_data_paths)

    @npc_io.cached_property
    def video_info_data(self) -> npc_io.LazyDict[str, npc_mvr.MVRInfoData]:
        return npc_io.LazyDict(
            (
                npc_mvr.get_camera_name(path.stem),
                (npc_mvr.get_video_info_data, (path,), {}),
            )
            for path in self.video_info_paths
        )

    @npc_io.cached_property
    def _stim_frame_times(self) -> dict[str, Exception | npt.NDArray[np.float64]]:
        """Frame times dict for all stims, containing time arrays or Exceptions."""
        frame_times = npc_stim.get_stim_frame_times(
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
        if self.ignore_stim_errors:
            return {
                path: times
                for path, times in self._stim_frame_times.items()
                if not isinstance(times, Exception)
            }
        asserted_stim_frame_times: dict[str, npt.NDArray[np.float64]] = {}
        for k, v in self._stim_frame_times.items():
            v = npc_stim.assert_stim_times(v)
            asserted_stim_frame_times[k] = v
        assert not any(
            isinstance(v, Exception) for v in asserted_stim_frame_times.values()
        )
        return asserted_stim_frame_times

    @property
    def ephys_record_node_dirs(self) -> tuple[upath.UPath, ...]:
        if getattr(self, "_ephys_record_node_dirs", None) is None:
            all_ = tuple(
                p
                for p in self.raw_data_paths
                if re.match(r"^Record Node [0-9]+$", p.name)
            )
            # if data has been reuploaded and lives in a modality subfolder as
            # well as the root, we want to use the modality subfolder
            modality = tuple(
                p for p in all_ if "/ecephys/ecephys_clipped" in p.as_posix()
            )
            self.ephys_record_node_dirs = modality or all_
        return self._ephys_record_node_dirs

    @ephys_record_node_dirs.setter
    def ephys_record_node_dirs(self, v: Iterable[npc_io.PathLike]) -> None:
        if isinstance(v, str) or not isinstance(v, Iterable):
            v = (v,)
        paths = tuple(npc_io.from_pathlike(path) for path in v)
        assert all("Record Node" in path.name for path in paths)
        self._ephys_record_node_dirs = paths

    @npc_io.cached_property
    def ephys_recording_dirs(self) -> tuple[upath.UPath, ...]:
        return tuple(
            p
            for record_node in self.ephys_record_node_dirs
            for p in record_node.glob("experiment*/recording*")
        )

    @npc_io.cached_property
    def ephys_timing_data(self) -> tuple[npc_ephys.EphysTimingInfo, ...]:
        pxi_data = (
            timing
            for timing in npc_ephys.get_ephys_timing_on_pxi(
                self.ephys_recording_dirs,
            )
            if (p := npc_session.extract_probe_letter(timing.device.name))
            is None  # nidaq
            or p in self.probe_letters_to_use
        )
        return tuple(
            npc_ephys.get_ephys_timing_on_sync(
                self.sync_data,
                devices=pxi_data,
            )
        )

    @npc_io.cached_property
    def drift_map_paths(self) -> tuple[upath.UPath, ...]:
        if npc_ephys.SpikeInterfaceKS25Data(self.id).is_pre_v0_99:
            return tuple(
                next(
                    d for d in self.sorted_data_paths if d.name == "drift_maps"
                ).iterdir()
            )

        return ()  # TODO: think about what to do, issue already open about making drift maps from scratch

    @npc_io.cached_property
    def ephys_sync_messages_path(self) -> upath.UPath:
        return next(
            p
            for p in itertools.chain(
                *(record_node.iterdir() for record_node in self.ephys_recording_dirs)
            )
            if "sync_messages.txt" == p.name
        )

    @npc_io.cached_property
    def ephys_nominal_start_time(self) -> datetime.datetime:
        """Start time from sync_messages.txt"""
        software_time_line = self.ephys_sync_messages_path.read_text().split("\n")[0]
        timestamp_value = float(
            software_time_line[software_time_line.index(":") + 2 :].strip()
        )
        timestamp = datetime.datetime.fromtimestamp(timestamp_value / 1e3)
        return timestamp

    @npc_io.cached_property
    def ephys_structure_oebin_paths(self) -> tuple[upath.UPath, ...]:
        return tuple(
            p
            for p in itertools.chain(
                *(record_node.iterdir() for record_node in self.ephys_recording_dirs)
            )
            if "structure.oebin" == p.name
        )

    @npc_io.cached_property
    def ephys_structure_oebin_data(
        self,
    ) -> dict[Literal["continuous", "events", "spikes"], list[dict[str, Any]]]:
        return npc_ephys.get_merged_oebin_file(self.ephys_structure_oebin_paths)

    @npc_io.cached_property
    def ephys_sync_messages_data(
        self,
    ) -> dict[str, dict[Literal["start", "rate"], int]]:
        return npc_ephys.get_sync_messages_data(self.ephys_sync_messages_path)

    @npc_io.cached_property
    def ephys_experiment_dirs(self) -> tuple[upath.UPath, ...]:
        return tuple(
            p
            for record_node in self.ephys_record_node_dirs
            for p in record_node.glob("experiment*")
        )

    @npc_io.cached_property
    def ephys_settings_xml_paths(self) -> tuple[upath.UPath, ...]:
        if not self.is_ephys:
            raise ValueError(
                f"{self.id} is not an ephys session (required for settings.xml)"
            )
        return tuple(
            next(record_node.glob("settings*.xml"))
            for record_node in self.ephys_record_node_dirs
        )

    @npc_io.cached_property
    def ephys_settings_xml_path(self) -> upath.UPath:
        """Single settings.xml path, if applicable"""
        if not self.ephys_settings_xml_paths:
            raise ValueError(
                f"settings.xml not found for {self.id} - check status of raw upload"
            )
        npc_ephys.assert_xml_files_match(*self.ephys_settings_xml_paths)
        return self.ephys_settings_xml_paths[0]

    @npc_io.cached_property
    def ephys_settings_xml_data(self) -> npc_ephys.SettingsXmlInfo:
        return npc_ephys.get_settings_xml_data(self.ephys_settings_xml_path)

    @property
    def electrode_group_description(self) -> str:
        # TODO get correct channels range from settings xml
        return "Neuropixels 1.0 lower channels (1:384)"

    @property
    def probe_letters_with_surface_channel_recording(
        self,
    ) -> tuple[npc_session.ProbeRecord, ...]:
        try:
            _ = self.surface_recording
        except AttributeError:
            return ()
        return self.surface_recording.probe_letters_to_use

    @property
    def probe_letters_with_sorted_data(self) -> tuple[npc_session.ProbeRecord, ...]:
        if not self.is_sorted:
            return ()
        return self.sorted_data.probes

    @property
    def probe_letters_skipped_by_sorting(self) -> tuple[npc_session.ProbeRecord, ...]:
        if not self.is_sorted:
            return ()
        return tuple(
            npc_session.ProbeRecord(p)
            for p in "ABCDEF"
            if p not in self.sorted_data.probes
        )

    @property
    def probe_letters_to_skip(self) -> tuple[npc_session.ProbeRecord, ...]:
        """Includes probes skipped by sorting"""
        if (v := getattr(self, "_probe_letters_to_skip", None)) is not None:
            probe_letters_to_skip = tuple(
                npc_session.ProbeRecord(letter) for letter in v
            )
        else:
            probe_letters_to_skip = ()
        return probe_letters_to_skip + self.probe_letters_skipped_by_sorting

    def remove_probe_letters_to_skip(
        self, letters: Iterable[str | npc_session.ProbeRecord]
    ) -> tuple[npc_session.ProbeRecord, ...]:
        return tuple(
            npc_session.ProbeRecord(letter)
            for letter in letters
            if npc_session.ProbeRecord(letter) not in self.probe_letters_to_skip
        )

    @npc_io.cached_property
    def probe_insertions(self) -> dict[str, Any] | None:
        if self.probe_insertion_info:
            return self.probe_insertion_info["probes"]
        return None

    @npc_io.cached_property
    def probe_insertion_info(self) -> dict[str, Any] | None:
        d = None
        with contextlib.suppress(ValueError):
            return npc_lims.get_probe_insertion_metadata(self.id)
        if d is None:
            path = next(
                (
                    path
                    for path in self.raw_data_paths
                    if path.name == "probe_insertions.json"
                ),
                None,
            )
            if path:
                d = json.loads(path.read_text())["probe_insertions"]
        if d is not None:
            key_to_probe = {}
            for k in d:
                with contextlib.suppress(ValueError):
                    key_to_probe[k] = npc_session.ProbeRecord(k)
            return {
                "probes": {key_to_probe[k]: d[k]["hole"] for k in key_to_probe},
                "notes": {key_to_probe[k]: d[k].get("notes") for k in key_to_probe},
                "implant": d["implant"],
            }
        path = next(
            (path for path in self.raw_data_paths if path.name == "insertions.json"),
            None,
        )
        if path:
            return json.loads(path.read_text())
        return None

    @npc_io.cached_property
    def probes_inserted(self) -> tuple[str, ...]:
        """('probeA', 'probeB', ...)"""
        return tuple(probe.name for probe in self.probe_letters_to_use)

    @npc_io.cached_property
    def probe_letters_to_use(self) -> tuple[npc_session.ProbeRecord, ...]:
        """('A', 'B', ...)"""
        from_annotation = from_insertion_record = None
        if self.is_annotated:
            from_annotation = tuple(
                npc_session.ProbeRecord(probe)
                for probe in utils.get_tissuecyte_electrodes_table(
                    self.id
                ).group_name.unique()
            )
            from_annotation = self.remove_probe_letters_to_skip(from_annotation)
        if self.probe_insertions is not None:
            from_insertion_record = tuple(
                npc_session.ProbeRecord(k)
                for k, v in self.probe_insertions.items()
                if v is not None
            )
            from_insertion_record = self.remove_probe_letters_to_skip(
                from_insertion_record
            )
        if from_annotation and from_insertion_record:
            if set(from_annotation).symmetric_difference(set(from_insertion_record)):
                logger.warning(
                    f"probe_insertions.json and annotation info do not match for {self.id} - using annotation info"
                )
        if from_annotation:
            return from_annotation
        if from_insertion_record:
            return from_insertion_record
        logger.warning(
            f"No probe_insertions.json or annotation info found for {self.id} - defaulting to ABCDEF"
        )
        return self.remove_probe_letters_to_skip("ABCDEF")

    @property
    def implant(self) -> str | None:
        if self.probe_insertion_info is None:
            # TODO get from sharepoint
            return None
        return self.probe_insertion_info["shield"]["name"]

    @npc_io.cached_property
    def _all_licks(self) -> tuple[ndx_events.Events, ...]:
        """First item is always `processing['licks']` - the following items are only if sync
        is available, and use raw rising/falling edges of the lick sensor to get lick duration
        for `acquisition`.

        If sync isn't available, we only have start frames of licks, so we can't
        filter by duration very accurately.
        """
        if not self.is_task:
            return ()
        if self.is_sync:
            max_contact = (
                0.5  # must factor-in lick_sensor staying high after end of contact
            )
            # https://www.nature.com/articles/s41586-021-03561-9/figures/1

            licks_on_sync: bool = False
            try:
                rising = self.sync_data.get_rising_edges("lick_sensor", units="seconds")
                falling = self.sync_data.get_falling_edges(
                    "lick_sensor", units="seconds"
                )
            except IndexError:
                logger.debug(f"No licks on sync line for {self.id}")
            else:
                licks_on_sync = rising.size > 0 and falling.size > 0
            if licks_on_sync:
                if falling[0] < rising[0]:
                    falling = falling[1:]
                if rising[-1] > falling[-1]:
                    rising = rising[:-1]
                assert len(rising) == len(falling)

                rising_falling = np.array([rising, falling]).T
                lick_duration = np.diff(rising_falling, axis=1).squeeze()

                filtered_idx = lick_duration <= max_contact

                # # remove licks that aren't part of a sequence of licks at at least ~3 Hz
                # max_interval = 0.5
                # for i, (r, f) in enumerate(rising_falling):
                #     prev_end = rising_falling[i-1, 1] if i > 0 else None
                #     next_start = rising_falling[i+1, 0] if i < len(rising_falling) - 1 else None
                #     if (
                #         (prev_end is None or r - prev_end > max_interval)
                #         and
                #         (next_start is None or next_start - f > max_interval)
                #     ):
                #         filtered_idx[i] = False

        description = "times at which the subject made contact with a combined lick sensor + water spout: putatively the starts of licks, but may include other events such as grooming"
        duration_description = "`data` contains the duration of each event"
        if self.is_sync and licks_on_sync:
            licks = pynwb.TimeSeries(
                timestamps=rising[filtered_idx],
                data=falling[filtered_idx] - rising[filtered_idx],
                name="licks",
                description=f"{description}; filtered to exclude events with duration >{max_contact} s; {duration_description}",
                unit="seconds",
            )
        else:
            licks = ndx_events.Events(
                timestamps=self.sam.lickTimes,
                name="licks",
                description=description,
            )

        if not (self.is_sync and licks_on_sync):
            return (licks,)
        return (
            licks,
            pynwb.TimeSeries(
                timestamps=rising,
                data=falling - rising,
                name="lick_sensor_events",
                description=f"{description}; {duration_description}",
                unit="seconds",
            ),
        )

    @npc_io.cached_property
    def _running_speed(self) -> pynwb.TimeSeries:
        name = "running_speed"
        description = (
            "linear forward running speed on a rotating disk, low-pass filtered "
            f"at {npc_stim.RUNNING_LOWPASS_FILTER_HZ} Hz with a 3rd order Butterworth filter"
        )
        unit = npc_stim.RUNNING_SPEED_UNITS
        # comments = f'Assumes mouse runs at `radius = {npc_stim.RUNNING_DISK_RADIUS} {npc_stim.RUNNING_SPEED_UNITS.split("/")[0]}` on disk.'
        data, timestamps = npc_stim.get_running_speed_from_stim_files(
            *self.stim_data_without_timing_issues.values(),
            sync=self.sync_data if self.is_sync else None,
            filt=npc_stim.lowpass_filter,
        )
        return pynwb.TimeSeries(
            name=name,
            description=description,
            data=data,
            timestamps=timestamps,
            unit=unit,
        )

    def get_all_spike_histogram(
        self, electrode_group_name: str | npc_session.ProbeRecord
    ) -> pynwb.TimeSeries:
        probe = npc_session.ProbeRecord(electrode_group_name)
        units: pd.DataFrame = (
            self.units[:]
            .query("default_qc")
            .query("electrode_group_name == @electrode_group_name")
        )
        bin_interval = 1  # seconds
        hist, bin_edges = npc_ephys.bin_spike_times(
            units["spike_times"].to_numpy(), bin_interval=bin_interval
        )

        return pynwb.TimeSeries(
            name=probe.name,
            description=f"joint spike rate across all good units on {probe.name}, binned at {bin_interval} second intervals",
            data=hist,
            timestamps=(bin_edges[:-1] + bin_edges[1:]) / 2,
            unit="spikes/s",
            resolution=1.0,
        )

    @npc_io.cached_property
    def all_spike_histograms(self) -> pynwb.core.MultiContainerInterface:
        ## using this as a generic multi-timeseries container
        # class BehavioralEvents(MultiContainerInterface):
        #     __clsconf__ = {
        #         'add': 'add_timeseries',
        #         'get': 'get_timeseries',
        #         'create': 'create_timeseries',
        #         'type': pynwb.TimeSeries,
        #         'attr': 'time_series'
        #     }
        module = pynwb.behavior.BehavioralEvents(
            name="all_spike_histograms",
        )
        for probe in sorted(set(self.units.electrode_group_name[:])):
            module.add_timeseries(self.get_all_spike_histogram(probe))
        return module

    @npc_io.cached_property
    def _video_frame_times(
        self,
    ) -> tuple[pynwb.core.NWBDataInterface | pynwb.core.DynamicTable, ...]:
        path_to_timestamps = npc_mvr.get_video_frame_times(
            self.sync_data, *self.video_paths
        )

        return tuple(
            ndx_events.Events(
                timestamps=timestamps,
                name=f"frametimes_{self.mvr_to_nwb_camera_name[npc_mvr.get_camera_name(path.stem)]}",
                description=f"start time of each frame exposure in {path.stem}",
            )
            for path, timestamps in path_to_timestamps.items()
        )

    @npc_io.cached_property
    def _LPFaceParts(self) -> tuple[pynwb.core.DynamicTable, ...]:
        """
        Stores the lightning pose output as a dynamic table for each of the relevant cameras (side, face)
        For each camera, 3 tables - the predictions, the pca error, and the temporal norm
        """
        LP_face_parts_dynamic_tables = []
        if not self.is_video:
            raise ValueError(f"{self.id} is not a session with video")

        for video_path in self.video_paths:
            camera_name = npc_mvr.get_camera_name(video_path.name)
            if camera_name == "eye":
                continue

            nwb_camera_name = self.mvr_to_nwb_camera_name[camera_name]
            timestamps = next(
                t for t in self._video_frame_times if nwb_camera_name in t.name
            ).timestamps
            assert len(timestamps) == npc_mvr.get_total_frames_in_video(video_path)

            for result_name in utils.LP_RESULT_TYPES:
                df = utils.get_LPFaceParts_result_dataframe(
                    self.id, utils.LP_MAPPING[camera_name], result_name
                )
                if len(timestamps) != len(df):
                    logger.warning(
                        f"{self.id} {camera_name} {result_name} lightning pose face parts output has wrong shape {len(df)}, expected {len(timestamps)} frames."
                        "\nLightning pose face parts capsule was likely run with an additional data asset attached"
                    )
                    continue

                df["timestamps"] = timestamps
                name = f"Lightning_Pose_FaceParts_{nwb_camera_name}_{result_name}"

                if result_name == "predictions":
                    table_description = (
                        f"Lightning Pose tracking model fit to {len(utils.LP_VIDEO_FEATURES_MAPPING[utils.LP_MAPPING[camera_name]])} facial features for each frame of {nwb_camera_name} video. "
                        "Output for every frame is x,y coordinates in pixels along with the likelihood of the model for each feature in the frame. "
                        f"Features tracked are {utils.LP_VIDEO_FEATURES_MAPPING[utils.LP_MAPPING[camera_name]]} "
                    )
                else:
                    table_description = f"Lightning Pose {nwb_camera_name} {utils.LP_RESULT_DESCRIPTIONS[result_name]}"

                table = pynwb.core.DynamicTable.from_dataframe(
                    name=name, table_description=table_description, df=df
                )
                LP_face_parts_dynamic_tables.append(table)

        return tuple(LP_face_parts_dynamic_tables)

    @npc_io.cached_property
    def _eye_tracking(self) -> pynwb.core.DynamicTable:
        if not self.is_video:
            raise ValueError(f"{self.id} is not a session with video")
        timestamps = next(
            t for t in self._video_frame_times if "eye" in t.name
        ).timestamps
        df = utils.get_ellipse_session_dataframe_from_h5(self.id)
        df["timestamps"] = timestamps
        name = "eye_tracking"
        table_description = (
            "Ellipses fit to three features in each frame of the eye video: "
            "pupil: perimeter of the pupil | eye: inner perimeter of the eyelid | cr: corneal reflection, a bright spot near the center of the eye which is always smaller than the pupil"
        )
        column_descriptions = {}
        for feature in ("cr", "eye", "pupil"):
            feature_name = "corneal reflection" if feature == "cr" else feature
            for column_suffix, description in dict(
                center_x=f"center of {feature_name} ellipse in pixels, with (0, 0) at top-left of frame",
                center_y=f"center of {feature_name} ellipse in pixels, with (0, 0) at top-left of frame",
                area=f"area of {feature_name} ellipse in pixels^2",
                width=f"length of semi-major axis of {feature_name} ellipse in pixels",
                height=f"length of semi-minor axis of {feature_name} ellipse in pixels",
                phi=f"counterclockwise rotation of major-axis of {feature_name} ellipse, relative to horizontal-axis of video, in radians",
                average_confidence=f"mean confidence [0-1] for the up-to-12 points from DLC used to fit {feature_name} ellipse",
                is_bad_frame=f"[bool] frames which should not be used due to low confidence in {feature_name} ellipse (typically caused by blinking, grooming, poor lighting)",
            ).items():
                column_descriptions[f"{feature}_{column_suffix}"] = description

        return pynwb.core.DynamicTable.from_dataframe(
            name=name,
            df=df,
            columns=None,
            table_description=table_description,
            column_descriptions=column_descriptions,
        )

    @npc_io.cached_property
    def _facemap(self) -> tuple[pynwb.TimeSeries, ...]:
        facemap_series = []
        for video_path in self.video_paths:
            camera_name = npc_mvr.get_camera_name(video_path.name)
            if camera_name == "eye":
                continue
            nwb_camera_name = self.mvr_to_nwb_camera_name[camera_name]
            timestamps = next(
                t for t in self._video_frame_times if nwb_camera_name in t.name
            ).timestamps
            assert len(timestamps) == npc_mvr.get_total_frames_in_video(video_path)
            try:
                face_motion_svd = utils.get_facemap_output_from_s3(
                    self.id, camera_name=camera_name, array_name="motSVD"
                )
            except FileNotFoundError:
                logger.warning(f"{camera_name} Facemap has not been run for {self.id}")
                continue
            if face_motion_svd.shape[0] != len(timestamps):
                logger.warning(
                    f"{self.id} {camera_name} Facemap output has wrong shape {face_motion_svd.shape}, expected {len(timestamps)} frames."
                    "\nFacemap capsule was likely run with an additional data asset attached"
                )
                continue
            facemap_series.append(
                pynwb.TimeSeries(
                    name=f"facemap_{nwb_camera_name}",
                    data=face_motion_svd,
                    unit="pixels",
                    timestamps=timestamps,
                    description=f"motion SVD for video from {nwb_camera_name.replace('_', ' ')}; shape is number of frames by number of components ({face_motion_svd.shape[1]})",
                )
            )
        return tuple(facemap_series)

    @npc_io.cached_property
    def _dlc(self) -> tuple[ndx_pose.pose.PoseEstimation, ...]:
        if not self.is_video:
            return ()
        camera_to_dlc_model = {
            "eye": "dlc_eye",
            "face": "dlc_face",
            "behavior": "dlc_side",
        }
        pose_estimations = []
        for video_path in self.video_paths:
            camera_name = npc_mvr.get_camera_name(video_path.name)
            nwb_camera_name = self.mvr_to_nwb_camera_name[camera_name]
            try:
                df = utils.get_dlc_session_model_dataframe_from_h5(
                    self.id,
                    model_name=camera_to_dlc_model[camera_name],
                )
            except FileNotFoundError:
                logger.warning(f"{camera_name} DLC has not been run for {self.id}")
                continue
            timestamps = next(
                t for t in self._video_frame_times if nwb_camera_name in t.name
            ).timestamps
            assert len(timestamps) == npc_mvr.get_total_frames_in_video(video_path)
            pose_estimation_series = utils.get_pose_series_from_dataframe(
                self.id, df, timestamps
            )
            pose_estimations.append(
                ndx_pose.pose.PoseEstimation(
                    name=f"dlc_{nwb_camera_name}",
                    pose_estimation_series=pose_estimation_series,
                    description=f"DeepLabCut analysis of video from {nwb_camera_name.replace('_', ' ')}",
                    original_videos=[video_path.as_posix()],
                    source_software="DeepLabCut",
                )
            )
        return tuple(pose_estimations)

    @npc_io.cached_property
    def _reward_frame_times(
        self,
    ) -> pynwb.core.NWBDataInterface | pynwb.core.DynamicTable:
        """As interpreted from task stim files, with timing corrected with sync if
        available.

        - includes manualRewardFrames
        """

        def get_reward_frames(data: h5py.File) -> list[int]:
            r = []
            for key in ("rewardFrames", "manualRewardFrames"):
                if (v := data.get(key, None)) is not None:
                    r.extend(v[:])
            return r

        reward_times: list[npt.NDArray[np.floating]] = []
        for stim_file, stim_data in self.stim_data_without_timing_issues.items():
            if any(name in stim_file.lower() for name in ("mapping", "tagging")):
                continue
            reward_times.extend(
                npc_stim.safe_index(
                    npc_stim.get_flip_times(
                        stim=stim_data,
                        sync=self.sync_data if self.is_sync else None,
                    ),
                    get_reward_frames(stim_data),
                )
            )
        return ndx_events.Events(
            timestamps=np.sort(np.unique(reward_times)),
            name="rewards",
            description="times at which the stimulus script triggered water rewards to be delivered to the subject",
        )

    @npc_io.cached_property
    def _reward_times_with_duration(self) -> pynwb.TimeSeries:
        """As interpreted from sync, after solenoid line was added ~March 2024."""
        if not self.is_sync:
            raise AttributeError(f"{self.id} is not a session with sync data")
        if self.id.date < datetime.date(2024, 4, 1):
            raise AttributeError(f"{self.id} does not have reward duration data")
        rising = self.sync_data.get_rising_edges(15, units="seconds")
        falling = self.sync_data.get_falling_edges(15, units="seconds")
        if falling[0] < rising[0]:
            falling = falling[1:]
        if rising[-1] > falling[-1]:
            rising = rising[:-1]
        assert len(rising) == len(falling)
        return pynwb.TimeSeries(
            timestamps=rising,
            data=falling - rising,
            name="rewards",
            unit="seconds",
            description="times at which the solenoid valve that controls water reward delivery to the subject was opened; `data` contains the length of time the solenoid was open for each event",
        )

    @npc_io.cached_property
    def _quiescent_interval_violations(
        self,
    ) -> pynwb.core.NWBDataInterface | pynwb.core.DynamicTable:
        frames: npt.NDArray[np.int32] = self.sam.quiescentViolationFrames
        times: npt.NDArray[np.floating] = npc_stim.safe_index(
            npc_stim.get_input_data_times(
                stim=self.task_data,
                sync=self.sync_data if self.is_sync else None,
            ),
            frames,
        )
        return ndx_events.Events(
            timestamps=np.sort(np.unique(times)),
            name="quiescent_interval_violations",
            description="times at which the subject made contact with the lick spout during a quiescent interval, triggering a restart of the trial",
        )

    @npc_io.cached_property
    def surface_root_path(self) -> upath.UPath:
        if self.root_path:
            surface_channel_root = (
                self.root_path.parent / f"{self.root_path.name}_surface_channels"
            )
            if not surface_channel_root.exists():
                raise FileNotFoundError(
                    f"Could not find surface channel root at expected {surface_channel_root}"
                )
        else:
            surface_channel_root = npc_lims.get_surface_channel_root(self.id)
        return surface_channel_root

    @npc_io.cached_property
    def surface_recording(self) -> DynamicRoutingSurfaceRecording:
        if not self.is_surface_channels:
            raise AttributeError(
                f"{self.id} is not a session with a surface channel recording"
            )
        return DynamicRoutingSurfaceRecording(
            self.surface_root_path,
            **self.kwargs,
        )

    @npc_io.cached_property
    def _aind_reward_delivery(
        self,
    ) -> aind_data_schema.core.session.RewardDeliveryConfig:
        return aind_data_schema.core.session.RewardDeliveryConfig(
            reward_solution="Water",
            reward_spouts=[
                aind_data_schema.core.session.RewardSpoutConfig(
                    side="Center",
                    starting_position=aind_data_schema.models.coordinates.RelativePosition(
                        device_position_transformations=[
                            aind_data_schema.models.coordinates.Translation3dTransform(
                                translation=[0.0, 0.0, 0.0],
                            )
                        ],
                        device_origin="Located on the tip of the spout (which is also the lick sensor), centered in front of the subject's mouth",
                        device_axes=[
                            aind_data_schema.models.coordinates.Axis(
                                name="X",
                                direction="Positive is from the centerline of the subject's mouth towards its right",
                            ),
                            aind_data_schema.models.coordinates.Axis(
                                name="Y",
                                direction="Positive is from the centerline of the subject's mouth towards the sky",
                            ),
                            aind_data_schema.models.coordinates.Axis(
                                name="Z",
                                direction="Positive is from the anterior-most part of the subject's mouth towards its tail",
                            ),
                        ],
                    ),
                    variable_position=False,
                )
            ],
        )

    @npc_io.cached_property
    def _aind_data_streams(self) -> tuple[aind_data_schema.core.session.Stream, ...]:
        data_streams = []
        # sync, mvr cameras, ephys probes
        modality = aind_data_schema.models.modalities.Modality
        if self.is_sync:
            data_streams.append(
                aind_data_schema.core.session.Stream(
                    stream_start_time=self.sync_data.start_time,
                    stream_end_time=self.sync_data.stop_time,
                    stream_modalities=[modality.BEHAVIOR],
                    daq_names=["Sync"],
                )
            )
        if self.is_video:
            data_streams.append(
                aind_data_schema.core.session.Stream(
                    stream_start_time=self.session_start_time
                    + datetime.timedelta(
                        seconds=min(
                            np.nanmin(times.timestamps)
                            for times in self._video_frame_times
                        )
                    ),
                    stream_end_time=self.session_start_time
                    + datetime.timedelta(
                        seconds=max(
                            np.nanmax(times.timestamps)
                            for times in self._video_frame_times
                        )
                    ),
                    camera_names=["Front camera", "Side camera", "Eye camera"],
                    stream_modalities=[modality.BEHAVIOR_VIDEOS],
                )
            )
        if self.is_ephys:

            data_streams.append(
                aind_data_schema.core.session.Stream(
                    stream_start_time=self.session_start_time
                    + datetime.timedelta(
                        seconds=min(
                            timing.start_time for timing in self.ephys_timing_data
                        )
                    ),
                    stream_end_time=self.session_start_time
                    + datetime.timedelta(
                        seconds=max(
                            timing.stop_time for timing in self.ephys_timing_data
                        )
                    ),
                    ephys_modules=(
                        ephys_modules := [
                            aind_data_schema.core.session.EphysModule(
                                assembly_name=probe.name.upper(),
                                arc_angle=0.0,
                                module_angle=0.0,
                                rotation_angle=0.0,
                                primary_targeted_structure="none",
                                manipulator_coordinates=(
                                    (
                                        aind_data_schema.models.coordinates.Coordinates3d(
                                            x=(
                                                row := self._manipulator_positions.to_dataframe().query(
                                                    f"electrode_group == '{probe.name}'"
                                                )
                                            )["x"].item(),
                                            y=row["y"].item(),
                                            z=row["z"].item(),
                                            unit="micrometer",
                                        )
                                    )  # some old sessions didn't have newscale logging enabled: no way to get their coords
                                    if hasattr(self, "_manipulator_info")
                                    else aind_data_schema.models.coordinates.Coordinates3d(
                                        x=0.0,
                                        y=0.0,
                                        z=0.0,
                                        unit="micrometer",
                                    )
                                ),
                                ephys_probes=[
                                    aind_data_schema.core.session.EphysProbeConfig(
                                        name=probe.name.upper(),
                                    )
                                ],
                            )
                            for probe in self.probe_letters_to_use
                        ]
                    ),
                    stick_microscopes=ephys_modules,  # cannot create ecephys modality without stick microscopes
                    stream_modalities=[modality.ECEPHYS],
                )
            )
        return tuple(data_streams)

    @npc_io.cached_property
    def _aind_stimulus_epochs(
        self,
    ) -> tuple[aind_data_schema.core.session.StimulusEpoch, ...]:

        def get_modalities(
            epoch_name: str,
        ) -> list[aind_data_schema.core.session.StimulusModality]:
            stim = aind_data_schema.core.session.StimulusModality
            modalities = []
            if any(
                name in epoch_name
                for name in (
                    "DynamicRouting",
                    "RFMapping",
                    "LuminanceTest",
                    "Spontaneous",
                )
            ):
                modalities.append(stim.VISUAL)
            if any(name in epoch_name for name in ("DynamicRouting", "RFMapping")):
                modalities.append(stim.AUDITORY)
            if self.is_opto and any(name in epoch_name for name in ("DynamicRouting",)):
                modalities.append(stim.OPTOGENETICS)
            if any(name in epoch_name for name in ("OptoTagging",)):
                modalities.append(stim.OPTOGENETICS)
            return modalities or [stim.NONE]

        def get_num_trials(epoch_name: str) -> int | None:
            intervals_name: str = utils.get_taskcontrol_intervals_table_name(epoch_name)
            if epoch_name == "RFMapping":
                return sum(
                    len(trials)
                    for name, trials in self.intervals.items()
                    if utils.get_taskcontrol_intervals_table_name("RFMapping") in name
                )
            if epoch_name == self.task_stim_name:
                trials = self.trials
            else:
                trials = self.intervals.get(intervals_name)
            if trials is None:
                return None
            return len(trials)

        def get_device_names(epoch_name: str) -> list[str]:
            stim = aind_data_schema.core.session.StimulusModality
            modalities = get_modalities(epoch_name)
            device_names = []
            if stim.VISUAL in modalities:
                device_names.append("Stim")
            if stim.AUDITORY in modalities:
                device_names.append("Speaker")
            if stim.OPTOGENETICS in modalities:
                device_names.append("Laser #0")  # TODO detect if second laser used
            return device_names

        def get_speaker_config(
            epoch_name: str,
        ) -> aind_data_schema.core.session.SpeakerConfig | None:
            stim = aind_data_schema.core.session.StimulusModality
            modalities = get_modalities(epoch_name)
            if stim.AUDITORY not in modalities:
                return None
            return aind_data_schema.core.session.SpeakerConfig(
                name="Speaker",
                volume=68.0,
                volume_unit="decibels",
            )

        def get_parameters(
            epoch_name: str,
        ) -> list[Any] | None:  # no baseclass for stim param classes
            stim = aind_data_schema.core.session.StimulusModality
            stimulus = aind_data_schema.models.stimulus
            modalities = get_modalities(epoch_name)
            if modalities == [stim.NONE]:
                return None
            parameters = []
            if epoch_name == "DynamicRouting1":
                parameters.extend(
                    [
                        stimulus.VisualStimulation(
                            stimulus_name="target and non-target visual grating stimuli",
                            stimulus_parameters={
                                "orientations_deg": [0, 90],
                                "position_xy": (0, 0),
                                "size_deg": 50,
                                "spatial_frequency_cycles_per_deg": 0.04,
                                "temporal_frequency_cycles_per_sec": 2,
                                "type": "sqr",
                                "phase": [0.0, 0.5],
                            },
                        ),
                        stimulus.AuditoryStimulation(
                            sitmulus_name="target amplitude-modulated noise stimulus",
                            sample_frequency=10_000,
                            amplitude_modulation_frequency=70,
                        ),
                        stimulus.AuditoryStimulation(
                            sitmulus_name="non-target amplitude-modulated noise stimulus",
                            sample_frequency=10_000,
                            amplitude_modulation_frequency=12,
                        ),
                    ]
                )
            if epoch_name == "RFMapping":
                parameters.extend(
                    [
                        stimulus.VisualStimulation(
                            stimulus_name="receptive-field mapping grating stimuli",
                            stimulus_parameters={
                                "orientations_deg": [
                                    0,
                                    45,
                                    90,
                                    135,
                                    180,
                                    225,
                                    270,
                                    315,
                                ],
                                "position_x": list(
                                    np.unique(
                                        self.intervals[
                                            utils.get_taskcontrol_intervals_table_name(
                                                "VisRFMapping"
                                            )
                                        ].grating_x
                                    )
                                ),
                                "position_y": list(
                                    np.unique(
                                        self.intervals[
                                            utils.get_taskcontrol_intervals_table_name(
                                                "VisRFMapping"
                                            )
                                        ].grating_y
                                    )
                                ),
                                "size_deg": 20,
                                "spatial_frequency_cycles_per_deg": 0.08,
                                "temporal_frequency_cycles_per_sec": 4,
                                "type": "sqr",
                                "duration_sec": 0.25,
                            },
                        ),
                        stimulus.VisualStimulation(
                            stimulus_name="full-field flash stimuli",
                            stimulus_parameters={
                                "gray_level": [-1, 0, 1],
                                "duration_sec": 0.25,
                            },
                            notes="-1.0: black | 0: mid-gray | 1.0: white",
                        ),
                        *[
                            stimulus.AuditoryStimulation(
                                sitmulus_name=f"receptive-field mapping amplitude-modulated noise stimulus {idx}",
                                sample_frequency=10_000,
                                amplitude_modulation_frequency=freq,
                            )
                            for idx, freq in enumerate([12, 20, 40, 80])
                        ],
                    ]
                )

            if epoch_name == "OptoTagging":
                # square waves of different lengths
                parameters.extend(
                    [
                        stimulus.OptoStimulation(
                            stimulus_name="short optotagging stimulus",
                            pulse_shape="Square",
                            pulse_frequency=[100],
                            number_pulse_trains=[1],
                            pulse_width=[10],  # ms
                            pulse_train_duration=[0.01],  # s
                            fixed_pulse_train_interval=False,
                            baseline_duration=0.2,  # s
                        ),
                        stimulus.OptoStimulation(
                            stimulus_name="long optotagging stimulus",
                            pulse_shape="Square",
                            pulse_frequency=[5],
                            number_pulse_trains=[1],
                            pulse_width=[200],  # ms
                            pulse_train_duration=[0.2],  # s
                            fixed_pulse_train_interval=False,
                            baseline_duration=0.2,  # s
                        ),
                    ]
                )
            if "Spontaneous" in epoch_name:
                # blank screen constant lum
                parameters.append(
                    stimulus.VisualStimulation(
                        stimulus_name="blank screen, constant luminance",
                        stimulus_parameters={
                            "gray_level": (
                                -0.95
                                if self.session_start_time.timestamp()
                                > datetime.datetime(2024, 4, 1).timestamp()
                                else -1.0
                            ),
                        },
                        notes="-1.0: black | 0: mid-gray | 1.0: white",
                    )
                )
            return parameters

        def get_num_trials_rewarded(epoch_name: str) -> int | None:
            if "DynamicRouting" in epoch_name:
                return len(self.sam.rewardTimes)
            return None

        def get_reward_consumed(epoch_name: str) -> float | None:
            num_trials_rewarded = get_num_trials_rewarded(epoch_name)
            if num_trials_rewarded is None:
                return None
            return np.nanmean(self.sam.rewardSize) * num_trials_rewarded

        def get_version() -> str | None:
            if "blob/main" in self.source_script:
                return None
            return (
                self.source_script.split("DynamicRoutingTask/")[-1]
                .split("/DynamicRouting1.py")[0]
                .strip("/")
            )

        def get_url(epoch_name: str) -> str:
            return self.source_script.replace(
                "DynamicRouting1", get_taskcontrol_file(epoch_name)
            )

        def get_taskcontrol_file(epoch_name: str) -> str:
            if upath.UPath(
                self.source_script.replace("DynamicRouting1", epoch_name)
            ).exists():
                return epoch_name
            return "TaskControl"

        def get_task_metrics() -> dict[str, dict]:
            blocks = {
                str(block_index): dict(
                    block_index=block_index,
                    block_stim_rewarded=block_stim_rewarded,
                    hit_count=hit_count,
                    dprime_same_modal=dprime_same_modal,
                    dprime_other_modal_go=dprime_other_modal_go,
                )
                for block_index, (
                    hit_count,
                    dprime_same_modal,
                    dprime_other_modal_go,
                    block_stim_rewarded,
                ) in enumerate(
                    zip(
                        [int(v) for v in self.sam.hitCount],
                        [float(v) for v in self.sam.dprimeSameModal],
                        [float(v) for v in self.sam.dprimeOtherModalGo],
                        [str(v) for v in self.sam.blockStimRewarded],
                    )
                )
            }
            return {"block_metrics": blocks, "task_version": self.sam.taskVersion}

        aind_epochs = []
        for nwb_epoch in self.epochs:
            epoch_name = nwb_epoch.stim_name.item()

            aind_epochs.append(
                aind_data_schema.core.session.StimulusEpoch(
                    stimulus_start_time=datetime.timedelta(
                        seconds=nwb_epoch.start_time.item()
                    )
                    + self.session_start_time,
                    stimulus_end_time=datetime.timedelta(
                        seconds=nwb_epoch.stop_time.item()
                    )
                    + self.session_start_time,
                    stimulus_name=epoch_name,
                    software=[
                        aind_data_schema.models.devices.Software(
                            name="PsychoPy",
                            version="2022.1.2",
                            url="https://www.psychopy.org/",
                        ),
                    ],
                    script=aind_data_schema.models.devices.Software(
                        name=get_taskcontrol_file(epoch_name),
                        version=get_version() or "unknown",
                        url=get_url(epoch_name),
                    ),
                    stimulus_modalities=get_modalities(epoch_name),
                    stimulus_parameters=get_parameters(epoch_name),
                    stimulus_device_names=get_device_names(epoch_name),
                    speaker_config=get_speaker_config(epoch_name),
                    reward_consumed_during_epoch=get_reward_consumed(epoch_name),
                    trials_finished=get_num_trials(epoch_name),
                    trials_rewarded=get_num_trials_rewarded(epoch_name),
                    reward_consumed_unit="milliliter",
                    trials_total=get_num_trials(epoch_name),
                    notes=nwb_epoch.notes.item(),
                    output_parameters=(
                        get_task_metrics() if "DynamicRouting" in epoch_name else {}
                    ),
                )
            )
        return tuple(aind_epochs)

    @npc_io.cached_property
    def _aind_rig_id(self) -> str:
        rig = self.rig.strip(".")
        # - room can't be found out in the cloud, and hard-coded map is not allowed
        # - last update time could be worked out from document db in future
        last_updated = "240401"
        return f"unknown_{rig}_{last_updated}"

    @npc_io.cached_property
    def _aind_session_metadata(self) -> aind_data_schema.core.session.Session:
        return aind_data_schema.core.session.Session(
            experimenter_full_name=self.experimenter or ["NSB trainer"],
            session_start_time=self.session_start_time,
            session_end_time=(
                self.sync_data.stop_time
                if self.is_sync
                else (
                    (
                        self.session_start_time
                        + datetime.timedelta(seconds=max(self.epochs.stop_time))
                    )
                    if self.epochs.stop_time
                    else None
                )
            ),
            session_type=self.session_description.replace(
                " without CCF-annotated units", ""
            ),
            iacuc_protocol="2104",
            rig_id=self._aind_rig_id,
            subject_id=str(self.id.subject),
            data_streams=list(self._aind_data_streams),
            stimulus_epochs=list(self._aind_stimulus_epochs),
            mouse_platform_name="Mouse Platform",
            active_mouse_platform=False,
            reward_delivery=self._aind_reward_delivery if self.is_task else None,
            reward_consumed_total=(
                (np.nanmean(self.sam.rewardSize) * len(self.sam.rewardTimes))
                if self.is_task
                else None
            ),
            reward_consumed_unit="milliliter",
            notes=self.notes,
        )


class DynamicRoutingSurfaceRecording(DynamicRoutingSession):
    @npc_io.cached_property
    def ephys_timing_data(self) -> tuple[npc_ephys.EphysTimingInfo, ...]:
        """Sync data not available, so timing info not accurate"""
        return tuple(
            timing
            for timing in npc_ephys.get_ephys_timing_on_pxi(self.ephys_recording_dirs)
            if (p := npc_session.extract_probe_letter(timing.device.name)) is None
            or p in self.probe_letters_to_use
        )

    @property
    def probe_letters_to_skip(self) -> tuple[npc_session.ProbeRecord, ...]:
        probes_with_tip_channel_bank = {
            npc_session.ProbeRecord(letter)
            for letter, is_tip_channel_bank in zip(
                self.ephys_settings_xml_data.probe_letters,
                self.ephys_settings_xml_data.is_tip_channel_bank,
            )
            if is_tip_channel_bank
        }
        manual_skip = {
            npc_session.ProbeRecord(letter)
            for letter in getattr(self, "_surface_recording_probe_letters_to_skip", "")
        }
        return tuple(
            sorted(
                (set(super().probe_letters_to_skip) | probes_with_tip_channel_bank)
                - manual_skip
            )
        )

    @property
    def session_description(self) -> str:
        return "short ephys recording of spontaneous activity using probe channels at brain surface, to aid probe localization"

    @property
    def session_start_time(self) -> datetime.datetime:
        return utils.get_aware_dt(self.ephys_nominal_start_time)

    @npc_io.cached_property
    def stim_paths(self) -> tuple[upath.UPath, ...]:
        return ()

    class AP(DynamicRoutingSession.AP):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs | {"name": "surface_AP"})


if __name__ == "__main__":
    import doctest

    import dotenv

    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
