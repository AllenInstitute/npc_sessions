from __future__ import annotations

import contextlib
import copy
import datetime
import functools
import importlib.metadata
import io
import itertools
import json
import logging
import re
import typing
import uuid
from collections.abc import Iterable, Iterator
from typing import Any, Literal

import h5py
import hdmf
import ndx_events
import npc_lims
import npc_session
import numpy as np
import numpy.typing as npt
import pandas as pd
import PIL.Image
import pynwb
import upath
import zarr
from DynamicRoutingTask.Analysis.DynamicRoutingAnalysisUtils import DynRoutData

import npc_sessions.plots as plots
import npc_sessions.trials as TaskControl
import npc_sessions.utils as utils

logger = logging.getLogger(__name__)


@typing.overload
def get_sessions() -> Iterator[DynamicRoutingSession]:
    ...


@typing.overload
def get_sessions(session: str | npc_session.SessionRecord) -> DynamicRoutingSession:
    ...


def get_sessions(
    session: str | npc_session.SessionRecord | None = None,
    **all_session_kwargs,
):
    """Uploaded sessions, tracked in npc_lims via `get_session_info()`, newest
    to oldest.

    - if `session` is provided, a single session object is returned
    - sessions with known issues are excluded
    - `all_session_kwargs` will be applied on top of any session-specific kwargs
        - session-specific config from `npc_lims.get_session_kwargs`
          is always applied
        - add extra kwargs here if you want to override or append
          parameters passed to every session __init__
    - returns a generator not because objects take long to create (data is
      loaded lazily) but because attributes are cached, so we want to avoid
      keeping references to sessions that are no longer needed

    ## getting an indexable sequence of sessions
    Just convert the output to a list or tuple, but see the note below if you intend to
    loop over this sequence to process large amounts of data:
    >>> sessions = list(get_sessions())                         # doctest: +SKIP

    ## looping over sessions
    Data is cached in each session object after fetching from the cloud, so
    looping over all sessions can use a lot of memory if all sessions are
    retained.

    ### do this
    loop over the generator so each session object is discarded after use:
    >>> nwbs = []
    >>> for session in get_sessions():                          # doctest: +SKIP
    ...     nwbs.append(session.nwb)

    ### avoid this
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
    if session:
        return DynamicRoutingSession(session, **all_session_kwargs)
    for session_info in sorted(
        npc_lims.get_session_info(), key=lambda x: x.date, reverse=True
    ):
        if session_info.issues:
            continue

        yield DynamicRoutingSession(
            session_info.id,
            **all_session_kwargs,
        )


class DynamicRoutingSession:
    """Class for fetching & processing raw data for a session, making
    NWB modules and an NWBFile instance available as attributes.

    >>> s = DynamicRoutingSession('670248_2023-08-03')

    # paths/raw data processing:
    >>> 'DynamicRouting1' in s.stim_names
    True
    >>> s.stim_paths[0]
    S3Path('s3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15/behavior/DynamicRouting1_670248_20230803_123154.hdf5')
    >>> s.sync_path
    S3Path('s3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15/behavior/20230803T120415.h5')
    >>> s.ephys_timing_data[0].name, s.ephys_timing_data[0].sampling_rate, s.ephys_timing_data[0].start_time
    ('Neuropix-PXI-100.ProbeA-AP', 30000.070518634246, 20.080209634424037)
    >>> s.sam.dprimeSameModal
    [3.5501294698425694]

    # access nwb modules individually before compiling a whole nwb file:
    >>> s.session_start_time
    datetime.datetime(2023, 8, 3, 12, 4, 15, tzinfo=zoneinfo.ZoneInfo(key='America/Los_Angeles'))
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
    experimenter: tuple[str, ...] | list[str] | None = None
    institution: str | None = "Neural Circuits & Behavior | MindScope program | Allen Institute for Neural Dynamics"
    notes: str | None = None

    # --------------------------------------------------------------------------- #

    task_stim_name: str = "DynamicRouting1"

    def __init__(self, session_or_path: str | utils.PathLike, **kwargs) -> None:
        self.id = npc_session.SessionRecord(str(session_or_path))
        if any(
            char in (path := utils.from_pathlike(session_or_path)).as_posix()
            for char in "\\/."
        ):
            if path.is_dir():
                self._root_path: upath.UPath | None = path
            if path.is_file():
                self._root_path = path.parent
        if self.info is not None:
            if issues := self.info.issues:
                logger.warning(f"Session {self.id} has known issues: {issues}")
            kwargs = copy.copy(self.info.session_kwargs) | kwargs
            if self.info.is_sync and not self.info.is_uploaded:
                logger.warning(
                    f"Session {self.id} is marked as `is_sync=True` by `npc_lims`, but raw data has not been uploaded. "
                    "`is_sync` and `is_ephys` will be set to False for this session (disabling all related data)."
                )
                kwargs["is_sync"] = False
        if kwargs:
            logger.info(f"Applying session kwargs to {self.id}: {kwargs}")
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError:
                setattr(self, f"_{key}", value)
        self._add_plots_as_methods()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.id!r})"

    def __eq__(self, other: Any) -> bool:
        if other_id := getattr(other, "id", None):
            return self.id == other_id
        return self.id == str(other)

    def __hash__(self) -> int:
        return hash(self.id)

    def _add_plots_as_methods(self) -> None:
        """Add plots as methods to session object, so they can be called
        directly, e.g. `session.plot_drift_maps()`.

        Looks in `npc_sessions.plots` for functions starting with `plot_`
        """
        for attr in (attr for attr in plots.__dict__ if attr.startswith("plot_")):
            if getattr((fn := getattr(plots, attr)), "__call__", None) is not None:
                setattr(self, attr, functools.partial(fn, self))

    @property
    def nwb(self) -> pynwb.NWBFile:
        # if self._nwb_hdf5_path:
        #     self.nwb = pynwb.NWBHDF5IO(self._nwb_hdf5_path, "r").read()
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
            stimulus_notes=self.task_version,
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
            processing=tuple(self.processing.values()),
            analysis=self._analysis,
            devices=self._devices,
            electrode_groups=self._electrode_groups if self.is_ephys else None,
            electrodes=self.electrodes if self.is_ephys else None,
            units=self.units if self.is_sorted else None,
        )

    # metadata ------------------------------------------------------------------ #

    @utils.cached_property
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
            stimulus_notes=self.task_version,
            subject=self.subject,
            keywords=self.keywords,
            epoch_tags=self.epoch_tags,
        )

    @property
    def session_id(self) -> str:
        return str(self.id)

    @property
    def session_start_time(self) -> datetime.datetime:
        if self.is_sync:
            return utils.get_aware_dt(self.sync_data.start_time)
        return utils.get_aware_dt(utils.get_stim_start_time(self.task_data))

    @property
    def session_description(self) -> str:
        """Uses `is_` bools to construct a text description.
        - won't be correct if testing with datastreams manually disabled
        """
        opto = ", with optogenetic inactivation during task" if self.is_opto else ""
        video = ", with video recording of behavior" if self.is_video else ""
        sync = ", without precise timing information" if not self.is_sync else ""
        if not self.is_ephys:
            description = "training session with behavioral task data"
            description += opto
            description += video
            description += sync
            return description
        else:
            description = "ecephys session"
            description += " without sorted units," if not self.is_sorted else ""
            description += (
                " without CCF-annotated units,"
                if self.is_sorted and not self.is_annotated
                else ""
            )
            description += (
                f" {'with' if self.is_task else 'without'} behavioral task data"
            )
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
        else:
            desc = "visual-auditory task-switching behavior experiment"
        assert (
            "experiment" in desc
        ), "experiment description should contain 'experiment', due to other function which replaces the word"
        return desc

    @property
    def source_script(self) -> str:
        """`githubTaskScript` from the task stim file, if available.
        Otherwise, url to Sam's repo on github"""
        if self.is_task and (script := self.task_data.get("githubTaskScript", None)):
            if isinstance(script[()], bytes):
                return script.asstr()[()]
            if isinstance(script[()], np.floating) and not np.isnan(script[()]):
                return str(script[()])
        return "https://github.com/samgale/DynamicRoutingTask"

    @property
    def source_script_file_name(self) -> str:
        """url to tagged version of packaging code repo on github"""
        return f"https://github.com/AllenInstitute/npc_sessions/releases/tag/v{importlib.metadata.version('npc_sessions')}"

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
                if not self.is_sorted:
                    self.keywords.append("no units")
                elif not self.is_annotated:
                    self.keywords.append("no CCF")
            if self.is_opto:
                self.keywords.append("opto")
            if self.is_templeton:
                self.keywords.append("Templeton")
        return self._keywords

    @keywords.setter
    def keywords(self, value: Iterable[str]) -> None:
        keywords = getattr(self, "_keywords", [])
        keywords += list(value)
        self._keywords = list(set(keywords))

    @utils.cached_property
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

    @utils.cached_property
    def _acquisition(
        self,
    ) -> tuple[pynwb.core.NWBDataInterface | pynwb.core.DynamicTable, ...]:
        """The version passed to NWBFile.__init__"""
        modules: list[pynwb.core.NWBDataInterface | pynwb.core.DynamicTable] = []
        if self.is_sync:
            modules.extend(self._all_licks[1:])
        modules.append(self._rewards)
        if self.is_lfp:
            modules.append(self._raw_lfp)
        if self.is_ephys:
            modules.append(self._raw_ap)
        if self.is_video:
            modules.extend(self._video_frame_times)
        return tuple(modules)

    @property
    def processing(
        self,
    ) -> pynwb.core.LabelledDict[str, pynwb.base.ProcessingModule]:
        """Data after processing and filtering - raw data goes in
        `acquisition`.

        The property as it appears on an NWBFile."""
        # TODO replace with `nwb.add_processing_module`
        processing = pynwb.core.LabelledDict(label="processing", key_attr="name")
        for module_name in ("behavior",) + ("ecephys",) if self.is_ephys else ():
            module = getattr(self, f"_{module_name}")
            processing[module_name] = pynwb.base.ProcessingModule(
                name=module_name,
                description=f"processed {module_name} data",
                data_interfaces=module,
            )
        return processing

    @utils.cached_property
    def _behavior(
        self,
    ) -> tuple[pynwb.core.NWBDataInterface | pynwb.core.DynamicTable, ...]:
        modules: list[pynwb.core.NWBDataInterface | pynwb.core.DynamicTable] = []
        modules.append(self._all_licks[0])
        modules.append(self._running_speed)
        return tuple(modules)

    @utils.cached_property
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

    @utils.cached_property
    def _analysis(
        self,
    ) -> tuple[pynwb.core.NWBDataInterface | pynwb.core.DynamicTable, ...]:
        """The version passed to NWBFile.__init__"""
        # TODO add RF maps
        modules: list[pynwb.core.NWBDataInterface | pynwb.core.DynamicTable] = []
        if self.is_sorted:
            modules.append(self.all_spike_histograms)
            modules.append(self.drift_maps)
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
            TaskControl.OptoTagging: "opto-tagging trials",
        }

    @utils.cached_property
    def invalid_times(self) -> pynwb.epoch.TimeIntervals:
        """Time intervals when recording was interrupted, stim malfunctioned or
        otherwise invalid.

        A separate attribute can mark invalid times for individual ecephys units:
        see `NWBFile.Units.get_unit_obs_intervals()`
        """
        intervals = pynwb.epoch.TimeIntervals(
            name="invalid_times",
            description="time intervals to be removed from analysis",
        )
        intervals.add_column(
            name="reason",
            description="reason for invalidation",
        )
        if (
            self.info
            and (invalid_intervals := getattr(self, "_invalid_intervals", None))
            is not None
        ):
            for interval in invalid_intervals:
                if (
                    stop_time := interval.get("stop_time", None)
                ) is None or stop_time == -1:
                    interval["stop_time"] = self.epochs[:].stop_time.max()
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
            description=self.intervals_descriptions[self._trials.__class__],
        )
        for column in self._trials.to_add_trial_column():
            trials.add_column(**column)
        for trial in self._trials.to_add_trial():
            trials.add_interval(**trial)
        self._cached_nwb_trials = trials
        return trials

    @utils.cached_property
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
            raise KeyError(
                f"no intervals named {self.task_stim_name}* found for {self.id}"
            )

        trials = self._all_trials[stim_name.stem]
        assert isinstance(trials, TaskControl.DynamicRouting1)  # for mypy
        return trials

    @property
    def performance(self) -> pynwb.epoch.TimeIntervals:
        trials = self.trials[:]
        task_performance_by_block = {}

        for block_idx in trials.block_index.unique():
            block_performance: dict[str, float | str] = {}

            block_performance["block_index"] = block_idx
            rewarded_modality = (
                trials[trials["block_index"] == block_idx]["context_name"]
                .unique()
                .item()
            )
            block_performance["rewarded_modality"] = rewarded_modality
            block_performance["cross_modal_dprime"] = self.sam.dprimeOtherModalGo[
                block_idx
            ]
            block_performance["same_modal_dprime"] = self.sam.dprimeSameModal[block_idx]
            block_performance[
                "nonrewarded_modal_dprime"
            ] = self.sam.dprimeNonrewardedModal[block_idx]

            if rewarded_modality == "vis":
                block_performance[
                    "signed_cross_modal_dprime"
                ] = self.sam.dprimeOtherModalGo[block_idx]
                block_performance["vis_intra_dprime"] = self.sam.dprimeSameModal[
                    block_idx
                ]
                block_performance["aud_intra_dprime"] = self.sam.dprimeNonrewardedModal[
                    block_idx
                ]

            elif rewarded_modality in ("aud", "sound"):
                block_performance[
                    "signed_cross_modal_dprime"
                ] = -self.sam.dprimeOtherModalGo[block_idx]
                block_performance["aud_intra_dprime"] = self.sam.dprimeSameModal[
                    block_idx
                ]
                block_performance["vis_intra_dprime"] = self.sam.dprimeNonrewardedModal[
                    block_idx
                ]

            task_performance_by_block[block_idx] = block_performance

        nwb_intervals = pynwb.epoch.TimeIntervals(
            name="performance",
            description=f"behavioral performance for each context block in task (refers to `trials` or `intervals[{self.task_stim_name!r}])",
        )
        column_name_to_description = {
            "block_index": "presentation order in the task (0-indexed)",
            "rewarded_modality": "the modality of the target stimulus that was rewarded in each block: normally `vis` or `aud`",
            "cross_modal_dprime": "dprime across modalities; hits=response rate to rewarded target stimulus, false alarms=response rate to non-rewarded target stimulus",
            "signed_cross_modal_dprime": "same as cross_modal_dprime, but with sign flipped for auditory blocks",
            "same_modal_dprime": "dprime within rewarded modality; hits=response rate to rewarded target stimulus, false alarms=response rate to same modality non-target stimulus",
            "nonrewarded_modal_dprime": "dprime within non-rewarded modality; hits=response rate to non-rewarded target stimulus, false alarms=response rate to same modality non-target stimulus",
            "vis_intra_dprime": "dprime within visual modality; hits=response rate to visual target stimulus, false alarms=response rate to visual non-target stimulus",
            "aud_intra_dprime": "dprime within auditory modality; hits=response rate to auditory target stimulus, false alarms=response rate to auditory non-target stimulus",
        }
        for name, description in column_name_to_description.items():
            nwb_intervals.add_column(name=name, description=description)
        for block_index in task_performance_by_block:
            nwb_intervals.add_interval(
                start_time=trials[trials["block_index"] == block_index][
                    "start_time"
                ].min(),
                stop_time=trials[trials["block_index"] == block_index][
                    "stop_time"
                ].max(),
                **dict.fromkeys(column_name_to_description, np.nan)
                | task_performance_by_block[block_index],
            )

        return nwb_intervals

    @utils.cached_property
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

    @utils.cached_property
    def _intervals(self) -> tuple[pynwb.epoch.TimeIntervals, ...]:
        """The version passed to NWBFile.__init__"""
        intervals: list[pynwb.epoch.TimeIntervals] = []
        for k, v in self._all_trials.items():
            if self.task_stim_name in k and self.is_task:
                # intervals.append(self.trials)
                intervals.append(self.performance)
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

    @utils.cached_property
    def _all_trials(self) -> utils.LazyDict[str, TaskControl.TaskControl]:
        if self.is_sync:
            # get the only stims for which we have times:
            stim_paths = tuple(
                path
                for path in self.stim_paths
                if path.stem in self.stim_frame_times.keys()
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
        return utils.LazyDict((k, v) for k, v in lazy_dict_items.items())

    @utils.cached_property
    def epochs(self) -> pynwb.file.TimeIntervals:
        epochs = pynwb.file.TimeIntervals(
            name="epochs",
            description="time intervals corresponding to different phases of the session",
        )
        epochs.add_column(
            "notes",
            description="notes about the experiment or the data collected during the epoch",
        )
        records = []
        for stim in self.stim_paths:
            if self.is_sync and stim.stem not in self.stim_frame_times.keys():
                continue
            records.append(self.get_epoch_record(stim).nwb)

        for record in sorted(records, key=lambda _: str(_["start_time"]), reverse=True):
            # reverse=True puts intervals in chronological order (doesn't make
            # sense, but it's true)
            epochs.add_interval(
                **record,
            )
        return epochs

    # probes, devices, units ---------------------------------------------------- #

    @utils.cached_property
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
            if probe_letter in self.probe_letters_inserted
        )

    @property
    def _devices(self) -> tuple[pynwb.device.Device, ...]:
        """The version passed to NWBFile.__init__"""
        devices: list[pynwb.device.Device] = []
        if self.is_ephys:
            devices.extend(self._probes)
        return tuple(devices)  # add other devices as we need them

    @utils.cached_property
    def devices(self) -> pynwb.core.LabelledDict[str, pynwb.device.Device]:
        """Currently just probe model + serial number.

        Could include other devices: laser, monitor, etc.

        The property as it appears on an NWBFile"""
        devices = pynwb.core.LabelledDict(label="devices", key_attr="name")
        for module in self._devices:
            devices[module.name] = module
        return devices

    @utils.cached_property
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

    @utils.cached_property
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
            if probe_letter in self.probe_letters_inserted
        )

    @utils.cached_property
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
            column_names = ("structure", "x", "y", "z") + column_names
            ccf_df = utils.get_tissuecyte_electrodes_table(self.id)
        column_description = {
            "structure": "acronym for the Allen CCF structure that the electrode recorded from - less-specific than `location`",
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
            electrodes.add_column(name=column, description=column_description[column])

        for probe_letter, channel_pos_xy in zip(
            self.ephys_settings_xml_data.probe_letters,
            self.ephys_settings_xml_data.channel_pos_xy,
        ):
            if probe_letter not in self.probe_letters_inserted:
                continue
            group = self.electrode_groups[f"probe{probe_letter}"]
            for channel_label, (x, y) in channel_pos_xy.items():
                channel_idx = int(channel_label.strip("CH"))
                kwargs: dict[str, str | float] = {
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
                    kwargs |= (
                        annotated_probes.query(f"channel == {channel_idx}")
                        .iloc[0]
                        .to_dict()
                    )
                electrodes.add_row(
                    **kwargs,
                )

        return electrodes

    @utils.cached_property
    def _units(self) -> pd.DataFrame:
        if not self.is_sorted:
            raise AttributeError(f"{self.id} hasn't been spike-sorted")
        units = utils.add_global_unit_ids(
            units=utils.make_units_table_from_spike_interface_ks25(
                self.sorted_data,  # TODO keep spikeinterface obj in self
                self.ephys_timing_data,
                include_waveform_arrays=True,
            ),
            session=self.id,
        )
        # remove units from probes that weren't inserted
        units = units[units["electrode_group_name"].isin(self.probes_inserted)]
        if self.is_annotated:
            units = utils.add_tissuecyte_annotations(
                units=units,
                session=self.id,
            )
        return utils.good_units(units)

    @utils.cached_property
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
            units.add_column(name=column, description="")
        for _, row in self._units.iterrows():
            ## for ref:
            # add_unit(spike_times=None, obs_intervals=None, electrodes=None, electrode_group=None, waveform_mean=None, waveform_sd=None, waveforms=None, id=None)
            # TODO add obs_intervals - use invalid_times, presence?
            units.add_unit(
                **row,  # contains spike_times
                electrodes=[
                    self.electrodes[:]
                    .query(f"channel == {row['peak_channel']}")
                    .query(f"group_name == {row['electrode_group_name']!r}")
                    .index.item()
                ],
                electrode_group=self.electrode_groups[row["electrode_group_name"]],
            )
        return units

    @utils.cached_property
    def _raw_ap(self) -> pynwb.core.MultiContainerInterface:
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

        ap = AP()
        #! this will likely not write to disk as the class is not registered with 'CORE_NAMESPACE'
        # there's currently no appropriate ephys MultiContainerInterface
        # but `pynwb.ecephys.FilteredEphys()` would work if otherwise unused
        band: str = "0.3-10 kHz"
        for probe in self.electrode_groups.values():
            timing_info = next(
                d
                for d in self.ephys_timing_data
                if d.name.endswith("AP") and probe.name.lower() in d.name.lower()
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
                starting_time=timing_info.start_time,
                rate=timing_info.sampling_rate,
                channel_conversion=None,
                filtering=band,
                conversion=0.195e-6,  # bit/microVolt from open-ephys
                comments="",
                resolution=0.195e-6,
                description=f"action potential-band voltage timeseries ({band}) from electrodes on {probe.name}",
                # units=microvolts, # doesn't work - electrical series must be in volts
            )
        return ap

    @utils.cached_property
    def _raw_lfp(self) -> pynwb.ecephys.LFP:
        lfp = pynwb.ecephys.LFP()
        band: str = "0.5-500 Hz"

        for probe in self.electrode_groups.values():
            timing_info = next(
                d
                for d in self.ephys_timing_data
                if d.name.endswith("LFP") and probe.name.lower() in d.name.lower()
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
                starting_time=timing_info.start_time,
                rate=timing_info.sampling_rate,
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

    @utils.cached_property
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

    @utils.cached_property
    def info(self) -> npc_lims.SessionInfo | None:
        with contextlib.suppress(ValueError):
            return npc_lims.get_session_info(self.id)
        return None

    @utils.cached_property
    def is_task(self) -> bool:
        if (v := getattr(self, "_is_task", None)) is not None:
            return v
        with contextlib.suppress(
            FileNotFoundError, ValueError, StopIteration, npc_lims.MissingCredentials
        ):
            _ = self.task_data
            return True
        return False

    @utils.cached_property
    def is_sync(self) -> bool:
        if (v := getattr(self, "_is_sync", None)) is not None:
            return v
        if (v := getattr(self, "_sync_path", None)) is not None:
            return True
        if self.info:
            return self.info.is_sync
        with contextlib.suppress(
            FileNotFoundError, ValueError, npc_lims.MissingCredentials
        ):
            if self.get_sync_paths():
                return True
        return False

    @utils.cached_property
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

    @utils.cached_property
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

    @utils.cached_property
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
            _ = npc_lims.is_sorted_data_asset(self.id)
            return True
        return False

    @utils.cached_property
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

    @utils.cached_property
    def is_lfp(self) -> bool:
        if (v := getattr(self, "_is_lfp", None)) is not None:
            return v
        return self.is_ephys

    @utils.cached_property
    def is_opto(self) -> bool:
        """Opto during behavior task && not wt/wt (if genotype info available)"""
        genotype: str | None = (
            self.subject.genotype
        )  # won't exist if subject.json not found
        if not self.is_task:
            return False
        if utils.is_opto(self.task_data) and (
            genotype is None or "wt/wt" not in genotype.lower()
        ):
            if genotype is None:
                logger.warning(
                    f"Could not find genotype for {self.id}: returning is_opto = True regardless"
                )
            return True
        return False

    @property
    def is_templeton(self) -> bool:
        if (v := getattr(self, "_is_templeton", None)) is not None:
            return v
        if self.task_version is not None:
            return bool("templeton" in self.task_version)
        if self.info is not None:
            return bool(
                "templeton" in self.info.project.lower()
                or "templeton" in self.info.allen_path.as_posix().lower()
            )
        raise NotImplementedError(
            "Not enough information to tell if this is a Templeton session"
        )

    # helper properties -------------------------------------------------------- #

    @utils.cached_property
    def _raw_upload_metadata_json_paths(self) -> tuple[upath.UPath, ...]:
        return tuple(
            file
            for file in npc_lims.get_raw_data_root(self.id).iterdir()
            if file.suffix == ".json"
        )

    @utils.cached_property
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

    @utils.cached_property
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

    @utils.cached_property
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
        return pynwb.file.Subject(
            subject_id=metadata["subject_id"],
            species="Mus musculus",
            sex=metadata["sex"][0].upper(),
            date_of_birth=dob,
            genotype=metadata["genotype"],
            description=None,
            strain=metadata["background_strain"] or metadata["breeding_group"],
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
                file.path
                for file in npc_lims.get_hdf5_stim_files_from_s3(self.id)
                if self.task_stim_name in file.path.stem
            )
        except StopIteration:
            raise FileNotFoundError(
                f"Could not find file in {npc_lims.DR_DATA_REPO} for {self.id}"
            ) from None

    @utils.cached_property
    def raw_data_paths(self) -> tuple[upath.UPath, ...]:
        if self.root_path:
            return self.get_raw_data_paths_from_root()
        with contextlib.suppress(FileNotFoundError, ValueError):
            return npc_lims.get_raw_data_paths_from_s3(self.id)
        if (
            getattr(self, "_is_task", None) is not False
        ):  # using regular version will cause infinite recursion
            with contextlib.suppress(StopIteration):
                if stim_files := npc_lims.get_hdf5_stim_files_from_s3(self.id):
                    self._root_path = stim_files[0].path.parent
                    logger.warning(
                        f"Using {self._root_path} as root path for {self.id}"
                    )
                    return self.get_raw_data_paths_from_root()
        raise ValueError(
            f"{self.id} is either an ephys session with no Code Ocean upload, or a behavior session with no data in the synced s3 repo {npc_lims.DR_DATA_REPO}"
        )

    @utils.cached_property
    def sorted_data(self) -> utils.SpikeInterfaceKS25Data:
        return utils.SpikeInterfaceKS25Data(
            self.id,
            self.sorted_data_paths[0].parent,
        )

    @utils.cached_property
    def sorted_data_paths(self) -> tuple[upath.UPath, ...]:
        if not self.is_ephys:
            raise ValueError(f"{self.id} is not a session with ephys")
        return npc_lims.get_sorted_data_paths_from_s3(self.id)

    @utils.cached_property
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

    @utils.cached_property
    def raw_data_asset_id(self) -> str:
        if not self.is_ephys:
            raise ValueError(
                f"{self.id} currently only ephys sessions have raw data assets"
            )
        return npc_lims.get_session_raw_data_asset(self.id)["id"]

    @utils.cached_property
    def sync_data(self) -> utils.SyncDataset:
        return utils.SyncDataset(io.BytesIO(self.sync_path.read_bytes()))

    @property
    def stim_path_root(self) -> upath.UPath:
        return npc_lims.DR_DATA_REPO / str(self.id.subject)

    @utils.cached_property
    def stim_paths(self) -> tuple[upath.UPath, ...]:
        def is_valid_stim_file(p) -> bool:
            if not utils.is_stim_file(
                p, subject_spec=self.id.subject, date_spec=self.id.date
            ):
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
            if dt > self.sync_data.stop_time:
                return False
            return True

        if self.is_ephys:
            if stim_paths := tuple(
                p for p in self.raw_data_paths if is_valid_stim_file(p)
            ):
                return stim_paths
        if self.root_path:
            if stim_paths := tuple(
                p for p in self.root_path.iterdir() if is_valid_stim_file(p)
            ):
                return stim_paths
        if stim_paths := tuple(
            p for p in self.stim_path_root.iterdir() if is_valid_stim_file(p)
        ):
            return stim_paths
        raise FileNotFoundError(
            f"Could not find stim files for {self.id} in {self.stim_path_root}"
        )

    @utils.cached_property
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
        if getattr(self, "_sam", None) is None:
            self._sam = utils.get_sam(self.task_data)
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

    @utils.cached_property
    def stim_data(self) -> utils.LazyDict[str, h5py.File]:
        def h5_dataset(path: upath.UPath) -> h5py.File:
            return h5py.File(io.BytesIO(path.read_bytes()), "r")

        return utils.LazyDict(
            (path.stem, (h5_dataset, (path,), {})) for path in self.stim_paths
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

    @utils.cached_property
    def video_info_data(self) -> utils.LazyDict[str, utils.MVRInfoData]:
        return utils.LazyDict(
            (
                utils.extract_camera_name(path.stem),
                (utils.get_video_info_data, (path,), {}),
            )
            for path in self.video_info_paths
        )

    @utils.cached_property
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
        if utils.is_opto(h5):
            tags.append("opto")
        if (rewards := h5.get("rewardFrames", None)) is not None and any(rewards[:]):
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

        notes: list[str] = []
        for _, invalid_interval in self.invalid_times[:].iterrows():
            if any(
                start_time <= invalid_interval[time] <= stop_time
                for time in ("start_time", "stop_time")
            ):
                notes.append(invalid_interval["reason"])
                if "invalid_times" not in tags:
                    tags.append("invalid_times")
        return npc_lims.Epoch(
            session_id=self.id,
            start_time=start_time,
            stop_time=stop_time,
            tags=tags,
            notes=None if not notes else f"includes invalid times: {'; '.join(notes)}",
        )

    @property
    def ephys_record_node_dirs(self) -> tuple[upath.UPath, ...]:
        if getattr(self, "_ephys_record_node_dirs", None) is None:
            self.ephys_record_node_dirs = tuple(
                p
                for p in self.raw_data_paths
                if re.match(r"^Record Node [0-9]+$", p.name)
            )
        return self._ephys_record_node_dirs

    @ephys_record_node_dirs.setter
    def ephys_record_node_dirs(self, v: Iterable[utils.PathLike]) -> None:
        if isinstance(v, str) or not isinstance(v, Iterable):
            v = (v,)
        paths = tuple(utils.from_pathlike(path) for path in v)
        assert all("Record Node" in path.name for path in paths)
        self._ephys_record_node_dirs = paths

    @utils.cached_property
    def ephys_recording_dirs(self) -> tuple[upath.UPath, ...]:
        return tuple(
            p
            for record_node in self.ephys_record_node_dirs
            for p in record_node.glob("experiment*/recording*")
        )

    @utils.cached_property
    def ephys_timing_data(self) -> tuple[utils.EphysTimingInfoOnSync, ...]:
        return tuple(
            timing
            for timing in utils.get_ephys_timing_on_sync(
                self.sync_data, self.ephys_recording_dirs
            )
            if (p := npc_session.extract_probe_letter(timing.device.name)) is None
            or p in self.probe_letters_inserted
        )

    @utils.cached_property
    def drift_map_paths(self) -> tuple[upath.UPath, ...]:
        return tuple(
            next(d for d in self.sorted_data_paths if d.name == "drift_maps").iterdir()
        )

    @utils.cached_property
    def ephys_sync_messages_path(self) -> upath.UPath:
        return next(
            p
            for p in itertools.chain(
                *(record_node.iterdir() for record_node in self.ephys_recording_dirs)
            )
            if "sync_messages.txt" == p.name
        )

    @utils.cached_property
    def ephys_structure_oebin_paths(self) -> tuple[upath.UPath, ...]:
        return tuple(
            p
            for p in itertools.chain(
                *(record_node.iterdir() for record_node in self.ephys_recording_dirs)
            )
            if "structure.oebin" == p.name
        )

    @utils.cached_property
    def ephys_structure_oebin_data(
        self,
    ) -> dict[Literal["continuous", "events", "spikes"], list[dict[str, Any]]]:
        return utils.get_merged_oebin_file(self.ephys_structure_oebin_paths)

    @utils.cached_property
    def ephys_sync_messages_data(
        self,
    ) -> dict[str, dict[Literal["start", "rate"], int]]:
        return utils.get_sync_messages_data(self.ephys_sync_messages_path)

    @utils.cached_property
    def ephys_experiment_dirs(self) -> tuple[upath.UPath, ...]:
        return tuple(
            p
            for record_node in self.ephys_record_node_dirs
            for p in record_node.glob("experiment*")
        )

    @utils.cached_property
    def ephys_settings_xml_paths(self) -> tuple[upath.UPath, ...]:
        if not self.is_ephys:
            raise ValueError(
                f"{self.id} is not an ephys session (required for settings.xml)"
            )
        return tuple(
            next(record_node.glob("settings*.xml"))
            for record_node in self.ephys_record_node_dirs
        )

    @utils.cached_property
    def ephys_settings_xml_path(self) -> upath.UPath:
        """Single settings.xml path, if applicable"""
        if not self.ephys_settings_xml_paths:
            raise ValueError(
                f"settings.xml not found for {self.id} - check status of raw upload"
            )
        utils.assert_xml_files_match(*self.ephys_settings_xml_paths)
        return self.ephys_settings_xml_paths[0]

    @utils.cached_property
    def ephys_settings_xml_data(self) -> utils.SettingsXmlInfo:
        return utils.get_settings_xml_data(self.ephys_settings_xml_path)

    @property
    def electrode_group_description(self) -> str:
        # TODO get correct channels range from settings xml
        return "Neuropixels 1.0 lower channels (1:384)"

    @property
    def probe_letters_to_skip(self) -> tuple[npc_session.ProbeRecord, ...]:
        if (v := getattr(self, "_probe_letters_to_skip", None)) is not None:
            return tuple(npc_session.ProbeRecord(letter) for letter in v)
        return ()

    def remove_probe_letters_to_skip(
        self, letters: Iterable[str | npc_session.ProbeRecord]
    ) -> tuple[npc_session.ProbeRecord, ...]:
        return tuple(
            npc_session.ProbeRecord(letter)
            for letter in letters
            if npc_session.ProbeRecord(letter) not in self.probe_letters_to_skip
        )

    @utils.cached_property
    def probe_insertions(self) -> dict[str, Any] | None:
        path = next(
            (path for path in self.raw_data_paths if "probe_insertions" in path.stem),
            None,
        )
        if not path:
            return None
        return json.loads(path.read_text())["probe_insertions"]

    @utils.cached_property
    def probes_inserted(self) -> tuple[str, ...]:
        """('probeA', 'probeB', ...)"""
        return tuple(probe.name for probe in self.probe_letters_inserted)

    @utils.cached_property
    def probe_letters_inserted(self) -> tuple[npc_session.ProbeRecord, ...]:
        """('A', 'B', ...)"""
        from_annotation = from_insertion_record = None
        if self.is_annotated:
            from_annotation = tuple(
                npc_session.ProbeRecord(probe)
                for probe in utils.get_tissuecyte_electrodes_table(
                    self.id
                ).group_name.unique()
            )
        if self.probe_insertions is not None:
            from_insertion_record = tuple(
                npc_session.ProbeRecord(k)
                for k, v in self.probe_insertions.items()
                if npc_session.extract_probe_letter(k) is not None
                and v["hole"] is not None
                and k not in self.probe_letters_to_skip
            )
            if from_annotation and set(
                self.remove_probe_letters_to_skip(from_annotation)
            ).symmetric_difference(
                set(self.remove_probe_letters_to_skip(from_insertion_record))
            ):
                logger.warning(
                    f"probe_insertions.json and annotation info do not match for {self.id} - using annotation info"
                )
        if from_annotation:
            return self.remove_probe_letters_to_skip(from_annotation)
        if from_insertion_record:
            return self.remove_probe_letters_to_skip(from_insertion_record)
        logger.warning(
            f"No probe_insertions.json or annotation info found for {self.id} - defaulting to ABCDEF"
        )
        return tuple(npc_session.ProbeRecord(letter) for letter in "ABCDEF")

    @property
    def implant(self) -> str:
        if self.probe_insertions is None:
            # TODO get from sharepoint
            return "unknown implant"
        implant: str = self.probe_insertions["implant"]
        return "2002" if "2002" in implant else implant

    @utils.cached_property
    def _all_licks(self) -> tuple[ndx_events.Events, ...]:
        """First item is always `processing['licks']` - the following items are only if sync
        is available, and are raw rising/falling edges of the lick sensor,
        for `acquisition`.

        If sync isn't available, we only have start frames of licks, so we can't
        filter by duration very accurately.
        """
        if self.is_sync:
            max_contact = (
                0.5  # must factor-in lick_sensor staying high after end of contact
            )
            # https://www.nature.com/articles/s41586-021-03561-9/figures/1

            rising = self.sync_data.get_rising_edges("lick_sensor", units="seconds")
            falling = self.sync_data.get_falling_edges("lick_sensor", units="seconds")
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

            filtered = rising[filtered_idx]

        licks = ndx_events.Events(
            timestamps=self.sam.lickTimes if not self.is_sync else filtered,
            name="licks",
            description="times at which the subject made contact with a water spout"
            + (
                f" - filtered to exclude events with duration >{max_contact} s"
                if self.is_sync
                else " - putatively the starts of licks, but may include other events such as grooming"
            ),
        )

        if not self.is_sync:
            return (licks,)
        return (
            licks,
            ndx_events.Events(
                timestamps=rising,
                name="lick_sensor_rising",
                description=(
                    "times at which the subject made contact with a water spout - "
                    "putatively the starts of licks, but may include other events such as grooming"
                ),
            ),
            ndx_events.Events(
                timestamps=falling,
                name="lick_sensor_falling",
                description=(
                    "times at which the subject ceased making contact with a water spout - "
                    "putatively the ends of licks, but may include other events such as grooming"
                ),
            ),
        )

    @utils.cached_property
    def _running_speed(self) -> pynwb.TimeSeries:
        name = "running_speed"
        description = (
            "linear forward running speed on a rotating disk, low-pass filtered "
            f"at {utils.RUNNING_LOWPASS_FILTER_HZ} Hz with a 3rd order Butterworth filter"
        )
        unit = utils.RUNNING_SPEED_UNITS
        # comments = f'Assumes mouse runs at `radius = {utils.RUNNING_DISK_RADIUS} {utils.RUNNING_SPEED_UNITS.split("/")[0]}` on disk.'
        data, timestamps = utils.get_running_speed_from_stim_files(
            *self.stim_data.values(),
            sync=self.sync_data if self.is_sync else None,
            filt=utils.lowpass_filter,
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
        hist, bin_edges = utils.bin_spike_times(
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

    @utils.cached_property
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
        for probe in self.probes_inserted:
            module.add_timeseries(self.get_all_spike_histogram(probe))
        return module

    @utils.cached_property
    def _video_frame_times(
        self,
    ) -> tuple[pynwb.core.NWBDataInterface | pynwb.core.DynamicTable, ...]:
        # currently doesn't require opening videos
        path_to_timestamps = utils.get_video_frame_times(
            self.sync_data, *self.video_paths
        )
        nwb_names = {
            "eye": "eye",
            "face": "front",
            "behavior": "side",
        }
        return tuple(
            ndx_events.Events(
                timestamps=timestamps,
                name=f"{nwb_names[utils.extract_camera_name(path.stem)]}_camera",
                description=f"start time of each frame exposure for {path.stem}",
            )
            for path, timestamps in path_to_timestamps.items()
        )

    @utils.cached_property
    def _rewards(self) -> pynwb.core.NWBDataInterface | pynwb.core.DynamicTable:
        def get_reward_frames(data: h5py.File) -> list[int]:
            r = []
            for key in ("rewardFrames", "manualRewardFrames"):
                if (v := data.get(key, None)) is not None:
                    r.extend(v[:])
            return r

        reward_times: list[npt.NDArray[np.floating]] = []
        for stim_file, stim_data in self.stim_data.items():
            if any(name in stim_file.lower() for name in ("mapping", "tagging")):
                continue
            reward_times.extend(
                utils.safe_index(
                    utils.get_flip_times(
                        stim_data, sync=self.sync_data if self.is_sync else None
                    ),
                    get_reward_frames(stim_data),
                )
            )
        return ndx_events.Events(
            timestamps=np.sort(np.unique(reward_times)),
            name="rewards",
            description="individual water rewards delivered to the subject",
        )


if __name__ == "__main__":
    import doctest

    import dotenv

    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
