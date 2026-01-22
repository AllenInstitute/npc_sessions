import contextlib
import datetime
import itertools
import json
import logging
import re
from collections.abc import Iterable
from typing import TypeGuard

import aind_data_schema.components.configs
import aind_data_schema.components.coordinates
import aind_data_schema.components.identifiers
import aind_data_schema.core.acquisition
import aind_data_schema_models.brain_atlas
import aind_data_schema_models.modalities
import aind_data_schema_models.stimulus_modality
import aind_data_schema_models.units
import numpy as np
import upath

from npc_sessions.aind_data_schema import instrument
from npc_sessions.sessions import DynamicRoutingSession, DynamicRoutingSurfaceRecording

logger = logging.getLogger(__name__)


def is_surface_recording(
    session: DynamicRoutingSession,
) -> TypeGuard[DynamicRoutingSurfaceRecording]:
    return isinstance(session, DynamicRoutingSurfaceRecording)


def _merge_data_streams(
    data_streams: Iterable[aind_data_schema.core.acquisition.DataStream],
) -> aind_data_schema.core.acquisition.DataStream:
    data_streams = tuple(data_streams)
    return aind_data_schema.core.acquisition.DataStream(
        stream_start_time=min(ds.stream_start_time for ds in data_streams),
        stream_end_time=max(ds.stream_end_time for ds in data_streams),
        modalities=sorted(
            {mod for ds in data_streams for mod in ds.modalities}, key=lambda m: m.name
        ),
        code=sorted(
            itertools.chain.from_iterable(ds.code for ds in data_streams if ds.code),
            key=lambda c: c.url,
        )
        or None,
        active_devices=sorted(
            itertools.chain.from_iterable(ds.active_devices for ds in data_streams),
        ),
        configurations=list(
            itertools.chain.from_iterable(ds.configurations for ds in data_streams)
        ),
    )


def get_acquisition_model(
    session: DynamicRoutingSession,
) -> aind_data_schema.core.acquisition.Acquisition:
    """Get the Pydantic model corresponding to the 'acquisition.json' for a given session."""
    subject_id = str(session.id.subject)
    acquisition_start_time = session.session_start_time
    if session.is_sync:
        acquisition_end_time = session.sync_data.stop_time
    elif session.epochs.stop_time:
        acquisition_end_time = session.session_start_time + datetime.timedelta(
            seconds=max(session.epochs.stop_time)
        )
    elif session.ephys_timing_data:  # eg surface channel recording
        acquisition_end_time = session.session_start_time + datetime.timedelta(
            seconds=session.ephys_timing_data[0].stop_time
        )
    else:
        raise ValueError(f"Unable to determine acquisition end time for {session}")
    if isinstance(session, DynamicRoutingSurfaceRecording):
        experimenters = session.main_recording.experimenter or ["unknown"]
    else:
        experimenters = session.experimenter or ["NSB trainer"]
    if is_surface_recording(session):
        instrument_id = session.main_recording.rig
        acquisition_type = "ecephys surface recording"
    else:
        instrument_id = session.rig
        keywords: list[str] = []
        if session.is_ephys:
            keywords.append("ecephys")
        if session.is_task:
            keywords.append("behavior")
        for k in [
            "naive",
            "injection_perturbation",
            "injection_control",
            "opto_perturbation",
            "opto_control",
        ]:
            if k in session.keywords:
                keywords.append(k)
        acquisition_type = " ".join(k.replace("_", " ") for k in keywords)
    coordinate_system = None
    data_streams = _get_data_streams(session)
    stimulus_epochs = _get_stimulus_epochs(session)
    if is_surface_recording(session):
        subject_details = None
    else:
        subject_details = aind_data_schema.core.acquisition.AcquisitionSubjectDetails(
            mouse_platform_name="Brain Observatory running disc",
            reward_consumed_total=(
                (np.nanmean(session.sam.rewardSize) * len(session.sam.rewardTimes))
                if session.is_task
                else None
            ),
            reward_consumed_unit=aind_data_schema_models.units.VolumeUnit.ML,
        )
    return aind_data_schema.core.acquisition.Acquisition(
        subject_id=subject_id,
        acquisition_start_time=acquisition_start_time,
        acquisition_end_time=acquisition_end_time,
        experimenters=experimenters,
        protocol_id=None,
        ethics_review_id=["2104"],
        instrument_id=instrument_id,
        acquisition_type=acquisition_type,
        notes=session.notes,
        coordinate_system=coordinate_system,
        data_streams=[_merge_data_streams(data_streams)],
        stimulus_epochs=stimulus_epochs,
        subject_details=subject_details,
    )


def get_active_devices(script_name: str, session: DynamicRoutingSession) -> list[str]:
    stim = aind_data_schema_models.stimulus_modality.StimulusModality
    modalities = get_modalities(script_name, session)
    device_names = [instrument.TASKCONTROL_DAQ.name, instrument.CAMSTIM_DAQ.name]
    if stim.VISUAL in modalities:
        device_names.append(instrument.MONITOR.name)
        if session.is_sync:
            device_names.extend([instrument.PHOTODIODE.name])
    if stim.AUDITORY in modalities:
        device_names.append(instrument.SPEAKER.name)
        if session.is_sync:
            device_names.extend([instrument.MICROPHONE.name])
    if stim.OPTOGENETICS in modalities:
        device_names.extend(
            [
                instrument.LASER_488.name,
                instrument.LASER_633.name,
                instrument.LASER_GALVO_X.name,
                instrument.LASER_GALVO_Y.name,
                instrument.OPTO_DAQ.name,
            ]
        )
    if session.is_sync:
        device_names.append(instrument.SYNC_DAQ.name)

    return sorted(set(device_names))


def get_modalities(
    script_name: str,
    session: DynamicRoutingSession,
) -> list[aind_data_schema_models.stimulus_modality.StimulusModality]:
    stim = aind_data_schema_models.stimulus_modality.StimulusModality
    modalities = []
    if any(
        name in script_name
        for name in (
            "DynamicRouting",
            "RFMapping",
            "LuminanceTest",
            "Spontaneous",
        )
    ):
        modalities.append(stim.VISUAL)
    if any(name in script_name for name in ("DynamicRouting", "RFMapping")):
        modalities.append(stim.AUDITORY)
    if session.is_opto and any(name in script_name for name in ("DynamicRouting",)):
        modalities.append(stim.OPTOGENETICS)
    if any(name in script_name for name in ("OptoTagging",)):
        modalities.append(stim.OPTOGENETICS)
    return modalities or [stim.NO_STIMULUS]


def _get_stimulus_epochs(
    session: DynamicRoutingSession,
) -> list[aind_data_schema.core.acquisition.StimulusEpoch]:
    if is_surface_recording(session):
        return []

    def get_speaker_config(
        script_name: str,
    ) -> aind_data_schema.components.configs.SpeakerConfig | None:
        stim = aind_data_schema_models.stimulus_modality.StimulusModality
        if stim.AUDITORY not in get_modalities(script_name, session):
            return None
        return aind_data_schema.components.configs.SpeakerConfig(
            device_name="Speaker",
            volume=68.0,
            volume_unit="decibels",
        )

    def get_performance_metrics(
        script_name: str,
    ) -> aind_data_schema.core.acquisition.PerformanceMetrics | None:
        if "DynamicRouting" not in script_name or not session.is_task:
            return None
        block_metrics = {}
        for block_index, (
            hit_count,
            dprime_same_modal,
            dprime_other_modal_go,
            block_stim_rewarded,
        ) in enumerate(
            zip(
                [int(v) for v in session.sam.hitCount],
                [float(v) for v in session.sam.dprimeSameModal],
                [float(v) for v in session.sam.dprimeOtherModalGo],
                [str(v) for v in session.sam.blockStimRewarded],
                strict=False,
            )
        ):
            block_metrics[str(block_index)] = dict(
                block_index=block_index,
                block_stim_rewarded=block_stim_rewarded,
                hit_count=hit_count,
                dprime_same_modal=dprime_same_modal,
                dprime_other_modal_go=dprime_other_modal_go,
            )
        return aind_data_schema.core.acquisition.PerformanceMetrics(
            output_parameters={
                "block_metrics": block_metrics,
                "task_version": session.sam.taskVersion,
            },
            reward_consumed_during_epoch=np.nanmean(session.sam.rewardSize)
            * sum(session.trials[:].is_rewarded),
            reward_consumed_unit=aind_data_schema_models.units.VolumeUnit.ML,
            trials_total=len(session.trials[:]),
            trials_rewarded=sum(session.trials[:].is_contingent_reward),
        )

    def get_laser_configs(
        script_name: str,
    ) -> list[aind_data_schema.components.configs.LaserConfig] | None:
        configs = []
        for laser in [instrument.LASER_488, instrument.LASER_633]:
            if laser.name not in get_active_devices(script_name, session):
                continue
            if script_name == "OptoTagging":
                column_name = "power"
            elif script_name == "DynamicRouting1":
                column_name = "opto_power"
            else:
                raise NotImplementedError(
                    f"Unknown script name {script_name}: unsure how to get laser power from intervals table"
                )
            trials = next(v for k, v in session._all_trials.items() if script_name in k)
            max_power = np.nanmax(getattr(trials, column_name))
            if np.isnan(max_power):  # control session
                max_power = 0.0
            configs.append(
                aind_data_schema.components.configs.LaserConfig(
                    device_name=laser.name,
                    wavelength=laser.wavelength,
                    wavelength_unit=laser.wavelength_unit,
                    power=max_power,
                    power_unit=aind_data_schema_models.units.PowerUnit.MW,
                    power_measured_at="Brain surface",
                )
            )
        return configs or None

    def get_version() -> str | None:
        if "blob/main" in session.source_script:
            return None
        return (
            session.source_script.split("DynamicRoutingTask/")[-1]
            .split("/DynamicRouting1.py")[0]
            .strip("/")
        )

    def get_url(epoch_name: str) -> str:
        return session.source_script.replace(
            "DynamicRouting1", get_taskcontrol_file(epoch_name)
        )

    def get_taskcontrol_file(epoch_name: str) -> str:
        if upath.UPath(
            session.source_script.replace("DynamicRouting1", epoch_name)
        ).exists():
            return epoch_name
        return "TaskControl"

    def get_code(script_name: str) -> aind_data_schema.core.acquisition.Code:
        return aind_data_schema.core.acquisition.Code(
            url=get_url(script_name),
            version=get_version(),
            container=None,
            language="Python",
            language_version="3.9",
            core_dependency=aind_data_schema.components.identifiers.Software(
                name="PsychoPy",
                version="2022.1.2",
            ),
        )

    def get_configurations(
        script_name: str,
    ) -> list[aind_data_schema.components.configs.DeviceConfig]:
        configurations = []
        speaker_config = get_speaker_config(script_name)
        if speaker_config:
            configurations.append(speaker_config)
        laser_configs = get_laser_configs(script_name)
        if laser_configs:
            configurations.extend(laser_configs)
        return configurations

    def get_training_protocol_name(
        script_name: str,
        session: DynamicRoutingSession,
    ) -> str | None:
        if "DynamicRouting" not in script_name or not session.is_task:
            return None
        protocol = "dynamic_routing"
        if session.is_context_naive:
            protocol += "_context_naive"
        if session.is_naive:
            protocol += "_naive"
        return protocol

    def get_curriculum_status(
        script_name: str,
        session: DynamicRoutingSession,
    ) -> str | None:
        if "DynamicRouting" not in script_name or not session.is_task:
            return None
        # extract 'stage X'
        stages = re.findall(
            r"stage\s?(\d+)",
            session.sam.taskVersion.lower(),
        )
        if not stages:
            return None
        return f"stage {stages[0]}"

    aind_epochs = []
    for nwb_epoch in session.epochs:
        script_name = nwb_epoch.script_name.item()
        aind_epochs.append(
            aind_data_schema.core.acquisition.StimulusEpoch(
                stimulus_start_time=datetime.timedelta(
                    seconds=nwb_epoch.start_time.item()
                )
                + session.session_start_time,
                stimulus_end_time=datetime.timedelta(seconds=nwb_epoch.stop_time.item())
                + session.session_start_time,
                stimulus_name=script_name,
                code=get_code(script_name),
                stimulus_modalities=get_modalities(script_name, session),
                performance_metrics=get_performance_metrics(script_name),
                notes=nwb_epoch.notes.item(),
                active_devices=get_active_devices(script_name, session),
                configurations=get_configurations(script_name),
                training_protocol_name=get_training_protocol_name(script_name, session),
                curriculum_status=get_curriculum_status(script_name, session),
            )
        )
    return aind_epochs


def _get_data_streams(
    session: DynamicRoutingSession,
) -> list[aind_data_schema.core.acquisition.DataStream]:
    def get_core_dependency(
        stream_name: str,
    ) -> aind_data_schema.core.acquisition.Code | None:
        match stream_name:
            case "Ephys":
                assert (
                    session.is_ephys
                ), "Ephys dependency info requested for non-ephys session - should not be possible"
                try:
                    version = session.ephys_settings_xml_data.open_ephys_version
                except (AttributeError, FileNotFoundError):
                    version = "0.6.6"  # most-commonly used version
                return aind_data_schema.components.identifiers.Software(
                    name="Open Ephys GUI",
                    version=version,
                )
            case "Camstim":
                return aind_data_schema.components.identifiers.Software(
                    name="PsychoPy",
                    version="2022.1.2",
                )

            case "MVR":
                assert (
                    session.is_video
                ), "MVR dependency info requested for non-video session - should not be possible"
                try:
                    version = next(iter(session.mvr.info_data.values()))["MVR Version"]
                except (StopIteration, AttributeError, KeyError, FileNotFoundError):
                    version = "1.1.7"  # most-commonly used version
                return aind_data_schema.components.identifiers.Software(
                    name="MultiVideoRecorder",
                    version=version,
                )
            case "Sync":
                return aind_data_schema.components.identifiers.Software(
                    name="nidaqmx",
                    version="0.6.2",
                )
            case _:
                return None

    def get_np_services_code(
        stream_name: str,
    ) -> list[aind_data_schema.core.acquisition.Code] | None:
        if stream_name not in ("MVR", "Sync", "Camstim", "Ephys"):
            raise ValueError(f"Unknown stream name {stream_name} for NP services code")
        script_name = {
            "MVR": "proxies",
            "Sync": "proxies",
            "Camstim": "proxies",
            "Ephys": "open_ephys",
        }
        url = f"https://github.com/AllenInstitute/np_services/blob/main/src/np_services/{script_name[stream_name]}.py"
        return [
            aind_data_schema.core.acquisition.Code(
                url=url,
                container=None,
                language="Python",
                language_version="3.9",
                core_dependency=get_core_dependency(stream_name),
            )
        ]

    data_streams = []
    modality = aind_data_schema_models.modalities.Modality

    if session.is_sync:
        data_streams.append(
            aind_data_schema.core.acquisition.DataStream(
                stream_start_time=session.sync_data.start_time,
                stream_end_time=session.sync_data.stop_time,
                modalities=[modality.BEHAVIOR, modality.BEHAVIOR_VIDEOS],
                code=get_np_services_code("Sync"),
                active_devices=[instrument.SYNC_DAQ.name],
                configurations=[],
            )
        )
    if session.stim_paths and len(session.epochs.script_name):
        data_streams.append(
            aind_data_schema.core.acquisition.DataStream(
                stream_start_time=session.session_start_time
                + datetime.timedelta(seconds=min(session.epochs.stop_time)),
                stream_end_time=session.session_start_time
                + datetime.timedelta(seconds=max(session.epochs.stop_time)),
                modalities=[modality.BEHAVIOR],
                active_devices=sorted(
                    set(
                        itertools.chain.from_iterable(
                            get_active_devices(s, session)
                            for s in session.epochs.script_name
                        )
                    )
                ),
                code=get_np_services_code("Camstim"),
                configurations=[],
            )
        )
    if session.is_video:
        if session.is_sync:
            active_cameras = [
                instrument.EYE_CAMERA,
                instrument.FRONT_CAMERA,
                instrument.SIDE_CAMERA,
                instrument.NOSE_CAMERA,
            ]
        else:
            active_cameras = [instrument.SIDE_CAMERA]
        data_streams.append(
            aind_data_schema.core.acquisition.DataStream(
                stream_start_time=session.session_start_time
                + datetime.timedelta(
                    seconds=min(
                        np.nanmin(times.timestamps)
                        for times in session._video_frame_times
                    )  # min frame time across all vids
                ),
                stream_end_time=session.session_start_time
                + datetime.timedelta(
                    seconds=max(
                        np.nanmax(times.timestamps)
                        for times in session._video_frame_times
                    )  # max frame time across all vids
                ),
                modalities=[modality.BEHAVIOR_VIDEOS],
                active_devices=[cam.name for cam in active_cameras],
                code=get_np_services_code("MVR"),
                configurations=[],
            )
        )
    if session.is_ephys:
        # Build manipulator configurations for each probe

        if is_surface_recording(session):
            rec = session.main_recording
        else:
            rec = session

        if (
            path := next(
                (p for p in rec.raw_data_paths if p.name.endswith("_dye.json")), None
            )
        ) is not None:
            dye = json.loads(path.read_text())["dye"]
        else:
            dye = "unknown"

        ephys_configs = []
        for probe in session.probe_letters_to_use:
            translation = []
            # a few sessions didn't have newscale logging enabled: no way to get their coords
            with contextlib.suppress(AttributeError):
                row = rec._manipulator_positions[:].query(
                    f"electrode_group_name == '{probe.name}'"
                )
                if not row.empty:
                    translation = [row[axis].item() for axis in ["x", "y", "z"]]

            manipulator_config = aind_data_schema.components.configs.ManipulatorConfig(
                device_name=probe.id,
                coordinate_system=aind_data_schema.components.coordinates.CoordinateSystem(
                    name=f"{probe.name} XYZ",
                    origin=aind_data_schema.components.coordinates.Origin.TIP,
                    axes=[
                        aind_data_schema.components.coordinates.Axis(
                            name=aind_data_schema.components.coordinates.AxisName.X,
                            direction=aind_data_schema.components.coordinates.Direction.LR,
                        ),
                        aind_data_schema.components.coordinates.Axis(
                            name=aind_data_schema.components.coordinates.AxisName.Y,
                            direction=aind_data_schema.components.coordinates.Direction.BF,
                        ),
                        aind_data_schema.components.coordinates.Axis(
                            name=aind_data_schema.components.coordinates.AxisName.Z,
                            direction=aind_data_schema.components.coordinates.Direction.SI,
                        ),
                    ],
                    axis_unit=aind_data_schema.components.coordinates.SizeUnit.UM,
                ),
                local_axis_positions=aind_data_schema.components.coordinates.Translation(
                    translation=translation,
                ),
            )
            match probe:
                case "A":
                    primary_targeted_structure = (
                        aind_data_schema_models.brain_atlas.CCFv3.ROOT
                    )
                case "B":
                    primary_targeted_structure = (
                        aind_data_schema_models.brain_atlas.CCFv3.ROOT
                    )
                case "C":
                    primary_targeted_structure = (
                        aind_data_schema_models.brain_atlas.CCFv3.ROOT
                    )
                case "D":
                    primary_targeted_structure = (
                        aind_data_schema_models.brain_atlas.CCFv3.ROOT
                    )
                case "E":
                    primary_targeted_structure = (
                        aind_data_schema_models.brain_atlas.CCFv3.ROOT
                    )
                case "F":
                    primary_targeted_structure = (
                        aind_data_schema_models.brain_atlas.CCFv3.ROOT
                    )

            probe_config = aind_data_schema.components.configs.ProbeConfig(
                device_name=probe.name,
                primary_targeted_structure=primary_targeted_structure,
                coordinate_system=manipulator_config.coordinate_system,
                transform=[manipulator_config.local_axis_positions],
                dye=dye,
            )
            ephys_assembly = aind_data_schema.components.configs.EphysAssemblyConfig(
                device_name=probe.id,
                manipulator=manipulator_config,
                probes=[probe_config],
                modules=None,
            )
            ephys_configs.append(ephys_assembly)
        data_streams.append(
            aind_data_schema.core.acquisition.DataStream(
                stream_start_time=session.session_start_time
                + datetime.timedelta(
                    seconds=min(
                        timing.start_time for timing in session.ephys_timing_data
                    )
                ),
                stream_end_time=session.session_start_time
                + datetime.timedelta(
                    seconds=max(
                        timing.stop_time for timing in session.ephys_timing_data
                    )
                ),
                modalities=[modality.ECEPHYS],
                active_devices=[
                    probe.name for probe in session.probe_letters_to_use
                ],  # keep names synced with 'instrument.py'
                configurations=ephys_configs,
                code=get_np_services_code("Ephys"),
            )
        )
    return data_streams


if __name__ == "__main__":
    session = DynamicRoutingSession("814666_20251107")
    metadata = get_acquisition_model(session)
    print(metadata.model_dump_json(indent=2))
    with open("acquisition_814666_2025-11-07.json", "w") as f:
        f.write(metadata.model_dump_json(indent=2))
    # session = DynamicRoutingSession('628801_2022-09-20')
    # metadata = get_acquisition_model(session)
    # print(metadata.model_dump_json(indent=2))
    # with open('acquisition_628801_2022-09-20.json', 'w') as f:
    #     f.write(metadata.model_dump_json(indent=2))
