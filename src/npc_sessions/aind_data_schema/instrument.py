import contextlib
import logging
import re
from typing import Literal

import aind_data_schema.components.connections
import aind_data_schema.components.coordinates
import aind_data_schema.components.devices
import aind_data_schema.components.identifiers
import aind_data_schema.core.instrument
import aind_data_schema_models.coordinates
import aind_data_schema_models.devices
import aind_data_schema_models.modalities
import aind_data_schema_models.organizations
import aind_data_schema_models.units
from aind_data_schema.components.coordinates import Axis, AxisName, Direction

from npc_sessions.sessions import DynamicRoutingSession

logger = logging.getLogger(__name__)


RIG_COORDINATE_SYSTEM = aind_data_schema.components.coordinates.CoordinateSystem(
    name="BREGMA_PRU",
    origin=aind_data_schema.components.coordinates.Origin.BREGMA,
    axes=[
        Axis(name=AxisName.X, direction=Direction.PA),
        Axis(name=AxisName.Y, direction=Direction.RL),
        Axis(name=AxisName.Z, direction=Direction.DU),
    ],
    axis_unit=aind_data_schema.components.coordinates.SizeUnit.M,
)
MONITOR_COORDINATE_SYSTEM = aind_data_schema.components.coordinates.CoordinateSystem(
    name="MINDSCOPE_COPA_MONITOR_RDB",
    origin=aind_data_schema.components.coordinates.Origin.FRONT_CENTER,
    axes=[
        Axis(name=AxisName.X, direction=Direction.RL),
        Axis(name=AxisName.Y, direction=Direction.DU),
        Axis(name=AxisName.Z, direction=Direction.BF),
    ],
    axis_unit=aind_data_schema.components.coordinates.SizeUnit.M,
)
CAMERA_COORDINATE_SYSTEM = aind_data_schema.components.coordinates.CoordinateSystem(
    name="MINDSCOPE_COPA_CAMERA_LUB",
    origin=aind_data_schema.components.coordinates.Origin.FRONT_CENTER,
    axes=[
        Axis(name=AxisName.X, direction=Direction.LR),
        Axis(name=AxisName.Y, direction=Direction.UD),
        Axis(name=AxisName.Z, direction=Direction.BF),
    ],
    axis_unit=aind_data_schema.components.coordinates.SizeUnit.M,
)


def _normalize_rig_name(rig_name: str) -> str:
    return rig_name.replace(".", "").replace("-", "").lower()


def is_np_rig(rig_name: str) -> bool:
    return _normalize_rig_name(rig_name).startswith("np")


def is_behavior_box(rig_name: str) -> bool:
    return _normalize_rig_name(rig_name).startswith("b")


def is_og_rig(rig_name: str) -> bool:
    return _normalize_rig_name(rig_name).startswith("og")


def get_location(rig_name: str) -> str:
    """Get a Location object for a given rig name."""
    rig_name = _normalize_rig_name(rig_name)
    if rig_name.startswith("np"):
        return {
            "np0": "325",
            "np1": "325",
            "np2": "327",
            "np3": "342",
        }[rig_name]
    if rig_name == "og1":
        return "342"
    if rig_name.startswith("beh"):
        return "NSB"
    else:
        return "342"  # behavior B


DISC = aind_data_schema.components.devices.Disc(
    name="Brain Observatory running disc",
    radius="4.69",
    radius_unit="centimeter",
    output=aind_data_schema_models.devices.DaqChannelType.DO,
    surface_material="rubber",
    notes="Radius is distance from center to subject",
)

MONITOR = aind_data_schema.components.devices.Monitor(
    name="Stimulus monitor",
    manufacturer=aind_data_schema_models.organizations.Organization.ASUS,
    model="PA248",
    refresh_rate=60,
    width=1920,
    height=1200,
    size_unit=aind_data_schema_models.units.SizeUnit.PX,
    viewing_distance=15.3,
    viewing_distance_unit=aind_data_schema_models.units.SizeUnit.CM,
    brightness=43,
    contrast=50,
    coordinate_system=MONITOR_COORDINATE_SYSTEM,
    relative_position=[
        aind_data_schema_models.coordinates.AnatomicalRelative.RIGHT,
        aind_data_schema_models.coordinates.AnatomicalRelative.ANTERIOR,
        aind_data_schema_models.coordinates.AnatomicalRelative.SUPERIOR,
    ],
    transform=[
        aind_data_schema.components.coordinates.Affine(
            affine_transform=[
                [-0.80914, -0.58761, 0],
                [-0.12391, 0.17063, 0.97751],
                [-0.5744, 0.79095, -0.21087],
                [0.08751, -0.12079, 0.02298],
            ]
        ),
    ],
)


SPEAKER = aind_data_schema.components.devices.Speaker(
    name="Stimulus speaker",
    manufacturer=aind_data_schema_models.organizations.Organization.ISL,
    model="SPK-I-81345",
    coordinate_system=MONITOR_COORDINATE_SYSTEM,  # same system as monitor
    relative_position=[
        aind_data_schema_models.coordinates.AnatomicalRelative.RIGHT,
        aind_data_schema_models.coordinates.AnatomicalRelative.ANTERIOR,
        aind_data_schema_models.coordinates.AnatomicalRelative.SUPERIOR,
    ],
    transform=[
        aind_data_schema.components.coordinates.Affine(
            affine_transform=[
                [-0.82783, -0.4837, -0.28412],
                [-0.55894, 0.75426, 0.34449],
                [0.04767, 0.44399, -0.89476],
                [-0.00838, -0.09787, 0.18228],
            ]
        ),
    ],
)

LICK_SPOUT_ASSEMBLY = aind_data_schema.components.devices.LickSpoutAssembly(
    name="Reward delivery assembly",
    lick_spouts=[
        aind_data_schema.components.devices.LickSpout(
            name="Reward spout",
            manufacturer=aind_data_schema_models.organizations.Organization.HAMILTON,
            model="8649-01",
            spout_diameter=0.672,
            spout_diameter_unit=aind_data_schema_models.units.SizeUnit.MM,
            notes="Spout diameter is inner. Outer diameter is 1.575 mm.",
            solenoid_valve=aind_data_schema.components.devices.Device(
                name="Solenoid valve",
                manufacturer=aind_data_schema_models.organizations.Organization.NRESEARCH_INC,
                model="161K011",
            ),
            lick_sensor=aind_data_schema.components.devices.Device(
                name="Lick sensor",
                model="1007079-1",
                manufacturer=aind_data_schema_models.organizations.Organization.TE_CONNECTIVITY,
            ),
            lick_sensor_type=aind_data_schema_models.devices.LickSensorType.PIEZOELECTIC,
        ),
    ],
)

SYNC_DAQ = aind_data_schema.components.devices.DAQDevice(
    name="Sync DAQ",
    manufacturer=aind_data_schema_models.organizations.Organization.NATIONAL_INSTRUMENTS,
    model="NI-6612",
    data_interface=aind_data_schema_models.devices.DataInterface.PCIE,
)
TASKCONTROL_DAQ = aind_data_schema.components.devices.DAQDevice(
    name="TaskControl DAQ",
    manufacturer=aind_data_schema_models.organizations.Organization.NATIONAL_INSTRUMENTS,
    model="6001",
    data_interface=aind_data_schema_models.devices.DataInterface.USB,
)
CAMSTIM_DAQ = aind_data_schema.components.devices.DAQDevice(
    name="Camstim DAQ",
    manufacturer=aind_data_schema_models.organizations.Organization.NATIONAL_INSTRUMENTS,
    model="6323",
    data_interface=aind_data_schema_models.devices.DataInterface.PCIE,
)
OPTO_DAQ = aind_data_schema.components.devices.DAQDevice(
    manufacturer=aind_data_schema_models.organizations.Organization.NATIONAL_INSTRUMENTS,
    name="Opto DAQ",
    model="9264",
    data_interface=aind_data_schema_models.devices.DataInterface.ETH,
)
EPHYS_DAQ = aind_data_schema.components.devices.DAQDevice(
    manufacturer=aind_data_schema_models.organizations.Organization.NATIONAL_INSTRUMENTS,
    name="Ephys DAQ",
    model="6133",
    data_interface=aind_data_schema_models.devices.DataInterface.PXI,
)

PHOTODIODE = aind_data_schema.components.devices.Detector(
    name="Stimulus photodiode",
    manufacturer=aind_data_schema_models.organizations.Organization.THORLABS,
    model="PDA25K",
    data_interface=aind_data_schema_models.devices.DataInterface.COAX,
    detector_type=aind_data_schema_models.devices.DetectorType.OTHER,
    cooling=aind_data_schema_models.devices.Cooling.NO_COOLING,
    notes="Photodiode used to measure stimulus monitor frame updates",
)
MICROPHONE = aind_data_schema.components.devices.Device(
    name="Stimulus microphone",
    manufacturer=aind_data_schema_models.organizations.Organization.DODOTRONIC,
    model="momimic",
    notes="Microphone used to record stimulus speaker output from approximately 50 mm",
)

LASER_488 = aind_data_schema.components.devices.Laser(
    name="Opto laser #1",
    manufacturer=aind_data_schema_models.organizations.Organization.VORTRAN,
    model="Stradus 488-50",
    wavelength=488,
    wavelength_unit=aind_data_schema_models.units.SizeUnit.NM,
)
LASER_633 = LASER_488.model_copy(
    update={
        "name": "Opto laser #2",
        "model": "Stradus 633-50",
        "wavelength": 633,
    }
)

LASER_GALVO_X = aind_data_schema.components.devices.AdditionalImagingDevice(
    name="Opto laser galvo X",
    manufacturer=aind_data_schema_models.organizations.Organization.THORLABS,
    model="GVS012",
    imaging_device_type=aind_data_schema.components.devices.ImagingDeviceType.GALVO,
)
LASER_GALVO_Y = LASER_GALVO_X.model_copy(
    update={"name": LASER_GALVO_X.name.replace("X", "Y")}
)

ALLIED_CAMERA = aind_data_schema.components.devices.Camera(
    name="Generic camera",
    manufacturer=aind_data_schema_models.organizations.Organization.ALLIED,
    model="G-032",
    data_interface=aind_data_schema_models.devices.DataInterface.ETH,
    cooling=aind_data_schema_models.devices.Cooling.NO_COOLING,
    frame_rate=60,
    frame_rate_unit=aind_data_schema_models.units.FrequencyUnit.HZ,
    sensor_width=636,
    sensor_height=508,
    size_unit=aind_data_schema_models.units.SizeUnit.PX,
    chroma=aind_data_schema_models.devices.CameraChroma.BW,
    recording_software=aind_data_schema.components.identifiers.Software(
        name="MultiVideoRecorder",
        version="1.1.7",
    ),
)
FRONT_CAMERA = ALLIED_CAMERA.model_copy(update={"name": "Front camera"})
FRONT_CAMERA_ASSEMBLY = aind_data_schema.components.devices.CameraAssembly(
    name="Front camera assembly",
    target=aind_data_schema.components.devices.CameraTarget.FACE,
    camera=FRONT_CAMERA,
    lens=aind_data_schema.components.devices.Lens(
        name="Front camera lens",
        manufacturer=aind_data_schema_models.organizations.Organization.EDMUND_OPTICS,
        model="86604",
        additional_settings=dict(
            focal_length=8.5,
            focal_length_unit=aind_data_schema_models.units.SizeUnit.MM,
        ),
    ),
    filter=aind_data_schema.components.devices.Filter(
        name="Front camera filter",
        manufacturer=aind_data_schema_models.organizations.Organization.SEMROCK,
        model="FF01-715",
        filter_type=aind_data_schema_models.devices.FilterType.LONGPASS,
        cut_on_wavelength=715,
        wavelength_unit=aind_data_schema_models.units.SizeUnit.NM,
    ),
    coordinate_system=CAMERA_COORDINATE_SYSTEM,
    relative_position=[
        aind_data_schema_models.coordinates.AnatomicalRelative.ANTERIOR,
        aind_data_schema_models.coordinates.AnatomicalRelative.SUPERIOR,
    ],
    transform=[
        aind_data_schema.components.coordinates.Affine(
            affine_transform=[
                [-0.17365, 0.98481, 0],
                [0.44709, 0.07883, -0.89101],
                [-0.87747, -0.15472, -0.45399],
                [0.154, 0.03078, 0.06346],
            ],
        ),
    ],
)

SIDE_CAMERA = ALLIED_CAMERA.model_copy(update={"name": "Side camera"})
SIDE_CAMERA_ASSEMBLY = aind_data_schema.components.devices.CameraAssembly(
    name="Side camera assembly",
    target=aind_data_schema.components.devices.CameraTarget.BODY,
    camera=SIDE_CAMERA,
    filter=aind_data_schema.components.devices.Filter(
        name="Side camera filter",
        manufacturer=aind_data_schema_models.organizations.Organization.SEMROCK,
        model="FF01-747",
        filter_type=aind_data_schema_models.devices.FilterType.BANDPASS,
        cut_on_wavelength=730,
        cut_off_wavelength=764,
        wavelength_unit=aind_data_schema_models.units.SizeUnit.NM,
    ),
    lens=aind_data_schema.components.devices.Lens(
        name="Side camera lens",
        manufacturer=aind_data_schema_models.organizations.Organization.NAVITAR,
        additional_settings=dict(
            focal_length=6.0,
            focal_length_unit=aind_data_schema_models.units.SizeUnit.MM,
        ),
    ),
    coordinate_system=CAMERA_COORDINATE_SYSTEM,
    relative_position=[
        aind_data_schema_models.coordinates.AnatomicalRelative.LEFT,
    ],
    transform=[
        aind_data_schema.components.coordinates.Affine(
            affine_transform=[
                [-1, 0, 0],
                [0, -1, 0],
                [-1, 0, -0.03617],
                [0.23887, -0.02535],
            ],
        ),
    ],
)

EYE_CAMERA = ALLIED_CAMERA.model_copy(
    update={
        "name": "Eye camera",
        "notes": "There is a mirror in the light path between the eye and the camera.",
    }
)
EYE_CAMERA_ASSEMBLY = aind_data_schema.components.devices.CameraAssembly(
    name="Eye camera assembly",
    target=aind_data_schema.components.devices.CameraTarget.EYE,
    camera=EYE_CAMERA,
    filter=aind_data_schema.components.devices.Filter(
        name="Eye camera filter",
        manufacturer=aind_data_schema_models.organizations.Organization.SEMROCK,
        model="FF01-850",
        filter_type=aind_data_schema_models.devices.FilterType.BANDPASS,
        cut_on_wavelength=845,
        cut_off_wavelength=855,
        wavelength_unit=aind_data_schema_models.units.SizeUnit.NM,
    ),
    lens=aind_data_schema.components.devices.Lens(
        name="Eye camera lens",
        manufacturer=aind_data_schema_models.organizations.Organization.INFINITY_PHOTO_OPTICAL,
        model="213073",
        additional_settings=dict(
            focal_length=6.0,
            focal_length_unit=aind_data_schema_models.units.SizeUnit.MM,
        ),
    ),
    coordinate_system=CAMERA_COORDINATE_SYSTEM,
    relative_position=[
        aind_data_schema_models.coordinates.AnatomicalRelative.RIGHT,
    ],
    transform=[
        aind_data_schema.components.coordinates.Affine(
            affine_transform=[
                [-0.5, -0.86603, 0],
                [-0.366, 0.21131, -0.90631],
                [0.78489, -0.45315, -0.42262],
                [-0.14259, 0.06209, 0.09576],
                [0.78489, -0.45315, -0.42262],
                [-0.14259, 0.06209, 0.09576],
            ],
        ),
    ],
)

NOSE_CAMERA = ALLIED_CAMERA.model_copy(update={"name": "Nose camera"})
NOSE_CAMERA_ASSEMBLY = aind_data_schema.components.devices.CameraAssembly(
    name="Nose camera assembly",
    target=aind_data_schema.components.devices.CameraTarget.FACE,
    camera=NOSE_CAMERA,
    filter=aind_data_schema.components.devices.Filter(
        name="Nose camera filter",
        manufacturer=aind_data_schema_models.organizations.Organization.SEMROCK,
        model="FF01-715",
        filter_type=aind_data_schema_models.devices.FilterType.LONGPASS,
        cut_on_wavelength=715,
        wavelength_unit=aind_data_schema_models.units.SizeUnit.NM,
    ),
    lens=aind_data_schema.components.devices.Lens(
        name="Nose camera lens",
        manufacturer=aind_data_schema_models.organizations.Organization.EDMUND_OPTICS,
        model="85360",
        additional_settings=dict(
            focal_length=25,
            focal_length_unit=aind_data_schema_models.units.SizeUnit.MM,
        ),
    ),
    relative_position=[
        aind_data_schema_models.coordinates.AnatomicalRelative.ANTERIOR,
        aind_data_schema_models.coordinates.AnatomicalRelative.LEFT,
    ],
    # !pending:
    # coordinate_system=CAMERA_COORDINATE_SYSTEM,
    # transform=[
    #     aind_data_schema.components.coordinates.Affine(
    #         affine_transform=[[], [], [], []]
    #     ),
    # ],
)
FRONT_LED = aind_data_schema.components.devices.LightEmittingDiode(
    manufacturer=aind_data_schema_models.organizations.Organization.AMS_OSRAM,
    name="Front camera illumination LED",
    model="LZ4-40R308-0000",
    wavelength=740,
    wavelength_unit=aind_data_schema_models.units.SizeUnit.NM,
)
SIDE_LED = FRONT_LED.model_copy(
    update={"name": FRONT_LED.name.replace("Front", "Side")}
)
NOSE_LED = FRONT_LED.model_copy(
    update={"name": FRONT_LED.name.replace("Front", "Nose")}
)

EYE_LED = aind_data_schema.components.devices.LightEmittingDiode(
    manufacturer=aind_data_schema_models.organizations.Organization.AMS_OSRAM,
    name="Eye camera illumination LED",
    model="LZ4-40R608-0000",
    wavelength=850,
    wavelength_unit=aind_data_schema_models.units.SizeUnit.NM,
)


def get_basestation(
    slot: Literal[2, 3], session: DynamicRoutingSession | None = None
) -> aind_data_schema.components.devices.NeuropixelsBasestation:
    probe_letters = "ABC" if slot == 2 else "DEF"
    basestation_firmware_version = "240.1120"
    bsc_firmware_version = "1.0.144"
    if session is not None and session.is_ephys:
        with contextlib.suppress(
            ValueError, FileNotFoundError, AttributeError, IndexError
        ):
            settings_xml_path = session.ephys_settings_xml_path
            text = settings_xml_path.read_text()
            basestation_firmware_version = re.search(r'bs_firmware_version="([\d.]+)"', text).group(1)  # type: ignore [union-attr]
            bsc_firmware_version = re.search(r'bsc_firmware_version="([\d.]+)"', text).group(1)  # type: ignore [union-attr]
    return aind_data_schema.components.devices.NeuropixelsBasestation(
        name=f"probes {probe_letters}",
        manufacturer=aind_data_schema_models.organizations.Organization.IMEC,
        model="PXIe",
        basestation_firmware_version=basestation_firmware_version,
        bsc_firmware_version=bsc_firmware_version,
        slot=slot,
        ports=[
            aind_data_schema.components.devices.ProbePort(
                index=i, probes=[f"probe{letter}"]
            )
            for i, letter in zip(range(3), probe_letters, strict=False)
        ],
    )


def get_computers(
    session: DynamicRoutingSession,
) -> list[aind_data_schema.components.devices.Computer]:
    rig_name = _normalize_rig_name(session.rig)
    computer_names = []
    if rig_name.startswith("og") or rig_name.startswith("np"):
        computer_names.extend(["STIM", "SYNC", "MON"])
    else:
        computer_names.append("BEH")
    if rig_name.startswith("np"):
        computer_names.append("ACQ")
    return [
        aind_data_schema.components.devices.Computer(
            name=name, operating_system="Windows 10"
        )
        for name in computer_names
    ]


def get_components_and_connections(
    session: DynamicRoutingSession,
) -> tuple[
    list[aind_data_schema.components.devices.Device],
    list[aind_data_schema.components.connections.Connection],
]:
    components = [
        DISC,
        MONITOR,
        SPEAKER,
        LICK_SPOUT_ASSEMBLY,
        CAMSTIM_DAQ,
        TASKCONTROL_DAQ,
        *get_computers(session),
    ]
    connections = [
        aind_data_schema.components.connections.Connection(
            source_device=LICK_SPOUT_ASSEMBLY.name,
            target_device=CAMSTIM_DAQ.name,
            send_and_receive=True,
        ),
        aind_data_schema.components.connections.Connection(
            source_device=DISC.name,
            target_device=CAMSTIM_DAQ.name,
        ),
        aind_data_schema.components.connections.Connection(
            target_device=TASKCONTROL_DAQ.name,
            source_device=SPEAKER.name,
        ),
    ]
    if is_behavior_box(session.rig):
        components.extend(
            [
                SIDE_CAMERA_ASSEMBLY,
                SIDE_LED,
            ]
        )
    if is_og_rig(session.rig) or is_np_rig(session.rig):
        components.extend(
            [
                SYNC_DAQ,
                MICROPHONE,
                PHOTODIODE,
                FRONT_CAMERA_ASSEMBLY,
                FRONT_LED,
                SIDE_CAMERA_ASSEMBLY,
                SIDE_LED,
                EYE_CAMERA_ASSEMBLY,
                EYE_LED,
                NOSE_CAMERA_ASSEMBLY,
                NOSE_LED,
                OPTO_DAQ,
                LASER_488,
                LASER_633,
                LASER_GALVO_X,
                LASER_GALVO_Y,
            ]
        )
        connections.extend(
            [
                aind_data_schema.components.connections.Connection(
                    source_device=source_device.name,
                    target_device=SYNC_DAQ.name,
                )
                for source_device in (
                    PHOTODIODE,
                    FRONT_CAMERA_ASSEMBLY.camera,
                    SIDE_CAMERA_ASSEMBLY.camera,
                    EYE_CAMERA_ASSEMBLY.camera,
                    NOSE_CAMERA_ASSEMBLY.camera,
                    LICK_SPOUT_ASSEMBLY.lick_spouts[0],
                    LASER_488,
                    LASER_633,
                )
            ]
        )
        connections.extend(
            [
                aind_data_schema.components.connections.Connection(
                    source_device=OPTO_DAQ.name,
                    target_device=target_device.name,
                )
                for target_device in (
                    LASER_488,
                    LASER_633,
                    LASER_GALVO_X,
                    LASER_GALVO_Y,
                )
            ]
        )
    if is_np_rig(session.rig):
        components.extend(
            [
                *get_ephys_assemblies(session),
                get_basestation(slot=2, session=session),
                get_basestation(slot=3, session=session),
                EPHYS_DAQ,
            ]
        )
        connections.extend(
            [
                aind_data_schema.components.connections.Connection(
                    source_device=source_device.name,
                    target_device=EPHYS_DAQ.name,
                )
                for source_device in (
                    SPEAKER,
                    MICROPHONE,
                    PHOTODIODE,
                )
            ]
        )
    return components, connections


def get_modalities(
    session: DynamicRoutingSession,
) -> list[aind_data_schema_models.modalities.Modality]:
    modality = aind_data_schema_models.modalities.Modality
    modalities = [modality.BEHAVIOR, modality.BEHAVIOR_VIDEOS]
    if is_np_rig(session.rig):
        modalities.append(modality.ECEPHYS)
    return modalities


def get_ephys_assemblies(
    session: DynamicRoutingSession,
) -> list[aind_data_schema.components.devices.EphysAssembly]:
    """Get ephys assemblies for the session."""
    if not session.is_ephys:
        return []
    assemblies = []
    for probe_letter in ["A", "B", "C", "D", "E", "F"]:
        assemblies.append(
            aind_data_schema.components.devices.EphysAssembly(
                name=probe_letter,
                manipulator=aind_data_schema.components.devices.Manipulator(
                    name=f"Manipulator {probe_letter}",
                    manufacturer=aind_data_schema_models.organizations.Organization.NEW_SCALE_TECHNOLOGIES,
                    model="M3-LS-3.4-15",
                ),
                probes=[
                    aind_data_schema.components.devices.EphysProbe(
                        name=f"probe{probe_letter}",
                        manufacturer=aind_data_schema_models.organizations.Organization.IMEC,
                        probe_model=aind_data_schema.components.devices.ProbeModel.NP1,
                    )
                ],
            )
        )
    return assemblies


def get_instrument_model(
    session: DynamicRoutingSession,
) -> aind_data_schema.core.instrument.Instrument:
    """Get the Pydantic model corresponding to the 'instrument.json' for a given session."""

    return aind_data_schema.core.instrument.Instrument(
        location=get_location(session.rig),
        instrument_id=session.rig,
        modification_date=session.session_start_time.date(),
        modalities=get_modalities(session),
        calibrations=[],
        coordinate_system=RIG_COORDINATE_SYSTEM,
        temperature_control=None,
        notes=None,
        connections=(
            components_and_connections := get_components_and_connections(session)
        )[1],
        components=components_and_connections[0],
    )


if __name__ == "__main__":
    session = DynamicRoutingSession("814666_20251107")
    metadata = get_instrument_model(session)
    print(metadata.model_dump_json(indent=2))
    with open("instrument_814666_2025-11-07.json", "w") as f:
        f.write(metadata.model_dump_json(indent=2))
