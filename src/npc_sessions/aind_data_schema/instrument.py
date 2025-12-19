
import logging
from typing import TypeGuard

import aind_data_schema.base
import aind_data_schema.components.configs
import aind_data_schema.components.coordinates
import aind_data_schema.components.devices
import aind_data_schema.components.identifiers
import aind_data_schema.core.acquisition
import aind_data_schema.core.instrument
import aind_data_schema_models.modalities
import aind_data_schema_models.devices
import aind_data_schema_models.coordinates
import aind_data_schema_models.units
import aind_data_schema_models.organizations


from npc_sessions.sessions import DynamicRoutingSession, DynamicRoutingSurfaceRecording

logger = logging.getLogger(__name__)

COPA_NOTES = (
    "The rotation matrix is represented as: a,b,c,d,e,f,g,h,i. Wherein a, b, "
    "c correspond to the first row of the matrix. The translation matrix is "
    "represented as: x,y,z."
)

def _normalize_rig_name(rig_name: str) -> str:
    return rig_name.replace('.', '').replace('-', '').lower()

def is_np_rig(rig_name: str) -> bool:
    return _normalize_rig_name(rig_name).startswith('np')

def is_behavior_box(rig_name: str) -> bool:
    return _normalize_rig_name(rig_name).startswith('b')

def is_og_rig(rig_name: str) -> bool:
    return _normalize_rig_name(rig_name).startswith('og')

def get_location(rig_name: str) -> str:
    """Get a Location object for a given rig name."""
    rig_name = _normalize_rig_name(rig_name)
    if rig_name.startswith('np'):
        return {
            'np0': '325',
            'np1': '325',
            'np2': '327',
            'np3': '342',
        }[rig_name]
    if rig_name == 'og1':
        return '342'
    if rig_name.startswith('beh'):
        return 'NSB'
    else:
        return '342' # behavior B

DISC = aind_data_schema.components.devices.Disc(
    name="Brain Observatory Mouse Platform",
    radius="4.69",
    radius_unit="centimeter",
    output=aind_data_schema_models.devices.DaqChannelType.DO,
    surface_material="rubber",
    notes="Radius is distance from center to subject",
)

MONITOR = aind_data_schema.components.devices.Monitor(
    name="Monitor",
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
    coordinate_system=aind_data_schema.components.coordinates.CoordinateSystemLibrary.SIPE_MONITOR_RTF,
    relative_position=[
        aind_data_schema_models.coordinates.AnatomicalRelative.RIGHT,
        aind_data_schema_models.coordinates.AnatomicalRelative.ANTERIOR,
        aind_data_schema_models.coordinates.AnatomicalRelative.SUPERIOR,
    ],
    transform=[
        aind_data_schema.components.coordinates.Rotation(
            angles=[
                -0.80914,
                -0.58761,
                0,
                -0.12391,
                0.17063,
                0.97751,
                0.08751,
                -0.12079,
                0.02298,
            ],
            angles_unit=aind_data_schema_models.units.AngleUnit.RAD,
        ),
        aind_data_schema.components.coordinates.Translation(
            distances=[0.08751, -0.12079, 0.02298],
            distances_unit=aind_data_schema_models.units.SizeUnit.M,
        ),
    ],
    notes=COPA_NOTES,
)

def get_modalities(session: DynamicRoutingSession) -> list[aind_data_schema_models.modalities.Modality]:
    modality = aind_data_schema_models.modalities.Modality
    modalities = [modality.BEHAVIOR, modality.BEHAVIOR_VIDEOS]
    if is_np_rig(session.rig):
        modalities.append(modality.ECEPHYS)
    return modalities

def get_components(session: DynamicRoutingSession) -> list[aind_data_schema.components.devices.Device]:
    devices = aind_data_schema.components.devices
    components = [
        DISC,
        MONITOR,
        devices.Speaker,
        devices.LickSpoutAssembly,
    ]
    if is_behavior_box(session.rig):
        components.extend([
            devices.CameraAssembly,
            devices.Computer,
        ])
    if is_og_rig(session.rig) or is_np_rig(session.rig):
        components.extend([
            devices.DAQDevice, # sync
            devices.Laser,
            devices.CameraAssembly,
            devices.CameraAssembly,
            devices.CameraAssembly,
            devices.CameraAssembly,
            devices.Computer,
            devices.Computer,
            devices.Computer,
        ])
    if is_np_rig(session.rig):
        components.extend([
            devices.EphysAssembly,
            devices.EphysAssembly,
            devices.EphysAssembly,
            devices.EphysAssembly,
            devices.EphysAssembly,
            devices.EphysAssembly,
            devices.NeuropixelsBasestation,
            devices.Computer,
        ])
    return components

coordinate_system = aind_data_schema.components.coordinates.CoordinateSystem(
    name="rig-based XYZ",
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
)

shared_camera_props = {
    "detector_type": "Camera",
    "model": "G-032",
    "max_frame_rate": 102,
    "sensor_width": 7400,
    "sensor_height": 7400,
    "size_unit": devices.SizeUnit.NM,
    "notes": "Max frame rate is at maximum resolution.",
    "cooling": devices.Cooling.NONE,
}

def get_instrument_model(session: DynamicRoutingSession) -> aind_data_schema.core.instrument.Instrument:
    """Get the Pydantic model corresponding to the 'instrument.json' for a given session."""

    return aind_data_schema.core.instrument.Instrument(
        location=get_location(session.rig),
        instrument_id=session.rig,
        modification_date=session.session_start_time.date(),
        modalities=get_modalities(session),
        calibrations=[],
        coordinate_system=coordinate_system,
        temperature_control=None,
        notes=None,
        connections=[],
        components=get_components(session),
    )

if __name__ == "__main__":
    session = DynamicRoutingSession('814666_20251107')
    metadata = get_instrument_model(session)
    print(metadata.model_dump_json(indent=2))
    with open('instrument_814666_2025-11-07.json', 'w') as f:
        f.write(metadata.model_dump_json(indent=2))
