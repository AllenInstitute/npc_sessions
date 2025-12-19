import datetime
import logging

from aind_data_schema.components import coordinates, devices
from aind_data_schema.core import rig
from aind_data_schema_models import organizations
from np_aind_metadata import common, rigs

COPA_NOTES = (
    "The rotation matrix is represented as: a,b,c,d,e,f,g,h,i. Wherein a, b, "
    "c correspond to the first row of the matrix. The translation matrix is "
    "represented as: x,y,z."
)
DEFAULT_HOSTNAME = "127.0.0.1"


logger = logging.getLogger(__name__)


def init(
    rig_name: common.RigName,
    modification_date: datetime.date | None = None,
    manipulator_infos: list[common.ManipulatorInfo] = [],
    mon_computer_name: str = DEFAULT_HOSTNAME,
    stim_computer_name: str = DEFAULT_HOSTNAME,
    sync_computer_name: str = DEFAULT_HOSTNAME,
) -> rig.Rig:
    """Initializes a rig model for the dynamic routing project.

    >>> rig_model = init("NP3")

    Notes
    -----
    - rig_id is expected to be in the format:
        <ROOM NAME>_<RIG NAME>_<MODIFICATION DATE>
    - The DR task does not set the brightness and contrast of the monitor.
     These are hardcoded and assumed to be static.
    """
    if not modification_date:
        modification_date = datetime.date.today()

    room_name = rigs.get_rig_room(rig_name)
    rig_id = f"{rig_name}_{modification_date.strftime('%y%m%d')}"
    if room_name is not None:
        rig_id = f"{room_name}_{rig_id}"

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

    shared_camera_assembly_relative_position_props = {
        "device_origin": (
            "Located on the face of the lens mounting surface at its center, "
            "ie. just ahead of the camera sensor."
        ),
        "device_axes": [
            coordinates.Axis(
                name=coordinates.AxisName.X,
                direction=(
                    "Positive is from the center of the sensor towards the "
                    "left of the assembly, from the subject's perspective."
                ),
            ),
            coordinates.Axis(
                name=coordinates.AxisName.Y,
                direction=(
                    "Positive is from the center of the sensor towards the "
                    "bottom of the assembly."
                ),
            ),
            coordinates.Axis(
                name=coordinates.AxisName.Z,
                direction=("Positive is from the sensor towards the subject."),
            ),
        ],
        "notes": COPA_NOTES,
    }

    ephys_assemblies = []
    for assembly_letter in ["A", "B", "C", "D", "E", "F"]:
        serial_number = None
        for manipulator_info in manipulator_infos:
            if manipulator_info.assembly_name == f"Ephys Assembly {assembly_letter}":
                serial_number = manipulator_info.serial_number
                break
        ephys_assemblies.append(
            rig.EphysAssembly(
                name=f"Ephys Assembly {assembly_letter}",
                manipulator=devices.Manipulator(
                    name=f"Ephys Assembly {assembly_letter} Manipulator",
                    manufacturer=(organizations.Organization.NEW_SCALE_TECHNOLOGIES),
                    model="06591-M-0004",
                    serial_number=serial_number,
                ),
                probes=[
                    devices.EphysProbe(
                        name=f"Probe{assembly_letter}",
                        probe_model="Neuropixels 1.0",
                        manufacturer=organizations.Organization.IMEC,
                    )
                ],
            )
        )

    model = rig.Rig(
        rig_id=rig_id,
        modification_date=modification_date,
        modalities=[
            rig.Modality.BEHAVIOR_VIDEOS,
            rig.Modality.BEHAVIOR,
            rig.Modality.ECEPHYS,
        ],
        mouse_platform=devices.Disc(
            name="Mouse Platform",
            radius="4.69",
            radius_unit="centimeter",
            notes=(
                "Radius is the distance from the center of the wheel to the " "mouse."
            ),
        ),
        stimulus_devices=[
            devices.Monitor(
                name="Stim",
                model="PA248",
                manufacturer=organizations.Organization.ASUS,
                width=1920,
                height=1200,
                size_unit="pixel",
                viewing_distance=15.3,
                viewing_distance_unit="centimeter",
                refresh_rate=60,
                brightness=43,
                contrast=50,
                position=coordinates.RelativePosition(
                    device_position_transformations=[
                        coordinates.Rotation3dTransform(
                            rotation=[
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
                        ),
                        coordinates.Translation3dTransform(
                            translation=[0.08751, -0.12079, 0.02298]
                        ),
                    ],
                    device_origin=(
                        "Located at the center of the screen. Right and left "
                        "conventions are relative to the screen side of the "
                        "monitor, ie. from the subject's perspective."
                    ),
                    device_axes=[
                        coordinates.Axis(
                            name=coordinates.AxisName.X,
                            direction=(
                                "Positive is from the center towards the "
                                "right of the screen, from the subject's "
                                "perspective."
                            ),
                        ),
                        coordinates.Axis(
                            name=coordinates.AxisName.Y,
                            direction=(
                                "Positive is from the center towards the top "
                                "of the screen."
                            ),
                        ),
                        coordinates.Axis(
                            name=coordinates.AxisName.Z,
                            direction=(
                                "Positive is from the front of the screen "
                                "towards the subject."
                            ),
                        ),
                    ],
                    notes=COPA_NOTES,
                ),
            ),
            devices.Speaker(
                name="Speaker",
                manufacturer=organizations.Organization.ISL,
                model="SPK-I-81345",
                position=coordinates.RelativePosition(
                    device_position_transformations=[
                        coordinates.Rotation3dTransform(
                            rotation=[
                                -0.82783,
                                -0.4837,
                                -0.28412,
                                -0.55894,
                                0.75426,
                                0.34449,
                                0.04767,
                                0.44399,
                                -0.89476,
                            ],
                        ),
                        coordinates.Translation3dTransform(
                            translation=[-0.00838, -0.09787, 0.18228]
                        ),
                    ],
                    device_origin=(
                        "Located on the front mounting flange face. Right "
                        "and left conventions are relative to the front side "
                        "of the speaker, ie. from the subject's perspective."
                    ),
                    device_axes=[
                        coordinates.Axis(
                            name=coordinates.AxisName.X,
                            direction=(
                                "Positive is from the center of the speaker "
                                "towards the right of the speaker, from the "
                                "subject's perspective."
                            ),
                        ),
                        coordinates.Axis(
                            name=coordinates.AxisName.Y,
                            direction=(
                                "Positive is from the center towards the top "
                                "of the speaker."
                            ),
                        ),
                        coordinates.Axis(
                            name=coordinates.AxisName.Z,
                            direction=(
                                "Positive is from the speaker towards the " "subject."
                            ),
                        ),
                    ],
                    notes=COPA_NOTES
                    + (
                        " Speaker to be mounted with the X axis pointing to "
                        "the right when viewing the speaker along the Z axis"
                    ),
                ),
            ),
            devices.RewardDelivery(
                reward_spouts=[
                    devices.RewardSpout(
                        name="Reward Spout",
                        manufacturer=organizations.Organization.HAMILTON,
                        model="8649-01 Custom",
                        spout_diameter=0.672,
                        spout_diameter_unit="millimeter",
                        side=devices.SpoutSide.CENTER,
                        solenoid_valve=devices.Device(
                            name="Solenoid Valve",
                            device_type="Solenoid Valve",
                            manufacturer=organizations.Organization.NRESEARCH_INC,
                            model="161K011",
                            notes="Model number is product number.",
                        ),
                        lick_sensor=devices.Device(
                            name="Lick Sensor",
                            device_type="Lick Sensor",
                            manufacturer=organizations.Organization.OTHER,
                        ),
                        lick_sensor_type=devices.LickSensorType.PIEZOELECTIC,
                        notes=(
                            "Spout diameter is for inner diameter. "
                            "Outer diameter is 1.575mm. "
                        ),
                    ),
                ]
            ),
        ],
        ephys_assemblies=ephys_assemblies,
        light_sources=[
            devices.LightEmittingDiode(
                manufacturer=organizations.Organization.OTHER,
                name="Face forward LED",
                model="LZ4-40R308-0000",
                wavelength=740,
                wavelength_unit=devices.SizeUnit.NM,
            ),
            devices.LightEmittingDiode(
                manufacturer=organizations.Organization.OTHER,
                name="Body LED",
                model="LZ4-40R308-0000",
                wavelength=740,
                wavelength_unit=devices.SizeUnit.NM,
            ),
            devices.LightEmittingDiode(
                manufacturer=organizations.Organization.OSRAM,
                name="Eye LED",
                model="LZ4-40R608-0000",
                wavelength=850,
                wavelength_unit=devices.SizeUnit.NM,
            ),
            devices.Laser(
                name="Laser #0",
                manufacturer=organizations.Organization.VORTRAN,
                wavelength=488.0,
                model="Stradus 488-50",
                wavelength_unit="nanometer",
            ),
            devices.Laser(
                name="Laser #1",
                manufacturer=organizations.Organization.VORTRAN,
                wavelength=633.0,
                model="Stradus 633-80",
                wavelength_unit="nanometer",
            ),
        ],
        cameras=[
            devices.CameraAssembly(
                name="Front",
                camera_target="Face forward",
                camera=devices.Camera(
                    name="Front camera",
                    manufacturer=organizations.Organization.ALLIED,
                    chroma="Monochrome",
                    data_interface="Ethernet",
                    computer_name=mon_computer_name,
                    **shared_camera_props,
                ),
                filter=devices.Filter(
                    name="Front filter",
                    manufacturer=organizations.Organization.SEMROCK,
                    model="FF01-715_LP-25",
                    filter_type=devices.FilterType.LONGPASS,
                ),
                lens=devices.Lens(
                    name="Front lens",
                    manufacturer=organizations.Organization.EDMUND_OPTICS,
                    focal_length=8.5,
                    focal_length_unit="millimeter",
                    model="86604",
                ),
                position=coordinates.RelativePosition(
                    device_position_transformations=[
                        coordinates.Rotation3dTransform(
                            rotation=[
                                -0.17365,
                                0.98481,
                                0,
                                0.44709,
                                0.07883,
                                -0.89101,
                                -0.87747,
                                -0.15472,
                                -0.45399,
                            ]
                        ),
                        coordinates.Translation3dTransform(
                            translation=[0.154, 0.03078, 0.06346],
                        ),
                    ],
                    **shared_camera_assembly_relative_position_props,
                ),
            ),
            devices.CameraAssembly(
                name="Side",
                camera_target="Body",
                camera=devices.Camera(
                    name="Side camera",
                    manufacturer=organizations.Organization.ALLIED,
                    chroma="Monochrome",
                    data_interface="Ethernet",
                    computer_name=mon_computer_name,
                    **shared_camera_props,
                ),
                filter=devices.Filter(
                    name="Side filter",
                    manufacturer=organizations.Organization.SEMROCK,
                    model="FF01-747/33-25",
                    filter_type=devices.FilterType.BANDPASS,
                ),
                lens=devices.Lens(
                    name="Side lens",
                    manufacturer=organizations.Organization.NAVITAR,
                    focal_length=6.0,
                    focal_length_unit="millimeter",
                ),
                position=coordinates.RelativePosition(
                    device_position_transformations=[
                        coordinates.Rotation3dTransform(
                            rotation=[-1, 0, 0, 0, 0, -1, 0, -1, 0]
                        ),
                        coordinates.Translation3dTransform(
                            translation=[-0.03617, 0.23887, -0.02535],
                        ),
                    ],
                    **shared_camera_assembly_relative_position_props,
                ),
            ),
            devices.CameraAssembly(
                name="Eye",
                camera_target="Eye",
                camera=devices.Camera(
                    name="Eye camera",
                    manufacturer=organizations.Organization.ALLIED,
                    chroma="Monochrome",
                    data_interface="Ethernet",
                    computer_name=mon_computer_name,
                    **shared_camera_props,
                ),
                filter=devices.Filter(
                    name="Eye filter",
                    manufacturer=organizations.Organization.SEMROCK,
                    model="FF01-850/10-25",
                    filter_type=devices.FilterType.BANDPASS,
                ),
                lens=devices.Lens(
                    name="Eye lens",
                    manufacturer=(organizations.Organization.INFINITY_PHOTO_OPTICAL),
                    focal_length=6.0,
                    focal_length_unit="millimeter",
                    model="213073",
                    notes="Model number is SKU.",
                ),
                position=coordinates.RelativePosition(
                    device_position_transformations=[
                        coordinates.Rotation3dTransform(
                            rotation=[
                                -0.5,
                                -0.86603,
                                0,
                                -0.366,
                                0.21131,
                                -0.90631,
                                0.78489,
                                -0.45315,
                                -0.42262,
                            ]
                        ),
                        coordinates.Translation3dTransform(
                            translation=[-0.14259, 0.06209, 0.09576],
                        ),
                    ],
                    **shared_camera_assembly_relative_position_props,
                ),
            ),
        ],
        daqs=[
            devices.DAQDevice(
                manufacturer=organizations.Organization.NATIONAL_INSTRUMENTS,
                name="Sync",
                computer_name=sync_computer_name,
                model="NI-6612",
                data_interface=devices.DataInterface.PCIE,
            ),
            devices.DAQDevice(
                manufacturer=organizations.Organization.NATIONAL_INSTRUMENTS,
                name="Behavior",
                computer_name=stim_computer_name,
                model="NI-6323",
                data_interface=devices.DataInterface.USB,
            ),
            devices.DAQDevice(
                manufacturer=organizations.Organization.NATIONAL_INSTRUMENTS,
                name="BehaviorSync",
                computer_name=stim_computer_name,
                model="NI-6001",
                data_interface=devices.DataInterface.PCIE,
            ),
            devices.DAQDevice(
                manufacturer=organizations.Organization.NATIONAL_INSTRUMENTS,
                name="Opto",
                computer_name=stim_computer_name,
                model="NI-9264",
                data_interface=devices.DataInterface.ETH,
            ),
        ],
        detectors=[
            devices.Detector(
                name="vsync photodiode",
                model="PDA25K",
                manufacturer=organizations.Organization.THORLABS,
                data_interface=devices.DataInterface.OTHER,
                notes="Data interface is unknown.",
                detector_type=devices.DetectorType.OTHER,
                cooling=devices.Cooling.AIR,
            ),
        ],
        calibrations=[],
        additional_devices=[
            devices.Detector(
                name="microphone",
                manufacturer=organizations.Organization.DODOTRONIC,
                model="MOM",
                data_interface=devices.DataInterface.OTHER,
                notes="Data interface is unknown.",
                detector_type=devices.DetectorType.OTHER,
                cooling=devices.Cooling.AIR,
            ),
            devices.AdditionalImagingDevice(
                name="Galvo x",
                imaging_device_type=devices.ImagingDeviceType.GALVO,
            ),
            devices.AdditionalImagingDevice(
                name="Galvo y",
                imaging_device_type=devices.ImagingDeviceType.GALVO,
            ),
        ],
        rig_axes=[
            coordinates.Axis(
                name=coordinates.AxisName.X,
                direction=(
                    "The world horizontal. Lays on the Mouse Sagittal Plane. "
                    "Positive direction is towards the nose of the mouse. "
                ),
            ),
            coordinates.Axis(
                name=coordinates.AxisName.Y,
                direction=(
                    "Perpendicular to Y. Positive direction is "
                    "away from the nose of the mouse. "
                ),
            ),
            coordinates.Axis(
                name=coordinates.AxisName.Z,
                direction="Positive pointing up.",
            ),
        ],
        origin=coordinates.Origin.BREGMA,
        patch_cords=[
            devices.Patch(
                name="Patch Cord #1",
                manufacturer=organizations.Organization.THORLABS,
                model="SM450 Custom Length, FC/PC Ends",
                core_diameter=125.0,
                numerical_aperture=0.10,
                notes=("Numerical aperture is approximately between 0.10 and " "0.14."),
            ),
        ],
    )

    return rig.Rig.model_validate(model)


if __name__ == "__main__":
    from np_aind_metadata import testmod

    testmod()
