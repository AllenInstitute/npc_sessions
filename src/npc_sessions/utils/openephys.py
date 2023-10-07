"""Tools for working with Open Ephys raw data files."""
from __future__ import annotations

import doctest
import io
import json
import logging
import re
from collections.abc import Generator, Iterable, Sequence
from typing import Any, Literal, NamedTuple

import numpy as np
import numpy.typing as npt
import upath
import zarr

import npc_sessions.utils as utils

logger = logging.getLogger(__name__)

DEFAULT_PROBES = "ABCDEF"


def get_sync_messages_data(
    sync_messages_path: utils.PathLike,
) -> dict[str, dict[Literal["start", "rate"], int]]:
    """
    Start Time for Neuropix-PXI (107) - ProbeA-AP @ 30000 Hz: 210069564
    Start Time for Neuropix-PXI (107) - ProbeA-LFP @ 2500 Hz: 17505797
    Start Time for NI-DAQmx (109) - PXI-6133 @ 30000 Hz: 210265001

    >>> path = 's3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15/ecephys_clipped/Record Node 102/experiment1/recording1/sync_messages.txt'
    >>> dirname_to_sample = get_sync_messages_data(path)
    >>> dirname_to_sample['NI-DAQmx-105.PXI-6133']
    {'start': 257417001, 'rate': 30000}
    """

    def label(line) -> str:
        return "".join(
            line.split("Start Time for ")[-1]
            .split(" @")[0]
            .replace(") - ", ".")
            .replace(" (", "-")
        )

    def start(line) -> int:
        return int(line.strip(" ").split("Hz:")[-1])

    def rate(line) -> int:
        return int(line.split("@ ")[-1].split(" Hz")[0])

    return {
        label(line): {
            "start": start(line),
            "rate": rate(line),
        }
        for line in utils.from_pathlike(sync_messages_path).read_text().splitlines()[1:]
    }


class EphysTimingInfoOnPXI(NamedTuple):
    name: str
    continuous: upath.UPath
    """Abs path to device's folder within raw data/continuous/"""
    events: upath.UPath
    """Abs path to device's folder within raw data/events/"""
    ttl: upath.UPath
    """Abs path to device's folder within events/"""
    compressed: upath.UPath | None
    """Abs path to device's zarr storage within ecephys_compressed/, or None if not found"""
    start_sample: int
    """Start sample reported in sync_messages.txt"""
    sampling_rate: float
    """Nominal sample rate reported in sync_messages.txt"""
    ttl_sample_numbers: npt.NDArray
    """Sample numbers on open ephys clock, after subtracting first sample reported in
    sync_messages.txt"""
    ttl_states: npt.NDArray
    """Contents of ttl/states.npy"""


class EphysTimingInfoOnSync(NamedTuple):
    name: str
    device: EphysTimingInfoOnPXI
    """Info with paths"""
    sampling_rate: float
    """Sample rate assessed on the sync clock"""
    start_time: float
    """First sample time (sec) relative to the start of the sync clock"""


def read_array_range_from_npy(path: utils.PathLike, _range: int | slice) -> npt.NDArray:
    """Read specific range without downloading entire array. For 1-D array only, currently."""
    if not isinstance(_range, slice):
        _range = slice(_range, _range + 1)
    path = utils.from_pathlike(path)
    ver_major = int.from_bytes(path.fs.read_bytes(path, start=6, end=7), "little")
    header_len_stop = 10 if ver_major == 1 else 12
    header_len = int.from_bytes(
        path.fs.read_bytes(path, start=8, end=header_len_stop), "little"
    )
    array_start = header_len_stop + header_len
    header_bytes = path.fs.read_bytes(path, start=header_len_stop, end=array_start)
    header = eval(header_bytes.decode("utf-8").strip("\n").strip())
    assert len(header["shape"]) == 1, "Currently supporting 1-D array only"
    dtype = header["descr"]
    num_bytes_per_value = np.dtype(dtype).itemsize
    return np.frombuffer(
        path.fs.read_bytes(
            path,
            start=array_start + _range.start * num_bytes_per_value,
            end=array_start + _range.stop * num_bytes_per_value,
        ),
        dtype=dtype,
    )


def get_ephys_timing_on_pxi(
    recording_dirs: Iterable[utils.PathLike],
    only_devices_including: str | None = None,
) -> Generator[EphysTimingInfoOnPXI, None, None]:
    """
    >>> path = upath.UPath('s3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15/ecephys_clipped/Record Node 102/experiment1/recording1')
    >>> next(get_ephys_timing_on_pxi(path)).sampling_rate
    30000
    """
    if not isinstance(recording_dirs, Iterable):
        recording_dirs = (recording_dirs,)

    for recording_dir in recording_dirs:
        recording_dir = utils.from_pathlike(recording_dir)
        device_to_sync_messages_data = get_sync_messages_data(
            recording_dir / "sync_messages.txt"
        )  # includes name of each input device used (probe, nidaq)
        for device in device_to_sync_messages_data:
            if (
                only_devices_including
                and only_devices_including.lower() not in device.lower()
            ):
                continue
            continuous = recording_dir / "continuous" / device
            if not continuous.exists():
                continue
            events = recording_dir / "events" / device
            ttl = next(events.glob("TTL*"))

            first_sample_from_continuous_sample_numbers = read_array_range_from_npy(
                continuous / "sample_numbers.npy", 0
            ).item()
            first_sample_from_sync_messages = device_to_sync_messages_data[device][
                "start"
            ]
            if (
                first_sample_from_continuous_sample_numbers
                != first_sample_from_sync_messages
            ):
                logger.debug(
                    f"{first_sample_from_sync_messages =} != {first_sample_from_continuous_sample_numbers =}. This may be due to Record Nodes being out-of-sync (green indicator in GUI). Using value from sample_numbers.npy"
                )
            first_sample_on_ephys_clock = first_sample_from_continuous_sample_numbers

            sampling_rate = device_to_sync_messages_data[device]["rate"]
            ttl_sample_numbers = (
                np.load(io.BytesIO((ttl / "sample_numbers.npy").read_bytes()))
                - first_sample_on_ephys_clock
            )
            ttl_states = np.load(io.BytesIO((ttl / "states.npy").read_bytes()))
            try:
                compressed = clipped_path_to_compressed(continuous)
            except ValueError:
                logger.info(f"No compressed data found for {continuous}")
                compressed = None
            yield EphysTimingInfoOnPXI(
                name=device,
                continuous=continuous,
                events=events,
                ttl=ttl,
                compressed=compressed,
                start_sample=first_sample_on_ephys_clock,
                sampling_rate=sampling_rate,
                ttl_sample_numbers=ttl_sample_numbers,
                ttl_states=ttl_states,
            )


def clipped_path_to_compressed(path: utils.PathLike) -> upath.UPath:
    """
    >>> path = upath.UPath('s3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15/ecephys_clipped/Record Node 102/experiment1/recording1/continuous/Neuropix-PXI-100.ProbeA-AP')
    >>> clipped_path_to_compressed(path)
    S3Path('s3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15/ecephys_compressed/experiment1_Record Node 102#Neuropix-PXI-100.ProbeA-AP.zarr')
    """
    path = utils.from_pathlike(path)
    if "ecephys_clipped" not in path.as_posix():
        raise ValueError(f'Expected path to contain "ecephys_clipped", got {path}')
    experiment_re = re.search(r".*(experiment\d+)", path.as_posix())
    record_node_re = re.search(r".*(Record Node \d+)", path.as_posix())
    # /recording?/ isn't part of compressed path: assumes 1 recording per folder, or concats multiple recordings
    if not (experiment_re and record_node_re):
        raise ValueError(f"Could not parse experiment and record node from {path}")
    experiment, record_node = experiment_re.groups()[0], record_node_re.groups()[0]
    device_re = re.match(
        rf".*{record_node}/{experiment}/recording\d+/[^/]+/(.*)", path.as_posix()
    )
    if not device_re:
        raise ValueError(f"Could not parse device from {path}")
    compressed_name = f"{experiment}_{record_node}#{device_re.groups()[0]}.zarr"
    root_path = next(p for p in path.parents if p.name == "ecephys_clipped")
    # cannot construct S3Path de novo from a string including `#`, but we can return
    # the actual path that exists
    return next(
        path
        for path in root_path.with_name("ecephys_compressed").iterdir()
        if path.name == compressed_name
    )


def get_pxi_nidaq_data(
    *recording_dirs: utils.PathLike,
    device_name: str | None = None,
) -> npt.NDArray[np.int16]:
    """
    -channel_idx: 0-indexed
    - if device_name not specified, first and only (assumed) NI-DAQ will be used

    ```
    speaker_channel, mic_channel = 1, 3
    ```

    >>> path = upath.UPath('s3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15/ecephys_clipped/Record Node 102/experiment1/recording1')
    >>> get_pxi_nidaq_data(path).shape
    (142823472, 8)
    """
    if device_name:
        device = next(
            get_ephys_timing_on_pxi(recording_dirs, only_devices_including=device_name)
        )
    else:
        device = get_pxi_nidaq_device(recording_dirs)
    if device.compressed:
        data = zarr.open(device.compressed, mode="r")
        return data["traces_seg0"]
    else:
        device_metadata = next(
            (
                _
                for _ in get_merged_oebin_file(
                    next(utils.from_pathlike(p).glob("*.oebin")) for p in recording_dirs
                )["continuous"]
                if device.name in _["folder_name"]
            ),
            None,
        )
        if device_metadata is None:
            raise ValueError(
                f"Could not find device metadata for {device.name}: looked for `structure.oebin` files in {recording_dirs}"
            )
        num_channels: int = device_metadata["num_channels"]

        if not device.continuous.as_uri().startswith("file"):
            # local file we can memory-map
            dat = np.load(device.continuous / "continuous.dat", mmap_mode="r")
        else:
            logger.warning(
                f"Reading entirety of uncompressed OpenEphys NI-DAQ data from {device.continuous}. If you only need part of this data, consider using `read_array_range_from_npy` with the path instead."
            )
            dat = np.frombuffer(
                (device.continuous / "continuous.dat").read_bytes(), dtype=np.int16
            )
        return np.reshape(dat, (int(dat.size / num_channels), -1))


def get_pxi_nidaq_device(
    recording_dir: Iterable[utils.PathLike],
) -> EphysTimingInfoOnPXI:
    """NI-DAQmx device info

    >>> path = upath.UPath('s3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15/ecephys_clipped/Record Node 102/experiment1/recording1')
    >>> get_pxi_nidaq_device(path).ttl.parent.name
    'NI-DAQmx-105.PXI-6133'
    """
    device = tuple(
        get_ephys_timing_on_pxi(recording_dir, only_devices_including="NI-DAQmx-")
    )
    if not device:
        raise FileNotFoundError(
            f"No */continuous/NI-DAQmx-*/ dir found in {recording_dir = }"
        )
    if device and len(device) != 1:
        raise FileNotFoundError(
            f"Expected a single NI-DAQmx folder to exist, but found: {[d.continuous for d in device]}"
        )
    return device[0]


def get_ephys_timing_on_sync(
    sync: utils.SyncPathOrDataset,
    recording_dirs: Iterable[utils.PathLike] | None = None,
    devices: Iterable[EphysTimingInfoOnPXI] | None = None,
    only_devices_including: str | None = None,
) -> Generator[EphysTimingInfoOnSync, None, None]:
    """
    One of `recording_dir` or `devices` must be supplied:
        - all devices in `recording_dir` will be returned
        - or just those specified in `devices`
        (use `get_ephys_timing_on_pxi()` to get a filtered iterable of devices
        in a recording_dir)

    >>> path = upath.UPath('s3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15/ecephys_clipped/Record Node 102/experiment1/recording1')
    >>> sync = upath.UPath('s3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15/behavior/20230803T120415.h5')
    >>> device = next(get_ephys_timing_on_sync(sync, path))
    >>> device.sampling_rate, device.start_time
    (30000.070518634246, 20.080209634424037)
    """
    if not (recording_dirs or devices):
        raise ValueError("Must specify recording_dir or devices")

    sync = utils.get_sync_data(sync)

    sync_barcode_times, sync_barcode_ids = utils.extract_barcodes_from_times(
        on_times=sync.get_rising_edges("barcode_ephys", units="seconds"),
        off_times=sync.get_falling_edges("barcode_ephys", units="seconds"),
    )
    if devices and not isinstance(devices, Iterable):
        devices = (devices,)

    if recording_dirs and not isinstance(recording_dirs, Iterable):
        recording_dirs = (recording_dirs,)

    if recording_dirs and not devices:
        devices = get_ephys_timing_on_pxi(recording_dirs, only_devices_including)

    assert devices is not None
    for device in devices:
        (
            ephys_barcode_times,
            ephys_barcode_ids,
        ) = utils.extract_barcodes_from_times(
            on_times=device.ttl_sample_numbers[device.ttl_states > 0]
            / device.sampling_rate,
            off_times=device.ttl_sample_numbers[device.ttl_states < 0]
            / device.sampling_rate,
        )

        timeshift, sampling_rate, _ = utils.get_probe_time_offset(
            master_times=sync_barcode_times,
            master_barcodes=sync_barcode_ids,
            probe_times=ephys_barcode_times,
            probe_barcodes=ephys_barcode_ids,
            acq_start_index=0,
            local_probe_rate=device.sampling_rate,
        )
        start_time = -timeshift
        if (np.isnan(sampling_rate)) | (~np.isfinite(sampling_rate)):
            sampling_rate = device.sampling_rate

        yield EphysTimingInfoOnSync(
            name=device.name,
            device=device,
            sampling_rate=sampling_rate,
            start_time=start_time,
        )


def is_new_ephys_folder(path: utils.PathLike) -> bool:
    """Look for any hallmarks of a v0.6.x Open Ephys recording in path or subfolders."""
    path = utils.from_pathlike(path)
    globs = (
        "Record Node*",
        "structure*.oebin",
    )
    components = tuple(_.replace("*", "") for _ in globs)

    if any(_.lower() in path.as_posix().lower() for _ in components):
        return True

    for glob in globs:
        if next(path.rglob(glob), None):
            return True
    return False


def is_complete_ephys_folder(path: utils.PathLike) -> bool:
    """Look for all hallmarks of a complete v0.6.x Open Ephys recording."""
    # TODO use structure.oebin to check for completeness
    path = utils.from_pathlike(path)
    if not is_new_ephys_folder(path):
        return False
    for glob in ("continuous.dat", "spike_times.npy", "spike_clusters.npy"):
        if not next(path.rglob(glob), None):
            logger.debug(f"Could not find {glob} in {path}")
            return False
    return True


def is_valid_ephys_folder(
    path: utils.PathLike,
    min_size_gb: int | float | None = None,
) -> bool:
    """Check a single dir of raw data for size and v0.6.x+ Open Ephys."""
    path = utils.from_pathlike(path)
    if not path.is_dir():
        return False
    if not is_new_ephys_folder(path):
        return False
    if min_size_gb is not None and utils.dir_size(path) < min_size_gb * 1024**3:  # GB
        return False
    return True


def get_ephys_root(path: utils.PathLike) -> upath.UPath:
    """Returns the parent of the first `Record Node *` found in the supplied
    path.

    >>> get_ephys_root(upath.UPath('A:/test/Record Node 0/Record Node test')).as_posix()
    'A:/test'
    """
    path = utils.from_pathlike(path)
    if "Record Node" not in path.as_posix():
        raise ValueError(
            f"Could not find 'Record Node' in {path} - is this a valid raw ephys path?"
        )
    return next(
        p.parent for p in path.parents if "Record Node".lower() in p.name.lower()
    )


def get_filtered_ephys_paths_relative_to_record_node_parents(
    toplevel_ephys_path: utils.PathLike,
) -> Generator[tuple[upath.UPath, upath.UPath], None, None]:
    """For restructuring the raw ephys data in a session folder, we want to
    discard superfluous recording folders and only keep the "good" data, but
    with the overall directory structure relative to `Record Node*` folders intact.

    Supply a top-level path that contains `Record Node *`
    subfolders somewhere in its tree.

    Returns a generator akin to `path.rglob('Record Node*')` except:
    - only paths associated with the "good" ephys data are returned (with some
    assumptions made about the ephys session)
    - a tuple of two paths is supplied:
        - `(abs_path, abs_path.relative_to(record_node.parent))`
        ie. path[1] should always start with `Record Node *`

    Expectation is:
    - `/npexp_path/ephys_*/Record Node */ recording1 / ... / continuous.dat`

    ie.
    - one recording per `Record Node *` folder
    - one subfolder in `npexp_path/` per `Record Node *` folder (following the
    pipeline `*_probeABC` / `*_probeDEF` convention for recordings split across two
    drives)


    Many folders have:
    - `/npexp_path/Record Node */ recording*/ ...`

    ie.
    - multiple recording folders per `Record Node *` folder
    - typically there's one proper `recording` folder: the rest are short,
    aborted recordings during setup

    We filter out the superfluous small recording folders here.

    Some folders (Templeton) have:
    - `/npexp_path/Record Node */ ...`

    ie.
    - contents of expected ephys subfolders directly deposited in npexp_path

    """
    toplevel_ephys_path = utils.from_pathlike(toplevel_ephys_path)
    record_nodes = toplevel_ephys_path.rglob("Record Node*")

    for record_node in record_nodes:
        superfluous_recording_dirs = tuple(
            _.parent for _ in get_superfluous_oebin_paths(record_node)
        )
        logger.debug(
            f"Found {len(superfluous_recording_dirs)} superfluous recording dirs to exclude: {superfluous_recording_dirs}"
        )

        for abs_path in record_node.rglob("*"):
            is_superfluous_path = any(
                _ in abs_path.parents for _ in superfluous_recording_dirs
            )

            if is_superfluous_path:
                continue

            yield abs_path, abs_path.relative_to(record_node.parent)


def get_raw_ephys_subfolders(
    path: utils.PathLike, min_size_gb: int | float | None = None
) -> tuple[upath.UPath, ...]:
    """
    Return raw ephys recording folders, defined as the root that Open Ephys
    records to, e.g. `A:/1233245678_366122_20220618_probeABC`.
    """
    path = utils.from_pathlike(path)

    subfolders = set()

    for f in upath.UPath(path).rglob("continuous.dat"):
        if any(
            k in f.as_posix().lower()
            for k in [
                "sorted",
                "extracted",
                "curated",
            ]
        ):
            # skip sorted/extracted folders
            continue

        subfolders.add(get_ephys_root(f))

    if min_size_gb is not None:
        subfolders = {
            _ for _ in subfolders if utils.dir_size(_) < min_size_gb * 1024**3
        }

    return tuple(sorted(subfolders, key=lambda s: str(s)))


# - If we have probeABC and probeDEF raw data folders, each one has an oebin file:
#     we'll need to merge the oebin files and the data folders to create a single session
#     that can be processed in parallel
def get_single_oebin_path(path: utils.PathLike) -> upath.UPath:
    """Get the path to a single structure.oebin file in a folder of raw ephys data.

    - There's one structure.oebin per `recording*` folder
    - Raw data folders may contain multiple `recording*` folders
    - Datajoint expects only one structure.oebin file per Session for sorting
    - If we have multiple `recording*` folders, we assume that there's one
        good folder - the largest - plus some small dummy / accidental recordings
    """
    path = utils.from_pathlike(path)
    if not path.is_dir():
        raise ValueError(f"{path} is not a directory")

    oebin_paths = tuple(path.rglob("structure*.oebin"))

    if not oebin_paths:
        raise FileNotFoundError(f"No structure.oebin file found in {path}")

    if len(oebin_paths) == 1:
        return oebin_paths[0]

    oebin_parents = (_.parent for _ in oebin_paths)
    dir_sizes = tuple(utils.dir_size(_) for _ in oebin_parents)
    return oebin_paths[dir_sizes.index(max(dir_sizes))]


def get_superfluous_oebin_paths(path: utils.PathLike) -> tuple[upath.UPath, ...]:
    """Get the paths to any oebin files in `recording*` folders that are not
    the largest in a folder of raw ephys data.

    Companion to `get_single_oebin_path`.
    """
    path = utils.from_pathlike(path)
    all_oebin_paths = tuple(path.rglob("structure*.oebin"))

    if len(all_oebin_paths) == 1:
        return ()

    return tuple(set(all_oebin_paths) - {(get_single_oebin_path(path))})


def assert_xml_files_match(*path: utils.PathLike) -> None:
    """Check that all xml files are identical, as they should be for
    recordings split across multiple locations e.g. A:/*_probeABC, B:/*_probeDEF
    or raise an error.

    Update: xml files on two nodes can be created at slightly different times, so their `date`
    fields may differ. Everything else should be identical.
    """
    paths = tuple(utils.from_pathlike(p) for p in path)
    if not all(s == ".xml" for s in [p.suffix for p in paths]):
        raise ValueError(f"Not all paths are XML files: {paths}")
    if not all(p.is_file() for p in paths):
        raise FileNotFoundError(
            f"Not all paths are files, or they do not exist: {paths}"
        )
    if not utils.checksums_match(*paths):
        data = [utils.get_settings_xml_data(p) for p in paths]
        if not all(d == data[0] for d in data):
            raise AssertionError(f"XML files do not match: {paths}")


def get_merged_oebin_file(
    paths: Iterable[utils.PathLike], exclude_probe_names: Sequence[str] | None = None
) -> dict[Literal["continuous", "events", "spikes"], list[dict[str, Any]]]:
    """Merge two or more structure.oebin files into one.

    For recordings split across multiple locations e.g. A:/*_probeABC,
    B:/*_probeDEF
    - if items in the oebin files have 'folder_name' values that match any
    entries in `exclude_probe_names`, they will be removed from the merged oebin
    """
    if not isinstance(paths, Iterable):
        paths = tuple(paths)
    oebin_paths = tuple(utils.from_pathlike(p) for p in paths)
    if len(oebin_paths) == 1:
        return read_oebin(oebin_paths[0])

    # ensure oebin files can be merged - if from the same exp they will have the same settings.xml file
    if any(p.suffix != ".oebin" for p in oebin_paths):
        raise ValueError(f"Not all paths are .oebin files: {oebin_paths}")
    assert_xml_files_match(
        *[p / "settings.xml" for p in [o.parent.parent.parent for o in oebin_paths]]
    )

    logger.debug(f"Creating merged oebin file from {oebin_paths}")
    merged_oebin: dict = {}
    for oebin_path in sorted(oebin_paths):
        oebin_data = read_oebin(oebin_path)

        for key in oebin_data:
            # skip if already in merged oebin
            if merged_oebin.get(key) == oebin_data[key]:
                continue

            # 'continuous', 'events', 'spikes' are lists, which we want to concatenate across files
            if isinstance(oebin_data[key], list):
                for item in oebin_data[key]:
                    # skip if already in merged oebin
                    if item in merged_oebin.get(key, []):
                        continue

                    # skip probes in excl list (ie. not inserted)
                    if exclude_probe_names and any(
                        p.lower() in item.get("folder_name", "").lower()
                        for p in exclude_probe_names
                    ):
                        continue

                    # insert in merged oebin
                    merged_oebin.setdefault(key, []).append(item)

    if not merged_oebin:
        raise ValueError("No data found in structure.oebin files")
    return merged_oebin


def read_oebin(
    path: utils.PathLike,
) -> dict[Literal["continuous", "events", "spikes"], list[dict[str, Any]]]:
    return json.loads(utils.from_pathlike(path).read_text())


if __name__ == "__main__":
    import doctest

    import dotenv

    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
