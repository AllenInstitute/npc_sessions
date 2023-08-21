"""Tools for working with Open Ephys raw data files."""
from __future__ import annotations

import doctest
import io
import json
import pathlib
import tempfile
from typing import Any, Generator, Literal, NamedTuple, Optional, Sequence, Iterable
import logging
import warnings

import numpy as np
import numpy.typing as npt
import upath

import npc_sessions.tools.file_io as file_io
import npc_sessions.tools.sync_dataset as sync_dataset
import npc_sessions.tools.ephys_utils as ephys_utils


logger = logging.getLogger(__name__)

DEFAULT_PROBES = 'ABCDEF'


def get_ephys_timing_info(sync_messages_path: str | pathlib.Path | upath.UPath) -> dict[str, dict[Literal['start', 'rate'], int]]:
    """
    Start Time for Neuropix-PXI (107) - ProbeA-AP @ 30000 Hz: 210069564
    Start Time for Neuropix-PXI (107) - ProbeA-LFP @ 2500 Hz: 17505797
    Start Time for NI-DAQmx (109) - PXI-6133 @ 30000 Hz: 210265001
    
    >>> path = 's3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15/ecephys_clipped/Record Node 102/experiment1/recording1/sync_messages.txt'
    >>> dirname_to_sample = get_ephys_timing_info(path)
    >>> dirname_to_sample['NI-DAQmx-105.PXI-6133']
    {'start': 257417001, 'rate': 30000}
    """
    label = lambda line: ''.join(line.split('Start Time for ')[-1].split(' @')[0].replace(') - ', '.').replace(' (', '-'))
    sample = lambda line: int(line.strip(' ').split('Hz:')[-1])
    rate = lambda line: int(line.split('@ ')[-1].split(' Hz')[0])
    
    return {
        label(line): {
            'start': sample(line),
            'rate': rate(line),
        }
        for line in upath.UPath(sync_messages_path).read_text().splitlines()[1:]
    } 
    
class EphysDevice(NamedTuple):
    continuous: upath.UPath
    """Abs path to device's folder within raw data/continuous/"""
    events: upath.UPath
    """Abs path to device's folder within raw data/events/"""
    ttl: upath.UPath
    """Abs path to device's folder within events/"""
    sampling_rate: float
    """Nominal sample rate reported in sync_messages.txt"""
    ttl_sample_numbers: npt.NDArray
    """Sample numbers on open ephys clock, after subtracting first sample reported in
    sync_messages.txt"""
    ttl_states: npt.NDArray
    """Contents of ttl/states.npy"""

class EphysTimingOnSync(NamedTuple):
    device: EphysDevice
    """Info with paths"""
    sampling_rate: float
    """Sample rate assessed on the sync clock"""
    start_time: float
    """First sample time (sec) relative to the start of the sync clock"""

def get_ephys_timing_on_pxi(recording_dir: upath.UPath, only_dirs_including: str = '') -> Generator[EphysDevice, None, None]:
    """
    >>> path = upath.UPath('s3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15/ecephys_clipped/Record Node 102/experiment1/recording1')
    >>> next(get_ephys_timing_on_pxi(path)).sampling_rate
    30000
    """
    device_to_first_sample_number = get_ephys_timing_info(recording_dir / 'sync_messages.txt') # includes name of each input device used (probe, nidaq)
    for device in device_to_first_sample_number:
        if only_dirs_including not in device:
            continue
        continuous = recording_dir / 'continuous' / device
        if not continuous.exists():
            continue
        events = recording_dir / 'events' / device
        ttl = next(events.glob('TTL*'))
        if 'ephys_clipped' in continuous.as_posix():
            warnings.warn(f'ephys_clipped folder detected: must update paths to read compressed zarr')
        first_sample_on_ephys_clock = device_to_first_sample_number[device]['start']
        sampling_rate = device_to_first_sample_number[device]['rate']
        ttl_sample_numbers = np.load(io.BytesIO((ttl / 'sample_numbers.npy').read_bytes())) - first_sample_on_ephys_clock
        ttl_states = np.load(io.BytesIO((ttl / 'states.npy').read_bytes()))
        yield EphysDevice(
            continuous, events, ttl, sampling_rate, ttl_sample_numbers, ttl_states
        )
                
def get_pxi_nidaq_data(
    recording_dir: upath.UPath,
    ) -> npt.NDArray[np.int16 | np.float64]:
    """
    >>> path = upath.UPath('s3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15/ecephys_clipped/Record Node 102/experiment1/recording1')
    >>> get_pxi_nidaq_data(path).shape
    (8, 25)
    """
    device = get_pxi_nidaq_device(recording_dir)
    if device.continuous.name.endswith('6133'):
        num_channels = 8
        speaker_channel, mic_channel = 1, 3
    else:
        raise IndexError(f'Unknown channel configuration for {device.continuous.name = }')
    dat = np.frombuffer((((device.continuous / 'continuous.dat').read_bytes())))
    data = np.reshape(dat, (int(dat.size / num_channels), -1)).T
    return data
    
    
def get_pxi_nidaq_device(recording_dir: upath.UPath) -> EphysDevice:
    """NI-DAQmx device info

    >>> path = upath.UPath('s3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15/ecephys_clipped/Record Node 102/experiment1/recording1')
    >>> get_pxi_nidaq_device(path).ttl.parent.name
    'NI-DAQmx-105.PXI-6133'
    """
    device = tuple(get_ephys_timing_on_pxi(recording_dir, only_dirs_including='NI-DAQmx-'))
    if not device:
        raise FileNotFoundError(f'No */continuous/NI-DAQmx-*/ dir found in {recording_dir = }')
    if device and len(device) != 1:
        raise FileNotFoundError(f'Expected a single NI-DAQmx folder to exist, but found: {[d.continuous for d in device]}')
    return device[0]
    
def get_ephys_timing_on_sync(
    sync_file: upath.UPath, 
    recording_dir: Optional[upath.UPath] = None,
    devices: Optional[EphysDevice | Iterable[EphysDevice]] = None
) -> Generator[EphysTimingOnSync, None, None]:
    """
    >>> path = upath.UPath('s3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15/ecephys_clipped/Record Node 102/experiment1/recording1')
    >>> sync = upath.UPath('s3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15/behavior/20230803T120415.h5')
    >>> device = next(get_ephys_timing_on_sync(sync, path))
    >>> device.sampling_rate, device.start_time
    (30000.07066388302, 0.7318752524919077)
    """
    if not (recording_dir or devices):
        raise ValueError('Must specify recording_dir or devices')
    
    sync = sync_dataset.SyncDataset(sync_file)
    
    sync_barcode_times, sync_barcode_ids = ephys_utils.extract_barcodes_from_times(
        on_times=sync.get_rising_edges('barcode_ephys', units='seconds'),
        off_times=sync.get_falling_edges('barcode_ephys', units='seconds'),
    )
    if isinstance(devices, EphysDevice):
        devices = (devices,)
    
    if recording_dir and not devices:
        devices = get_ephys_timing_on_pxi(recording_dir)
        
    assert devices  
    for device in devices:
        
        ephys_barcode_times, ephys_barcode_ids = ephys_utils.extract_barcodes_from_times(
            on_times=device.ttl_sample_numbers[device.ttl_states > 0] / device.sampling_rate,
            off_times=device.ttl_sample_numbers[device.ttl_states < 0] / device.sampling_rate,
            )
        
        timeshift, sampling_rate, _ = ephys_utils.get_probe_time_offset(
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
            
        yield EphysTimingOnSync(
            device, sampling_rate, start_time
        )
    

def is_new_ephys_folder(path: upath.UPath) -> bool:
    """Look for any hallmarks of a v0.6.x Open Ephys recording in path or subfolders."""

    globs = (
        'Record Node*',
        'structure*.oebin',
    )
    components = tuple(_.replace('*', '') for _ in globs)

    if any(_.lower() in path.as_posix().lower() for _ in components):
        return True

    for glob in globs:
        if next(path.rglob(glob), None):
            return True
    return False


def is_complete_ephys_folder(path: upath.UPath) -> bool:
    """Look for all hallmarks of a complete v0.6.x Open Ephys recording."""
    # TODO use structure.oebin to check for completeness
    if not is_new_ephys_folder(path):
        return False
    for glob in ('continuous.dat', 'spike_times.npy', 'spike_clusters.npy'):
        if not next(path.rglob(glob), None):
            logger.debug(f'Could not find {glob} in {path}')
            return False
    return True


def is_valid_ephys_folder(
    path: upath.UPath, min_size_gb: Optional[int | float] = None,
) -> bool:
    """Check a single dir of raw data for size and v0.6.x+ Open Ephys."""
    if not path.is_dir():
        return False
    if not is_new_ephys_folder(path):
        return False
    if min_size_gb is not None and file_io.dir_size(path) < min_size_gb * 1024**3: # GB
        return False
    return True


def get_ephys_root(path: upath.UPath) -> upath.UPath:
    """Returns the parent of the first `Record Node *` found in the supplied
    path.

    >>> get_ephys_root(upath.UPath('A:/test/Record Node 0/Record Node test')).as_posix()
    'A:/test'
    """
    if 'Record Node' not in path.as_posix():
        raise ValueError(
            f"Could not find 'Record Node' in {path} - is this a valid raw ephys path?"
        )
    return next(
        p.parent
        for p in path.parents
        if 'Record Node'.lower() in p.name.lower()
    )



def get_filtered_ephys_paths_relative_to_record_node_parents(
    toplevel_ephys_path: upath.UPath
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
    record_nodes = toplevel_ephys_path.rglob('Record Node*')
    
    for record_node in record_nodes:
        
        superfluous_recording_dirs = tuple(
            _.parent for _ in get_superfluous_oebin_paths(record_node)
        )
        logger.debug(f'Found {len(superfluous_recording_dirs)} superfluous recording dirs to exclude: {superfluous_recording_dirs}')
        
        for abs_path in record_node.rglob('*'):
            is_superfluous_path = any(_ in abs_path.parents for _ in superfluous_recording_dirs)
            
            if is_superfluous_path:
                continue
            
            yield abs_path, abs_path.relative_to(record_node.parent)
       
       
def get_raw_ephys_subfolders(
    path: upath.UPath, min_size_gb: Optional[int | float] = None
) -> tuple[upath.UPath, ...]:
    """
    Return raw ephys recording folders, defined as the root that Open Ephys
    records to, e.g. `A:/1233245678_366122_20220618_probeABC`.
    """ 

    subfolders = set()

    for f in upath.UPath(path).rglob('continuous.dat'):

        if any(
            k in f.as_posix().lower()
            for k in [
                'sorted',
                'extracted',
                'curated',
            ]
        ):
            # skip sorted/extracted folders
            continue

        subfolders.add(get_ephys_root(f))

    if min_size_gb is not None:
        subfolders = {_ for _ in subfolders if file_io.dir_size(_) < min_size_gb * 1024**3}

    return tuple(sorted(list(subfolders), key=lambda s: str(s)))


# - If we have probeABC and probeDEF raw data folders, each one has an oebin file:
#     we'll need to merge the oebin files and the data folders to create a single session
#     that can be processed in parallel
def get_single_oebin_path(path: upath.UPath) -> upath.UPath:
    """Get the path to a single structure.oebin file in a folder of raw ephys data.

    - There's one structure.oebin per `recording*` folder
    - Raw data folders may contain multiple `recording*` folders
    - Datajoint expects only one structure.oebin file per Session for sorting
    - If we have multiple `recording*` folders, we assume that there's one
        good folder - the largest - plus some small dummy / accidental recordings
    """
    if not path.is_dir():
        raise ValueError(f'{path} is not a directory')

    oebin_paths = tuple(path.rglob('structure*.oebin'))
        
    if not oebin_paths:
        raise FileNotFoundError(f'No structure.oebin file found in {path}')

    if len(oebin_paths) == 1:
        return oebin_paths[0]
    
    oebin_parents = (_.parent for _ in oebin_paths)
    dir_sizes = tuple(file_io.dir_size(_) for _ in oebin_parents)
    return oebin_paths[dir_sizes.index(max(dir_sizes))]


def get_superfluous_oebin_paths(path: upath.UPath) -> tuple[upath.UPath, ...]:
    """Get the paths to any oebin files in `recording*` folders that are not
    the largest in a folder of raw ephys data. 
    
    Companion to `get_single_oebin_path`.
    """
    
    all_oebin_paths = tuple(path.rglob('structure*.oebin'))
    
    if len(all_oebin_paths) == 1:
        return tuple()
    
    return tuple(set(all_oebin_paths) - {(get_single_oebin_path(path))})


def assert_xml_files_match(*paths: upath.UPath) -> None:
    """Check that all xml files are identical, as they should be for
    recordings split across multiple locations e.g. A:/*_probeABC, B:/*_probeDEF
    or raise an error.
    
    Update: xml files on two nodes can be created at slightly different times, so their `date`
    fields may differ. Everything else should be identical.
    """
    if not all(s == '.xml' for s in [p.suffix for p in paths]):
        raise ValueError('Not all paths are XML files')
    if not all(p.is_file() for p in paths):
        raise FileNotFoundError(
            'Not all paths are files, or they do not exist'
        )
    if not file_io.checksums_match(*paths):
        
        # if the files are the same size and were created within +/- 1 second
        # of each other, we'll assume they're the same
        
        created_times = tuple(file_io.ctime(p) for p in paths)
        created_times_equal = all(created_times[0] - 1 <= t <= created_times[0] + 1 for t in created_times[1:])
        
        sizes = tuple(file_io.file_size(p) for p in paths)
        sizes_equal = all(s == sizes[0] for s in sizes[1:])
        
        if not (sizes_equal and created_times_equal):
            raise AssertionError('XML files do not match')


def get_merged_oebin_file(
    paths: Sequence[upath.UPath], exclude_probe_names: Optional[Sequence[str]] = None
) -> upath.UPath:
    """Merge two or more structure.oebin files into one.

    For recordings split across multiple locations e.g. A:/*_probeABC,
    B:/*_probeDEF
    - if items in the oebin files have 'folder_name' values that match any
    entries in `exclude_probe_names`, they will be removed from the merged oebin
    """
    if isinstance(paths, upath.UPath):
        return paths
    if any(not p.suffix == '.oebin' for p in paths):
        raise ValueError('Not all paths are .oebin files')
    if len(paths) == 1:
        return paths[0]

    # ensure oebin files can be merged - if from the same exp they will have the same settings.xml file
    assert_xml_files_match(
        *[p / 'settings.xml' for p in [o.parent.parent.parent for o in paths]]
    )

    logger.debug(f'Creating merged oebin file from {paths}')
    merged_oebin: dict = {}
    for oebin_path in sorted(paths):
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
                        p.lower() in item.get('folder_name', '').lower()
                        for p in exclude_probe_names
                    ):
                        continue
                    
                    # insert in merged oebin
                    merged_oebin.setdefault(key, []).append(item)

    if not merged_oebin:
        raise ValueError('No data found in structure.oebin files')
    
    merged_oebin_path = upath.UPath(tempfile.gettempdir()) / 'structure.oebin'
    merged_oebin_path.write_text(json.dumps(merged_oebin, indent=4))
    return merged_oebin_path


def read_oebin(path: str | pathlib.Path | upath.UPath) -> dict[str, Any]:
    return json.loads(file_io.from_pathlike(path).read_bytes())


if __name__ == "__main__":
    import doctest

    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
