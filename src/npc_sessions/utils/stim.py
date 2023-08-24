from __future__ import annotations

import json
import logging
from collections.abc import Container, Iterable
from typing import Literal, TypeVar

import numpy as np
import numpy.typing as npt
import upath
import h5py
import io
import datetime
import upath
import warnings

import npc_sessions.utils.file_io as file_io
import npc_sessions.utils.sync as sync
import npc_sessions.utils as utils
import npc_session

logger = logging.getLogger(__name__)

def load_h5_dataset(path: upath.UPath) -> h5py.File:
    return h5py.File(io.BytesIO(path.read_bytes()), "r")

def get_stim_frame_times(
        stim_paths: tuple | file_io.PathLike, 
        sync_path: file_io.PathLike,
        frame_time_type='display_time'
        ) -> dict[upath.UPath, npt.NDArray[np.float64]]:

    #load sync file once
    sync_data = sync.SyncDataset(file_io.from_pathlike(sync_path))
    #get sync start time
    sync_start_time = sync_data.start_time
    #get vsync_times_in_blocks
    if frame_time_type=='vsync':
        frame_times_in_blocks = sync_data.vsync_times_in_blocks
    #get frame_display_time_blocks
    elif frame_time_type=='display_time':
        frame_times_in_blocks = sync_data.frame_display_time_blocks
    #get num frames in each block
    n_frames_per_block = np.asarray([len(x) for x in frame_times_in_blocks])
    #get first frame time in each block
    first_frame_per_block = np.asarray([x[0] for x in frame_times_in_blocks])


    stim_frame_times: dict[upath.UPath, npt.NDArray[np.float64]] = {}

    if isinstance(stim_paths, file_io.PathLike):
        stim_paths=tuple([stim_paths])

    #loop through stim files (if multiple)
    for stim_path in stim_paths:
        #load each stim file once
        stim_data = load_h5_dataset(stim_path)
        #get start time
        stim_start_time = get_stim_start_time(stim_data)
        #get number of frames
        n_stim_frames = get_n_frames(stim_data)
        #get duration
        stim_duration = get_stim_duration(stim_data) #
        #get start time relative to sync
        stim_start_time_rel_sync = (stim_start_time - sync_start_time).seconds
        #try to match to vsyncs by start time
        block_match = np.where(first_frame_per_block>stim_start_time_rel_sync)[0]
        if len(block_match)>0:
            block_match = block_match[0]
            block_frames_match = n_stim_frames == n_frames_per_block[block_match]
        else:
            block_match = np.nan
            block_frames_match = False

        if np.isnan(block_match):
            warnings.warn('Block match not found')
            stim_frame_times[stim_path] = []

        elif block_frames_match == False:
            warnings.warn('Number of block frames do not match: off by'+str(n_stim_frames-n_frames_per_block[block_match]))
            stim_frame_times[stim_path] = []

        elif np.isnan(block_match)==False and block_frames_match==True:
            stim_frame_times[stim_path] = frame_times_in_blocks[block_match]

    return stim_frame_times


def get_stim_start_time(stim_path_or_data: file_io.PathLike | h5py.File) -> datetime.datetime:

    #load dataset
    if isinstance(stim_path_or_data, h5py.File):
        stim_data = stim_path_or_data
    else:
        stim_data = load_h5_dataset(file_io.from_pathlike(stim_path_or_data))

    #get stim start time & convert to datetime
    return npc_session.DatetimeRecord(stim_data['startTime'][()].decode()).dt


def get_n_frames(stim_path_or_data: file_io.PathLike | h5py.File) -> int:

    #load dataset
    if isinstance(stim_path_or_data, h5py._hl.files.File):
        stim_data = stim_path_or_data
    else:
        stim_data = load_h5_dataset(file_io.from_pathlike(stim_path_or_data))

    n_stim_frames=len(stim_data['frameIntervals'][:])+1

    return n_stim_frames

def get_stim_duration(stim_path_or_data: file_io.PathLike | h5py.File) -> float:

    #load dataset
    if isinstance(stim_path_or_data, h5py._hl.files.File):
        stim_data = stim_path_or_data
    else:
        stim_data = load_h5_dataset(file_io.from_pathlike(stim_path_or_data))

    stim_duration = np.sum(stim_data['frameIntervals'][:])

    return stim_duration


if __name__ == "__main__":
    import doctest

    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
