from __future__ import annotations

import json
import logging
from collections.abc import Container, Iterable
from typing import Literal, TypeVar

import numpy as np
import numpy.typing as npt
import upath

import npc_sessions.utils as utils

logger = logging.getLogger(__name__)


def get_video_frame_times(
    sync_path_or_dataset: utils.PathLike | utils.SyncDataset,
    *video_paths: utils.PathLike,
) -> dict[upath.UPath, npt.NDArray[np.float64]]:
    """Returns number of frames on sync line for each video file (after
    subtracting lost frames).

    If a single directory is passed, video files in that directory will be
    found. If multiple paths are passed, the video files will be filtered out.

    - keys are video file paths
    - values are arrays of frame times in seconds

    - if the number of frames in a video file doesn't match the number of frames
    returned here, just truncate the excess frames in the video file:

        MVR previously ceased all TTL pulses before the recording was
        stopped, resulting in frames in the video that weren't registered
        in sync. MVR was fixed July 2023 after Corbett discovered the issue.

    >>> sync_path = 's3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15/behavior/20230803T120415.h5'
    >>> video_path = 's3://aind-ephys-data/ecephys_670248_2023-08-03_12-04-15/behavior'
    >>> frame_times = get_video_frame_times(sync_path, video_path)
    >>> [len(frames) for frames in frame_times.values()]
    [304229, 304229, 304229]
    """
    videos = get_video_file_paths(*video_paths)
    jsons = get_video_info_file_paths(*video_paths)
    camera_to_video_path = {
        utils.extract_camera_name(path.stem): path for path in videos
    }
    camera_to_json_path = {utils.extract_camera_name(path.stem): path for path in jsons}
    camera_exposing_times = get_cam_exposing_times_on_sync(sync_path_or_dataset)
    frame_times = {}
    for camera in camera_exposing_times:
        if camera in camera_to_video_path:
            frame_times[camera_to_video_path[camera]] = remove_lost_frame_times(
                camera_exposing_times[camera],
                get_lost_frames_from_camera_info(camera_to_json_path[camera]),
            )
            # Insert a nan frame time at the beginning to account for metadata frame
            frame_times[camera_to_video_path[camera]] = np.insert(
                frame_times[camera_to_video_path[camera]], 0, np.nan
            )
    return frame_times


def get_cam_exposing_times_on_sync(
    sync_path_or_dataset: utils.PathLike | utils.SyncDataset,
) -> dict[Literal["behavior", "eye", "face"], npt.NDArray[np.float64]]:
    if isinstance(sync_path_or_dataset, utils.SyncDataset):
        sync_data = sync_path_or_dataset
    else:
        sync_data = utils.SyncDataset(utils.from_pathlike(sync_path_or_dataset))

    frame_times = {}
    for line in (line for line in sync_data.line_labels if "_cam_exposing" in line):
        camera_name = utils.extract_camera_name(line)
        frame_times[camera_name] = sync_data.get_rising_edges(line, units="seconds")
    return frame_times


def get_cam_exposing_falling_edge_times_on_sync(
    sync_path_or_dataset: utils.PathLike | utils.SyncDataset,
) -> dict[Literal["behavior", "eye", "face"], npt.NDArray[np.float64]]:
    if isinstance(sync_path_or_dataset, utils.SyncDataset):
        sync_data = sync_path_or_dataset
    else:
        sync_data = utils.SyncDataset(utils.from_pathlike(sync_path_or_dataset))

    frame_times = {}
    for line in (line for line in sync_data.line_labels if "_cam_exposing" in line):
        camera_name = utils.extract_camera_name(line)
        frame_times[camera_name] = sync_data.get_falling_edges(line, units="seconds")
    return frame_times


def get_cam_transfer_times_on_sync(
    sync_path_or_dataset: utils.PathLike | utils.SyncDataset,
) -> dict[Literal["behavior", "eye", "face"], npt.NDArray[np.float64]]:
    if isinstance(sync_path_or_dataset, utils.SyncDataset):
        sync_data = sync_path_or_dataset
    else:
        sync_data = utils.SyncDataset(utils.from_pathlike(sync_path_or_dataset))

    frame_times = {}
    for line in (
        line for line in sync_data.line_labels if "_cam_frame_readout" in line
    ):
        camera_name = utils.extract_camera_name(line)
        frame_times[camera_name] = sync_data.get_rising_edges(line, units="seconds")
    return frame_times


def get_lost_frames_from_camera_info(
    info_path_or_dict: dict | utils.PathLike,
) -> npt.NDArray[np.int32]:
    """
    >>> get_lost_frames_from_camera_info({'LostFrames': ['1-2,4-5,7']})
    array([0, 1, 3, 4, 6])
    """
    if not isinstance(info_path_or_dict, dict):
        info = json.loads(utils.from_pathlike(info_path_or_dict).read_bytes())
    else:
        info = info_path_or_dict

    if "RecordingReport" in info:
        info = info["RecordingReport"]

    if info.get("FramesLostCount") == 0:
        return np.array([])

    lost_frame_spans: list[str] = info["LostFrames"][0].split(",")

    lost_frames = []
    for span in lost_frame_spans:
        start_end = span.split("-")
        if len(start_end) == 1:
            lost_frames.append(int(start_end[0]))
        else:
            lost_frames.extend(np.arange(int(start_end[0]), int(start_end[1]) + 1))

    return np.subtract(lost_frames, 1)  # lost frames in info are 1-indexed


NumericT = TypeVar("NumericT", bound=np.generic, covariant=True)


def remove_lost_frame_times(
    frame_times: Iterable[NumericT], lost_frame_idx: Container[int]
) -> npt.NDArray[NumericT]:
    """
    >>> remove_lost_frame_times([1., 2., 3., 4., 5.], [1, 3])
    array([1., 3., 5.])
    """
    return np.array(
        [t for idx, t in enumerate(frame_times) if idx not in lost_frame_idx]
    )


def get_video_file_paths(*paths: utils.PathLike) -> tuple[upath.UPath, ...]:
    if len(paths) == 1 and utils.from_pathlike(paths[0]).is_dir():
        upaths = tuple(utils.from_pathlike(paths[0]).glob("*"))
    else:
        upaths = tuple(utils.from_pathlike(p) for p in paths)
    return tuple(
        p
        for p in upaths
        if p.suffix in (".avi", ".mp4", ".zip")
        and any(label in p.stem.lower() for label in ("eye", "face", "beh"))
    )


def get_video_info_file_paths(*paths: utils.PathLike) -> tuple[upath.UPath, ...]:
    return tuple(
        p.with_suffix(".json").with_stem(p.stem.replace(".mp4", "").replace(".avi", ""))
        for p in get_video_file_paths(*paths)
    )


if __name__ == "__main__":
    import doctest

    import dotenv

    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
