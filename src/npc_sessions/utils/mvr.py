from __future__ import annotations

import json
import logging
from collections.abc import Container, Iterable, Mapping
from typing import Literal, TypeVar, Union

import cv2
import numpy as np
import numpy.typing as npt
import upath
from typing_extensions import TypeAlias

import npc_sessions.utils as utils

logger = logging.getLogger(__name__)


def get_video_frame_times(
    sync_path_or_dataset: utils.PathLike | utils.SyncDataset,
    *video_paths: utils.PathLike,
) -> dict[upath.UPath, npt.NDArray[np.float64]]:
    """Returns frametimes as measured on sync clock for each video file.

    If a single directory is passed, video files in that directory will be
    found. If multiple paths are passed, the video files will be filtered out.

    - keys are video file paths
    - values are arrays of frame times in seconds
    - the first frametime will be a nan value (corresponding to a metadata frame)
    - frames at the end may also be nan values:

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
    camera_to_json_data = {
        utils.extract_camera_name(path.stem): get_video_info_data(path)
        for path in jsons
    }
    camera_exposing_times = get_cam_exposing_times_on_sync(sync_path_or_dataset)
    frame_times: dict[upath.UPath, npt.NDArray[np.floating]] = {}
    for camera in camera_exposing_times:
        if camera in camera_to_video_path:
            camera_frame_times = remove_lost_frame_times(
                camera_exposing_times[camera],
                get_lost_frames_from_camera_info(camera_to_json_data[camera]),
            )
            # Insert a nan frame time at the beginning to account for metadata frame
            camera_frame_times = np.insert(camera_frame_times, 0, np.nan)
            # append nan frametimes for frames in the video file but are
            # unnaccounted for on sync:
            if (
                frames_missing_from_sync := len(frame_times)
                - get_total_frames_from_camera_info(camera_to_json_data[camera])
            ) > 0:
                camera_frame_times = np.append(
                    camera_frame_times,
                    np.full(frames_missing_from_sync, np.nan),
                )
            frame_times[camera_to_video_path[camera]] = camera_frame_times
    return frame_times


def get_cam_line_times_on_sync(
    sync_path_or_dataset: utils.PathLike | utils.SyncDataset,
    sync_line_suffix: str,
    edge_type: Literal["rising", "falling"] = "rising",
) -> dict[Literal["behavior", "eye", "face"], npt.NDArray[np.float64]]:
    sync_data = utils.get_sync_data(sync_path_or_dataset)

    edge_getter = (
        sync_data.get_rising_edges
        if edge_type == "rising"
        else sync_data.get_falling_edges
    )

    line_times = {}
    for line in (line for line in sync_data.line_labels if sync_line_suffix in line):
        camera_name = utils.extract_camera_name(line)
        line_times[camera_name] = edge_getter(line, units="seconds")
    return line_times


def get_cam_exposing_times_on_sync(
    sync_path_or_dataset: utils.PathLike | utils.SyncDataset,
) -> dict[Literal["behavior", "eye", "face"], npt.NDArray[np.float64]]:
    return get_cam_line_times_on_sync(sync_path_or_dataset, "_cam_exposing")


def get_cam_exposing_falling_edge_times_on_sync(
    sync_path_or_dataset: utils.PathLike | utils.SyncDataset,
) -> dict[Literal["behavior", "eye", "face"], npt.NDArray[np.float64]]:
    return get_cam_line_times_on_sync(sync_path_or_dataset, "_cam_exposing", "falling")


def get_cam_transfer_times_on_sync(
    sync_path_or_dataset: utils.PathLike | utils.SyncDataset,
) -> dict[Literal["behavior", "eye", "face"], npt.NDArray[np.float64]]:
    return get_cam_line_times_on_sync(sync_path_or_dataset, "_cam_frame_readout")


def get_lost_frames_from_camera_info(
    info_path_or_data: MVRInfoData | utils.PathLike,
) -> npt.NDArray[np.int32]:
    """
    >>> get_lost_frames_from_camera_info({'LostFrames': ['1-2,4-5,7']})
    array([0, 1, 3, 4, 6])
    """
    info = get_video_info_data(info_path_or_data)

    if info.get("FramesLostCount") == 0:
        return np.array([])

    assert isinstance(_lost_frames := info["LostFrames"], list)
    lost_frame_spans: list[str] = _lost_frames[0].split(",")

    lost_frames: list[int] = []
    for span in lost_frame_spans:
        start_end = span.split("-")
        if len(start_end) == 1:
            lost_frames.append(int(start_end[0]))
        else:
            lost_frames.extend(np.arange(int(start_end[0]), int(start_end[1]) + 1))

    return np.subtract(lost_frames, 1)  # lost frames in info are 1-indexed


def get_total_frames_from_camera_info(
    info_path_or_data: MVRInfoData | utils.PathLike,
) -> int:
    """`FramesRecorded` in info.json plus 1 (for metadata frame)."""
    info = get_video_info_data(info_path_or_data)
    assert isinstance((reported := info.get("FramesRecorded")), int)
    return reported + 1


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


MVRInfoData: TypeAlias = Mapping[str, Union[str, int, float, list[str]]]
"""Contents of `RecordingReport` from a camera's info.json for an MVR
recording."""


def get_video_info_data(path_or_info_data: utils.PathLike | Mapping) -> MVRInfoData:
    if isinstance(path_or_info_data, Mapping):
        if "RecordingReport" in path_or_info_data:
            return path_or_info_data["RecordingReport"]
        return path_or_info_data
    return json.loads(utils.from_pathlike(path_or_info_data).read_bytes())[
        "RecordingReport"
    ]


def get_video_data(
    video_or_video_path: cv2.VideoCapture | utils.PathLike,
) -> cv2.VideoCapture:
    if isinstance(video_or_video_path, cv2.VideoCapture):
        return video_or_video_path

    video_path = utils.from_pathlike(video_or_video_path)
    # check if this is a local or cloud path
    is_local = video_path.as_uri()[:4] == "file"
    if not is_local:
        raise NotImplementedError(
            "Getting video data not yet implemented for cloud resources"
        )
    return cv2.VideoCapture(video_path.as_posix())


def get_total_frames_in_video(
    video_path: utils.PathLike,
) -> int:
    v = get_video_data(video_path)
    num_frames = v.get(cv2.CAP_PROP_FRAME_COUNT)

    return int(num_frames)


def get_augmented_camera_info(
    sync_path_or_dataset: utils.PathLike | utils.SyncDataset,
    *video_paths: utils.PathLike,
) -> dict[Literal["eye", "face", "behavior"], dict[str, int | float]]:
    videos = get_video_file_paths(*video_paths)
    jsons = get_video_info_file_paths(*video_paths)
    camera_to_video_path = {
        utils.extract_camera_name(path.stem): path for path in videos
    }
    camera_to_json_path = {utils.extract_camera_name(path.stem): path for path in jsons}

    cam_exposing_times = get_cam_exposing_times_on_sync(sync_path_or_dataset)
    cam_transfer_times = get_cam_transfer_times_on_sync(sync_path_or_dataset)
    cam_exposing_falling_edge_times = get_cam_exposing_falling_edge_times_on_sync(
        sync_path_or_dataset
    )

    augmented_camera_info = {}
    for camera, video_path in camera_to_video_path.items():
        camera_info = json.loads(camera_to_json_path[camera].read_bytes())[
            "RecordingReport"
        ]

        frames_lost = camera_info["FramesLostCount"]
        num_exposures = cam_exposing_times[camera].size
        num_transfers = cam_transfer_times[camera].size

        num_frames_in_video = get_total_frames_in_video(video_path)
        num_expected_from_sync = num_transfers - frames_lost + 1
        signature_exposures = (
            cam_exposing_falling_edge_times[camera][:10]
            - cam_exposing_times[camera][:10]
        )

        camera_info["num_frames_exposed"] = num_exposures
        camera_info["num_frames_transfered"] = num_transfers
        camera_info["num_frames_in_video"] = num_frames_in_video
        camera_info["num_expected_from_sync"] = num_expected_from_sync
        camera_info["expected_minus_actual"] = (
            num_expected_from_sync - num_frames_in_video
        )
        camera_info["signature_exposure_duration"] = np.round(
            np.median(signature_exposures), 3
        )
        augmented_camera_info[camera] = camera_info

    return augmented_camera_info


if __name__ == "__main__":
    import doctest

    import dotenv

    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
