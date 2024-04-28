from __future__ import annotations

import tempfile
from typing import Literal

import ndx_pose
import npc_lims
import npc_mvr
import numpy as np
import numpy.typing as npt
import pandas as pd
import upath
import zarr

# nice little trick from carter peene
MODEL_FUNCTION_MAPPING = {
    "dlc_eye": npc_lims.get_dlc_eye_s3_paths,
    "dlc_side": npc_lims.get_dlc_side_s3_paths,
    "dlc_face": npc_lims.get_dlc_face_s3_paths,
}

FACEMAP_CAMERA_NAMES: tuple[npc_mvr.CameraName, ...] = ("behavior", "face")


def get_dlc_session_paf_graph(session: str, model_name: str) -> list:
    """
    https://github.com/DeepLabCut/DLC2NWB/blob/main/dlc2nwb/utils.py#L139
    >>> get_dlc_session_paf_graph('676909_2023-12-12', 'dlc_eye')
    []
    """
    model_s3_paths = MODEL_FUNCTION_MAPPING[model_name](session)
    metadata_pickle_file = tuple(
        path for path in model_s3_paths if path.suffix == ".pickle"
    )
    if not metadata_pickle_file:
        raise FileNotFoundError(
            f"No metadata pickle found for session {session}. Check {model_name} capsule"
        )

    metadata_pickle = pd.read_pickle(metadata_pickle_file[0])
    test_config = metadata_pickle["data"]["DLC-model-config file"]
    paf_graph = test_config.get("partaffinityfield_graph", [])
    if paf_graph:
        paf_inds = test_config.get("paf_best")
        if paf_inds is not None:
            paf_graph = [paf_graph[i] for i in paf_inds]

    return paf_graph


def h5_to_dataframe(h5_path: upath.UPath, key_name: str | None = None) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as tempdir:
        with open(f"{tempdir}/{h5_path.stem}", "wb") as h5_file:
            h5_file.write(h5_path.read_bytes())
            if key_name is not None:
                df_h5 = pd.read_hdf(f"{tempdir}/{h5_path.stem}", key=key_name)
            else:
                df_h5 = pd.read_hdf(f"{tempdir}/{h5_path.stem}")

    return df_h5


def get_ellipse_session_dataframe_from_h5(session: str) -> pd.DataFrame:
    """
    >>> df_ellipse = get_ellipse_session_dataframe_from_h5('676909_2023-12-12')
    >>> len(df_ellipse)
    512347
    """
    eye_s3_paths = npc_lims.get_dlc_eye_s3_paths(session)
    ellipse_h5_path = tuple(
        path for path in eye_s3_paths if path.stem == "ellipses_processed"
    )
    if not ellipse_h5_path:
        raise FileNotFoundError(
            f"No ellipse h5 file found for {session}. Check dlc eye capsule"
        )

    # verbatim from allensdk: allensdk.brain_observatory.behavior.eye_tracking_processing.py
    eye_tracking_fields = ["cr", "eye", "pupil"]

    eye_tracking_dfs = []
    for field_name in eye_tracking_fields:
        df_ellipse = h5_to_dataframe(ellipse_h5_path[0], key_name=field_name)
        new_col_name_map = {
            col_name: f"{field_name}_{col_name}"
            for col_name in df_ellipse.columns
            if "center" in col_name
            or "width" in col_name
            or "height" in col_name
            or "phi" in col_name
        }
        df_ellipse.rename(new_col_name_map, axis=1, inplace=True)
        eye_tracking_dfs.append(df_ellipse)

    eye_tracking_data = pd.concat(eye_tracking_dfs, axis=1)
    eye_tracking_data.index.name = "frame"

    # Values in the hdf5 may be complex (likely an artifact of the ellipse
    # fitting process). Take only the real component.
    eye_tracking_data = eye_tracking_data.apply(lambda x: np.real(x.to_numpy()))

    return eye_tracking_data


def get_dlc_session_model_dataframe_from_h5(
    session: str, model_name: str
) -> pd.DataFrame:
    """
    >>> df_model = get_dlc_session_model_dataframe_from_h5('676909_2023-12-12', 'dlc_eye')
    >>> len(df_model)
    512347
    """
    model_s3_paths = MODEL_FUNCTION_MAPPING[model_name](session)
    h5_files = tuple(
        path
        for path in model_s3_paths
        if path.suffix == ".h5" and "ellipses" not in path.stem
    )

    if not h5_files:
        raise FileNotFoundError(
            f"No h5 files found for {session}. Check {model_name} capsule"
        )

    df_model = h5_to_dataframe(h5_files[0])
    return df_model


def get_pose_series_from_dataframe(
    session: str, df: pd.DataFrame, video_timestamps: npt.NDArray[np.float64]
) -> list[ndx_pose.pose.PoseEstimationSeries]:
    # https://github.com/DeepLabCut/DLC2NWB/blob/main/dlc2nwb/utils.py#L189
    if df.shape[0] != len(video_timestamps):
        raise ValueError(
            f"{session} pose dataframe has {df.shape[0]} rows, but {len(video_timestamps)} timestamps were provided."
            "\nMake sure another data asset wasn't attached to the capsule when run."
        )
    pose_estimations_series = []
    for keypoint, xy_positions in df.T.groupby(level="bodyparts", sort=False):
        assert xy_positions.shape[0] == 3
        data = xy_positions.to_numpy().T
        pose_estimation_series = ndx_pose.pose.PoseEstimationSeries(
            name=keypoint,
            description=f"{keypoint} keypoint position in each frame",  # TODO use a lookup table of abbreviation: description
            data=data[:, :2],
            unit="pixels",
            reference_frame="(0,0) corresponds to the top left corner of the video frame",
            timestamps=video_timestamps,
            confidence=data[:, 2],
            confidence_definition="softmax output of the deep neural network",
        )
        pose_estimations_series.append(pose_estimation_series)

    return pose_estimations_series


def get_facemap_output_from_s3(
    session: str, camera_name: npc_mvr.CameraName, array_name: Literal["motSVD", "proc"]
) -> zarr.Array:  # currently only saving motSVD
    """
    >>> behavior_motion_svd = get_facemap_output_from_s3('646318_2023-01-17', 'Behavior', 'motSVD')
    >>> behavior_motion_svd.shape
    (284190, 500)
    """
    session_facemap_paths = npc_lims.get_facemap_s3_paths(session)
    if not session_facemap_paths:
        raise FileNotFoundError(
            f"No facemap files found for session {session}. Check codeocean"
        )
    camera_name = npc_mvr.get_camera_name(camera_name)
    if camera_name not in FACEMAP_CAMERA_NAMES:
        raise ValueError(f"{camera_name} not part of facemap output")
    try:
        file_path = next(
            path
            for path in session_facemap_paths
            if camera_name in path.stem.lower()
            and array_name.lower() in path.stem.lower()
        )
    except StopIteration:
        raise FileNotFoundError(
            f"No {array_name} file found for session {session} {camera_name} camera, despite other paths existing. Check codeocean"
        ) from None
    else:
        return zarr.open(file_path)


if __name__ == "__main__":
    import doctest

    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
