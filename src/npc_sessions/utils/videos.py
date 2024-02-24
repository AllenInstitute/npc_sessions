from __future__ import annotations

import io
import tempfile
from typing import Any

import ndx_pose
import npc_lims
import numpy as np
import numpy.typing as npt
import pandas as pd
import upath
from scipy import ndimage, stats

# nice little trick from carter peene
MODEL_FUNCTION_MAPPING = {
    "dlc_eye": npc_lims.get_dlc_eye_s3_paths,
    "dlc_side": npc_lims.get_dlc_side_s3_paths,
    "dlc_face": npc_lims.get_dlc_face_s3_paths,
}

FACEMAP_VIDEO_OUTPUT_CATEGORIES = ["Behavior", "Face"]


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


# helper functions below taken from allensdk
def compute_elliptical_area(df_row: pd.Series) -> float:
    """Calculate the area of corneal reflection (cr) or eye ellipse fits using
    the ellipse formula.

    Parameters
    ----------
    df_row : pd.Series
        A row from an eye tracking dataframe containing either:
        "cr_width", "cr_height"
        or
        "eye_width", "eye_height"

    Returns
    -------
    float
        The elliptical area of the eye or cr in pixels^2
    """
    return np.pi * df_row.iloc[0] * df_row.iloc[1]


def compute_circular_area(df_row: pd.Series) -> float:
    """Calculate the area of the pupil as a circle using the max of the
    height/width as radius.

    Note: This calculation assumes that the pupil is a perfect circle
    and any eccentricity is a result of the angle at which the pupil is
    being viewed.

    Parameters
    ----------
    df_row : pd.Series
        A row from an eye tracking dataframe containing only "pupil_width"
        and "pupil_height".

    Returns
    -------
    float
        The circular area of the pupil in pixels^2.
    """
    max_dim = max(df_row.iloc[0], df_row.iloc[1])
    return np.pi * max_dim * max_dim


def determine_outliers(data_df: pd.DataFrame, z_threshold: float) -> pd.Series:
    """Given a dataframe and some z-score threshold return a pandas boolean
    Series where each entry indicates whether a given row contains at least
    one outlier (where outliers are calculated along columns).

    Parameters
    ----------
    data_df : pd.DataFrame
        A dataframe containing only columns where outlier detection is
        desired. (e.g. "cr_area", "eye_area", "pupil_area")
    z_threshold : float
        z-score values higher than the z_threshold will be considered outliers.

    Returns
    -------
    pd.Series
        A pandas boolean Series whose length == len(data_df.index).
        True denotes that a row in the data_df contains at least one outlier.
    """

    outliers = (
        data_df.apply(stats.zscore, nan_policy="omit").apply(np.abs) > z_threshold
    )
    return pd.Series(outliers.any(axis=1))


def determine_likely_blinks(
    eye_areas: pd.Series,
    pupil_areas: pd.Series,
    outliers: pd.Series,
    dilation_frames: int = 2,
) -> pd.Series:
    """Determine eye tracking frames which contain likely blinks or outliers

    Parameters
    ----------
    eye_areas : pd.Series
        A pandas series of eye areas.
    pupil_areas : pd.Series
        A pandas series of pupil areas.
    outliers : pd.Series
        A pandas series containing bool values of outlier rows.
    dilation_frames : int, optional
        Determines the number of additional adjacent frames to mark as
        'likely_blink', by default 2.

    Returns
    -------
    pd.Series
        A pandas series of bool values that has the same length as the number
        of eye tracking dataframe rows (frames).
    """
    blinks = pd.isnull(eye_areas) | pd.isnull(pupil_areas) | outliers
    if dilation_frames > 0:
        likely_blinks = ndimage.binary_dilation(blinks, iterations=dilation_frames)
    else:
        likely_blinks = blinks
    return pd.Series(likely_blinks, index=eye_areas.index)


def filter_on_blinks(eye_tracking_data: pd.DataFrame):
    """Set data is specified columns where likely_blink is true to NaN.

    Modify the DataFrame in place.

    Parameters
    ----------
    eye_tracking_data : pandas.DataFrame
        Data frame containing eye tracking data.
    """
    likely_blinks = eye_tracking_data["likely_blink"]
    eye_tracking_data.loc[likely_blinks, "eye_area"] = np.nan
    eye_tracking_data.loc[likely_blinks, "pupil_area"] = np.nan
    eye_tracking_data.loc[likely_blinks, "cr_area"] = np.nan

    eye_tracking_data.loc[likely_blinks, "eye_width"] = np.nan
    eye_tracking_data.loc[likely_blinks, "eye_height"] = np.nan
    eye_tracking_data.loc[likely_blinks, "eye_phi"] = np.nan

    eye_tracking_data.loc[likely_blinks, "pupil_width"] = np.nan
    eye_tracking_data.loc[likely_blinks, "pupil_height"] = np.nan
    eye_tracking_data.loc[likely_blinks, "pupil_phi"] = np.nan


def get_ellipse_session_dataframe_from_h5(session: str) -> pd.DataFrame:
    """
    >>> df_ellipse = get_ellipse_session_dataframe_from_h5('676909_2023-12-12')
    >>> len(df_ellipse)
    512347
    """
    eye_s3_paths = npc_lims.get_dlc_eye_s3_paths(session)
    ellipse_h5_path = tuple(path for path in eye_s3_paths if path.stem == "ellipses")
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
            col_name: f"{field_name}_{col_name}" for col_name in df_ellipse.columns
        }
        df_ellipse.rename(new_col_name_map, axis=1, inplace=True)
        eye_tracking_dfs.append(df_ellipse)

    eye_tracking_data = pd.concat(eye_tracking_dfs, axis=1)
    eye_tracking_data.index.name = "frame"

    # Values in the hdf5 may be complex (likely an artifact of the ellipse
    # fitting process). Take only the real component.
    eye_tracking_data = eye_tracking_data.apply(lambda x: np.real(x.to_numpy()))

    return eye_tracking_data.astype(float)


def get_computed_ellipse_metrics_dataframe(
    session: str, z_threshold: float = 3.0, dilation_frames: int = 2
) -> pd.DataFrame:
    """
    >>> df_eliipse_computed_metrics = get_computed_ellipse_metrics_dataframe('676909_2023-12-12')
    >>> df_eliipse_computed_metrics.columns
    Index(['cr_area', 'eye_area', 'pupil_area', 'likely_blink', 'pupil_area_raw',
           'cr_area_raw', 'eye_area_raw', 'cr_center_x', 'cr_center_y', 'cr_width',
           'cr_height', 'cr_phi', 'eye_center_x', 'eye_center_y', 'eye_width',
           'eye_height', 'eye_phi', 'pupil_center_x', 'pupil_center_y',
           'pupil_width', 'pupil_height', 'pupil_phi'],
          dtype='object')
    """
    eye_data = get_ellipse_session_dataframe_from_h5(session)
    cr_areas = eye_data[["cr_width", "cr_height"]].apply(
        compute_elliptical_area, axis=1
    )
    eye_areas = eye_data[["eye_width", "eye_height"]].apply(
        compute_elliptical_area, axis=1
    )
    pupil_areas = eye_data[["pupil_width", "pupil_height"]].apply(
        compute_circular_area, axis=1
    )

    # only use eye and pupil areas for outlier detection
    area_df = pd.concat([eye_areas, pupil_areas], axis=1)
    outliers = determine_outliers(area_df, z_threshold=z_threshold)

    likely_blinks = determine_likely_blinks(
        eye_areas, pupil_areas, outliers, dilation_frames=dilation_frames
    )

    # remove outliers/likely blinks `pupil_area`, `cr_area`, `eye_area`
    pupil_areas_raw = pupil_areas.copy()
    cr_areas_raw = cr_areas.copy()
    eye_areas_raw = eye_areas.copy()

    eye_data.insert(0, "cr_area", cr_areas)
    eye_data.insert(1, "eye_area", eye_areas)
    eye_data.insert(2, "pupil_area", pupil_areas)
    eye_data.insert(3, "likely_blink", likely_blinks)
    eye_data.insert(4, "pupil_area_raw", pupil_areas_raw)
    eye_data.insert(5, "cr_area_raw", cr_areas_raw)
    eye_data.insert(6, "eye_area_raw", eye_areas_raw)

    # Apply blink fliter to additional columns.
    filter_on_blinks(eye_data)

    return eye_data


def get_dlc_session_model_dataframe_from_h5(
    session: str, model_name: str
) -> pd.DataFrame:
    """
    >>> df_model = get_dlc_session_model_dataframe_from_h5('676909_2023-12-12', 'dlc_eye')
    >>> len(df_model)
    512347
    """
    model_s3_paths = MODEL_FUNCTION_MAPPING[model_name](session)
    # TODO figure out ellipses.h5
    h5_files = tuple(path for path in model_s3_paths if path.suffix == ".h5")

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
    pose_estimations_series = []
    for keypoint, xy_positions in df.groupby(level="bodyparts", axis=1, sort=False):
        data = xy_positions.to_numpy()
        pose_estimation_series = ndx_pose.pose.PoseEstimationSeries(
            name=f"{session}_{keypoint}",
            description=f"Keypoint {keypoint} from session {session}.",
            data=data[:, :2],
            unit="pixels",
            reference_frame="(0,0) corresponds to the bottom left corner of the video.",
            timestamps=video_timestamps,
            confidence=data[:, 2],
            confidence_definition="Softmax output of the deep neural network.",
        )
        pose_estimations_series.append(pose_estimation_series)

    return pose_estimations_series


def get_facemap_output_from_s3(session: str, video_type: str) -> dict[str, Any]:
    """
    >>> proc = get_facemap_output_from_s3('676909_2023-12-12', 'Behavior')
    >>> proc['motSVD'][1].shape
    (512370, 500)
    """
    session_facemap_paths = npc_lims.get_facemap_s3_paths(session)

    if video_type not in FACEMAP_VIDEO_OUTPUT_CATEGORIES:
        raise ValueError(f"{video_type} not part of facemap output")

    facemap_path = tuple(
        path for path in session_facemap_paths if video_type in path.stem
    )

    if not facemap_path:
        raise FileNotFoundError(
            f"No {video_type} proc file found for session {session}. Check codeocean"
        )

    facemap_proc_path = facemap_path[0]
    with io.BytesIO(facemap_proc_path.read_bytes()) as f:
        proc = np.load(f, allow_pickle=True).item()

    return proc


if __name__ == "__main__":
    import doctest

    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
