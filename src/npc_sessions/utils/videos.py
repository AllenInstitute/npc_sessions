import tempfile

import ndx_pose
import npc_lims
import numpy as np
import numpy.typing as npt
import pandas as pd
import upath

# nice little trick from carter peene
MODEL_FUNCTION_MAPPING = {
    "dlc_eye": npc_lims.get_dlc_eye_s3_paths,
    "dlc_side": npc_lims.get_dlc_side_s3_paths,
    "dlc_face": npc_lims.get_dlc_face_s3_paths,
}


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


def h5_to_dataframe(h5_path: upath.UPath) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as tempdir:
        with open(f"{tempdir}/{h5_path.stem}", "wb") as h5_file:
            h5_file.write(h5_path.read_bytes())
            df_h5 = pd.read_hdf(f"{tempdir}/{h5_path.stem}")

    return df_h5


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


if __name__ == "__main__":
    import doctest

    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
