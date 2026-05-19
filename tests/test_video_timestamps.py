import numpy as np
import pandas as pd
import pynwb

from npc_sessions.utils.videos import get_timestamps_from_dynamic_table


def test_dynamic_table_timestamps_can_be_reused_as_flat_float_column() -> None:
    timestamps = np.array([1.1, 2.2, 3.3], dtype=np.float64)
    frame_times = pynwb.core.DynamicTable.from_dataframe(
        name="frametimes_face_camera",
        table_description="test frame times",
        df=pd.DataFrame({"timestamps": timestamps}),
    )

    df = pd.DataFrame({"x": [10, 20, 30]})
    df["timestamps"] = get_timestamps_from_dynamic_table(frame_times)
    table = pynwb.core.DynamicTable.from_dataframe(
        name="lp_face_camera",
        table_description="test LP output",
        df=df,
    )

    assert df["timestamps"].dtype == np.float64
    np.testing.assert_allclose(np.asarray(table["timestamps"]), timestamps)
