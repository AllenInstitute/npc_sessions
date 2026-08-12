from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from pandas.api.types import is_float_dtype

from npc_sessions.utils import electrodes


def test_tissuecyte_electrode_coordinates_are_floats(monkeypatch) -> None:
    monkeypatch.setattr(
        electrodes.npc_lims,
        "get_tissuecyte_annotation_files_from_s3",
        lambda _: [Path("annotation.csv")],
    )
    monkeypatch.setattr(
        electrodes.npc_session,
        "ProbeRecord",
        lambda _: SimpleNamespace(name="probeA"),
    )
    monkeypatch.setattr(
        electrodes.pd,
        "read_csv",
        lambda _: pd.DataFrame(
            {
                "AP": [1],
                "DV": [2],
                "ML": [3],
                "region": ["VISp"],
                "region_stripped": ["VISp"],
                "channel": [0],
            }
        ),
    )

    result = electrodes.get_tissuecyte_electrodes_table("662892_2023-08-21")

    assert all(is_float_dtype(result[column]) for column in ("x", "y", "z"))
