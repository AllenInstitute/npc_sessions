from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from npc_sessions.utils import atlas


@pytest.fixture
def synthetic_atlas(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Empty label volume with a few labeled voxels, at known CCF positions."""
    labels_path = tmp_path / atlas.UNIFIED_ATLAS_LABELS_FILENAME
    with labels_path.open("wb") as f:
        f.truncate(atlas._HEADER_BYTES + 2 * int(np.prod(atlas._SHAPE)))
    volume = np.memmap(
        labels_path,
        dtype="<u2",
        mode="r+",
        offset=atlas._HEADER_BYTES,
        shape=atlas._SHAPE,
        order="F",
    )
    # CCF (AP, DV, ML) = (5180, 3000, 7380) um -> voxel (570-1-369, 150, 660-1-259)
    volume[200, 150, 400] = 385
    # anterior-superior-left corner of the CCF -> opposite corner of RSP volume
    volume[-1, 0, -1] = 728
    volume.flush()
    del volume
    (tmp_path / atlas.UNIFIED_ATLAS_ONTOLOGY_FILENAME).write_text(
        "id,name,acronym,structure_id_path\n"
        "385,Primary visual cortex,V1 ,1\n"
        ",,,\n",
        encoding="utf-8-sig",
    )
    monkeypatch.setattr(atlas, "UNIFIED_ATLAS_DIR", tmp_path)
    atlas.get_unified_atlas_dir.cache_clear()
    atlas.get_unified_atlas_ontology.cache_clear()
    yield
    atlas.get_unified_atlas_dir.cache_clear()
    atlas.get_unified_atlas_ontology.cache_clear()


def test_add_atlas_labels_to_units(synthetic_atlas: None) -> None:
    units = pd.DataFrame(
        {
            "location": ["VISp5", "arb", "root", "out of brain", "out of brain"],
            "structure": ["VISp", "arb", "root", "out of brain", "out of brain"],
            "ccf_ap": [5199.9, 0.0, 100.0, np.nan, 13200.0],
            "ccf_dv": [3000.0, 0.0, 100.0, np.nan, 0.0],
            "ccf_ml": [7380.0, 19.9, 100.0, np.nan, 0.0],
        }
    )
    labeled = atlas.add_atlas_labels_to_units(units)
    assert labeled["allen_ccf_acronym"].equals(units["location"])
    assert labeled["allen_ccf_area"].equals(units["structure"])
    assert labeled["unified_atlas_acronym"].tolist() == [
        "V1",
        "arb",
        *[atlas.OUT_OF_BRAIN] * 3,
    ]
    assert labeled["unified_atlas_name"].tolist() == [
        "Primary visual cortex",
        "arbor vitae",
        *[atlas.OUT_OF_BRAIN] * 3,
    ]
