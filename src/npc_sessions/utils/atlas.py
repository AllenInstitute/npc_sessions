"""Brain-area labels for units, from Allen CCFv3 coordinates.

In addition to the Allen CCFv3 (2017) labels assigned during probe annotation,
units are labeled with the Unified Mouse Brain Atlas v2 (Chon et al., 2019),
which uses Franklin-Paxinos nomenclature, subdivides several Allen areas (e.g.
monocular/binocular V1) and redraws some boundaries.

https://figshare.com/articles/dataset/Unified_mouse_brain_atlas_v2/25750983
"""

from __future__ import annotations

import csv
import functools
import json
import logging
import os
import pathlib
import shutil
import urllib.request

import numpy as np
import numpy.typing as npt
import pandas as pd

logger = logging.getLogger(__name__)

UNIFIED_ATLAS_FIGSHARE_FILES_URL = "https://api.figshare.com/v2/articles/25750983/files"
UNIFIED_ATLAS_LABELS_FILENAME = "UnifiedAtlas_Label_v2_20um-isotropic.nii"
UNIFIED_ATLAS_ONTOLOGY_FILENAME = "UnifiedAtlas_Label_ontology_v2.csv"
UNIFIED_ATLAS_DIR = pathlib.Path(
    os.environ.get(
        "UNIFIED_ATLAS_DIR",
        pathlib.Path.home() / ".cache" / "npc_sessions" / "unified_atlas",
    )
)
"""Where atlas files are cached after the first download (override with env var
`UNIFIED_ATLAS_DIR`)"""

# The released NIfTI has no usable spatial transform: it is a 20 um grid in RSP
# orientation (axis 0: right->left, 1: dorsal->ventral, 2: posterior->anterior),
# stored Fortran-order after a 352-byte header. Allen CCF coords are AP, DV, ML
# from the anterior-superior-left corner.
_SHAPE = (570, 400, 660)
_RESOLUTION_UM = 20
_HEADER_BYTES = 352

OUT_OF_BRAIN = "out of brain"
"""Label for units without coordinates or outside annotated tissue, matching
the convention for `location` and `structure`"""
# IDs in the v2 volume without an entry in its ontology CSV (728 is arbor vitae
# in the Allen CCF ontology)
_MISSING_ONTOLOGY_LABELS = {
    0: (OUT_OF_BRAIN, OUT_OF_BRAIN),
    728: ("arb", "arbor vitae"),
}

UNITS_ATLAS_COLUMN_DESCRIPTIONS = {
    "ccf_ap": "anterior-posterior position of the unit's peak channel in the Allen CCFv3, in microns (increasing values are posterior)",
    "ccf_dv": "dorsal-ventral position of the unit's peak channel in the Allen CCFv3, in microns (increasing values are ventral)",
    "ccf_ml": "medial-lateral position of the unit's peak channel in the Allen CCFv3, in microns (increasing values are rightward)",
    "location": "same as `allen_ccf_acronym` (retained for compatibility)",
    "structure": "same as `allen_ccf_area` (retained for compatibility)",
    "allen_ccf_acronym": "acronym for the Allen CCFv3 (2017) area at the unit's peak channel, including cortical layer (e.g. `VISp5`)",
    "allen_ccf_area": "acronym for the Allen CCFv3 (2017) area at the unit's peak channel, without cortical layer (e.g. `VISp`)",
    "unified_atlas_acronym": "acronym for the Unified Mouse Brain Atlas v2 (Chon et al., 2019) area at `ccf_ap`, `ccf_dv`, `ccf_ml`, including cortical layer (e.g. `V1B-5`); the atlas uses Franklin-Paxinos nomenclature and subdivides or redraws some Allen CCFv3 areas",
    "unified_atlas_name": "full name of the area in `unified_atlas_acronym` (e.g. `Primary visual cortex, binocular area, layer5`)",
}
"""Descriptions of the annotation columns in the units table, for annotated sessions"""


def _download_figshare_file(item: dict, dest: pathlib.Path) -> None:
    logger.info(f"Downloading {item['name']} to {dest.parent}")
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.download")
    try:
        with (
            urllib.request.urlopen(item["download_url"], timeout=60) as response,
            tmp.open("wb") as f,
        ):
            shutil.copyfileobj(response, f, length=2**20)
        if (size := tmp.stat().st_size) != item["size"]:
            raise ValueError(
                f"Incomplete download of {item['name']}: {size} of {item['size']} bytes"
            )
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)


@functools.cache
def get_unified_atlas_dir() -> pathlib.Path:
    """Local directory containing the Unified Atlas v2 label volume and ontology,
    downloaded from Figshare (~300 MB) on first use."""
    paths = {
        name: UNIFIED_ATLAS_DIR / name
        for name in (UNIFIED_ATLAS_LABELS_FILENAME, UNIFIED_ATLAS_ONTOLOGY_FILENAME)
    }
    if not all(path.exists() for path in paths.values()):
        UNIFIED_ATLAS_DIR.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(
            UNIFIED_ATLAS_FIGSHARE_FILES_URL, timeout=60
        ) as response:
            files = {item["name"]: item for item in json.load(response)}
        for name, path in paths.items():
            if not path.exists():
                _download_figshare_file(files[name], path)
    labels_size = paths[UNIFIED_ATLAS_LABELS_FILENAME].stat().st_size
    if labels_size != _HEADER_BYTES + 2 * np.prod(_SHAPE):
        raise ValueError(f"Unexpected size for Unified Atlas labels: {labels_size}")
    return UNIFIED_ATLAS_DIR


@functools.cache
def get_unified_atlas_ontology() -> dict[int, tuple[str, str]]:
    """Region ID -> (acronym, name), including 0 for out of brain"""
    path = get_unified_atlas_dir() / UNIFIED_ATLAS_ONTOLOGY_FILENAME
    with path.open(newline="", encoding="utf-8-sig") as f:
        ontology = {
            int(row["id"]): (row["acronym"].strip(), row["name"].strip())
            for row in csv.DictReader(f)
            if row["id"].strip()  # the CSV has trailing blank rows
        }
    return ontology | _MISSING_ONTOLOGY_LABELS


def get_unified_atlas_region_ids(
    ccf_ap: npt.ArrayLike, ccf_dv: npt.ArrayLike, ccf_ml: npt.ArrayLike
) -> npt.NDArray[np.uint16]:
    """Unified Atlas v2 region IDs at Allen CCFv3 coordinates (microns), with 0
    for points outside the brain or with missing coordinates."""
    ap, dv, ml = (np.asarray(c, dtype=np.float64) for c in (ccf_ap, ccf_dv, ccf_ml))
    idx = np.floor_divide(np.stack([ap, dv, ml]), _RESOLUTION_UM)
    is_valid = np.all(
        np.isfinite(idx)
        & (idx >= 0)
        & (idx < np.array([_SHAPE[2], _SHAPE[1], _SHAPE[0]])[:, None]),
        axis=0,
    )
    ap_idx, dv_idx, ml_idx = idx[:, is_valid].astype(np.intp)
    volume = np.memmap(
        get_unified_atlas_dir() / UNIFIED_ATLAS_LABELS_FILENAME,
        dtype="<u2",
        mode="r",
        offset=_HEADER_BYTES,
        shape=_SHAPE,
        order="F",
    )
    region_ids = np.zeros(ap.shape, dtype=np.uint16)
    region_ids[is_valid] = volume[
        _SHAPE[0] - 1 - ml_idx, dv_idx, _SHAPE[2] - 1 - ap_idx
    ]
    return region_ids


def add_atlas_labels_to_units(units: pd.DataFrame) -> pd.DataFrame:
    """Add brain-area labels to an annotated units table (with columns
    `location`, `structure`, `ccf_ap`, `ccf_dv`, `ccf_ml`).

    - `allen_ccf_acronym` and `allen_ccf_area` duplicate `location` and
      `structure` with clearer names
    - `unified_atlas_acronym` and `unified_atlas_name` are the finest Unified
      Atlas v2 labels at each unit's CCF coordinates
    """
    ontology = get_unified_atlas_ontology()
    region_ids = get_unified_atlas_region_ids(
        units["ccf_ap"], units["ccf_dv"], units["ccf_ml"]
    )
    labels = [ontology[region_id] for region_id in region_ids.tolist()]
    return units.assign(
        allen_ccf_acronym=units["location"],
        allen_ccf_area=units["structure"],
        unified_atlas_acronym=[acronym for acronym, _ in labels],
        unified_atlas_name=[name for _, name in labels],
    )
