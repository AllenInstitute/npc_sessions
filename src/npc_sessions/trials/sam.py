from __future__ import annotations

import threading
from typing import Any

import h5py
import npc_io
import npc_lims
import npc_samstim
import npc_stim
import numpy as np
from DynamicRoutingTask import TaskUtils
from DynamicRoutingTask.Analysis.DynamicRoutingAnalysisUtils import DynRoutData

_LOAD_SAM_LOCK = threading.Lock()


def _get_stim_data(stim_path_or_data: npc_io.PathLike | h5py.File) -> h5py.File:
    stim_data = npc_stim.get_stim_data(stim_path_or_data)
    if not isinstance(stim_data, h5py.File):
        raise TypeError(f"Expected h5py.File, got {type(stim_data)}")
    return stim_data


def is_opto(stim_path_or_data: npc_io.PathLike | h5py.File) -> bool:
    """Whether a stimulus file contains at least one optogenetic trial."""
    stim_data = _get_stim_data(stim_path_or_data)
    onset_frames = stim_data.get("trialOptoOnsetFrame")
    return bool(
        onset_frames is not None
        and onset_frames.size
        and np.any(np.isfinite(onset_frames[()]))
    )


def is_galvo_opto(stim_path_or_data: npc_io.PathLike | h5py.File) -> bool:
    """Whether a stimulus file contains finite galvo coordinates."""
    stim_data = _get_stim_data(stim_path_or_data)
    for key in ("trialGalvoVoltage", "trialGalvoX", "trialGalvoY"):
        values = stim_data.get(key)
        if values is not None and values.size and np.any(np.isfinite(values[()])):
            return True
    return False


def _txt_to_dict(txt: str) -> dict[str, list[float]]:
    rows = [line.split("\t") for line in txt.splitlines() if line]
    return {
        values[0]: [float(value) for value in values[1:]]
        for values in zip(*rows, strict=False)
    }


def get_bregma_galvo_calibration_data(
    stim_path_or_data: npc_io.PathLike | h5py.File,
) -> dict[str, Any]:
    """Get embedded calibration data, falling back to the cloud-backed copy."""
    stim_data = _get_stim_data(stim_path_or_data)
    if (embedded := stim_data.get("bregmaGalvoCalibrationData")) is not None:
        if not isinstance(embedded, h5py.Group):
            raise TypeError(
                "Expected bregmaGalvoCalibrationData to be an HDF5 group, "
                f"got {type(embedded)}"
            )
        return {key: value[()] for key, value in embedded.items()}

    rig = stim_data["rigName"].asstr()[()]
    root = npc_lims.DR_DATA_REPO.parent / "OptoGui" / rig
    return _txt_to_dict((root / f"{rig}_bregma_galvo.txt").read_text())


def _requires_legacy_opto_calibration(stim_data: h5py.File) -> bool:
    """Whether newer DynRoutData will try to read its hard-coded UNC path."""
    if isinstance(stim_data.get("optoParams"), h5py.Group):
        return False
    if (regions := stim_data.get("optoRegions")) is not None and regions.size:
        return True
    if (probability := stim_data.get("optoProb")) is not None:
        return bool(probability[()] > 0)
    return is_opto(stim_data)


def get_sam(stim_path_or_data: npc_io.PathLike | h5py.File) -> DynRoutData:
    """Load DynRoutData while keeping legacy opto calibration cloud-compatible.

    DynamicRoutingTask 0.1.106 began loading galvo calibration from a hard-coded
    network path for legacy opto files. npc_sessions already keeps a synchronized
    copy of those files in its data repository, so supply that copy during load.
    """
    stim_data = _get_stim_data(stim_path_or_data)
    if not _requires_legacy_opto_calibration(stim_data):
        return npc_samstim.get_sam(stim_data)

    calibration_data = get_bregma_galvo_calibration_data(stim_data)
    with _LOAD_SAM_LOCK:
        original = TaskUtils.getBregmaGalvoCalibrationData
        TaskUtils.getBregmaGalvoCalibrationData = lambda _rig: calibration_data
        try:
            return npc_samstim.get_sam(stim_data)
        finally:
            TaskUtils.getBregmaGalvoCalibrationData = original
