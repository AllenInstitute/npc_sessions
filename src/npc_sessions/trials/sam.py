from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator
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


def _has_finite_values(values: h5py.Dataset) -> bool:
    data = values[()]
    if np.asarray(data).dtype != object:
        return bool(np.any(np.isfinite(data)))
    return any(
        np.any(np.isfinite(np.asarray(value, dtype=float)))
        for value in np.asarray(data, dtype=object).flat
    )


def is_galvo_opto(stim_path_or_data: npc_io.PathLike | h5py.File) -> bool:
    """Whether a stimulus file contains finite galvo coordinates."""
    stim_data = _get_stim_data(stim_path_or_data)
    for key in ("trialGalvoVoltage", "trialGalvoX", "trialGalvoY"):
        values = stim_data.get(key)
        if (
            isinstance(values, h5py.Dataset)
            and values.size
            and _has_finite_values(values)
        ):
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


@contextlib.contextmanager
def _repair_legacy_variable_length_galvo_voltage(
    stim_data: h5py.File,
) -> Iterator[h5py.File]:
    """Give DynRoutData the two-column galvo data it expects.

    Some 670248 stimulus files stored non-opto X/Y pairs as variable-length
    two-value arrays but stored only X for opto trials. The missing Y is
    recoverable when the session records exactly one fixed X/Y coordinate in
    ``galvoVoltage``. Keep the source file unchanged and repair only the
    in-memory copy passed to DynamicRoutingTask.
    """
    trial_galvo_voltage = stim_data.get("trialGalvoVoltage")
    if (
        not isinstance(trial_galvo_voltage, h5py.Dataset)
        or not trial_galvo_voltage.size
        or trial_galvo_voltage.ndim != 1
    ):
        yield stim_data
        return

    galvo_values = tuple(np.asarray(value) for value in trial_galvo_voltage[()])
    if all(value.shape == (2,) for value in galvo_values):
        normalized_galvo_voltage = np.stack(galvo_values)
    else:
        fixed_galvo_voltage = stim_data.get("galvoVoltage")
        if (
            not isinstance(fixed_galvo_voltage, h5py.Dataset)
            or fixed_galvo_voltage.shape != (1, 2)
            or any(value.shape not in ((1,), (2,)) for value in galvo_values)
        ):
            yield stim_data
            return
        normalized_galvo_voltage = np.stack(
            [
                fixed_galvo_voltage[0] if value.shape == (1,) else value
                for value in galvo_values
            ]
        )
    if normalized_galvo_voltage.shape != (trial_galvo_voltage.size, 2):
        yield stim_data
        return

    trial_opto_voltage = stim_data.get("trialOptoVoltage")
    squeeze_trial_opto_voltage = (
        isinstance(trial_opto_voltage, h5py.Dataset)
        and trial_opto_voltage.ndim == 2
        and trial_opto_voltage.shape[1] == 1
    )
    with h5py.File(
        "legacy-variable-length-galvo-repaired.h5",
        "w",
        driver="core",
        backing_store=False,
    ) as repaired:
        for key, value in stim_data.attrs.items():
            repaired.attrs[key] = value
        for key in stim_data:
            if key != "trialGalvoVoltage" and not (
                key == "trialOptoVoltage" and squeeze_trial_opto_voltage
            ):
                stim_data.copy(key, repaired)
        repaired.create_dataset(
            "trialGalvoVoltage",
            data=normalized_galvo_voltage,
        )
        if squeeze_trial_opto_voltage:
            assert isinstance(trial_opto_voltage, h5py.Dataset)
            repaired.create_dataset("trialOptoVoltage", data=trial_opto_voltage[:, 0])
        yield repaired


def get_sam(stim_path_or_data: npc_io.PathLike | h5py.File) -> DynRoutData:
    """Load DynRoutData while keeping legacy opto calibration cloud-compatible.

    DynamicRoutingTask 0.1.106 began loading galvo calibration from a hard-coded
    network path for legacy opto files. npc_sessions already keeps a synchronized
    copy of those files in its data repository, so supply that copy during load.
    """
    stim_data = _get_stim_data(stim_path_or_data)
    with _repair_legacy_variable_length_galvo_voltage(stim_data) as load_data:
        if not _requires_legacy_opto_calibration(load_data):
            return npc_samstim.get_sam(load_data)

        calibration_data = get_bregma_galvo_calibration_data(load_data)
        with _LOAD_SAM_LOCK:
            original = TaskUtils.getBregmaGalvoCalibrationData
            TaskUtils.getBregmaGalvoCalibrationData = lambda _rig: calibration_data
            try:
                return npc_samstim.get_sam(load_data)
            finally:
                TaskUtils.getBregmaGalvoCalibrationData = original
