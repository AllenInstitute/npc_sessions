from pathlib import Path
from types import SimpleNamespace

import pytest

import npc_sessions.sessions as sessions


def test_epochs_only_reference_existing_taskcontrol_interval_tables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sessions.npc_samstim, "is_opto", lambda h5: False)
    monkeypatch.setattr(sessions.npc_samstim, "is_galvo_opto", lambda h5: False)
    monkeypatch.setattr(sessions.npc_stim, "get_stim_duration", lambda h5: 1.0)

    session = sessions.DynamicRoutingSession.__new__(sessions.DynamicRoutingSession)
    session.task_stim_name = "DynamicRouting1"
    session.is_sync = False
    session.is_task = True
    session.is_opto = False
    session.is_wildtype = False
    session.invalid_times = None
    session.stim_paths = [
        Path("Spontaneous_123.hdf5"),
        Path("SpontaneousRewards_123.hdf5"),
        Path("DynamicRouting1_123.hdf5"),
    ]
    session.stim_data = {path.stem: {"rewardFrames": []} for path in session.stim_paths}
    session.stim_data_without_timing_issues = {
        path.stem: session.stim_data[path.stem] for path in session.stim_paths
    }

    epochs = session.epochs.to_dataframe()
    interval_names_by_script = dict(
        zip(epochs["script_name"], epochs["interval_names"], strict=True)
    )

    assert interval_names_by_script["Spontaneous"] == []
    assert interval_names_by_script["SpontaneousRewards"] == []
    assert interval_names_by_script["DynamicRouting1"] == ["trials", "performance"]


def test_stim_frame_times_records_whole_session_timing_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exc = AssertionError("unmapped sync block")
    monkeypatch.setattr(
        sessions.npc_stim,
        "get_stim_frame_times",
        lambda *args, **kwargs: (_ for _ in ()).throw(exc),
    )

    session = sessions.DynamicRoutingSession.__new__(sessions.DynamicRoutingSession)
    session.stim_paths = [
        Path("DynamicRouting1_123.hdf5"),
        Path("RFMapping_123.hdf5"),
    ]
    session.sync_path = Path("session.sync")

    frame_times = session._stim_frame_times

    assert frame_times == {
        "DynamicRouting1_123": exc,
        "RFMapping_123": exc,
    }


def test_stim_data_without_timing_issues_tolerates_whole_session_timing_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sessions.npc_stim,
        "get_stim_frame_times",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError),
    )

    session = sessions.DynamicRoutingSession.__new__(sessions.DynamicRoutingSession)
    session.is_sync = True
    session.stim_paths = [
        Path("DynamicRouting1_123.hdf5"),
        Path("RFMapping_123.hdf5"),
    ]
    session.sync_path = Path("session.sync")
    session.stim_data = {path.stem: object() for path in session.stim_paths}

    assert session.stim_data_without_timing_issues == {}


def test_epochs_skip_sync_stims_with_unrecoverable_timing_failure() -> None:
    session = sessions.DynamicRoutingSession.__new__(sessions.DynamicRoutingSession)
    session.is_sync = True
    session.stim_paths = [
        Path("DynamicRouting1_123.hdf5"),
        Path("RFMapping_123.hdf5"),
    ]
    session.stim_data_without_timing_issues = {}

    assert len(session.epochs) == 0


def test_unsynced_ephys_timing_uses_pxi_timing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timings = (
        SimpleNamespace(device=SimpleNamespace(name="Neuropix-PXI-100.ProbeA-AP")),
        SimpleNamespace(device=SimpleNamespace(name="NI-DAQmx-101.PXI-6133")),
    )
    monkeypatch.setattr(
        sessions.npc_ephys,
        "get_ephys_timing_on_pxi",
        lambda recording_dirs: timings,
    )

    session = sessions.DynamicRoutingSession.__new__(sessions.DynamicRoutingSession)
    session.is_sync = False
    session.ephys_recording_dirs = (Path("Record Node 1/experiment1/recording1"),)
    session.probe_letters_to_use = ("A",)

    assert session.ephys_timing_data == timings
