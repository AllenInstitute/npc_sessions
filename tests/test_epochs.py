from pathlib import Path

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
    session.stim_data = {
        path.stem: {"rewardFrames": []} for path in session.stim_paths
    }
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
