import npc_sessions


def test_surface_recording_units_use_surface_channels() -> None:
    surface_recording = npc_sessions.Session("703880_2024-04-17").surface_recording

    units = surface_recording.units[:]
    electrodes = surface_recording.electrodes[:]

    assert len(units) > 0
    assert electrodes["channel"].min() == 385
    assert electrodes["channel"].max() == 768
    assert units["peak_channel"].between(385, 768).all()
    assert units["obs_intervals"].map(bool).all()
