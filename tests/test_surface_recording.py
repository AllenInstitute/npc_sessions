import npc_sessions


def test_surface_recording_units_use_surface_channels() -> None:
    surface_recording = npc_sessions.Session("703880_2024-04-17").surface_recording

    units = surface_recording.units[:]
    electrodes = surface_recording.electrodes[:]

    assert len(units) > 0
    assert electrodes["channel"].min() == 384
    assert electrodes["channel"].max() == 767
    assert units["peak_channel"].between(384, 767).all()
    assert units["obs_intervals"].map(bool).all()


def test_surface_recording_electrodes_partially_annotated() -> None:
    """Some probes annotated for 768 channels, others only 384: surface
    recording electrodes should fill unannotated channels with sentinels rather
    than raise `ValueError: column 'structure' missing`."""
    session = npc_sessions.Session("834408_2025-12-17")
    electrodes = session.surface_recording.electrodes[:]
    assert electrodes["channel"].between(384, 767).all()
    assert "structure" in electrodes
    unannotated = electrodes["structure"] == "unannotated"
    assert unannotated.any() and not unannotated.all()
    assert electrodes.loc[unannotated, "x"].isna().all()
    assert (session.electrodes[:]["structure"] != "unannotated").all()
