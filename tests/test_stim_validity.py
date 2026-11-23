"""Bottom-up validity of stim blocks (`npc_sessions.qc.get_stim_validity`,
`DynamicRoutingSession.stim_validity`) and what follows from it: `is_task`, trials,
intervals tables, keywords, notes and the epochs table (one row per stim file, `is_qc_pass`).

Synthetic: a session object with stim files and display times made here, the criterion (b),
(c) and (d) checks replaced by fixed results, and stand-ins for the TaskControl classes.
Nothing is read from S3. NWBs are written to and read back from a temp dir.
"""

from __future__ import annotations

import datetime
import json
import pathlib
import types
from typing import Any

import h5py
import numpy as np
import pynwb
import pytest
import upath

import npc_stim
import npc_sessions
import npc_sessions.trials
from npc_sessions import qc

if not hasattr(npc_stim, "FrameIndexCheck"):
    pytest.skip(
        "needs npc_stim with BlockDisplayTimes.frame_index_check", allow_module_level=True
    )

DAY = "20240101"
SUBJECT = "123456"
SYNC_START = datetime.datetime(2024, 1, 1, 12, 0, 0)
TASK = f"DynamicRouting1_{SUBJECT}_{DAY}_121000"
RF_A = f"RFMapping_{SUBJECT}_{DAY}_120100"
RF_B = f"RFMapping_{SUBJECT}_{DAY}_120500"
SPONTANEOUS = f"Spontaneous_{SUBJECT}_{DAY}_123000"
OPTO = f"OptoTagging_{SUBJECT}_{DAY}_124000"
N_FRAMES = 600


# qc.get_stim_validity ----------------------------------------------------------------- #


def frame_check(passed: bool | None = True) -> npc_stim.FrameIndexCheck:
    return npc_stim.FrameIndexCheck(
        passed=passed,
        reason=None if passed else "1 long frame(s) don't match",
        n_long_sync=3,
        n_long_stim=3,
        n_same_index=3,
        n_adjacent_index=0,
        n_unpaired=0,
        n_unmatched=0 if passed else 1,
        max_step=0.0001,
        duration_diff=0.001,
        duration_tolerance=0.005,
        refresh_period=1 / 60,
    )


def make_block(
    stem: str,
    index: int,
    source: str = "photodiode",
    a: bool | None = True,
    start: float | None = None,
) -> npc_stim.BlockDisplayTimes:
    if start is None:
        t = datetime.datetime.strptime(stem.split("_")[-1], "%H%M%S")
        start = (t.hour - 12) * 3600 + t.minute * 60 + t.second + 2.0
    vsyncs = start + np.arange(N_FRAMES) / 60
    constant_lag = source == "constant_lag"
    return npc_stim.BlockDisplayTimes(
        block_index=index,
        stim_path=upath.UPath(f"{stem}.hdf5"),
        vsync_times=vsyncs,
        display_times=vsyncs + (0.02755 if constant_lag else 0.029),
        source=source,  # type: ignore[arg-type]
        reason="10245 diode flips, but 10629 vsyncs" if constant_lag else None,
        frame_index_check=frame_check(a) if constant_lag else None,
    )


class Checks:
    """Fixed results for (b), (c) and (d), and a record of which were run"""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        b: str = "passed",
        c: str | None = None,
        d: bool | None = True,
    ) -> None:
        self.calls: list[tuple[str, str]] = []

        def latency(session: Any, stim: str, **kwargs: Any) -> Any:
            assert kwargs["display_times"] is not None
            self.calls.append(("b", stim))
            return types.SimpleNamespace(
                verdict=b,
                summary=f"visual response latency check {b}: synthetic",
                to_dict=lambda include_arrays=False: {"verdict": b},
            )

        def reference(session: Any, stim: str, **kwargs: Any) -> Any:
            self.calls.append(("c", stim))
            if c is None:
                return None
            return types.SimpleNamespace(
                verdict=c,
                summary=f"photodiode reference latency check {c}: synthetic",
                to_dict=lambda include_arrays=False: {"verdict": c},
            )

        def diode(session: Any, vsync_times: Any) -> Any:
            self.calls.append(("d", ""))
            if d is None:
                return None
            import npc_sync

            verification = getattr(npc_sync, "DiodeVerification", None)
            if verification is None:
                pytest.skip("needs npc_sync with verify_constant_lag")
            return verification(
                passed=d,
                reason="" if d else "photodiode edges are 'rejected' in the sync patch",
                status="cleaned" if d else "rejected",
                n_frames=10,
                n_toggles=10,
                landmarks=5,
                rising_latency_p50_ms=21.2,
                abs_deviation_p99_ms=1.7,
                fraction_over_half_refresh=0.0004,
            )

        monkeypatch.setattr(qc, "quartile_visual_latency", latency)
        monkeypatch.setattr(qc, "reference_visual_latency", reference)
        monkeypatch.setattr(qc, "get_diode_verification", diode)

    @property
    def run(self) -> str:
        return "".join(k for k, _ in self.calls)


SESSION = types.SimpleNamespace(id="123456_2024-01-01", task_stim_name="DynamicRouting1")


def test_photodiode_block_is_valid_without_checks(monkeypatch) -> None:
    checks = Checks(monkeypatch, b="failed")
    v = qc.get_stim_validity(SESSION, TASK, make_block(TASK, 0))  # type: ignore[arg-type]
    assert v.is_valid and v.reason == "" and v.summary == ""
    assert v.display_time_source == "photodiode"
    assert checks.calls == [] and v.checks == {}


@pytest.mark.parametrize("stim", [SPONTANEOUS, OPTO, "SpontaneousRewards_x", "LuminanceTest_x"])
def test_constant_lag_without_visual_stims_is_valid_without_checks(monkeypatch, stim) -> None:
    checks = Checks(monkeypatch, b="failed")
    v = qc.get_stim_validity(SESSION, stim, make_block(stim, 0, "constant_lag", start=0))  # type: ignore[arg-type]
    assert v.is_valid and v.has_visual_stims is False
    assert checks.calls == []
    assert "not checked" in v.summary


@pytest.mark.parametrize(
    "a, b, c, d, valid, run, in_reason",
    [
        (True, "passed", None, True, True, "bc", ""),
        (True, "passed", "passed", True, True, "bc", ""),
        (True, "passed", "untestable", True, True, "bc", ""),
        (False, "passed", None, True, False, "", "(a) frame index check failed"),
        (None, "passed", None, True, False, "", "(a) frame index check not run"),
        (True, "failed", "passed", True, False, "b", "(b) visual response latency check failed"),
        (True, "untestable", None, True, True, "bcd", ""),
        (True, "untestable", None, False, False, "bcd", "(d) photodiode verification failed"),
        (True, "untestable", None, None, False, "bcd", "(d) photodiode verification not available"),
        (True, "passed", "failed", True, False, "bc", "(c) photodiode reference latency check failed"),
        (True, "untestable", "failed", True, False, "bcd", "(c) photodiode reference"),
    ],
)
@pytest.mark.parametrize("stim", [TASK, RF_A])
def test_constant_lag_visual_block(monkeypatch, stim, a, b, c, d, valid, run, in_reason) -> None:
    """valid = (a) and [(b), or (b) untestable and (d)] and (c) doesn't fail"""
    checks = Checks(monkeypatch, b=b, c=c, d=d)
    block = make_block(stim, 0, "constant_lag", a=a)
    if a is None:
        block = types.SimpleNamespace(  # as from an npc_stim without the check
            source="constant_lag", display_times=block.display_times, vsync_times=block.vsync_times, reason="x"
        )
    v = qc.get_stim_validity(SESSION, stim, block)  # type: ignore[arg-type]
    assert v.is_valid is valid
    assert checks.run == run  # nothing is loaded for a block that fails (a)
    assert v.display_time_source == "constant_lag" and v.has_visual_stims
    if valid:
        assert v.reason == "" and v.usable_intervals == ()
        assert v.summary.startswith("constant-lag display times accepted: (a) frame index check passed")
        assert ("(d) photodiode verification passed" in v.summary) == ("d" in run)
    else:
        assert in_reason in v.reason and v.summary == f"invalid: {v.reason}"
        # an RFMapping block keeps its auditory table; a task block keeps nothing
        assert v.usable_intervals == (("aud_rf_mapping_trials",) if stim == RF_A else ())
    assert "\n" not in v.summary
    json.dumps(v.to_dict(), allow_nan=False)
    assert v.to_dict()["is_valid"] is valid


def test_block_without_display_times_is_invalid() -> None:
    block = types.SimpleNamespace(source="unavailable", display_times=None, reason="overlaps a replayed interval")
    v = qc.get_stim_validity(SESSION, TASK, block)  # type: ignore[arg-type]
    assert not v.is_valid and "overlaps a replayed interval" in v.reason
    assert v.display_time_source == "unavailable"


# DynamicRoutingSession ------------------------------------------------------------------ #


def write_stim_file(directory: pathlib.Path, stem: str, rewards: bool = False) -> upath.UPath:
    path = directory / f"{stem}.hdf5"
    with h5py.File(path, "w") as f:
        f["startTime"] = f"{DAY}_{stem.split('_')[-1]}"
        f["frameIntervals"] = np.full(N_FRAMES - 1, 1 / 60)
        f["rewardFrames"] = np.array([10, 20] if rewards else [], dtype=int)
    return upath.UPath(path)


class FakeTrials:
    """Stand-in for a TaskControl class: 3 trials at the start of the block"""

    def __init__(self, hdf5: Any, sync: Any = None, **kwargs: Any) -> None:
        self.path = upath.UPath(hdf5)
        self.display_times = np.asarray(kwargs["vis_display_times"])
        self.trial_index = np.arange(3)

    def keys(self) -> tuple[str, ...]:
        return ("start_time", "stop_time", "trial_index")

    def to_add_trial_column(self) -> tuple[dict[str, str], ...]:
        return ({"name": "trial_index", "description": "index"},)

    def to_add_trial(self) -> tuple[dict[str, Any], ...]:
        t0 = float(self.display_times[0])
        return tuple(
            {"start_time": t0 + i, "stop_time": t0 + i + 0.5, "trial_index": i} for i in range(3)
        )


class FakeSync:
    start_time = SYNC_START
    total_seconds = 3600.0
    invalid_intervals: tuple = ()
    patch = None


class FakeSession(npc_sessions.DynamicRoutingSession):
    """A sync session without ephys or video, from local stim files and given display times"""

    def __init__(
        self,
        stim_paths: list[upath.UPath],
        display_times: dict[str, Any],
        excluded: dict[upath.UPath, str] | None = None,
        **kwargs: Any,
    ) -> None:
        import npc_session

        self.id = npc_session.SessionRecord(f"{SUBJECT}_2024-01-01")
        self.kwargs = kwargs
        for k, v in kwargs.items():
            setattr(self, f"_{k}", v)
        cache = self.__dict__
        cache["info"] = None
        cache["_stim_paths_and_exclusions"] = (tuple(stim_paths), excluded or {})
        cache["stim_paths"] = tuple(stim_paths)
        cache["_stim_display_times"] = display_times
        cache["is_sync"] = True
        cache["is_ephys"] = False
        cache["is_video"] = False
        cache["is_sorted"] = False
        cache["is_annotated"] = False
        cache["is_surface_channels"] = False
        cache["sync_data"] = FakeSync()
        cache["_raw_sync_data"] = FakeSync()
        cache["sync_patch"] = None
        cache["rig"] = "NP3"
        cache["session_type"] = self._SessionType.SURVEY
        cache["subject"] = pynwb.file.Subject(subject_id=SUBJECT, genotype="wt/wt")
        cache["_running_speed"] = pynwb.TimeSeries(
            name="running_speed", data=[0.0, 1.0], unit="cm/s", timestamps=[0.0, 1.0]
        )
        cache["_reward_frame_times"] = pynwb.core.DynamicTable(name="rewards", description="rewards")
        self._is_templeton = False
        self._is_training = False
        self._is_hab = False

    session_start_time = SYNC_START.replace(tzinfo=datetime.timezone.utc)  # type: ignore[assignment]
    experimenter = None  # type: ignore[assignment]
    is_stage_5_passed = False  # type: ignore[assignment]
    source_script_file_name = "test"  # type: ignore[assignment]


@pytest.fixture
def fake_trials(monkeypatch) -> None:
    for name in ("DynamicRouting1", "AudRFMapping", "VisRFMapping", "OptoTagging"):
        monkeypatch.setattr(npc_sessions.trials, name, type(name, (FakeTrials,), {}))


def make_session(tmp_path: pathlib.Path, sources: dict[str, Any], **kwargs: Any) -> FakeSession:
    """`sources`: {stim file stem: 'photodiode' | 'constant_lag' | an Exception}"""
    paths = [write_stim_file(tmp_path, stem, rewards=stem == TASK) for stem in sources]
    display_times = {
        stem: (source if isinstance(source, Exception) else make_block(stem, i, source))
        for i, (stem, source) in enumerate(sources.items())
    }
    return FakeSession(paths, display_times, **kwargs)


def roundtrip(session: FakeSession, tmp_path: pathlib.Path) -> pynwb.NWBFile:
    path = session.write_nwb(tmp_path / "test.nwb", force=True)
    return pynwb.NWBHDF5IO(path.as_posix(), "r").read()


def test_invalid_task_block(tmp_path, monkeypatch, fake_trials) -> None:
    """Constant-lag task block that fails (b): no trials, performance or task keyword; the
    epoch stays, invalid; the rest of the session is untouched; the NWB writes and reads."""
    Checks(monkeypatch, b="failed")
    session = make_session(
        tmp_path, {RF_A: "photodiode", TASK: "constant_lag", SPONTANEOUS: "constant_lag", OPTO: "photodiode"}
    )
    validity = session.stim_validity
    assert list(validity) == [RF_A, TASK, SPONTANEOUS, OPTO]
    assert [v.is_valid for v in validity.values()] == [True, False, True, True]

    assert session.is_task is False
    assert "(b) visual response latency check failed" in (session.task_exclusion_reason or "")
    # AttributeError: npc_sessions_cache's component writer skips the component
    with pytest.raises(AttributeError, match="is invalid"):
        _ = session.trials
    with pytest.raises(AttributeError):
        _ = session.performance
    assert getattr(session, "trials", None) is None
    assert TASK not in session.stim_frame_times
    assert TASK not in session.stim_data_without_timing_issues
    assert list(session._all_trials) == [f"Aud{RF_A}", f"Vis{RF_A}", OPTO]
    assert sorted(session.intervals) == ["aud_rf_mapping_trials", "optotagging_trials", "vis_rf_mapping_trials"]
    assert session._licks is None

    keywords = session.keywords
    assert "task" not in keywords and "task_excluded" in keywords
    assert "invalid" not in keywords and "no_photodiode" in keywords
    assert not any(k in keywords for k in ("first_block_aud", "early_autorewards", "good_behavior"))
    assert f"{TASK} task excluded (no trials or performance tables): " in session.notes
    assert "behavioral task" not in session.session_description

    epochs = session.epochs[:]
    assert list(epochs.script_name) == ["RFMapping", "DynamicRouting1", "Spontaneous", "OptoTagging"]
    assert list(epochs.is_qc_pass) == [True, False, True, True]
    task = epochs.iloc[1]
    assert list(task.interval_names) == []
    assert set(task.tags) == {"task_excluded", "rewards", "invalid", "no_photodiode"}
    assert task.display_time_source == "constant_lag"
    assert task.notes.startswith("invalid: photodiode not usable, and vsync + constant lag display times rejected: (b)")
    # same start/stop as when the block was used: its first and last (constant-lag) display time
    block = session._stim_display_times[TASK]
    assert (task.start_time, task.stop_time) == (block.display_times[0], block.display_times[-1])
    spontaneous = epochs.iloc[2]
    assert "invalid" not in spontaneous.tags and "not checked" in spontaneous.notes

    nwb = roundtrip(session, tmp_path)
    assert nwb.trials is None
    assert "performance" not in nwb.intervals and "trials" not in nwb.intervals
    assert "licks" not in nwb.processing["behavior"].data_interfaces
    assert "quiescent_interval_violations" not in nwb.processing["behavior"].data_interfaces
    read = nwb.epochs[:]
    assert read.is_qc_pass.dtype == bool and list(read.is_qc_pass) == [True, False, True, True]
    assert list(read.interval_names.iloc[1]) == []
    assert "task" not in nwb.keywords[:] and "task_excluded" in nwb.keywords[:]
    assert "task excluded" in nwb.notes


def test_invalid_rf_mapping_block_keeps_its_auditory_table(tmp_path, monkeypatch, fake_trials) -> None:
    """Two RFMapping blocks; the constant-lag one fails (c): the visual table has the other
    block's rows only, the auditory table has both blocks' rows."""
    checks = Checks(monkeypatch, b="passed", c="failed")
    session = make_session(tmp_path, {RF_A: "photodiode", RF_B: "constant_lag", SPONTANEOUS: "photodiode"})
    validity = session.stim_validity
    assert checks.run == "bc" and {stim for _, stim in checks.calls} == {RF_B}
    assert not validity[RF_B].is_valid and validity[RF_B].usable_intervals == ("aud_rf_mapping_trials",)
    assert list(session._all_trials) == [f"Aud{RF_A}", f"Vis{RF_A}", f"Aud{RF_B}"]
    intervals = session.intervals
    assert len(intervals["vis_rf_mapping_trials"]) == 3
    assert len(intervals["aud_rf_mapping_trials"]) == 6
    # trial_index continues across the blocks of the auditory table
    assert list(intervals["aud_rf_mapping_trials"][:].trial_index) == [0, 1, 2, 3, 4, 5]
    vis_times = intervals["vis_rf_mapping_trials"][:].start_time
    b_start = session._stim_display_times[RF_B].display_times[0]
    assert (vis_times < b_start).all()

    epochs = session.epochs[:]
    assert list(epochs.is_qc_pass) == [True, False, True]
    assert list(epochs.interval_names.iloc[0]) == ["vis_rf_mapping_trials", "aud_rf_mapping_trials"]
    assert list(epochs.interval_names.iloc[1]) == ["aud_rf_mapping_trials"]
    assert {"invalid", "mapping", "no_photodiode"} <= set(epochs.tags.iloc[1])
    assert "(c) photodiode reference latency check failed" in epochs.notes.iloc[1]
    assert f"{RF_B} visual stimuli excluded: " in session.notes

    nwb = roundtrip(session, tmp_path)
    assert len(nwb.intervals["vis_rf_mapping_trials"]) == 3
    assert len(nwb.intervals["aud_rf_mapping_trials"]) == 6


def test_only_rf_mapping_block_invalid_no_visual_table(tmp_path, monkeypatch, fake_trials) -> None:
    Checks(monkeypatch, b="untestable", d=False)
    session = make_session(tmp_path, {RF_A: "constant_lag", SPONTANEOUS: "photodiode"})
    assert sorted(session.intervals) == ["aud_rf_mapping_trials"]
    nwb = roundtrip(session, tmp_path)
    assert "vis_rf_mapping_trials" not in nwb.intervals
    assert list(nwb.epochs[:].is_qc_pass) == [False, True]


def test_valid_constant_lag_block_has_checks_in_notes(tmp_path, monkeypatch, fake_trials) -> None:
    Checks(monkeypatch, b="untestable", d=True)
    session = make_session(tmp_path, {RF_A: "constant_lag", SPONTANEOUS: "photodiode"})
    assert session.stim_validity[RF_A].is_valid
    epochs = session.epochs[:]
    assert list(epochs.is_qc_pass) == [True, True]
    notes = epochs.notes.iloc[0]
    assert notes.startswith("frame display times are vsync + constant lag (27.55 ms), photodiode not used: ")
    for text in ("constant-lag display times accepted: (a) frame index check passed", "(b) visual response",
                 "(d) photodiode verification passed"):
        assert text in notes
    assert set(epochs.tags.iloc[0]) == {"mapping", "no_photodiode"}
    assert list(epochs.interval_names.iloc[0]) == ["vis_rf_mapping_trials", "aud_rf_mapping_trials"]
    assert session.notes is None
    assert len(session.intervals["vis_rf_mapping_trials"]) == 3


def test_photodiode_session_runs_no_checks(tmp_path, monkeypatch, fake_trials) -> None:
    checks = Checks(monkeypatch, b="failed")
    session = make_session(tmp_path, {RF_A: "photodiode", SPONTANEOUS: "photodiode", OPTO: "photodiode"})
    epochs = session.epochs[:]
    assert checks.calls == []
    assert list(epochs.is_qc_pass) == [True, True, True]
    assert list(epochs.notes) == ["", "", ""]
    assert list(epochs.display_time_source) == ["photodiode"] * 3
    # start/stop are the first and last display time, unchanged
    for (_, row), stem in zip(epochs.iterrows(), (RF_A, SPONTANEOUS, OPTO)):
        times = session._stim_display_times[stem].display_times
        assert (row.start_time, row.stop_time) == (times[0], times[-1])
        assert session.stim_frame_times[stem] is times


def test_every_stim_file_has_an_epoch_and_all_invalid_nwb_writes(tmp_path, monkeypatch, fake_trials) -> None:
    """Stims without frame times (no sync match), an excluded stim file and a constant-lag
    block that fails (a): every one has an invalid row with a reason and start/stop times;
    every `interval_names` is empty, and the NWB still writes."""
    checks = Checks(monkeypatch)
    excluded = write_stim_file(tmp_path, f"DynamicRouting1_{SUBJECT}_{DAY}_125000")
    session = make_session(
        tmp_path,
        {
            RF_A: IndexError("Frame count mismatch: block 0 has 590 vsyncs, stim file has 600 frames"),
            TASK: "constant_lag",
            SPONTANEOUS: ValueError("No frame display times: block overlaps a replayed sync interval"),
        },
    )
    session.__dict__["_stim_paths_and_exclusions"] = (session.stim_paths, {excluded: "more than one task stim file"})
    session.__dict__["_stim_display_times"][TASK] = make_block(TASK, 1, "constant_lag", a=False)
    assert checks.calls == []
    assert [v.is_valid for v in session.stim_validity.values()] == [False] * 4
    assert session.is_task is False
    assert session.stim_frame_times == {} and list(session._all_trials) == []
    assert len(session.intervals) == 0

    epochs = session.epochs[:]
    assert list(epochs.script_name) == ["RFMapping", "DynamicRouting1", "Spontaneous", "DynamicRouting1"]
    assert not epochs.is_qc_pass.any()
    assert all("invalid" in tags for tags in epochs.tags)
    assert all(len(names) == 0 for names in epochs.interval_names)
    assert list(epochs.display_time_source) == ["unavailable", "constant_lag", "unavailable", "unavailable"]
    assert ["timing_issues" in tags for tags in epochs.tags] == [True, False, True, True]
    assert "Frame count mismatch" in epochs.notes.iloc[0]
    assert "(a) frame index check failed" in epochs.notes.iloc[1]
    assert "more than one task stim file" in epochs.notes.iloc[3]
    # no vsync block to take times from here (the fake sync has none): the stim file's start
    # time on the sync clock, and its duration
    assert "from the stim file's start time and duration" in epochs.notes.iloc[0]
    assert epochs.start_time.iloc[0] == 60.0
    assert epochs.stop_time.iloc[0] == pytest.approx(60.0 + (N_FRAMES - 1) / 60)
    assert epochs.start_time.iloc[3] == 50 * 60.0
    assert (epochs.stop_time > epochs.start_time).all()
    assert len(session.notes.split("; ")) >= 4

    nwb = roundtrip(session, tmp_path)
    read = nwb.epochs[:]
    assert len(read) == 4 and not read.is_qc_pass.any()
    assert all(len(names) == 0 for names in read.interval_names)
    assert nwb.trials is None and set(nwb.intervals) == {"epochs"}
    assert "task_excluded" in nwb.keywords[:] and "task" not in nwb.keywords[:]


def test_epoch_times_from_the_vsync_block(tmp_path, monkeypatch, fake_trials) -> None:
    """A stim without frame times takes its epoch times from the vsync block that starts
    closest to the stim file's start time, unless another stim has that block."""
    Checks(monkeypatch)
    session = make_session(tmp_path, {RF_A: ValueError("no display times"), SPONTANEOUS: "photodiode"})
    spontaneous = session._stim_display_times[SPONTANEOUS]
    blocks = [61.5 + np.arange(N_FRAMES) / 60, spontaneous.vsync_times]
    assert spontaneous.block_index == 1
    import npc_sync  # the real module: `sessions.npc_sync` is a lazy proxy

    monkeypatch.setattr(
        npc_sync,
        "SyncDataset",
        lambda *args, **kwargs: types.SimpleNamespace(vsync_times_in_blocks=blocks),
    )
    epochs = session.epochs[:]
    assert (epochs.start_time.iloc[0], epochs.stop_time.iloc[0]) == (blocks[0][0], blocks[0][-1])
    assert "first/last vsync of sync block 0" in epochs.notes.iloc[0]
    assert list(epochs.is_qc_pass) == [False, True]


def test_task_disabled_by_config(tmp_path, monkeypatch, fake_trials) -> None:
    """`is_task=False`: the task stim is invalid without running checks, and there's no
    'task' tag or keyword."""
    checks = Checks(monkeypatch)
    session = make_session(tmp_path, {TASK: "constant_lag", SPONTANEOUS: "photodiode"}, is_task=False)
    assert session.is_task is False and checks.calls == []
    assert "is_task=False" in session.stim_validity[TASK].reason
    assert list(session._all_trials) == []
    epochs = session.epochs[:]
    assert list(epochs.is_qc_pass) == [False, True]
    assert "task" not in session.epoch_tags and "task" not in session.keywords
    assert "task_excluded" in session.keywords


def test_stim_validity_cannot_depend_on_itself(tmp_path, monkeypatch) -> None:
    session = make_session(tmp_path, {TASK: "constant_lag"})
    monkeypatch.setattr(qc, "quartile_visual_latency", lambda s, stim, **kw: s.is_task)
    with pytest.raises(RuntimeError, match="needs itself"):
        _ = session.stim_validity
