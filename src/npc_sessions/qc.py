"""Acceptance check for visual stim times that aren't measured with the photodiode.

Where a stim block's frame display times are vsync + constant lag
(`display_time_source == 'constant_lag'`), nothing on sync shows whether the display kept up:
in 3 of the 10 task blocks with constant lag in v0.0.301 the display ran 1-2 frames late from
partway through the block, although vsync and the stim file's `frameIntervals` agreed (hub
experiments/constant-lag-qc). Neural responses do show it: the latency of the population visual
response, relative to the reported stim onsets, steps by the display delay.

Criterion (b) for accepting such a block (`quartile_latency_check`):

- units: those passing `UNIT_QUERY`, with a fast response to the block's visual stims (rate
  increase >= `MIN_RATE_INCREASE_HZ` and mean / SEM >= `MIN_SCORE` in `RESPONSE_WINDOW`, vs
  `BASELINE_WINDOW`), in any structure
- the block's visual stim onsets are split, in time order, into 4 groups with equal numbers of
  stims ("quartiles")
- per quartile: population PSTH (mean over units of the baseline-subtracted, peak-normalised
  PSTH; 1-ms bins, 5-ms boxcar) and its half-max latency
- verdict, from the point estimates only:
    - `passed`: range of the 4 latencies <= `MAX_LATENCY_RANGE_MS`, with >=
      `MIN_RESPONSIVE_UNITS` responsive units
    - `failed`: range > `MAX_LATENCY_RANGE_MS`
    - `untestable`: too few responsive units or stims, or a latency is undefined

Also reported, not used for the verdict: bootstrap CIs of the latencies (fixed seed), the same
for auditory stims (control: sound onsets don't use display times) and a per-stim
maximum-likelihood onset shift (template: the first quartile's stims).

`quartile_visual_latency(session, block, display_times=...)` runs the check for a stim block
of a `DynamicRoutingSession`. It uses the stim file, the candidate display times and the sorted
units only: not `session.trials`, `session.epochs`, `session.intervals` or `session._units`
(which needs `trials`), so it can be called while display times are being decided.

The method is deterministic: the same inputs give the same result, including the CIs.

Limit: quarters are compared within a block, so a delay that is there from the start of the
block isn't detected (hub experiments/constant-lag-acceptance: 666986_2023-08-14 RFMapping).

Criterion (c) covers that where it can (`reference_latency_check`): if the session has a
block of the same stim with photodiode display times, the whole-block latency of the same
units must be within `MAX_REFERENCE_LATENCY_DIFF_MS` of it.

`get_stim_validity(session, stim, block)` combines the checks into the validity of a stim
block (`StimValidity`), which `DynamicRoutingSession.stim_validity` records per stim file:
a block with visual stims and constant-lag display times is valid if

- (a) the long frames in its vsyncs match the stim file's
  (`npc_stim.BlockDisplayTimes.frame_index_check`), and
- (b) passes, or (b) is untestable and (d) the photodiode edges, as aligned to the frames
  for the sync patch, verify a constant lag (`npc_sync.verify_constant_lag`), and
- (c) doesn't fail.

Blocks with photodiode display times aren't checked. Blocks without visual stims timed by
display times (OptoTagging, Spontaneous, SpontaneousRewards, LuminanceTest: their intervals
use sync lines, the NI-DAQ or vsyncs) are valid with constant lag.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import numpy as np
import numpy.typing as npt
import scipy.ndimage

if TYPE_CHECKING:
    import pandas as pd
    import upath

    import npc_sessions

logger = logging.getLogger(__name__)

Verdict = Literal["passed", "failed", "untestable"]
Modality = Literal["vis", "aud"]

UNIT_QUERY = (
    "amplitude_cutoff < 0.1 & isi_violations_ratio < 0.5 & presence_ratio > 0.9"
)
"""Units used for the check (as `npc_sessions_cache.plots.responses.UNIT_QUERY`)"""
RESPONSE_WINDOW: dict[Modality, tuple[float, float]] = {
    "vis": (0.02, 0.10),
    "aud": (0.005, 0.05),
}
"""Window after stim onset (s) for selecting responsive units: short, to favour fast responses"""
BASELINE_WINDOW = (-0.1, 0.0)
MIN_RATE_INCREASE_HZ = 5.0
MIN_SCORE = 5.0
"""Mean / SEM of the per-stim (response - baseline) rate difference"""
PSTH_BIN = 0.001
PSTH_WINDOW = (-0.1, 0.2)
SMOOTH_BINS = 5
LATENCY_WINDOW = (0.0, 0.15)
"""The PSTH peak and its half-max crossing are searched in this window after stim onset (s)"""
N_QUARTILES = 4
MAX_LATENCY_RANGE_MS = 8.0
"""Largest accepted range of the visual half-max latency over quartiles (half a 60 Hz frame)"""
MIN_RESPONSIVE_UNITS = 5
MIN_STIMS_PER_QUARTILE = 10
N_BOOTSTRAP = 200
BOOTSTRAP_SEED = 0
SHIFTS = np.arange(-0.05, 0.0505, 0.001)
"""Candidate onset shifts (s) for the per-stim shift estimate"""
SHIFT_WINDOW = (-0.05, 0.2)
"""Spikes in this window after stim onset are used for the per-stim shift estimate"""
MAX_STIM_SHIFT_MS = 8.0

_EDGES = np.arange(PSTH_WINDOW[0], PSTH_WINDOW[1] + PSTH_BIN, PSTH_BIN)
_CENTERS = _EDGES[:-1] + PSTH_BIN / 2
_IS_BASELINE = _CENTERS < 0
_IS_LATENCY_WINDOW = (_CENTERS >= LATENCY_WINDOW[0]) & (_CENTERS < LATENCY_WINDOW[1])


# results -------------------------------------------------------------------------- #


def _jsonable(value: Any) -> Any:  # noqa: ANN401
    """Python scalars / lists, with None for nan (valid JSON)"""
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


@dataclasses.dataclass(frozen=True)
class QuartileStats:
    """One quartile of a block's stims (of one modality)."""

    index: int
    start_time: float
    """Onset of the quartile's first stim (s)"""
    stop_time: float
    """Onset of the quartile's last stim (s)"""
    n_trials: int
    n_units: int
    """Responsive units with a positive PSTH peak in this quartile"""
    latency_ms: float
    """Half-max latency of the population PSTH; nan if undefined"""
    ci_low_ms: float
    ci_high_ms: float
    """95% bootstrap CI (stims resampled); nan if not computed"""
    psth: npt.NDArray[np.float64] = dataclasses.field(repr=False, compare=False)
    """Population PSTH (mean of peak-normalised units), at `ModalityStats.psth_time`"""

    def to_dict(self, include_arrays: bool = False) -> dict[str, Any]:
        d = {
            f.name: _jsonable(getattr(self, f.name))
            for f in dataclasses.fields(self)
            if include_arrays or f.name != "psth"
        }
        return d


@dataclasses.dataclass(frozen=True)
class ModalityStats:
    modality: Modality
    n_trials: int
    n_units_tested: int
    n_units: int
    """Responsive units"""
    unit_ids: tuple[str, ...]
    """Responsive units: keys of `spike_times`, or indices as str"""
    quartiles: tuple[QuartileStats, ...]
    """Empty if there were too few stims or no responsive units"""
    latency_range_ms: float
    """max - min of the quartiles' half-max latencies; nan if any is undefined"""
    psth_time: npt.NDArray[np.float64] = dataclasses.field(repr=False, compare=False)
    """Bin centres (s from stim onset) of the quartiles' PSTHs"""

    @property
    def latencies_ms(self) -> tuple[float, ...]:
        return tuple(q.latency_ms for q in self.quartiles)

    def to_dict(self, include_arrays: bool = False) -> dict[str, Any]:
        d: dict[str, Any] = {
            "modality": self.modality,
            "n_trials": self.n_trials,
            "n_units_tested": self.n_units_tested,
            "n_units": self.n_units,
            "latency_range_ms": _jsonable(self.latency_range_ms),
            "quartiles": [q.to_dict(include_arrays) for q in self.quartiles],
        }
        if include_arrays:
            d["unit_ids"] = list(self.unit_ids)
            d["psth_time"] = _jsonable(self.psth_time)
        return d


@dataclasses.dataclass(frozen=True)
class StimShiftStats:
    """Per-stim maximum-likelihood onset shift of the visual response (template: quartile 1)."""

    n_trials: int
    n_template_trials: int
    frac_shifted: float
    """Fraction of stims with |shift| > `max_shift_ms`"""
    frac_shifted_by_quartile: tuple[float, ...]
    median_shift_ms_by_quartile: tuple[float, ...]
    max_shift_ms: float
    onsets: npt.NDArray[np.float64] = dataclasses.field(repr=False, compare=False)
    shift_ms: npt.NDArray[np.float64] = dataclasses.field(repr=False, compare=False)
    """Positive: the response is later than the template's, i.e. the stim was displayed
    later than its reported onset; nan where no responsive unit covers the stim"""

    def to_dict(self, include_arrays: bool = False) -> dict[str, Any]:
        return {
            f.name: _jsonable(getattr(self, f.name))
            for f in dataclasses.fields(self)
            if include_arrays or f.name not in ("onsets", "shift_ms")
        }


@dataclasses.dataclass(frozen=True)
class QuartileLatencyResult:
    """Result of `quartile_latency_check`."""

    verdict: Verdict
    reason: str
    vis: ModalityStats
    aud: ModalityStats | None
    """Control; None if no auditory onsets were given"""
    stim_shifts: StimShiftStats | None
    """None if untestable, or not requested"""
    params: dict[str, Any]
    block: str | None = None

    @property
    def passed(self) -> bool:
        return self.verdict == "passed"

    @property
    def vis_range_ms(self) -> float:
        return self.vis.latency_range_ms

    @property
    def aud_range_ms(self) -> float:
        return np.nan if self.aud is None else self.aud.latency_range_ms

    @property
    def frac_shifted(self) -> float:
        return np.nan if self.stim_shifts is None else self.stim_shifts.frac_shifted

    @property
    def summary(self) -> str:
        """One line, for the notes of an epoch"""
        return f"visual response latency check {self.verdict}: {self.reason}"

    def to_dict(self, include_arrays: bool = False) -> dict[str, Any]:
        """JSON-serialisable. `include_arrays`: also the PSTHs, unit ids and per-stim
        shifts (for plots)."""
        return {
            "block": self.block,
            "verdict": self.verdict,
            "reason": self.reason,
            "summary": self.summary,
            "vis_range_ms": _jsonable(self.vis_range_ms),
            "aud_range_ms": _jsonable(self.aud_range_ms),
            "frac_shifted": _jsonable(self.frac_shifted),
            "vis": self.vis.to_dict(include_arrays),
            "aud": None if self.aud is None else self.aud.to_dict(include_arrays),
            "stim_shifts": (
                None
                if self.stim_shifts is None
                else self.stim_shifts.to_dict(include_arrays)
            ),
            "params": _jsonable(self.params),
        }


# core ----------------------------------------------------------------------------- #


def _counts_in_window(
    spike_times: npt.NDArray[np.floating],
    onsets: npt.NDArray[np.floating],
    window: tuple[float, float],
) -> npt.NDArray[np.int64]:
    return np.searchsorted(spike_times, onsets + window[1]) - np.searchsorted(
        spike_times, onsets + window[0]
    )


def _binned_counts(
    spike_times: npt.NDArray[np.floating],
    onsets: npt.NDArray[np.floating],
    edges: npt.NDArray[np.floating],
) -> npt.NDArray[np.uint16]:
    """Spike counts, stims x bins (bins as `np.histogram`)"""
    n_bins = len(edges) - 1
    lo = np.searchsorted(spike_times, onsets + edges[0])
    hi = np.searchsorted(spike_times, onsets + edges[-1])
    n = hi - lo
    total = int(n.sum())
    if total == 0:
        return np.zeros((len(onsets), n_bins), dtype=np.uint16)
    stim = np.repeat(np.arange(len(onsets)), n)
    first = (
        np.cumsum(n) - n
    )  # index in the concatenated spikes of each stim's first spike
    spike_idx = np.arange(total) - np.repeat(first, n) + np.repeat(lo, n)
    aligned = spike_times[spike_idx] - onsets[stim]
    bin_idx = np.searchsorted(edges, aligned, side="right") - 1
    bin_idx[aligned == edges[-1]] = n_bins - 1
    ok = (bin_idx >= 0) & (bin_idx < n_bins)
    counts = np.bincount(
        stim[ok] * n_bins + bin_idx[ok], minlength=len(onsets) * n_bins
    )
    return counts.reshape(len(onsets), n_bins).astype(np.uint16)


def _smooth(x: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
    """Boxcar over the last axis, zero-padded (as `np.convolve(x, ones(k) / k, 'same')`)"""
    return scipy.ndimage.uniform_filter1d(
        np.asarray(x, dtype=float), SMOOTH_BINS, axis=-1, mode="constant"
    )


def _population_psth(
    rates: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Mean over units (axis -2) of the smoothed, baseline-subtracted PSTH divided by its
    peak in `LATENCY_WINDOW`; units without a positive peak are left out. Returns the
    population PSTH(s) and the number of units used."""
    s = _smooth(rates)
    s = s - s[..., _IS_BASELINE].mean(axis=-1, keepdims=True)
    peak = s[..., _IS_LATENCY_WINDOW].max(axis=-1, keepdims=True)
    ok = np.isfinite(peak) & (peak > 0)
    n = ok.sum(axis=-2)
    normed = np.where(ok, s / np.where(ok, peak, 1.0), 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        pop = normed.sum(axis=-2) / n
    return pop, n[..., 0]


def _half_max_latency(population_psth: npt.NDArray[np.floating]) -> float:
    """Time (s) at which the smoothed, baseline-subtracted population PSTH first reaches half
    of its peak in `LATENCY_WINDOW` (linear interpolation between bins); nan if no peak.
    """
    if not np.all(np.isfinite(population_psth)):
        return np.nan
    s = _smooth(population_psth)
    s = s - s[_IS_BASELINE].mean()
    peak = s[_IS_LATENCY_WINDOW].max()
    if not peak > 0:
        return np.nan
    i = int(np.flatnonzero(_IS_LATENCY_WINDOW & (s >= peak / 2))[0])
    x0, x1, y0, y1 = _CENTERS[i - 1], _CENTERS[i], s[i - 1], s[i]
    if y1 == y0:
        return float(x1)
    return float(x0 + (peak / 2 - y0) * (x1 - x0) / (y1 - y0))


@dataclasses.dataclass
class _Unit:
    """A responsive unit: the stims it is tested on, and its spike counts around them"""

    unit_id: str
    spike_times: npt.NDArray[np.float64]
    stim_idx: npt.NDArray[np.intp]
    """Sorted indices in the block's onsets (all of them, unless conditions were given)"""
    counts: npt.NDArray[np.uint16]
    """len(stim_idx) x PSTH bins"""


def _responsive_units(
    spike_times: Mapping[str, npt.NDArray[np.floating]],
    onsets: npt.NDArray[np.float64],
    modality: Modality,
    conditions: npt.NDArray[np.intp] | None,
    min_rate_increase_hz: float,
    min_score: float,
) -> list[_Unit]:
    """Units with a fast response to the stims, or (with `conditions`) to the stims of at
    least one condition: then a unit is tested on the stims of its responsive conditions.
    """
    window = RESPONSE_WINDOW[modality]
    if conditions is None:
        masks = [np.ones(len(onsets), dtype=bool)]
    else:
        masks = [conditions == c for c in np.unique(conditions)]
        masks = [m for m in masks if m.sum() >= 2]
    units: list[_Unit] = []
    for unit_id, st in spike_times.items():
        st = np.asarray(st, dtype=float)
        if st.size == 0:
            continue
        diff = _counts_in_window(st, onsets, window) / (window[1] - window[0])
        diff = diff - _counts_in_window(st, onsets, BASELINE_WINDOW) / (
            BASELINE_WINDOW[1] - BASELINE_WINDOW[0]
        )
        responsive = np.zeros(len(onsets), dtype=bool)
        for mask in masks:
            d = diff[mask]
            sem = d.std(ddof=1) / np.sqrt(d.size)
            if (
                d.mean() >= min_rate_increase_hz
                and sem > 0
                and d.mean() / sem >= min_score
            ):
                responsive |= mask
        if responsive.any():
            stim_idx = np.flatnonzero(responsive)
            units.append(
                _Unit(
                    unit_id=str(unit_id),
                    spike_times=st,
                    stim_idx=stim_idx,
                    counts=_binned_counts(st, onsets[stim_idx], _EDGES),
                )
            )
    return units


def _modality_stats(
    spike_times: Mapping[str, npt.NDArray[np.floating]],
    onsets: npt.NDArray[np.float64],
    modality: Modality,
    conditions: npt.NDArray[np.intp] | None,
    n_quartiles: int,
    min_stims_per_quartile: int,
    min_rate_increase_hz: float,
    min_score: float,
    n_bootstrap: int,
    seed: Sequence[int],
) -> tuple[ModalityStats, list[_Unit]]:
    enough_stims = len(onsets) >= n_quartiles * max(min_stims_per_quartile, 1)
    units = (
        _responsive_units(
            spike_times, onsets, modality, conditions, min_rate_increase_hz, min_score
        )
        if enough_stims
        else []
    )
    quartiles: list[QuartileStats] = []
    if units:
        bounds = [
            a[[0, -1]] for a in np.array_split(np.arange(len(onsets)), n_quartiles)
        ]
        for k, (first, last) in enumerate(bounds):
            n = last - first + 1
            selections = [(u.stim_idx >= first) & (u.stim_idx <= last) for u in units]
            with np.errstate(invalid="ignore", divide="ignore"):
                rates = np.array(
                    [
                        u.counts[sel].sum(axis=0) / sel.sum() / PSTH_BIN
                        for u, sel in zip(units, selections, strict=True)
                    ]
                )
            pop, n_units = _population_psth(rates)
            # bootstrap: resample the quartile's stims (weights = number of times drawn)
            rng = np.random.default_rng([*seed, k])
            boot: list[float] = []
            for start in range(0, n_bootstrap, 50):
                n_rep = min(50, n_bootstrap - start)
                weights = np.zeros((n_rep, n))
                draws = rng.integers(0, n, size=(n_rep, n))
                np.add.at(weights, (np.arange(n_rep)[:, None], draws), 1)
                rep_rates = np.empty((n_rep, len(units), len(_CENTERS)))
                for i, (u, sel) in enumerate(zip(units, selections, strict=True)):
                    w = weights[:, u.stim_idx[sel] - first]
                    with np.errstate(invalid="ignore", divide="ignore"):
                        rep_rates[:, i] = (
                            w
                            @ u.counts[sel].astype(float)
                            / w.sum(axis=1, keepdims=True)
                            / PSTH_BIN
                        )
                rep_pop, _ = _population_psth(rep_rates)
                boot.extend(_half_max_latency(p) for p in rep_pop)
            finite = np.array(boot)[np.isfinite(boot)] if boot else np.array([])
            ci = (
                np.percentile(finite, [2.5, 97.5]) * 1e3
                if finite.size
                else (np.nan, np.nan)
            )
            quartiles.append(
                QuartileStats(
                    index=k,
                    start_time=float(onsets[first]),
                    stop_time=float(onsets[last]),
                    n_trials=int(n),
                    n_units=int(n_units),
                    latency_ms=_half_max_latency(pop) * 1e3,
                    ci_low_ms=float(ci[0]),
                    ci_high_ms=float(ci[1]),
                    psth=pop,
                )
            )
    latencies = np.array([q.latency_ms for q in quartiles])
    return (
        ModalityStats(
            modality=modality,
            n_trials=len(onsets),
            n_units_tested=len(spike_times),
            n_units=len(units),
            unit_ids=tuple(u.unit_id for u in units),
            quartiles=tuple(quartiles),
            latency_range_ms=(
                float(np.ptp(latencies))
                if latencies.size and np.all(np.isfinite(latencies))
                else np.nan
            ),
            psth_time=_CENTERS,
        ),
        units,
    )


def estimate_stim_shifts(
    spike_times_per_unit: Sequence[npt.NDArray[np.floating]],
    onsets: npt.NDArray[np.floating],
    template_mask: npt.NDArray[np.bool_],
    stim_idx_per_unit: Sequence[npt.NDArray[np.intp]] | None = None,
) -> npt.NDArray[np.float64]:
    """Per-stim onset shift (s): the shift s maximizing the Poisson log-likelihood of the
    spikes around the stim (all units) under each unit's mean rate, λ_u(t - s), estimated
    from the stims in `template_mask` (1-ms bins, 5-ms boxcar). A stim displayed one frame
    later than its reported onset has s ≈ +16.7 ms.

    As `npc_sessions_cache.plots.responses.estimate_trial_shifts`. With `stim_idx_per_unit`,
    each unit only contributes to (and has its template from) its own stims; stims without
    a unit are nan."""
    dt = 0.001
    pad = SHIFTS.max() + 0.01
    edges = np.arange(SHIFT_WINDOW[0] - pad, SHIFT_WINDOW[1] + pad + dt, dt)
    centers = edges[:-1] + dt / 2
    window_bins = np.flatnonzero(
        (centers >= SHIFT_WINDOW[0]) & (centers < SHIFT_WINDOW[1])
    )
    # λ(t - s) on the window bins: the template shifted right by s / dt bins
    shifted_bins = window_bins[None, :] - np.round(SHIFTS / dt).astype(int)[:, None]
    onsets = np.asarray(onsets, dtype=float)
    ll = np.zeros((len(onsets), len(SHIFTS)))
    has_unit = np.zeros(len(onsets), dtype=bool)
    for i, st in enumerate(spike_times_per_unit):
        stim_idx = (
            np.arange(len(onsets))
            if stim_idx_per_unit is None
            else np.asarray(stim_idx_per_unit[i])
        )
        is_template = template_mask[stim_idx]
        if not is_template.any():
            continue
        counts = _binned_counts(np.asarray(st, dtype=float), onsets[stim_idx], edges)
        rate = counts[is_template].mean(axis=0) / dt
        rate = np.convolve(rate, np.ones(5) / 5, mode="same") + 0.5  # floor: 0.5 Hz
        ll[stim_idx] += (
            counts[:, window_bins].astype(float) @ np.log(rate)[shifted_bins].T
            - rate[shifted_bins].sum(axis=1) * dt
        )
        has_unit[stim_idx] = True
    return np.where(has_unit, SHIFTS[np.argmax(ll, axis=1)], np.nan)


def _stim_shift_stats(
    units: Sequence[_Unit],
    onsets: npt.NDArray[np.float64],
    n_quartiles: int,
    max_shift_ms: float,
) -> StimShiftStats:
    groups = np.array_split(np.arange(len(onsets)), n_quartiles)
    template = np.zeros(len(onsets), dtype=bool)
    template[groups[0]] = True
    shift_ms = 1e3 * estimate_stim_shifts(
        [u.spike_times for u in units], onsets, template, [u.stim_idx for u in units]
    )

    def frac(x: npt.NDArray[np.float64]) -> float:
        x = x[np.isfinite(x)]
        # 1e-6: shifts are multiples of 1 ms with float error
        return float(np.mean(np.abs(x) > max_shift_ms + 1e-6)) if x.size else np.nan

    def median(x: npt.NDArray[np.float64]) -> float:
        x = x[np.isfinite(x)]
        return float(np.median(x)) if x.size else np.nan

    return StimShiftStats(
        n_trials=len(onsets),
        n_template_trials=int(template.sum()),
        frac_shifted=frac(shift_ms),
        frac_shifted_by_quartile=tuple(frac(shift_ms[g]) for g in groups),
        median_shift_ms_by_quartile=tuple(median(shift_ms[g]) for g in groups),
        max_shift_ms=max_shift_ms,
        onsets=onsets,
        shift_ms=shift_ms,
    )


def _clean_onsets(
    onsets: npt.ArrayLike | None,
    conditions: npt.ArrayLike | None,
    block_start: float | None,
    block_stop: float | None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.intp] | None]:
    """Finite onsets within the block, in time order (with their conditions)"""
    if onsets is None:
        return np.array([], dtype=float), None
    times = np.asarray(onsets, dtype=float).ravel()
    keep = np.isfinite(times)
    if block_start is not None:
        keep &= times >= block_start
    if block_stop is not None:
        keep &= times <= block_stop
    order = np.argsort(times[keep], kind="stable")
    labels = None
    if conditions is not None:
        raw = np.asarray(conditions).ravel()
        if raw.shape != times.shape:
            raise ValueError(
                f"conditions ({raw.shape}) and onsets ({times.shape}) differ in length"
            )
        labels = np.unique(raw[keep][order], return_inverse=True)[1].astype(np.intp)
    return times[keep][order], labels


def _fmt(values: Sequence[float]) -> str:
    return ", ".join("nan" if not np.isfinite(v) else f"{v:.1f}" for v in values)


def quartile_latency_check(
    spike_times: (
        Mapping[str, npt.NDArray[np.floating]] | Sequence[npt.NDArray[np.floating]]
    ),
    vis_onsets: npt.ArrayLike,
    aud_onsets: npt.ArrayLike | None = None,
    *,
    vis_conditions: npt.ArrayLike | None = None,
    block_start: float | None = None,
    block_stop: float | None = None,
    block: str | None = None,
    max_latency_range_ms: float = MAX_LATENCY_RANGE_MS,
    min_responsive_units: int = MIN_RESPONSIVE_UNITS,
    min_stims_per_quartile: int = MIN_STIMS_PER_QUARTILE,
    min_rate_increase_hz: float = MIN_RATE_INCREASE_HZ,
    min_score: float = MIN_SCORE,
    n_quartiles: int = N_QUARTILES,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
    stim_shifts: bool = True,
) -> QuartileLatencyResult:
    """Is the latency of the population visual response stable over a stim block?
    (criterion (b) for display times that aren't measured with the photodiode: see module
    docstring). Pure function of its arguments; deterministic.

    - `spike_times`: sorted spike times (s) per unit, for the units to consider (already
      filtered by quality, e.g. `UNIT_QUERY`); same clock as the onsets
    - `vis_onsets`: onsets (s) of the block's visual stims, as they would be reported
      (i.e. from the display times being checked). nan are dropped
    - `aud_onsets`: onsets of the block's auditory stims (control, optional)
    - `vis_conditions`: a label per visual onset, for stims that drive different units
      (e.g. the position of an RF-mapping stim). A unit is then tested on the stims of the
      conditions it responds to. Quartiles are still quarters of all the visual stims
    - `block_start`, `block_stop`: onsets outside are dropped
    - `block`: a name for the result

    The verdict uses the point estimates of the visual latencies only:
    range <= `max_latency_range_ms` with >= `min_responsive_units` units -> 'passed';
    range > `max_latency_range_ms` -> 'failed'; otherwise 'untestable'.
    """
    if not isinstance(spike_times, Mapping):
        spike_times = {str(i): st for i, st in enumerate(spike_times)}
    vis, conditions = _clean_onsets(vis_onsets, vis_conditions, block_start, block_stop)
    aud, _ = _clean_onsets(aud_onsets, None, block_start, block_stop)
    params = {
        "unit_query": UNIT_QUERY,
        "response_window": RESPONSE_WINDOW,
        "baseline_window": BASELINE_WINDOW,
        "max_latency_range_ms": max_latency_range_ms,
        "min_responsive_units": min_responsive_units,
        "min_stims_per_quartile": min_stims_per_quartile,
        "min_rate_increase_hz": min_rate_increase_hz,
        "min_score": min_score,
        "n_quartiles": n_quartiles,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "n_vis_conditions": (
            None
            if conditions is None or not conditions.size
            else int(conditions.max() + 1)
        ),
    }
    vis_stats, vis_units = _modality_stats(
        spike_times,
        vis,
        "vis",
        conditions,
        n_quartiles=n_quartiles,
        min_stims_per_quartile=min_stims_per_quartile,
        min_rate_increase_hz=min_rate_increase_hz,
        min_score=min_score,
        n_bootstrap=n_bootstrap,
        seed=(seed, 0),
    )
    aud_stats = None
    if aud_onsets is not None:
        aud_stats, _ = _modality_stats(
            spike_times,
            aud,
            "aud",
            None,
            n_quartiles=n_quartiles,
            min_stims_per_quartile=min_stims_per_quartile,
            min_rate_increase_hz=min_rate_increase_hz,
            min_score=min_score,
            n_bootstrap=n_bootstrap,
            seed=(seed, 1),
        )

    verdict: Verdict
    shifts = None
    if not spike_times:
        verdict, reason = "untestable", "no units"
    elif len(vis) < n_quartiles * max(min_stims_per_quartile, 1):
        verdict = "untestable"
        reason = f"{len(vis)} visual stims (< {n_quartiles} x {min_stims_per_quartile})"
    elif vis_stats.n_units < min_responsive_units:
        verdict = "untestable"
        reason = (
            f"{vis_stats.n_units} visually responsive units (< {min_responsive_units}) "
            f"of {len(spike_times)}"
        )
    else:
        latencies = vis_stats.latencies_ms
        detail = (
            f"quartile latencies {_fmt(latencies)} ms, {vis_stats.n_units} units, "
            f"{len(vis)} stims"
        )
        if stim_shifts:
            shifts = _stim_shift_stats(vis_units, vis, n_quartiles, MAX_STIM_SHIFT_MS)
            detail += (
                f", {shifts.frac_shifted:.0%} of stims shifted > "
                f"{MAX_STIM_SHIFT_MS:g} ms"
            )
        if aud_stats is not None and np.isfinite(aud_stats.latency_range_ms):
            detail += f", aud range {aud_stats.latency_range_ms:.1f} ms"
        if not np.isfinite(vis_stats.latency_range_ms):
            verdict = "untestable"
            reason = f"visual half-max latency undefined in a quartile ({detail})"
        elif vis_stats.latency_range_ms <= max_latency_range_ms:
            verdict = "passed"
            reason = (
                f"visual half-max latency range over quartiles "
                f"{vis_stats.latency_range_ms:.1f} ms (<= {max_latency_range_ms:g} ms; "
                f"{detail})"
            )
        else:
            verdict = "failed"
            reason = (
                f"visual half-max latency range over quartiles "
                f"{vis_stats.latency_range_ms:.1f} ms (> {max_latency_range_ms:g} ms; "
                f"{detail})"
            )
    return QuartileLatencyResult(
        verdict=verdict,
        reason=reason,
        vis=vis_stats,
        aud=aud_stats,
        stim_shifts=shifts,
        params=params,
        block=block,
    )


# session wrapper -------------------------------------------------------------------- #


class StimOnsets(NamedTuple):
    """Onsets of a stim block's visual stims, from the stim file and display times."""

    vis: npt.NDArray[np.float64]
    conditions: npt.NDArray[np.intp] | None
    """RFMapping: index of each grating's position; otherwise None"""
    start_time: float
    stop_time: float
    """First and last display time of the block"""
    condition_values: npt.NDArray[np.float64] | None = None
    """RFMapping: (x, y) of each grating, for comparing conditions between blocks (the
    indices in `conditions` are per block); otherwise None"""


def get_stim_path(
    session: npc_sessions.DynamicRoutingSession, block: str
) -> upath.UPath:
    """Path of the stim file for `block`: the stem of a stim file (as the keys of
    `session.stim_frame_times`), or the start of one if unambiguous (e.g. 'RFMapping').
    """
    paths = tuple(session.stim_paths)
    for path in paths:
        if path.stem == block:
            return path
    matches = [path for path in paths if path.stem.startswith(block)]
    if len(matches) != 1:
        raise ValueError(
            f"{session.id}: {len(matches)} stim files match {block!r}: "
            f"{[p.stem for p in paths]}"
        )
    return matches[0]


def get_stim_onsets(
    session: npc_sessions.DynamicRoutingSession,
    block: str,
    display_times: npt.ArrayLike,
) -> StimOnsets:
    """Visual stim onsets of a stim block, for the given frame display times (one per frame
    of the stim file), as the block's intervals table would have them:

    - DynamicRouting1: `stim_start_time` of the trials with a visual stim, without opto
    - RFMapping: `stim_start_time` of the small gratings; conditions are their positions
      (a unit is tested on the positions in its receptive field). The ~72 full-field
      flashes are left out: too few for enough responsive units on their own, and pooled
      with the gratings they add a response with another time course, which makes the
      half-max latency less stable (hub experiments/constant-lag-acceptance). A block
      without gratings uses its flashes, without conditions

    Reads the stim file only: no trials table is made, and no sound or opto alignment run.
    """
    import npc_stim

    import npc_sessions.trials as TaskControl  # noqa: N812

    stim_path = get_stim_path(session, block)
    times = np.asarray(display_times, dtype=float)
    finite = times[np.isfinite(times)]
    if finite.size == 0:
        raise ValueError(f"{session.id}: no display times for {stim_path.stem}")
    conditions: npt.NDArray[np.intp] | None = None
    condition_values: npt.NDArray[np.float64] | None = None
    sync = session.sync_data if session.is_sync else None
    if "RFMapping" in stim_path.stem:
        rf = TaskControl.VisRFMapping(stim_path, sync, vis_display_times=times)
        frames = np.asarray(rf._hdf5_data["stimStartFrame"][()])[rf._idx]
        xy = np.round(np.column_stack([rf.grating_x, rf.grating_y]), 3)
        is_grating = np.asarray(rf.is_small_field_grating, dtype=bool)
        if is_grating.any():
            frames = frames[is_grating]
            condition_values = xy[is_grating]
            conditions = (
                np.unique(xy[is_grating], axis=0, return_inverse=True)[1]
                .ravel()
                .astype(np.intp)
            )
    elif stim_path.stem.startswith(session.task_stim_name):
        task = TaskControl.DynamicRouting1(stim_path, sync, vis_display_times=times)
        frames = np.asarray(task._sam.stimStartFrame, dtype=float)
        is_vis = np.asarray(task.is_vis_stim, dtype=bool)
        n = min(len(frames), len(is_vis))
        is_opto = np.zeros(n, dtype=bool)
        if task._is_opto:
            triggered = np.array(
                [
                    f is not None and np.isfinite(f)
                    for f in npc_stim.get_stim_trigger_frames(
                        stim_path, stim_type="opto"
                    )
                ][:n],
                dtype=bool,
            )
            is_opto[: len(triggered)] = triggered
        frames = frames[:n][is_vis[:n] & ~is_opto]
    else:
        raise ValueError(
            f"{session.id}: no visual stims known for {stim_path.stem} "
            "(expected the task or RFMapping)"
        )
    frames = np.asarray(frames, dtype=float)
    in_range = np.isfinite(frames) & (frames >= 0) & (frames < len(times))
    vis = np.full(len(frames), np.nan)
    vis[in_range] = times[frames[in_range].astype(int)]
    return StimOnsets(
        vis=vis,
        conditions=conditions,
        start_time=float(finite[0]),
        stop_time=float(finite[-1]),
        condition_values=condition_values,
    )


def select_units(units: pd.DataFrame) -> pd.DataFrame:
    """Rows of a units table passing `UNIT_QUERY` (all rows, with a warning, if the metrics
    are missing)"""
    try:
        return units.query(UNIT_QUERY)
    except Exception as exc:  # metrics columns missing
        logger.warning(f"unit QC query not applied: {exc!r}")
        return units


def get_unit_spike_times(
    session: npc_sessions.DynamicRoutingSession,
) -> dict[str, npt.NDArray[np.float64]]:
    """{unit_id: spike times (s, on sync)} for the session's units passing `UNIT_QUERY`;
    empty if the session isn't sorted.

    Doesn't use `session._units` unless it's already made, as that needs `session.trials`
    (activity drift), which needs display times, which may be waiting for this check. The
    table is made as in `_units` up to the columns used here: units of the probes inserted,
    with `amplitude_cutoff` recomputed from the spike amplitudes. It is kept on the
    session (`_qc_unit_spike_times`), so the check, repeated for other blocks or for a
    plot, loads spikes once."""
    import npc_ephys

    from npc_sessions import utils

    cached = session.__dict__.get("_qc_unit_spike_times")
    if cached is not None:
        return cached
    if not session.is_sorted:
        return {}
    if (units := session.__dict__.get("_units")) is None:
        units = npc_ephys.make_units_table_from_spike_interface_ks25(
            session.sorted_data,
            session.ephys_timing_data,
            include_waveform_arrays=False,
        )
        units = units[units["electrode_group_name"].isin(session.probes_inserted)]
        units = npc_ephys.add_global_unit_ids(units=units.copy(), session=session.id)
        units["amplitude_cutoff"] = units["spike_amplitudes"].map(
            utils.get_amplitude_cutoff
        )
    units = select_units(units)
    spike_times = {
        str(unit_id): np.asarray(st, dtype=float)
        for unit_id, st in zip(units["unit_id"], units["spike_times"], strict=True)
    }
    session.__dict__["_qc_unit_spike_times"] = spike_times
    return spike_times


def quartile_visual_latency(
    session: npc_sessions.DynamicRoutingSession,
    block: str,
    *,
    display_times: npt.ArrayLike | None = None,
    vis_onsets: npt.ArrayLike | None = None,
    vis_conditions: npt.ArrayLike | None = None,
    aud_onsets: npt.ArrayLike | None = None,
    spike_times: Mapping[str, npt.NDArray[np.floating]] | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> QuartileLatencyResult:
    """`quartile_latency_check` for a stim block of a session with visual stims (the task,
    or RFMapping): is the latency of the population visual response, relative to the
    block's stim onsets, stable over the block?

    - `block`: the stem of the stim file (as the keys of `session.stim_frame_times`), or
      the start of it if unambiguous (e.g. 'DynamicRouting1', 'RFMapping')
    - `display_times`: the block's candidate frame display times (e.g. vsync + constant
      lag), one per frame of the stim file. Visual stim onsets are computed from them and
      the stim file (`get_stim_onsets`). Pass them when the check decides whether display
      times are used: then nothing here needs the session's trials, epochs or intervals.
      If None (and no `vis_onsets`): `session._stim_display_times[block].display_times`
    - `vis_onsets` (and `vis_conditions`): onsets to use instead, e.g. from an intervals
      table already made with the candidate display times
    - `aud_onsets`: onsets of the block's auditory stims, for the control (not part of the
      verdict; left out if None)
    - `spike_times`: {unit_id: spike times} to use instead of `get_unit_spike_times(session)`
    - `kwargs`: thresholds etc. of `quartile_latency_check`

    A session without sorted units is 'untestable'. Units are only loaded here: sessions
    that don't call this for any block aren't affected.
    """
    stim_path = get_stim_path(session, block)
    start: float | None = None
    stop: float | None = None
    if vis_onsets is None:
        if display_times is None:
            block_times = session._stim_display_times[stim_path.stem]
            if isinstance(block_times, Exception):
                raise ValueError(
                    f"{session.id}: no display times for {stim_path.stem}: {block_times!r}"
                )
            display_times = block_times.display_times
        assert display_times is not None
        onsets = get_stim_onsets(session, stim_path.stem, display_times)
        vis_onsets, vis_conditions = onsets.vis, onsets.conditions
        start, stop = onsets.start_time, onsets.stop_time
        if aud_onsets is not None:
            # sounds can end up to a stim duration after the last frame
            stop += 1.0
    if spike_times is None:
        spike_times = get_unit_spike_times(session)
    return quartile_latency_check(
        spike_times,
        vis_onsets,
        aud_onsets,
        vis_conditions=vis_conditions,
        block_start=start,
        block_stop=stop,
        block=f"{session.id} {stim_path.stem}",
        **kwargs,
    )


# criterion (c): whole-block latency vs photodiode blocks of the same stim ------------- #

MAX_REFERENCE_LATENCY_DIFF_MS = 8.0
"""Largest accepted difference between a block's whole-block visual half-max latency and that
of the session's photodiode block(s) of the same stim, for the same units (half a 60 Hz
frame, as `MAX_LATENCY_RANGE_MS`)"""
MIN_REFERENCE_STIMS = N_QUARTILES * MIN_STIMS_PER_QUARTILE
"""Fewest visual stims in a block (the one checked, or a reference) for a whole-block latency"""
MIN_POOLED_REFERENCE_STIMS = 500
"""Fewest visual stims in the reference blocks together. Units are selected on the pooled
reference stims, and with a small reference the comparison isn't reliable: splitting photodiode
RFMapping blocks, a 288-grating reference gave differences > 8 ms in 4 of 50 blocks (max 25
ms), a reference of the rest of the block vs 240 gratings in 0 of 101 (max 5.9 ms). See
hub/experiments/constant-lag-acceptance (criterion (c) null)."""


@dataclasses.dataclass(frozen=True)
class BlockLatency:
    """Whole-block visual latency of one stim block, for `reference_latency_check`."""

    block: str | None
    is_reference: bool
    n_trials: int
    n_units: int
    """Selected units with a positive PSTH peak in this block"""
    latency_ms: float
    """Half-max latency of the population PSTH; nan if undefined"""
    ci_low_ms: float
    ci_high_ms: float
    """95% bootstrap CI (the block's stims resampled); nan if not computed"""
    psth: npt.NDArray[np.float64] = dataclasses.field(repr=False, compare=False)

    def to_dict(self, include_arrays: bool = False) -> dict[str, Any]:
        return {
            f.name: _jsonable(getattr(self, f.name))
            for f in dataclasses.fields(self)
            if include_arrays or f.name != "psth"
        }


@dataclasses.dataclass(frozen=True)
class ReferenceLatencyResult:
    """Result of `reference_latency_check`."""

    verdict: Verdict
    reason: str
    block: BlockLatency
    references: tuple[BlockLatency, ...]
    n_units: int
    """Units selected on the reference block(s)"""
    unit_ids: tuple[str, ...]
    reference_latency_ms: float
    """Median of the reference blocks' latencies; nan if undefined"""
    latency_diff_ms: float
    """Block latency - `reference_latency_ms`; nan if undefined"""
    params: dict[str, Any]
    psth_time: npt.NDArray[np.float64] = dataclasses.field(repr=False, compare=False)

    @property
    def passed(self) -> bool:
        return self.verdict == "passed"

    @property
    def summary(self) -> str:
        """One line, for the notes of an epoch"""
        return f"photodiode reference latency check {self.verdict}: {self.reason}"

    def to_dict(self, include_arrays: bool = False) -> dict[str, Any]:
        d: dict[str, Any] = {
            "verdict": self.verdict,
            "reason": self.reason,
            "summary": self.summary,
            "n_units": self.n_units,
            "reference_latency_ms": _jsonable(self.reference_latency_ms),
            "latency_diff_ms": _jsonable(self.latency_diff_ms),
            "block": self.block.to_dict(include_arrays),
            "references": [r.to_dict(include_arrays) for r in self.references],
            "params": _jsonable(self.params),
        }
        if include_arrays:
            d["unit_ids"] = list(self.unit_ids)
            d["psth_time"] = _jsonable(self.psth_time)
        return d


def _sorted_onsets(
    onsets: npt.ArrayLike, labels: npt.NDArray[np.intp] | None
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.intp] | None]:
    """Finite onsets in time order, with their labels (unchanged, unlike `_clean_onsets`)"""
    times = np.asarray(onsets, dtype=float).ravel()
    keep = np.flatnonzero(np.isfinite(times))
    keep = keep[np.argsort(times[keep], kind="stable")]
    return times[keep], None if labels is None else labels[keep]


def _common_labels(
    conditions: Sequence[npt.ArrayLike],
) -> list[npt.NDArray[np.intp]]:
    """Integer labels shared between blocks, from each block's condition values (one value,
    or one row of values, per stim)"""
    arrays = [np.asarray(c) for c in conditions]
    arrays = [a.reshape(len(a), -1) for a in arrays]
    inverse = np.unique(np.concatenate(arrays), axis=0, return_inverse=True)[1]
    inverse = inverse.ravel().astype(np.intp)
    return np.split(inverse, np.cumsum([len(a) for a in arrays])[:-1])


def _block_latency(
    unit_spike_times: Sequence[npt.NDArray[np.float64]],
    unit_conditions: Sequence[npt.NDArray[np.intp] | None],
    onsets: npt.NDArray[np.float64],
    labels: npt.NDArray[np.intp] | None,
    n_bootstrap: int,
    seed: Sequence[int],
) -> tuple[float, int, float, float, npt.NDArray[np.float64]]:
    """(half-max latency in ms, units used, CI low, CI high, population PSTH) over a block's
    stims: each unit's PSTH is over the stims of its conditions (all stims if None)"""
    n = len(onsets)
    selections = [
        (
            np.arange(n)
            if labels is None or conds is None
            else np.flatnonzero(np.isin(labels, conds))
        )
        for conds in unit_conditions
    ]
    counts = [
        _binned_counts(st, onsets[sel], _EDGES).astype(float)
        for st, sel in zip(unit_spike_times, selections, strict=True)
    ]
    with np.errstate(invalid="ignore", divide="ignore"):
        rates = np.array([c.sum(axis=0) / len(c) / PSTH_BIN for c in counts])
        pop, n_units = _population_psth(rates)
    boot: list[float] = []
    rng = np.random.default_rng(list(seed))
    for start in range(0, n_bootstrap if n else 0, 50):
        n_rep = min(50, n_bootstrap - start)
        weights = np.zeros((n_rep, n))
        np.add.at(
            weights,
            (np.arange(n_rep)[:, None], rng.integers(0, n, size=(n_rep, n))),
            1,
        )
        rep_rates = np.empty((n_rep, len(counts), len(_CENTERS)))
        with np.errstate(invalid="ignore", divide="ignore"):
            for i, (c, sel) in enumerate(zip(counts, selections, strict=True)):
                w = weights[:, sel]
                rep_rates[:, i] = w @ c / w.sum(axis=1, keepdims=True) / PSTH_BIN
            rep_pop, _ = _population_psth(rep_rates)
        boot.extend(_half_max_latency(p) for p in rep_pop)
    finite = np.array(boot)[np.isfinite(boot)] if boot else np.array([])
    ci = np.percentile(finite, [2.5, 97.5]) * 1e3 if finite.size else (np.nan, np.nan)
    return (
        _half_max_latency(pop) * 1e3,
        int(n_units),
        float(ci[0]),
        float(ci[1]),
        pop,
    )


def reference_latency_check(
    spike_times: (
        Mapping[str, npt.NDArray[np.floating]] | Sequence[npt.NDArray[np.floating]]
    ),
    vis_onsets: npt.ArrayLike,
    reference_vis_onsets: Sequence[npt.ArrayLike],
    *,
    vis_conditions: npt.ArrayLike | None = None,
    reference_vis_conditions: Sequence[npt.ArrayLike] | None = None,
    block: str | None = None,
    reference_blocks: Sequence[str] | None = None,
    max_latency_diff_ms: float = MAX_REFERENCE_LATENCY_DIFF_MS,
    min_responsive_units: int = MIN_RESPONSIVE_UNITS,
    min_stims: int = MIN_REFERENCE_STIMS,
    min_pooled_reference_stims: int = MIN_POOLED_REFERENCE_STIMS,
    min_rate_increase_hz: float = MIN_RATE_INCREASE_HZ,
    min_score: float = MIN_SCORE,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> ReferenceLatencyResult:
    """Is the visual response latency of a stim block the same as in the session's blocks of
    the same stim whose display times are measured with the photodiode? (criterion (c) for
    display times that aren't: see module docstring). `quartile_latency_check` compares
    quarters within a block, so it can't see a display that is late for the whole block; this
    can, for sessions that have a reference block. Pure function of its arguments;
    deterministic.

    - `spike_times`: as for `quartile_latency_check`, covering all the blocks
    - `vis_onsets`: visual stim onsets (s) of the block being checked, from its candidate
      display times
    - `reference_vis_onsets`: the same for each reference (photodiode) block
    - `vis_conditions`, `reference_vis_conditions`: a value (or row of values, e.g. x, y) per
      stim, comparable between blocks, for stims that drive different units (the position of
      an RF-mapping grating). Give both or neither

    Method (parameters as `quartile_latency_check`):

    - units, and for each unit its conditions, are selected on the reference blocks' stims
      (pooled): the timing of those is known, and a late display could move responses out
      of the selection window
    - per block: population PSTH (mean over the selected units of the baseline-subtracted,
      peak-normalised PSTH over the block's stims of the unit's conditions) and its half-max
      latency. Blocks with fewer than `min_stims` visual stims aren't used
    - verdict, from the point estimates only: `passed` if |block latency - median of the
      reference blocks' latencies| <= `max_latency_diff_ms`, with >= `min_responsive_units`
      units in each block; `failed` if the difference is larger; `untestable` otherwise
      (too few units or stims, or a latency is undefined)
    """
    if not isinstance(spike_times, Mapping):
        spike_times = {str(i): st for i, st in enumerate(spike_times)}
    n_refs = len(reference_vis_onsets)
    names = list(reference_blocks) if reference_blocks is not None else [None] * n_refs
    if len(names) != n_refs:
        raise ValueError(f"{len(names)} reference block names for {n_refs} blocks")
    labels: list[npt.NDArray[np.intp] | None]
    if vis_conditions is None or reference_vis_conditions is None:
        if (vis_conditions is None) != (reference_vis_conditions is None):
            raise ValueError("conditions are needed for all blocks, or none")
        labels = [None] * (n_refs + 1)
    else:
        if len(reference_vis_conditions) != n_refs:
            raise ValueError("one array of conditions is needed per reference block")
        labels = list(_common_labels([vis_conditions, *reference_vis_conditions]))
    all_onsets = [vis_onsets, *reference_vis_onsets]
    for onsets_, labels_ in zip(all_onsets, labels, strict=True):
        if labels_ is not None and len(labels_) != np.size(onsets_):
            raise ValueError(
                f"conditions ({len(labels_)}) and onsets ({np.size(onsets_)}) differ in length"
            )
    cleaned = [
        _sorted_onsets(o, lab) for o, lab in zip(all_onsets, labels, strict=True)
    ]
    (onsets, block_labels), refs = cleaned[0], cleaned[1:]
    used = [i for i, (o, _) in enumerate(refs) if len(o) >= min_stims]
    params = {
        "unit_query": UNIT_QUERY,
        "response_window": RESPONSE_WINDOW,
        "baseline_window": BASELINE_WINDOW,
        "max_latency_diff_ms": max_latency_diff_ms,
        "min_responsive_units": min_responsive_units,
        "min_stims": min_stims,
        "min_pooled_reference_stims": min_pooled_reference_stims,
        "min_rate_increase_hz": min_rate_increase_hz,
        "min_score": min_score,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
    }
    n_pooled = sum(len(refs[i][0]) for i in used)

    # units: selected on the pooled stims of the reference blocks in use
    units: list[_Unit] = []
    unit_conditions: list[npt.NDArray[np.intp] | None] = []
    if used and spike_times:
        pooled = np.concatenate([refs[i][0] for i in used])
        order = np.argsort(pooled, kind="stable")
        pooled_labels = None
        if block_labels is not None:
            pooled_labels = np.concatenate([refs[i][1] for i in used])[order]
        units = _responsive_units(
            spike_times,
            pooled[order],
            "vis",
            pooled_labels,
            min_rate_increase_hz,
            min_score,
        )
        unit_conditions = [
            None if pooled_labels is None else np.unique(pooled_labels[u.stim_idx])
            for u in units
        ]

    def latency(
        k: int,
        name: str | None,
        o: npt.NDArray[np.float64],
        lab: npt.NDArray[np.intp] | None,
    ) -> BlockLatency:
        if units and len(o) >= min_stims:
            est, n_units, lo, hi, pop = _block_latency(
                [u.spike_times for u in units],
                unit_conditions,
                o,
                lab,
                n_bootstrap,
                (seed, 2, k),
            )
        else:
            est, n_units, lo, hi = np.nan, 0, np.nan, np.nan
            pop = np.full(len(_CENTERS), np.nan)
        return BlockLatency(
            block=name,
            is_reference=k > 0,
            n_trials=len(o),
            n_units=n_units,
            latency_ms=est,
            ci_low_ms=lo,
            ci_high_ms=hi,
            psth=pop,
        )

    this = latency(0, block, onsets, block_labels)
    references = tuple(latency(i + 1, names[i], *refs[i]) for i in range(n_refs))
    ref_latencies = np.array([references[i].latency_ms for i in used])
    reference_latency = (
        float(np.median(ref_latencies))
        if ref_latencies.size and np.all(np.isfinite(ref_latencies))
        else np.nan
    )
    diff = this.latency_ms - reference_latency

    verdict: Verdict = "untestable"
    if not spike_times:
        reason = "no units"
    elif not used:
        reason = f"no reference block with >= {min_stims} visual stims"
    elif n_pooled < min_pooled_reference_stims:
        reason = (
            f"{n_pooled} visual stims in the reference block(s) "
            f"(< {min_pooled_reference_stims})"
        )
    elif len(onsets) < min_stims:
        reason = f"{len(onsets)} visual stims (< {min_stims})"
    elif len(units) < min_responsive_units:
        reason = (
            f"{len(units)} visually responsive units (< {min_responsive_units}) of "
            f"{len(spike_times)} in the reference block(s)"
        )
    else:
        detail = (
            f"{_fmt([this.latency_ms])} ms vs {_fmt(ref_latencies.tolist())} ms in "
            f"{len(used)} photodiode block(s), {len(units)} units, {len(onsets)} stims"
        )
        n_min = min(this.n_units, *(references[i].n_units for i in used))
        if n_min < min_responsive_units:
            reason = (
                f"{n_min} units with a response in a block "
                f"(< {min_responsive_units}; {detail})"
            )
        elif not np.isfinite(diff):
            reason = f"visual half-max latency undefined in a block ({detail})"
        elif abs(diff) <= max_latency_diff_ms:
            verdict = "passed"
            reason = (
                f"visual half-max latency {diff:+.1f} ms from the photodiode block(s) "
                f"(within {max_latency_diff_ms:g} ms; {detail})"
            )
        else:
            verdict = "failed"
            reason = (
                f"visual half-max latency {diff:+.1f} ms from the photodiode block(s) "
                f"(more than {max_latency_diff_ms:g} ms; {detail})"
            )
    return ReferenceLatencyResult(
        verdict=verdict,
        reason=reason,
        block=this,
        references=references,
        n_units=len(units),
        unit_ids=tuple(u.unit_id for u in units),
        reference_latency_ms=reference_latency,
        latency_diff_ms=float(diff),
        params=params,
        psth_time=_CENTERS,
    )


def get_script_name(stim: str) -> str:
    """'RFMapping' for the stim file (stem) 'RFMapping_666986_20230814_120000'"""
    return stim.split("_")[0]


def get_reference_display_times(
    session: npc_sessions.DynamicRoutingSession, block: str
) -> dict[str, npt.NDArray[np.float64]]:
    """{stim file stem: frame display times} of the session's other blocks of the same stim
    as `block` whose display times are measured with the photodiode (empty if none). Uses
    `session._stim_display_times` only (the candidates: not `stim_validity`)."""
    stem = get_stim_path(session, block).stem
    out = {}
    for name, times in session._stim_display_times.items():
        if (
            name != stem
            and get_script_name(name) == get_script_name(stem)
            and not isinstance(times, Exception)
            and times.source == "photodiode"
            and times.display_times is not None
        ):
            out[name] = times.display_times
    return out


def reference_visual_latency(
    session: npc_sessions.DynamicRoutingSession,
    block: str,
    *,
    display_times: npt.ArrayLike,
    reference_display_times: Mapping[str, npt.ArrayLike] | None = None,
    spike_times: Mapping[str, npt.NDArray[np.floating]] | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> ReferenceLatencyResult | None:
    """`reference_latency_check` for a stim block of a session with visual stims (the task,
    or RFMapping), given its candidate frame display times: is the latency of the population
    visual response the same as in the session's photodiode blocks of the same stim?

    - `reference_display_times`: {stim file stem: frame display times} of the reference
      blocks; by default `get_reference_display_times(session, block)`
    - other arguments as `quartile_visual_latency`

    None if the session has no reference block (the check doesn't apply). Like
    `quartile_visual_latency`, it needs the stim files and the sorted units only.
    """
    stim_path = get_stim_path(session, block)
    if reference_display_times is None:
        reference_display_times = get_reference_display_times(session, stim_path.stem)
    if not reference_display_times:
        return None
    this = get_stim_onsets(session, stim_path.stem, display_times)
    references = {
        name: get_stim_onsets(session, name, times)
        for name, times in reference_display_times.items()
    }
    # e.g. an RFMapping block without gratings uses its flashes: then no block's stims are
    # split by condition
    use_conditions = all(
        o.condition_values is not None for o in (this, *references.values())
    )
    if spike_times is None:
        spike_times = get_unit_spike_times(session)
    return reference_latency_check(
        spike_times,
        this.vis,
        [o.vis for o in references.values()],
        vis_conditions=this.condition_values if use_conditions else None,
        reference_vis_conditions=(
            [o.condition_values for o in references.values()]  # type: ignore[misc]
            if use_conditions
            else None
        ),
        block=f"{session.id} {stim_path.stem}",
        reference_blocks=[f"{session.id} {name}" for name in references],
        **kwargs,
    )


# validity of a stim block ----------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class StimValidity:
    """Whether a stim block (one stim file) of a session can be used, and why not: one per
    stim file in `DynamicRoutingSession.stim_validity`. It is the source of `is_qc_pass` in
    the epochs table, and of the reason in the epoch's `notes`."""

    stim: str
    """Stem of the stim file"""
    is_valid: bool
    reason: str = ""
    """Why the block is invalid ('' if valid)"""
    display_time_source: str = "unavailable"
    """'photodiode', 'constant_lag' (candidate display times: vsync + constant lag),
    'unavailable' (no frame times on sync) or 'no_sync'"""
    frame_index_check: Any | None = None
    """(a) `npc_stim.FrameIndexCheck`: constant-lag blocks only"""
    latency_check: QuartileLatencyResult | None = None
    """(b): constant-lag blocks with visual stims that passed (a)"""
    reference_check: ReferenceLatencyResult | None = None
    """(c): as (b), if (b) didn't fail and the session has a photodiode block of the same
    stim"""
    diode_check: Any | None = None
    """(d) `npc_sync.DiodeVerification`: only if (b) is untestable"""
    has_visual_stims: bool | None = None
    """Whether the block's intervals use display times (None: not determined)"""
    usable_intervals: tuple[str, ...] = ()
    """For an invalid block: its intervals tables that are written anyway, because they
    don't use display times (the auditory table of an RFMapping block whose constant-lag
    display times failed the checks)"""

    @property
    def checks(self) -> dict[str, Any]:
        """The checks that were run, by criterion"""
        return {
            k: v
            for k, v in (
                ("a", self.frame_index_check),
                ("b", self.latency_check),
                ("c", self.reference_check),
                ("d", self.diode_check),
            )
            if v is not None
        }

    @property
    def summary(self) -> str:
        """One line, for the notes of an epoch: '' for a valid block that needed no checks"""
        if not self.is_valid:
            return f"invalid: {self.reason}"
        if self.display_time_source != "constant_lag":
            return ""
        if not self.has_visual_stims:
            return (
                "constant-lag display times not checked: no visual stimuli timed by "
                "display times"
            )
        return "constant-lag display times accepted: " + "; ".join(
            f"({k}) {v.summary}" for k, v in self.checks.items()
        )

    def to_dict(self, include_arrays: bool = False) -> dict[str, Any]:
        """JSON-serialisable. `include_arrays`: also the PSTHs etc. of (b) and (c)."""
        a = self.frame_index_check
        d = self.diode_check
        return {
            "stim": self.stim,
            "is_valid": bool(self.is_valid),
            "reason": self.reason,
            "summary": self.summary,
            "display_time_source": self.display_time_source,
            "has_visual_stims": self.has_visual_stims,
            "usable_intervals": list(self.usable_intervals),
            "frame_index_check": (
                None
                if a is None
                else _jsonable(dataclasses.asdict(a) | {"summary": a.summary})
            ),
            "latency_check": (
                None
                if self.latency_check is None
                else self.latency_check.to_dict(include_arrays)
            ),
            "reference_check": (
                None
                if self.reference_check is None
                else self.reference_check.to_dict(include_arrays)
            ),
            "diode_check": (
                None
                if d is None
                else _jsonable(dataclasses.asdict(d) | {"summary": d.summary})
            ),
        }


def has_visual_stims(session: npc_sessions.DynamicRoutingSession, stim: str) -> bool:
    """Whether the stim's intervals tables have visual stim times from frame display times:
    the task (`TaskControl.DynamicRouting1`) and RFMapping (`TaskControl.VisRFMapping`).
    The others take their times from sync lines, the NI-DAQ or vsyncs (OptoTagging,
    AudRFMapping, LuminanceTest), or have no intervals table (Spontaneous,
    SpontaneousRewards)."""
    return stim.startswith(session.task_stim_name) or "RFMapping" in stim


def get_diode_verification(
    session: npc_sessions.DynamicRoutingSession,
    vsync_times: npt.ArrayLike,
) -> Any | None:  # noqa: ANN401
    """Criterion (d): `npc_sync.verify_constant_lag` on the sync patch's photodiode record
    for the block with these vsyncs. None with an npc_sync that doesn't have it."""
    import npc_sync

    verify = getattr(npc_sync, "verify_constant_lag", None)
    if verify is None:
        return None
    patch = session.sync_patch
    vsyncs = np.asarray(vsync_times, dtype=float)
    record = (
        None
        if patch is None or not len(vsyncs)
        else patch.diode_block(float(vsyncs[0]), float(vsyncs[-1]))
    )
    return verify(record)


def get_stim_validity(
    session: npc_sessions.DynamicRoutingSession,
    stim: str,
    block: Any,  # noqa: ANN401
) -> StimValidity:
    """Validity of a stim block from its candidate display times (`block`: the
    `npc_stim.BlockDisplayTimes` in `session._stim_display_times[stim]`): see the module
    docstring for the rule. Photodiode blocks, and constant-lag blocks without visual stims,
    are valid without checks: nothing is loaded for them. Spike times are loaded for a
    constant-lag block with visual stims that passes (a)."""
    source = str(block.source)
    if block.display_times is None:
        return StimValidity(
            stim, False, f"no frame display times: {block.reason}", "unavailable"
        )
    if source != "constant_lag":
        return StimValidity(stim, True, "", source)
    a = getattr(block, "frame_index_check", None)
    if not has_visual_stims(session, stim):
        return StimValidity(
            stim, True, "", source, frame_index_check=a, has_visual_stims=False
        )
    usable: tuple[str, ...] = ()
    if "RFMapping" in stim:
        from npc_sessions import utils

        usable = (utils.get_taskcontrol_intervals_table_name("AudRFMapping"),)

    def result(
        is_valid: bool,
        reason: str = "",
        b: QuartileLatencyResult | None = None,
        c: ReferenceLatencyResult | None = None,
        d: Any | None = None,  # noqa: ANN401
    ) -> StimValidity:
        return StimValidity(
            stim,
            is_valid,
            (
                ""
                if is_valid
                else "photodiode not usable, and vsync + constant lag display times "
                f"rejected: {reason}"
            ),
            source,
            frame_index_check=a,
            latency_check=b,
            reference_check=c,
            diode_check=d,
            has_visual_stims=True,
            usable_intervals=() if is_valid else usable,
        )

    if a is None:
        return result(False, "(a) frame index check not run (npc_stim without it)")
    if a.passed is not True:
        return result(False, f"(a) {a.summary}")
    b = quartile_visual_latency(session, stim, display_times=block.display_times)
    if b.verdict == "failed":
        return result(False, f"(b) {b.summary}", b)
    c = reference_visual_latency(session, stim, display_times=block.display_times)
    d = None
    if b.verdict != "passed":
        d = get_diode_verification(session, block.vsync_times)
        if d is None or not d.passed:
            d_summary = (
                "photodiode verification not available (npc_sync without it)"
                if d is None
                else d.summary
            )
            return result(False, f"(b) {b.summary}; (d) {d_summary}", b, c, d)
    if c is not None and c.verdict == "failed":
        return result(False, f"(c) {c.summary}", b, c, d)
    return result(True, "", b, c, d)
