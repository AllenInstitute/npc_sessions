"""
getting trials tables for experiments based on sam's TaskControl requires
timing information about each 'frame' in the experiment event loop, plus any
metadata about trials/stimuli.

Files required:

- a single hdf5 file, incl trial/stim data and frame display times (not
  synced with other devices)

- [optional for single hdf5] vsync times on sync clock, which give more precise, synced timing
  of each 'frame' of the experiment, used in place of the times in the hdf5

- [optional] for aud, opto, other non-visual stimuli: latency estimates from
  synced recordings of stimulus presentations, added to vsync times

Timing info will be obtained from the first available source from the left:

custom overrides (e.g non-vis stim latencies) > vsync times > hdf5 times

Info for custom overrides should be passed as kwargs to the constructor, e.g.:
TaskControl('stim.hdf5', sync='sync.h5', aud_stim_latencies=np.array([np.nan, 0.01, np.nan]))

`aud_stim_latencies` will be added as an instance attribute with a leading
underscore, and can then used where needed in the class, e.g.:
@property
def stim_start_times(self):
    return self._stim_onsets_from_vsyncs + self._aud_stim_latencies

# Concatenating trials
Where there are multiple sets of trials for the same experiment (e.g.
optotagging pre- and post-task) it should be possible to concatentate them
by just adding trials to an existing nwb table, via `nwb_file.add_trial(**trial)`
 - just be careful that shared parameters are referring to the same across all
   concatenated trials, e.g.  stim_idx 1 in both sets must be the same stim
 - any `trial_idx` type counters might need updating too
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Optional, SupportsFloat

import h5py
import numpy as np
import numpy.typing as npt

import npc_sessions.trials.property_dict as property_dict
import npc_sessions.utils as utils


class TaskControl(property_dict.PropertyDict):
    _sync: utils.SyncDataset | None
    _hdf5: h5py.File
    _frame_times: npt.NDArray[np.float64]
    _display_times: npt.NDArray[np.float64]

    def __init__(
        self,
        hdf5: utils.StimPathOrDataset,
        sync: utils.SyncPathOrDataset | None = None,
        **kwargs,
    ) -> None:
        for k, v in kwargs.items():
            setattr(self, f"_{k.strip('_')}", v)
        self._hdf5 = utils.get_h5_stim_data(hdf5)
        if not sync:
            # - all times in nwb are relative to start of first frame in hdf5
            # - there can only be one hdf5 file
            self._sync = None
            self._frame_times = np.concatenate(
                ([0], np.cumsum(self._hdf5["frameIntervals"][:]))
            )
            """Best-estimate time of 'frame' in event loop, in seconds, from start
            of experiment. Uses vsync time if available."""
            self._display_times = self._frame_times
            """Best-estimate time of screen update. Without sync, this equals frame times."""
        else:
            # - all times in nwb are relative to start of first sample on sync
            # - there can be multiple hdf5 files, all recorded on sync
            self._sync = utils.get_sync_data(sync)
            self._frame_times = utils.assert_stim_times(
                utils.get_stim_frame_times(
                    self._hdf5, sync=self._sync, frame_time_type="vsync"
                )[self._hdf5],
            )
            self._display_times = utils.assert_stim_times(
                utils.get_stim_frame_times(
                    self._hdf5, sync=self._sync, frame_time_type="display_time"
                )[self._hdf5],
            )

    def get_script_frame_time(
        self, frame: SupportsFloat | Iterable[SupportsFloat]
    ) -> npt.NDArray[np.float64]:
        return utils.safe_index(self._frame_times, frame)

    def get_vis_display_time(
        self, frame: SupportsFloat | Iterable[SupportsFloat]
    ) -> npt.NDArray[np.float64]:
        return utils.safe_index(self._display_times, frame)
