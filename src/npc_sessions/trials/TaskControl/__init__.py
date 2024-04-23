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

import functools
from collections.abc import Iterable
from typing import Optional, SupportsFloat

import h5py
import npc_ephys
import npc_io
import npc_lims
import npc_mvr
import npc_samstim
import npc_session
import npc_stim
import npc_sync
import numpy as np
import numpy.typing as npt

import npc_sessions.trials.property_dict as property_dict
import npc_sessions.utils as utils


class TaskControl(property_dict.PropertyDict):
    _sync: npc_sync.SyncDataset | None
    _hdf5: h5py.File

    def __init__(
        self,
        hdf5: npc_stim.StimPathOrDataset,
        sync: npc_sync.SyncPathOrDataset | None = None,
        **kwargs,
    ) -> None:
        for k, v in kwargs.items():
            setattr(self, f"_{k.strip('_')}", v)
        self._hdf5 = npc_stim.get_h5_stim_data(hdf5)
        if not sync:
            # - all times in nwb are relative to start of first frame in hdf5
            # - there can only be one hdf5 file
            self._sync = None
        else:
            # - all times in nwb are relative to start of first sample on sync
            # - there can be multiple hdf5 files, all recorded on sync
            self._sync = npc_sync.get_sync_data(sync)

    @npc_io.cached_property
    def _input_data_times(self) -> npt.NDArray[np.float64]:
        """Best-estimate time of `getInputData()` in psychopy event loop, in seconds, from start
        of experiment. Uses preceding frame's vsync time if available."""
        return npc_stim.get_input_data_times(self._hdf5, self._sync)

    @npc_io.cached_property
    def _flip_times(self) -> npt.NDArray[np.float64]:
        """Best-estimate time of `flip()` in psychopy event loop, in seconds, from start
        of experiment. Uses frame's vsync time if available."""
        return npc_stim.get_flip_times(self._hdf5, self._sync)

    @npc_io.cached_property
    def _vis_display_times(self) -> npt.NDArray[np.float64]:
        """Best-estimate time of monitor update. Without sync, this equals frame times."""
        return npc_stim.get_vis_display_times(self._hdf5, self._sync)
