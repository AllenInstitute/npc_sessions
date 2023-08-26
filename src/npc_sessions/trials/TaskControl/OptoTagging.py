"""
getting Optotagging trials table requires:
- one or more hdf5 files with trial/stim data, called 'OptoTagging_*.hdf5'
- frame display times from sync, to use in place of frametimes in hdf5 file
- latency estimate for each stim presentation, to be added to frame
display times to get stim onset times
"""
import contextlib
import functools

import numpy as np
import numpy.typing as npt

import npc_sessions.utils as utils
from npc_sessions.trials.TaskControl import TaskControl


class OptoTagging(TaskControl):
    _stim_onset_times: npt.NDArray[np.float64]
    """`[1 x num trials]` onset time of each opto stim relative to start of
    sync. There should be no nans: the times will be used as trial start_time."""
    _stim_offset_times: npt.NDArray[np.float64]

    @functools.cached_property
    def start_time(self) -> npt.NDArray[np.float64]:
        with contextlib.suppress(AttributeError):
            return self._stim_onset_times
        return utils.safe_index(self._frame_times, self._hdf5["trialOptoOnsetFrame"][:])

    @functools.cached_property
    def stop_time(self) -> npt.NDArray[np.float64]:
        with contextlib.suppress(AttributeError):
            return self._stim_offset_times
        return self.start_time + self._hdf5["trialOptoDur"][:]

    @functools.cached_property
    def trial_index(self) -> npt.NDArray[np.int32]:
        """0-indexed"""
        return np.arange(len(self.start_time))

    @functools.cached_property
    def stim_id(self) -> npt.NDArray[np.int32]:
        return np.full_like(self.start_time, np.nan)
