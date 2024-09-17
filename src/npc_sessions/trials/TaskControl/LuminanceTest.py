"""
getting LuminanceTest trials table (for testing pupil size) requires:
- one or more hdf5 files with trial/stim data, called 'Luminance_*.hdf5'
- frame display times from sync, to use in place of frametimes in hdf5 file
- latency estimate for each stim presentation, to be added to frame
display times to get stim onset times

# >>> stim = npc_stim.get_h5_stim_data("//allen/programs/mindscope/workgroups/dynamicrouting/PilotEphys/Task 2 pilot/DRpilot_708019_20240322/LuminanceTest_708019_20240322_153324.hdf5")
# >>> sync = npc_sync.get_sync_data('//allen/programs/mindscope/workgroups/dynamicrouting/PilotEphys/Task 2 pilot/DRpilot_708019_20240322/20240322T153301.h5')
# >>> lum_trials = LuminanceTest(stim, sync)
# >>> assert not lum_trials.to_dataframe().empty
"""

from __future__ import annotations

from collections.abc import Iterable

import npc_io
import npc_stim
import npc_sync
import numpy as np
import numpy.typing as npt

from npc_sessions.trials.TaskControl import TaskControl


class LuminanceTest(TaskControl):

    def __init__(
        self,
        hdf5: npc_stim.StimPathOrDataset,
        sync: npc_sync.SyncPathOrDataset,
        ephys_recording_dirs: Iterable[npc_io.PathLike] | None = None,
        **kwargs,
    ) -> None:
        if sync is None:
            raise ValueError(
                f"sync data is required for {self.__class__.__name__} trials table"
            )
        self._ephys_recording_dirs = ephys_recording_dirs
        super().__init__(
            hdf5, sync, ephys_recording_dirs=ephys_recording_dirs, **kwargs
        )

    def find(self, key: str) -> npt.NDArray[np.bool_] | None:
        if key in self._hdf5:
            return ~np.isnan(self._hdf5[key][()])
        return None

    @npc_io.cached_property
    def _len_all_trials(self) -> int:
        return len(self._hdf5["trialStartFrame"][()])

    @npc_io.cached_property
    def _idx(self) -> npt.NDArray[np.int32]:
        """Used for extracting a subset of inds throughout all properties.
        Must be constructed from private properties directly from the hdf5 file"""
        return np.arange(self._len_all_trials)

    @npc_io.cached_property
    def start_time(self) -> npt.NDArray[np.float64]:
        """falling edge of first vsync in each trial"""
        return npc_stim.safe_index(
            self._flip_times, self._hdf5["trialStartFrame"][self._idx]
        )

    @npc_io.cached_property
    def stop_time(self) -> npt.NDArray[np.float64]:
        """falling edge of vsync after stimulus end + inter-stim frames"""
        return npc_stim.safe_index(
            self._flip_times,
            self._hdf5["trialStartFrame"][self._idx] + self._hdf5["framesPerLevel"][()],
        )

    @npc_io.cached_property
    def trial_index(self) -> npt.NDArray[np.int32]:
        return np.arange(self._len_all_trials)[self._idx]

    @npc_io.cached_property
    def level(self) -> npt.NDArray[np.int32]:
        # round because some values end up as -0.40000000000000013
        return np.round(self._hdf5["trialLevel"][self._idx], 3)

    @npc_io.cached_property
    def _len(self) -> int:
        return len(self._idx)


if __name__ == "__main__":
    import doctest

    import dotenv

    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )
