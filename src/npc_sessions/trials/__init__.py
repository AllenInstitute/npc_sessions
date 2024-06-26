from typing import Optional

import npc_ephys
import npc_io
import npc_lims
import npc_mvr
import npc_samstim
import npc_session
import npc_stim
import npc_sync

import npc_sessions.utils as utils
from npc_sessions.trials.property_dict import PropertyDict
from npc_sessions.trials.TaskControl import TaskControl
from npc_sessions.trials.TaskControl.DynamicRouting1 import DynamicRouting1
from npc_sessions.trials.TaskControl.LuminanceTest import LuminanceTest
from npc_sessions.trials.TaskControl.OptoTagging import OptoTagging
from npc_sessions.trials.TaskControl.RFMapping import AudRFMapping, VisRFMapping


def get_trials(
    *stim_path: npc_io.PathLike,
    sync_path_or_data: Optional[npc_sync.SyncPathOrDataset],
    **kwargs,
) -> tuple[TaskControl, ...]:
    """Get trials for a given stimulus file.

    Parameters
    ----------
    stim_path : npc_io.PathLike
        Path to stimulus file.

    Returns
    -------
    PropertyDict
        Trials for the given stimulus file.
    """
    trials: list[TaskControl] = []
    for stim in stim_path:
        stim_name = npc_io.from_pathlike(stim).name
        if "DynamicRouting1" in stim_name:
            trials.append(DynamicRouting1(stim, sync_path_or_data, **kwargs))
        elif "OptoTagging" in stim_name:
            trials.append(OptoTagging(stim, sync_path_or_data, **kwargs))
        elif "RFMapping" in stim_name:
            trials.append(AudRFMapping(stim, sync_path_or_data, **kwargs))
            trials.append(VisRFMapping(stim, sync_path_or_data, **kwargs))
        elif "LuminanceTest" in stim_name:
            trials.append(LuminanceTest(stim, sync_path_or_data, **kwargs))
        else:
            raise NotImplementedError(f"Stimulus file {stim_name} not supported.")
    return tuple(trials)


__all__ = [
    "PropertyDict",
    "TaskControl",
    "OptoTagging",
    "AudRFMapping",
    "VisRFMapping",
    "DynamicRouting1",
    "LuminanceTest",
    "get_trials",
]
