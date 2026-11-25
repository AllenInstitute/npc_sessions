from npc_sessions.utils.atlas import *
from npc_sessions.utils.electrodes import *
from npc_sessions.utils.intervals import *
from npc_sessions.utils.misc import *
from npc_sessions.utils.units import *
from npc_sessions.utils.videos import *


def __getattr__(name: str):  # lazily-built aliases, see `intervals.__getattr__`
    if name in ("Interval", "UnitSelection"):
        from npc_sessions.utils import intervals

        return getattr(intervals, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
