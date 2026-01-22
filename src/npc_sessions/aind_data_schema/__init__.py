"""
Create pydantic metadata models for the current variants of sessions (DR, Templeton, surface
recordings).

Code is tightly coupled to current version of npc_sessions, so it doesn't make sense to split into a
different repo, but required dependencies are made optional.
"""

try:
    import aind_data_schema  # noqa
except ImportError:
    raise ImportError(
        "Optional dependencies are required to use this module: install `npc_sessions[metadata]`"
    ) from None

from npc_sessions.aind_data_schema.acquisition import get_acquisition_model
from npc_sessions.aind_data_schema.data_description import get_data_description_model
from npc_sessions.aind_data_schema.instrument import get_instrument_model

__all__ = [
    "get_acquisition_model",
    "get_data_description_model",
    "get_instrument_model",
]
