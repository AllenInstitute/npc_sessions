from collections.abc import Generator

import dotenv
from npc_lims import tracked

from npc_sessions.sessions import *
from npc_sessions.trials import *
from npc_sessions.utils import *

_ = dotenv.load_dotenv(
    dotenv.find_dotenv(usecwd=True)
)  # take environment variables from .env

Session = DynamicRoutingSession
"""Temp alias for backwards compatibility"""
