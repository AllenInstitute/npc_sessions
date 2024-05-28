import importlib.metadata

import dotenv
import numpy as np

from npc_sessions.sessions import *
from npc_sessions.trials import *
from npc_sessions.utils import *

__version__ = importlib.metadata.version("npc-sessions")

_ = dotenv.load_dotenv(
    dotenv.find_dotenv(usecwd=True)
)  # take environment variables from .env

np.seterr(divide="ignore", invalid="ignore")
# suppress common warning from sam's DynamicRoutingAnalysisUtils

Session = DynamicRoutingSession
"""Temp alias for backwards compatibility"""
