import dotenv

from npc_sessions.plots import *
from npc_sessions.sessions import *
from npc_sessions.trials import *
from npc_sessions.utils import *
from npc_sessions.widgets import PSTHWidget, session_widget

_ = dotenv.load_dotenv(
    dotenv.find_dotenv(usecwd=True)
)  # take environment variables from .env

Session = DynamicRoutingSession
"""Temp alias for backwards compatibility"""
