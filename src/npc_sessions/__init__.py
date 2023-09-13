import dotenv
from npc_lims import tracked
from npc_session import *

from npc_sessions.sessions import *
from npc_sessions.trials import *
from npc_sessions.utils import *

Session = DynamicRoutingSession  # temp alias for backwards compatibility

_ = dotenv.load_dotenv(
    dotenv.find_dotenv(usecwd=True)
)  # take environment variables from .env
