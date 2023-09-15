import dotenv
from npc_lims import tracked
from npc_session import *

from npc_sessions.sessions import *
from npc_sessions.trials import *
from npc_sessions.utils import *

_ = dotenv.load_dotenv(
    dotenv.find_dotenv(usecwd=True)
)  # take environment variables from .env

Session = DynamicRoutingSession  # temp alias for backwards compatibility

session_kwargs = {
    '670248_2023-08-02': dict(suppress_errors=True),
    '628801_2022-09-20': dict(is_video=False),
}
issues = dict.fromkeys(('660023_2023-08-08', '666986_2023-08-14', '644867_2023-02-21', '628801_2022-09-20'))
sessions = [
    DynamicRoutingSession(info.session, **session_kwargs.get(info.session, {})) 
    for info in sorted(tracked, key=lambda x: x.date)
    if (
        info.is_uploaded 
        and info.session not in issues
    )
]
"""Uploaded sessions, tracked in npc_lims via `tracked_sessions.yaml`. Sessions
with known issues are excluded. Session-specific config from `session_kwargs`
is applied."""