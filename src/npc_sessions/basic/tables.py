"""
- project
    - subjects
    - sessions

    - subject
        - sessions
        * project

    - session
        * subject
        * project
"""
import enum
import pathlib

import pandas as pd

from npc_sessions.basic import Session

class Project(enum.Enum):
    DG = 'dynamic_gating'
    DR1 = 'dynamic_routing_aud_vis_1'

sessions = pd.read_csv(
    pathlib.Path(__file__).parent / 'sessions.csv',
    parse_dates=['datetime'],
    dtype_backend='pyarrow',
    )
print(sessions)
