import pathlib

import pandas as pd

from npc_sessions import types


df = pd.read_csv(
    pathlib.Path(__file__).parent / 'sessions.csv'
    )
available_sessions = tuple(
    types.SessionSpec(
        project=project,
        subject=subject,
        dt=datetime,
    ) for project, subject, datetime in df.values
)
print(types.SessionSpec(dt=20221201, subject=676910) in available_sessions)