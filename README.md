# npc_sessions
**n**euro**p**ixels **c**loud **sessions**

Tools and interfaces for working with behavior and epyhys sessions from the
Mindscope Neuropixels team, in the cloud.

## quickstart
To get minimal info on all tracked sessions:

```bash
pip install npc_sessions
```
```python
>>> import npc_sessions
>>> npc_sessions.tracked
```

Each object in the returned sequence has info about one session:
```python
>>> sessions = npc_sessions.tracked
>>> sessions[0].__class__.__name__
'SessionInfo'
>>> sessions[0].is_ephys
True
>>> any(s for s in sessions if s.date.year < 2021)
False
```