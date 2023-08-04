# npc_sessions
**n**euro**p**ixels **c**loud **sessions**

Tools and interfaces for working with behavior and epyhys sessions from the
Mindscope Neuropixels team, in the cloud.

## quickstart

```bash
pip install npc_sessions
```

Get some minimal info on all the tracked sessions available to work with:
```python
>>> from npc_sessions import tracked as tracked_sessions;

# each record in the sequence has info about one session:
>>> tracked_sessions[0]._fields
('session', 'subject', 'date', 'idx', 'project', 'is_ephys', 'is_sync')
>>> tracked_sessions[0].is_ephys
True
>>> all(s.date.year >= 2022 for s in tracked_sessions)
True

```