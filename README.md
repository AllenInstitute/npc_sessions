# npc_sessions
**n**euro**p**ixels **c**loud **sessions**
	
Tools for accessing data and metadata for behavior and epyhys sessions from the
Mindscope Neuropixels team - in the cloud.

[![PyPI](https://img.shields.io/pypi/v/npc-sessions.svg?label=PyPI&color=blue)](https://pypi.org/project/npc-sessions/)
[![Python version](https://img.shields.io/pypi/pyversions/npc-sessions)](https://pypi.org/project/npc-sessions/)

[![Coverage](https://img.shields.io/codecov/c/github/alleninstitute/npc_sessions?logo=codecov)](https://app.codecov.io/github/AllenInstitute/npc_sessions)
[![CI/CD](https://img.shields.io/github/actions/workflow/status/alleninstitute/npc_sessions/publish.yml?label=CI/CD&logo=github)](https://github.com/alleninstitute/npc_sessions/actions/workflows/publish.yml)
[![GitHub issues](https://img.shields.io/github/issues/alleninstitute/npc_sessions?logo=github)](https://github.com/alleninstitute/npc_sessions/issues)


## quickstart
Make a conda environment with python>=3.9 and simply pip install the npc_sessions package:

```bash
conda create -n npc_sessions python>=3.9
conda activate npc_sessions
pip install npc_sessions
```

Get some minimal info on all the tracked sessions available to work with:
```python
>>> from npc_sessions import tracked as tracked_sessions;

# each record in the sequence has info about one session:
>>> tracked_sessions[0]._fields
('session', 'subject', 'date', 'idx', 'project', 'is_ephys', 'is_sync', 'allen_path')
>>> tracked_sessions[0].is_ephys
True
>>> all(s.date.year >= 2022 for s in tracked_sessions)
True

```

## to develop with conda
To install with the intention of contributing to this package:

1) create a conda environment:
```bash
conda create -n npc_sessions python>=3.9
conda activate npc_sessions
```
2) clone npc_sessions from github:
```bash
git clone git@github.com:AllenInstitute/npc_sessions.git
```
3) pip install all dependencies:
```bash
cd npc_sessions
pip install -e .
```
