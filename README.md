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

```python
>>> from npc_sessions import DynamicRoutingSession, get_sessions;

# each object is used to get metadata and paths for a session:         
>>> sesssion = DynamicRoutingSession('668755_2023-08-31')  
>>> session.is_ephys                                
True
>>> session.stim_paths[0].stem                      
'DynamicRouting1_626791_20220815_112336'

# data is processed on-demand to generate individual pynwb modules:
>>> session.subject                                 
subject pynwb.file.Subject at 0x...
Fields:
  age: P145D
  age__reference: birth
  date_of_birth: 2022-03-22 20:22:03-07:00
  genotype: wt/wt
  sex: M
  species: Mus musculus
  strain: C57BL6J(NP)
  subject_id: 626791

# a full NWBFile instance can also be generated with all currently-available data:
>>> session.nwb                                     # doctest: +SKIP
root pynwb.file.NWBFile at 0x...
Fields:
  acquisition: {
    lick spout <class 'ndx_events.events.Events'>
  }
  devices: {
    18005102491 <class 'pynwb.device.Device'>,
    18005114452 <class 'pynwb.device.Device'>,
    18005123131 <class 'pynwb.device.Device'>,
    18194810652 <class 'pynwb.device.Device'>,
    19192719021 <class 'pynwb.device.Device'>,
    19192719061 <class 'pynwb.device.Device'>
  }
   ...     
# loop over all currently-tracked ephys sessions using the session-generator:
>>> all(s.session_start_time.year >= 2022 for s in get_sessions())
True
>>> trials_dfs = {}
>>> for session in get_sessions():                  # doctest: +SKIP
...     trials_dfs[session.id] = session.trials[:]

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
