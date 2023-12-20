from typing import Any, NamedTuple

import pytest

import npc_sessions

_kwargsets = (
    dict(is_ephys=False, is_sync=False), # fastest
    dict(is_ephys=False),
    dict(), # slowest, but most complete
    ) 

@pytest.fixture(params=_kwargsets)
def kwargsets(request):
    """Fixture to test different combinations of kwargs applied to Session"""
    return request.param

class Parameters(NamedTuple):
    
    session_id: str
    session_kwargs: dict[str, Any]
    """session-specific config: kwargsets will be updated with this"""
    expected: dict[str, dict[str, Any]]

opto_sessions: tuple[Parameters, ...] = (
    
    # early experiments with galvo ------------------------------------- #
    # always a single device 
    
    Parameters(
        session_id='636766_2023-01-23',
        session_kwargs=dict(),
        expected=dict(
            eval={
                "df.index.size": 531,
                "df.query('is_opto').index.size": 313,
                "df.query('is_opto').opto_stim_name.unique().size": 3,
                },
            ),
        ),
    
    # a 'simple' mode opto experiment ---------------------------------- #
    
    Parameters(
        session_id='670248_2023-08-01',
        session_kwargs=dict(),
        expected=dict(
            eval={
                "df.index.size": 538,
                "df.query('is_opto').index.size": 172,
                "df.query('is_opto').opto_stim_name.unique().size": 1,
                },
            ),
        ),
    # - single device
    # - trialOptoOnsetFrame values in subarrays [specific bug]
    # - trialGalvoVoltage only one value per trial (but also only one
    #   possibility in `galvoVoltage`) [specific bug]
    
    # after multi-device update ---------------------------------------- #
    
    # multiple devices possible:
    # - arrays that were [1 x ntrials] now may be:
    #      [n_devices x ntrials] (e.g. trialOptoVoltage)
    #   or [n_devices x nlocations x ntrials] (`trialGalvoVoltage`)  
    
    Parameters(
        session_id='658096 2023-08-15',
        session_kwargs=dict(is_sync=False),
        expected=dict(
            eval={
                "df.index.size": 532,
                "df.query('is_opto').index.size": 187,
                "df.query('is_opto').opto_stim_name.unique().size": 6,
                },
            ),
        ),
    # - behavior only, no ephys
)

@pytest.mark.parametrize("session, session_kwargs, expected", opto_sessions)
def test_opto(session, session_kwargs, expected, kwargsets):
    s = npc_sessions.Session(session, **kwargsets | session_kwargs)
    df = s.trials[:]    # get nwb module
    for k, v in expected.get('eval', {}).items():
        assert (result := eval(k)) == v, f'{s!r} opto mismatch: {k}={result} != expected {v}'
    assert all(df.query('is_opto').start_time < df.query('is_opto').opto_start_time), f'{s!r} opto_start_time < start_time'
    assert all(df.query('is_opto').opto_start_time < df.query('is_opto').opto_stop_time), f'{s!r} opto_stop_time < opto_start_time'
