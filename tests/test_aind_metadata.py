import tempfile

import npc_sessions

def test_aind_session_metadata():
    ephys_session = 'DRpilot_676909_20231213-08-03'
    behavior_session = '715706_20240529'
    for session in [behavior_session, ephys_session]:
        s = npc_sessions.DynamicRoutingSession(session)
        m = s._aind_session_metadata
        m.write_standard_file(tempfile.mkdtemp())
    
if __name__ == "__main__":
    import pytest
    pytest.main(['-s', __file__])