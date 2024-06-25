import npc_sessions

def test_aind_session_metadata():
    s = npc_sessions.DynamicRoutingSession('DRpilot_676909_20231213-08-03')
    _ = s._aind_session_metadata
    
if __name__ == "__main__":
    import pytest
    pytest.main(['-s', __file__])