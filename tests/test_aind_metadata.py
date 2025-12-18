import tempfile

import npc_sessions
import npc_sessions.aind_data_schema

def test_data_description_metadata():
    ephys_session = 'DRpilot_676909_20231213-08-03'
    templeton_session = '628801_2022-09-20'
    behavior_session = '715706_20240529'
    for session in [behavior_session, ephys_session, templeton_session]:
        s = npc_sessions.DynamicRoutingSession(session)
        m = npc_sessions.aind_data_schema.get_data_description_model(s)
        m.write_standard_file(tempfile.mkdtemp())
    
if __name__ == "__main__":
    import pytest
    pytest.main(['-s', __file__])