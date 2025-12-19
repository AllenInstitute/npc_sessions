import tempfile

import pytest
import npc_sessions
import npc_sessions.aind_data_schema


@pytest.fixture(scope="module")
def ephys_session():
    return npc_sessions.DynamicRoutingSession('DRpilot_676909_20231213-08-03')


@pytest.fixture(scope="module")
def templeton_session():
    return npc_sessions.DynamicRoutingSession('628801_2022-09-20')


@pytest.fixture(scope="module")
def behavior_session():
    return npc_sessions.DynamicRoutingSession('715706_20240529')


@pytest.fixture(scope="module")
def all_sessions(ephys_session, templeton_session, behavior_session):
    return [behavior_session, ephys_session, templeton_session]

def test_data_description_metadata(all_sessions):
    for s in all_sessions:
        m = npc_sessions.aind_data_schema.get_data_description_model(s)
        m.write_standard_file(tempfile.mkdtemp())
    
def test_acquisition_metadata(all_sessions):
    for s in all_sessions:
        m = npc_sessions.aind_data_schema.get_acquisition_model(s)
        m.write_standard_file(tempfile.mkdtemp())
    
if __name__ == "__main__":
    import pytest
    pytest.main(['-s', __file__])