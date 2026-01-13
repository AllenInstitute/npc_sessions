import npc_sessions
import npc_sessions.aind_data_schema

session_id = '676909_2023-12-13' # DR
suffix = f"_{session_id}"
output_directory = '.sample_aind_metadata'

session = npc_sessions.DynamicRoutingSession(session_id)

m = npc_sessions.aind_data_schema.get_data_description_model(session)
m.write_standard_file(output_directory, suffix=suffix)

m = npc_sessions.aind_data_schema.get_instrument_model(session)
m.write_standard_file(output_directory, suffix=suffix)

m = npc_sessions.aind_data_schema.get_acquisition_model(session)
m.write_standard_file(output_directory, suffix=suffix)