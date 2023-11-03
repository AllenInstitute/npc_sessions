import npc_lims
import upath

import npc_sessions
import npc_sessions.utils as utils

session = next(npc_sessions.get_sessions(is_ephys=False, is_sync=False))
nwb = session.nwb
utils.write_nwb(npc_lims.NWB_REPO / f'{session.id}.nwb.zarr', nwb)
