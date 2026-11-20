import os, time
os.environ["NPC_SESSIONS_SYNC_PATCH_DIR"] = r"C:\Users\ben.hardcastle\github\hub\experiments\qc-patched-sessions\patches_drops"
import npc_sessions, npc_lims
from npc_sessions import timing
t=time.time()
sid="708020_2024-05-17"
s = npc_sessions.DynamicRoutingSession(sid)
print(s.stim_paths)
p = npc_lims.get_h5_sync_from_s3(sid); print(p)
raw = timing.true_clock_view(p); print(raw, raw.invalid_intervals[:5], len(raw.invalid_intervals), time.time()-t)
