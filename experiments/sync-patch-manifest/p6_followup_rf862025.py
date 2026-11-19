import os, logging
os.environ["NPC_SESSIONS_SYNC_PATCH_DIR"]="none"; logging.basicConfig(level=logging.ERROR)
import numpy as np, npc_sessions, npc_sync
s=npc_sessions.DynamicRoutingSession("862025_2026-08-05")
rf=s.intervals["aud_rf_mapping_trials"][:]
st=rf.stim_start_time.to_numpy()
inside=npc_sync.in_intervals(st, s.sync_data.invalid_intervals)
print("aud RF trials", len(rf), "NaN", int(np.isnan(st).sum()), "onsets inside replay", np.round(st[inside],4).tolist())
