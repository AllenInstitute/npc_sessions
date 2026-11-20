"""End to end on builder 9: session.licks for the two affected sessions (patch built fresh into
./patches_b9, not the shared cache)."""
import os, sys, json, pathlib
os.environ["NPC_SESSIONS_SYNC_PATCH_DIR"] = str(pathlib.Path(__file__).parent / "patches_b9")
import numpy as np, npc_sessions
from npc_sessions import timing
from evaluate import nearest, TOL
from common import lick_onsets
out = []
for sid in sys.argv[1:]:
    s = npc_sessions.DynamicRoutingSession(sid)
    sd = s.sync_data
    r = sd.get_rising_edges("lick_sensor", units="seconds"); f = sd.get_falling_edges("lick_sensor", units="seconds")
    src = np.asarray(sd.get_edge_sources("lick_sensor", "rising"))
    on = lick_onsets(sd, s.stim_paths)
    licks = s._licks
    licks = None if licks is None else licks[:]
    row = dict(session=sid, builder=sd.patch.builder if sd.patch else None, n_rising=len(r), n_falling=len(f),
               is_task=s.is_task, n_licks_table=None if licks is None else len(licks),
               timing_source=None if licks is None or "timing_source" not in licks else licks["timing_source"].value_counts().to_dict(),
               synthetic_without_lick_frame=int(np.sum(nearest(r[src == 1], on) > TOL)))
    print(row, flush=True); out.append(row)
json.dump(out, open("results/e2e_licks.json", "w"), indent=1, default=str)
