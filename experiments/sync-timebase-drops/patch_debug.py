"""Build a session's patch without the NI-DAQ sound step and print spans that stay unknown
on the vsync, photodiode and camera exposing lines. Usage: patch_debug.py <session id>"""
import sys, logging, json, collections
logging.basicConfig(level=logging.ERROR)
import numpy as np, npc_sessions, npc_sync
from npc_sessions import timing
s = npc_sessions.DynamicRoutingSession(sys.argv[1])
patch = timing.build_session_sync_patch(s._raw_sync_data, s.stim_paths)
print(sys.argv[1], "invalid", len(patch.invalid_intervals), "notes", patch.notes)
for line, spans in patch.lines.items():
    if line.startswith("bit") or "readout" in line: continue
    c = collections.Counter((sp.status, sp.method) for sp in spans)
    print(f" {line:22s}", dict(c))
    if line in ("vsync_stim", "stim_photodiode") or line.endswith("_cam_exposing"):
        reasons = collections.Counter(sp.reason[:160] for sp in spans if sp.status == "unknown")
        for r, n in reasons.most_common(6): print("      ", n, "x", r)
        if line == "stim_photodiode":
            for sp in spans:
                if sp.status == "unknown": print("       span", [round(x, 4) for x in sp.span], {k: v for k, v in sp.diagnostics.items() if not isinstance(v, (list, dict))})
