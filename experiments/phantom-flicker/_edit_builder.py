"""One-off: rewrite the vsync + photodiode section of npc_sync.patch.build_sync_patch (phantom
flicker removal, box-1 span prediction). Kept for provenance; not needed to reproduce results."""

import pathlib

p = pathlib.Path("../../../npc_sync/src/npc_sync/patch.py")
s = p.read_text(encoding="utf-8")
start = s.index("def build_sync_patch(")
end = s.index("    # --- periodic lines ---------------------------------------------------------------")
new = pathlib.Path("_builder_new.txt").read_text(encoding="utf-8")
s = s[:start] + new + s[end:]
helpers = pathlib.Path("_builder_helpers.txt").read_text(encoding="utf-8")
s = s.replace("def _extend_unknown_spans_over_splice(", helpers + "def _extend_unknown_spans_over_splice(", 1)
p.write_text(s, encoding="utf-8")
print("ok")
