"""§4 of outputs/display-times-issues.md: blocks rejected for extra flips in stalls
(../display-times-issues/flip_extra.csv). For each: TTL edges in the block window minus frames,
and how the alignment accounts for every edge (survey_blocks.csv): phantom flicker (inside a
stim-recorded held frame), glitches, outside the block's toggles, unexplained, missing toggles.

Run: ../../.venv/Scripts/python.exe section4_accounting.py > section4_accounting.log
"""
import pandas as pd

pd.set_option("display.width", 250)
fx = pd.read_csv("../display-times-issues/flip_extra.csv")
sv = pd.read_csv("survey_blocks.csv")
cp = pd.read_csv("compare_population.csv")
m = fx.merge(sv, on=["folder", "block"], how="left", suffixes=("", "_s"))
m = m.merge(cp[["folder", "block", "B", "B_reason"]], on=["folder", "block"], how="left")
m["extra_edges"] = m.n_edges - m.n_frames
m["accounted"] = m.n_phantom + m.n_glitch + m.n_outside + m.n_unexplained - m.n_missing
m["all_extra_in_holds"] = (m.n_unexplained == 0) & (m.extra_edges == m.accounted)
cols = ["folder", "block", "stim", "excess", "extra_edges", "n_phantom", "n_holds_with_phantom", "n_glitch", "n_outside",
        "n_unexplained", "n_missing", "accounted", "validated", "B"]
print(m[cols].to_string())
print(f"\n§4 blocks: {len(m)}; every extra TTL edge accounted for: {int((m.extra_edges == m.accounted).sum())}; "
      f"no unexplained edge: {int((m.n_unexplained == 0).sum())}; phantom-flicker edges: {int(m.n_phantom.sum())} "
      f"(all inside stim-recorded held frames of >= 3 refreshes, by construction); glitches {int(m.n_glitch.sum())}")
print("after the patch (npc_stim, B):", m.B.value_counts().to_dict())
print(m[m.B != "photodiode"][["folder", "block", "stim", "B_reason"]].to_string())
