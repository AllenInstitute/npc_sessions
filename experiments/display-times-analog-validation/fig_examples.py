"""Report figures from raw NI-DAQ traces (read-only; reads S3) and the channel survey.
    C:/data/t2p_qc/venv_analog/Scripts/python fig_examples.py"""
import json, pathlib, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from raw_window import plot

HERE = pathlib.Path(__file__).parent
panels = {
    "ex_late_falling_edge.png": [("DRpilot_744279_20250116", 1, 1827, 0.10, 0.25, "§1a, box 1: 744279_20250116 b1 frame 1827 (falling edge 156 ms late)"),
                                 ("DRpilot_807082_20250930", 3, 81435, 0.10, 0.15, "§1a/§1b, box 3: 807082_20250930 b3, toggle frame 81435 (falling edge 58 ms late); frame 81436 re-placed")],
    "ex_box3_replaced.png": [("DRpilot_813586_20251205", 5, 27876, 0.10, 0.18, "§1b, box 3: 813586_20251205 b5, group starting at toggle 27876 (frame 27877 re-placed)"),
                             ("DRpilot_840160_20260501", 3, 177240, 0.10, 0.18, "§1b, box 3: 840160_20260501 b3, toggle 177240")],
    "ex_stall_flicker.png": [("DRpilot_744279_20250116", 1, 877, 0.25, 0.08, "§4, box 1 OptoTagging: 744279_20250116 b1 frame 877 (11-refresh stall; TTL shows nothing)"),
                             ("DRpilot_762526_20250320", 1, 78659, 0.12, 0.10, "§4, box 1 DR1 (rejected, +11 TTL flips): 762526_20250320 b1 frame 78659 (3-refresh stall)")],
    "ex_after_stall_timing.png": [("DRpilot_744279_20250116", 1, 185, 0.12, 0.08, "box 1 OptoTagging: 744279_20250116 b1 frame 185 (after a 4-refresh stall; analog - TTL = -13 ms)"),
                                  ("DRpilot_744279_20250116", 0, 41075, 0.12, 0.08, "box 1 RFMapping: 744279_20250116 b0 frame 41075 (after a 3-refresh stall)")],
}
for name, ps in panels.items():
    fig, axs = plt.subplots(len(ps), 1, figsize=(13, 4.2 * len(ps)))
    for ax, (f, b, fr, bef, aft, title) in zip(np.atleast_1d(axs), ps):
        plot(ax, f, b, fr, bef, aft, title=title)
    fig.tight_layout(); fig.savefig(HERE / "figs" / name, dpi=100); plt.close(fig)
    print(name)

# channel survey
rows = []
for l in open(HERE / "survey_channels.jsonl"):
    r = json.loads(l)
    d = re.search(r"_(\d{8})$", r["folder"]).group(1)
    if r["status"] != "ok":
        rows.append((d, np.nan, np.nan)); continue
    w = r["windows"]
    rows.append((d, max(max(x["f10"][0], x["f30"][0]) for x in w), max(max(max(x["f10"][c], x["f30"][c]) for x in w) for c in range(1, 8))))
import datetime
dt = [datetime.datetime.strptime(x[0], "%Y%m%d") for x in rows]
fig, ax = plt.subplots(figsize=(11, 3.2))
ax.plot(dt, [x[1] for x in rows], "o", ms=4, label="AI0")
ax.plot(dt, [x[2] for x in rows], "x", ms=4, color="gray", label="max of AI1-AI7")
bl = [t for t, x in zip(dt, rows) if np.isnan(x[1])]
ax.plot(bl, np.full(len(bl), -0.03), "|", color="r", ms=10, label="no NI-DAQ data on S3")
ax.set_ylabel("fraction of 1-200 Hz spectrum\nin the 10 / 30 Hz band"); ax.legend(fontsize=7)
ax.set_title("NI-DAQ channel survey (2 s at 20/40/60/80 % of each recording)", fontsize=9)
fig.tight_layout(); fig.savefig(HERE / "figs" / "channel_survey.png", dpi=110)
print("channel_survey.png")
