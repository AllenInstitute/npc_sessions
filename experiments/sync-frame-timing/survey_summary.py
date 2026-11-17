"""Summarise the replay survey CSV produced by npc_sync/scripts/scan_sync_faults.py (2026-11-03).

Input: ../../../npc_sync/sync_replay_issues.csv (consecutive-identical-0.5s-chunk detector;
pre-dates the npc_sync 0.2.7/0.2.8 slot-fingerprint detector, so counts are not identical
to `SyncDataset.invalid_intervals`).

Run: ../../.venv/Scripts/python.exe survey_summary.py
"""

import pathlib
import re

import numpy as np
import pandas as pd

SEP = re.compile(r"[\\/]")

csv = pathlib.Path(__file__).resolve().parents[3] / "npc_sync" / "sync_replay_issues.csv"
df = pd.read_csv(csv)
parts = df.path.map(lambda p: SEP.split(p))
df["folder"] = parts.map(lambda x: x[-2])
df["root"] = parts.map(lambda x: x[-3])
df["date"] = df.folder.str.extract(r"_(\d{8})$")[0]
print("survey file:", csv)
print("files scanned:", len(df), "| status:", df.status.value_counts().to_dict())
print("roots:", df.root.value_counts().to_dict())
r = df[df.n_replay_runs > 0].copy()
print("files with >=1 replay run:", len(r))
print("files by year (all):", df.date.str[:4].value_counts().sort_index().to_dict())
print("files by year (replay):", r.date.str[:4].value_counts().sort_index().to_dict())
print("sync versions (replay):", r.version.value_counts().to_dict())
print("sync versions (all):", df.version.value_counts().to_dict())
w, during = [], []
for s, d in zip(r.replay_windows_s.fillna(""), r.replay_during_stim.fillna("").astype(str)):
    segs = [x for x in s.split(";") if x]
    flags = [x for x in d.split(";") if x]
    for i, seg in enumerate(segs):
        a, z = map(float, seg.split("-"))
        w.append(z - a)
        during.append(int(float(flags[i])) if i < len(flags) else -1)
w = np.array(w)
during = np.array(during)
print(
    "replay windows:", len(w),
    "| during stim:", int((during == 1).sum()),
    "| not during stim:", int((during == 0).sum()),
    "| unknown:", int((during == -1).sum()),
)
print(
    "window duration incl. genuine last chunk (s): value counts:",
    pd.Series(w).value_counts().sort_index().to_dict(),
)
print("window duration quantiles [50,90,99,100]% (s):", np.quantile(w, [0.5, 0.9, 0.99, 1.0]).round(2).tolist())
print("runs per affected file: median", r.n_replay_runs.median(), "max", r.n_replay_runs.max())
print("files with >5 runs:")
print(
    r[r.n_replay_runs > 5][["folder", "n_replay_runs", "replayed_s", "version"]]
    .sort_values("n_replay_runs", ascending=False)
    .to_string(index=False)
)
