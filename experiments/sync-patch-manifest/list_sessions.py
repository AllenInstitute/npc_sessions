"""List every sync file to survey: the Task 2 Pilot share plus npc_lims-tracked sessions with
sync (S3), deduplicated by session (subject + date).

Output: sessions.csv (one row per sync file): session, source, sync_path, stim_paths (';'),
in_npc_lims, is_ephys, project.

Local copies in C:/data/t2p_cache are used where the cache has the same file (same name and
size) to reduce share reads.

Run (npc_sessions venv, needs npc_lims):
    ../../.venv/Scripts/python.exe list_sessions.py > list_sessions.log 2>&1
"""

from __future__ import annotations

import csv
import pathlib
import re
import time

import npc_lims

from common import CACHE, SHARE

HERE = pathlib.Path(__file__).parent
SYNC_RE = re.compile(r"^\d{8}T\d{6}\.h5$")

t0 = time.time()
rows = []
share_keys = set()
folders = sorted(p for p in SHARE.iterdir() if p.is_dir() and p.name.startswith("DRpilot_"))
print(f"share: {len(folders)} DRpilot_ folders ({time.time() - t0:.0f} s)", flush=True)
for i, d in enumerate(folders):
    m = re.match(r"DRpilot_(\d+)_(\d{8})", d.name)
    if not m:
        continue
    try:
        files = list(d.iterdir())
    except OSError as exc:
        print(f"  {d.name}: {exc}")
        continue
    syncs = sorted(f for f in files if SYNC_RE.match(f.name))
    stims = sorted(f for f in files if f.suffix == ".hdf5")
    key = f"{m.group(1)}_{m.group(2)[:4]}-{m.group(2)[4:6]}-{m.group(2)[6:]}"
    for s in syncs:
        local = CACHE / d.name / s.name
        path = s
        try:
            if local.exists() and local.stat().st_size == s.stat().st_size:
                path = local
        except OSError:
            pass
        stim_paths = []
        for st in stims:
            lst = CACHE / d.name / st.name
            stim_paths.append(lst if lst.exists() else st)
        rows.append(dict(session=d.name, key=key, source="share", sync_path=str(path), stim_paths=";".join(map(str, stim_paths))))
    if syncs:
        share_keys.add(key)
    if i % 200 == 0:
        print(f"  {i}/{len(folders)} folders ({time.time() - t0:.0f} s)", flush=True)

infos = [i for i in npc_lims.get_session_info() if i.is_sync]
print(f"npc_lims: {len(infos)} sessions with sync", flush=True)
lims_keys = {}
for info in infos:
    key = f"{info.subject}_{info.date}"
    lims_keys[key] = info
    if key in share_keys:
        continue
    try:
        sync = npc_lims.get_h5_sync_from_s3(info.id)
        stims = npc_lims.get_hdf5_stim_files_from_s3(info.id)
        stim_paths = [s.path for s in stims]
    except Exception as exc:  # noqa: BLE001
        print(f"  {info.id}: {type(exc).__name__}: {exc}", flush=True)
        rows.append(dict(session=str(info.id), key=key, source="s3", sync_path="", stim_paths="", error=f"{type(exc).__name__}: {exc}"))
        continue
    rows.append(dict(session=str(info.id), key=key, source="s3", sync_path=str(sync), stim_paths=";".join(map(str, stim_paths))))

for r in rows:
    info = lims_keys.get(r["key"])
    r["in_npc_lims"] = info is not None
    r["is_ephys"] = bool(info.is_ephys) if info else ""
    r["project"] = info.project if info else ""
    r.setdefault("error", "")

with open(HERE / "sessions.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["session", "key", "source", "sync_path", "stim_paths", "in_npc_lims", "is_ephys", "project", "error"])
    w.writeheader()
    w.writerows(rows)
n_share = sum(r["source"] == "share" for r in rows)
n_s3 = sum(r["source"] == "s3" for r in rows)
print(f"rows: {len(rows)} ({n_share} share sync files, {n_s3} npc_lims sessions not on the share); "
      f"npc_lims sessions on the share: {len(share_keys & set(lims_keys))}; {time.time() - t0:.0f} s")
