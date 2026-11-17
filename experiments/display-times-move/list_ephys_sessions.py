"""List ephys sessions in the Task 2 pilot share.

Ephys session := top-level folder DRpilot_<mouse>_<yyyymmdd> (no suffix, so excludes
*_surface_channels) containing a subfolder whose name starts with the folder name
(OpenEphys data: `<name>`, `<name>_probeABCDEF`, ...) or a `Record Node*` folder,
plus at least one sync .h5 file at top level.
Output: ephys_sessions.csv (folder, n_sync, n_stim_hdf5)
"""
import csv, os, re, sys, pathlib
ROOT = "//allen/programs/mindscope/workgroups/dynamicrouting/PilotEphys/Task 2 pilot"
pat = re.compile(r"^DRpilot_\d+_\d{8}$")
rows = []
for name in sorted(os.listdir(ROOT)):
    if not pat.match(name):
        continue
    d = os.path.join(ROOT, name)
    try:
        entries = list(os.scandir(d))
    except OSError as e:
        print("skip", name, e, file=sys.stderr); continue
    subdirs = [e.name for e in entries if e.is_dir()]
    is_ephys = any(s.startswith(name) or s.startswith("Record Node") for s in subdirs)
    if not is_ephys:
        continue
    files = [e.name for e in entries if e.is_file()]
    sync = [f for f in files if re.match(r"^\d{8}T\d{6}\.h5$", f)]
    stim = [f for f in files if f.endswith(".hdf5")]
    rows.append((name, len(sync), len(stim)))
with open("ephys_sessions.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["folder", "n_sync", "n_stim_hdf5"]); w.writerows(rows)
print(len(rows), "ephys sessions;", sum(r[1] == 1 for r in rows), "with exactly 1 sync;",
      sum(r[1] == 0 for r in rows), "with none;", sum(r[1] > 1 for r in rows), "with >1")
