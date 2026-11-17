"""Copy sync .h5 + stim .hdf5 (top level only) for each ephys session to a local cache so
the baseline and new code read identical bytes without hitting the share twice.
Read-only on the share. Skips files already cached with matching size."""
import csv, os, re, shutil, sys
from concurrent.futures import ThreadPoolExecutor
ROOT = "//allen/programs/mindscope/workgroups/dynamicrouting/PilotEphys/Task 2 pilot"
DEST = sys.argv[1] if len(sys.argv) > 1 else "C:/data/t2p_cache"
jobs = []
for row in csv.DictReader(open("ephys_sessions.csv")):
    d = os.path.join(ROOT, row["folder"])
    for f in os.listdir(d):
        if re.match(r"^\d{8}T\d{6}\.h5$", f) or f.endswith(".hdf5"):
            jobs.append((os.path.join(d, f), os.path.join(DEST, row["folder"], f)))
def cp(j):
    src, dst = j
    try:
        if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
            return 0
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst + ".part"); os.replace(dst + ".part", dst)
        return os.path.getsize(dst)
    except Exception as e:
        print("FAIL", src, e, flush=True); return 0
n = 0
with ThreadPoolExecutor(8) as ex:
    for i, b in enumerate(ex.map(cp, jobs)):
        n += b
        if i % 200 == 0: print(i, len(jobs), round(n / 1e9, 1), "GB", flush=True)
print("done", len(jobs), "files", round(n / 1e9, 1), "GB copied", flush=True)
