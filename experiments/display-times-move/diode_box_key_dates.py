"""When did stim files start containing `diodeBoxFrameInterval`? Scans every cached stim
file (Task 2 pilot ephys sessions). Output: diode_box_key_dates.log"""
import pathlib, h5py, collections
rows = []
for p in sorted(pathlib.Path("C:/data/t2p_cache").glob("*/*.hdf5")):
    try:
        with h5py.File(p, "r") as f:
            st = f["startTime"][()].decode() if "startTime" in f else None
            v = int(f["diodeBoxFrameInterval"][()]) if "diodeBoxFrameInterval" in f else None
            n = f["frameIntervals"].shape[0] if "frameIntervals" in f else None
    except Exception as e:
        print("FAIL", p, e); continue
    rows.append((st, v, p.parent.name, p.name, n))
rows.sort(key=lambda r: r[0] or "")
with_key = [r for r in rows if r[1] is not None]
without = [r for r in rows if r[1] is None and r[0]]
print("stim files:", len(rows), "| with key:", len(with_key), "| without:", len(without))
print("earliest with key:", with_key[0][:4] if with_key else None)
print("latest without key:", without[-1][:4] if without else None)
print("without key on/after earliest-with-key date:")
for r in without:
    if with_key and r[0] >= with_key[0][0][:8]:
        print("  ", r[:4])
print("values of key:", collections.Counter(r[1] for r in with_key))
print("value by session date (first 5 days with key):")
d = collections.OrderedDict()
for r in with_key:
    d.setdefault(r[0][:8], collections.Counter())[r[1]] += 1
for k in list(d)[:5]: print("  ", k, dict(d[k]))
