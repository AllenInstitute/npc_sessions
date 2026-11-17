"""For each session: locate NI-DAQ zarr on S3, read 2 s at 20/40/60/80 % of the recording and report,
per AI channel, the fraction of 1-200 Hz spectral magnitude in the 10 Hz (box-3) and 30 Hz (box-1) bands
and the peak-to-peak range. Read-only. Output: survey_channels.jsonl (appended)."""
import json, pathlib, sys
import numpy as np, zarr, s3fs, wavpack_numcodecs  # noqa: F401
from locate import find_session, nidaq_info

CACHE = pathlib.Path("C:/data/t2p_cache")
fs = s3fs.S3FileSystem()


def sync_stem(folder):
    import re
    hs = sorted((p for p in (CACHE / folder).iterdir() if re.match(r"^\d{8}T\d{6}\.h5$", p.name)), key=lambda p: p.stat().st_size)
    return hs[-1].stem


def survey(folder):
    out = dict(folder=folder)
    c = find_session(folder, sync_stem(folder))
    if not c:
        out["status"] = "blocked: no ecephys session on S3 within 10 min of sync start"; return out
    _, b, p = c[0]
    comp, clipped = nidaq_info(b, p)
    out.update(bucket=b, prefix=p, zarrs=comp, n_clipped=len(clipped))
    if not comp:
        out["status"] = "blocked: no NI-DAQ zarr"; return out
    z = zarr.open(s3fs.S3Map(f"{b}/{comp[0]}", s3=fs), mode="r")
    segs = sorted(k for k in z.keys() if k.startswith("traces_seg"))
    out["segments"] = {s: list(z[s].shape) for s in segs}
    out["channel_ids"] = [str(x) for x in z["channel_ids"][:]]
    tr = z[segs[0]]; n = tr.shape[0]
    rows = []
    for frac in (0.2, 0.4, 0.6, 0.8):
        i = int(n * frac)
        x = tr[i:i + 60000].astype(float)
        f = np.fft.rfftfreq(len(x), 1 / 30000); P = np.abs(np.fft.rfft(x - x.mean(0), axis=0))
        tot = P[(f > 1) & (f < 200)].sum(0) + 1e-9
        rows.append(dict(frac=frac,
                         f10=np.round(P[(f > 9.4) & (f < 10.6)].sum(0) / tot, 3).tolist(),
                         f30=np.round(P[(f > 29.4) & (f < 30.6)].sum(0) / tot, 3).tolist(),
                         ptp=np.ptp(x, 0).astype(int).tolist()))
    out["windows"] = rows
    out["status"] = "ok"
    return out


if __name__ == "__main__":
    for folder in sys.argv[1:]:
        try:
            r = survey(folder)
        except Exception as e:
            r = dict(folder=folder, status=f"error: {type(e).__name__}: {str(e)[:200]}")
        with open("survey_channels.jsonl", "a") as f:
            f.write(json.dumps(r) + "\n")
        w = r.get("windows", [])
        print(folder, r["status"], [(x["frac"], x["f10"][:2], x["f30"][:2], x["ptp"][0]) for x in w], flush=True)
