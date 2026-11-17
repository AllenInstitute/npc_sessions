"""Scan a NI-DAQ zarr: 2 s every `step` s, per-channel min/max/std and spectral band fractions (read-only)."""
import sys, numpy as np, zarr, s3fs, wavpack_numcodecs  # noqa: F401
fs = s3fs.S3FileSystem(anon=False)
z = zarr.open(s3fs.S3Map(sys.argv[1], s3=fs), mode="r"); tr = z["traces_seg0"]
step = int(sys.argv[2]) if len(sys.argv) > 2 else 300
n = tr.shape[0] // 30000
np.set_printoptions(linewidth=200)
for s in range(0, n, step):
    x = tr[s*30000:(s+2)*30000].astype(float)
    f = np.fft.rfftfreq(len(x), 1/30000); P = np.abs(np.fft.rfft(x - x.mean(0), axis=0)); tot = P[(f > 1) & (f < 200)].sum(0) + 1e-9
    fr = {h: np.round(P[(f > h-0.6) & (f < h+0.6)].sum(0)/tot, 2) for h in (10, 30)}
    print(f"t={s:5d}s min={x.min(0).astype(int)} max={x.max(0).astype(int)} f10={fr[10]} f30={fr[30]}")
