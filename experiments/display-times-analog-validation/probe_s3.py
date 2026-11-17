"""Throughput / format probe: read 20 s of NI-DAQ analog from S3 zarr (read-only)."""
import time, sys, numpy as np, zarr, s3fs, wavpack_numcodecs  # noqa: F401
fs = s3fs.S3FileSystem(anon=False)
url = sys.argv[1]
z = zarr.open(s3fs.S3Map(url, s3=fs), mode="r")
tr = z["traces_seg0"]; print(tr.shape, tr.chunks, tr.dtype)
print("times_seg0 first/last", z["times_seg0"][:3], z["times_seg0"][-1])
print("channel_ids", z["channel_ids"][:])
t = time.time(); x = tr[30000*600:30000*620]; dt = time.time() - t
print(f"20 s read {dt:.2f}s ({20/dt:.1f}x realtime)")
np.save("probe_662892.npy", x)
f = np.fft.rfftfreq(len(x), 1/30000); P = np.abs(np.fft.rfft(x - x.mean(0), axis=0))
for fr in (10, 20, 30, 60):
    b = (f > fr-0.5) & (f < fr+0.5); print(fr, "Hz band frac", np.round(P[b].sum(0)/P[(f>1)&(f<200)].sum(0), 3))
print("ranges", x.min(0), x.max(0))
