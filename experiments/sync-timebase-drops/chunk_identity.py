"""Which 0.5 s chunks around a complementary barcode pair are identical copies of another chunk,
at ANY lag (find_replayed_buffers only uses the most common lag)? Fingerprint = set of
(sample offset in chunk, bit state) of all events in the chunk."""
import logging, sys
import numpy as np, npc_sync
logging.disable(logging.WARNING)
CH = 50_000
def run(path, t0, t1):
    s = npc_sync.SyncDataset(path, on_invalid="ignore")
    times = np.asarray(s.get_all_times("samples"), dtype=np.int64); bits = np.asarray(s.get_all_bits(), dtype=np.int64)
    print("=" * 90); print(path.split("/")[3], t0, t1, "invalid:", [(a, z) for a, z in s.invalid_intervals if z > t0 and a < t1])
    fp = {}
    c0, c1 = int(t0 * 2), int(t1 * 2)
    for c in range(c0, c1):
        i, j = np.searchsorted(times, [c * CH, (c + 1) * CH])
        fp[c] = (tuple(times[i:j] - c * CH), tuple(bits[i:j]))
    bc = s._line_to_bit("barcode_ephys")
    for c in range(c0, c1):
        same = [d for d in range(c0, c1) if d != c and fp[d] == fp[c] and len(fp[c][0]) > 0]
        i, j = np.searchsorted(times, [c * CH, (c + 1) * CH])
        st = (bits[i:j] >> bc) & 1
        nb = int(np.sum(np.diff(np.r_[(bits[i - 1] >> bc) & 1, st]) != 0))
        # partial identity: events identical ignoring... also time-only identity at lags
        print(f"  chunk {c/2:8.1f}: {len(fp[c][0]):4d} events, barcode edges {nb:2d}, identical to: {[d/2 for d in same]}")
run("C:/data/t2p_cache/DRpilot_703880_20240418/20240418T130656.h5", 924.0, 933.0)
run("C:/data/t2p_cache/DRpilot_703880_20240418/20240418T130656.h5", 2007.0, 2017.0)
