import sys, numpy as np, npc_sync
from common import *
sid = sys.argv[1]
raw, stim_paths, base = load(sid)
bit = raw._line_to_bit(LICK)
t = np.asarray(raw.get_all_times("seconds"), float); b = raw.get_bit(bit).astype(int)
print("times monotonic:", bool(np.all(np.diff(t) >= 0)))
ch = np.ediff1d(b, to_begin=0)
rr, ff = edges(raw)
print("rising from words == get_rising_edges:", np.array_equal(np.sort(t[ch == 1]), np.sort(rr)), " falling:", np.array_equal(np.sort(t[ch == -1 % 256]) if False else np.sort(t[ch == -1]), np.sort(ff)))
dt = 1/raw.sample_freq
spans = sorted(s.span for s in base.lines[LICK])
def state_before(a):
    i = np.searchsorted(t, a - 1.5*dt, side="left") - 1   # last event strictly before the mask
    return b[i], t[i]
def state_after(z):
    i = np.searchsorted(t, z + 0.5*dt, side="left")       # first event at/after the mask end
    if ch[i] != 0:  # a lick edge: the state before it
        return 1 - b[i], t[i]
    return b[i], t[i]
pre = np.array([state_before(a) for a, z in spans]); post = np.array([state_after(z) for a, z in spans])
print("max distance to the nearest event used: before %.4f s, after %.4f s" % (np.max([a - x for (a, z), x in zip(spans, pre[:,1])]), np.max([x - z for (a, z), x in zip(spans, post[:,1])])))
print("spans high before:", int(pre[:,0].sum()), " high after:", int(post[:,0].sum()), " of", len(spans))
# gaps between consecutive spans
gaps = [(spans[i][1], spans[i+1][0]) for i in range(len(spans)-1)]
print("gap lengths min %.3f s" % min(z - a for a, z in gaps))
