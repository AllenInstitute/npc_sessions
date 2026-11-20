"""Summary of survey.jsonl -> survey_summary.log"""
import json, collections, numpy as np, pathlib
rows = [json.loads(l) for l in open(pathlib.Path(__file__).parent / "survey.jsonl")]
err = [r for r in rows if "error" in r]; rows = [r for r in rows if "error" not in r]
print("files", len(rows), "errors", len(err)); [print("  ERR", r["session"], r["error"][:150]) for r in err]
clean = [r for r in rows if not r["n_invalid"] and not r["n_disc"]]
faulty = [r for r in rows if r["n_invalid"] or r["n_disc"]]
print("\n== clean files (no replays, no barcode discontinuity in the P0 manifest):", len(clean))
print("  with any drop found:", [(r["session"], r["drops_measured"]) for r in clean if r["drops_measured"]])
print("  with notes:", [(r["session"], r["notes"]) for r in clean if r["notes"]])
print("  with no camera lines:", sum(r["n_lines"] == 0 for r in clean))
print("  boundaries observed:", sum(r["n_boundaries_observed"] for r in clean), "of", sum(r["n_chunks"] for r in clean))
mf = [r["max_frac_observed"] for r in clean if r["max_frac_observed"] is not None]
print("  max |jump - integer| over observed boundaries: median over files %.3f, max %.3f" % (np.median(mf), max(mf)))
print("  boundaries with an estimate, not observed, |value| > 0.5:", sum(r["n_unobserved_nonzero"] for r in clean), [(r["session"], r["n_unobserved_nonzero"]) for r in clean if r["n_unobserved_nonzero"]][:20])
print("\n== files with replays or discontinuities:", len(faulty))
n_copies = sum(r["n_copies"] for r in faulty); n_meas = sum(k for r in faulty for _, k, _ in r["drops_measured"]); n_ass = sum(k for r in faulty for _, k in r["drops_assumed"])
print("  replayed copies:", n_copies, "| dropped chunks located on cameras:", n_meas, "| assumed at run start:", n_ass, "| net shift chunks:", sum(round(r["net_shift_s"] * 2) for r in faulty))
print("  files with assumed drops:", [(r["session"], len(r["drops_assumed"]), r["n_lines"]) for r in faulty if r["drops_assumed"]])
print("  files with notes:"); [print("    ", r["session"], r["notes"]) for r in faulty if r["notes"]]
print("  chunk sizes:", collections.Counter(r["chunk"] for r in faulty))
lags = np.array([l for r in faulty for l in r["lags_s"] if l is not None])
print("  lag from a located drop to the start of the next replayed run (s): n %d; ==0: %d (%.0f%%); >0: %d; median of >0 %.1f; p90 %.1f; max %.1f" % (len(lags), (lags == 0).sum(), 100 * (lags == 0).mean(), (lags > 0).sum(), np.median(lags[lags > 0]), np.percentile(lags[lags > 0], 90), lags.max()))
print("  lag histogram (s: count):", {k: int(v) for k, v in zip(*np.unique(np.minimum(lags, 10), return_counts=True))})
sizes = collections.Counter(k for r in faulty for _, k, _ in r["drops_measured"]); print("  located drop sizes (chunks: count):", dict(sorted(sizes.items())))
frac = [abs(v - round(v)) for r in faulty for _, _, v in r["drops_measured"]]; print("  |measured jump - integer| at located drops: median %.3f p99 %.3f max %.3f" % (np.median(frac), np.percentile(frac, 99), max(frac)))
files_displaced = [r for r in faulty if any(l for l in r["lags_s"] if l)]
print("  files with >= 1 displaced stretch (lag > 0):", len(files_displaced), "of", sum(1 for r in faulty if r["n_runs"]), "files with replays")
disp_s = sum(l for r in faulty for l in r["lags_s"] if l); print("  total recorded time between a drop and its run (upper bound of displaced data): %.0f s" % disp_s)
print("  barcode discontinuities: recorded clock %d intervals in %d files -> on the timebase %d intervals in %d files" % (sum(len(r["disc_recorded"]) for r in faulty), sum(bool(r["disc_recorded"]) for r in faulty), sum(len(r["disc_true"]) for r in faulty), sum(bool(r["disc_true"]) for r in faulty)))
for r in faulty:
    if r["disc_true"]: print("    still irregular:", r["session"], r["disc_true"], "recorded:", r["disc_recorded"])
print("  net shift files:", [(r["session"], r["net_shift_s"], r["n_runs"]) for r in faulty if r["net_shift_s"]])
mf = [r["max_frac_observed"] for r in faulty if r["max_frac_observed"] is not None]; print("  max |jump - integer| over observed boundaries: median %.3f max %.3f" % (np.median(mf), max(mf)))
print("  unobserved nonzero:", [(r["session"], r["n_unobserved_nonzero"]) for r in faulty if r["n_unobserved_nonzero"]])
