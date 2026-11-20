"""Scan cached patches (S3 shared cache) for #201's symptom in the lick-line diagnostics:
reconstructed spans with no lickFrames onset that still got a synthetic edge."""
import json, collections, upath
r = upath.UPath('s3://aind-scratch-data/dynamic-routing/cache/sync_patches')
rows = []
for f in sorted(r.glob('*.json')):
    d = json.loads(f.read_text())
    spans = d["lines"].get("lick_sensor", [])
    rec = [s for s in spans if s["status"] == "reconstructed"]
    noon_r = [s for s in rec if s["diagnostics"].get("n_onsets") == 0 and len(s["rising"])]
    noon_f = [s for s in rec if s["diagnostics"].get("n_onsets") == 0 and not len(s["rising"]) and len(s["falling"])]
    rows.append(dict(file=f.name, builder=d["builder"], n_invalid=len(d["invalid_intervals"]), n_lick_spans=len(spans),
                     n_reconstructed=len(rec), no_onset_with_rising=len(noon_r), no_onset_falling_only=len(noon_f)))
for x in rows:
    if x["n_lick_spans"]:
        print(x)
print("files", len(rows), "with lick spans", sum(1 for x in rows if x["n_lick_spans"]), "builders", collections.Counter(x["builder"] for x in rows))
json.dump(rows, open("results/scan_cache.json", "w"), indent=1)
