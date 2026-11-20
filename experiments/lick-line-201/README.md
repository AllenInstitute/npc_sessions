# lick-line-201

npc_sessions #201: fake lick onsets / falling edges in runs of replayed spans. Report:
`../../outputs/issue-201-lick-line.pdf`. Fixed in builder 9 (`timing._line_states_at_bounds`).

| Script | Output | What |
|---|---|---|
| `common.py` | `cache/` (not tracked) | loaders; base patch with lick spans unknown |
| `diagnose.py` | stdout | reproduces the issue's numbers on builder 8 |
| `inspect_runs.py`, `bound_states.py`, `bound_distances.py` | stdout, `results/bound_distances.json` | span groups, edges at +0.5 sample inside spans, data-word states (<= 6.2 ms to the nearest event) |
| `fix.py` | – | prototype of the fix |
| `evaluate.py` | `results/summary.jsonl`, `results/*.log` | builder 8 vs prototype on real spans and on injected spans with ground truth (seed 201) |
| `checks.py` | `results/checks.json` | what the remaining synthetic onsets are |
| `scan_cache.py` | `results/scan_cache.json` | shared S3 patch cache contents |
| `verify_src.py` | `results/verify_src.json` | builder 9 (src/) == prototype on all 8 sessions, 0 alternation errors |
| `e2e_licks.py` | `results/e2e_licks.json`, `.log` | builder 9 end to end: `sync_data` lick line and the licks table |
| `make_pdf.py` | `../../outputs/issue-201-lick-line.pdf` | report |
