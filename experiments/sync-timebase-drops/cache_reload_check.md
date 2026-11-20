# Patch cache reload check

Second run in a new process (`e2e_release_reload.json`) against the first (`e2e_release.json`), released npc_sync 0.3.8. Patches are in `s3://aind-scratch-data/dynamic-routing/cache/sync_patches/`.

| Session | Patch file | Loaded, not rebuilt (same `created`) | Timebase in the loaded patch | Same invalid intervals | Same block statuses | Same frame-time counts | Same ephys statuses | Barcode discontinuities |
|---|---|---|---|---|---|---|---|---|
| 840160_2026-04-28 | `840160_2026-04-28_60030a394068.json` | True | True | True | True | True | True | 0 |
| 715709_2024-08-08 | `715709_2024-08-08_c786c2f32ef7.json` | True | True | True | True | True | True | 0 |
| 703880_2024-04-18 | `703880_2024-04-18_326e488667f0.json` | True | True | True | True | True | True | 0 |
| 708020_2024-05-17 | `708020_2024-05-17_6dc0b1c21830.json` | True | True | True | True | True | True | 0 |
| 714748_2024-06-28 | `714748_2024-06-28_4a6e30b42aed.json` | True | True | True | True | True | True | 0 |
