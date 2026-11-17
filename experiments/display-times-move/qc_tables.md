### Coverage

| status           |   sessions |
|:-----------------|-----------:|
| compared         |        441 |
| no_run:NotCached |          2 |
| no_run:Timeout   |          1 |

### Consistency, baseline vs new (sessions)

| code path                                                                       |   identical |   different |
|:--------------------------------------------------------------------------------|------------:|------------:|
| vsync blocks (`vsync_times_in_blocks`, unchanged code: sanity)                  |         441 |           0 |
| A: per-block display times (`frame_display_time_blocks` vs `get_display_times`) |         441 |           0 |
| A: per-block source (photodiode / none)                                         |         441 |           0 |
| A: block-to-stim-file matching                                                  |         441 |           0 |
| A2: deprecated npc_sync shim == npc_stim result (new only)                      |         441 |           0 |
| B: `get_stim_frame_times(display_time)` (npc_sessions path)                     |         406 |          35 |
| C: `get_stim_frame_times(vsync)`                                                |         441 |           0 |
| D: `stim_onsets`                                                                |         441 |           0 |
| D: `stim_offsets`                                                               |         438 |           3 |

### Block-level display times (all blocks in compared sessions where A ran)

|   blocks |   identical to baseline |   different |
|---------:|------------------------:|------------:|
|     2000 |                    2000 |           0 |

### Sessions that differ, and why

| session                 | differs in   | baseline A / B error                                            | new A / B error   | B arrays base→new   | B arrays changed   |
|:------------------------|:-------------|:----------------------------------------------------------------|:------------------|:--------------------|:-------------------|
| DRpilot_626791_20220815 | D_offsets    | none / none                                                     | none / none       | 2→2                 |                    |
| DRpilot_626791_20220816 | B, D_offsets | none / TypeError: None display-time block (no photodiode times) | none / none       | 0→1                 |                    |
| DRpilot_626791_20220817 | D_offsets    | none / none                                                     | none / none       | 2→2                 |                    |
| DRpilot_660023_20230808 | B            | none / TypeError: None display-time block (no photodiode times) | none / none       | 0→5                 |                    |
| DRpilot_666986_20230816 | B            | none / TypeError: None display-time block (no photodiode times) | none / none       | 0→6                 |                    |
| DRpilot_667252_20230927 | B            | none / TypeError: None display-time block (no photodiode times) | none / none       | 0→6                 |                    |
| DRpilot_674562_20231002 | B            | none / TypeError: None display-time block (no photodiode times) | none / none       | 0→5                 |                    |
| DRpilot_674562_20231003 | B            | none / TypeError: None display-time block (no photodiode times) | none / none       | 0→6                 |                    |
| DRpilot_674562_20231004 | B            | none / TypeError: None display-time block (no photodiode times) | none / none       | 0→6                 |                    |
| DRpilot_674562_20231005 | B            | none / TypeError: None display-time block (no photodiode times) | none / none       | 0→6                 |                    |
| DRpilot_699847_20240415 | B            | none / TypeError: None display-time block (no photodiode times) | none / none       | 0→7                 |                    |
| DRpilot_699847_20240417 | B            | none / TypeError: None display-time block (no photodiode times) | none / none       | 0→6                 |                    |
| DRpilot_699847_20240418 | B            | none / TypeError: None display-time block (no photodiode times) | none / none       | 0→4                 |                    |
| DRpilot_702136_20240305 | B            | none / TypeError: None display-time block (no photodiode times) | none / none       | 0→5                 |                    |
| DRpilot_702136_20240306 | B            | none / TypeError: None display-time block (no photodiode times) | none / none       | 0→4                 |                    |
| DRpilot_703881_20240509 | B            | none / TypeError: None display-time block (no photodiode times) | none / none       | 0→1                 |                    |
| DRpilot_703882_20240424 | B            | none / TypeError: None display-time block (no photodiode times) | none / none       | 0→7                 |                    |
| DRpilot_706401_20240422 | B            | none / TypeError: None display-time block (no photodiode times) | none / none       | 0→5                 |                    |
| DRpilot_706401_20240423 | B            | none / TypeError: None display-time block (no photodiode times) | none / none       | 0→5                 |                    |
| DRpilot_706401_20240424 | B            | none / TypeError: None display-time block (no photodiode times) | none / none       | 0→4                 |                    |
| DRpilot_706401_20240425 | B            | none / TypeError: None display-time block (no photodiode times) | none / none       | 0→5                 |                    |
| DRpilot_708016_20240501 | B            | none / TypeError: None display-time block (no photodiode times) | none / none       | 0→7                 |                    |
| DRpilot_708016_20240502 | B            | none / TypeError: None display-time block (no photodiode times) | none / none       | 0→7                 |                    |
| DRpilot_712141_20240612 | B            | none / TypeError: None display-time block (no photodiode times) | none / none       | 0→2                 |                    |
| DRpilot_712815_20240523 | B            | none / TypeError: None display-time block (no photodiode times) | none / none       | 0→5                 |                    |
| DRpilot_715706_20240626 | B            | none / TypeError: None display-time block (no photodiode times) | none / none       | 0→2                 |                    |
| DRpilot_715706_20240627 | B            | none / TypeError: None display-time block (no photodiode times) | none / none       | 0→2                 |                    |
| DRpilot_716718_20240730 | B            | none / TypeError: None display-time block (no photodiode times) | none / none       | 0→0                 |                    |
| DRpilot_726088_20240620 | B            | none / TypeError: None display-time block (no photodiode times) | none / none       | 0→5                 |                    |
| DRpilot_733780_20240905 | B            | none / TypeError: None display-time block (no photodiode times) | none / none       | 0→7                 |                    |
| DRpilot_733891_20240916 | B            | none / TypeError: None display-time block (no photodiode times) | none / none       | 0→7                 |                    |
| DRpilot_733891_20240917 | B            | none / TypeError: None display-time block (no photodiode times) | none / none       | 0→7                 |                    |
| DRpilot_733891_20240919 | B            | none / TypeError: None display-time block (no photodiode times) | none / none       | 0→7                 |                    |
| DRpilot_741148_20241016 | B            | none / TypeError: None display-time block (no photodiode times) | none / none       | 0→5                 |                    |
| DRpilot_761583_20241219 | B            | none / TypeError: None display-time block (no photodiode times) | none / none       | 0→5                 |                    |
| DRpilot_762526_20250320 | B            | none / TypeError: None display-time block (no photodiode times) | none / none       | 0→1                 |                    |
| DRpilot_818720_20251219 | B            | none / TypeError: None display-time block (no photodiode times) | none / none       | 0→5                 |                    |

### Session-level errors (A: all blocks; B: npc_sessions path)

| error                                                       |   A baseline |   A new |   B baseline |   B new |
|:------------------------------------------------------------|-------------:|--------:|-------------:|--------:|
| AssertionError:                                             |            1 |       1 |            1 |       1 |
| AssertionError: replayed sync data (invalid_intervals)      |           64 |      64 |           64 |      64 |
| AssertionError: timebase discontinuity                      |            2 |       2 |            1 |       1 |
| IndexError: index N is out of bounds for axis N with size N |            3 |       3 |            3 |       3 |
| TypeError: None display-time block (no photodiode times)    |            0 |       0 |           35 |       0 |
| none                                                        |          371 |     371 |          337 |     372 |

### Block sources, new (blocks in sessions where A ran)

| source      | reason (numbers → N)                                                                       |   blocks |
|:------------|:-------------------------------------------------------------------------------------------|---------:|
| photodiode  |                                                                                            |     1938 |
| unavailable | N diode flips (after basic filtering), but N vsyncs                                        |       36 |
| unavailable | N vsyncs, expected N from stim file                                                        |        1 |
| unavailable | estimator failed: AssertionError('DRpilot_N_N_N_DynamicRoutingN_N_N_N: Too many frames are |        8 |
| unavailable | estimator failed: AssertionError('DRpilot_N_N_N_DynamicRoutingN_N_N_N: found fewer than N  |        1 |
| unavailable | estimator failed: AssertionError('Expected first display time to have higher than usual la |        2 |
| unavailable | estimator failed: AssertionError()                                                         |        1 |
| unavailable | no stim file matched to block: number of vsyncs and diode-box interval can't be checked    |       13 |

### Sanity checks on new photodiode display times

| check                                                 | value          |
|:------------------------------------------------------|:---------------|
| blocks with photodiode display times                  | 1938           |
| n display times == n vsyncs                           | 1938/1938      |
| n display times == n frames in stim file              | 1938/1938      |
| strictly increasing                                   | 1938/1938      |
| blocks with >=1 display time before its vsync         | 169            |
| total frames with display time before vsync           | 4065           |
| median vsync→display latency, median over blocks (ms) | 27.66          |
| median latency, range over blocks (ms)                | 27.17 – 109.66 |
| p0.1 latency, min over blocks (ms)                    | -6.38          |
| p99.9 latency, max over blocks (ms)                   | 138.75         |
| min display-time interval, min over blocks (ms)       | 0.13           |

By diode-box frame interval:

|   diode_box_frame_interval |   blocks |   median_latency_ms |   min_p001_ms |   max_p999_ms |   blocks_display_before_vsync |
|---------------------------:|---------:|--------------------:|--------------:|--------------:|------------------------------:|
|                          1 |     1151 |               27.61 |         -6.38 |        138.75 |                            86 |
|                          3 |      787 |               27.69 |         22.13 |         28.24 |                            83 |

Blocks with any display time before its vsync, or p99.9 latency > 60 ms (pre-existing behaviour, identical in baseline):

187 blocks (first 40 shown; full list in qc_blocks.csv)

| folder                  |   block | stim                                           |   diode_box_frame_interval |   n_display_before_vsync |   lat_ms_min |   lat_ms_median |   lat_ms_p999 |   lat_ms_max | identical_to_baseline   |
|:------------------------|--------:|:-----------------------------------------------|---------------------------:|-------------------------:|-------------:|----------------:|--------------:|-------------:|:------------------------|
| DRpilot_366122_20260306 |       3 | DynamicRouting1_366122_20260306_154813.hdf5    |                          3 |                        1 |        -2.52 |           27.67 |         27.95 |        30.11 | True                    |
| DRpilot_660023_20230808 |       4 | Spontaneous_660023_20230808_131619.hdf5        |                          1 |                        0 |        34.68 |           44.25 |         60.93 |        61.02 | True                    |
| DRpilot_662892_20230821 |       2 | Spontaneous_662892_20230821_130510.hdf5        |                          1 |                        0 |        42.69 |           44.26 |         60.98 |        61.11 | True                    |
| DRpilot_662892_20230822 |       2 | Spontaneous_662892_20230822_122954.hdf5        |                          1 |                        0 |        43.42 |           44.22 |         60.96 |        61.05 | True                    |
| DRpilot_662892_20230823 |       2 | Spontaneous_662892_20230823_142344.hdf5        |                          1 |                        0 |        42.99 |           44.25 |         60.98 |        61.08 | True                    |
| DRpilot_662892_20230824 |       2 | Spontaneous_662892_20230824_144954.hdf5        |                          1 |                        0 |        42.57 |           44.25 |         60.93 |        61.05 | True                    |
| DRpilot_664851_20231113 |       0 | RFMapping_664851_20231113_125122.hdf5          |                          1 |                        2 |       -45.76 |           27.76 |         28.34 |        28.46 | True                    |
| DRpilot_664851_20231113 |       4 | DynamicRouting1_664851_20231113_134144.hdf5    |                          1 |                        1 |        -7.32 |           27.46 |         27.63 |        27.69 | True                    |
| DRpilot_664851_20231116 |       4 | DynamicRouting1_664851_20231116_133302.hdf5    |                          1 |                        1 |       -10.72 |           27.49 |         27.62 |        27.67 | True                    |
| DRpilot_666986_20230815 |       5 | Spontaneous_666986_20230815_121636.hdf5        |                          1 |                        0 |        31.55 |           44.29 |         61.01 |        61.09 | True                    |
| DRpilot_666986_20230816 |       2 | Spontaneous_666986_20230816_153257.hdf5        |                          1 |                        0 |        43.37 |           44.24 |         60.93 |        61.06 | True                    |
| DRpilot_666986_20230816 |       4 | DynamicRouting1_666986_20230816_155324.hdf5    |                          1 |                        1 |        -2.8  |           27.46 |         28.06 |        28.17 | True                    |
| DRpilot_666986_20230817 |       2 | Spontaneous_666986_20230817_143955.hdf5        |                          1 |                        0 |        34.53 |           44.25 |         60.97 |        61.08 | True                    |
| DRpilot_666986_20230817 |       4 | DynamicRouting1_666986_20230817_150019.hdf5    |                          1 |                        1 |        -7.4  |           27.46 |         28.21 |        28.27 | True                    |
| DRpilot_667252_20230928 |       0 | RFMapping_667252_20230928_150117.hdf5          |                          1 |                        1 |       -24.24 |           27.52 |         27.63 |        27.67 | True                    |
| DRpilot_667252_20230928 |       4 | DynamicRouting1_667252_20230928_154120.hdf5    |                          1 |                        1 |        -3.48 |           27.48 |         28.15 |        28.22 | True                    |
| DRpilot_668755_20230828 |       4 | DynamicRouting1_668755_20230828_134632.hdf5    |                          1 |                        1 |        -2.65 |           27.47 |         28.06 |        28.14 | True                    |
| DRpilot_668755_20230829 |       3 | SpontaneousRewards_668755_20230829_134134.hdf5 |                          1 |                        1 |       -46.49 |           27.95 |         28.39 |        28.42 | True                    |
| DRpilot_676909_20231212 |       4 | DynamicRouting1_676909_20231212_134539.hdf5    |                          1 |                        1 |        -9.48 |           27.46 |         27.74 |        27.8  | True                    |
| DRpilot_676909_20231213 |       0 | RFMapping_676909_20231213_134420.hdf5          |                          1 |                        1 |        -3.23 |           27.58 |         28.12 |        28.22 | True                    |
| DRpilot_676909_20231214 |       4 | DynamicRouting1_676909_20231214_132021.hdf5    |                          1 |                        2 |       -30.31 |           27.47 |         27.64 |        27.7  | True                    |
| DRpilot_681532_20231017 |       4 | DynamicRouting1_681532_20231017_143927.hdf5    |                          1 |                        1 |        -4.33 |           27.54 |         28.43 |        28.55 | True                    |
| DRpilot_686740_20231023 |       1 | OptoTagging_686740_20231023_142650.hdf5        |                          1 |                        1 |       -92.57 |           27.31 |         28.75 |        28.86 | True                    |
| DRpilot_690706_20231127 |       0 | RFMapping_690706_20231127_151053.hdf5          |                          1 |                        1 |        -0.98 |           27.96 |         28.5  |        28.67 | True                    |
| DRpilot_690706_20231130 |       4 | DynamicRouting1_690706_20231130_131725.hdf5    |                          1 |                        1 |       -48.41 |           27.49 |         27.69 |        27.78 | True                    |
| DRpilot_699847_20240415 |       0 | RFMapping_699847_20240415_124856.hdf5          |                          1 |                        1 |       -14.06 |           29.18 |         31.37 |        31.8  | True                    |
| DRpilot_699847_20240415 |       2 | Spontaneous_699847_20240415_130740.hdf5        |                          1 |                        0 |        39.63 |           45.04 |         61.93 |        63.1  | True                    |
| DRpilot_699847_20240416 |       1 | OptoTagging_699847_20240416_123651.hdf5        |                          1 |                        1 |       -26.61 |           27.83 |         28.7  |        28.77 | True                    |
| DRpilot_702131_20240226 |       3 | DynamicRouting1_702131_20240226_161335.hdf5    |                          1 |                        1 |       -81.41 |           27.53 |         27.77 |        27.84 | True                    |
| DRpilot_702131_20240227 |       3 | DynamicRouting1_702131_20240227_140115.hdf5    |                          1 |                        1 |       -67.67 |           27.87 |         28.94 |        29.13 | True                    |
| DRpilot_702131_20240227 |       5 | Spontaneous_702131_20240227_143450.hdf5        |                          1 |                        0 |        74.79 |          109.66 |        138.75 |       140.89 | True                    |
| DRpilot_703333_20240408 |       2 | Spontaneous_703333_20240408_144610.hdf5        |                          1 |                        0 |        39.36 |           44.4  |         61.27 |        61.82 | True                    |
| DRpilot_703333_20240409 |       2 | Spontaneous_703333_20240409_132510.hdf5        |                          1 |                        0 |        37.16 |           44.52 |         61.28 |        61.8  | True                    |
| DRpilot_703333_20240410 |       2 | Spontaneous_703333_20240410_123952.hdf5        |                          1 |                        0 |        36.52 |           44.3  |         61.3  |        61.87 | True                    |
| DRpilot_703880_20240415 |       0 | RFMapping_703880_20240415_140310.hdf5          |                          1 |                        1 |        -1.14 |           28.05 |         28.69 |        28.83 | True                    |
| DRpilot_703880_20240415 |       2 | Spontaneous_703880_20240415_142056.hdf5        |                          1 |                        0 |        32.92 |           44.28 |         61.27 |        61.78 | True                    |
| DRpilot_703880_20240416 |       2 | Spontaneous_703880_20240416_145325.hdf5        |                          1 |                        0 |        32.56 |           44.16 |         61.22 |        61.65 | True                    |
| DRpilot_703882_20240422 |       0 | RFMapping_703882_20240422_142032.hdf5          |                          1 |                        1 |       -90.64 |           27.79 |         28.41 |        28.54 | True                    |
| DRpilot_703882_20240422 |       2 | Spontaneous_703882_20240422_143918.hdf5        |                          1 |                        0 |        29.77 |           44.54 |         61.14 |        61.69 | True                    |
| DRpilot_703882_20240422 |       5 | SpontaneousRewards_703882_20240422_160046.hdf5 |                          1 |                        1 |       -44.83 |           27.57 |         27.68 |        27.71 | True                    |

### Runtime (s per session, median)

|   A baseline |   A new |   B baseline |   B new |
|-------------:|--------:|-------------:|--------:|
|         4.84 |    4.55 |         4.92 |    4.56 |
