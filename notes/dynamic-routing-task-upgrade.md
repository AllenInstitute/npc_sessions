# DynamicRoutingTask upgrade

Date: 2026-07-28
Author: Codex

## What this dependency is used for

`DynamicRoutingTask` is both the task-control code that produced the behavioral
HDF5 files and a set of analysis utilities for reading those files.

`npc_sessions` does not ask `DynamicRoutingTask` to create the NWB trials
table directly. The flow is:

1. A session's `DynamicRouting1_*.hdf5` file contains the values recorded by
   the task, such as the stimulus shown on each trial, responses, rewards, opto
   voltage, and galvo position.
2. `DynamicRoutingTask.Analysis.DynRoutData` reads the common behavioral
   fields and exposes them as Python attributes.
3. `npc_sessions.trials.DynamicRouting1` combines those attributes with the
   raw HDF5 values and synchronized timestamps to create the NWB trials table.

The HDF5 format changed over the lifetime of the experiment. We need the
current `DynamicRoutingTask` library to read both current and legacy files.
We do not need to support old releases of the library.

## Dependency change

The upper dependency cap on `DynamicRoutingTask` was removed:

```toml
DynamicRoutingTask>=0.1.136
```

Version `0.1.136` is the minimum supported `DynamicRoutingTask` release. The
lock file currently resolves to that version.

The previous `<0.1.106` cap was added in commit `91c3e9b` as a rollback to the
version before upstream optogenetics changes. The cap protected legacy files,
but also prevented this package from receiving later fixes.

## What the legacy opto format looked like

Before the opto format changed, galvo X and Y were stored together:

| Purpose | Legacy HDF5 field | Typical shape |
| --- | --- | --- |
| Galvo coordinates used on each trial | `trialGalvoVoltage` | trials x 2, where the final values are X and Y |
| Available galvo locations for the session | `galvoVoltage` | locations x 2 |
| Label for each available location | `optoRegions` | one label per location |
| Laser voltage used on each trial | `trialOptoVoltage` | one value per trial |
| Available laser voltages | `optoVoltage` | one value per opto condition |

Some later legacy files added extra dimensions for multiple devices or
locations, but X and Y were still paired in `trialGalvoVoltage`.

These files did not have the modern `optoParams` group containing the complete
opto condition definitions. Converting a galvo voltage into a brain location
therefore required a separate calibration file for the rig. The calibration
maps galvo X/Y voltages to bregma X/Y coordinates.

## What changed in the newer format

Starting on 2024-03-29, the task began storing the per-trial coordinates in
separate fields:

| Legacy field | New fields |
| --- | --- |
| `trialGalvoVoltage[..., 0]` | `trialGalvoX` |
| `trialGalvoVoltage[..., 1]` | `trialGalvoY` |

Newer files can also contain an `optoParams` group. It describes opto
conditions such as device, label, galvo coordinates, bregma coordinates, and
waveform parameters. A per-trial parameter index connects each trial to the
appropriate condition.

`DynamicRoutingTask 0.1.106` changed `DynRoutData` to normalize old files into
the new Python representation. When it encountered `trialGalvoVoltage`, it
created `trialGalvoX` and `trialGalvoY` attributes in memory. This was useful
for upstream analysis because callers could use one representation regardless
of the file's age.

The same change also reconstructed missing opto parameters for legacy files.
To calculate bregma locations, it loaded the rig calibration from a hard-coded
Allen network path under `DynamicRoutingTask/OptoGui`.

## Why that broke `npc_sessions`

`npc_sessions` previously decided which HDF5 format it was reading by checking
the attributes on `DynRoutData`:

- `trialGalvoVoltage` meant a legacy file;
- `trialGalvoX` and `trialGalvoY` meant a newer file.

After upstream normalization, a legacy file appeared to be a newer file
because `DynRoutData` had created X and Y attributes. `npc_sessions` would then
try to read modern fields such as `optoParams` from a file that did not contain
them. That could break galvo and bregma columns in the trials table.

Loading a legacy file could also fail earlier if the hard-coded network
calibration path was unavailable. This matters for cloud and CI processing,
where the Allen file share is not mounted.

## How compatibility works now

- The raw HDF5 keys, rather than normalized `DynRoutData` attributes, determine
  whether a file uses combined `trialGalvoVoltage` or split
  `trialGalvoX`/`trialGalvoY`.
- Galvo coordinates used to build the trials table are read from the raw file,
  preserving its actual dimensions.
- Legacy calibration comes from `bregmaGalvoCalibrationData` embedded in the
  stimulus file when available. Otherwise, it uses the existing synchronized
  cloud copy.
- `DynRoutData` receives that calibration while loading a legacy file, so it
  does not need access to the hard-coded network path.
- Session-level opto detection reads `trialOptoOnsetFrame` directly instead of
  loading all of `DynRoutData`.

The result is one current library version with two supported input schemas:
legacy stimulus files and modern stimulus files.

## Trial tables

The fields directly affected by the format change were the galvo coordinates
and the bregma locations calculated from them. However, `DynRoutData` is loaded
before the NWB trials table is assembled. If legacy calibration or format
detection failed, the entire trials table could fail to build, including its
ordinary stimulus, response, and reward columns.

The related opto fields were also checked:

- opto onset and duration still come from their existing per-trial fields;
- opto power still comes from `trialOptoVoltage` and power calibration;
- opto device wavelength still comes from the recorded device name.

Trial-column construction remains local to `npc_sessions`. Regression tests
cover both legacy combined-galvo and modern split-coordinate layouts under
`DynamicRoutingTask 0.1.136`.

## Laser metadata

`LaserConfig.power` is derived from the local `opto_power` trial column. That
column uses `trialOptoVoltage` and the recorded power calibration. It does not
depend on whether galvo X and Y were stored together or separately. The new
tests confirm that the power column remains available when a legacy file is
read with `DynamicRoutingTask 0.1.136`.

The acquisition device inventory is independent of the dependency version.
It currently lists both 488 nm and 633 nm lasers, plus both galvos, for every
opto epoch. Selecting only the laser devices actually used in a session would
be a separate metadata improvement.

## Verification

- Three new compatibility tests pass.
- Ruff passes for the new compatibility code and tests.
- Mypy passes for the affected modules.
- All 38 repository tests collect successfully.
- `uv lock --check` passes.
- Source distribution and wheel builds pass.

The live S3-backed trial and AIND metadata tests could not be executed because
the local AWS SSO token was expired. After refreshing AWS credentials, run:

```powershell
pytest tests/test_task_trials.py tests/test_aind_metadata.py -n=0
```
