"""Smoke test of the moved plot functions (npc_stim) and the deprecated npc_sync shims on one
cached session. Writes PNGs to plots/."""
import pathlib, warnings, sys
import matplotlib; matplotlib.use("Agg")
import npc_stim, npc_sync
folder = pathlib.Path("C:/data/t2p_cache") / (sys.argv[1] if len(sys.argv) > 1 else "DRpilot_839146_20260512")
sync_path = next(folder.glob("*T*.h5")); stims = sorted(folder.glob("*.hdf5"))
sync = npc_sync.SyncDataset(sync_path)
blocks = npc_stim.get_display_times(sync, stims)
print([(b.block_index, b.stim_path.stem if b.stim_path else None, b.source) for b in blocks])
out = pathlib.Path("plots")
npc_stim.plot_stim_onsets(sync, blocks).savefig(out / f"{folder.name}_onsets.png", dpi=50)
npc_stim.plot_stim_offsets(sync, blocks).savefig(out / f"{folder.name}_offsets.png", dpi=50)
npc_stim.plot_diode_measured_sync_square_flips(sync).savefig(out / f"{folder.name}_diode.png", dpi=80)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    shim = npc_sync.SyncDataset(sync_path, stim_paths=stims)
    shim.plot_stim_onsets(); rate = shim.expected_diode_flip_rate
    print("shim diode rate", rate, "| deprecation warnings:", sorted({str(x.message)[:70] for x in w if issubclass(x.category, DeprecationWarning)}))
print("ok")
