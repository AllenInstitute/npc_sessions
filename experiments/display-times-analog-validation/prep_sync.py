"""Export the sync-side inputs for the analog validation (read-only on the cache).

    C:/data/t2p_qc/venv_new/Scripts/python prep_sync.py <folder> [<folder> ...]

Writes C:/data/t2p_qc/analog/<folder>/sync.npz with:
  bc_t, bc_v              sync barcode onsets (s) and values (barcode_ephys line)
  v_rise, v_fall          vsync_stim edges (s)
  d_rise, d_fall          stim_photodiode edges (s)
  block{i}                vsync falling edges of block i (SyncDataset.vsync_times_in_blocks)
  total_seconds, invalid_intervals, discontinuities
"""
import pathlib, re, sys, logging, warnings
import numpy as np
import npc_sync

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from npc_ephys_barcodes import extract_barcodes_from_times  # local copy of npc_ephys.barcodes

warnings.simplefilter("ignore")
logging.disable(logging.WARNING)
CACHE = pathlib.Path("C:/data/t2p_cache")
OUT = pathlib.Path("C:/data/t2p_qc/analog")


def sync_path(folder):
    hs = sorted((p for p in (CACHE / folder).iterdir() if re.match(r"^\d{8}T\d{6}\.h5$", p.name)), key=lambda p: p.stat().st_size)
    return hs[-1]


def main(folder):
    p = sync_path(folder)
    s = npc_sync.SyncDataset(p)
    on = s.get_rising_edges("barcode_ephys", units="seconds")
    off = s.get_falling_edges("barcode_ephys", units="seconds")
    bc_t, bc_v = extract_barcodes_from_times(on, off, total_time_on_line=s.total_seconds)
    blocks = s.vsync_times_in_blocks
    out = dict(
        bc_t=np.asarray(bc_t, float), bc_v=np.asarray(bc_v, float),
        v_rise=s.get_rising_edges("vsync_stim", units="seconds"),
        v_fall=s.get_falling_edges("vsync_stim", units="seconds"),
        d_rise=s.get_rising_edges("stim_photodiode", units="seconds"),
        d_fall=s.get_falling_edges("stim_photodiode", units="seconds"),
        total_seconds=float(s.total_seconds),
        invalid_intervals=np.asarray(getattr(s, "invalid_intervals", ()), float).reshape(-1, 2),
        discontinuities=np.asarray(getattr(s, "timebase_discontinuities", ()), float).ravel(),
        sync_file=str(p),
        **{f"block{i}": b for i, b in enumerate(blocks)},
    )
    d = OUT / folder
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "sync.npz", **out)
    print(f"{folder}: {p.name} {len(bc_t)} barcodes, {len(blocks)} blocks {[len(b) for b in blocks]}, "
          f"invalid={out['invalid_intervals'].tolist()} disc={out['discontinuities'].tolist()}")


if __name__ == "__main__":
    for f in sys.argv[1:]:
        try:
            main(f)
        except Exception as e:
            print(f"{f}: ERROR {type(e).__name__}: {e}")
