"""Run npc_stim.get_display_times (code path A only) on cached sessions with whatever npc_stim is
first on sys.path, and save per-block arrays + provenance. Read-only on data.
    PYTHONPATH=C:/data/t2p_qc/wt/npc_stim_fix/src <venv_new>/python run_display_times.py <label> [--workers N] [--only F,F]
Output: C:/data/t2p_qc/out_issues/<label>/<folder>.{npz,json}"""
import argparse, concurrent.futures, json, logging, pathlib, subprocess, sys, time, traceback, warnings
import pandas as pd
OUT = pathlib.Path("C:/data/t2p_qc/out_issues")
TIMEOUT = 1800

class H(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING); self.records = []
    def emit(self, r):
        self.records.append(f"{r.levelname}:{r.getMessage().splitlines()[0][:300]}")

def run_one(label, folder):
    import numpy as np, npc_stim
    from common import sync_path, stim_paths
    warnings.simplefilter("ignore")
    h = H(); logging.getLogger().addHandler(h); logging.getLogger().setLevel(logging.WARNING)
    meta = dict(folder=folder, npc_stim_file=npc_stim.__file__); arrays = {}
    t = time.time()
    try:
        res = npc_stim.get_display_times(str(sync_path(folder)), stim_paths(folder))
        meta["sources"] = [b.source for b in res]; meta["reasons"] = [b.reason for b in res]
        meta["stim_paths"] = [b.stim_path.name if b.stim_path else None for b in res]
        meta["box"] = [b.diode_box_frame_interval for b in res]
        for b in res:
            if b.display_times is not None:
                arrays[f"A_block{b.block_index}"] = b.display_times
    except Exception as e:
        meta["error"] = {"type": type(e).__name__, "msg": str(e)[:500]}; meta["traceback"] = traceback.format_exc()[-1500:]
    meta["seconds"] = round(time.time() - t, 2); meta["warnings"] = h.records
    d = OUT / label; d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"{folder}.npz", **arrays); (d / f"{folder}.json").write_text(json.dumps(meta, indent=1))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("label"); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only"); ap.add_argument("--one"); a = ap.parse_args()
    if a.one:
        return run_one(a.label, a.one)
    folders = a.only.split(",") if a.only else sorted(pd.read_csv("../display-times-move/ephys_sessions.csv").folder)
    from common import CACHE
    folders = [f for f in folders if (CACHE / f).exists()]
    def sub(f):
        t = time.time()
        try:
            p = subprocess.run([sys.executable, __file__, a.label, "--one", f], capture_output=True, text=True, timeout=TIMEOUT)
            return f"{f}: exit {p.returncode} {time.time() - t:.0f}s" + (f" {p.stderr.strip().splitlines()[-1:]}" if p.returncode else "")
        except subprocess.TimeoutExpired:
            (OUT / a.label).mkdir(parents=True, exist_ok=True)
            (OUT / a.label / f"{f}.json").write_text(json.dumps(dict(folder=f, error={"type": "Timeout"})))
            return f"{f}: timeout"
    with concurrent.futures.ThreadPoolExecutor(a.workers) as ex:
        for i, m in enumerate(ex.map(sub, folders)):
            print(i, m, flush=True)

if __name__ == "__main__":
    main()
