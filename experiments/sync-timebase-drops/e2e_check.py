"""End-to-end check of npc_sessions on sessions with dropped sync buffers.

Usage: python e2e_check.py <label> <session id> [...]   (label: 'new' or 'old', for the log)
  new: PYTHONPATH=../../../npc_sync/src;../../src   (npc_sync branch `timebase`, working tree)
  old: PYTHONPATH=<worktree of npc_sessions HEAD>/src (released npc_sync from the venv)
Set NPC_SESSIONS_SYNC_PATCH_DIR to a local dir (or 'none').
Output: one JSON line per session -> e2e_<label>.jsonl
"""
from __future__ import annotations
import json, logging, sys, time, traceback, pathlib
import numpy as np
logging.basicConfig(level=logging.ERROR)
import npc_sync, npc_sessions
from npc_sessions import timing

label, sessions = sys.argv[1], sys.argv[2:]
out_path = pathlib.Path(__file__).parent / f"e2e_{label}.jsonl"
print("npc_sync", npc_sync.__file__, "| npc_sessions", npc_sessions.__file__, "| builder", timing.BUILDER, "| patch root", timing.get_patch_root(), flush=True)

def jsonable(x):
    if isinstance(x, dict): return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [jsonable(v) for v in x]
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, float) and not np.isfinite(x): return None
    return x

for sid in sessions:
    t0 = time.time(); rec: dict = dict(session=sid, label=label)
    try:
        s = npc_sessions.DynamicRoutingSession(sid)
        patch = s.sync_patch
        sync = s.sync_data
        tb = getattr(patch, "timebase", None)
        fs = sync.sample_freq
        rec["timebase"] = None if tb is None else dict(
            drops=[(j.sample / fs, j.n_chunks, j.source) for j in tb.drops], n_copies=sum(j.n_chunks == -1 for j in tb.jumps),
            net_shift_s=tb.net_shift_samples / fs, notes=list(tb.notes))
        try:  # patch cache: where it is, and whether this run loaded it or built it
            ppath = timing.get_patch_path(str(s.id), s._raw_sync_data.content_hash)
            on_disk = timing.load_patch(ppath) if ppath is not None and ppath.exists() else None
            rec["patch_cache"] = dict(path=str(ppath), exists=on_disk is not None, created=patch.created, builder=patch.builder,
                                      saved_versions=on_disk[1] if on_disk else None, saved_created=on_disk[0].created if on_disk else None,
                                      saved_has_timebase=(on_disk[0].timebase is not None) if on_disk else None)
        except Exception as exc:
            rec["patch_cache_error"] = repr(exc)[:300]
        rec["n_invalid_intervals"] = len(sync.invalid_intervals)
        rec["invalid_s"] = float(sum(z - a for a, z in sync.invalid_intervals))
        rec["barcode_discontinuities"] = [[round(v, 4) for v in d] for d in sync.timebase_discontinuities]
        rec["patch_notes"] = list(patch.notes) if patch is not None else None
        rec["line_summary"] = {k: v for k, v in patch.summary().items() if any(v.values())} if patch is not None else None
        # blocks
        try:
            rep = s.sync_timing_report
            rec["blocks"] = [dict(i=int(r.block_index), stim=r.stim, status=r.status, reason=str(r.reason)[:200], n_stim=r.n_stim_frames, n_rec=int(r.n_vsyncs_recorded),
                                  n_spans=int(r.n_patched_spans), resid_ms_max=(max(np.abs(r.anchor_residuals_ms)) if len(r.anchor_residuals_ms) else None),
                                  source=getattr(r, "display_time_source", None), n_interp=getattr(r, "n_interpolated_frames", None)) for r in rep.itertuples()]
        except Exception as exc:
            rec["blocks_error"] = repr(exc)[:300]
        # frame times per stim
        ft = {}
        for name, v in s.stim_frame_times.items():
            ft[name] = repr(v)[:200] if isinstance(v, Exception) else dict(n=len(v), nan=int(np.isnan(v).sum()), first=float(v[0]), last=float(v[-1]))
        rec["frame_times"] = ft
        # ephys timing
        if s.is_ephys:
            try:
                rec["ephys"] = [dict(device=t.device.name, status=t.status, reason=t.reason[:200], rate=t.sampling_rate, start=t.start_time, qc=dict(t.qc)) for t in s.ephys_timing_qc]
            except Exception as exc:
                rec["ephys_error"] = repr(exc)[:300]
        # trials / licks
        try:
            tr = s.trials[:]
            rec["trials"] = dict(n=len(tr), n_nan_stim_start=int(tr["stim_start_time"].isna().sum()), n_response=int(tr["response_time"].notna().sum()),
                                 first_start=float(tr["start_time"].iloc[0]), last_stop=float(tr["stop_time"].iloc[-1]))
            rec["is_task"] = bool(s.is_task)
        except Exception as exc:
            rec["trials_error"] = repr(exc)[:300]
        try:
            ep = s.epochs[:]
            rec["epoch_columns"] = list(ep.columns); rec["epochs"] = [dict(name=getattr(r, "stim_name", None) or getattr(r, "script_name", None), start=float(r.start_time), stop=float(r.stop_time), tags=list(r.tags), notes=str(r.notes)[:300]) for r in ep.itertuples()]
        except Exception as exc:
            rec["epochs_error"] = repr(exc)[:300]
        rec["ok"] = True
    except Exception as exc:
        rec |= dict(ok=False, error=repr(exc)[:500], tb=traceback.format_exc()[-2500:])
    rec["sec"] = round(time.time() - t0)
    with open(out_path, "a") as f: f.write(json.dumps(jsonable(rec), default=str) + "\n")
    out_path.with_suffix(".json").write_text(json.dumps([json.loads(x) for x in open(out_path)], indent=1))
    print(sid, "ok" if rec.get("ok") else rec.get("error"), rec["sec"], "s", flush=True)
