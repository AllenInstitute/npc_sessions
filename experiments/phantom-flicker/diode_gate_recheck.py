"""Re-run the photodiode gate on every replay-reconcilable block of the P0 manifest (the spans of
../sync-patch-manifest/diode_gate_check.csv), with the branch builder
(npc_sessions.timing.build_session_sync_patch -> npc_sync.build_sync_patch): box interval 1 uses
the block alignment (`align_diode_toggles`, `predict_box1_diode_edges`), box interval 3 the
existing `predict_diode_edges` (now ignoring copies of other spans).

Output: diode_gate_recheck.csv / .log (old vs new per span)
Run: PYTHONPATH="../../../npc_sync/src;../../../npc_stim/src;../../src;." ../../.venv/Scripts/python.exe diode_gate_recheck.py > diode_gate_recheck.log 2>&1
"""

from __future__ import annotations

import concurrent.futures as cf
import csv
import io
import logging
import traceback

import numpy as np
import pandas as pd

MAN = "../sync-patch-manifest/"


def run(args) -> list[dict]:
    session, row, spans_old = args
    import npc_io
    import npc_sync

    from npc_sessions import timing

    logging.disable(logging.WARNING)
    out = []
    try:
        raw = npc_sync.SyncDataset(row["sync_path"], on_invalid="ignore")
        stim_paths = [p for p in row["stim_paths"].split(";") if p]
        if row["source"] == "s3":
            # stim files on S3: npc_stim reads them through npc_io
            stim_paths = [npc_io.from_pathlike(p) for p in stim_paths]
        patch = timing.build_session_sync_patch(raw, stim_paths)
        recs = {b.block_index: b for b in patch.diode_blocks}
        for _, old in spans_old.iterrows():
            a, z = (float(x) for x in old.span.split("-"))
            sp = [s for s in patch.lines.get("stim_photodiode", []) if s.span[0] <= a + 1e-3 and s.span[1] >= z - 1e-3]
            rec = recs.get(int(old.block_index))
            r = dict(session=session, block_index=int(old.block_index), box=int(old.box), span=old.span,
                     old_fail=bool(old.fail_new), old_failures=old.failures_new, old_baseline_ms=old.baseline_ms,
                     old_after_ms=old.after_0p5s_ms,
                     diode_block=rec.status if rec else None, diode_block_reason=(rec.reason if rec else "")[:200])
            if not sp:
                r["new_status"] = "no diode span"
            else:
                s = sp[0]
                d = s.diagnostics or {}
                r.update(new_status=s.status, new_reason=s.reason[:200], new_method=s.method,
                         new_baseline_ms=(d.get("baseline_error_sec") or np.nan) * 1e3,
                         new_after_ms=(d.get("after_error_sec") or np.nan) * 1e3,
                         new_lat_r_ms=(d.get("latency_rising_sec") or np.nan) * 1e3,
                         new_lat_f_ms=(d.get("latency_falling_sec") or np.nan) * 1e3,
                         new_polarity_errors=d.get("n_polarity_errors_after"), new_inserted=d.get("n_inserted"))
            if rec is not None and rec.status == "cleaned":
                r["phantom_removed_block"] = rec.n_phantom_flicker
                r["glitch_removed_block"] = rec.n_glitch
            out.append(r)
    except Exception:  # noqa: BLE001
        for _, old in spans_old.iterrows():
            out.append(dict(session=session, block_index=int(old.block_index), box=int(old.box), span=old.span,
                            old_fail=bool(old.fail_new), new_status="error: " + traceback.format_exc().splitlines()[-1]))
    return out


def main() -> None:
    gate = pd.read_csv(MAN + "diode_gate_check.csv")
    rows = {r["session"]: r for r in csv.DictReader(open(MAN + "sessions.csv"))}
    jobs = [(s, rows[s], g) for s, g in gate.groupby("session")]
    out = []
    with cf.ProcessPoolExecutor(max_workers=8) as ex:
        for r in ex.map(run, jobs):
            out += r
    d = pd.DataFrame(out)
    d.to_csv("diode_gate_recheck.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 120)
    d["new_pass"] = d.new_status == "reconstructed"
    for box in (1, 3):
        x = d[d.box == box]
        print(f"box {box}: {len(x)} spans; old pass {int((~x.old_fail).sum())}; new pass {int(x.new_pass.sum())}")
        print(pd.crosstab(x.old_fail.map({False: 'old pass', True: 'old fail'}), x.new_status.str.slice(0, 40)).to_string())
        print()
    b1 = d[d.box == 1]
    print("box-1 spans, new outcome:")
    print(b1[["session", "span", "old_fail", "new_status", "new_baseline_ms", "new_after_ms", "new_lat_r_ms", "new_lat_f_ms", "diode_block", "new_reason", "diode_block_reason"]].round(3).to_string())
    print("\nbox-3 spans whose outcome changed:")
    b3 = d[(d.box == 3) & (d.old_fail == d.new_pass)]
    print(b3[["session", "span", "old_fail", "old_failures", "new_status", "new_reason", "old_baseline_ms", "new_baseline_ms", "old_after_ms", "new_after_ms"]].round(3).to_string())
    print("\nnew baseline (pre-span max |toggle residual|) ms quantiles, box 1:", b1.new_baseline_ms.quantile([0.5, 0.9, 1]).round(3).to_dict())
    print("old baseline ms quantiles, box 1:", b1.old_baseline_ms.quantile([0.5, 0.9, 1]).round(3).to_dict())


if __name__ == "__main__":
    main()
