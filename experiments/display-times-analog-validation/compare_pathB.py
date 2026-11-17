"""Path B (get_stim_frame_times(..., 'display_time'), the npc_sessions call), the A2 shim and stim
onsets/offsets (D) on the branch, against baseline and against the branch's path A (fixE).

    C:/data/t2p_qc/venv_analog/Scripts/python compare_pathB.py > compare_pathB.log

Inputs: C:/data/t2p_qc/out_issues/fixE_qc (qc_run.py on the branch), C:/data/t2p_qc/out/baseline,
C:/data/t2p_qc/out_issues/fixE (run_display_times.py on the branch). Read-only.
"""
import json, pathlib
import numpy as np, pandas as pd

Q = pathlib.Path("C:/data/t2p_qc/out_issues/fixE_qc")
BASE = pathlib.Path("C:/data/t2p_qc/out/baseline")
FIX = pathlib.Path("C:/data/t2p_qc/out_issues/fixE")
HERE = pathlib.Path(__file__).parent

rows = []
for jq in sorted(Q.glob("*.json")):
    f = jq.stem
    mq = json.loads(jq.read_text())
    if "error" in mq:
        rows.append(dict(folder=f, kind="session error (qc)", detail=str(mq["error"])[:120])); continue
    zq = np.load(Q / f"{f}.npz")
    mb = json.loads((BASE / f"{f}.json").read_text()) if (BASE / f"{f}.json").exists() else {}
    zb = np.load(BASE / f"{f}.npz") if (BASE / f"{f}.npz").exists() else {}
    mf = json.loads((FIX / f"{f}.json").read_text()) if (FIX / f"{f}.json").exists() else {}
    zf = np.load(FIX / f"{f}.npz") if (FIX / f"{f}.npz").exists() else {}
    stem_to_block = {pathlib.Path(p).stem: i for i, p in enumerate(mf.get("stim_paths", [])) if p}
    # B vs A (same code version): each B array must equal the A array of its matched block
    for k in [k for k in zq.files if k.startswith("B_")]:
        stem = k[2:]
        b = stem_to_block.get(stem)
        a = zf[f"A_block{b}"] if b is not None and f"A_block{b}" in getattr(zf, "files", []) else None
        if a is None:
            rows.append(dict(folder=f, stim=stem, kind="B has times, fixE A has none")); continue
        if not np.array_equal(a, zq[k]):
            rows.append(dict(folder=f, stim=stem, kind="B != fixE A", detail=f"max {np.nanmax(np.abs(a - zq[k])) * 1e3:.3f} ms" if len(a) == len(zq[k]) else f"len {len(a)} vs {len(zq[k])}"))
        # B vs baseline B
        kb = k
        if kb in getattr(zb, "files", []):
            if not (len(zb[kb]) == len(zq[k]) and np.array_equal(zb[kb], zq[k])):
                d = np.abs(zb[kb] - zq[k]) if len(zb[kb]) == len(zq[k]) else None
                rows.append(dict(folder=f, stim=stem, block=b, kind="B changed vs baseline",
                                 detail=f"{int(np.sum(d > 1e-9))} frames, max {d.max() * 1e3:.2f} ms" if d is not None else "length"))
        else:
            rows.append(dict(folder=f, stim=stem, block=b, kind="B new (baseline had exception)",
                             detail=str(mb.get("B_exceptions", {}).get(stem + ".hdf5", ""))[:120]))
    for stem, e in mq.get("B_exceptions", {}).items():
        st = pathlib.Path(stem).stem
        if f"B_{st}" in getattr(zb, "files", []):
            rows.append(dict(folder=f, stim=st, block=stem_to_block.get(st), kind="B lost (now exception)", detail=str(e)[:160]))
    if "B_error" in mq:
        rows.append(dict(folder=f, kind="B session error", detail=str(mq["B_error"])[:160]))
    # A2 shim == A
    for k in [k for k in zq.files if k.startswith("A2_block")]:
        ka = "A_block" + k[len("A2_block"):]
        if ka not in zq.files or not np.array_equal(zq[k], zq[ka]):
            rows.append(dict(folder=f, kind="A2 shim != A", detail=k))
    # D onsets/offsets
    for k in ("D_onsets", "D_offsets"):
        if k in zq.files and k in getattr(zb, "files", []):
            if not (len(zq[k]) == len(zb[k]) and np.array_equal(zq[k], zb[k])):
                rows.append(dict(folder=f, kind=f"{k} changed", detail=f"{len(zb[k])} -> {len(zq[k])}"))
        elif (k in zq.files) != (k in getattr(zb, "files", [])):
            rows.append(dict(folder=f, kind=f"{k} presence changed"))
    rows.append(dict(folder=f, kind="_checked", detail=str(len([k for k in zq.files if k.startswith("B_")]))))

df = pd.DataFrame(rows)
df.to_csv(HERE / "compare_pathB.csv", index=False)
chk = df[df.kind == "_checked"]
print(f"sessions checked: {len(chk)}; B arrays: {chk.detail.astype(int).sum()}")
print(df[df.kind != "_checked"].kind.value_counts().to_string())
# are B changes vs baseline exactly the blocks that changed in path A (compare_fixE.csv)?
cf = pd.read_csv(HERE.parent / "display-times-issues" / "compare_fixE.csv")
changedA = set(zip(cf[cf.change != "identical"].folder, cf[cf.change != "identical"].block))
bc = df[df.kind.isin(["B changed vs baseline", "B new (baseline had exception)", "B lost (now exception)"])]
inA = bc.apply(lambda r: (r.folder, r.block) in changedA, axis=1) if len(bc) else pd.Series(dtype=bool)
print(f"B changes vs baseline: {len(bc)}; in blocks changed in path A: {int(inA.sum())}; not: {int((~inA).sum())}")
if len(bc) and (~inA).any():
    print(bc[~inA].to_string(index=False))
other = df[~df.kind.isin(["_checked", "B changed vs baseline", "B new (baseline had exception)", "B lost (now exception)"])]
if len(other):
    print(other.to_string(index=False))
