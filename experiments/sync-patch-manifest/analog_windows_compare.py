"""Step 6 (validate), stage 2: compare patched photodiode edges and display times inside replayed
windows with the NI-DAQ analog photodiode (AI0; Task 2 pilot rigs from 2025-01-13, confirmed in
outputs/display-times-analog-validation.md). Reads only [span - 4 s, span + 3 s] of AI0 from the
S3 zarr. Read-only on S3.

Per span (checks defined before running):
  V1 digital diode edges (patched) vs analog transitions, per-polarity lag taken from the 3 s
     before the span: max error inside the span <= max error before + 0.5 ms.
  V2 toggle-frame display times (npc_stim on the patched sync) vs analog transitions, same rule.
  V3 every analog transition inside the span has a patched digital edge within 5 ms of
     (transition + lag), and vice versa (no missing / extra predicted toggles).

Uses functions from ../display-times-analog-validation (locate, barcode map, detect).
Run: C:/data/t2p_qc/venv_analog/Scripts/python analog_windows_compare.py > analog_windows_compare.log 2>&1
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

import numpy as np
import s3fs
import zarr

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "display-times-analog-validation"))
from detect import detect  # noqa: E402
from fetch_analog import FS, _pw, barcode_map, load_npy, npy_first_last  # noqa: E402
from locate import find_session, nidaq_info  # noqa: E402

ROOT = pathlib.Path("C:/data/t2p_qc/analog_windows")
s3 = s3fs.S3FileSystem()
rows = []


def pairs(x, ref, ref_pol):
    """For each x: (nearest ref - x, polarity of that ref)."""
    j = np.clip(np.searchsorted(ref, x), 1, len(ref) - 1)
    j = np.where(np.abs(x - ref[j - 1]) < np.abs(ref[j] - x), j - 1, j)
    return ref[j] - x, ref_pol[j]


for d in sorted(p for p in ROOT.iterdir() if (p / "prep.npz").exists()):
    P = np.load(d / "prep.npz")
    folder = d.name
    rec = dict(folder=folder)
    try:
        c = find_session(folder, str(P["sync_file"]))
        if not c:
            raise RuntimeError("blocked: no ecephys upload on S3 within 10 min of the sync start")
        _, bucket, prefix = c[0]
        comp, clipped = nidaq_info(bucket, prefix)
        if not comp:
            raise RuntimeError("blocked: no NI-DAQ zarr on S3")
        zpath = comp[0]
        exp, node = re.search(r"/(experiment\d+)_(Record Node \d+)#", zpath).groups()
        cl = [k for k in clipped if f"/{node}/{exp}/" in k]
        recs = sorted({k.split(f"/{exp}/")[1].split("/")[0] for k in cl if f"/{exp}/recording" in k})
        z = zarr.open(s3fs.S3Map(f"{bucket}/{zpath}", s3=s3), mode="r")
        segs = sorted(k for k in z.keys() if k.startswith("traces_seg"))
        ch_ids = [str(x) for x in z["channel_ids"][:]]
        if len(segs) != 1 or len(recs) != 1:
            raise RuntimeError(f"blocked: {len(segs)} zarr segments / {len(recs)} recordings (not handled)")
        base = next(k for k in cl if k.endswith(f"{recs[0]}/continuous/" + zpath.split('#')[1].replace('.zarr/', '') + "/sample_numbers.npy"))
        first, last, n_sn = npy_first_last(f"{bucket}/{base}")
        tr = z[segs[0]]
        if not (last - first + 1 == n_sn == tr.shape[0]):
            raise RuntimeError("blocked: NI-DAQ samples not contiguous")
        ttl_dir = base.replace("/continuous/", "/events/").replace("sample_numbers.npy", "TTL/")
        states = load_npy(f"{bucket}/{ttl_dir}states.npy").astype(int)
        sn = load_npy(f"{bucket}/{ttl_dir}sample_numbers.npy").astype(np.int64)
        line, pe, ps = barcode_map(states, sn, P["bc_t"], P["bc_v"])
        A = np.polyfit(pe, ps, 1)
        to_sync = lambda t: _pw(t, pe, ps, A)  # noqa: E731
        Ai = np.polyfit(ps, pe, 1)
        to_ni = lambda t: _pw(t, ps, pe, Ai)  # noqa: E731
        ch = ch_ids.index("AI0")
        for a, zz in P["spans"]:
            i0, i1 = int(round(to_ni(a - 4) * FS)) - first, int(round(to_ni(zz + 3) * FS)) - first
            r = dict(rec, span=f"{a:.3f}-{zz:.3f}")
            if i0 < 0 or i1 > tr.shape[0]:
                rows.append(dict(r, status="outside NI-DAQ recording"))
                continue
            x = np.asarray(tr[i0:i1, ch]).astype(np.float32)
            pos, pol, sw, _ = detect(x, FS)
            ta = np.sort(to_sync((first + i0 + pos) / FS))
            res = {}
            for name, ref, ref_pol in (("digital", P["d_t"], P["d_pol"]),
                                       ("display", P["f_t"], np.arange(len(P["f_t"])) % 2 == 0)):
                if len(ref) < 10:
                    res[name] = None
                    continue
                pre = ta[(ta >= a - 3) & (ta < a)]
                dd, pp = pairs(pre, ref, ref_pol)
                lag = {True: np.median(dd[pp]), False: np.median(dd[~pp])}

                def err(lo, hi):
                    xx = ta[(ta >= lo) & (ta < hi)]
                    if not len(xx):
                        return np.array([np.nan])
                    dd, pp = pairs(xx, ref, ref_pol)
                    return np.abs(dd - np.where(pp, lag[True], lag[False]))

                e_pre, e_in, e_post = err(a - 3, a), err(a, zz), err(zz, zz + 2.5)
                res[name] = dict(pre_max_ms=float(np.nanmax(e_pre) * 1e3), in_max_ms=float(np.nanmax(e_in) * 1e3),
                                 post_max_ms=float(np.nanmax(e_post) * 1e3), n_in=int(np.isfinite(e_in).sum()),
                                 lag_ms=[float(lag[True] * 1e3), float(lag[False] * 1e3)],
                                 pass_=bool(np.nanmax(e_in) <= np.nanmax(e_pre) + 0.5e-3))
            # V3: one-to-one inside the span (digital, lag-corrected)
            dig = P["d_t"][(P["d_t"] >= a - 0.005) & (P["d_t"] < zz + 0.005)]  # lag up to a few ms: edges just outside the span pair with transitions inside
            ain = ta[(ta >= a) & (ta < zz)]
            if res.get("digital"):
                lagm = np.mean(res["digital"]["lag_ms"]) / 1e3
                miss = int(sum(np.min(np.abs(dig - (t + lagm))) > 0.005 for t in ain)) if len(dig) else len(ain)
                dig_in = dig[(dig >= a) & (dig < zz)]
                extra = int(sum(np.min(np.abs(ta + lagm - t)) > 0.005 for t in dig_in)) if len(ta) else len(dig_in)
            else:
                miss = extra = None
            rows.append(dict(r, status="ok", n_analog_in=int(len(ain)), n_digital_in=int(len(dig)),
                             analog_without_digital=miss, digital_without_analog=extra, **{f"V_{k}": v for k, v in res.items()}))
            print(folder, rows[-1], flush=True)
    except Exception as exc:  # noqa: BLE001
        rows.append(dict(rec, status=f"{type(exc).__name__}: {exc}"))
        print(folder, rows[-1]["status"], flush=True)

(HERE / "analog_windows_compare.json").write_text(json.dumps(rows, indent=1))
ok = [r for r in rows if r.get("status") == "ok"]
v1 = [r["V_digital"]["pass_"] for r in ok if r.get("V_digital")]
v2 = [r["V_display"]["pass_"] for r in ok if r.get("V_display")]
v3 = [(r["analog_without_digital"] == 0 and r["digital_without_analog"] == 0) for r in ok if r.get("analog_without_digital") is not None]
print(f"\n===== summary: sessions {len({r['folder'] for r in rows})}, spans compared {len(ok)}, not compared {len(rows) - len(ok)}")
print(f"V1 digital: {sum(v1)}/{len(v1)} pass; V2 display: {sum(v2)}/{len(v2)} pass; V3 one-to-one: {sum(v3)}/{len(v3)} pass")
for r in rows:
    if r.get("status") != "ok":
        print("  not compared:", r["folder"], r.get("span", ""), r["status"])
for r in ok:
    bad = [k for k in ("V_digital", "V_display") if r.get(k) and not r[k]["pass_"]]
    if bad or not (r["analog_without_digital"] == 0 and r["digital_without_analog"] == 0):
        print("  FAIL:", r["folder"], r["span"], {k: r.get(k) for k in ("V_digital", "V_display", "analog_without_digital", "digital_without_analog")})
