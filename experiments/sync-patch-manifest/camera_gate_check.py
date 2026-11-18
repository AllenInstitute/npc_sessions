"""Camera gate with the startup PWM section excluded from the period fit
(npc_sync.patch.startup_pwm_section), vs the manifest run (intervals.jsonl, no exclusion).
Every camera line x invalid interval in every file with invalid intervals.

Checks: the 857681_20260720 eye-cam falling-edge failure passes; no span that passed before
fails now; how many lines have a PWM section, and where it is.

Run: PYTHONPATH=../../../npc_sync/src ../../.venv/Scripts/python.exe camera_gate_check.py > camera_gate_check.log 2>&1
"""

import dataclasses
import json
import logging

import numpy as np
import pandas as pd

import npc_sync
from npc_sync import patch as P

logging.disable(logging.WARNING)
f = pd.read_json("manifest_files.jsonl", lines=True)
old = pd.read_json("intervals.jsonl", lines=True)
old = old[old.kind == "camera"]
rows, pwm_rows = [], []
for _, r in f[f.n_invalid > 0].iterrows():
    s = npc_sync.SyncDataset(r.sync_path, on_invalid="ignore")
    fs = s.sample_freq
    spans = [(a - 1 / fs, z) for a, z in s.invalid_intervals]
    for bit, lab in enumerate(s.line_labels):
        if not lab.endswith(P.CAMERA_LINE_SUFFIXES):
            continue
        ri, fa = s.get_rising_edges(bit, units="seconds"), s.get_falling_edges(bit, units="seconds")
        pwm = P.startup_pwm_section(ri[~P.in_intervals(ri, spans)], fa[~P.in_intervals(fa, spans)])
        if pwm:
            pwm_rows.append(dict(session=r.session, line=lab, sections=[[round(a, 3), round(z, 3)] for a, z in pwm],
                                 first_rise=round(float(ri[0]), 3) if len(ri) else None))
        for sp in spans:
            for edge, e in (("rising", ri), ("falling", fa)):
                _, sol = P.interpolate_periodic_edges(e, sp, fit_exclude=pwm)
                rows.append(dict(session=r.session, line=lab, edge=edge, span=[sp[0], sp[1]], status=sol.status,
                                 residual_sec=sol.residual_sec, fraction=sol.fraction,
                                 fit_residual_max_sec=sol.fit_residual_max_sec, fail_new=bool(sol.failures()),
                                 reason=sol.reason, n_invalid=r.n_invalid, n_disc=r.n_discontinuities))
new = pd.DataFrame(rows)
old = old.assign(fail_old=old.failures.apply(len) > 0, key=old.session + "|" + old.line + "|" + old.edge + "|" + old.span.apply(lambda x: f"{x[0]:.5f}"))
new["key"] = new.session + "|" + new.line + "|" + new.edge + "|" + new.span.apply(lambda x: f"{x[0]:.5f}")
m = new.merge(old[["key", "fail_old", "residual_sec"]].rename(columns={"residual_sec": "residual_old"}), on="key", how="left")
m["pop"] = np.where(m.n_disc > 0, "discontinuities", np.where(m.n_invalid >= 20, ">=20 runs", "<20 runs, no discontinuity"))
m["type"] = np.where(m.line.str.endswith("_exposing"), "exposing", "readout")
pd.set_option("display.width", 250)
print(f"camera line x span x edge: {len(m)} (matched to manifest run: {m.fail_old.notna().sum()})")
print("\nfailures old -> new, by population and line type (edge-level):")
print(m.groupby(["pop", "type"]).agg(n=("key", "size"), fail_old=("fail_old", "sum"), fail_new=("fail_new", "sum")).to_string())
print("\nnow pass, failed before:")
print(m[(m.fail_old == True) & ~m.fail_new][["session", "line", "edge", "span", "residual_old", "residual_sec", "fit_residual_max_sec"]].to_string())  # noqa: E712
print("\nnow fail, passed before:")
print(m[(m.fail_old == False) & m.fail_new][["session", "line", "edge", "span", "residual_old", "residual_sec", "reason"]].to_string())  # noqa: E712
p = pd.DataFrame(pwm_rows)
print(f"\ncamera lines with a PWM section: {len(p)} in {p.session.nunique() if len(p) else 0} files "
      f"(of {len(set(zip(m.session, m.line)))} lines in {m.session.nunique()} files)")
if len(p):
    p["n_sections"] = p.sections.apply(len)
    p["starts_at_first_rise"] = [abs(s[0][0] - fr) < 1e-3 for s, fr in zip(p.sections, p.first_rise)]
    print("sections per line:", p.n_sections.value_counts().to_dict(), "| first section starts at the line's first edge:", int(p.starts_at_first_rise.sum()))
    print(p[~p.starts_at_first_rise | (p.n_sections > 1)].head(20).to_string())
m.drop(columns=["key"]).to_csv("camera_gate_check.csv", index=False)
