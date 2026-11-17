"""Locate NI-DAQ data for a DRpilot session folder on S3 (read-only).
Prints: bucket/prefix, NI-DAQ zarr(s) and their segments, clipped TTL/sample_numbers paths."""
from __future__ import annotations
import datetime, json, re, sys
import boto3

BUCKETS = ("aind-ephys-data", "aind-private-data-prod-o5171v", "aind-open-data")
s3 = boto3.client("s3")


def prefixes(bucket, pre):
    out = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=pre, Delimiter="/"):
        out += [p["Prefix"] for p in page.get("CommonPrefixes", [])]
    return out


def keys(bucket, pre):
    out = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=pre):
        out += [(o["Key"], o["Size"]) for o in page.get("Contents", [])]
    return out


def find_session(folder: str, sync_stem: str):
    """sync_stem e.g. '20250930T140330' -> ecephys_<mouse>_2025-09-30_14-03-30 (closest start within 10 min)."""
    mouse = folder.split("_")[1]
    t_sync = datetime.datetime.strptime(sync_stem, "%Y%m%dT%H%M%S")
    cands = []
    for b in BUCKETS:
        for p in prefixes(b, f"ecephys_{mouse}_{t_sync:%Y-%m-%d}"):
            m = re.match(rf"ecephys_{mouse}_(\d{{4}}-\d\d-\d\d_\d\d-\d\d-\d\d)/$", p)
            if not m:
                continue
            t = datetime.datetime.strptime(m.group(1), "%Y-%m-%d_%H-%M-%S")
            cands.append((abs((t - t_sync).total_seconds()), b, p))
    cands.sort()
    return [c for c in cands if c[0] < 600]


def nidaq_info(bucket, prefix):
    for root in (prefix, prefix + "ecephys/"):
        comp = [p for p in prefixes(bucket, root + "ecephys_compressed/") if "NI-DAQ" in p]
        clipped = [k for k, _ in keys(bucket, root + "ecephys_clipped/") if "NI-DAQ" in k or k.endswith("structure.oebin")]
        if comp or clipped:
            return comp, clipped
    return [], []


if __name__ == "__main__":
    folder, sync_stem = sys.argv[1], sys.argv[2]
    c = find_session(folder, sync_stem)
    res = dict(folder=folder, candidates=[(d, b, p) for d, b, p in c])
    if c:
        _, b, p = c[0]
        comp, clipped = nidaq_info(b, p)
        res.update(bucket=b, prefix=p, zarrs=comp, clipped=clipped)
    print(json.dumps(res, indent=1))
