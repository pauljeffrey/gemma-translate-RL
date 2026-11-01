"""Schema + licence probe of Hub datasets. Reads metadata only (dataset card, datasets-server info / first rows, parquet footers): no dataset is downloaded.

    python -m sft_new.probe --repos repos.txt --out probe.json [--threads 8]

For every repo: licence (card + `license:` tag), gated/private, last modified, number of files, configs with splits / row counts / feature names, and two sample
rows (each field cut to 240 characters) from the first config's first split.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
SERVER = "https://datasets-server.huggingface.co"


def _cut(v, n=240):
    if isinstance(v, (bytes, bytearray)):
        return f"<{len(v)} bytes>"
    if isinstance(v, str):
        return v[:n]
    if isinstance(v, list):
        return [_cut(x, n) for x in v[:6]]
    if isinstance(v, dict):
        return {k: _cut(x, n) for k, x in list(v.items())[:12]}
    return v


def probe_one(repo: str, token: str) -> dict:
    from huggingface_hub import HfApi
    hdr = {"Authorization": f"Bearer {token}"}
    out: dict = {"repo": repo}
    info, last = None, None
    for attempt in range(7):                                         # the Hub rate-limits (HTTP 429): back off and retry
        try:
            info = HfApi(token=token).dataset_info(repo)
            break
        except Exception as e:                                       # missing, gated, renamed -- or throttled
            last = e
            if "429" in str(e) or "Too Many" in str(e) or "rate" in str(e).lower():
                time.sleep(15 * (attempt + 1))
                continue
            break
    if info is None:
        return {"repo": repo, "error": f"{type(last).__name__}: {str(last)[:120]}"}
    cd = info.card_data.to_dict() if info.card_data else {}
    lic = cd.get("license")
    out.update(license=lic if isinstance(lic, str) else (",".join(lic) if lic else None), license_name=cd.get("license_name"),
               gated=bool(info.gated), private=bool(info.private), last_modified=str(info.last_modified)[:10], files=len(info.siblings or []),
               languages=cd.get("language") if isinstance(cd.get("language"), list) else ([cd["language"]] if cd.get("language") else []),
               task_categories=cd.get("task_categories"))
    for attempt in range(3):
        try:
            r = requests.get(f"{SERVER}/info", params={"dataset": repo}, headers=hdr, timeout=45)
            if r.status_code == 200:
                di = r.json().get("dataset_info", {})
                out["configs"] = {c: {"splits": {s: v.get("num_examples") for s, v in (d.get("splits") or {}).items()},
                                      "columns": list((d.get("features") or {}).keys())} for c, d in di.items()}
                break
            out["info_status"] = r.status_code
            if r.status_code in (404, 400, 422, 501):
                break
        except Exception as e:
            out["info_error"] = type(e).__name__
        time.sleep(2 * (attempt + 1))
    cfgs = out.get("configs") or {}
    if cfgs:
        c0 = next(iter(cfgs))
        s0 = next(iter(cfgs[c0]["splits"]), None)
        if s0:
            try:
                r = requests.get(f"{SERVER}/first-rows", params={"dataset": repo, "config": c0, "split": s0}, headers=hdr, timeout=60)
                if r.status_code == 200:
                    out["sample"] = {"config": c0, "split": s0, "rows": [_cut(x["row"]) for x in r.json().get("rows", [])[:2]]}
            except Exception as e:
                out["sample_error"] = type(e).__name__
    if not cfgs:                                                     # script-based or unsupported by the viewer: read a parquet/jsonl footer instead
        files = [s.rfilename for s in (info.siblings or [])]
        out["file_sample"] = files[:8]
        try:
            import pyarrow.parquet as pq
            from huggingface_hub import HfFileSystem
            pf = next((f for f in files if f.endswith(".parquet")), None)
            if pf:
                with HfFileSystem(token=token).open(f"datasets/{repo}/{pf}", "rb") as fh:
                    f = pq.ParquetFile(fh)
                    out["parquet"] = {"file": pf, "rows": f.metadata.num_rows, "columns": f.schema_arrow.names}
                    out["sample"] = {"config": None, "split": None, "rows": [_cut(r) for r in f.read_row_group(0).slice(0, 2).to_pylist()]}
        except Exception as e:
            out["parquet_error"] = f"{type(e).__name__}: {str(e)[:80]}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY") or ""
    repos = [l.strip() for l in a.repos.read_text().splitlines() if l.strip() and not l.startswith("#")]
    res = {}
    with cf.ThreadPoolExecutor(a.threads) as ex:
        for r in ex.map(lambda x: probe_one(x, token), repos):
            res[r["repo"]] = r
            print(r["repo"], r.get("license"), "ERR" if "error" in r else "", flush=True)
    a.out.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=lambda o: f"<{type(o).__name__}>"))
    print(len(res), "repos probed;", sum("error" in r for r in res.values()), "unreachable")


if __name__ == "__main__":
    main()
