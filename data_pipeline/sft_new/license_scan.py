"""Licence classification of every probed dataset -> license_scan.json.

    python -m sft_new.license_scan --probe probe.json --out license_scan.json

Declared licence = the dataset card's `license` (+ `license_name`). Classes:
  commercial_ok   permissive or share-alike licences that allow commercial use (apache, mit, bsd, cc-by, cc-by-sa, cc0, odc-by, odbl, mpl, gfdl...)
  non_commercial  any NC / research-only licence -> SKIPPED (listed with its size at the end of the build)
  unknown         no licence on the card, or `other` -> USED but flagged (every row carries metadata.license_flag), listed for the owner's decision
Overrides follow the owner's explicit instructions (OVERRIDE below). `upstream_hint` records what the dataset is known to be derived from when that
changes the picture (Alpaca-derived instruction sets are CC-BY-NC upstream); a hint never changes the decision by itself, it is for the owner.
Sizes are the sum of the repository's file sizes (bytes).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
from pathlib import Path

OK = re.compile(r"^(apache-2\.0|mit|bsd.*|cc-by-[234]\.\d|cc-by-sa-[34]\.\d|cc0-1\.0|odc-by|odbl|pddl|mpl-2\.0|cdla-permissive.*|unlicense|gfdl|wtfpl|afl-3\.0|artistic.*|isc|zlib|bsl-1\.0|cc-by)$")
NC = re.compile(r"(^|-)nc(-|$)|non-?commercial|research-only|cc-by-nd|noncommercial", re.I)
OVERRIDE = {
    "ghananlpcommunity/pristine-twi-english": ("use", "owner: use regardless of licence (declared cc-by-nc-4.0)"),
    "AfriSpeech/african-corpus-jw": ("use", "owner: jw.org text allowed (copyright not cleared)"),
    "AfriSpeech/africa-corpus": ("use", "owner: allowed"),
    "African-Languages-Lab/multi-open": ("use", "owner: named explicitly"),
}
HINTS = [
    (re.compile(r"alpaca", re.I), "Alpaca-derived instruction data: upstream Alpaca is CC-BY-NC-4.0 and its answers were generated with OpenAI models"),
    (re.compile(r"(^|/)(sib200|.*sib200.*)$", re.I), "SIB-200 is CC-BY-SA-4.0; its test split is a benchmark"),
    (re.compile(r"mafand", re.I), "MAFAND-MT is CC-BY-NC-4.0 upstream"),
    (re.compile(r"afrisenti", re.I), "AfriSenti is CC-BY-NC-SA-4.0 for part of its languages"),
    (re.compile(r"xp3", re.I), "xP3 is Apache-2.0 but built from many source datasets with their own licences"),
    (re.compile(r"aya", re.I), "Aya collection: Apache-2.0; contains machine-translated templates"),
    (re.compile(r"wiki", re.I), "Wikipedia text is CC-BY-SA-3.0/4.0 + GFDL"),
    (re.compile(r"(gsm8k)", re.I), "GSM8K is MIT"),
    (re.compile(r"(uhura|eval|_test$)", re.I), "benchmark / evaluation set"),
]


def classify(lic: str | None, name: str | None) -> str:
    if not lic:
        return "unknown"
    parts = [p.strip().lower() for p in lic.split(",") if p.strip()]
    if any(NC.search(p) for p in parts):
        return "non_commercial"
    if lic.lower() == "other" and name and NC.search(name):
        return "non_commercial"
    if parts and all(OK.match(p) or p == "other" for p in parts) and any(OK.match(p) for p in parts):
        return "commercial_ok"
    return "unknown"


def size_of(repo: str, token: str) -> int | None:
    from huggingface_hub import HfApi
    try:
        info = HfApi(token=token).dataset_info(repo, files_metadata=True)
        return sum((s.size or 0) for s in (info.siblings or []))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY") or ""
    probe = json.loads(a.probe.read_text())
    repos = [r for r, v in probe.items() if "error" not in v]
    with cf.ThreadPoolExecutor(a.threads) as ex:
        sizes = dict(zip(repos, ex.map(lambda r: size_of(r, token), repos)))
    out = {}
    for r, v in probe.items():
        if "error" in v:
            out[r] = {"declared": None, "class": "unreachable", "decision": "skip", "reason": v["error"], "bytes": None}
            continue
        cls = classify(v.get("license"), v.get("license_name"))
        dec, why = {"commercial_ok": ("use", "licence allows commercial use"), "non_commercial": ("skip", "non-commercial licence"),
                    "unknown": ("use_flagged", "no usable licence on the card; used and flagged")}[cls]
        if r in OVERRIDE:
            dec, why = OVERRIDE[r]
        hint = next((h for p, h in HINTS if p.search(r)), None)
        out[r] = {"declared": v.get("license"), "license_name": v.get("license_name"), "class": cls, "decision": dec, "reason": why,
                  "bytes": sizes.get(r), "gated": v.get("gated"), "private": v.get("private"), "last_modified": v.get("last_modified"),
                  "upstream_hint": hint}
    summary = {}
    for d in ("use", "use_flagged", "skip", "unreachable"):
        rows = {r: x for r, x in out.items() if x["decision"] == d or x["class"] == d}
        summary[d] = {"repos": len(rows), "bytes": sum(x["bytes"] or 0 for x in rows.values())}
    a.out.write_text(json.dumps({"summary": summary, "datasets": out}, indent=1, ensure_ascii=False))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
