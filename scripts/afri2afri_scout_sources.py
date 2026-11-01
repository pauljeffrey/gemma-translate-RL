#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""List, for each of the 77 languages, which open monolingual corpora of the Hub have a folder for it (metadata calls only, no download) -> json.

    python scripts/afri2afri_scout_sources.py --out sources.json

Sources and layouts (checked 2026-10-20): wikimedia/wikipedia `20231101.<iso1 or iso3>/train-*.parquet` (CC-BY-SA-3.0 + GFDL), HuggingFaceFW/fineweb-2 `data/<iso3>_<Script>/train/*.parquet` (ODC-By),
cis-lmu/GlotCC-V1 `v1.0/<iso3>-<Script>/*.parquet` (CC0), HPLT/HPLT2.0_cleaned `<iso3>_<Script>/*.parquet` (CC0), allenai/c4 `multilingual/c4-<iso1>.tfrecord-*.json.gz` (ODC-By),
allenai/madlad-400 (ODC-By; clean split). Language codes come from `languages.py` (GlotLID labels give iso3_Script).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from gemma_mt.languages import ALL_ISOS, LANGUAGES  # noqa: E402

SFT77 = [i for i in ALL_ISOS if i not in ("nde", "fat", "knc", "gaa", "men", "luy", "tzm", "ful")]
WIKI = {"afr": ["af"], "amh": ["am"], "bam": ["bm"], "ewe": ["ee"], "hau": ["ha"], "ibo": ["ig"], "kik": ["ki"], "kin": ["rw"], "lin": ["ln"], "lug": ["lg"], "mlg": ["mg"], "nya": ["ny"],
        "orm": ["om"], "run": ["rn"], "sna": ["sn"], "som": ["so"], "sot": ["st"], "swh": ["sw"], "tsn": ["tn"], "tso": ["ts"], "tum": ["tum"], "twi": ["tw", "ak"], "ven": ["ve"],
        "wol": ["wo"], "xho": ["xh"], "yor": ["yo"], "zul": ["zu"], "tir": ["ti"], "ssw": ["ss"], "fon": ["fon"], "kbp": ["kbp"], "pcm": ["pcm"], "kab": ["kab"], "dag": ["dag"], "kon": ["kg"],
        "sag": ["sg"], "nso": ["nso"], "bin": ["bin"], "efi": ["efi"], "nbl": ["nr"], "kea": ["kea"], "ibb": ["ibb"], "lua": ["lua"], "tzm": ["tzm"], "tiv": ["tiv"], "wes": ["wes"]}
C4 = {"afr": "af", "amh": "am", "hau": "ha", "ibo": "ig", "mlg": "mg", "nya": "ny", "sna": "sn", "som": "so", "sot": "st", "swh": "sw", "xho": "xh", "yor": "yo", "zul": "zu"}


def script_of(iso: str) -> str:
    return "Ethi" if iso in ("amh", "tir") else "Latn"


def codes(iso: str):
    """iso3_Script candidates: GlotLID labels first, then <iso>_<script>."""
    out = list(LANGUAGES[iso].glotlid) + [f"{iso}_{script_of(iso)}"]
    return list(dict.fromkeys(out))


def main(argv=None) -> int:
    from huggingface_hub import HfApi

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    api = HfApi(token=os.environ.get("HF_TOKEN"))

    def ls(repo, path):
        try:
            return [(f.path, getattr(f, "size", None) or 0) for f in api.list_repo_tree(repo, path_in_repo=path, repo_type="dataset", recursive=False) if getattr(f, "size", None)]
        except Exception:
            return []

    res = {}
    for iso in SFT77:
        r = {}
        for c in codes(iso):
            f = ls("HuggingFaceFW/fineweb-2", f"data/{c}/train")
            if f:
                r["fineweb-2"] = {"code": c, "files": f}
                break
        for c in codes(iso):
            f = ls("cis-lmu/GlotCC-V1", f"v1.0/{c.replace('_', '-')}")
            if f:
                r["glotcc"] = {"code": c, "files": f}
                break
        for c in codes(iso):
            f = ls("HPLT/HPLT2.0_cleaned", c)
            if f:
                r["hplt2"] = {"code": c, "files": f}
                break
        for w in WIKI.get(iso, []):
            f = ls("wikimedia/wikipedia", f"20231101.{w}")
            if f:
                r["wikipedia"] = {"code": w, "files": f}
                break
        if iso in C4:
            f = ls("allenai/c4", "multilingual")
            r["c4"] = {"code": C4[iso], "files": [x for x in f if x[0].startswith(f"multilingual/c4-{C4[iso]}.tfrecord")][:3]} if f else None
            if r["c4"] is None or not r["c4"]["files"]:
                r.pop("c4", None)
        res[iso] = r
        print(iso, {k: (v["code"], len(v["files"]), round(sum(s for _, s in v["files"]) / 1e6)) for k, v in r.items()}, flush=True)
    json.dump(res, open(a.out, "w"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
