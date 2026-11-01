#!/usr/bin/env python3
"""Which public dataset is the Luganda stream that carries FLORES+ English? (2026-10-26 audit, agent C1, section 1)

The SFT-window sample (r10/pool/lug.jsonl, 2,142 Luganda windows with `src_idx`) holds 16 windows with a FLORES+ English sentence, all single-sentence windows with
src_idx 7.83M-7.91M. This script takes the single-sentence Luganda windows of that index range and measures what share of their (English, target) texts occurs in each
candidate public dataset (any string cell equal after normalisation). A candidate that covers a large share of the range AND contains the FLORES-hit windows is the
source; candidates are streamed read-only, nothing is stored. Output: reports/data/contamination_audit_2026-10-26/luganda_source_match.csv
"""
import csv, json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
from contam_audit_common import OUT, hf_token, normalise
from huggingface_hub import HfApi, HfFileSystem
sys.path.insert(0, os.path.dirname(__file__))
from contam_audit_03_public_source_scan import iter_rows  # noqa: E402

REPO_ID = (os.environ['HF_ORG_LEGACY'] + "/sabiyarn-box-backup-r10-weak-language-data")
fs = HfFileSystem(token=hf_token())
hits = {int(r["src_idx"]) for r in csv.DictReader(open(os.path.join(OUT, "sft_pool_flores_hits.csv"))) if r["iso"] == "lug"}
wins = []
with fs.open(f"datasets/{REPO_ID}/r10/pool/lug.jsonl", "rb") as f:
    for line in f:
        r = json.loads(line)
        si = int(r["src_idx"])
        if 7_800_000 <= si <= 7_930_000 and len(re.split(r"(?<=[.!?])\s+", r["english"].strip())) == 1:
            wins.append((si, normalise(r["english"]), normalise(r["target"])))
print("single-sentence lug windows in range:", len(wins), "flores-hit windows:", len(hits))
allw = []
with fs.open(f"datasets/{REPO_ID}/r10/pool/lug.jsonl", "rb") as f:
    for line in f:
        r = json.loads(line)
        allw.append((int(r["src_idx"]), normalise(r["english"]), normalise(r["target"])))
print("lug windows total", len(allw), "src_idx min/max", min(a[0] for a in allw), max(a[0] for a in allw))
CANDS = [("pkyoyetera/luganda_english_dataset", "*"), ("kambale/luganda-english-parallel-corpus", "*"), ("gimmy256/ugandan_languages", "*"),
         ("yigagilbert/synthetic-parallel-salt", "*"), ("Sunbird/salt", "*text*"), ("michsethowusu/english-ganda_sentence-pairs_mt560", "*"), ("taresco/afri-dict", "*")]
api = HfApi(token=hf_token())
import fnmatch
out = []
for repo, pat in CANDS:
    try:
        files = [x for x in api.list_repo_files(repo, repo_type="dataset") if fnmatch.fnmatch(x, pat) and x.lower().endswith((".parquet", ".jsonl", ".json", ".csv", ".tsv"))][:20]
    except Exception as e:
        print(repo, "ERR", str(e)[:60]); continue
    cells = set()
    for p in files:
        try:
            for r in iter_rows(fs, f"datasets/{repo}/{p}"):
                for v in r.values():
                    if v and 8 < len(v) < 2000:
                        cells.add(normalise(v))
        except Exception as e:
            print("  skip", p, str(e)[:60])
    inr = sum(1 for si, en, tg in wins if en in cells or tg in cells)
    allh = sum(1 for si, en, tg in allw if en in cells or tg in cells)
    fl = sum(1 for si, en, tg in allw if si in hits and (en in cells or tg in cells))
    out.append({"candidate": repo, "files": len(files), "cells": len(cells), "range_windows": len(wins), "range_windows_found": inr, "all_lug_windows": len(allw), "all_lug_windows_found": allh,
                "flores_hit_windows": len(hits), "flores_hit_windows_found": fl})
    print(out[-1], flush=True)
with open(os.path.join(OUT, "luganda_source_match.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
