#!/usr/bin/env python3
"""Re-scan of PUBLIC Hub sources for FLORES+ dev/devtest text (2026-10-26 contamination audit, agent C1).

Why: the SFT windows (`gemma/windows_sft`) were never mirrored (research_archive_2026-10-23/BOX_ONLY_ITEMS.md), so the source that carries the
English side of 95.3% of the FLORES+ devtest sentences cannot be read off the scan outputs. This script answers it for every candidate source that is
still reachable: it STREAMS the files of a Hub dataset (no download to disk, read-only, token from the repo-root .env, never printed) and counts, per
file, the rows in which a field equals (NFKC, case-fold, punctuation-free, the normalisation of gemma_mt/eval/contamination.py)

  * a FLORES+ English sentence of devtest or dev       -> `en_devtest`, `en_dev`
  * a FLORES+ sentence of one of our target languages  -> `tgt_devtest`, `tgt_dev`  (the FLORES+ code of the language is recorded)

Only counts and FLORES ids are written, never FLORES text. Usage:
    python contam_audit_03_public_source_scan.py --repo UBC-NLP/AfroLingu-MT --glob '*.jsonl' [--max-rows N] [--tag name]
Outputs append to reports/data/contamination_audit_2026-10-26/public_source_scan.csv.
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from contam_audit_common import OUT, REPO, hf_token, normalise  # noqa: E402

from contam_audit_common import CACHE


def our_flores_codes() -> list:
    sys.path.insert(0, os.path.join(REPO, "gemma_mt"))
    from gemma_mt.languages import LANGUAGES
    codes = {l.flores for l in LANGUAGES.values() if l.flores}
    codes |= {"twi_Latn_asan1239", "aka_Latn", "kon_Latn", "swc_Latn", "nde_Latn", "fuv_Latn", "gaz_Latn", "orm_Latn"}      # FLORES-200 variants of our targets
    return sorted(codes)


def load_index(codes):
    from huggingface_hub import hf_hub_download
    os.makedirs(CACHE, exist_ok=True)
    idx = defaultdict(set)      # normalised text -> {(kind, split, code)}
    sizes = {}
    for split in ("devtest", "dev"):
        for code in ["eng_Latn"] + list(codes):
            try:
                p = hf_hub_download("openlanguagedata/flores_plus", f"{split}/{code}.jsonl", repo_type="dataset", token=hf_token(), cache_dir=CACHE)
            except Exception:
                continue
            n = 0
            for line in open(p, encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line)
                t = normalise(r["text"])
                if len(t.split()) >= 3:
                    idx[t].add(("en" if code == "eng_Latn" else "tgt", split, code))
                    n += 1
            sizes[(split, code)] = n
    return idx, sizes


def iter_rows(fs, path):
    """Yield dicts of the string fields of every row (jsonl / json lines / parquet), streaming."""
    low = path.lower()
    if low.endswith(".parquet"):
        import pyarrow.parquet as pq
        with fs.open(path, "rb") as f:
            pf = pq.ParquetFile(f)
            cols = [c for c in pf.schema_arrow.names if str(pf.schema_arrow.field(c).type) in ("string", "large_string")]
            for b in pf.iter_batches(batch_size=20000, columns=cols):
                d = b.to_pydict()
                for i in range(b.num_rows):
                    yield {c: d[c][i] for c in cols}
    elif low.endswith((".jsonl", ".json")):
        with fs.open(path, "rb") as f:
            for line in f:
                line = line.strip()
                if not line.startswith(b"{"):
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                yield {k: v for k, v in r.items() if isinstance(v, str)}
    elif low.endswith((".tsv", ".csv", ".txt")):
        with fs.open(path, "rb") as f:
            for line in f:
                yield {f"c{j}": c for j, c in enumerate(line.decode("utf-8", "replace").rstrip("\n").split("\t"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--glob", default="*")
    ap.add_argument("--max-files", type=int, default=50)
    ap.add_argument("--max-rows", type=int, default=0, help="0 = all rows of each file")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    from huggingface_hub import HfApi, HfFileSystem
    idx, sizes = load_index(our_flores_codes())
    print(f"index: {len(idx)} distinct FLORES+ sentences", flush=True)
    api = HfApi(token=hf_token())
    files = [f for f in api.list_repo_files(a.repo, repo_type="dataset") if fnmatch.fnmatch(f, a.glob)][: a.max_files]
    fs = HfFileSystem(token=hf_token())
    out = os.path.join(OUT, "public_source_scan.csv")
    new = not os.path.exists(out)
    with open(out, "a", newline="") as fo:
        w = csv.writer(fo)
        if new:
            w.writerow(["date", "repo", "file", "rows_scanned", "en_devtest_rows", "en_dev_rows", "tgt_devtest_rows", "tgt_dev_rows", "en_devtest_distinct", "en_dev_distinct", "tgt_langs_hit", "seconds", "tag"])
        for p in files:
            t0 = time.time()
            rows = 0
            c = Counter()
            en_dt, en_d = set(), set()
            langs = Counter()
            try:
                for r in iter_rows(fs, f"datasets/{a.repo}/{p}"):
                    rows += 1
                    if a.max_rows and rows > a.max_rows:
                        break
                    hit_flags = set()
                    for v in r.values():
                        if not v or len(v) < 12 or len(v) > 4000:
                            continue
                        n = normalise(v)
                        m = idx.get(n)
                        if m is None:
                            # multi-sentence fields: test the sentences separately
                            parts = re.split(r"(?<=[.!?])\s+|\n+", v)
                            if len(parts) > 1:
                                for s in parts:
                                    mm = idx.get(normalise(s))
                                    if mm:
                                        for k in mm:
                                            hit_flags.add((k, normalise(s)))
                            continue
                        for k in m:
                            hit_flags.add((k, n))
                    seen_row = set()
                    for (kind, split, code), n in hit_flags:
                        key = (kind, split)
                        if key not in seen_row:
                            c[key] += 1
                            seen_row.add(key)
                        if kind == "en":
                            (en_dt if split == "devtest" else en_d).add(n)
                        else:
                            langs[code] += 1
            except Exception as e:
                print("  ERR", p, str(e)[:100], flush=True)
            w.writerow([time.strftime("%Y-%m-%d"), a.repo, p, rows, c[("en", "devtest")], c[("en", "dev")], c[("tgt", "devtest")], c[("tgt", "dev")], len(en_dt), len(en_d),
                        ";".join(f"{k}:{v}" for k, v in langs.most_common(12)), round(time.time() - t0), a.tag])
            fo.flush()
            print(f"{p}: rows {rows} en_devtest {c[('en','devtest')]} en_dev {c[('en','dev')]} tgt_devtest {c[('tgt','devtest')]} tgt_dev {c[('tgt','dev')]} {dict(langs.most_common(5))}", flush=True)


if __name__ == "__main__":
    main()
