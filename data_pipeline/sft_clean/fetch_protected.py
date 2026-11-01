#!/usr/bin/env python3
"""Build the protected index (step 1 of the clean SFT rebuild, 2026-10-28).

Streams the protected evaluation sets from the Hub ONE FILE AT A TIME (download -> read every string cell -> index -> delete the file), so nothing but the
hashed index is kept. Benchmark text is never written anywhere by this script (no cache is kept, nothing is pushed). Output:
    <out>/protected_index.npz        sorted uint64 arrays (sentence hashes + set bitmasks, 8-gram hashes, first-gram rows)  -> local use by the filter
    <out>/protected_manifest.json    versioned manifest: sets, repos, revisions (commit sha), files used, sentence counts, sha256 of the hash lists
    <out>/protected_sentence_hashes.npz   only the sentence hashes + set bitmasks (what may be pushed to the output repo)

Reading rule: every string cell of every selected file is a protected sentence (any language, source and reference side, ids/metadata are too short to matter; URLs are skipped),
so the multi-way structure of FLORES+/NTREX/SIB-200/... is covered on all sides. Sets with documents (SmolDoc, AfriLongDoc, AfriScience) are protected as a whole text AND sentence by sentence.

Usage:  python fetch_protected.py --out /workspace/work/protected [--only flores_plus]
Token: environment variable HF_TOKEN (never printed).
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import io
import json
import os
import re
import shutil
import sys
import time
from typing import Dict, Iterator, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decon as D  # noqa: E402

SPLIT_RE = re.compile(r"(^|/)(dev|devtest|test|validation|val)([-_./0-9]|$)")
TRAIN_RE = re.compile(r"(^|/)train([-_./0-9]|$)")
DATA_EXT = (".parquet", ".jsonl", ".json", ".csv", ".tsv", ".txt")

# name, repo, revision, selection ("splits" = only files whose path has a dev/devtest/test/validation component; "all" = every data file), extra include/exclude globs, doc?
SPECS = [
    dict(name="flores_plus", repo="openlanguagedata/flores_plus", sel="splits", fields=["text"]),
    dict(name="ibom_mt", repo="howard-nlp/ibom-mt", sel="splits"),
    dict(name="ibom_mt_efi_davlan", repo="Davlan/ibom-mt-en-efi", sel="splits"),
    dict(name="flores_moz", repo="MOZNLP/FLORES_MOZ", sel="splits", fields=["text"]),
    dict(name="ntrex128", repo="davidstap/NTREX", sel="all", include=["data/*/*.txt"]),
    dict(name="ntrex_african", repo="masakhane/ntrex_african", sel="all", include=["ntrex_african/*.tsv"]),
    dict(name="mafand", repo="masakhane/mafand", github="masakhane-io/lafand-mt"),
    dict(name="smol", repo="google/smol", sel="all", include=["smolsent/*", "smoldoc/*", "gatitos/*"], doc_dirs=["smoldoc/"]),
    dict(name="salt", repo="Sunbird/salt", sel="splits", include=["text-all/*"]),
    dict(name="glot_story_book", repo="cis-lmu/GlotStoryBook", sel="all", include=["*.csv"]),
    dict(name="afriscience_mt", repo="dsfsi/afriscience_mt", sel="splits", include=["corpus/*"], doc=True),
    dict(name="afrilongdoc", repo=(os.environ['HF_ORG'] + "/AfriLongDoc"), sel="all", include=["*.jsonl", "*.json"], doc=True),
    dict(name="tico19", repo="gmnlp/tico19", sel="splits"),
    dict(name="ssa_mte", repo="McGill-NLP/SSA-MTE", sel="splits"),
    dict(name="afrimte_wmt2024", repo="masakhane/AfriMTE-WMT2024", sel="all", include=["data/*.src.txt", "data/*.ref.txt"]),
    dict(name="wmt24pp", repo="google/wmt24pp", sel="all", include=["en-sw_KE.jsonl", "en-sw_TZ.jsonl", "en-zu_ZA.jsonl"]),
    dict(name="sib200", repo="Davlan/sib200", sel="splits"),
    dict(name="sib200_ibom", repo="howard-nlp/sib200-ibom", sel="splits"),
    dict(name="xsimplusplus", repo="jaygala24/xsimplusplus", sel="splits"),
    # not in the lead's list, added because they are African MT evaluation sets found on the Hub during recon (only strengthens the filter)
    dict(name="afridocmt", repo="masakhane/AfriDocMT", sel="splits", doc=True),
    dict(name="ssa_mt", repo="McGill-NLP/SSA-MT", sel="splits"),
    dict(name="naijas2st_news_mt", repo="McGill-NLP/NaijaS2ST-news-MT", sel="splits"),
]


def strings(obj, fields=None, depth=0) -> Iterator[str]:
    if isinstance(obj, str):
        if len(obj) >= 8 and not obj.startswith(("http://", "https://", "www.")):
            yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if fields and k not in fields and not isinstance(v, (dict, list)):
                continue
            yield from strings(v, fields, depth + 1)
    elif isinstance(obj, (list, tuple)) and depth < 6:
        for v in obj:
            yield from strings(v, fields, depth + 1)


def iter_file(path: str, fields=None) -> Iterator[str]:
    low = path.lower()
    if low.endswith(".parquet"):
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg)
            for col in t.column_names:
                if fields and col not in fields:
                    continue
                for v in t[col].to_pylist():
                    yield from strings(v, None)
    elif low.endswith(".jsonl"):
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line.startswith(("{", "[")):
                    try:
                        yield from strings(json.loads(line), fields)
                    except Exception:
                        pass
    elif low.endswith(".json"):
        with open(path, encoding="utf-8", errors="replace") as f:
            try:
                yield from strings(json.load(f), fields)
            except Exception:
                f.seek(0)
                for line in f:
                    try:
                        yield from strings(json.loads(line), fields)
                    except Exception:
                        pass
    elif low.endswith((".csv", ".tsv")):
        delim = "\t" if low.endswith(".tsv") else ","
        csv.field_size_limit(10 ** 8)
        with open(path, encoding="utf-8", errors="replace", newline="") as f:
            for row in csv.reader(f, delimiter=delim):
                for c in row:
                    yield from strings(c)
    else:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                for c in line.rstrip("\n").split("\t"):
                    yield from strings(c)


def split_label(path: str) -> str:
    m = SPLIT_RE.search(path)
    return m.group(2) if m else "all"


def select_files(files: List[str], spec: dict) -> List[str]:
    out = []
    for f in files:
        if not f.lower().endswith(DATA_EXT) or f.startswith(".") or "/." in f:
            continue
        if spec.get("include") and not any(fnmatch.fnmatch(f, g) for g in spec["include"]):
            continue
        if spec["sel"] == "splits":
            if not SPLIT_RE.search(f) or TRAIN_RE.search(f):
                continue
        out.append(f)
    return sorted(out)


def github_set(B, spec, manifest, tmp):
    """MAFAND-MT: the Hub repo is only a loading script; the data (data/json_files/<pair>/{train,dev,test}.json, jsonl with a `translation` dict) lives on GitHub.
    dev and test of EVERY pair (en-* and fr-*) are protected; train is not read here."""
    import urllib.request
    t0 = time.time()
    repo = spec["github"]
    def get(url):
        return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "sft-clean"}), timeout=60).read()
    tree = json.loads(get(f"https://api.github.com/repos/{repo}/git/trees/main?recursive=1"))
    sha = tree.get("sha")
    files = [t["path"] for t in tree["tree"] if re.match(r"data/json_files/[^/]+/(dev|test)\.json$", t["path"])]
    per = {}
    for f in files:
        sp = "dev" if f.endswith("/dev.json") else "test"
        raw = get(f"https://raw.githubusercontent.com/{repo}/main/{f}").decode("utf-8", "replace")
        name = f"mafand:{sp}"
        n0 = B.count.get(name, 0)
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    for s_ in strings(json.loads(line)):
                        B.add(name, s_)
                except Exception:
                    pass
        per[sp] = per.get(sp, 0) + B.count.get(name, 0) - n0
    manifest["sets"]["mafand"] = {"repo": "github.com/" + repo + " (Hub masakhane/mafand is a loading script that points there)", "revision": "main", "commit": sha, "files_used": len(files),
                                  "files_selected": len(files), "sentences_by_split": per, "seconds": round(time.time() - t0)}
    print(f"mafand: {len(files)} files, sentences {per}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp", default=None)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--extend", default=None, help="existing protected_index.npz to extend (new sets get new bits); output goes to --out")
    ap.add_argument("--dump-text", default=None, help="LOCAL file with the normalised protected sentences (for the independent exposure check; delete afterwards, never push)")
    ap.add_argument("--max-files", type=int, default=0, help="debug: limit files per set")
    a = ap.parse_args()
    from huggingface_hub import HfApi, hf_hub_download
    tok = os.environ["HF_TOKEN"]
    api = HfApi(token=tok)
    os.makedirs(a.out, exist_ok=True)
    tmp = a.tmp or os.path.join(a.out, "_dl")
    B = D.Builder.extend(D.Index.load(a.extend)) if a.extend else D.Builder()
    if a.dump_text:
        B.dump = open(a.dump_text, "a" if a.extend else "w", encoding="utf-8")
    manifest = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "rule": "every string cell of every selected dev/devtest/test/validation file; see fetch_protected.py", "sets": {}, "missing": {}}
    if a.extend:
        prev = json.load(open(os.path.join(os.path.dirname(a.extend), "protected_manifest.json")))
        manifest["sets"] = prev["sets"]
        manifest["missing"] = {k: v for k, v in prev.get("missing", {}).items() if not (a.only and k in a.only)}
        manifest["history"] = prev.get("history", []) + [{"extended_at": manifest["created"], "sets": a.only}]
    for spec in SPECS:
        if a.only and spec["name"] not in a.only:
            continue
        rev = spec.get("revision")
        t0 = time.time()
        if spec.get("github"):
            github_set(B, spec, manifest, tmp)
            continue
        try:
            info = api.dataset_info(spec["repo"], revision=rev)
            files = select_files([s.rfilename for s in info.siblings], spec)
            sha = info.sha
        except Exception as e:
            manifest["missing"][spec["name"]] = f"{spec['repo']}: {type(e).__name__} {str(e)[:100]}"
            print("MISSING", spec["name"], manifest["missing"][spec["name"]], flush=True)
            continue
        if a.max_files:
            files = files[:a.max_files]
        per_split: Dict[str, int] = {}
        nfiles = 0
        for f in files:
            sp = split_label(f)
            setname = f"{spec['name']}:{sp}"
            is_doc = spec.get("doc") or any(f.startswith(d) for d in spec.get("doc_dirs", []))
            try:
                p = hf_hub_download(spec["repo"], f, repo_type="dataset", revision=rev, token=tok, local_dir=os.path.join(tmp, spec["name"]))
            except Exception as e:
                manifest["missing"][f"{spec['name']}/{f}"] = f"{type(e).__name__} {str(e)[:80]}"
                print("  skip", spec["name"], f, type(e).__name__, flush=True)
                continue
            n0 = B.count.get(setname, 0)
            for s in iter_file(p, spec.get("fields")):
                (B.add_document if is_doc else B.add)(setname, s)
            per_split[sp] = per_split.get(sp, 0) + B.count.get(setname, 0) - n0
            nfiles += 1
            try:
                os.remove(p)
            except OSError:
                pass
        shutil.rmtree(os.path.join(tmp, spec["name"]), ignore_errors=True)
        manifest["sets"][spec["name"]] = {"repo": spec["repo"], "revision": rev or "main", "commit": sha, "files_used": nfiles, "files_selected": len(files),
                                          "sentences_by_split": per_split, "seconds": round(time.time() - t0)}
        print(f"{spec['name']}: {nfiles}/{len(files)} files, sentences {per_split}, {time.time()-t0:.0f}s", flush=True)
    if B.dump:
        B.dump.close()
    ix = B.finish()
    ix.save(os.path.join(a.out, "protected_index.npz"))
    import numpy as np
    np.savez_compressed(os.path.join(a.out, "protected_sentence_hashes.npz"), s_keys=ix.s_keys, s_bits=ix.s_bits, meta=np.array(json.dumps({"set_names": ix.set_names})))
    manifest["set_names_in_bit_order"] = ix.set_names
    manifest["sentences_added_by_set"] = ix.count
    manifest["digest"] = ix.digest()
    manifest["hash"] = "xxh64 of the UTF-8 normalised sentence (decon.norm: NFKC, quote fold, casefold, non-alphanumerics -> one space); a sentence is indexed in two forms (marks kept / marks stripped); sets without >= 3 tokens skipped"
    json.dump(manifest, open(os.path.join(a.out, "protected_manifest.json"), "w"), indent=1, ensure_ascii=False)
    print("done", json.dumps(manifest["digest"]), flush=True)


if __name__ == "__main__":
    main()
