#!/usr/bin/env python3
"""Verification of the clean RL prompt pools (agent S3, 2026-10-28).

(A) INDEPENDENT check: pure-Python implementation (no hashing, no decon.py code, plain string sets) against the raw English text of FLORES+ (eng_Latn dev + devtest, ALL of it),
    NTREX-128 English, AfriScience-MT dev + test English, AfriLongDoc English (core, hard) fetched from the Hub for this check only (kept in memory, never written). A kept prompt fails when
    one of its normalised sentence parts equals a reference sentence of >= 3 tokens, or it shares a word 8-gram with any reference text. Checked: all kept prompts whose
    xxh64(english) % --py-mod == 0 (1 = all).
(B) The pipeline's own index re-run on a hash-selected share of the kept prompts (xxh64 % --ix-mod == 0), with a new Checker (different chunking): must give 0 hits.
(C) Positive controls: reference sentences planted into kept prompts (start / middle / end) must be caught by BOTH checkers.
Result JSON: --out. No benchmark text is written.
"""
import argparse
import collections
import glob
import json
import os
import re
import sys
import time
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import xxhash  # noqa: E402

import s3_common as C  # noqa: E402

_NON = re.compile(r"[\W_]+", re.UNICODE)
_Q = str.maketrans({"‘": "'", "’": "'", "‚": "'", "‛": "'", "“": '"', "”": '"', "″": '"', "ʼ": "'", "ʹ": "'"})
_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def nrm(t, strip=False):
    t = unicodedata.normalize("NFKC", t or "").translate(_Q).casefold()
    if strip:
        t = "".join(c for c in unicodedata.normalize("NFD", t) if not unicodedata.combining(c))
    return _NON.sub(" ", t).strip()


def forms(t):
    out = [nrm(t)]
    if not t.isascii():
        out.append(nrm(t, True))
    return out


class Ref:
    def __init__(self):
        self.sents = {}      # normalised sentence (>= 3 tokens) -> set name
        self.grams = {}      # 8-gram string -> set name
        self.n = {}

    def add(self, name, text, doc=False):
        """doc=True: a long text (whole document): its sentence parts are protected as well; doc=False: one benchmark sentence = one unit (same as decon.Builder.add)."""
        self.n[name] = self.n.get(name, 0) + 1
        for f in forms(text):
            toks = f.split()
            if len(toks) >= 3:
                self.sents.setdefault(f, name)
            for i in range(len(toks) - 7):
                self.grams.setdefault(" ".join(toks[i:i + 8]), name)
        for p in (_SPLIT.split(text) if doc else []):
            if p.strip() and p != text:
                for f in forms(p):
                    if len(f.split()) >= 3:
                        self.sents.setdefault(f, name)

    def hit(self, text):
        for sm, f in enumerate(forms(text)):
            toks = f.split()
            if f in self.sents:
                return ("exact", self.sents[f], len(toks))
            for p in _SPLIT.split(text):
                if p.strip():
                    g = nrm(p, bool(sm))
                    if len(g.split()) >= 3 and g in self.sents:
                        return ("exact_part", self.sents[g], len(g.split()))
            for i in range(len(toks) - 7):
                k = " ".join(toks[i:i + 8])
                if k in self.grams:
                    return ("8gram", self.grams[k], 8)
        return None


def build_ref(work, tok):
    from huggingface_hub import hf_hub_download
    R = Ref()

    def get(repo, f):
        return hf_hub_download(repo, f, repo_type="dataset", token=tok, local_dir=work)
    for f, nm in (("dev/eng_Latn.jsonl", "flores_plus_dev"), ("devtest/eng_Latn.jsonl", "flores_plus_devtest")):
        p = get("openlanguagedata/flores_plus", f)
        for line in open(p, encoding="utf-8"):
            j = json.loads(line)
            R.add(nm, j.get("text") or "")
        os.remove(p)
    p = get("davidstap/NTREX", "data/eng_Latn/newstest2019-ref.eng.txt")
    for line in open(p, encoding="utf-8"):
        R.add("ntrex128_eng", line.strip())
    os.remove(p)
    for s in ("dev", "test"):
        p = get("dsfsi/afriscience_mt", f"corpus/{s}-00000.parquet")
        d = pq.read_table(p, columns=["source", "paper_id", "target_lang", "sentence_id"]).to_pydict()
        papers = {}
        for i, src in enumerate(d["source"]):
            if d["target_lang"][i] == "amh":
                R.add(f"afriscience_{s}", src)
                papers.setdefault(d["paper_id"][i], []).append((d["sentence_id"][i], src))
        for pid, v in papers.items():
            R.add(f"afriscience_{s}", " ".join(x[1] for x in sorted(v)), doc=True)
        os.remove(p)
    for f in ("core/docs.jsonl", "hard/docs.jsonl"):
        p = get((os.environ['HF_ORG'] + "/AfriLongDoc"), f)
        for line in open(p, encoding="utf-8"):
            j = json.loads(line)
            R.add("afrilongdoc_" + f.split("/")[0], j.get("text") or "", doc=True)
        os.remove(p)
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--py-mod", type=int, default=8)
    ap.add_argument("--ix-mod", type=int, default=10)
    ap.add_argument("--nproc", type=int, default=2)
    a = ap.parse_args()
    tok = os.environ["HF_TOKEN"]
    t0 = time.time()
    R = build_ref(a.work, tok)
    print(C.now(), "reference built:", R.n, "sentences", len(R.sents), "8-grams", len(R.grams), f"{time.time() - t0:.0f}s", flush=True)
    files = sorted(f for d in a.dirs for f in glob.glob(os.path.join(d, "*.parquet")))
    res = dict(created=C.now(), files=len(files), ref_counts=R.n, ref_sentences=len(R.sents), ref_8grams=len(R.grams), py_mod=a.py_mod, ix_mod=a.ix_mod, per_dir={})
    seen_py, seen_ix = set(), set()
    py_checked = py_hits = 0
    ix_texts = []
    sample_for_plant = []
    py_hit_examples = []
    hit_kinds = collections.Counter()
    for k, f in enumerate(files):
        t = pq.read_table(f, columns=["english"]).column("english").to_pylist()
        dname = os.path.basename(os.path.dirname(f))
        for e in t:
            h = xxhash.xxh64_intdigest((e or "").encode("utf-8", "surrogatepass"))
            if h % a.py_mod == 0 and h not in seen_py:
                seen_py.add(h)
                r = R.hit(e or "")
                py_checked += 1
                res["per_dir"].setdefault(dname, {"py_checked": 0, "py_hits": 0})["py_checked"] += 1
                if r:
                    py_hits += 1
                    res["per_dir"][dname]["py_hits"] += 1
                    hit_kinds[f"{r[0]}|{r[1]}|tokens={min(r[2], 9)}"] += 1
                    if len(py_hit_examples) < 30:
                        py_hit_examples.append({"file": os.path.basename(f), "xxh64": format(h, "016x"), "kind": r[0], "set": r[1], "matched_tokens": r[2]})
            if h % a.ix_mod == 0 and h not in seen_ix:
                seen_ix.add(h)
                ix_texts.append(e or "")
            if len(sample_for_plant) < 400 and h % 997 == 0 and len((e or "").split()) > 40:
                sample_for_plant.append(e)
        if k % 10 == 0:
            print(C.now(), f"{k + 1}/{len(files)} py_checked {py_checked} py_hits {py_hits} ix_texts {len(ix_texts)}", flush=True)
    res.update(py_checked=py_checked, py_hits=py_hits, py_hit_examples=py_hit_examples, py_hit_kinds=dict(hit_kinds))
    # ---- (B) pipeline index on a share of the kept prompts
    ix, ex, man = C.load_strict(a.index, a.manifest)
    chk = C.Checker(ix, ex, None, nproc=a.nproc)
    out = chk.run(ix_texts)
    nh = sum(1 for r in out if (r[0] | r[1] | r[2]))
    res.update(index_version=man["version"], index_checked=len(ix_texts), index_hits=nh)
    print(C.now(), "index re-check:", len(ix_texts), "texts,", nh, "hits", flush=True)
    # ---- (C) positive controls (the planted sentences are reference sentences; nothing is written)
    allr = [s for s in R.sents if len(s.split()) >= 10]
    rs = allr[::max(1, len(allr) // 300)][:300]
    planted, labels = [], []
    for j, s in enumerate(rs):
        base = sample_for_plant[j % len(sample_for_plant)] if sample_for_plant else "A clean sentence about nothing in particular."
        words = base.split()
        pos = (0, len(words) // 2, len(words))[j % 3]
        planted.append(" ".join(words[:pos] + [s + "."] + words[pos:]))
    outp = chk.run(planted)
    caught_ix = sum(1 for r in outp if (r[0] | r[1] | r[2]))
    caught_py = sum(1 for p in planted if R.hit(p))
    res.update(control_planted=len(planted), control_caught_by_index=caught_ix, control_caught_by_python=caught_py)
    chk.close()
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "py_hit_examples"}, indent=1))


if __name__ == "__main__":
    main()
