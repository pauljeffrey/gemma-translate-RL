#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Prompt set of the SCIENCE RL run (DECISIONS.md D-52, owner 2026-10-21): English scientific text windows for the 77 RL languages plus a 30 % replay share of the
general prompts, in the format of `rl_prompts_v3` with two extra columns (`domain`: science | general, `field`).

Sources of the science windows (the owner allows any licence for RL prompts; source and licence are recorded per row):
  afriscience   AfriScience-MT TRAIN English side (dsfsi/afriscience_mt, Apache-2.0; 177 papers, whole documents + sub-windows)
  pmc           PubMed Central open access full text (common-pile/pubmed; CC-BY family, per-document licence kept)
  arxiv         arXiv full text under an open licence (common-pile/arxiv_papers; per-document licence kept)
  libretexts    LibreTexts science textbooks (common-pile/libretexts; CC-BY / CC-BY-SA / GFDL, per-document licence kept)
  arxiv_abs     arXiv abstracts (gfissore/arxiv-abstracts-2021, CC0 metadata)
HELD OUT of every prompt (science and general): the AfriScience-MT dev + test English sentences and documents, and the AfriLongDoc hard (+ core) documents: a window is
dropped when one of its sentences equals a held-out sentence (NFC, lower case, white space collapsed) or when it shares ANY word 8-gram with a held-out text.

Windows are CONTIGUOUS WHOLE SENTENCES of one document joined by single spaces (no newline, D-23), measured in Davlan/afro-xlmr-large-76L tokens (special tokens counted, exactly as
`src_afr_tokens` of rl_prompts_v3), assigned to the bins b64 .. b2048 of gemma_mt.rl.data.

Sub-commands (CPU only, run with nice and <= 8 processes):

  windows  --source S --out FILE.parquet [--max-bytes N] [--workers 6]    cut + filter the windows of one source (network sources are STREAMED and cut off after --max-bytes of compressed data)
  holdout  --out FILE.json                                                 the held-out sentence / 8-gram index (reads afs_raw dev+test and the AfriLongDoc files)
  general  --v3 DIR --out FILE.parquet --per-cell 120                      replay rows from rl_prompts_v3 (per language and bin), without the 2c hold-out prompts
  assemble --windows F1 F2 ... --general FILE --out DIR [--cells ...]      language assignment, dedup against the hold-out, parquet per language + stats.json
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import hashlib
import io
import json
import multiprocessing as mp
import os
import random
import re
import sys
import time
import unicodedata
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

SEED = 20261021
BINS = ("b64", "b256", "b512", "b1024", "b2048")
EDGES = {"b64": (0, 65), "b256": (65, 257), "b512": (257, 513), "b1024": (513, 1025), "b2048": (1025, 2049)}
AFRO_TOKENIZER_GLOB = "/workspace/sft_build/hf/hub/models--Davlan--afro-xlmr-large-76L/snapshots/*/tokenizer.json"
AFS_DIR = "/workspace/jobs/a4/afs_raw/corpus"
LONGDOC_DIR = "/workspace/jobs/a4/holdout/afrilongdoc"
HF = "https://huggingface.co/datasets"
SOURCES = {
    "afriscience": dict(name="dsfsi/afriscience_mt:train(en)", licence="apache-2.0"),
    "pmc": dict(name="common-pile/pubmed", url=f"{HF}/common-pile/pubmed/resolve/main/data/00000_pubmedcentral.jsonl.gz", licence="cc-by (per document)"),
    "arxiv": dict(name="common-pile/arxiv_papers", url=f"{HF}/common-pile/arxiv_papers/resolve/main/00000_arxiv-papers.jsonl.gz", licence="open licence (per document)"),
    "libretexts": dict(name="common-pile/libretexts", url=f"{HF}/common-pile/libretexts/resolve/main/00000_libretext.json.gz", licence="cc-by / cc-by-sa / gfdl (per document)"),
    "arxiv_abs": dict(name="gfissore/arxiv-abstracts-2021", url=f"{HF}/gfissore/arxiv-abstracts-2021/resolve/main/arxiv-abstracts.jsonl.gz", licence="cc0-1.0 (metadata)"),
}
# bins a document of a long source is cut into (probability of the bin of a window); b2048/b1024 dominate because the benchmark documents are 600-1,800 AfriXLM-R tokens
LONG_SOURCE_BIN_P = {"b64": 0.05, "b256": 0.15, "b512": 0.20, "b1024": 0.30, "b2048": 0.30}
WINDOWS_PER_DOC = {"pmc": 3, "arxiv": 2, "libretexts": 2}
LIBRE_SCIENCE_HOSTS = {"bio": "biology", "chem": "chemistry", "phys": "physics", "geo": "earth science", "eng": "engineering", "med": "medicine", "stats": "statistics",
                       "biz": None, "math": None, "human": None, "socialsci": "social science", "workforce": None, "k12": None, "ds": None, "query": None, "batch": None}


# ------------------------------------------------------------------------------------------------------------------------------ text cleaning
NFC = lambda s: unicodedata.normalize("NFC", s)                                                    # noqa: E731
_WS = re.compile(r"\s+")
_URL = re.compile(r"https?://|www\.|\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b|\bdoi:|\barXiv:", re.I)
_HTML = re.compile(r"<[a-zA-Z/][^>]*>|&[a-z]+;|\{\{|\}\}")
_PMC_CITE = re.compile(r"\s*\[@[^\]]*\]")
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)(?:\{[^}]*\})?")
_MD_ATTR = re.compile(r"\{[#.][^}]*\}|\{ref-type=[^}]*\}")
_MD_EMPH = re.compile(r"(?<!\w)[*_]{1,3}(?=\S)([^*_\n]{1,120}?)(?<=\S)[*_]{1,3}(?!\w)")
_BAD_LIBRE = re.compile(r"Page ID|Save as PDF|Last updated|Learning Objectives|Skills to Develop|\bFigure \d+\.\d|Watch the video|click here|Exercises?\b|LibreTexts|Creative Commons|Attribution|All rights reserved", re.I)
_BAD_SENT = re.compile(r"Corresponding author|Competing interests|Conflict of interest|Acknowledg|Funding:|Supplementary|Table \d|Fig\.? \d|Figure \d|e-?mail|ORCID|\bET AL\b|©|Received:|Accepted:|Published:|Keywords?:|Abbreviations?:|Copyright", re.I)
_TERMINAL = (".", "!", "?", ")", '"', "”", "’")
LATEX_TOKENS = re.compile(r"\\[A-Za-z]{2,}|\\\(|\\\[|\$\$")


def clean_pmc(text: str) -> str:
    t = _PMC_CITE.sub("", text)
    t = _MD_LINK.sub(lambda m: m.group(1), t)
    t = _MD_ATTR.sub("", t)
    t = t.replace("\\'", "'").replace("\\*", "*").replace("~", "")
    t = _MD_EMPH.sub(lambda m: m.group(1), t)
    return t


def paragraphs_of(text: str) -> List[str]:
    """Blank-line separated paragraphs of a document with the hard line wrapping removed (arXiv text is wrapped at 80 columns). Headings (#, ====) and table lines are dropped."""
    out = []
    for blk in re.split(r"\n\s*\n", NFC(text)):
        lines = [l.strip() for l in blk.split("\n") if l.strip()]
        if not lines or all(re.fullmatch(r"[=\-_*|:+ ]+", l) for l in lines):
            continue
        if lines[0].startswith("#") or any(l.startswith(("|", "+--", "!", "<")) for l in lines):
            continue
        out.append(_WS.sub(" ", " ".join(lines)).strip())
    return out


def prose_ok(s: str, libre: bool = False) -> bool:
    """A sentence a window may contain: running prose of a scientific text (numbers, units and symbols are fine; tables, headings, references, markup are not)."""
    n = len(s)
    if n < 25 or n > 1100:
        return False
    w = s.split()
    if not (6 <= len(w) <= 90):
        return False
    if not (s[0].isupper() or s[0].isdigit() or s[0] in "\"“‘(["):
        return False
    if not s.endswith(_TERMINAL):
        return False
    if _URL.search(s) or _HTML.search(s) or LATEX_TOKENS.search(s) or s.count("$") > 2 or _BAD_SENT.search(s):
        return False
    if libre and _BAD_LIBRE.search(s):
        return False
    letters = sum(c.isalpha() for c in s)
    if letters / n < 0.55 or sum(c.isdigit() for c in s) / n > 0.20:
        return False
    if sum(1 for c in s if c in "=<>±×÷∑∫√≈≠≤≥^\\|{}_") > 6:
        return False
    if sum(1 for c in s if ord(c) > 0x2FF and not c.isspace() and c not in "–—‘’“”…°±×−μµ") > 3:      # mostly Latin-script English
        return False
    return True


class Worker:
    """Per-process state: AfriXLM-R tokenizer + the repo's sentence splitter."""

    def __init__(self):
        from tokenizers import Tokenizer

        sys.path.insert(0, os.environ.get("GEMMA_MT_SRC", "/workspace/jobs/a4/gemma_mt_sci"))
        from gemma_mt.rl.segments import split_sentences

        self.split = split_sentences
        self.tk = Tokenizer.from_file(glob.glob(AFRO_TOKENIZER_GLOB)[0])

    def ntok(self, text: str) -> int:
        return len(self.tk.encode(text, add_special_tokens=True).ids)

    def sentences(self, paragraph: str) -> List[str]:
        if len(paragraph) <= 3000:
            return self.split(paragraph)
        out: List[str] = []
        for piece in re.split(r"(?<=[.!?])\s+(?=[A-Z])", paragraph):                 # one-line documents (LibreTexts): linear pre-split, then the splitter per ~2.5k characters
            out.append(piece)
        merged, cur = [], ""
        for p in out:
            if len(cur) + len(p) > 2500 and cur:
                merged.append(cur)
                cur = ""
            cur = (cur + " " + p).strip()
        if cur:
            merged.append(cur)
        return [x for m in merged for x in self.split(m)]


def runs_of(worker: Worker, paragraphs: Sequence[str], libre: bool = False) -> List[List[Tuple[str, int]]]:
    """Runs of consecutive prose sentences [(sentence, afro tokens)] (a paragraph break or a non-prose sentence ends a run)."""
    runs: List[List[Tuple[str, int]]] = []
    for p in paragraphs:
        cur: List[Tuple[str, int]] = []
        sents = worker.sentences(p)
        cnt = [len(e.ids) for e in worker.tk.encode_batch(sents, add_special_tokens=False)] if sents else []
        for s, c in zip(sents, cnt):
            if prose_ok(s, libre) and c <= 400:
                cur.append((s, c))
            else:
                if cur:
                    runs.append(cur)
                cur = []
        if cur:
            runs.append(cur)
    return runs


def cut_windows(runs: List[List[Tuple[str, int]]], rng: random.Random, n_windows: int, bin_p: Dict[str, float], tries: int = 6) -> List[Tuple[str, int, int]]:
    """Up to n_windows (text, afro tokens, n_sentences) from the runs of ONE document: a bin is drawn, a token budget inside it, a start sentence, sentences are added while the budget holds.
    Runs of the same document are joined when a bin needs more than one run holds (paragraph boundaries inside a document do not matter for a window: it is running text)."""
    flat = [x for r in runs for x in r]
    if not flat:
        return []
    out, used = [], set()
    names = list(bin_p)
    probs = [bin_p[b] for b in names]
    for _ in range(tries * n_windows):
        if len(out) >= n_windows:
            break
        b = rng.choices(names, probs)[0]
        lo, hi = EDGES[b]
        budget = rng.randint(max(lo, 8), hi - 1)
        # candidate start sentences: those from which the rest of the document can still reach the bin's lower edge
        tail = [0] * (len(flat) + 1)
        for k in range(len(flat) - 1, -1, -1):
            tail[k] = tail[k + 1] + flat[k][1]
        starts = [k for k in range(len(flat)) if tail[k] + 2 >= max(lo, 8)]
        if not starts:
            continue
        a = rng.choice(starts)
        toks, e = 2, a
        while e < len(flat) and toks + flat[e][1] <= budget:
            toks += flat[e][1]
            e += 1
        if e == a or toks < max(lo, 8) or toks > hi - 1 or (a, e) in used:
            continue
        used.add((a, e))
        out.append((" ".join(s for s, _ in flat[a:e]), toks, e - a))
    return out


def bin_of(n: int) -> str:
    for b in BINS:
        lo, hi = EDGES[b]
        if lo <= n < hi:
            return b
    return BINS[-1]


# ------------------------------------------------------------------------------------------------------------------------------ source readers
def stream_jsonl_gz(url: str, max_bytes: int) -> Iterator[dict]:
    """Rows of a remote .jsonl.gz, cut off after `max_bytes` COMPRESSED bytes were read (nothing is written to disk)."""
    import requests

    tok = os.environ.get("HF_TOKEN")
    r = requests.get(url, headers={"Authorization": f"Bearer {tok}"} if tok else {}, stream=True, timeout=120)
    r.raise_for_status()
    seen = [0]

    class Counting(io.RawIOBase):
        def readable(self):
            return True

        def readinto(self, b):
            chunk = r.raw.read(len(b))
            seen[0] += len(chunk)
            if seen[0] > max_bytes:
                return 0
            b[: len(chunk)] = chunk
            return len(chunk)

    g = io.BufferedReader(gzip.GzipFile(fileobj=io.BufferedReader(Counting())))
    try:
        for line in g:
            try:
                yield json.loads(line)
            except Exception:
                return                                  # the cut-off truncated the last row
    except (EOFError, OSError):
        return
    finally:
        r.close()


def doc_pmc(d: dict) -> Optional[Tuple[str, str, str, str]]:
    lic = (d.get("metadata") or {}).get("license") or ""
    short = "cc-by" if "Attribution" in lic and "NonCommercial" not in lic and "ShareAlike" not in lic else ("cc0" if "Zero" in lic or "Public" in lic else None)
    if short is None:
        return None
    jr = ((d.get("metadata") or {}).get("journal") or "").split(".")[0].strip()
    return d["id"], clean_pmc(d["text"]), short, f"pmc:{jr}"


def doc_arxiv(d: dict) -> Optional[Tuple[str, str, str, str]]:
    lic = (d.get("metadata") or {}).get("license") or ""
    short = ("cc-by" if "Attribution" in lic and "NonCommercial" not in lic and "ShareAlike" not in lic and "NoDerivative" not in lic
             else "cc0" if ("Zero" in lic or "Public Domain" in lic) else None)
    if short is None:
        return None
    return str(d["id"]), d["text"], short, "arxiv"


def doc_libre(d: dict) -> Optional[Tuple[str, str, str, str]]:
    md = d.get("metadata") or {}
    m = re.match(r"https?://([a-z0-9]+)\.libretexts\.org", d.get("id") or md.get("url") or "")
    field = LIBRE_SCIENCE_HOSTS.get(m.group(1)) if m else None
    if not field:
        return None
    lic = md.get("license") or ""
    short = "cc-by" if "Attribution" in lic and "ShareAlike" not in lic and "NonCommercial" not in lic else ("cc-by-sa" if "ShareAlike" in lic and "NonCommercial" not in lic else None)
    if short is None:
        return None
    return d["id"], d["text"], short, f"libretexts:{field}"


def _cut_doc(args) -> List[dict]:
    """worker: one document -> window rows"""
    global _W
    try:
        _W
    except NameError:
        _W = Worker()
    source, doc_id, text, licence, field, n_win = args
    rng = random.Random(f"{SEED}|{source}|{doc_id}")
    libre = source == "libretexts"
    if source == "libretexts":
        paras = [_WS.sub(" ", NFC(text)).strip()]
    else:
        paras = paragraphs_of(text)
    runs = runs_of(_W, paras, libre)
    if sum(c for r in runs for _, c in r) < 700:
        return []
    rows = []
    for txt, n, ns in cut_windows(runs, rng, n_win, LONG_SOURCE_BIN_P):
        rows.append({"english": txt, "src_afr_tokens": n, "length_bin": bin_of(n), "n_sentences": ns, "source_dataset": SOURCES[source]["name"], "licence": licence, "field": field,
                     "doc_id": f"{source}:{doc_id}", "window_id": hashlib.blake2b(txt.encode(), digest_size=8).hexdigest()})
    return rows


def _abs_row(args) -> Optional[dict]:
    global _W
    try:
        _W
    except NameError:
        _W = Worker()
    d, = args
    t = _WS.sub(" ", NFC(d.get("abstract") or "")).strip()
    if not t or LATEX_TOKENS.search(t) or t.count("$") > 2 or _URL.search(t) or _HTML.search(t):
        return None
    sents = _W.sentences(t)
    if len(sents) < 2 or not all(prose_ok(s) for s in sents):
        return None
    n = _W.ntok(t)
    cat = (d.get("categories") or ["?"])[0].split()[0].split(".")[0]
    return {"english": t, "src_afr_tokens": n, "length_bin": bin_of(n), "n_sentences": len(sents), "source_dataset": SOURCES["arxiv_abs"]["name"], "licence": "cc0-1.0",
            "field": f"arxiv:{cat}", "doc_id": f"arxiv_abs:{d['id']}", "window_id": hashlib.blake2b(t.encode(), digest_size=8).hexdigest()}


def afriscience_docs(split: str = "train") -> List[Tuple[str, str, List[str], str]]:
    import pandas as pd

    d = pd.read_parquet(os.path.join(AFS_DIR, f"{split}-00000.parquet"))
    e = d[d.lang_pair == "eng-hau"].sort_values(["paper_id", "sentence_id"])        # the English side is identical in all six lang_pairs (checked)
    return [(str(pid), g.domain.iloc[0], [NFC(x) for x in g.source.tolist()], split) for pid, g in e.groupby("paper_id")]


def cmd_windows(a) -> int:
    import pandas as pd

    t0 = time.time()
    src = a.source
    rows: List[dict] = []
    if src == "afriscience":
        W = Worker()
        rng = random.Random(f"{SEED}|afs")
        for pid, dom, sents, _ in afriscience_docs("train"):
            toks = [len(e.ids) for e in W.tk.encode_batch(sents, add_special_tokens=False)]
            whole = " ".join(sents)
            nt = sum(toks) + 2
            base = {"source_dataset": SOURCES[src]["name"], "licence": "apache-2.0", "field": f"afriscience:{dom}", "doc_id": f"afriscience:{pid}"}
            rows.append({**base, "english": whole, "src_afr_tokens": W.ntok(whole), "length_bin": bin_of(W.ntok(whole)), "n_sentences": len(sents),
                         "window_id": hashlib.blake2b(whole.encode(), digest_size=8).hexdigest()})          # the WHOLE document (what the benchmark translates)
            runs = [[(s, c) for s, c in zip(sents, toks)]]
            per_bin = {"b64": 3, "b256": 3, "b512": 3, "b1024": 1}
            for b, k in per_bin.items():
                for txt, n, ns in cut_windows(runs, rng, k, {b: 1.0}):
                    rows.append({**base, "english": txt, "src_afr_tokens": n, "length_bin": bin_of(n), "n_sentences": ns, "window_id": hashlib.blake2b(txt.encode(), digest_size=8).hexdigest()})
    elif src == "arxiv_abs":
        docs = []
        for k, d in enumerate(stream_jsonl_gz(SOURCES[src]["url"], a.max_bytes)):
            if k % 3 == 0:
                docs.append((d,))
        with mp.Pool(a.workers) as pool:
            rows = [r for r in pool.imap(_abs_row, docs, chunksize=200) if r]
    else:
        reader = {"pmc": doc_pmc, "arxiv": doc_arxiv, "libretexts": doc_libre}[src]
        jobs = []
        n_docs = 0
        for d in stream_jsonl_gz(SOURCES[src]["url"], a.max_bytes):
            n_docs += 1
            x = reader(d)
            if x is None or len(x[1]) < 6000:
                continue
            jobs.append((src, x[0], x[1], x[2], x[3], WINDOWS_PER_DOC[src]))
        print(f"[windows:{src}] streamed {n_docs} documents, {len(jobs)} usable (licence / length)", flush=True)
        with mp.Pool(a.workers) as pool:
            for r in pool.imap_unordered(_cut_doc, jobs, chunksize=4):
                rows.extend(r)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.to_parquet(a.out, index=False)
    print(f"[windows:{src}] {len(df)} windows in {time.time() - t0:.0f}s -> {a.out}; per bin {df.length_bin.value_counts().to_dict() if len(df) else {}}", flush=True)
    return 0


# ------------------------------------------------------------------------------------------------------------------------------ hold-out
_WORD = re.compile(r"\w+", re.U)


def norm_sentence(s: str) -> str:
    return _WS.sub(" ", NFC(s)).strip().lower()


def ngrams8(text: str) -> Iterator[int]:
    w = _WORD.findall(NFC(text).lower())
    for i in range(len(w) - 7):
        yield hash(" ".join(w[i:i + 8]))


class HoldOut:
    """Held-out sentences (exact, normalised) and 8-gram hashes. `hit(text)` is True when any sentence of `text` equals a held-out sentence or any word 8-gram is held out.
    Python's str hash is randomised per process: the index is rebuilt per process from the texts (cheap), never pickled."""

    def __init__(self, texts: Iterable[str], sentences: Iterable[str]):
        self.sent = {norm_sentence(s) for s in sentences if len(s) > 20}
        self.grams = set()
        for t in texts:
            self.grams.update(ngrams8(t))

    def hit(self, text: str, sents: Optional[Sequence[str]] = None) -> Optional[str]:
        if sents is None:
            sents = re.split(r"(?<=[.!?])\s+", text)
        for s in sents:
            if norm_sentence(s) in self.sent:
                return "sentence"
        for g in ngrams8(text):
            if g in self.grams:
                return "ngram8"
        return None


def holdout_texts() -> Tuple[List[str], List[str], Dict[str, int]]:
    texts, sents, cnt = [], [], {}
    for split in ("dev", "test"):
        for pid, dom, ss, _ in afriscience_docs(split):
            texts.append(" ".join(ss))
            texts.extend(ss)
            sents.extend(ss)
            cnt[f"afriscience_{split}_docs"] = cnt.get(f"afriscience_{split}_docs", 0) + 1
    for rel in ("hard/docs.jsonl", "core/docs.jsonl"):
        p = os.path.join(LONGDOC_DIR, rel)
        if not os.path.exists(p):
            continue
        n = 0
        for line in open(p, encoding="utf-8"):
            d = json.loads(line)
            texts.append(d.get("text") or "")
            sents.extend(d.get("sents") or [])
            n += 1
        cnt[f"afrilongdoc_{rel.split('/')[0]}_docs"] = n
    return texts, sents, cnt


def cmd_holdout(a) -> int:
    texts, sents, cnt = holdout_texts()
    h = HoldOut(texts, sents)
    json.dump({"counts": cnt, "n_sentences": len(h.sent), "n_8grams": len(h.grams)}, open(a.out, "w"), indent=1)
    print(f"[holdout] {cnt}; {len(h.sent)} sentences, {len(h.grams)} word 8-grams -> {a.out}", flush=True)
    return 0


# ------------------------------------------------------------------------------------------------------------------------------ general replay
def cmd_general(a) -> int:
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq

    sys.path.insert(0, os.environ.get("GEMMA_MT_SRC", "/workspace/jobs/a4/gemma_mt_sci"))
    from gemma_mt.data.prepare_sft import split_fraction

    files = sorted(glob.glob(os.path.join(a.v3, "**", "*.parquet"), recursive=True))
    t0 = time.time()
    # pass 1: the 2c hold-out is the `eval_prompts`(128) smallest-hash English texts (+ the per-bin top-up, <= 16 per bin and its duplicates): a superset is excluded = the `--exclude-smallest` smallest
    # hashes of EVERY bin (default 600), which contains every prompt the 2c run ever evaluated on
    keep: Dict[str, List[Tuple[float, str]]] = {b: [] for b in BINS}
    for f in files:
        t = pq.read_table(f, columns=["english", "length_bin"]).to_pandas()
        for b in BINS:
            tb = t[t.length_bin == b].english.tolist()
            fr = [(split_fraction(e, a.seed), e) for e in tb]
            keep[b] = sorted(keep[b] + fr)[: a.exclude_smallest]
    excl = {e for b in BINS for _, e in keep[b]}
    print(f"[general] excluding {len(excl)} 2c hold-out candidates; pass 1 {time.time() - t0:.0f}s", flush=True)
    rng = np.random.default_rng(SEED)
    out = []
    cols = ["english", "iso", "language", "src_afr_tokens", "length_bin", "n_sentences", "source_dataset", "licence"]
    for f in files:
        t = pq.read_table(f).to_pandas()
        t = t[[c for c in cols if c in t.columns]]
        t = t[~t.english.isin(excl)]
        for (iso, b), g in t.groupby(["iso", "length_bin"]):
            take = g.sample(n=min(a.per_cell, len(g)), random_state=int(rng.integers(2 ** 31)))
            out.append(take)
        del t
    df = pd.concat(out, ignore_index=True)
    df["domain"] = "general"
    df["field"] = "general"
    df["doc_id"] = ""
    df["window_id"] = [hashlib.blake2b(e.encode(), digest_size=8).hexdigest() for e in df.english]
    df.to_parquet(a.out, index=False)
    print(f"[general] {len(df)} rows ({df.iso.nunique()} languages) in {time.time() - t0:.0f}s -> {a.out}; per bin {df.length_bin.value_counts().to_dict()}", flush=True)
    return 0


# ------------------------------------------------------------------------------------------------------------------------------ assemble
# rows per language and bin of the SCIENCE share (the schedule draws with replacement; the 500-1,000 steps x 42 prompts x 0.7 = 15-30k science draws spread over 77 languages)
CELLS = {"b64": 80, "b256": 120, "b512": 150, "b1024": 220, "b2048": 220}
# share of each cell taken from AfriScience-MT train windows (the benchmark's own domain): the six benchmark languages see it more
AFS_SHARE = {"benchmark": 0.5, "other": 0.2}
BENCH_LANGS = ("amh", "hau", "lug", "nso", "yor", "zul")
# the rest of a cell: by source (fallback when a source has no window in the bin)
OTHER_SOURCE_W = {"pmc": 0.40, "libretexts": 0.30, "arxiv": 0.15, "arxiv_abs": 0.15}


def cmd_assemble(a) -> int:
    import pandas as pd

    sys.path.insert(0, os.environ.get("GEMMA_MT_SRC", "/workspace/jobs/a4/gemma_mt_sci"))
    from gemma_mt.languages import LANGUAGES, SUPPORTED_RL

    texts, sents, hcount = holdout_texts()
    ho = HoldOut(texts, sents)
    print(f"[assemble] hold-out: {hcount}; {len(ho.sent)} sentences, {len(ho.grams)} 8-grams", flush=True)
    frames = []
    drop = collections.Counter()
    for f in a.windows:
        df = pd.read_parquet(f)
        df["src"] = os.path.basename(f).split(".")[0].replace("windows_", "")
        frames.append(df)
    pool = pd.concat(frames, ignore_index=True).drop_duplicates("window_id")
    keep = []
    for i, e in enumerate(pool.english.tolist()):
        why = ho.hit(e)
        if why:
            drop[(pool.src.iloc[i], why)] += 1
        keep.append(why is None)
    pool = pool[keep].reset_index(drop=True)
    print(f"[assemble] science pool after hold-out dedup: {len(pool)} windows; dropped {dict(drop)}", flush=True)
    gen = pd.read_parquet(a.general)
    gk = [ho.hit(e) is None for e in gen.english.tolist()]
    print(f"[assemble] general replay: dropped {len(gen) - sum(gk)} of {len(gen)} by the hold-out", flush=True)
    gen = gen[gk].reset_index(drop=True)

    rng = random.Random(SEED)
    by = {(b, s): pool[(pool.length_bin == b) & (pool.src == s)].index.tolist() for b in BINS for s in pool.src.unique()}
    pool_stats = {f"{b}/{s}": len(v) for (b, s), v in by.items()}
    out_rows = []
    short = collections.Counter()
    for iso in sorted(SUPPORTED_RL):
        for b in BINS:
            n = CELLS[b]
            share = AFS_SHARE["benchmark" if iso in BENCH_LANGS else "other"]
            plan = {"afriscience": int(round(n * share))}
            avail = {s: OTHER_SOURCE_W[s] for s in OTHER_SOURCE_W if by.get((b, s))}
            tot = sum(avail.values())
            for s, w in avail.items():
                plan[s] = int(round((n - plan["afriscience"]) * w / tot)) if tot else 0
            chosen = []
            for s, k in plan.items():
                cand = by.get((b, s), [])
                take = rng.sample(cand, min(k, len(cand)))
                chosen += take
                if len(take) < k:
                    short[(b, s)] += k - len(take)
            # fill a shortfall from any source of the bin (no duplicates inside a language)
            if len(chosen) < n:
                rest = [i for s in pool.src.unique() for i in by.get((b, s), []) if i not in set(chosen)]
                chosen += rng.sample(rest, min(n - len(chosen), len(rest)))
            for i in chosen:
                r = pool.loc[i]
                out_rows.append({"english": r.english, "iso": iso, "language": LANGUAGES[iso].name, "src_afr_tokens": int(r.src_afr_tokens), "length_bin": b, "n_sentences": int(r.n_sentences),
                                 "source_dataset": r.source_dataset, "licence": r.licence, "domain": "science", "field": r.field, "doc_id": r.doc_id, "window_id": r.window_id})
    sci = pd.DataFrame(out_rows)
    allr = pd.concat([sci, gen[sci.columns]], ignore_index=True)
    allr["src_afr_tokens"] = allr.src_afr_tokens.astype("int32")
    allr["n_sentences"] = allr.n_sentences.fillna(0).astype("int32")
    os.makedirs(a.out, exist_ok=True)
    for iso, g in allr.groupby("iso"):
        g.sample(frac=1.0, random_state=SEED).to_parquet(os.path.join(a.out, f"{iso}.parquet"), index=False)
    stats = {"seed": SEED, "rows": len(allr), "languages": int(allr.iso.nunique()), "holdout": hcount, "holdout_sentences": len(ho.sent), "holdout_8grams": len(ho.grams),
             "dropped_by_holdout": {f"{k[0]}/{k[1]}": v for k, v in drop.items()}, "pool_windows": pool_stats, "cell_shortfall": {f"{k[0]}/{k[1]}": v for k, v in short.items()},
             "per_domain_bin": {f"{d}/{b}": int(n) for (d, b), n in allr.groupby(["domain", "length_bin"]).size().items()},
             "per_domain_source": {f"{d}/{s}": int(n) for (d, s), n in allr.groupby(["domain", "source_dataset"]).size().items()},
             "per_domain_licence": {f"{d}/{s}": int(n) for (d, s), n in allr.groupby(["domain", "licence"]).size().items()},
             "science_unique_english": int(sci.english.nunique()), "science_cells": CELLS, "afs_share": AFS_SHARE, "other_source_weights": OTHER_SOURCE_W}
    json.dump(stats, open(os.path.join(os.path.dirname(os.path.abspath(a.out)), os.path.basename(a.out.rstrip("/")) + "_stats.json"), "w"), indent=1)
    print(json.dumps({k: stats[k] for k in ("rows", "languages", "science_unique_english", "dropped_by_holdout", "cell_shortfall")}, indent=1), flush=True)
    print("per domain/bin:", stats["per_domain_bin"], flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("windows"); p.add_argument("--source", required=True, choices=list(SOURCES)); p.add_argument("--out", required=True)
    p.add_argument("--max-bytes", type=int, default=150_000_000); p.add_argument("--workers", type=int, default=6); p.set_defaults(fn=cmd_windows)
    p = sub.add_parser("holdout"); p.add_argument("--out", required=True); p.set_defaults(fn=cmd_holdout)
    p = sub.add_parser("general"); p.add_argument("--v3", required=True); p.add_argument("--out", required=True); p.add_argument("--per-cell", type=int, default=120)
    p.add_argument("--exclude-smallest", type=int, default=600); p.add_argument("--seed", type=int, default=1234); p.set_defaults(fn=cmd_general)
    p = sub.add_parser("assemble"); p.add_argument("--windows", nargs="+", required=True); p.add_argument("--general", required=True); p.add_argument("--out", required=True); p.set_defaults(fn=cmd_assemble)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
