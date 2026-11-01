#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build the b2048 length bin of the Gemma RL prompts (DECISIONS.md D-45) and the prompt directory `rl_prompts_v3`.

b2048 = English source windows of 1,025..2,048 AfriCOMET-encoder tokens (Davlan/afro-xlmr-large-76L, `tokenizer.json`, special tokens counted, exactly as
`src_afr_tokens` of `rl_prompts_v2`), made of CONTIGUOUS WHOLE SENTENCES of ONE document (sentences = `gemma_mt.rl.segments.split_sentences`, joined by single
spaces, no newlines), from the same two English corpora as the other bins (HuggingFaceFW/fineweb-edu sample-10BT, odc-by; wikimedia/wikipedia 20231101.en, share-alike),
passed through the same window filters as `gemma/rl_prompts_v2_meta/stats.json` lists them.

THE ORIGINAL BUILDER (`gemma_rl_windows.py` / `gemma_rl_assemble.py`, old box) IS NOT ON THE RENTAL NOR IN THE REPO. The filters below were therefore RE-DERIVED from the
1.4 M kept windows of rl_prompts_v2 (envelope method: a rule is accepted when 100 % of the kept windows of a 58k sample pass it AND the sample shows a sharp cliff at the
threshold; see `window_filter` and reports/b2048_bin_2026-10-20.md section 3). Exact (a cliff in the old data): repetition (no 4-gram of [a-z']+ words seen >= 3 times),
numeric (digits / characters <= 0.06), technical vocabulary (share of words with >= 13 letters <= 0.03), hard to read (Flesch reading ease >= 30 with the vowel-group
syllable rule). Approximated (no cliff): artifact, shouting / acronyms, math symbols.

Sub-commands (CPU only; run with CUDA_VISIBLE_DEVICES= and nice):

  windows   --source fineweb|wiki --shard K --out FILE.parquet [--workers 24]     download ONE raw shard, cut windows, filter, write them, delete the shard
  hashes    --v2 DIR --out FILE.npy                                                sentence hashes of the existing b1024 + b512 windows (dedup index)
  assemble  --windows F1 F2 ... --v2 DIR --hashes FILE.npy --out DIR_v3 --per-language 600 --doc-ids FILE.txt
            builds DIR_v3 = hard links of the 77 old per-language files (unchanged, same order) + DIR_v3/zz_b2048/<iso>.parquet (the new rows; the subdirectory
            name sorts after every old file, so every old row keeps its global row index in `load_rl_prompts`)
  tokens    --v3 DIR [--model PATH]                                                Gemma prompt-token statistics of the b2048 rows (for vllm_max_model_len)
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import multiprocessing as mp
import os
import re
import sys
import time
import unicodedata
from typing import Dict, List, Optional, Sequence, Tuple

LO, HI = 1025, 2048                        # b2048 = AfriCOMET tokens (special tokens counted) in [1025, 2048]
SEED = 20261020
SOURCES = {
    "fineweb": dict(repo="HuggingFaceFW/fineweb-edu", file="sample/10BT/{k:03d}_00000.parquet", name="HuggingFaceFW/fineweb-edu:sample-10BT",
                    licence="odc-by", id_col="id", tag="fineweb-edu"),
    "wiki": dict(repo="wikimedia/wikipedia", file="20231101.en/train-{k:05d}-of-00041.parquet", name="wikimedia/wikipedia:20231101.en",
                 licence="share_alike", id_col="id", tag="wikipedia"),
}
AFRO_TOKENIZER_GLOB = "/workspace/sft_build/hf/hub/models--Davlan--afro-xlmr-large-76L/snapshots/*/tokenizer.json"


# ---------------------------------------------------------------------------------------------------------------- window filters (pure functions)
HARD_ARTIFACT = re.compile(r"https?:|www\.|\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b|©|[\u00a0\t\x00-\x08\x0b-\x1f\x7f\u200b-\u200f\u2028\u2029\ufeff\ufffd]|\[\d{1,3}\]|\[citation|\{\{|\}\}|<[a-z/][^>]*>")
SOFT_ARTIFACT = re.compile(r"(?i)click here|read more|subscribe|share this|cookies?\b|all rights reserved|sign up|log in|follow us|privacy policy|terms of (?:use|service)|"
                           r"\badvertisement\b|\bnewsletter\b|\bcomments? (?:are|is) closed\b|\bleave a (?:reply|comment)\b|\bposted by\b|\bfile:|\bthumb\||\bimage:")
VERSE = re.compile(r"\b\d{1,3}:\d{1,3}\b")
MENU = re.compile(r"[|•►▶»«]")
MATH_CHARS = re.compile(r"[=<>±×÷∑∫√∞≈≠≤≥^\\]")
LATEX = re.compile(r"\\(?:frac|sum|int|alpha|beta|begin|end|left|right|sqrt)\b")
WORD_A = re.compile(r"[A-Za-z]+")
WORD_APOS = re.compile(r"[A-Za-z']+")
FOUR_GRAM_WORD = re.compile(r"[a-z']+")
VOWEL_GROUP = re.compile(r"[aeiouy]+")

DIGIT_MAX = 0.06                 # digits / characters; old kept windows: p99.99 0.0597, max 0.0600  (cliff)
LONG_WORD_MAX = 0.03             # share of [A-Za-z]+ words with >= 13 letters; old kept: max 0.0287..0.0300 in every bin (cliff)
CAPS_MAX = 0.15                  # share of ALL-CAPS words with >= 3 letters (old kept: max 0.11-0.14; no cliff: approximated; rule is 0.15% of windows)
FLESCH_MIN = 30.0                # rl_prompts_v2_meta/stats.json reading_ease_min; old kept (vowel-group syllables, [.!?]+ sentences): min 29.8, p0.1 30.1 (cliff)
REPEAT_4GRAM_MAX = 2             # no 4-gram of lower-cased [a-z']+ words seen >= 3 times; old kept: max exactly 2 in 29k b512/b1024 windows (cliff)


def nfc_clean(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def flesch(text: str) -> float:
    words = WORD_APOS.findall(text)
    if not words:
        return 0.0
    sents = max(1, len([s for s in re.split(r"[.!?]+", text) if s.strip()]))
    syl = sum(max(1, len(VOWEL_GROUP.findall(w.lower()))) for w in words)
    return 206.835 - 1.015 * len(words) / sents - 84.6 * syl / len(words)


def max_4gram_count(text: str) -> int:
    w = FOUR_GRAM_WORD.findall(text.lower())
    if len(w) < 4:
        return 0
    return max(collections.Counter(zip(w, w[1:], w[2:], w[3:])).values())


def bad_sentence(s: str) -> bool:
    """A sentence no window may contain (hard web artifacts); used to cut a document into runs before windows are drawn."""
    return bool(HARD_ARTIFACT.search(s))


def window_filter(text: str) -> Optional[str]:
    """None = the window is kept, else the first failing reason (names as in stats.json `window_filter`)."""
    if HARD_ARTIFACT.search(text) or len(SOFT_ARTIFACT.findall(text)) >= 2 or len(VERSE.findall(text)) >= 8 or len(MENU.findall(text)) >= 6:
        return "artifact"
    n = max(len(text), 1)
    if sum(c.isdigit() for c in text) / n > DIGIT_MAX or len(MATH_CHARS.findall(text)) >= 20 or LATEX.search(text):
        return "numeric_or_math"
    words = WORD_A.findall(text)
    nw = max(len(words), 1)
    if sum(1 for w in words if len(w) >= 3 and w.isupper()) / nw > CAPS_MAX:
        return "shouting_or_acronyms"
    if sum(1 for w in words if len(w) >= 13) / nw > LONG_WORD_MAX:
        return "technical_vocabulary"
    if flesch(text) < FLESCH_MIN:
        return "hard_to_read"
    if max_4gram_count(text) > REPEAT_4GRAM_MAX:
        return "repetition"
    return None


# ---------------------------------------------------------------------------------------------------------------- document -> windows
_TERMINAL = ('.', '!', '?', '"', '”', '’', ')', '።')


def clean_paragraphs(text: str) -> List[str]:
    """Complete-sentence lines of a document: NFC, white space collapsed; headings, menus, list items and captions (lines that are not >= 5 words ending in a sentence
    terminator) are dropped."""
    keep = []
    for line in unicodedata.normalize("NFC", text).split("\n"):
        line = re.sub(r"\s+", " ", line).strip()
        if len(line.split()) >= 5 and line.endswith(_TERMINAL):
            keep.append(line)
    return keep


def clean_doc(text: str) -> str:
    """Running prose of a document: `clean_paragraphs` joined by one space. The windows carry no newline (D-23: the model must not learn to translate correctly
    only when a line break follows every sentence)."""
    return " ".join(clean_paragraphs(text))


class Worker:
    """Per-process state: tokenizer + splitter."""

    def __init__(self):
        from tokenizers import Tokenizer

        sys.path.insert(0, os.environ.get("GEMMA_MT_SRC", "/workspace/sft_build/gemma_mt_2c"))
        from gemma_mt.rl.segments import split_sentences

        self.split = split_sentences
        self.tk = Tokenizer.from_file(glob.glob(AFRO_TOKENIZER_GLOB)[0])

    def ntok(self, text: str) -> int:
        return len(self.tk.encode(text, add_special_tokens=True).ids)

    def windows(self, doc_id: str, text: str, max_windows: int = 2, tries: int = 4) -> Tuple[List[dict], collections.Counter]:
        stat: collections.Counter = collections.Counter()
        stat["docs"] += 1
        if len(text) < 3600:
            stat["doc_short_chars"] += 1
            return [], stat
        paras = clean_paragraphs(text)
        if sum(len(p) + 1 for p in paras) < 3600:
            stat["doc_short_clean"] += 1
            return [], stat
        sents = [s for p in paras for s in self.split(p)]          # per paragraph: `split_sentences` is quadratic in the length of its input (a whole document took 0.1 s)
        counts = [len(e.ids) for e in self.tk.encode_batch(sents, add_special_tokens=False)]
        if sum(counts) + 2 < LO:
            stat["doc_short_tokens"] += 1
            return [], stat
        stat["docs_long_enough"] += 1
        # runs of consecutive sentences without a hard artifact
        runs, cur = [], []
        for i, s in enumerate(sents):
            if bad_sentence(s) or counts[i] > 600:
                if cur:
                    runs.append(cur)
                cur = []
            else:
                cur.append(i)
        if cur:
            runs.append(cur)
        runs = [r for r in runs if sum(counts[i] for i in r) + 2 >= LO]
        if not runs:
            stat["doc_no_clean_run"] += 1
            return [], stat
        rng = __import__("numpy").random.default_rng(int.from_bytes(hashlib.blake2b(f"{SEED}|{doc_id}".encode(), digest_size=8).digest(), "little"))
        out: List[dict] = []
        used: List[Tuple[int, int]] = []
        for _ in range(tries):
            if len(out) >= max_windows:
                break
            mass = [sum(counts[i] for i in r) for r in runs]
            r = runs[int(rng.choice(len(runs), p=[m / sum(mass) for m in mass]))]
            budget = int(rng.integers(LO, HI + 1))
            # starts with enough tokens left in the run
            tail = [0] * (len(r) + 1)
            for k in range(len(r) - 1, -1, -1):
                tail[k] = tail[k + 1] + counts[r[k]]
            starts = [k for k in range(len(r)) if tail[k] + 2 >= min(budget, LO)]
            if not starts:
                continue
            a = starts[int(rng.integers(len(starts)))]
            budget = min(budget, tail[a] + 2)
            tot, b = 2, a
            while b < len(r) and tot + counts[r[b]] <= HI and tot < budget:
                tot += counts[r[b]]
                b += 1
            if tot < max(LO, int(0.9 * budget)) or b - a < 8:
                stat["try_budget_miss"] += 1
                continue
            lo_i, hi_i = r[a], r[b - 1]
            if any(not (hi_i < u0 or lo_i > u1) for u0, u1 in used):
                stat["try_overlap"] += 1
                continue
            text_w = " ".join(sents[r[k]] for k in range(a, b))
            n = self.ntok(text_w)
            if not LO <= n <= HI:
                stat["try_token_miss"] += 1
                continue
            why = window_filter(text_w)
            stat["win_" + (why or "kept")] += 1
            if why:
                continue
            used.append((lo_i, hi_i))
            out.append({"english": text_w, "src_afr_tokens": n, "n_sentences": len(self.split(text_w)), "doc_id": doc_id})
        return out, stat


_W: Optional[Worker] = None


def _work(batch):
    global _W
    if _W is None:
        _W = Worker()
    rows, stat = [], collections.Counter()
    for doc_id, url, text in batch:
        ws, st = _W.windows(doc_id, text)
        stat.update(st)
        for w in ws:
            w["doc_url"] = url
            rows.append(w)
    return rows, stat


def cmd_windows(a):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    src = SOURCES[a.source]
    t0 = time.time()
    path = hf_hub_download(src["repo"], src["file"].format(k=a.shard), repo_type="dataset", cache_dir=a.cache_dir)
    print(f"downloaded {path} ({os.path.getsize(path) / 1e9:.2f} GB, {time.time() - t0:.0f}s)", flush=True)
    pf = pq.ParquetFile(path)
    cols = [c for c in ("text", src["id_col"], "url") if c in pf.schema_arrow.names]
    rows: List[dict] = []
    stat: collections.Counter = collections.Counter()
    pool = mp.Pool(a.workers)

    def batches():
        buf = []
        for rb in pf.iter_batches(batch_size=4096, columns=cols):
            d = rb.to_pydict()
            for i in range(len(d["text"])):
                t = d["text"][i]
                if t is not None and len(t) >= 3600:                       # a 1,025-token window needs ~4,300+ characters of English; cheap pre-screen
                    buf.append((f"{src['tag']}:{d[src['id_col']][i]}", (d.get("url") or [""] * len(d["text"]))[i] or "", t))
                else:
                    stat["docs_pre_short"] += 1
                if len(buf) >= 64:
                    yield buf
                    buf = []
        if buf:
            yield buf

    n = 0
    for r, st in pool.imap_unordered(_work, batches(), chunksize=1):
        rows += r
        stat.update(st)
        n += 1
        if n % 200 == 0:
            print(f"  {n * 64} long docs processed, {len(rows)} windows ({time.time() - t0:.0f}s) {dict(stat)}", flush=True)
        if a.max_windows and len(rows) >= a.max_windows:
            print(f"  --max-windows {a.max_windows} reached after {n * 64} long docs: stopping early", flush=True)
            break
    pool.terminate()
    pool.join()
    for r in rows:
        r["source_dataset"], r["licence"] = src["name"], src["licence"]
    tbl = pa.Table.from_pylist(rows) if rows else pa.table({"english": []})
    pq.write_table(tbl, a.out, compression="zstd")
    json.dump({"source": a.source, "shard": a.shard, "windows": len(rows), "stat": dict(stat), "seconds": round(time.time() - t0)}, open(a.out + ".stats.json", "w"), indent=1)
    print(f"wrote {len(rows)} windows -> {a.out}; {dict(stat)} ({time.time() - t0:.0f}s)", flush=True)
    if not a.keep_raw:
        try:
            os.remove(os.path.realpath(path))
            os.remove(path)
        except OSError:
            pass


# ---------------------------------------------------------------------------------------------------------------- dedup index
_FAST_SPLIT = re.compile(r"(?<=[.!?])\s+")


def fast_sentences(text: str) -> List[str]:
    """Cheap sentence split used ONLY for the duplicate index (same function on both sides, so it need not equal `split_sentences`; that one costs ~5 ms per window)."""
    return [s for s in _FAST_SPLIT.split(text) if len(s) >= 25]


def sent_hash(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(re.sub(r"\s+", " ", s).strip().lower().encode(), digest_size=8).digest(), "little")


def _hash_file(args):
    path, bins = args
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    sys.path.insert(0, os.environ.get("GEMMA_MT_SRC", "/workspace/sft_build/gemma_mt_2c"))
    from gemma_mt.rl.segments import split_sentences

    t = pq.read_table(path, columns=["english", "length_bin"])
    t = t.filter(pc.is_in(t["length_bin"], value_set=__import__("pyarrow").array(list(bins))))
    seen, hs = set(), []
    for e in t["english"].to_pylist():
        wh = hash(e)
        if wh in seen:
            continue
        seen.add(wh)
        hs.extend(sent_hash(s) for s in fast_sentences(e))
    import numpy as np

    return np.unique(np.asarray(hs, dtype=np.uint64))


def cmd_hashes(a):
    import numpy as np

    files = sorted(glob.glob(os.path.join(a.v2, "*.parquet")))
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        parts = pool.map(_hash_file, [(f, ("b1024", "b512")) for f in files], chunksize=1)
    allh = np.unique(np.concatenate(parts))
    np.save(a.out, allh)
    print(f"{len(files)} files -> {len(allh)} distinct sentence hashes ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------------------------------------------------------- assemble
def cmd_assemble(a):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    sys.path.insert(0, os.environ.get("GEMMA_MT_SRC", "/workspace/sft_build/gemma_mt_2c"))
    from gemma_mt.languages import LANGUAGES, SUPPORTED_RL
    from gemma_mt.rl.segments import split_sentences

    langs = sorted(SUPPORTED_RL)
    old = {l: os.path.join(a.v2, f"{l}.parquet") for l in langs}
    missing = [l for l, p in old.items() if not os.path.exists(p)]
    assert not missing, f"v2 files missing for {missing}"
    cand = pa.concat_tables([pq.read_table(f) for f in a.windows]).to_pandas()
    print(f"candidate windows: {len(cand)} from {len(a.windows)} files", flush=True)
    cand = cand.drop_duplicates("english").reset_index(drop=True)
    assert cand.src_afr_tokens.between(LO, HI).all()
    # drop windows that share sentences with the existing b1024 / b512 windows
    known = np.load(a.hashes)
    shared = []
    for e in cand.english:
        hs = np.fromiter((sent_hash(s) for s in fast_sentences(e)), dtype=np.uint64)
        pos = np.minimum(np.searchsorted(known, hs), len(known) - 1)          # `known` is sorted (np.unique); np.isin would re-sort 13 M hashes per window
        shared.append(float((known[pos] == hs).mean()) if len(hs) else 0.0)
    cand["shared"] = shared
    # the trainer's hold-out is the 128 prompts with the smallest hash of the English text (seed 1234): keep every b2048 window out of that selection, so the hold-out of the
    # running trainer (v2 rows) stays exactly the same set (a window with hash fraction < 3e-5 is dropped: ~2 of 20,000; the 128th smallest of 10 M rows is ~1.3e-5)
    from gemma_mt.data.prepare_sft import split_fraction

    low = np.array([split_fraction(e, 1234) < a.holdout_guard for e in cand.english])
    print(f"windows inside the hold-out hash range (< {a.holdout_guard}): {int(low.sum())} dropped", flush=True)
    cand = cand[~low].reset_index(drop=True)
    shared = cand["shared"].tolist()
    n0 = len(cand)
    cand = cand[cand.shared <= a.max_shared].reset_index(drop=True)
    print(f"dedup vs the b1024/b512 windows of {a.v2}: kept {len(cand)} of {n0} (windows with > {a.max_shared:.0%} shared sentences dropped; {int((np.array(shared) > 0).sum())} share any)", flush=True)
    # a window may not collide with an old English text (any bin) either: exact text
    rng = np.random.default_rng(SEED)
    need = a.per_language
    share_wiki = a.wiki_share
    per_lang_wiki = int(round(need * share_wiki))
    per_lang_fw = need - per_lang_wiki
    fw = cand[cand.source_dataset.str.startswith("HuggingFaceFW")].sample(frac=1.0, random_state=int(rng.integers(1 << 30))).reset_index(drop=True)
    wk = cand[cand.source_dataset.str.startswith("wikimedia")].sample(frac=1.0, random_state=int(rng.integers(1 << 30))).reset_index(drop=True)
    print(f"pool after dedup: fineweb-edu {len(fw)}, wikipedia {len(wk)}; need {per_lang_fw}+{per_lang_wiki} per language x {len(langs)}", flush=True)
    reuse = {"fineweb": len(fw) < per_lang_fw * len(langs), "wiki": len(wk) < per_lang_wiki * len(langs)}

    def take(pool, k, n):                                               # block k of n rows, wrapping (no duplicate inside a language while n <= len(pool))
        n = min(n, len(pool))
        idx = [(k * n + j) % len(pool) for j in range(n)]
        return pool.iloc[idx]

    os.makedirs(os.path.join(a.out, "zz_b2048"), exist_ok=True)
    for f in glob.glob(os.path.join(a.v2, "*")):                        # old files: hard links (same inode, never copied, v2 untouched)
        dst = os.path.join(a.out, os.path.basename(f))
        if os.path.isfile(f) and not os.path.exists(dst):
            os.link(f, dst)
    used_docs: Dict[str, str] = {}
    rows_per_lang = {}
    for k, l in enumerate(langs):
        part = __import__("pandas").concat([take(fw, k, per_lang_fw), take(wk, k, per_lang_wiki)]).sample(frac=1.0, random_state=int(rng.integers(1 << 30)))
        t = pa.table({
            "english": part.english.tolist(), "iso": [l] * len(part), "language": [LANGUAGES[l].name] * len(part),
            "src_afr_tokens": pa.array(part.src_afr_tokens.tolist(), pa.int32()), "length_bin": ["b2048"] * len(part),
            "n_sentences": pa.array(part.n_sentences.tolist(), pa.int32()), "source_dataset": part.source_dataset.tolist(),
            "licence": part.licence.tolist(), "doc_id": part.doc_id.tolist(),
            "window_id": [hashlib.blake2b(e.encode(), digest_size=8).hexdigest() for e in part.english]})
        pq.write_table(t, os.path.join(a.out, "zz_b2048", f"{l}.parquet"), compression="zstd")
        rows_per_lang[l] = len(part)
        for d, u in zip(part.doc_id, part.doc_url):
            used_docs[d] = u
    with open(a.doc_ids, "w") as f:
        f.write("# source_document_id<TAB>url   (documents whose text is in the b2048 RL prompts rl_prompts_v3/zz_b2048; ids are the `id` column of the raw corpus)\n")
        for d in sorted(used_docs):
            f.write(f"{d}\t{used_docs[d]}\n")
    summ = {"rows": sum(rows_per_lang.values()), "languages": len(langs), "rows_per_language": rows_per_lang, "distinct_windows": int(cand.english.nunique()),
            "distinct_documents_used": len(used_docs), "reuse": reuse, "src_afr_tokens": cand.src_afr_tokens.describe().round(1).to_dict(),
            "source_mix": cand.source_dataset.value_counts().to_dict(), "max_shared": a.max_shared}
    meta = a.out.rstrip("/") + "_meta"
    os.makedirs(meta, exist_ok=True)
    json.dump(summ, open(os.path.join(meta, "stats_b2048.json"), "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in summ.items() if k != "rows_per_language"}, indent=1, default=str), flush=True)


def cmd_tokens(a):
    import numpy as np
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    sys.path.insert(0, os.environ.get("GEMMA_MT_SRC", "/workspace/sft_build/gemma_mt_2c"))
    from gemma_mt import format as fmt

    tok = AutoTokenizer.from_pretrained(a.model)
    rng = np.random.default_rng(0)
    out = []
    for f in sorted(glob.glob(os.path.join(a.v3, "zz_b2048", "*.parquet")))[:: max(1, a.stride)]:
        t = pq.read_table(f, columns=["english", "iso", "src_afr_tokens"]).to_pandas()
        for i in rng.choice(len(t), size=min(a.per_file, len(t)), replace=False):
            ids = fmt.build_prompt(t.english.iloc[i], t.iso.iloc[i], tok)
            out.append((len(ids), int(t.src_afr_tokens.iloc[i])))
    arr = np.array(out)
    print(json.dumps({"n": len(arr), "prompt_tokens_pcts[0,50,90,99,100]": np.percentile(arr[:, 0], [0, 50, 90, 99, 100]).tolist(),
                      "gemma_per_afro_token_pcts[0,50,99,100]": np.percentile(arr[:, 0] / arr[:, 1], [0, 50, 99, 100]).round(3).tolist(), "afro_pcts": np.percentile(arr[:, 1], [0, 50, 90, 100]).tolist()}))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    w = sp.add_parser("windows")
    w.add_argument("--source", choices=sorted(SOURCES), required=True)
    w.add_argument("--shard", type=int, required=True)
    w.add_argument("--out", required=True)
    w.add_argument("--workers", type=int, default=24)
    w.add_argument("--cache-dir", default="/workspace/sft_build/b2048_build/raw_cache")
    w.add_argument("--keep-raw", action="store_true")
    w.add_argument("--max-windows", type=int, default=0, help="stop after this many windows (0 = the whole shard); the shard is in crawl order, so this is a prefix of it")
    h = sp.add_parser("hashes")
    h.add_argument("--v2", required=True)
    h.add_argument("--out", required=True)
    h.add_argument("--workers", type=int, default=16)
    s = sp.add_parser("assemble")
    s.add_argument("--windows", nargs="+", required=True)
    s.add_argument("--v2", required=True)
    s.add_argument("--hashes", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--doc-ids", required=True)
    s.add_argument("--per-language", type=int, default=600)
    s.add_argument("--wiki-share", type=float, default=0.25)
    s.add_argument("--max-shared", type=float, default=0.10)
    s.add_argument("--holdout-guard", type=float, default=3e-5, help="drop windows whose hold-out hash fraction is below this (the trainer's 128-prompt hash hold-out lives far below it)")
    t = sp.add_parser("tokens")
    t.add_argument("--v3", required=True)
    t.add_argument("--model", default="/workspace/sft_build/runs/rl_2b_stage_b64")
    t.add_argument("--stride", type=int, default=4)
    t.add_argument("--per-file", type=int, default=40)
    a = ap.parse_args(argv)
    {"windows": cmd_windows, "hashes": cmd_hashes, "assemble": cmd_assemble, "tokens": cmd_tokens}[a.cmd](a)


if __name__ == "__main__":
    main()
