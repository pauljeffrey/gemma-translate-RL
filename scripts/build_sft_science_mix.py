#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build the data of the SHORT SCIENCE-SFT STAGE that runs between the general RL (step 1433) and the science RL (D-58).

    python scripts/build_sft_science_mix.py --out /workspace/jobs/r4/mix_v1 [--seq-len 8192] [--science-share 0.35] ...

What it writes (the packed format of `gemma_mt.data.prepare_sft`, so `gemma_mt.train_sft` reads it unchanged):
    OUT/train/      science + replay samples packed together (seq_len tokens per pack, whole samples only, mixed languages per pack)
    OUT/val/        REPLAY validation: <= --val-per-language samples of every one of the 77 languages from the ORIGINAL SFT validation split
                    (hash-split on the English text, never trained on by the original SFT either): the forgetting monitor
    OUT/val_sci/    SCIENCE validation = AfriScience-MT DEV (sentence level 6 x 843 + whole papers 6 x 25): the progress monitor
                    (read by train_sft through `extra_val_splits: [val_sci]`); the TEST split is never read here
    OUT/meta.json   tokenizer fingerprint (must equal the fingerprint of the model's tokenizer), iso table, counts
    OUT/mixture_tables.{json,md}   examples / tokens / target tokens per language for science vs replay (train) and the validation sets

Science part (AfriScience-MT TRAIN, English -> amh hau lug nso yor zul; 177 papers, 5,792 sentence pairs per language, multi-parallel)
  * every sentence is used exactly once per PASS (--passes, default 2 = "two epochs over the science part"), but each pass cuts every
    paper into DIFFERENT windows: a paper gets one granularity per pass, drawn from  S (single sentences, the paper's sentence-level setting),
    A (2-4 consecutive sentences), B (5-12 consecutive sentences) or W (the WHOLE paper, the paper-level setting: sentences joined by single
    spaces exactly like `scripts/eval_afriscience.py` builds its documents). A paper never gets the same granularity twice. The cut is the
    same for the six languages of a paper (multi-parallel).
  * sentences are dropped when (a) the target/source length ratio is an outlier for that language (misaligned or truncated pair), or
    (b) the sentence equals a held-out sentence or shares a word 8-gram with the hold-out (below).
  * a window never crosses a paper; windows are joined by single spaces (no newline), as in the original SFT windows (D-23).

Replay part (the ORIGINAL packed SFT windows, `gemma_packed_fwd_sft_windows.tar.zst`: en->X only, 11.33 M samples, already tokenised with the
tokenizer of the model, so the token ids are copied, not re-tokenised)
  * UniMax-style per-language token budget: water-filling so that every language gets the same token budget c (languages with less data than
    c*... use all their data, at most --replay-max-epochs repetitions), total = science tokens x (1-share)/share; at the default share 0.35 the
    replay is 65 % of the tokens. A big language (Igbo: 260 M target tokens) therefore contributes ~0.1 % of its data and a small one (Kéa: 0.2 M
    tokens) most of it. At least --min-examples samples per language (the smallest language has 1,066).
  * samples are drawn at random within a language (the natural mixture of window lengths), without replacement.
  * decontamination: replay samples whose English side equals a held-out sentence or shares a word 8-gram with the hold-out are dropped.

Hold-out (never in train, in either part): AfriScience-MT dev + test (English), AfriLongDoc hard + core documents, the general dev set
(bench_dev_clean.jsonl), FLORES+ dev English, the sentences of the rl-vs-sft evaluation pool.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import unicodedata
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

SCI_LANGS = ("amh", "hau", "lug", "nso", "yor", "zul")
MODES = ("S", "A", "B", "W")
MODE_WEIGHTS = {"S": 0.35, "A": 0.25, "B": 0.20, "W": 0.20}
TAR_MEMBER = "gemma/packed_fwd/train/tokens.u32.bin"
_WS = re.compile(r"\s+")
_WORD = re.compile(r"\w+", re.U)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------------------------------------ hold-out
def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def norm_sentence(s: str) -> str:
    return _WS.sub(" ", nfc(s)).strip().lower()


def ngrams8(text: str) -> Iterator[int]:
    w = _WORD.findall(nfc(text).lower())
    for i in range(len(w) - 7):
        yield hash(" ".join(w[i:i + 8]))


class HoldOut:
    """Held-out sentences (exact after normalisation, only sentences longer than 20 characters) and word 8-gram hashes.
    `hit(text)` -> "sentence" | "ngram8" | None. The hash() values are per process (never stored)."""

    def __init__(self) -> None:
        self.sent: set = set()
        self.grams: set = set()
        self.counts: Dict[str, int] = collections.Counter()

    def add(self, source: str, texts: Iterable[str], sentences: Iterable[str]) -> None:
        for s in sentences:
            if s and len(s) > 20:
                self.sent.add(norm_sentence(s))
        for t in texts:
            if t:
                self.grams.update(ngrams8(t))
        self.counts[source] += 1

    def hit(self, text: str) -> Optional[str]:
        for s in re.split(r"(?<=[.!?])\s+", text):
            if norm_sentence(s) in self.sent:
                return "sentence"
        if self.sent and norm_sentence(text) in self.sent:
            return "sentence"
        for g in ngrams8(text):
            if g in self.grams:
                return "ngram8"
        return None


def read_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def build_holdout(a, afs_all: "Any") -> HoldOut:
    ho = HoldOut()
    # AfriScience dev + test: the English side is identical in the six lang_pairs; use eng-hau
    for split in ("dev", "test"):
        d = afs_all[split]
        e = d[d.lang_pair == "eng-hau"].sort_values(["paper_id", "sentence_id"])
        for pid, g in e.groupby("paper_id"):
            ss = [nfc(x) for x in g.source.tolist()]
            ho.add(f"afriscience_{split}_papers", [" ".join(ss)] + ss, ss)
    for rel in ("hard", "core"):
        p = os.path.join(a.longdoc_dir, rel, "docs.jsonl")
        if os.path.exists(p):
            for d in read_jsonl(p):
                ho.add(f"afrilongdoc_{rel}_docs", [d.get("text") or ""], d.get("sents") or [])
    if a.bench_dev and os.path.exists(a.bench_dev):
        rows = read_jsonl(a.bench_dev)
        en = sorted({r["english"] for r in rows})
        ho.add("general_dev_sentences", en, en)
        ho.counts["general_dev_sentences"] = len(en)
    for p in glob.glob(a.flores_dev_eng):
        en = [r["text"] for r in read_jsonl(p)]
        ho.add("flores_plus_dev_english", en, en)
        ho.counts["flores_plus_dev_english"] = len(en)
    if a.eval_pool and os.path.exists(a.eval_pool):
        en = sorted({r["english"] for r in read_jsonl(a.eval_pool)})
        ho.add("rl_vs_sft_pool", en, en)
        ho.counts["rl_vs_sft_pool"] = len(en)
    log(f"hold-out: {dict(ho.counts)}; {len(ho.sent)} sentences, {len(ho.grams)} word 8-grams")
    return ho


# ------------------------------------------------------------------------------------------------------------------------------ science
def paper_modes(pid: Any, passes: int, seed: int, weights: Dict[str, float] = MODE_WEIGHTS) -> List[str]:
    """One window granularity per pass for a paper, never the same twice while there are other granularities left."""
    rng = random.Random(f"{seed}|{pid}|modes")
    left, out = list(MODES), []
    for _ in range(passes):
        if not left:
            left = list(MODES)
        m = rng.choices(left, weights=[weights[x] for x in left])[0]
        out.append(m)
        left.remove(m)
    return out


def segment(n: int, mode: str, rng: random.Random) -> List[Tuple[int, int]]:
    """Index ranges [a, b) that cut a paper of n sentences into consecutive windows (every sentence in exactly one window)."""
    if n <= 0:
        return []
    if mode == "W":
        return [(0, n)]
    lo, hi = {"S": (1, 1), "A": (2, 4), "B": (5, 12)}[mode]
    out, i = [], 0
    while i < n:
        j = min(n, i + rng.randint(lo, hi))
        out.append((i, j))
        i = j
    return out


def length_ratio_bounds(df, lang: str) -> Tuple[float, float]:
    g = df[df.lang_pair == f"eng-{lang}"]
    r = (g.target.str.len() / g.source.str.len().clip(lower=1)).to_numpy()
    m = float(np.median(r))
    return 0.3 * m, 3.0 * m


def science_windows(train_df, ho: HoldOut, passes: int, seed: int, languages: Sequence[str] = SCI_LANGS) -> Tuple[List[dict], Dict[str, Any]]:
    """-> (windows, stats). window = {english, target, iso, paper, domain, pass, mode, n_sent}."""
    stats: Dict[str, Any] = {"dropped_ratio": collections.Counter(), "dropped_holdout_sentence": 0, "dropped_holdout_ngram8": 0,
                             "kept_sentences": collections.Counter(), "windows": collections.Counter()}
    ref = train_df[train_df.lang_pair == f"eng-{languages[0]}"].sort_values(["paper_id", "sentence_id"])
    ids = {pid: g.sentence_id.tolist() for pid, g in ref.groupby("paper_id")}
    # screen the (shared) English sentences once
    bad_en: Dict[Tuple[Any, int], str] = {}
    for r in ref.itertuples():
        h = ho.hit(nfc(r.source))
        if h:
            bad_en[(r.paper_id, int(r.sentence_id))] = h
    stats["dropped_holdout_sentence"] = sum(1 for v in bad_en.values() if v == "sentence")
    stats["dropped_holdout_ngram8"] = sum(1 for v in bad_en.values() if v == "ngram8")
    out: List[dict] = []
    for iso in languages:
        lo, hi = length_ratio_bounds(train_df, iso)
        g = train_df[train_df.lang_pair == f"eng-{iso}"].sort_values(["paper_id", "sentence_id"])
        by_paper: Dict[Any, List[Tuple[int, str, str, str]]] = {}
        for r in g.itertuples():
            by_paper.setdefault(r.paper_id, []).append((int(r.sentence_id), nfc(r.source).strip(), nfc(r.target).strip(), r.domain))
        for pid in sorted(by_paper):
            rows = by_paper[pid]
            if [x[0] for x in rows] != ids[pid]:
                raise ValueError(f"paper {pid}: sentence ids differ between eng-{languages[0]} and eng-{iso} (not multi-parallel)")
            keep = []
            for sid, src, tgt, dom in rows:
                if (pid, sid) in bad_en:
                    keep.append(False)
                    continue
                ok = bool(src) and bool(tgt) and lo <= len(tgt) / max(len(src), 1) <= hi
                if not ok:
                    stats["dropped_ratio"][iso] += 1
                keep.append(ok)
            modes = paper_modes(pid, passes, seed)
            for p in range(passes):
                rng = random.Random(f"{seed}|{pid}|{p}|cut")           # same cut for the six languages of a paper
                for a, b in segment(len(rows), modes[p], rng):
                    sel = [rows[k] for k in range(a, b) if keep[k]]
                    if not sel:
                        continue
                    out.append({"english": " ".join(x[1] for x in sel), "target": " ".join(x[2] for x in sel), "iso": iso, "paper": int(pid),
                                "domain": sel[0][3], "pass": p, "mode": modes[p], "n_sent": len(sel)})
                    stats["windows"][f"{iso}/{modes[p]}"] += 1
            stats["kept_sentences"][iso] += sum(keep)
    stats = {k: (dict(v) if isinstance(v, collections.Counter) else v) for k, v in stats.items()}
    return out, stats


def science_dev(dev_df, languages: Sequence[str] = SCI_LANGS) -> List[dict]:
    """The AfriScience-MT DEV split in the two evaluation settings: every sentence on its own + every whole paper (sentences joined by single spaces)."""
    out = []
    for iso in languages:
        g = dev_df[dev_df.lang_pair == f"eng-{iso}"].sort_values(["paper_id", "sentence_id"])
        for r in g.itertuples():
            out.append({"english": nfc(r.source).strip(), "target": nfc(r.target).strip(), "iso": iso, "paper": int(r.paper_id), "mode": "S", "n_sent": 1})
        for pid, gg in g.groupby("paper_id", sort=True):
            out.append({"english": " ".join(nfc(x).strip() for x in gg.source), "target": " ".join(nfc(x).strip() for x in gg.target), "iso": iso,
                        "paper": int(pid), "mode": "W", "n_sent": len(gg)})
    return out


# ------------------------------------------------------------------------------------------------------------------------------ replay: budget
def unimax_budgets(available: Sequence[float], total: float, max_epochs: float = 1.0, weights: Optional[Sequence[float]] = None) -> np.ndarray:
    """UniMax water-filling. b_l = min(w_l * c, max_epochs * available_l) with the level c chosen so that sum(b) = total (or every language is at
    its cap). With equal weights and max_epochs = 1 a language smaller than c contributes ALL its data, bigger ones exactly c tokens."""
    n = np.asarray(available, dtype=np.float64)
    w = np.ones_like(n) if weights is None else np.asarray(weights, dtype=np.float64)
    cap = max_epochs * n
    if total >= cap.sum():
        return cap
    lo, hi = 0.0, float(cap.max() / w.min()) + 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if np.minimum(w * mid, cap).sum() > total:
            hi = mid
        else:
            lo = mid
    return np.minimum(w * lo, cap)


def plan_replay(iso_id: np.ndarray, sample_len: np.ndarray, n_langs: int, budgets: np.ndarray, oversample: float, min_examples: int,
                seed: int) -> Dict[int, np.ndarray]:
    """Per language: a seeded permutation of its samples, cut after `budget * oversample` tokens (and at least `min_examples * oversample` samples);
    the candidates beyond the budget replace the ones the hold-out filter will reject."""
    order = np.argsort(iso_id, kind="stable")
    bounds = np.searchsorted(iso_id[order], np.arange(n_langs + 1))
    plan: Dict[int, np.ndarray] = {}
    for l in range(n_langs):
        idx = order[bounds[l]:bounds[l + 1]]
        if len(idx) == 0:
            continue
        perm = idx[np.random.default_rng([seed, l]).permutation(len(idx))]
        cum = np.cumsum(sample_len[perm].astype(np.int64))
        k = int(np.searchsorted(cum, budgets[l] * oversample)) + 1
        k = max(k, int(math.ceil(min_examples * oversample)))
        plan[l] = perm[:min(k, len(perm))]               # permutation order (the streaming pass sorts them by storage offset)
    return plan


# ------------------------------------------------------------------------------------------------------------------------------ replay: streaming extraction
def select_from_stream(stream, offsets: Sequence[int], lengths: Sequence[int], chunk: int = 64 << 20) -> List[bytes]:
    """`offsets` / `lengths` in BYTES, sorted ascending by offset, non-overlapping. Reads `stream` (a binary file object positioned at byte 0 of the
    file) once, front to back, and returns the requested byte ranges. Memory: one chunk + the longest range."""
    n = len(offsets)
    out: List[Optional[bytes]] = [None] * n
    buf, base, k = bytearray(), 0, 0
    while k < n:
        data = stream.read(chunk)
        if not data:
            break
        buf += data
        end = base + len(buf)
        while k < n and offsets[k] + lengths[k] <= end:
            a = offsets[k] - base
            out[k] = bytes(buf[a:a + lengths[k]])
            k += 1
        if k < n:
            cut = min(len(buf), max(0, offsets[k] - base))
            if cut:
                del buf[:cut]
                base += cut
    if k < n:
        raise EOFError(f"stream ended after {k} of {n} ranges")
    return out  # type: ignore[return-value]


def extract_tokens(tar_zst: str, sample_offset: np.ndarray, sample_len: np.ndarray, idx_sorted: np.ndarray) -> List[np.ndarray]:
    cmd = f"nice -n 10 zstd -dc {tar_zst} | tar -xOf - --occurrence=1 {TAR_MEMBER}"
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
    try:
        offs = (sample_offset[idx_sorted].astype(np.int64) * 4).tolist()
        lens = (sample_len[idx_sorted].astype(np.int64) * 4).tolist()
        raw = select_from_stream(p.stdout, offs, lens)
    finally:
        p.kill()
        p.wait()
    return [np.frombuffer(b, dtype=np.uint32) for b in raw]


_PROMPT_RE = re.compile(r"^<bos><start_of_turn>user\nTranslate the text to [^:\n]*:  (.*)<end_of_turn>\n<start_of_turn>model\n$", re.S)


def english_of_prompt(tok, ids: np.ndarray, prompt_len: int) -> Optional[str]:
    m = _PROMPT_RE.match(tok.decode(ids[:prompt_len].tolist(), skip_special_tokens=False))
    return m.group(1) if m else None


# ------------------------------------------------------------------------------------------------------------------------------ assemble
def _ex(ids: np.ndarray, prompt_len: int, target_len: int) -> SimpleNamespace:
    return SimpleNamespace(input_ids=np.asarray(ids, dtype=np.uint32), prompt_len=int(prompt_len), target_len=int(target_len))


def write_split(items: List[Tuple[Any, int]], out_dir: str, seq_len: int, seed: int, split: str = "train") -> Dict[str, Any]:
    from gemma_mt.data.prepare_sft import PrepareConfig, _SampleStore, _write_split

    tmp = os.path.join(out_dir, "_tmp_store")
    store = _SampleStore(os.path.join(tmp, split))
    for ex, iso_id in items:
        store.add(ex, iso_id, 0)
    cfg = PrepareConfig(input="-", out_dir=out_dir, seq_len=seq_len, buffer_size=max(100_000, len(items) + 1), seed=seed)
    meta = _write_split(store, os.path.join(out_dir, split), cfg, "train" if split == "train" else "val")
    import shutil

    shutil.rmtree(tmp, ignore_errors=True)
    return meta


def per_language_table(items: List[Tuple[Any, int, str]], isos: Sequence[str], id_table: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, int]]:
    """examples / tokens / target tokens per language; `iso_id` indexes `id_table` (default: gemma_mt.languages.ALL_ISOS), `isos` are the rows of the table."""
    if id_table is None:
        from gemma_mt.languages import ALL_ISOS as id_table
    t = {iso: {"examples": 0, "tokens": 0, "target_tokens": 0} for iso in isos}
    for ex, iso_id, src in items:
        d = t[id_table[iso_id]]
        d["examples"] += 1
        d["tokens"] += len(ex.input_ids)
        d["target_tokens"] += ex.target_len
    return t


def markdown_table(isos: Sequence[str], sci: Dict, rep: Dict, val: Dict, valsci: Dict, names: Dict[str, str]) -> str:
    tot = lambda t, k: sum(v[k] for v in t.values())
    L = ["| tag | language | science ex | science tokens | science target tok | replay ex | replay tokens | replay target tok | science share of tokens | val ex (replay) | val ex (sci dev) |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for iso in isos:
        s, r = sci[iso], rep[iso]
        share = s["tokens"] / max(s["tokens"] + r["tokens"], 1)
        L.append(f"| {iso} | {names.get(iso, iso)} | {s['examples']:,} | {s['tokens']:,} | {s['target_tokens']:,} | {r['examples']:,} | {r['tokens']:,} | {r['target_tokens']:,} | "
                 f"{share * 100:.1f}% | {val[iso]['examples']:,} | {valsci[iso]['examples']:,} |")
    L.append(f"| **all** | | {tot(sci, 'examples'):,} | {tot(sci, 'tokens'):,} | {tot(sci, 'target_tokens'):,} | {tot(rep, 'examples'):,} | {tot(rep, 'tokens'):,} | "
             f"{tot(rep, 'target_tokens'):,} | {tot(sci, 'tokens') / max(tot(sci, 'tokens') + tot(rep, 'tokens'), 1) * 100:.1f}% | {tot(val, 'examples'):,} | {tot(valsci, 'examples'):,} |")
    return "\n".join(L)


# ------------------------------------------------------------------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="/workspace/sft_build/runs/final_1433", help="its tokenizer is used (fingerprint-checked against the replay data)")
    ap.add_argument("--corpus-dir", default="/workspace/jobs/a4/afs_raw/corpus")
    ap.add_argument("--longdoc-dir", default="/workspace/jobs/a4/holdout/afrilongdoc")
    ap.add_argument("--bench-dev", default="/workspace/sft_build/bench/bench_dev_clean.jsonl")
    ap.add_argument("--flores-dev-eng", default="/workspace/sft_build/hf/hub/datasets--openlanguagedata--flores_plus/snapshots/*/dev/eng_Latn.jsonl")
    ap.add_argument("--eval-pool", default="/workspace/rlcheck/res/pool.jsonl")
    ap.add_argument("--replay-tar", default="/workspace/jobs/r4/raw/gemma_packed_fwd_sft_windows.tar.zst")
    ap.add_argument("--replay-meta", default="/workspace/jobs/r4/raw/meta/gemma/packed_fwd", help="dir with the tarball's train/ + val/ npy files and meta.json")
    ap.add_argument("--seq-len", type=int, default=8192, help="pack length (the original SFT used 4,096; a whole Yoruba paper needs up to ~8,000 tokens)")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--science-share", type=float, default=0.35, help="share of the TRAIN tokens that is science")
    ap.add_argument("--replay-max-epochs", type=float, default=1.0)
    ap.add_argument("--min-examples", type=int, default=600, help="every language gets at least this many replay examples (when it has them)")
    ap.add_argument("--sci-replay-weight", type=float, default=2.0, help="the six science languages get this multiple of the replay token budget of an ordinary language (they receive ~8x more science updates, so their general ability is the most at risk)")
    ap.add_argument("--oversample", type=float, default=1.3, help="candidate pool / budget (the hold-out filter drops some)")
    ap.add_argument("--val-per-language", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--limit-replay-tokens", type=float, default=0.0, help="testing: override the replay token total")
    a = ap.parse_args(argv)

    import pandas as pd
    from transformers import AutoTokenizer

    from gemma_mt import format as fmt
    from gemma_mt.data.prepare_sft import tokenizer_fingerprint
    from gemma_mt.languages import ALL_ISOS, LANGUAGES, iso_to_id

    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    fmt.assert_template_matches_reference(tok)
    fp = tokenizer_fingerprint(tok)
    rmeta = json.load(open(os.path.join(a.replay_meta, "meta.json")))
    if rmeta["tokenizer"]["fingerprint"] != fp:
        raise SystemExit("the replay data was tokenised with a different tokenizer than the model's")
    if rmeta["iso_table"] != list(ALL_ISOS):
        raise SystemExit("the language table of the replay data differs from gemma_mt.languages.ALL_ISOS")
    isos = [i for i in ALL_ISOS if i in rmeta["splits"]["train"]["per_language"]]
    log(f"{len(isos)} languages in the replay data")

    afs = {sp: pd.read_parquet(os.path.join(a.corpus_dir, f"{sp}-00000.parquet")) for sp in ("train", "dev", "test")}
    ho = build_holdout(a, afs)

    # ---------------------------------------------------------------- science (train + dev)
    wins, sstats = science_windows(afs["train"], ho, a.passes, a.seed)
    sci_items: List[Tuple[Any, int, str]] = []
    n_long = 0
    for w in wins:
        ex = fmt.build_example(w["english"], w["iso"], w["target"], tok)
        if ex.n_tokens > a.seq_len:
            n_long += 1
            continue
        sci_items.append((_ex(ex.input_ids, ex.prompt_len, ex.target_len), iso_to_id(w["iso"]), f"{w['mode']}"))
    sci_tab = per_language_table(sci_items, isos)
    sci_tokens = sum(v["tokens"] for v in sci_tab.values())
    log(f"science: {len(wins)} windows ({n_long} longer than {a.seq_len} tokens dropped), {sci_tokens:,} tokens over {a.passes} passes; "
        f"sentences dropped: ratio {sstats['dropped_ratio']}, hold-out sentence {sstats['dropped_holdout_sentence']}, hold-out 8-gram {sstats['dropped_holdout_ngram8']}")
    dev_items: List[Tuple[Any, int, str]] = []
    for w in science_dev(afs["dev"]):
        ex = fmt.build_example(w["english"], w["iso"], w["target"], tok)
        if ex.n_tokens <= a.seq_len:
            dev_items.append((_ex(ex.input_ids, ex.prompt_len, ex.target_len), iso_to_id(w["iso"]), w["mode"]))
    log(f"science dev (val_sci): {len(dev_items)} samples")

    # ---------------------------------------------------------------- replay: plan
    d = os.path.join(a.replay_meta, "train")
    iso_id = np.load(os.path.join(d, "iso_id.npy"))
    s_len = np.load(os.path.join(d, "sample_len.npy"))
    s_off = np.load(os.path.join(d, "sample_offset.npy"))
    p_len = np.load(os.path.join(d, "prompt_len.npy"))
    t_len = np.load(os.path.join(d, "target_len.npy"))
    n_langs = len(ALL_ISOS)
    avail = np.bincount(iso_id, weights=s_len.astype(np.float64), minlength=n_langs)
    lang_ids = [iso_to_id(i) for i in isos]
    replay_total = a.limit_replay_tokens or sci_tokens * (1.0 - a.science_share) / a.science_share
    bud = np.zeros(n_langs)
    wts = np.array([a.sci_replay_weight if i in SCI_LANGS else 1.0 for i in isos])
    bud[lang_ids] = unimax_budgets(avail[lang_ids], replay_total, a.replay_max_epochs, weights=wts)
    log(f"replay budget {replay_total:,.0f} tokens; per-language budget min {bud[lang_ids].min():,.0f} max {bud[lang_ids].max():,.0f} (science languages x{a.sci_replay_weight})")
    plan = plan_replay(iso_id, s_len, n_langs, bud, a.oversample, a.min_examples, a.seed)
    cand = np.sort(np.concatenate([v for v in plan.values()]))
    log(f"replay candidates: {len(cand):,} samples ({s_len[cand].astype(np.int64).sum():,} tokens); streaming the tarball ...")
    import hashlib

    cache = os.path.join(a.out, f"replay_pool_{hashlib.md5(cand.tobytes()).hexdigest()[:12]}.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        pool = {int(i): t for i, t in zip(z["idx"], z["toks"])}
    else:
        toks = extract_tokens(a.replay_tar, s_off, s_len, cand)
        pool = dict(zip(cand.tolist(), toks))
        arr = np.empty(len(toks), dtype=object)
        for j, t in enumerate(toks):
            arr[j] = t
        np.savez(cache, idx=cand, toks=arr)
    log(f"extracted {len(pool):,} samples in {time.time() - t0:.0f}s")

    # ---------------------------------------------------------------- replay: hold-out filter + trim to the budget
    rep_items: List[Tuple[Any, int, str]] = []
    rstat = {"hit_sentence": 0, "hit_ngram8": 0, "undecodable": 0, "short_of_budget": []}
    for l in lang_ids:
        got_tok, got_n = 0, 0
        for i in plan.get(l, []):
            if got_tok >= bud[l] and got_n >= a.min_examples:
                break
            ids = pool[int(i)]
            en = english_of_prompt(tok, ids, int(p_len[i]))
            if en is None:
                rstat["undecodable"] += 1
                continue
            h = ho.hit(en)
            if h:
                rstat["hit_" + h] += 1
                continue
            rep_items.append((_ex(ids, p_len[i], t_len[i]), l, ""))
            got_tok += len(ids)
            got_n += 1
        if got_tok < 0.98 * bud[l] and got_tok < avail[l] * a.replay_max_epochs * 0.98:
            rstat["short_of_budget"].append(ALL_ISOS[l])
    rep_tab = per_language_table(rep_items, isos)
    log(f"replay: {len(rep_items):,} samples, {sum(v['tokens'] for v in rep_tab.values()):,} tokens; rejected by the hold-out {rstat['hit_sentence']} (sentence) + {rstat['hit_ngram8']} (8-gram), "
        f"undecodable {rstat['undecodable']}; short of budget: {rstat['short_of_budget']}")

    # ---------------------------------------------------------------- replay validation (original SFT val split)
    vd = os.path.join(a.replay_meta, "val")
    v_iso, v_len, v_off = (np.load(os.path.join(vd, n)) for n in ("iso_id.npy", "sample_len.npy", "sample_offset.npy"))
    v_p, v_t = np.load(os.path.join(vd, "prompt_len.npy")), np.load(os.path.join(vd, "target_len.npy"))
    vtoks = np.memmap(os.path.join(vd, "tokens.u32.bin"), dtype=np.uint32, mode="r")
    val_items: List[Tuple[Any, int, str]] = []
    for l in lang_ids:
        idx = np.flatnonzero(v_iso == l)
        idx = idx[np.random.default_rng([a.seed, 99, l]).permutation(len(idx))]
        n = 0
        for i in idx:
            if n >= a.val_per_language:
                break
            ids = np.asarray(vtoks[int(v_off[i]): int(v_off[i]) + int(v_len[i])])
            en = english_of_prompt(tok, ids, int(v_p[i]))
            if en is None or ho.hit(en):
                continue
            val_items.append((_ex(ids, v_p[i], v_t[i]), l, ""))
            n += 1
    val_tab = per_language_table(val_items, isos)
    valsci_tab = per_language_table(dev_items, isos)
    log(f"val (replay): {len(val_items)} samples; val_sci: {len(dev_items)}")

    # ---------------------------------------------------------------- write
    train_items = [(e, l) for e, l, _ in sci_items] + [(e, l) for e, l, _ in rep_items]
    meta_splits = {"train": write_split(train_items, a.out, a.seq_len, a.seed, "train"),
                   "val": write_split([(e, l) for e, l, _ in val_items], a.out, a.seq_len, a.seed, "val")}
    # the third split directory has another name: write it as "val" into a scratch dir and move
    import shutil

    tmp_sci = os.path.join(a.out, "_val_sci_tmp")
    meta_splits["val_sci"] = write_split([(e, l) for e, l, _ in dev_items], tmp_sci, a.seq_len, a.seed + 1, "val")
    shutil.rmtree(os.path.join(a.out, "val_sci"), ignore_errors=True)
    shutil.move(os.path.join(tmp_sci, "val"), os.path.join(a.out, "val_sci"))
    shutil.rmtree(tmp_sci, ignore_errors=True)

    meta = {
        "format_version": 1, "dtype": "uint32", "seq_len": a.seq_len, "instruction_template": fmt.INSTRUCTION_TEMPLATE, "end_of_turn": fmt.END_OF_TURN,
        "ignore_index": fmt.IGNORE_INDEX,
        "tokenizer": {"name": getattr(tok, "name_or_path", a.model), "requested": a.model, "revision": None, "fingerprint": fp, "vocab_size": len(tok),
                      "bos_id": tok.bos_token_id, "eos_id": tok.eos_token_id, "pad_id": tok.pad_token_id,
                      "end_of_turn_id": tok.convert_tokens_to_ids(fmt.END_OF_TURN)},
        "iso_table": list(ALL_ISOS), "directions": ["en2x"], "pivot_tag": fmt.PIVOT_ISO,
        "splits": meta_splits,
        "mixture": {"science_share_target": a.science_share, "passes": a.passes, "mode_weights": MODE_WEIGHTS, "replay_tokens_budget": replay_total,
                    "replay_max_epochs": a.replay_max_epochs, "min_examples": a.min_examples, "science_stats": sstats, "replay_stats": rstat,
                    "holdout": dict(ho.counts), "science_windows_too_long": n_long, "seed": a.seed},
        "config": vars(a), "seconds": round(time.time() - t0, 1),
    }
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1, ensure_ascii=False)
    names = {i: LANGUAGES[i].name for i in isos}
    tables = {"train_science": sci_tab, "train_replay": rep_tab, "val_replay": val_tab, "val_science_dev": valsci_tab}
    json.dump(tables, open(os.path.join(a.out, "mixture_tables.json"), "w"), indent=1)
    open(os.path.join(a.out, "mixture_tables.md"), "w").write(markdown_table(isos, sci_tab, rep_tab, val_tab, valsci_tab, names) + "\n")
    tr = meta_splits["train"]
    log(f"TRAIN: {tr['n_samples']:,} samples, {tr['n_packs']:,} packs of {a.seq_len}, {tr['n_tokens']:,} tokens ({tr['target_tokens']:,} target), "
        f"packing efficiency {tr['packing_efficiency']:.4f}; science share of tokens {sum(v['tokens'] for v in sci_tab.values()) / tr['n_tokens']:.3f}; done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
