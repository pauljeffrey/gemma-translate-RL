#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""News-style English RL prompt pool (2026-11-03): BBC News windows, decontaminated, one parquet per RL language.

Stages (each resumable, files in --work):
  extract   BBC articles -> cleaned paragraphs -> sentences -> windows on sentence boundaries, one window per article and length bin (b64 b256 b512 b1024, bins by the number of
            Davlan/afro-xlmr-large-76L tokens INCLUDING <s></s> = the `src_afr_tokens` convention of the other pools)  -> <work>/pool_raw.parquet
  decon     every window against the protected index (data_pipeline/sft_clean/decon.py rules R1 exact sentence, R2 contained sentence, R3 shared word 8-gram, strict, no dev-half exemption)
            -> <work>/pool_clean.parquet + <work>/decon_stats.json
  draw      per RL language (gemma_mt.languages.SUPPORTED_RL minus arz) a seeded draw without replacement from each bin of the clean pool -> <out>/<iso>.parquet

    python scripts/build_rl_prompts_news.py extract --src data/src --work work
    python scripts/build_rl_prompts_news.py decon --work work --index protected_index.npz --manifest protected_manifest.json --nproc 4
    python scripts/build_rl_prompts_news.py draw --work work --out artifacts/rl_prompts_news
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import sys
import time
import zlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import xxhash

BINS = ("b64", "b256", "b512", "b1024")
BIN_RANGE = {"b64": (24, 64), "b256": (65, 256), "b512": (257, 512), "b1024": (513, 1024)}   # afro-xlmr tokens with <s> </s>
SRC_LICENCE = {
    "RealTimeData/bbc_news_alltime": "bbc-copyright-no-hub-licence",
    "permutans/c4-bbc-news": "odc-by",
    "SetFit/bbc-news": "bbc-copyright-noncommercial-research",
}
COLS = ("english", "iso", "language", "src_afr_tokens", "length_bin", "n_sentences", "source_dataset", "licence", "domain",
        "prov_pool", "prov_file", "prov_row", "prov_en_xxh64", "prov_index", "flag_ssa_mte_train")
POOL_VERSION = "news_v1_2026-11-03"
ABBR = {"mr", "mrs", "ms", "dr", "st", "prof", "jr", "sr", "lt", "col", "gen", "sgt", "capt", "cpl", "maj", "cmdr", "adm", "vs", "inc", "ltd", "co", "corp", "mt", "gov", "rep",
        "sen", "rev", "hon", "u.s", "u.k", "u.n", "e.g", "i.e", "a.m", "p.m", "approx", "dept", "est", "fig", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
        "ft", "oz", "lb", "bros", "ave", "blvd", "mph", "pres", "supt", "insp", "det", "cllr", "ph.d", "d.c", "n.y", "l.a"}

_WS = re.compile(r"[ \t    ]+")
_END = re.compile(r"""[.!?]["'”’)\]]*$""")
_BOUND = re.compile(r"""(?<=[.!?])(["'”’)\]]*)\s+(?=["'“‘(\[]?[A-Z0-9])""")
_BAD_PARA = re.compile(
    r"(?i)(^|\b)(image caption|media caption|image source|image copyright|getty images|related topics|follow us|follow bbc|get in touch|sign up|subscribe|bbc sounds|bbc iwonder|"
    r"read more|watch:|listen:|share this|all rights reserved|copyright|click here|whatsapp|bbc news app|download the|email us|send your|you can contact|reporting by|additional reporting|"
    r"\bsource:|\bpicture:|\bphoto:|\bvideo:|\bgraphic:|this article is about|find out more|more on this story|top stories|bbc is not responsible|external sites?|will open in a new)")
_BAD_SENT = re.compile(r"(https?://|www\.|\.com\b|\.co\.uk\b|@\w|#\w|\bbit\.ly\b|\bclick\b|\bsubscribe\b)", re.I)
_OK_CHARS = re.compile(r"[A-Za-z0-9 \t.,;:!?'\"()\[\]\-–—‘’“”…%$£€&/+*=<>éèáóíúñüöäçêâîôûÉ°]")


def now():
    return time.strftime("%H:%M:%S")


def h64(s: str) -> int:
    return xxhash.xxh64_intdigest(s.encode("utf-8", "surrogatepass"))


def clean_paragraphs(text: str):
    text = (text or "").replace("\r", "\n").replace("​", "").replace("﻿", "")
    out = []
    for line in re.split(r"\n+", text):
        p = _WS.sub(" ", line).strip()
        if len(p.split()) < 6 or not _END.search(p) or _BAD_PARA.search(p) or p.isupper():
            continue
        out.append(p)
    return out


def split_sentences(par: str):
    cuts, sents, start = [], [], 0
    for m in _BOUND.finditer(par):
        end = m.start() + len(m.group(1))                      # sentence ends after the closing quote/bracket
        cand = par[start:end]
        last = cand.rstrip(".!?\"'”’)]").split(" ")[-1].lower().strip("([\"'“‘")
        if cand.rstrip().endswith(".") or cand.rstrip("\"'”’)]").endswith("."):
            if last in ABBR or (last == "no" and par[m.end():m.end() + 1].isdigit()) or (len(last) == 1 and last.isalpha()) or re.fullmatch(r"(?:[a-z]\.)+[a-z]?", last):
                continue
        sents.append(cand.strip())
        start = m.end()
    tail = par[start:].strip()
    if tail:
        sents.append(tail)
    return sents


def good_sentence(s: str) -> bool:
    w = len(s.split())
    if w < 3 or w > 70 or _BAD_SENT.search(s):
        return False
    bad = len(s) - len(_OK_CHARS.findall(s))
    return bad <= 0.02 * len(s)


def iter_articles(src: str):
    """yield (source_dataset, prov_file, prov_row, text)"""
    d = os.path.join(src, "RealTimeData__bbc_news_alltime")
    for f in sorted(glob.glob(os.path.join(d, "*", "*.parquet"))):
        t = pq.read_table(f, columns=["content"]).column("content").to_pylist()
        rel = os.path.relpath(f, d)
        for i, c in enumerate(t):
            yield "RealTimeData/bbc_news_alltime", rel, i, c
    d = os.path.join(src, "permutans__c4-bbc-news")
    for f in sorted(glob.glob(os.path.join(d, "*", "*.parquet"))):
        pf = pq.ParquetFile(f)
        rel = os.path.relpath(f, d)
        row = 0
        for b in pf.iter_batches(batch_size=5000, columns=["text"]):
            for c in b.column("text").to_pylist():
                yield "permutans/c4-bbc-news", rel, row, c
                row += 1
    f = os.path.join(src, "SetFit__bbc-news", "bbc-text.csv")
    if os.path.exists(f):
        import csv
        csv.field_size_limit(10 ** 9)
        for i, r in enumerate(csv.DictReader(open(f, encoding="utf-8"))):
            yield "SetFit/bbc-news", "bbc-text.csv", i, r["text"]


def make_windows(sents, cum, rng, seg_starts):
    """sents: list of sentence ids in one clean segment (consecutive sentences); cum: cumulative token counts (len n+1). -> {bin: (i, j)} half-open range or None."""
    n = len(sents)
    res = {}
    for b in BINS:
        lo, hi = BIN_RANGE[b]
        T = int(rng.integers(lo, hi + 1))
        # candidate starts: paragraph starts are preferred (60 %)
        cands = None
        if rng.random() < 0.6 and seg_starts:
            cands = list(seg_starts)
        if not cands:
            cands = list(range(n))
        rng.shuffle(cands)
        for i in cands[:6]:
            j = i
            while j < n and cum[j + 1] - cum[i] <= hi and (cum[j] - cum[i] < T):
                j += 1
            tok = cum[j] - cum[i]
            if j > i and lo <= tok <= hi:
                res[b] = (i, j)
                break
    return res


def cmd_extract(a):
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("Davlan/afro-xlmr-large-76L").backend_tokenizer
    tk.no_truncation()
    os.makedirs(a.work, exist_ok=True)
    stats = collections.Counter()
    t0 = time.time()
    # ---- pass 1: paragraph frequencies (boilerplate that repeats in many articles is dropped)
    pc = collections.Counter()
    for _s, _f, _r, text in iter_articles(a.src):
        for p in set(clean_paragraphs(text)):
            pc[h64(p)] += 1
    boiler = {h for h, c in pc.items() if c >= a.boiler_min}
    print(now(), "paragraph pass done", len(pc), "distinct paragraphs,", len(boiler), "repeated >=", a.boiler_min, "articles (dropped)", flush=True)
    del pc
    # ---- pass 2: sentences + windows
    seen_sent = set()
    rows = []
    batch = []

    def flush(batch):
        """tokenise the sentences of a batch of articles, build windows"""
        allsents = [s for art in batch for seg in art["segs"] for s in seg]
        enc = tk.encode_batch(allsents, add_special_tokens=False)
        ntok = [len(e.ids) for e in enc]
        k = 0
        for art in batch:
            rng = np.random.default_rng([a.seed, h64(f"{art['src']}|{art['file']}|{art['row']}") & 0xFFFFFFFF])
            got = {}
            for seg, starts in zip(art["segs"], art["starts"]):
                toks = ntok[k:k + len(seg)]
                k += len(seg)
                if len(seg) < 1:
                    continue
                cum = [0]
                for x in toks:
                    cum.append(cum[-1] + x)
                w = make_windows(seg, cum, rng, starts)
                for b, (i, j) in w.items():
                    got.setdefault(b, []).append((len(seg), seg[i:j]))
            for b, lst in got.items():
                # pick one window of the bin per article, among the segments (the longest segments first is not wanted: uniform)
                _n, ws = lst[int(rng.integers(len(lst)))]
                rows.append((" ".join(ws), len(ws), art["src"], art["file"], art["row"]))
    n_art = 0
    for src, f, r, text in iter_articles(a.src):
        n_art += 1
        stats[f"articles_in:{src}"] += 1
        if a.limit and n_art > a.limit:
            break
        segs, starts = [], []
        cur, cur_starts, n_good, n_dup, n_all = [], [], 0, 0, 0
        for p in clean_paragraphs(text):
            if h64(p) in boiler:
                stats["paragraphs_boilerplate"] += 1
                if cur:
                    segs.append(cur); starts.append(cur_starts); cur, cur_starts = [], []
                continue
            first = True
            for s in split_sentences(p):
                n_all += 1
                if not good_sentence(s):
                    stats["sentences_bad"] += 1
                    if cur:
                        segs.append(cur); starts.append(cur_starts); cur, cur_starts = [], []
                    first = False
                    continue
                n_good += 1
                n_dup += h64(s) in seen_sent
                if first:
                    cur_starts.append(len(cur))
                    first = False
                cur.append(s)
        if cur:
            segs.append(cur); starts.append(cur_starts)
        if n_good == 0 or n_dup / max(1, n_good) > 0.5:
            stats[f"articles_dropped_duplicate_or_empty:{src}"] += 1
            continue
        for seg in segs:
            for s in seg:
                seen_sent.add(h64(s))
        stats[f"articles_used:{src}"] += 1
        batch.append(dict(src=src, file=f, row=r, segs=segs, starts=starts))
        if len(batch) >= 2000:
            flush(batch); batch = []
            if n_art % 20000 < 2000:
                print(now(), n_art, "articles", len(rows), "windows", f"{time.time() - t0:.0f}s", flush=True)
    if batch:
        flush(batch)
    print(now(), "windows built", len(rows), flush=True)
    # ---- exact token counts, bins, de-duplication of identical windows
    eng = [r[0] for r in rows]
    ntok = []
    for i in range(0, len(eng), 20000):
        ntok += [len(e.ids) + 2 for e in tk.encode_batch(eng[i:i + 20000], add_special_tokens=False)]
    seen, out = set(), []
    for r, n in zip(rows, ntok):
        if n > 1024 or n < 24:
            stats["windows_outside_range"] += 1
            continue
        key = r[0].casefold()
        if key in seen:
            stats["windows_duplicate"] += 1
            continue
        seen.add(key)
        b = "b64" if n <= 64 else "b256" if n <= 256 else "b512" if n <= 512 else "b1024"
        out.append((r[0], n, b, r[1], r[2], r[3], r[4]))
    t = pa.table({
        "english": [o[0] for o in out], "src_afr_tokens": pa.array([o[1] for o in out], pa.int64()), "length_bin": [o[2] for o in out],
        "n_sentences": pa.array([o[3] for o in out], pa.int64()), "source_dataset": [o[4] for o in out], "licence": [SRC_LICENCE[o[4]] for o in out],
        "prov_file": [o[5] for o in out], "prov_row": pa.array([o[6] for o in out], pa.int64()), "prov_en_xxh64": [format(h64(o[0]), "016x") for o in out]})
    pq.write_table(t, os.path.join(a.work, "pool_raw.parquet"), compression="zstd")
    cnt = collections.Counter(zip(t.column("length_bin").to_pylist(), t.column("source_dataset").to_pylist()))
    stats = dict(stats)
    stats["windows_raw"] = len(out)
    stats["windows_raw_by_bin_source"] = {f"{b}|{s}": c for (b, s), c in sorted(cnt.items())}
    json.dump(stats, open(os.path.join(a.work, "extract_stats.json"), "w"), indent=1)
    print(now(), json.dumps(stats, indent=1), flush=True)


def cmd_decon(a):
    here = os.path.dirname(os.path.abspath(__file__))
    sc = os.path.join(here, "..", "data_pipeline", "sft_clean")
    sys.path.insert(0, os.path.abspath(sc))
    import decon as D
    import s3_common as C
    ix, ex, man = C.load_strict(a.index, a.manifest)
    print(now(), "index", man["version"], man["index_file_sha256"][:16], flush=True)
    t = pq.read_table(os.path.join(a.work, "pool_raw.parquet"))
    eng = t.column("english").to_pylist()
    n = len(eng)
    chk = C.Checker(ix, None, None, nproc=a.nproc)
    keep = np.ones(n, bool)
    by_set, by_rule, by_bin_rm = collections.Counter(), collections.Counter(), collections.Counter()
    bins = t.column("length_bin").to_pylist()
    srcs = t.column("source_dataset").to_pylist()
    by_src_rm = collections.Counter()
    t0 = time.time()
    step = 20000
    for i in range(0, n, step):
        res = chk.run(eng[i:i + step])
        for j, r in enumerate(res):
            m = r[0] | r[1] | r[2]
            if m:
                keep[i + j] = False
                by_bin_rm[bins[i + j]] += 1
                by_src_rm[srcs[i + j]] += 1
                for nm in ix.names(m):
                    by_set[nm] += 1
                for rule, mm in zip(C.RULES, r[:3]):
                    if mm:
                        by_rule[rule] += 1
        print(now(), f"{min(n, i + step)}/{n} removed so far {int((~keep).sum())} {time.time() - t0:.0f}s", flush=True)
    chk.close()
    kt = t.filter(pa.array(keep))
    kt = kt.append_column("prov_index", pa.array([man["version"]] * kt.num_rows, pa.string()))
    pq.write_table(kt, os.path.join(a.work, "pool_clean.parquet"), compression="zstd")
    before = collections.Counter(bins)
    after = collections.Counter(kt.column("length_bin").to_pylist())
    st = dict(index_version=man["version"], index_sha256=man["index_file_sha256"], rule="R1 exact sentence, R2 contained sentence, R3 shared word 8-gram (strict, both normal forms), English side only, no FLORES+ dev train-half exemption",
              windows_in=n, windows_removed=int((~keep).sum()), windows_kept=int(keep.sum()),
              before_by_bin=dict(before), after_by_bin=dict(after), removed_by_bin=dict(by_bin_rm), removed_by_source=dict(by_src_rm),
              removed_by_set_any_rule=dict(by_set.most_common()), removed_by_rule=dict(by_rule))
    json.dump(st, open(os.path.join(a.work, "decon_stats.json"), "w"), indent=1)
    print(now(), json.dumps(st, indent=1), flush=True)


def cmd_draw(a):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
    from gemma_mt.languages import LANGUAGES, SUPPORTED_RL
    isos = sorted(i for i in SUPPORTED_RL if i not in set(a.exclude))
    print(len(isos), "languages", flush=True)
    t = pq.read_table(os.path.join(a.work, "pool_clean.parquet"))
    lb = np.asarray(t.column("length_bin").to_pylist())
    per_bin = dict(zip(BINS, a.per_bin))
    idx = {b: np.nonzero(lb == b)[0] for b in BINS}
    for b in BINS:
        assert len(idx[b]) >= per_bin[b], f"pool has {len(idx[b])} rows in {b}, need {per_bin[b]}"
    os.makedirs(a.out, exist_ok=True)
    n = t.num_rows
    base_cols = {c: t.column(c) for c in ("english", "src_afr_tokens", "length_bin", "n_sentences", "source_dataset", "licence", "prov_file", "prov_row", "prov_en_xxh64", "prov_index")}
    use = collections.Counter()
    for iso in isos:
        rng = np.random.default_rng([a.seed, zlib.crc32(iso.encode())])
        picks = np.concatenate([np.sort(rng.choice(idx[b], size=per_bin[b], replace=False)) for b in BINS])
        picks = picks[rng.permutation(len(picks))]
        use.update(picks.tolist())
        sub = pa.table({c: col.take(pa.array(picks)) for c, col in base_cols.items()})
        k = len(picks)
        sub = sub.append_column("iso", pa.array([iso] * k, pa.string())).append_column("language", pa.array([LANGUAGES[iso].name] * k, pa.string()))
        sub = sub.append_column("domain", pa.array(["news"] * k, pa.string())).append_column("prov_pool", pa.array([POOL_VERSION] * k, pa.string()))
        sub = sub.append_column("flag_ssa_mte_train", pa.array([False] * k, pa.bool_()))
        sub = sub.select(list(COLS))
        pq.write_table(sub, os.path.join(a.out, f"{iso}.parquet"), compression="zstd")
    d = {b: int(sum(1 for i in idx[b] if use[int(i)] > 0)) for b in BINS}
    print(now(), "wrote", len(isos), "files; distinct windows used per bin", d, flush=True)
    json.dump(dict(languages=isos, per_bin=per_bin, seed=a.seed, distinct_windows_used=d), open(os.path.join(a.work, "draw_stats.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("extract"); e.add_argument("--src", required=True); e.add_argument("--work", required=True); e.add_argument("--seed", type=int, default=1234)
    e.add_argument("--boiler-min", type=int, default=8); e.add_argument("--limit", type=int, default=0)
    d = sp.add_parser("decon"); d.add_argument("--work", required=True); d.add_argument("--index", required=True); d.add_argument("--manifest", required=True); d.add_argument("--nproc", type=int, default=4)
    w = sp.add_parser("draw"); w.add_argument("--work", required=True); w.add_argument("--out", required=True); w.add_argument("--seed", type=int, default=1234)
    w.add_argument("--per-bin", type=int, nargs=4, default=[6000, 6000, 5000, 5000]); w.add_argument("--exclude", nargs="*", default=["arz"])
    a = ap.parse_args()
    {"extract": cmd_extract, "decon": cmd_decon, "draw": cmd_draw}[a.cmd](a)


if __name__ == "__main__":
    main()
