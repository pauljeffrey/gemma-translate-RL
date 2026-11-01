# SPDX-License-Identifier: AGPL-3.0-or-later
"""English source set of the long-document benchmark (D-44): per (domain, length bin) selection from the licence-clean POOL, with the contamination filter.

    python scripts/longdoc_build_source_set.py --pool-dir longdoc_bench/pool --contam-dir longdoc_bench/contam --out longdoc_bench/source_set \\
        --licence-classes pd permissive --n 500

Inputs: pool/*.jsonl (rows {id, source, domain, licence, licence_class, url, title, attribution, text, ...}; paragraphs separated by a blank line) and
contam/<pool file stem>.flags.jsonl ({id, flag: clean | weak | contaminated, sft_share, rl_share, bench_share, dup_of, ...}; written by longdoc_contam_scan.py).
Steps
  1. keep documents with licence class in `--licence-classes`, an English-looking text, sane sentences (no tables / lists of fragments), contamination flag "clean"
     (`--allow-weak` also keeps "weak"), not a duplicate (exact normalised hash, `dup_of` of the scanner);
  2. [six bins, owner decision 2026-10-20 evening] the long bins 512 / 1024 / 2048 / 4096 are Gemma tokens of the English text; the short bins b64 / b256 are AfroXLM-R tokens (the AfriCOMET
     encoder, 1-64 and 65-256: the RL bins) and their items are SHORT WINDOWS of whole sentences/paragraphs from any document (id `<parent>#b64`), at a seeded random target length.
     For every remaining document and every long bin the EXCERPT: the longest prefix of whole paragraphs whose token count
     lies in the bin's range (`longdoc.BIN_RANGES`), closest to the nominal length; a document can serve any bin it is long enough for, but is used at most ONCE;
  3. quotas per (domain, bin) = domain share x bin share x N (`--bin-shares`, `--domain-shares`; default equal domains, 35/30/22/13% bins); candidates of a cell are ordered
     round-robin over sources (diversity first), deterministic by a seeded hash; shortfalls are filled by the other domains of the same bin (`--no-fill` to disable);
  4. caps: a global share cap per source (Gutenberg <= 10%, EB1911 <= 4%, every other source <= 15% of the set) so that no archaic or single-site source dominates;
     a document whose URL is in the b2048 RL-prompt list (`--exclude-urls-file`) is dropped; 10% of the selected documents (same strata) are marked `split: private`
     (leakage-check half-out, never published).
Outputs (out/core = all N documents with their `split`, out/core_public, out/core_private): docs.jsonl (doc_id, domain, source, licence, licence_class, url, title, attribution, text, sents, para_starts, bin, n_tokens, n_tokens_full,
excerpt, contam), ATTRIBUTION.jsonl (the credit line of every document), and out/SELECTION_REPORT.json (pool funnel, cells, shortfalls, composition).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from gemma_mt.eval import longdoc as L
from gemma_mt.eval.longdoc_release import CANARY

QUALITY_CHARS = 40_000
DEFAULT_BIN_SHARES_4 = {512: 0.35, 1024: 0.30, 2048: 0.22, 4096: 0.13}                      # the first 500-document design (kept for the tests)
DEFAULT_BIN_SHARES = {64: 0.16, 256: 0.20, 512: 0.22, 1024: 0.20, 2048: 0.14, 4096: 0.08}  # six bins: 80 / 100 / 110 / 100 / 70 / 40 of 500
SHORT_RANGE = {64: (8, 64), 256: (65, 256)}                                                # AfroXLM-R tokens (b64 needs >= 8 so that an item is more than a fragment)
# Owner decision 2026-10-20 (evening): thin domains are merged into neighbours; the original label survives as `subdomain`
DOMAIN_MERGE = {"opinion_magazine": "narrative_opinion", "stories": "narrative_opinion", "encyclopedia": "encyclopedia_travel", "travel": "encyclopedia_travel"}
DEFAULT_DOMAIN_SHARES = {"news": 0.22, "encyclopedia_travel": 0.20, "education": 0.16, "science_explainer": 0.16, "government_public": 0.14, "narrative_opinion": 0.12}
BENCH_SOURCE_PREFIXES = ("wmt", "ntrex", "flores", "smoldoc", "mafand", "glotstory")          # evaluation corpora by design: overlap with other benchmarks is not contamination, SFT/RL overlap still is
EXCLUDE_ID_PREFIXES = ("govreport:crs_",)                                # owner decision: drop the CRS half of GovReport (keep GAO)
CLEAN_MAX, WEAK_MAX = 0.02, 0.10
_STOP = frozenset("the of and to in a is that for it as was with be by on not he she they this are from at or an have has had his her which were but their".split())


def norm_hash(text: str) -> str:
    t = re.sub(r"[\W_]+", " ", text.casefold()).strip()
    return hashlib.sha1(t.encode("utf-8")).hexdigest()


def english_ratio(text: str) -> float:
    """Share of the words that are very common English function words (natural English prose is ~0.35-0.5; other languages and tables are far lower)."""
    w = re.findall(r"[A-Za-z']+", text.lower())
    return sum(1 for x in w if x in _STOP) / max(len(w), 1)


_ARTIFACT = re.compile(r"\b(?:the|a|an|of|to|by|with|in|for|and|from|at|on)\s+[\u2019']s\b|\s['\u2019]\s|\(\s*\)|\[\s*\]|\{\{|\}\}|\[\[|\]\]|&[a-z]+;|<[a-z/][^>]*>")
_ABBREV_WORD = re.compile(r"^[A-Za-z]{1,4}\.$")
_OBJECTIVES = re.compile(r"^(by the end of this (section|chapter)|learning objectives|key terms|chapter outline|section summary)", re.IGNORECASE)


def paragraph_ok(p: str) -> bool:
    """Per-paragraph cleaning guard (defence in depth after the fetchers): broken template remnants (`the 's`), markup, learning-objective / key-term blocks, abbreviation-heavy
    reference text (weights tables, indexes), unterminated fragments, shouting."""
    w = p.split()
    if len(w) < 6:
        return False
    if _ARTIFACT.search(p) or _OBJECTIVES.match(p.strip()):
        return False
    if sum(1 for x in w if _ABBREV_WORD.match(x)) / len(w) > 0.10:
        return False
    if p.rstrip()[-1] not in ".!?\"'\u201d\u2019)":
        return False
    letters = [c for c in p if c.isalpha()]
    if letters and sum(1 for c in letters if c.isupper()) / len(letters) > 0.45:
        return False
    return True


def text_quality(text: str, sents: Sequence[str]) -> Tuple[bool, str]:
    """(ok, reason). Rejects non-English, table-like and fragment-like documents."""
    if not sents:
        return False, "no_sentences"
    letters = sum(c.isalpha() for c in text)
    if letters / max(len(text), 1) < 0.70:
        return False, "low_letter_share"
    nonascii = sum(1 for c in text if ord(c) > 0x24F and c.isalpha())
    if nonascii / max(letters, 1) > 0.02:
        return False, "non_latin"
    if english_ratio(text) < 0.15:
        return False, "not_english"
    wl = [len(s.split()) for s in sents]
    if sum(wl) / len(wl) < 6 or sum(wl) / len(wl) > 45:
        return False, "sentence_length"
    if sum(1 for n in wl if n > 100) / len(wl) > 0.03:
        return False, "long_sentences"
    if sum(1 for n in wl if n < 3) / len(wl) > 0.25:
        return False, "fragments"
    digits = sum(c.isdigit() for c in text)
    if digits / max(len(text), 1) > 0.08:
        return False, "numeric"
    if re.search(r"https?://|www\.\S+|\S+@\S+\.\S+", text):
        return False, "url_or_email"
    return True, ""


def excerpt_for_bin(paras: Sequence[str], para_tokens: Sequence[int], bin_nominal: int) -> Optional[Tuple[int, int]]:
    """(number of leading paragraphs, tokens) of the longest paragraph prefix whose token count is inside the bin's range, the one closest to the nominal length;
    None if no prefix lands inside (the first paragraph is already too long, or the whole document is too short)."""
    lo, hi = L.BIN_RANGES[bin_nominal]
    total, best = 0, None
    for k, t in enumerate(para_tokens, 1):
        total += t
        if total > hi:
            break
        if total >= lo and (best is None or abs(total - bin_nominal) <= abs(best[1] - bin_nominal)):
            best = (k, total)
    return best


# Owner decision 2026-10-20: NO medicine and NO law in the core. The pool labels some general-interest documents with those domains (a ministerial speech filed under law, an American-government
# textbook); they are re-labelled, everything else with those labels is dropped. `source` is the `source` field of the pool row (the prefix of its id).
DROP_DOMAINS = ("health", "law_government")
DOMAIN_REMAP: Dict[str, Dict[str, str]] = {
    "govuk": {"law_government": "government_public", "news": "government_public"},        # press releases, speeches, guidance: public information, not legal text
    "openstax": {"law_government": "education"},                                            # civics / business textbooks
}
DEFAULT_EXCLUDE_FILES = ("openstax_ccby_snapshot__permissive", "federalregister*", "fedhealth*", "pmc*", "legislation*", "billsum*", "eurlex*", "medlineplus*", "cdc*", "cp_usgpo*", "hard_*", "unnews*", "nalibali*", "flores_plus*")   # unnews: UN terms forbid redistribution; nalibali: licence unverified; flores_plus: HF-gated (its terms ask not to re-host the text) - all three stay out of the public core. hard_*: the optional hard / domain split (medicine, law, maths-science) has its own builder, scripts/longdoc_build_hard.py
SOURCE_GROUP = {"eb1911": "gutenberg"}            # Britannica 1911 is cut from Project Gutenberg slices: it counts inside the Gutenberg cap


def source_group(source: str) -> str:
    """Cap group of a source: Gutenberg + EB1911; every Common Pile slice (`cp_*`) together (owner: Common Pile <= 20%)."""
    if source in SOURCE_GROUP:
        return SOURCE_GROUP[source]
    return "commonpile" if source.startswith("cp_") else source


def remap_domain(source: str, domain: str) -> str:
    return DOMAIN_REMAP.get(source, DOMAIN_REMAP.get(source.split("-")[0], {})).get(domain, domain)


def read_pool(pool_dir: str, licence_classes: Sequence[str], exclude: Sequence[str] = (), drop_domains: Sequence[str] = (), *, merge_domains: bool = True,
              exclude_id_prefixes: Sequence[str] = EXCLUDE_ID_PREFIXES) -> List[Dict[str, Any]]:
    """`licence_classes` may be ["all"]. Domains are re-labelled (DOMAIN_REMAP), dropped (`drop_domains`, on the re-labelled name), then merged (DOMAIN_MERGE; the pre-merge label stays in `subdomain`)."""
    docs = []
    for p in sorted(glob.glob(os.path.join(pool_dir, "*.jsonl"))):
        stem = os.path.splitext(os.path.basename(p))[0]
        if stem in exclude or any(stem.startswith(x[:-1]) for x in exclude if x.endswith("*")):
            continue
        for r in L.read_jsonl(p):
            r["_pool_file"] = stem
            if "all" not in licence_classes and r.get("licence_class") not in licence_classes:
                continue
            if any(str(r.get("id", "")).startswith(x) for x in exclude_id_prefixes):
                continue
            r["domain_orig"] = r.get("domain", "")
            r["domain"] = remap_domain(r.get("source", ""), r.get("domain", ""))
            if r["domain"] in drop_domains:
                continue
            r["subdomain"] = r["domain"]
            if merge_domains:
                r["domain"] = DOMAIN_MERGE.get(r["domain"], r["domain"])
            docs.append(r)
    return docs


def read_flags(contam_dir: Optional[str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not contam_dir:
        return out
    for p in sorted(glob.glob(os.path.join(contam_dir, "*.flags.jsonl"))):
        for r in L.read_jsonl(p):
            out[r["id"]] = r
    return out


def _order_key(seed: int, doc_id: str, bin_nominal: int) -> str:
    return hashlib.sha1(f"{seed}|{bin_nominal}|{doc_id}".encode("utf-8")).hexdigest()


def round_robin_by_source(cands: Sequence[Dict[str, Any]], seed: int, bin_nominal: int) -> List[Dict[str, Any]]:
    """Order candidates so that sources alternate (a cell of 12 documents gets 12 different sources before any source repeats)."""
    by: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in sorted(cands, key=lambda c: _order_key(seed, c["id"], bin_nominal)):
        by[c["source"]].append(c)
    keys = sorted(by, key=lambda s: _order_key(seed, s, bin_nominal))
    out, i = [], 0
    while any(by.values()):
        for s in keys:
            if by[s]:
                out.append(by[s].pop(0))
        i += 1
    return out


def quotas(n: int, domains: Sequence[str], bin_shares: Dict[int, float], domain_shares: Optional[Dict[str, float]] = None) -> Dict[Tuple[str, int], int]:
    """Integer quota per (domain, bin) summing exactly to n (largest remainder)."""
    ds = dict(domain_shares or {})
    known = [v for v in ds.values() if v > 0]
    mean = (sum(known) / len(known)) if known else 1.0
    ds = {d: ds.get(d, mean) for d in domains}                      # a domain without a configured share gets the mean share
    z = sum(ds.values())
    raw = {(d, b): n * (ds[d] / z) * (s / sum(bin_shares.values())) for d in domains for b, s in bin_shares.items()}
    out = {k: int(v) for k, v in raw.items()}
    rest = n - sum(out.values())
    for k in sorted(raw, key=lambda k: raw[k] - out[k], reverse=True)[:rest]:
        out[k] += 1
    return out


def effective_flag(fl: Dict[str, Any], source: str) -> str:
    """Contamination flag used for the selection. Evaluation corpora (WMT, NTREX, FLORES+, SmolDoc) are in the scanner's benchmark list by design, so their `bench_share` is ignored:
    the flag is recomputed from the SFT / RL overlap alone (same thresholds as the scanner). Everything else keeps the scanner's flag."""
    if source.lower().startswith(BENCH_SOURCE_PREFIXES) and fl.get("sft_share") is not None:
        m = max(fl.get("sft_share") or 0.0, fl.get("sft_word_share") or 0.0, fl.get("rl_share") or 0.0)
        if (fl.get("sft_longest_run") or 0) >= 64 or (fl.get("sft_word_longest_run") or 0) >= 50 or (fl.get("rl_longest_run") or 0) >= 50:
            return "contaminated"                                  # a verbatim run in the training data (scanner thresholds: 64 tokens / 50 words)
        return "clean" if m < CLEAN_MAX else ("weak" if m < WEAK_MAX else "contaminated")
    return fl.get("flag", "")


def short_window(paras: Sequence[str], sent_lists: Sequence[Sequence[str]], sent_tok: Sequence[Sequence[int]], bin_nominal: int, seed: int, doc_id: str,
                 max_starts: int = 12, max_paras: int = 4) -> Optional[Tuple[str, int]]:
    """A short item of bin 64 / 256 (AfroXLM-R tokens): whole sentences from the start of one of the first `max_starts` paragraphs, over at most `max_paras` paragraphs, with a token
    count in the bin's range, closest to a seeded random target length (so the bin is covered evenly; paragraph-final ends are preferred). Returns (text, summed tokens) or None."""
    lo, hi = SHORT_RANGE[bin_nominal]
    h = int(hashlib.sha1(f"{seed}|short|{bin_nominal}|{doc_id}".encode()).hexdigest()[:8], 16)
    target = lo + h % (hi - lo + 1)
    best = None
    for i in range(min(len(paras), max_starts)):
        total = 0
        done = False
        for j in range(i, min(len(paras), i + max_paras)):
            for k, t in enumerate(sent_tok[j], 1):
                total += t
                if total > hi:
                    done = True
                    break
                if total >= lo:
                    end_of_para = k == len(sent_tok[j])
                    score = abs(total - target) + (0 if end_of_para else 6)
                    if best is None or score < best[0]:
                        text = "\n\n".join(list(paras[i:j]) + [paras[j] if end_of_para else " ".join(sent_lists[j][:k])])
                        best = (score, text, total)
            if done:
                break
    return None if best is None else (best[1], best[2])


def build_candidates(pool: Sequence[Dict[str, Any]], flags: Dict[str, Dict[str, Any]], count_tokens: Callable[[str], int], *, allow_weak: bool = False,
                     require_flags: bool = True, min_tokens: int = 400, excluded_urls: Optional[set] = None, afro_count: Optional[Callable[[str], int]] = None,
                     seed: int = 0, exclude_ids: Optional[set] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """The funnel: pool -> candidate documents with their per-bin excerpts. Returns (candidates, funnel counts).
    `afro_count` (AfroXLM-R token counter) switches the short bins 64 / 256 on: `excerpts[64]` / `excerpts[256]` = (0, tokens) and the text is in `short_text`."""
    funnel: Counter = Counter()
    seen_hash: Dict[str, str] = {}
    cands = []
    excluded_urls = excluded_urls or set()
    exclude_ids = exclude_ids or set()
    many = lambda f, xs: f.many(xs) if hasattr(f, "many") else [f(x) for x in xs]
    for r in pool:
        funnel["pool"] += 1
        if r["id"] in exclude_ids:
            funnel["drop_excluded_id"] += 1
            continue
        if excluded_urls and norm_url(r.get("url", "")) in excluded_urls:
            funnel["drop_url_in_rl_b2048_prompts"] += 1
            continue
        fl = flags.get(r["id"])
        if fl is None:
            if require_flags:
                funnel["drop_no_contamination_flag"] += 1
                continue
        else:
            allowed = ("clean", "weak") if allow_weak else ("clean",)
            ef = effective_flag(fl, r.get("source", ""))
            if ef not in allowed:
                funnel[f"drop_contam_{ef}"] += 1
                continue
            if fl.get("dup_of"):
                funnel["drop_dup_of_other_pool_doc"] += 1
                continue
        long_possible = len(r["text"]) >= min_tokens * 2.5       # English is ~4-5 characters per Gemma token: far too short for a long bin, no need to tokenize
        if not long_possible and afro_count is None:
            funnel["drop_too_short"] += 1
            continue
        paras = L.paragraphs(r["text"])
        n_all = len(paras)
        paras = [p for p in paras if paragraph_ok(p)]
        funnel["paragraphs_removed"] += n_all - len(paras)
        if not paras:
            funnel["drop_empty"] += 1
            continue
        if n_all - len(paras) > max(2, 0.2 * n_all):
            funnel["drop_too_many_bad_paragraphs"] += 1
            continue
        h = norm_hash(" ".join(paras))
        if h in seen_hash:
            funnel["drop_exact_duplicate"] += 1
            continue
        ex, ptok, total, short_text = {}, [], 0, {}
        if long_possible:
            ptok = many(count_tokens, paras)
            total = sum(ptok)
            if total >= min_tokens:
                for b in L.BINS:
                    e = excerpt_for_bin(paras, ptok, b)
                    if e:
                        ex[b] = e
        else:
            total = 0
        joined = " ".join(paras)[:QUALITY_CHARS]              # judge the first ~40k characters (> the longest excerpt)
        ok, why = text_quality(joined, L.split_sentences(joined))
        if not ok:
            funnel[f"drop_{why}"] += 1
            continue
        if afro_count is not None:
            head = paras[:12]
            sl = [L.split_sentences(p) or [p] for p in head]
            flat = [x for ss in sl for x in ss]
            tk = many(afro_count, flat)
            st, i = [], 0
            for ss in sl:
                st.append(tk[i:i + len(ss)])
                i += len(ss)
            for b in L.SHORT_BINS:
                if f"{r['id']}#b{b}" in exclude_ids:
                    funnel[f"drop_short_excluded_id_b{b}"] += 1
                    continue
                w = short_window(head, sl, st, b, seed, r["id"])
                if w is None:
                    continue
                n_afro = afro_count(w[0])                      # the real count of the joined text (the per-sentence sum is only the search guide)
                lo, hi = SHORT_RANGE[b]
                if lo <= n_afro <= hi:
                    ex[b] = (0, n_afro)
                    short_text[b] = w[0]
        if not ex:
            funnel["drop_no_bin_fits"] += 1
            continue
        seen_hash[h] = r["id"]
        funnel["candidates"] += 1
        for b in ex:
            funnel[f"candidates_bin{b}"] += 1
        cands.append({**{k: r.get(k, "") for k in ("id", "source", "domain", "subdomain", "licence", "licence_class", "url", "title", "attribution", "date")},
                      "paras": paras, "para_tokens": ptok, "n_tokens_full": total, "excerpts": ex, "short_text": short_text,
                      "contam": ({k: fl.get(k) for k in ("flag", "sft_share", "rl_share", "bench_share")} if fl else None), "pool_file": r["_pool_file"]})
    return cands, dict(funnel)


def select_cells(cands: Sequence[Dict[str, Any]], n_extended: int, n_core: int, *, bin_shares: Optional[Dict[int, float]] = None, domain_shares: Optional[Dict[str, float]] = None,
                 seed: int = 0, fill: bool = True, source_cap: float = 0.35, max_share: Optional[Dict[str, float]] = None, default_max_share: float = 1.0,
                 group_domains: Optional[Dict[str, Sequence[str]]] = None) -> Dict[str, Any]:
    """Quota selection. A document is used once. Returns {"extended": [(cand, bin)], "core": [...], "shortfall": {...}, "cells": {...}}.
    The core set takes the first `quota_core(cell)` documents of each cell's ordering, so core is a subset of extended with the same strata proportions."""
    bin_shares = bin_shares or DEFAULT_BIN_SHARES
    domains = sorted({c["domain"] for c in cands})
    q_ext = quotas(n_extended, domains, bin_shares, domain_shares)
    q_core = quotas(n_core, domains, bin_shares, domain_shares)
    used: set = set()
    n_total = max(n_extended, n_core)
    caps = max_share or {}
    src_total: Counter = Counter()

    grp = lambda c: source_group(c["source"])

    gd = group_domains or {}

    def allowed(c: Dict[str, Any]) -> bool:       # a capped group (Gutenberg) may only fill its own domains (stories, encyclopedia), so that the 10% budget is not eaten elsewhere
        return grp(c) not in gd or c["domain"] in gd[grp(c)]

    def room(c: Dict[str, Any]) -> bool:          # global cap per source group: Gutenberg (+ EB1911) <= 10% of the set, any other source <= default_max_share
        return src_total[grp(c)] < max(1, int(caps.get(grp(c), default_max_share) * n_total))

    order: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    shortfall: Dict[Tuple[str, int], int] = {}
    chosen: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    # long bins first: they have the fewest candidates; short documents can always serve a short bin
    for b in sorted(bin_shares, reverse=True):
        supply = {d: sum(1 for c in cands if c["domain"] == d and b in c["excerpts"] and allowed(c)) for d in domains}
        for d in sorted(domains, key=lambda x: (supply[x], x)):                       # scarcest domain first
            pool = [c for c in cands if c["domain"] == d and b in c["excerpts"] and c["id"] not in used and allowed(c)]
            seq = round_robin_by_source(pool, seed, b)
            cap = max(1, int(source_cap * q_ext[(d, b)]))
            per_source: Counter = Counter()
            take = []
            for c in seq:
                if len(take) >= q_ext[(d, b)]:
                    break
                if per_source[c["source"]] >= cap or not room(c):
                    continue
                per_source[c["source"]] += 1
                src_total[grp(c)] += 1
                take.append(c)
            for c in take:
                used.add(c["id"])
            chosen[(d, b)] = take
            if len(take) < q_ext[(d, b)]:
                shortfall[(d, b)] = q_ext[(d, b)] - len(take)
    filled = 0
    if fill and shortfall:
        for (d, b), missing in sorted(shortfall.items(), key=lambda kv: -kv[0][1]):
            others: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for c in round_robin_by_source([c for c in cands if b in c["excerpts"] and c["id"] not in used and c["domain"] != d and allowed(c)], seed, b):
                others[c["domain"]].append(c)
            while missing > 0 and any(others.values()):
                # the other domain with the fewest documents so far gets the next one (keeps the domains balanced); documents of a capped group are skipped when the cap is reached
                dd = min((x for x in others if others[x]), key=lambda x: (sum(len(v) for (k0, _), v in chosen.items() if k0 == x), x))
                c = others[dd].pop(0)
                if not room(c):
                    continue
                src_total[grp(c)] += 1
                chosen[(c["domain"], b)].append(c)       # joins the cell of its own domain, beyond that domain's quota
                used.add(c["id"])
                missing -= 1
                filled += 1
            shortfall[(d, b)] = missing
    shortfall = {k: v for k, v in shortfall.items() if v > 0}
    ext, core = [], []
    for (d, b), docs in sorted(chosen.items()):
        k_core = q_core.get((d, b), 0)
        for i, c in enumerate(docs):
            ext.append((c, b))
            if i < k_core:
                core.append((c, b))
    # a cell that could not fill its extended quota leaves the core short as well: top the core up from the extended set, bin by bin (documents of the bins with a deficit first)
    if len(core) < n_core:
        in_core = {c["id"] for c, _ in core}
        target_bin = Counter()
        for (d, b), q in q_core.items():
            target_bin[b] += q
        have_bin = Counter(b for _, b in core)
        spare = [(c, b) for c, b in ext if c["id"] not in in_core]
        for pass_ in (0, 1):
            for c, b in spare:
                if len(core) >= n_core:
                    break
                if c["id"] in in_core or (pass_ == 0 and have_bin[b] >= target_bin[b]):
                    continue
                core.append((c, b))
                in_core.add(c["id"])
                have_bin[b] += 1
    return {"extended": ext, "core": core, "shortfall": shortfall, "filled_from_other_domains": filled, "quota_extended": q_ext, "quota_core": q_core, "domains": domains}


def assign_private(chosen: Sequence[Tuple[Dict[str, Any], int]], share: float = 0.10, seed: int = 0) -> set:
    """Ids of the private leakage-check documents: `share` of the set (largest remainder over the (domain, bin) cells, so the private half has the same strata), the documents of
    a cell with the smallest seeded hash."""
    cells: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for c, b in chosen:
        cells[(c["domain"], b)].append(c)
    n_priv = int(round(share * len(chosen)))
    raw = {k: share * len(v) for k, v in cells.items()}
    take = {k: int(x) for k, x in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - take[k], reverse=True)[: max(0, n_priv - sum(take.values()))]:
        take[k] += 1
    out = set()
    for k, v in cells.items():
        for c in sorted(v, key=lambda c: hashlib.sha1(f"{seed}|private|{c['id']}".encode()).hexdigest())[: take[k]]:
            out.add(c["id"])
    return out


def norm_url(u: str) -> str:
    u = (u or "").strip().lower()
    u = re.sub(r"^https?://(www\.)?", "", u)
    return u.split("#")[0].split("?")[0].rstrip("/")


def read_excluded_urls(path: Optional[str]) -> set:
    """`source_document_id<TAB>url` lines (comment lines start with #), e.g. /workspace/sft_build/gemma/rl_prompts_b2048_doc_ids.txt: documents whose text is in the b2048 RL prompts."""
    out = set()
    if path and os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("#") or "\t" not in line:
                    continue
                out.add(norm_url(line.rstrip("\n").split("\t", 1)[1]))
    return out


def doc_row(c: Dict[str, Any], b: int, count_tokens: Callable[[str], int], afro_count: Optional[Callable[[str], int]] = None) -> Dict[str, Any]:
    """One release row. `bin` is the nominal bin: Gemma tokens for 512..4096, AfroXLM-R tokens for 64 / 256 (`bin_unit`); `n_tokens` is always the Gemma count of the text and
    `n_tokens_afro` the AfroXLM-R count (the unit of the RL length bins and of the AfriCOMET chunking)."""
    k, ntok = c["excerpts"][b]
    short = b in L.SHORT_BINS
    if short:
        text = c["short_text"][b]
        paras = L.paragraphs(text)
        k, ntok, doc_id, excerpt = len(paras), count_tokens(text), f"{c['id']}#b{b}", True
        n_afro = c["excerpts"][b][1]
    else:
        paras = c["paras"][:k]
        text = "\n\n".join(paras)
        doc_id, excerpt = c["id"], k < len(c["paras"])
        n_afro = afro_count(text) if afro_count else None
    sents = L.canonical_sentences(text)
    return {"canary": CANARY, "doc_id": doc_id, "parent_id": c["id"], "domain": c["domain"], "subdomain": c.get("subdomain", c["domain"]), "source": c["source"], "licence": c["licence"],
            "licence_class": c["licence_class"], "url": c["url"], "title": c["title"], "attribution": c["attribution"], "date": c.get("date", ""), "text": text, "sents": sents,
            "para_starts": L.paragraph_starts(text, sents), "bin": b, "bin_unit": "afroxlmr_tokens" if short else "gemma_tokens", "n_tokens": ntok, "n_tokens_afro": n_afro,
            "n_tokens_full": c["n_tokens_full"], "excerpt": excerpt, "n_paragraphs": k, "n_sentences": len(sents), "contam": c["contam"]}


def write_set(rows: Sequence[Dict[str, Any]], out_dir: str) -> None:
    L.write_jsonl(os.path.join(out_dir, "docs.jsonl"), rows)
    L.write_jsonl(os.path.join(out_dir, "ATTRIBUTION.jsonl"), [{"doc_id": r["doc_id"], "title": r["title"], "attribution": r["attribution"], "url": r["url"], "licence": r["licence"],
                                                              "source": r["source"], "excerpt": r["excerpt"]} for r in rows])


def summarise(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    toks = sorted(r["n_tokens"] for r in rows)
    return {"n": len(rows), "tokens_total": sum(toks), "tokens_mean": (sum(toks) / len(toks)) if toks else 0, "sentences_total": sum(r["n_sentences"] for r in rows),
            "by_bin": dict(sorted(Counter(r["bin"] for r in rows).items())), "by_domain": dict(sorted(Counter(r["domain"] for r in rows).items())),
            "by_source": dict(sorted(Counter(r["source"] for r in rows).items())), "by_licence": dict(sorted(Counter(r["licence"] for r in rows).items())),
            "by_licence_class": dict(sorted(Counter(r["licence_class"] for r in rows).items())),
            "by_source_bin": {f"{sr}|{b}": n for (sr, b), n in sorted(Counter((r["source"], r["bin"]) for r in rows).items())},
            "by_source_domain": {f"{sr}|{d}": n for (sr, d), n in sorted(Counter((r["source"], r["domain"]) for r in rows).items())},
            "by_source_licence": {f"{sr}|{l}": n for (sr, l), n in sorted(Counter((r["source"], r["licence"]) for r in rows).items())},
            "by_subdomain": dict(sorted(Counter(r.get("subdomain", r["domain"]) for r in rows).items())),
            "by_domain_bin": {f"{d}|{b}": n for (d, b), n in sorted(Counter((r["domain"], r["bin"]) for r in rows).items())}}


def examples_markdown(rows: Sequence[Dict[str, Any]], per_domain: int = 3, words: int = 70, seed: int = 0) -> str:
    """`per_domain` example documents of every domain (different sources first, different bins when possible): metadata and the first `words` words."""
    by: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[r["domain"]].append(r)
    out = []
    for dom in sorted(by):
        out += [f"#### {dom}", ""]
        picked, seen_src, seen_bin = [], set(), set()
        pool = sorted(by[dom], key=lambda r: hashlib.sha1(f"{seed}|{r['doc_id']}".encode()).hexdigest())
        for r in pool:
            if len(picked) >= per_domain:
                break
            if r["source"] not in seen_src and r["bin"] not in seen_bin:
                picked.append(r)
                seen_src.add(r["source"])
                seen_bin.add(r["bin"])
        for r in pool:
            if len(picked) >= per_domain:
                break
            if r not in picked:
                picked.append(r)
        for r in picked:
            w = r["text"].split()
            out += [f"* `{r['doc_id']}` | bin {r['bin']} ({r['n_tokens']} Gemma tokens, {r['n_sentences']} sentences{', excerpt' if r['excerpt'] else ''}) | {r['licence']} | "
                    f"{r['attribution'] or r['source']} | {r['url']}", f"  > {' '.join(w[:words])}{' ...' if len(w) > words else ''}", ""]
    return "\n".join(out)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool-dir", required=True)
    ap.add_argument("--contam-dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--licence-classes", nargs="+", default=["all"], help="licence classes kept (default all: owner decision 2026-10-20 evening, any data licence is acceptable)")
    ap.add_argument("--exclude-files", nargs="*", default=list(DEFAULT_EXCLUDE_FILES),
                    help="pool file stems to leave out (`name*` = prefix). Default: the OpenStax snapshot (relicensed to CC-BY-NC-SA upstream in 2025-26) and every medicine / law source "
                         "(owner decision 2026-10-20: no medicine, no law in the core); pass '--exclude-files' with no names to keep everything")
    ap.add_argument("--drop-domains", nargs="*", default=list(DROP_DOMAINS), help="domains left out after the re-labelling (default health, law_government)")
    ap.add_argument("--n", type=int, default=500, help="size of the ONE shared English set (owner decision 2026-10-20: 500; no core / extended split)")
    ap.add_argument("--private-share", type=float, default=0.10, help="share of the set held back as the private leakage-check split (same strata)")
    ap.add_argument("--max-share", nargs="*", default=["gutenberg=0.10", "commonpile=0.20"], metavar="SOURCE=FRACTION", help="global cap per source as a fraction of the set")
    ap.add_argument("--group-domains", nargs="*", default=["gutenberg=stories,encyclopedia"], metavar="GROUP=DOM1,DOM2", help="a capped source group may only fill these domains")
    ap.add_argument("--default-max-share", type=float, default=0.15, help="cap for every other source (diversity: no source above this share)")
    ap.add_argument("--exclude-urls-file", nargs="*", default=["/workspace/sft_build/gemma/rl_prompts_b2048_doc_ids.txt"],
                    help="tsv lines `id<TAB>url` of documents in the b2048 RL prompts: a pool document with such a URL is dropped")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-weak", action="store_true")
    ap.add_argument("--no-require-flags", action="store_true", help="keep documents that have no contamination flag (NOT for a release)")
    ap.add_argument("--no-fill", action="store_true")
    ap.add_argument("--domains", nargs="*", default=None, help="restrict to these domains")
    ap.add_argument("--bin-shares", nargs=6, type=float, default=[DEFAULT_BIN_SHARES[b] for b in L.ALL_BINS], metavar=("S64", "S256", "S512", "S1024", "S2048", "S4096"))
    ap.add_argument("--domain-shares", nargs="*", default=[f"{k}={v}" for k, v in DEFAULT_DOMAIN_SHARES.items()], metavar="DOMAIN=SHARE")
    ap.add_argument("--afro-tokenizer", default=None, help="tokenizer.json of Davlan/afro-xlmr-large-76L (default: the HF cache); the short bins need it")
    ap.add_argument("--no-short-bins", action="store_true")
    ap.add_argument("--exclude-ids-file", nargs="*", default=[], help="text file(s) with one document id per line to leave out (e.g. items the excerpt-level contamination check rejected)")
    ap.add_argument("--no-merge-domains", action="store_true")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--examples-md", default=None, help="write 3 example documents per domain of the set to this markdown file")
    a = ap.parse_args(argv)
    count = L.load_token_counter(a.tokenizer)
    afro = None if a.no_short_bins else L.load_afro_counter(a.afro_tokenizer)
    pool = read_pool(a.pool_dir, a.licence_classes, a.exclude_files, a.drop_domains, merge_domains=not a.no_merge_domains)
    if a.domains:
        pool = [r for r in pool if r.get("domain") in a.domains]
    flags = read_flags(a.contam_dir)
    excluded = set()
    for p in a.exclude_urls_file or []:
        excluded |= read_excluded_urls(p)
    ex_ids = set()
    for p in a.exclude_ids_file:
        with open(p, encoding="utf-8") as f:
            ex_ids |= {x.strip() for x in f if x.strip()}
    cands, funnel = build_candidates(pool, flags, count, allow_weak=a.allow_weak, require_flags=not a.no_require_flags, excluded_urls=excluded, afro_count=afro, seed=a.seed, exclude_ids=ex_ids)
    bin_shares = dict(zip(L.ALL_BINS, a.bin_shares))
    if a.no_short_bins:
        bin_shares = {b: v for b, v in bin_shares.items() if b in L.BINS}
    dshares = {k: float(v) for k, v in (x.split("=", 1) for x in a.domain_shares or [])} or None
    caps = {k: float(v) for k, v in (x.split("=", 1) for x in a.max_share or [])}
    sel = select_cells(cands, a.n, a.n, bin_shares=bin_shares, domain_shares=dshares, seed=a.seed, fill=not a.no_fill, max_share=caps, default_max_share=a.default_max_share,
                       group_domains={k: v.split(",") for k, v in (x.split("=", 1) for x in a.group_domains or [])})
    chosen = sel["core"]
    priv = assign_private(chosen, a.private_share, a.seed)
    rows = []
    for c, b in chosen:
        r = doc_row(c, b, count, afro)
        r["subset"] = "core"
        r["split"] = "private" if c["id"] in priv else "public"
        rows.append(r)
    if a.examples_md:
        with open(a.examples_md, "w", encoding="utf-8") as f:
            f.write(examples_markdown([r for r in rows if r["split"] == "public"]))
    write_set(rows, os.path.join(a.out, "core"))                                              # the master file: all documents, `split` says public / private
    write_set([r for r in rows if r["split"] == "public"], os.path.join(a.out, "core_public"))
    write_set([r for r in rows if r["split"] == "private"], os.path.join(a.out, "core_private"))
    cand_cells = Counter()
    for c in cands:
        for b in c["excerpts"]:
            cand_cells[f"{c['domain']}|{b}"] += 1
    rep = {"funnel": funnel, "candidate_cells": dict(sorted(cand_cells.items())), "quota": {f"{d}|{b}": n for (d, b), n in sorted(sel["quota_core"].items())},
           "shortfall": {f"{d}|{b}": n for (d, b), n in sorted(sel["shortfall"].items())}, "filled_from_other_domains": sel["filled_from_other_domains"],
           "set": summarise(rows), "public": summarise([r for r in rows if r["split"] == "public"]), "private": summarise([r for r in rows if r["split"] == "private"]),
           "licence_classes": a.licence_classes, "allow_weak": a.allow_weak, "seed": a.seed, "max_share": caps, "default_max_share": a.default_max_share,
           "excluded_urls": len(excluded), "n": a.n, "private_share": a.private_share}
    with open(os.path.join(a.out, "SELECTION_REPORT.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print(json.dumps({"funnel": funnel, "n": rep["set"]["n"], "private": rep["private"]["n"], "shortfall": rep["shortfall"], "by_source": rep["set"]["by_source"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
