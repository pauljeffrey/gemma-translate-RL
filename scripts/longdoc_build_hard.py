#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build the HARD / DOMAIN split (medicine, law, maths-science) of the long-document benchmark from the pool.

    python scripts/longdoc_build_hard.py --pool-dir longdoc_bench/pool [--stage-dir longdoc_bench/hard_stage] --contam-dir longdoc_bench/contam_hard \\
        --out longdoc_bench/hard --n 150

Inputs   pool/*.jsonl rows (schema of `longdoc_fetch_sources.py`) of the files that belong to this split (medicine: pmc*, fedhealth, health-labelled rows of govuk / cp_pressbooks /
         cp_libretexts / openstax; law: federalregister, legislation, billsum, cp_usgpo*; maths-science: science rows of nasa / cp_libretexts / cp_pressbooks / openstax), plus the
         `hard_*__<class>.jsonl` files of `longdoc_fetch_hard.py` (arxiv, caselaw, eurlex, cuad, wikipedia, europepmc, who) from --pool-dir and --stage-dir; the contamination flags
         `<stem>.flags.jsonl` of `longdoc_contam_scan.py` (a document whose flag is not `clean`, i.e. more than 2% of its windows in the SFT stream / RL prompts v3 incl. b2048 /
         existing benchmarks / another pool document, is dropped).
Steps    1. domain mapping of every pool row; 2. cleaning with the domain rules of `longdoc_hard` (formulas, numbers, symbols, citation markers and section numbers stay; URLs, e-mail
         addresses and telephone numbers removed; running prose only: the first run of acceptable paragraphs); 3. the EXCERPT of every bin (512 / 1024 / 2048 / 4096 Gemma tokens, whole
         paragraphs, one excerpt per document); 4. selection stratified by (domain, bin) with a diversity cap (<= 35% of a domain per source, round-robin over sources, documents with
         domain features first: numerals in medicine, legal references in law, formulas in maths-science); a shortfall of a bin is carried to the next smaller bin; 5. 10% private split.
Outputs  out/docs.jsonl (PUBLIC rows only), out/docs_private.jsonl (private rows), out/docs_master_do_not_publish.jsonl (all rows, `split` says which), out/ATTRIBUTION.jsonl,
         out/ATTRIBUTION_private.jsonl, out/SELECTION_REPORT.json, out/examples.md, out/metric_config.json.
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from gemma_mt.eval import longdoc as L  # noqa: E402
from gemma_mt.eval import longdoc_hard as H  # noqa: E402
from gemma_mt.eval import longdoc_select as S  # noqa: E402
from gemma_mt.eval.longdoc_release import CANARY  # noqa: E402

HARD_BIN_SHARES = {512: 0.40, 1024: 0.35, 2048: 0.20, 4096: 0.05}
# pool files of the earlier campaigns that belong to the hard split (stem prefixes)
HARD_POOL_PREFIXES = ("pmc", "fedhealth", "federalregister", "legislation", "billsum", "cp_usgpo", "govuk", "cp_pressbooks", "cp_libretexts", "openstax_ccby_snapshot", "hard_")
MIN_TOKENS = 400


def domain_of_pool_row(r: Dict[str, Any]) -> Tuple[Optional[str], str]:
    """(hard domain | None, source group) of a pool row.  `source` of the row is the diversity-cap key (`cp_usgpo` covers both usgpo files)."""
    src, dom = r.get("source", ""), r.get("domain", "")
    if src == "cp_books":                                            # Common Pile Pressbooks and LibreTexts share `source`; the pool file tells them apart
        src = (r.get("_pool_file") or "cp_books").split("__")[0]
    if dom in H.DOMAINS:                                              # rows written by longdoc_fetch_hard.py
        return dom, src
    if src == "pmc":
        return "medicine", src
    if src == "fedhealth":
        return ("medicine", src) if dom == "health" else (None, src)
    if src in ("govuk", "cp_pressbooks", "cp_libretexts", "openstax", "openstax_ccby_snapshot") or src.startswith("openstax"):
        if dom == "health":
            return "medicine", src
        if dom == "science_explainer" and src != "govuk":
            return "math_science", src
        return None, src
    if src in ("federalregister", "legislation", "billsum", "cp_usgpo") or src.startswith("cp_usgpo"):
        return ("law", "cp_usgpo" if src.startswith("cp_usgpo") else src) if dom == "law_government" else (None, src)
    return None, src


def hard_quotas(n: int, domains: Sequence[str], bin_shares: Optional[Dict[int, float]] = None) -> Dict[Tuple[str, int], int]:
    """Quota per (domain, bin): the same number of documents for every domain (n split over the domains, remainder to the first ones), each domain split over the bins by largest remainder."""
    out: Dict[Tuple[str, int], int] = {}
    doms = list(domains)
    for i, d in enumerate(doms):
        nd = n // len(doms) + (1 if i < n % len(doms) else 0)
        out.update(S.quotas(nd, [d], bin_shares or HARD_BIN_SHARES))
    return out


def read_hard_pool(pool_dirs: Sequence[str], exclude_prefix: Sequence[str] = ()) -> List[Dict[str, Any]]:
    rows, seen = [], set()
    for d in pool_dirs:
        if not d:
            continue
        for p in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
            stem = os.path.splitext(os.path.basename(p))[0]
            if stem in seen or not stem.startswith(HARD_POOL_PREFIXES) or any(stem.startswith(x) for x in exclude_prefix):
                continue
            seen.add(stem)
            for r in L.read_jsonl(p):
                r["_pool_file"] = stem
                rows.append(r)
    return rows


def hardness(domain: str, text: str) -> Dict[str, Any]:
    """Domain features of an excerpt (reported; the 'hard' tier is selected first inside a cell)."""
    fd = H.formula_density(text)
    nums = len(H.extract_numbers(text))
    prot = Counter(c for c, _ in H.protected_tokens(text))
    cues = sum(len(H.source_cues(s)) for s in L.split_sentences(text))
    legal = prot["statute_ref"] + prot["eu_number"] + prot["us_cite"]
    if domain == "medicine":
        ok = nums >= 4 or prot["acronym"] + prot["chem_gene"] + prot["drug"] >= 3
    elif domain == "law":
        ok = legal >= 1 or cues >= 3
    else:
        ok = fd["n_formulas"] >= 3
    return {"hard_features": ok, "n_numerals": nums, "n_protected": sum(prot.values()), "n_legal_refs": legal, "n_modality_cues": cues, **fd}


def build_candidates(pool: Sequence[Dict[str, Any]], flags: Dict[str, Dict[str, Any]], count, *, allow_weak: bool = False, require_flags: bool = True,
                     excluded_urls: Optional[set] = None, excluded_ids: Optional[set] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    funnel: Counter = Counter()
    pool_by_source: Counter = Counter()
    by_domain: Dict[str, Counter] = defaultdict(Counter)
    seen_hash: Dict[str, str] = {}
    cands = []
    excluded_urls = excluded_urls or set()
    for r in pool:
        dom, grp = domain_of_pool_row(r)
        if dom is None:
            funnel["not_in_hard_domains"] += 1
            continue
        funnel["pool"] += 1
        by_domain[dom]["pool"] += 1
        pool_by_source[f"{dom}|{grp}"] += 1
        if excluded_ids and r["id"] in excluded_ids:
            by_domain[dom]["drop_in_core_500"] += 1
            continue
        if excluded_urls and S.norm_url(r.get("url", "")) in excluded_urls:
            by_domain[dom]["drop_url_in_rl_b2048_prompts"] += 1
            continue
        fl = flags.get(r["id"])
        if fl is None:
            if require_flags:
                by_domain[dom]["drop_no_contamination_flag"] += 1
                continue
        else:
            if fl.get("flag") not in (("clean", "weak") if allow_weak else ("clean",)):
                by_domain[dom][f"drop_contam_{fl.get('flag')}"] += 1
                continue
            if fl.get("dup_of"):
                by_domain[dom]["drop_dup_of_other_pool_doc"] += 1
                continue
        raw = L.paragraphs(r["text"])
        paras = [H.clean_hard_paragraph(p) for p in raw]
        ok_flags = []
        for p in paras:
            ok, why = H.paragraph_ok_hard(p)
            ok_flags.append(p if ok else None)
        runs, cur = [], []
        for p in ok_flags + [None]:
            if p is None:
                if cur:
                    runs.append(cur)
                cur = []
            else:
                cur.append(p)
        if not runs:
            by_domain[dom]["drop_no_acceptable_paragraph"] += 1
            continue
        # running prose must stay coherent: the first run that reaches the minimum length, else the longest run
        best = None
        for run in runs:
            if sum(count.many(run)) >= MIN_TOKENS:
                best = run
                break
        best = best or max(runs, key=len)
        paras = list(dict.fromkeys(best))
        ptok = count.many(paras)
        if sum(ptok) < MIN_TOKENS:
            by_domain[dom]["drop_too_short"] += 1
            continue
        h = S.norm_hash(" ".join(paras))
        if h in seen_hash:
            by_domain[dom]["drop_exact_duplicate"] += 1
            continue
        ex = {}
        for b in L.BINS:
            e = S.excerpt_for_bin(paras, ptok, b)
            if e:
                ex[b] = e
        if not ex:
            by_domain[dom]["drop_no_bin_fits"] += 1
            continue
        joined = " ".join(paras)[:S.QUALITY_CHARS]
        ok, why = H.text_quality_hard(joined, L.split_sentences(joined))
        if not ok:
            by_domain[dom][f"drop_{why}"] += 1
            continue
        seen_hash[h] = r["id"]
        by_domain[dom]["candidates"] += 1
        for b in ex:
            by_domain[dom][f"candidates_bin{b}"] += 1
        cands.append({**{k: r.get(k, "") for k in ("id", "source", "licence", "licence_class", "licence_evidence", "url", "title", "attribution", "date", "subdomain")},
                      "domain": dom, "group": grp, "source": grp, "paras": paras, "para_tokens": ptok, "n_tokens_full": sum(ptok), "excerpts": ex, "pool_file": r["_pool_file"],
                      "licence_note": r.get("licence_note", ""),
                      "contam": ({k: fl.get(k) for k in ("flag", "sft_share", "rl_share", "bench_share", "dup_share")} if fl else None)})
    return cands, {"overall": dict(funnel), "by_domain": {d: dict(c) for d, c in by_domain.items()}, "pool_by_domain_source": dict(sorted(pool_by_source.items()))}


def _hkey(seed: int, did: str, b: int) -> str:
    return hashlib.sha1(f"{seed}|{b}|{did}".encode()).hexdigest()


def select(cands: Sequence[Dict[str, Any]], n: int, *, seed: int = 0, source_share: float = 0.35, bin_shares: Optional[Dict[int, float]] = None) -> Dict[str, Any]:
    """Stratified selection.  Bins from the longest to the shortest; inside a (domain, bin) cell the candidates are ordered 'hard features first, seeded hash', then taken round-robin
    over sources, with <= `source_share` of the DOMAIN total per source.  A shortfall of a cell moves to the next smaller bin of the same domain."""
    domains = sorted({c["domain"] for c in cands})
    q = hard_quotas(n, domains, bin_shares)
    per_domain_total = {d: sum(v for (dd, _), v in q.items() if dd == d) for d in domains}
    cap = {d: max(1, int(source_share * per_domain_total[d])) for d in domains}
    used: set = set()
    per_src: Dict[Tuple[str, str], int] = Counter()
    chosen: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    shortfall: Dict[Tuple[str, int], int] = {}
    carry: Dict[str, int] = defaultdict(int)
    for b in sorted(L.BINS, reverse=True):
        for d in domains:
            want = q.get((d, b), 0) + carry[d]
            carry[d] = 0
            pool = [c for c in cands if c["domain"] == d and b in c["excerpts"] and c["id"] not in used]
            take: List[Dict[str, Any]] = []
            for tier in (True, False):                     # pass 1: documents with the domain features of the split (numerals / legal references / formulas); pass 2: the rest
                by: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
                for c in sorted([c for c in pool if bool(c.get("hard_ok")) == tier], key=lambda c: _hkey(seed, c["id"], b)):
                    by[c["group"]].append(c)
                order = sorted(by, key=lambda s: _hkey(seed, s, b))
                progress = True
                while len(take) < want and progress:
                    progress = False
                    for s in order:
                        if len(take) >= want:
                            break
                        if per_src[(d, s)] >= cap[d]:
                            by[s] = []
                        if by[s]:
                            c = by[s].pop(0)
                            per_src[(d, s)] += 1
                            used.add(c["id"])
                            take.append(c)
                            progress = True
            chosen[(d, b)] = take
            if len(take) < want:
                carry[d] = want - len(take)
                shortfall[(d, b)] = want - len(take)
    final_short = dict(carry)
    sel = [(c, b) for (d, b), cs in sorted(chosen.items()) for c in cs]
    return {"selected": sel, "quota": q, "shortfall": shortfall, "unfilled_after_last_bin": {d: v for d, v in final_short.items() if v}, "source_cap": cap}


def doc_row(c: Dict[str, Any], b: int) -> Dict[str, Any]:
    k, ntok = c["excerpts"][b]
    paras = c["paras"][:k]
    text = "\n\n".join(paras)
    sents = L.canonical_sentences(text)
    row = {"canary": CANARY, "doc_id": c["id"], "domain": c["domain"], "subdomain": c.get("subdomain", "") or c["source"], "source": c["source"], "licence": c["licence"],
           "licence_class": c["licence_class"], "licence_evidence": c.get("licence_evidence", ""), "url": c["url"], "title": c["title"], "attribution": c["attribution"],
           "date": c.get("date", ""), "text": text, "sents": sents, "para_starts": L.paragraph_starts(text, sents), "bin": b, "n_tokens": ntok, "n_tokens_full": c["n_tokens_full"],
           "excerpt": k < len(c["paras"]), "n_paragraphs": k, "n_sentences": len(sents), "contam": c["contam"], "subset": "hard"}
    row.update(hardness(c["domain"], text))
    if c.get("licence_note"):
        row["licence_note"] = c["licence_note"]
    return row


def summarise(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    toks = [r["n_tokens"] for r in rows]
    comp: Dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        comp[f"{r['domain']}|{r['bin']}"][r["source"]] += 1
    return {"n": len(rows), "tokens_total": sum(toks), "tokens_mean": (sum(toks) / len(toks)) if toks else 0, "sentences_total": sum(r["n_sentences"] for r in rows),
            "by_domain": dict(sorted(Counter(r["domain"] for r in rows).items())), "by_bin": dict(sorted(Counter(r["bin"] for r in rows).items())),
            "by_domain_bin": {f"{d}|{b}": n for (d, b), n in sorted(Counter((r["domain"], r["bin"]) for r in rows).items())},
            "by_domain_source": {d: dict(sorted(Counter(r["source"] for r in rows if r["domain"] == d).items(), key=lambda kv: -kv[1])) for d in sorted({r["domain"] for r in rows})},
            "by_domain_licence_class": {d: dict(sorted(Counter(r["licence_class"] for r in rows if r["domain"] == d).items())) for d in sorted({r["domain"] for r in rows})},
            "by_licence": dict(sorted(Counter(r["licence"] for r in rows).items(), key=lambda kv: -kv[1])),
            "by_licence_class": dict(sorted(Counter(r["licence_class"] for r in rows).items())),
            "hard_features_share": {d: round(sum(1 for r in rows if r["domain"] == d and r["hard_features"]) / max(1, sum(1 for r in rows if r["domain"] == d)), 3) for d in sorted({r["domain"] for r in rows})},
            "docs_with_formulas": {d: sum(1 for r in rows if r["domain"] == d and r["n_formulas"] > 0) for d in sorted({r["domain"] for r in rows})},
            "composition_domain_bin_source": {k: dict(v) for k, v in sorted(comp.items())}}


def examples_markdown(rows: Sequence[Dict[str, Any]], per_domain: int = 2, words: int = 60, seed: int = 0) -> str:
    out = []
    for dom in H.DOMAINS:
        rs = sorted([r for r in rows if r["domain"] == dom and r["split"] == "public"], key=lambda r: hashlib.sha1(f"{seed}|{r['doc_id']}".encode()).hexdigest())
        picked, srcs = [], set()
        for r in rs:
            if r["source"] not in srcs and len(picked) < per_domain and r["hard_features"]:
                picked.append(r); srcs.add(r["source"])
        out += [f"#### {dom}", ""]
        for r in picked:
            t = r["text"]
            fs = H.find_formulas(t)
            start = 0
            if fs:                                        # show the first formula: begin ~15 words before it
                pre = t[: fs[0][0]].split()
                start = max(0, len(pre) - 15)
            w = t.split()
            shown = " ".join(w[start:start + words])
            out += [f"* `{r['doc_id']}` | bin {r['bin']} ({r['n_tokens']} Gemma tokens) | {r['licence']} | {r['attribution'][:140]} | {r['url']}",
                    f"  > {'... ' if start else ''}{shown}{' ...' if start + words < len(w) else ''}", ""]
    return "\n".join(out)


def write_all(rows: Sequence[Dict[str, Any]], out: str) -> None:
    pub = [r for r in rows if r["split"] == "public"]
    priv = [r for r in rows if r["split"] == "private"]
    L.write_jsonl(os.path.join(out, "docs.jsonl"), pub)
    L.write_jsonl(os.path.join(out, "docs_private.jsonl"), priv)
    L.write_jsonl(os.path.join(out, "docs_master_do_not_publish.jsonl"), rows)
    att = lambda rs: [{"doc_id": r["doc_id"], "title": r["title"], "attribution": r["attribution"], "url": r["url"], "licence": r["licence"], "licence_class": r["licence_class"],
                       "licence_evidence": r["licence_evidence"], "source": r["source"], "excerpt": r["excerpt"]} for r in rs]
    L.write_jsonl(os.path.join(out, "ATTRIBUTION.jsonl"), att(pub))
    L.write_jsonl(os.path.join(out, "ATTRIBUTION_private.jsonl"), att(priv))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool-dir", required=True)
    ap.add_argument("--stage-dir", default=None, help="second directory with hard_*.jsonl files not yet in the pool dir")
    ap.add_argument("--contam-dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--private-share", type=float, default=0.10)
    ap.add_argument("--source-share", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-weak", action="store_true")
    ap.add_argument("--no-require-flags", action="store_true", help="NOT for a release")
    ap.add_argument("--exclude-urls-file", nargs="*", default=["/workspace/sft_build/gemma/rl_prompts_b2048_doc_ids.txt"])
    ap.add_argument("--core-docs", nargs="*", default=["/workspace/sft_build/longdoc_bench/source_set/core/docs.jsonl"], help="docs.jsonl of the core: its doc_ids are never used here")
    ap.add_argument("--tokenizer", default=None)
    a = ap.parse_args(argv)
    tok = sorted(glob.glob(a.tokenizer))[0] if a.tokenizer and "*" in a.tokenizer else a.tokenizer
    count = L.load_token_counter(tok)
    pool = read_hard_pool([a.pool_dir, a.stage_dir])
    flags = S.read_flags(a.contam_dir)
    excluded = set()
    for p in a.exclude_urls_file or []:
        excluded |= S.read_excluded_urls(p)
    core_ids = set()
    for p in a.core_docs or []:
        if os.path.isfile(p):
            core_ids |= {r["doc_id"] for r in L.read_jsonl(p)}
    cands, funnel = build_candidates(pool, flags, count, allow_weak=a.allow_weak, require_flags=not a.no_require_flags, excluded_urls=excluded, excluded_ids=core_ids)
    # hard-feature flag on the shortest excerpt available (cheap proxy; the final row carries the exact features of the chosen excerpt)
    for c in cands:
        k = c["excerpts"][min(c["excerpts"])][0]
        c["hard_ok"] = hardness(c["domain"], " ".join(c["paras"][:k]))["hard_features"]
    sel = select(cands, a.n, seed=a.seed, source_share=a.source_share)
    chosen = sel["selected"]
    priv = S.assign_private(chosen, a.private_share, a.seed)
    rows = []
    for c, b in chosen:
        r = doc_row(c, b)
        r["split"] = "private" if c["id"] in priv else "public"
        rows.append(r)
    os.makedirs(a.out, exist_ok=True)
    write_all(rows, a.out)
    H.save_config(os.path.join(a.out, "metric_config.json"))
    with open(os.path.join(a.out, "examples.md"), "w", encoding="utf-8") as f:
        f.write(examples_markdown(rows))
    cand_cells = Counter()
    for c in cands:
        for b in c["excerpts"]:
            cand_cells[f"{c['domain']}|{b}"] += 1
    cand_src = Counter((c["domain"], c["group"]) for c in cands)
    rep = {"funnel": funnel, "candidate_cells": dict(sorted(cand_cells.items())), "candidates_by_domain_source": {f"{d}|{s}": n for (d, s), n in sorted(cand_src.items())},
           "quota": {f"{d}|{b}": n for (d, b), n in sorted(sel["quota"].items())}, "shortfall_cells": {f"{d}|{b}": n for (d, b), n in sorted(sel["shortfall"].items())},
           "unfilled_after_last_bin": sel["unfilled_after_last_bin"], "source_cap": sel["source_cap"], "set": summarise(rows), "public": summarise([r for r in rows if r["split"] == "public"]),
           "private": summarise([r for r in rows if r["split"] == "private"]), "n": a.n, "seed": a.seed, "private_share": a.private_share, "source_share": a.source_share,
           "allow_weak": a.allow_weak, "excluded_urls": len(excluded), "contam_flags_loaded": len(flags), "core_ids_excluded": len(core_ids),
           "contamination_of_selected": dict(Counter((r["contam"] or {}).get("flag", "none") for r in rows))}
    with open(os.path.join(a.out, "SELECTION_REPORT.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print(json.dumps({"funnel_by_domain": funnel["by_domain"], "n": rep["set"]["n"], "private": rep["private"]["n"], "by_domain_bin": rep["set"]["by_domain_bin"],
                      "shortfall": rep["shortfall_cells"], "unfilled": rep["unfilled_after_last_bin"], "by_domain_source": rep["set"]["by_domain_source"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
