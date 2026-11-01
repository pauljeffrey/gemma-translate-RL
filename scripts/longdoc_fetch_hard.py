#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fetchers of the HARD / DOMAIN split (medicine, law, maths-science) of the long-document benchmark.  They write pool files `hard_<source>__<class>.jsonl` with the
row schema of the core pool (id, source, domain, licence, licence_class, licence_evidence, url, title, attribution, date, text, n_chars, n_gemma_tokens, extra) plus
`subdomain`, `domain` in {medicine, law, math_science}.  Any data licence is accepted (the owner decision of 2026-10-20: research / education evaluation only) but
every row keeps its licence name, class and evidence URL.  Unlike the core fetcher, formulas, numbers with units, symbols, citation markers and section numbers STAY;
URLs, e-mail addresses and telephone numbers are removed (`longdoc_hard.clean_hard_paragraph`).

    python scripts/longdoc_fetch_hard.py list
    python scripts/longdoc_fetch_hard.py arxiv      --out pool_stage/hard_arxiv__permissive.jsonl --max-docs 180
    python scripts/longdoc_fetch_hard.py caselaw    --out ...                 (Common Pile / Caselaw Access Project, public domain)
    python scripts/longdoc_fetch_hard.py eurlex     --out ...                 (joelniklaus/eurlex_resources, EU legal acts, CC BY 4.0 reuse)
    python scripts/longdoc_fetch_hard.py cuad       --out ...                 (CUAD v1 contracts from SEC EDGAR exhibits, CC BY 4.0)
    python scripts/longdoc_fetch_hard.py wikipedia  --out ...                 (mathematics / physics articles with formulas as LaTeX, CC BY-SA 4.0)
    python scripts/longdoc_fetch_hard.py europepmc  --out ...                 (PubMed Central CC BY-NC(-ND/-SA) articles, via Europe PMC)
    python scripts/longdoc_fetch_hard.py who        --out ...                 (WHO fact sheets, CC BY-NC-SA 3.0 IGO)

Polite crawling: User-Agent with contact, robots.txt honoured, >= 1 s between requests to one host (`Http` of longdoc_fetch_sources.py).  Hub data are streamed, nothing
is stored besides the pool file.  Downloaded content is untrusted data: it is parsed, never executed.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import lzma
import os
import random
import re
import sys
import time
from collections import Counter, OrderedDict
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from gemma_mt.eval import longdoc_hard as H  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import longdoc_fetch_sources as F  # noqa: E402

MIN_TOKENS, MAX_TOKENS = 420, 5200


# ------------------------------------------------------------------------------------------------------------------------ pool writer
def best_run(paras: List[str], tc, min_tokens: int) -> List[str]:
    """Keep running prose coherent: the first run of consecutive acceptable paragraphs that reaches `min_tokens`, else the longest run (None entries = rejected paragraphs)."""
    runs, cur = [], []
    for p in paras + [None]:
        if p is None:
            if cur:
                runs.append(cur)
            cur = []
        else:
            cur.append(p)
    best = []
    for r in runs:
        if sum(tc.count_many(r)) >= min_tokens:
            return r
        if len(r) > len(best):
            best = r
    return best


def merge_display(paras: Sequence[str]) -> List[str]:
    """A paragraph that is only a formula, or that continues a sentence after a display formula (starts in lower case: 'where ...'), is joined to the previous paragraph."""
    out: List[str] = []
    for p in paras:
        p = p.strip()
        if not p:
            continue
        only_formula = bool(H.find_formulas(p)) and len(H.mask_formulas(p).split()) == 0
        cont = out and out[-1].rstrip().endswith(("$$", "\\]", "$")) and p[:1].islower()
        if out and (only_formula or cont):
            out[-1] = out[-1].rstrip() + " " + p
        elif only_formula:
            continue
        else:
            out.append(p)
    return out


class HardPool:
    def __init__(self, out: str, source: str, tc, max_docs: int, min_tokens: int = MIN_TOKENS, max_tokens: int = MAX_TOKENS):
        self.out, self.part, self.source, self.tc, self.max_docs = out, out + ".part", source, tc, max_docs
        self.min_tokens, self.max_tokens = min_tokens, max_tokens
        self.ids, self.hashes, self.n, self.stats = set(), set(), 0, Counter()
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        if os.path.exists(self.part):
            for line in open(self.part, encoding="utf-8"):
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                self.ids.add(d["id"]); self.hashes.add(F.fingerprint(d["text"])); self.n += 1
        self.fh = open(self.part, "a", encoding="utf-8")

    @property
    def full(self) -> bool:
        return self.n >= self.max_docs

    def seen(self, did: str) -> bool:
        return did in self.ids

    def add(self, d: Dict[str, Any]) -> bool:
        did = d["id"]
        if did in self.ids:
            return False
        paras = d.pop("paragraphs", None)
        if paras is None:
            paras = [p for p in re.split(r"\n\s*\n", d.pop("text")) if p.strip()]
        paras = merge_display([H.clean_hard_paragraph(p) for p in paras])
        flags = []
        for p in paras:
            ok, why = H.paragraph_ok_hard(p)
            flags.append(p if ok else None)
            if not ok:
                self.stats["para_" + why] += 1
        paras = best_run(flags, self.tc, self.min_tokens)
        paras = list(OrderedDict.fromkeys(paras))
        if len(paras) < 3:
            self.stats["few_paragraphs"] += 1
            return False
        paras, truncated = F.cut_to_tokens(paras, self.tc, self.max_tokens)
        text = "\n\n".join(paras)
        n_tok = self.tc.count(text)
        if n_tok < self.min_tokens:
            self.stats["too_short"] += 1
            return False
        sents = H.L.split_sentences(text[:40000])
        ok, why = H.text_quality_hard(text[:40000], sents)
        if not ok:
            self.stats["doc_" + why] += 1
            return False
        h = F.fingerprint(text)
        if h in self.hashes:
            self.stats["duplicate"] += 1
            return False
        rec = OrderedDict()
        rec["id"] = did
        rec["source"] = d.get("source", self.source)
        for k in ("domain", "subdomain", "licence", "licence_class", "licence_evidence", "url", "title", "attribution", "date"):
            rec[k] = d.get(k, "") or ""
        assert rec["domain"] in H.DOMAINS, rec["domain"]
        assert rec["licence_class"] in F.LICENCE_CLASSES, rec["licence_class"]
        rec["text"] = text
        rec["n_chars"] = len(text)
        rec["n_gemma_tokens"] = n_tok
        rec.update(H.formula_density(text))
        if truncated:
            rec["truncated"] = True
        for k in ("licence_note", "extra"):
            if d.get(k):
                rec[k] = d[k]
        self.fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.fh.flush()
        self.ids.add(did); self.hashes.add(h); self.n += 1; self.stats["accepted"] += 1
        return True

    def close(self, complete: bool = True):
        self.fh.close()
        if complete and os.path.exists(self.part):
            os.replace(self.part, self.out)


class Ctx:
    def __init__(self, args, http, pool):
        self.args, self.http, self.pool = args, http, pool

    def log(self, *a):
        print(f"[{self.pool.source}]", *a, file=sys.stderr, flush=True)


SOURCES: Dict[str, Dict[str, Any]] = {}


def source(name: str, domain: str, licence: str, cls: str):
    def deco(fn):
        SOURCES[name] = dict(fn=fn, domain=domain, licence=licence, cls=cls, doc=(fn.__doc__ or "").strip())
        return fn
    return deco


# ------------------------------------------------------------------------------------------------------------------------ markdown (pandoc) blocks of arXiv
_MD_IMG = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\((?:[^)]*)\)")
_MD_EMPH = re.compile(r"(?<![\w$\\])(\*\*|__|\*)(?=\S)(.+?)(?<=\S)\1(?![\w$])")


def arxiv_blocks(text: str) -> List[str]:
    """Paragraph strings of the pandoc-markdown text of an arXiv paper, from the abstract up to the references: headings, tables, figure lines, lists of fragments dropped,
    display formulas kept (merged to the paragraph by `merge_display`)."""
    out: List[str] = []
    started = False
    for blk in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
        b = blk.strip()
        if not b:
            continue
        if re.match(r"^#{1,6}\s*(?:\d+(?:\.\d+)*\.?\s*)?(?:references|bibliography|acknowledg|appendix|supplementary)", b, re.I):
            break
        if re.match(r"^#{1,6}\s*abstract\b", b, re.I) or re.match(r"^abstract[.:]?\s*$", b, re.I):
            started = True
            out = []
            continue
        if b.startswith("#"):
            if re.match(r"^#{1,6}\s*(?:\d+\.?\s*)?(?:introduction|overview|background)", b, re.I):
                started = True
            continue
        if re.match(r"^(?:\||!\[|<|\\begin\{(?:tabular|figure|table)|Table\s+\d|Figure\s+\d|Fig\.\s*\d|\[\d+\]\s+[A-Z]|Keywords|Key words|PACS|MSC|\\)", b) and not b.startswith("$$"):
            continue
        b = _MD_IMG.sub("", b)
        b = _MD_LINK.sub(r"\1", b)
        b = re.sub(r"-\n(?=[a-z])", "", b)                  # hyphenated line breaks
        b = " ".join(x.strip() for x in b.split("\n"))
        b = _MD_EMPH.sub(r"\2", b)
        b = re.sub(r"\{#[^}]*\}|\\label\{[^}]*\}", "", b)
        b = re.sub(r"\s+", " ", b).strip()
        if not b:
            continue
        if not started and len(b.split()) < 40 and not b.startswith("$$"):
            continue                                        # title / author block / affiliations
        started = True
        out.append(b)
    return out


@source("arxiv", "math_science", "CC BY 4.0 / CC BY 3.0 / CC0 (per paper)", "permissive")
def src_arxiv(ctx: Ctx):
    """arXiv full-text papers under CC BY / CC0 (Common Pile `arxiv_papers`, pandoc markdown with $LaTeX$ kept): abstract + introduction prose."""
    a = ctx.args
    repo = "common-pile/arxiv_papers"
    shards = F.cp_shards(repo, "", ".jsonl.gz")
    random.Random(a.seed).shuffle(shards)
    for sh in shards:
        if ctx.pool.full:
            break
        n = 0
        for rec in F.cp_stream(repo, sh):
            n += 1
            if n > a.scan_limit or ctx.pool.full:
                break
            md = rec.get("metadata") or {}
            lic = F.cp_licence(md.get("license", ""))
            text = rec.get("text") or ""
            if not lic or len(text) < 15000 or random.Random(f"{a.seed}-{rec.get('id')}").random() > a.sample:
                continue
            tl = next((l for l in text.split("\n")[:12] if l.startswith("#") and len(l.strip("# ")) > 8), "")
            title = re.sub(r"\s+", " ", re.sub(r"^#+\s*", "", tl)).strip()[:200] or f"arXiv:{rec['id']}"
            ok = ctx.pool.add(dict(
                id=f"hard_arxiv:{rec['id']}", domain="math_science", subdomain="arxiv_paper", licence=lic[0], licence_class=lic[1], licence_evidence=lic[2],
                url=md.get("url") or f"https://arxiv.org/abs/{rec['id']}", title=title, attribution=f"{md.get('authors', '')}, {title}, arXiv:{rec['id']}, {lic[0]}",
                date=(rec.get("created") or "")[:10], paragraphs=arxiv_blocks(text),
                licence_note="per-paper licence read by the Common Pile from the arXiv metadata; equations and inline LaTeX are kept as $...$ / $$...$$ on purpose (the point of the split)"))
        ctx.log(os.path.basename(sh), "docs", ctx.pool.n, dict(ctx.pool.stats))


# ------------------------------------------------------------------------------------------------------------------------ law
def plain_law_blocks(text: str) -> List[str]:
    blocks = []
    for blk in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
        lines = [l.strip() for l in blk.split("\n") if l.strip()]
        if lines:
            blocks.append(" ".join(lines))
    return blocks


@source("caselaw", "law", "Public Domain (Caselaw Access Project, Harvard LIL / Free Law Project)", "pd")
def src_caselaw(ctx: Ctx):
    """US court opinions (Caselaw Access Project via Common Pile `caselaw_access_project`; the party names are those of the published opinion)."""
    a = ctx.args
    repo = "common-pile/caselaw_access_project"
    shards = F.cp_shards(repo, "", ".jsonl.gz")
    random.Random(a.seed).shuffle(shards)
    for sh in shards[: a.max_shards]:
        if ctx.pool.full:
            break
        n = 0
        for rec in F.cp_stream(repo, sh):
            n += 1
            if n > a.scan_limit or ctx.pool.full:
                break
            text = rec.get("text") or ""
            md = rec.get("metadata") or {}
            if len(text) < 9000 or random.Random(f"{a.seed}-{rec.get('id')}").random() > a.sample:
                continue
            blocks = plain_law_blocks(text)
            title = (text.strip().split("\n", 1)[0].strip())[:160]
            i = next((k for k, b in enumerate(blocks) if len(b.split()) >= 40), None)
            if i is None:
                continue
            paras = blocks[i:]
            ok = ctx.pool.add(dict(
                id=f"hard_caselaw:{hashlib.sha1(str(rec.get('id')).encode()).hexdigest()[:14]}", domain="law", subdomain="court_opinion", licence="Public Domain / CC0 (Caselaw Access Project)",
                licence_class="pd", licence_evidence="https://case.law/about/ (CAP data: CC0 for case text; Common Pile field 'Public Domain')", url=md.get("url") or "https://static.case.law/",
                title=title, attribution=f"{title}, Caselaw Access Project (Harvard Law School Library Innovation Lab), record {rec.get('id')}", date=(rec.get("created") or "")[:10],
                paragraphs=paras, licence_note="court opinion text is public domain; the Common Pile record carries no court or date field, the id is the CAP static path",
                extra={"cap_id": rec.get("id")}))
        ctx.log(os.path.basename(sh), "docs", ctx.pool.n, dict(ctx.pool.stats))


@source("eurlex", "law", "CC BY 4.0 (EU reuse policy; dataset card of joelniklaus/eurlex_resources: cc-by-4.0)", "permissive")
def src_eurlex(ctx: Ctx):
    """EU regulations, directives, decisions (EUR-Lex via the Hub mirror joelniklaus/eurlex_resources, English): recitals and articles, one clause per paragraph."""
    from huggingface_hub import HfFileSystem
    a = ctx.args
    fs = HfFileSystem()
    kinds = ["regulation", "directive", "decision", "recommendation"]
    per_kind = max(5, a.max_docs // len(kinds) + 5)
    for kind in kinds:
        got = 0
        with fs.open(f"datasets/joelniklaus/eurlex_resources/data/en/{kind}.jsonl.xz", "rb") as raw:
            g = lzma.open(raw, "rt", encoding="utf-8")
            n = 0
            for line in g:
                n += 1
                if n > a.scan_limit or got >= per_kind or ctx.pool.full:
                    break
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                text = r.get("text") or ""
                if not (7000 <= len(text) <= 60000) or random.Random(f"{a.seed}-{r.get('celex')}").random() > a.sample:
                    continue
                if re.search(r"Date of birth|Place of birth|National identification|Passport no|Other information:|Nationality:", text):
                    continue                                   # sanctions lists name individuals: personal data
                paras = [re.sub(r"\s+", " ", l).strip() for l in text.split("\n")]
                paras = [p for p in paras if len(p.split()) >= 6]
                i = next((k for k, p in enumerate(paras) if re.match(r"(?:THE |The )(?:EUROPEAN|COUNCIL|COMMISSION)|Having regard|Whereas", p)), 0)
                celex = r.get("celex")
                if a.keep_title and r.get("title"):
                    pass
                if ctx.pool.add(dict(
                        id=f"hard_eurlex:{celex}", domain="law", subdomain=f"eu_{kind}", licence="CC-BY-4.0 (EU reuse)", licence_class="permissive",
                        licence_evidence="https://huggingface.co/datasets/joelniklaus/eurlex_resources (card: license cc-by-4.0); EU reuse policy Decision 2011/833/EU",
                        url=f"https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{celex}", title=(r.get("title") or "")[:200],
                        attribution=f"{r.get('title', '')}, CELEX {celex}, {r.get('date', '')}, EUR-Lex (Publications Office of the European Union), CC BY 4.0",
                        date=r.get("date", ""), paragraphs=paras[i:], licence_note="EUR-Lex text is reusable under Decision 2011/833/EU; the legal-notice page is behind a WAF from the rental, "
                                                                          "the licence is quoted from the Hub dataset card",
                        extra={"celex": celex, "kind": kind})):
                    got += 1
        ctx.log(kind, "docs", ctx.pool.n, dict(ctx.pool.stats))


@source("cuad", "law", "CC BY 4.0 (Contract Understanding Atticus Dataset v1); contracts are public SEC EDGAR exhibits", "permissive")
def src_cuad(ctx: Ctx):
    """Commercial contracts (CUAD v1 full_contract_txt, exhibits of SEC EDGAR filings): a window of the operative clauses."""
    from huggingface_hub import HfFileSystem
    a = ctx.args
    fs = HfFileSystem()
    files = []
    for part in ("Part_I", "Part_II", "Part_III"):
        try:
            files += [x for x in fs.ls(f"datasets/theatticusproject/cuad/CUAD_v1/full_contract_txt/{part}", detail=False) if x.endswith(".txt")]
        except Exception as e:
            ctx.log("ls", part, str(e)[:80])
    random.Random(a.seed).shuffle(files)
    ctx.log("contracts listed", len(files))
    for f in files:
        if ctx.pool.full:
            break
        try:
            text = fs.open(f, "rb").read().decode("utf-8", errors="replace")
        except Exception:
            continue
        name = os.path.basename(f)[:-4]
        blocks = plain_law_blocks(text)
        # operative text starts after the recitals / definitions heading: skip the first 10% (cover page, table of contents)
        paras = [b for b in blocks[max(3, len(blocks) // 12):] if len(b.split()) >= 12]
        while paras and re.match(r"^\(?[a-z0-9ivx]{1,4}[).]\s", paras[0]) and not re.match(r"^\(?1[).]", paras[0]):
            paras = paras[1:]                                  # start at a clause, not in the middle of an enumeration
        ctx.pool.add(dict(
            id="hard_cuad:" + hashlib.sha1(name.encode()).hexdigest()[:14], domain="law", subdomain="contract", licence="CC-BY-4.0 (CUAD v1)", licence_class="permissive",
            licence_evidence="https://huggingface.co/datasets/theatticusproject/cuad (card: license cc-by-4.0); https://www.atticusprojectai.org/cuad", url="https://www.atticusprojectai.org/cuad",
            title=name[:160], attribution=f"{name}; Hendrycks, Burns, Chen, Ball (2021), CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review, NeurIPS Datasets and Benchmarks; "
                                          "contract text from public SEC EDGAR filings; CC BY 4.0", date="", paragraphs=paras,
            licence_note="the contract text is a public SEC filing; the annotated dataset is CC BY 4.0", extra={"file": name}))
    ctx.log("docs", ctx.pool.n, dict(ctx.pool.stats))


# ------------------------------------------------------------------------------------------------------------------------ Wikipedia mathematics / physics
WP_CATS = ["Theorems in analysis", "Theorems in algebra", "Theorems in geometry", "Theorems in number theory", "Theorems in probability theory", "Theorems in statistics",
           "Theorems in calculus", "Theorems in linear algebra", "Equations of physics", "Concepts in physics", "Quantum mechanics", "Classical mechanics", "Thermodynamics",
           "Electromagnetism", "Special relativity", "General relativity", "Statistical mechanics", "Linear algebra", "Calculus", "Differential equations", "Fourier analysis",
           "Probability distributions", "Probability theory", "Numerical analysis", "Optimization algorithms and methods", "Mathematical analysis", "Group theory",
           "Mathematical physics", "Fluid dynamics", "Optics", "Orbital mechanics", "Combinatorics", "Graph theory", "Topology", "Number theory", "Cryptography", "Information theory",
           "Machine learning algorithms", "Statistical tests", "Regression analysis"]


def brace_group(s: str, i: int) -> int:
    """Index after the brace group that starts at s[i] == '{' (nesting aware), or -1."""
    depth = 0
    for j in range(i, len(s)):
        c = s[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return j + 1
    return -1


def wp_extract_to_paras(ex: str) -> List[str]:
    """Wikipedia plain-text extract (explaintext) -> paragraphs.  The extract spells every formula twice: first the rendered symbols, one per indented line, then
    `{\\displaystyle ...}` (the LaTeX).  The indented spill is dropped and `{\\displaystyle X}` becomes `$X$` (a paragraph that is only a formula becomes `$$X$$`)."""
    marker = "{\\displaystyle"
    out_parts: List[str] = []
    ex = ex.replace("\u2060", "")
    i = 0
    while True:
        j = ex.find(marker, i)
        if j < 0:
            out_parts.append(ex[i:])
            break
        k = brace_group(ex, j)
        if k < 0:
            out_parts.append(ex[i:])
            break
        lines = ex[i:j].split("\n")
        while len(lines) > 1 and lines[-1] != "" and (lines[-1].strip() == "" or lines[-1].startswith((" ", "\t"))):
            lines.pop()                                         # the rendered-symbol spill before the LaTeX
        out_parts.append("\n".join(lines))
        inner = re.sub(r"\s+", " ", ex[j + len(marker):k - 1]).strip()
        out_parts.append("\u0001" + inner + "\u0002")
        rest = ex[k:]
        m = re.match(r"(?:[ \t]*\n){1,3}[ \t]*", rest)                  # blank / indented lines after the formula: the sentence goes on
        if m and re.search(r"[ \t]", m.group(0)):
            k += m.end()
            out_parts.append(" ")
        i = k
    s = "".join(out_parts).replace("\u2060", "")
    lines = []
    for ln in s.split("\n"):
        t = ln.strip()
        if not t or re.match(r"^=+\s.*\s=+$", t):
            continue
        m = re.fullmatch(r"\u0001(.*)\u0002[.,;]?", t)
        if m:
            lines.append("$$" + m.group(1) + "$$")
            continue
        lines.append(re.sub(r"\u0001(.*?)\u0002", lambda mm: "$" + mm.group(1) + "$", t))
    return lines


@source("wikipedia", "math_science", "CC BY-SA 4.0 (Wikipedia contributors)", "share_alike")
def src_wikipedia(ctx: Ctx):
    """English Wikipedia mathematics / physics / statistics articles fetched through the MediaWiki API (plain-text extract with formulas as LaTeX)."""
    a = ctx.args
    api = "https://en.wikipedia.org/w/api.php"
    titles: List[str] = []
    for cat in WP_CATS:
        try:
            j = ctx.http.json(api, params={"action": "query", "list": "categorymembers", "cmtitle": f"Category:{cat}", "cmtype": "page", "cmlimit": "60", "format": "json"})
        except Exception as e:
            ctx.log("cat", cat, str(e)[:80])
            continue
        titles += [m["title"] for m in j.get("query", {}).get("categorymembers", []) if ":" not in m["title"] and not m["title"].startswith("List of")]
    titles = sorted(set(titles))
    random.Random(a.seed).shuffle(titles)
    ctx.log("titles", len(titles))
    for t in titles:
        if ctx.pool.full:
            break
        try:
            j = ctx.http.json(api, params={"action": "query", "prop": "extracts|revisions", "explaintext": 1, "exsectionformat": "plain", "rvprop": "ids|timestamp", "titles": t, "format": "json", "redirects": 1})
        except Exception as e:
            continue
        pages = j.get("query", {}).get("pages", {})
        for pid, pg in pages.items():
            ex = pg.get("extract") or ""
            rev = (pg.get("revisions") or [{}])[0]
            if len(ex) < 4000 or "{\\displaystyle" not in ex:
                continue
            title = pg.get("title", t)
            url = "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")
            ctx.pool.add(dict(
                id=f"hard_wikipedia:{pid}", domain="math_science", subdomain="wikipedia_math_physics", licence="CC-BY-SA-4.0", licence_class="share_alike",
                licence_evidence="https://en.wikipedia.org/wiki/Wikipedia:Text_of_the_Creative_Commons_Attribution-ShareAlike_4.0_International_License; https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use",
                url=url, title=title, attribution=f"Wikipedia contributors, \"{title}\", Wikipedia, The Free Encyclopedia, {url} (revision {rev.get('revid', '')}, {rev.get('timestamp', '')[:10]}), CC BY-SA 4.0",
                date=(rev.get("timestamp") or "")[:10], paragraphs=wp_extract_to_paras(ex), licence_note="share-alike: derivative works (translations, score tables of this article) must carry CC BY-SA 4.0",
                extra={"revid": rev.get("revid"), "category_seed": True}))
    ctx.log("docs", ctx.pool.n, dict(ctx.pool.stats))


# ------------------------------------------------------------------------------------------------------------------------ medicine
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
MED_TOPICS = ["randomized controlled trial AND treatment outcome", "clinical practice guideline recommendations", "diagnosis AND imaging", "cardiovascular disease AND risk",
              "type 2 diabetes AND management", "breast cancer AND chemotherapy", "antibiotic AND resistance AND infection", "asthma OR COPD AND therapy", "chronic kidney disease",
              "stroke AND rehabilitation", "pregnancy AND maternal outcome", "pediatric AND vaccination", "depression AND treatment", "HIV AND antiretroviral", "tuberculosis AND treatment",
              "malaria AND antimalarial", "surgery AND complications", "hypertension AND blood pressure", "pharmacokinetics AND dose", "anesthesia OR analgesia", "COVID-19 AND clinical",
              "obesity AND intervention", "dementia AND care", "sepsis AND critical care", "liver disease AND hepatitis"]
_SKIP_SEC = re.compile(r"method|materials|statistic|data (?:collection|analysis)|supplement|ethic|acknowledg|funding|competing|conflict|author|abbreviation|reference|appendix|availability|consent|"
                       r"participants and|study design|search strategy|sampling|instrument|procedure", re.I)


def jats_paragraphs_keep(xml_text: str) -> Tuple[str, str, List[str], str]:
    """(title, licence url, paragraphs, doi): abstract + introduction + results + discussion + conclusion prose of a JATS article.  Bibliographic citation markers are kept as
    '[3]'; figure / table pointers stay as text; a paragraph with an equation (MathML) is dropped."""
    from lxml import etree
    root = etree.fromstring(xml_text.encode("utf-8"), etree.XMLParser(recover=True, huge_tree=True))
    t = root.xpath("//article-meta/title-group/article-title")
    title = " ".join("".join(t[0].itertext()).split()) if t else ""
    lic = (root.xpath("//article-meta//*[local-name()='license_ref']/text()") or root.xpath("//article-meta//license/@*[local-name()='href']") or [""])[0].strip()
    doi = (root.xpath("//article-meta/article-id[@pub-id-type='doi']/text()") or [""])[0]

    def ptext(p) -> Optional[str]:
        parts: List[str] = []
        bad = [False]

        def rec(e):
            tg = F._local(e.tag)
            if tg in ("inline-formula", "disp-formula", "math", "table-wrap", "fig", "supplementary-material"):
                if tg in ("inline-formula", "disp-formula", "math"):
                    bad[0] = True
                return
            if tg == "xref" and e.get("ref-type") == "bibr":
                tx = "".join(e.itertext()).strip()
                parts.append(tx if tx.startswith(("[", "(")) else f"[{tx}]")
                return
            if e.text:
                parts.append(e.text)
            for c in e:
                rec(c)
                if c.tail:
                    parts.append(c.tail)
        rec(p)
        return None if bad[0] else "".join(parts)

    out: List[str] = []
    for p in root.xpath("//article-meta/abstract[not(@abstract-type)]//p"):
        s = ptext(p)
        if s:
            out.append(s)
    for sec in root.xpath("//body/sec"):
        st = " ".join("".join(sec.xpath("./title")[0].itertext()).split()) if sec.xpath("./title") else ""
        if _SKIP_SEC.search(st) or _SKIP_SEC.search(sec.get("sec-type") or ""):
            continue
        for p in sec.xpath(".//p[not(ancestor::table-wrap) and not(ancestor::fig) and not(ancestor::boxed-text)]"):
            s = ptext(p)
            if s:
                out.append(s)
    return title, lic, out, doi


@source("europepmc", "medicine", "CC BY-NC / BY-NC-ND / BY-NC-SA (per article; read from the article XML)", "noncommercial")
def src_europepmc(ctx: Ctx):
    """PubMed Central open-access articles that are NOT CC BY (CC BY-NC, -NC-ND, -NC-SA) via Europe PMC: abstract, introduction, results, discussion."""
    a = ctx.args
    base = '(LICENSE:"cc by-nc" OR LICENSE:"cc by-nc-nd" OR LICENSE:"cc by-nc-sa") AND OPEN_ACCESS:y AND SRC:MED AND HAS_FT:Y AND PUB_YEAR:[2014 TO 2025] AND LANG:eng AND ('
    per_topic = max(4, a.max_docs // len(MED_TOPICS) + 3)
    got = Counter()
    cursors = {t: "*" for t in MED_TOPICS}
    active = list(MED_TOPICS)
    while active and not ctx.pool.full:
        for terms in list(active):
            if ctx.pool.full:
                break
            if got[terms] >= per_topic:
                active.remove(terms)
                continue
            try:
                j = ctx.http.json(EPMC + "/search", params={"query": base + terms + ")", "format": "json", "pageSize": 25, "resultType": "core", "cursorMark": cursors[terms]})
            except Exception as e:
                ctx.log("search", terms, str(e)[:100])
                active.remove(terms)
                continue
            res = j.get("resultList", {}).get("result", [])
            nxt = j.get("nextCursorMark")
            if not res or not nxt or nxt == cursors[terms]:
                active.remove(terms)
                continue
            cursors[terms] = nxt
            for r in res:
                pm = r.get("pmcid")
                if not pm or got[terms] >= per_topic or ctx.pool.full or ctx.pool.seen(f"hard_pmcnc:{pm}"):
                    continue
                try:
                    x = ctx.http.text(f"{EPMC}/{pm}/fullTextXML", allow_404=True)
                except Exception:
                    continue
                if not x:
                    continue
                title, lic_url, paras, doi = jats_paragraphs_keep(x)
                lic, cls = F.pmc_licence(lic_url)
                if not lic_url:                                    # no <license> link in the XML: Europe PMC's own licence field of the search result (the query selected cc by-nc*)
                    el = (r.get("license") or "").strip().lower()
                    lic, cls = ("CC-" + el.replace("cc ", "").upper() + " (version not stated; Europe PMC licence field)" if el else "CC BY-NC family (Europe PMC query)"), "noncommercial"
                ok = ctx.pool.add(dict(
                    id=f"hard_pmcnc:{pm}", domain="medicine", subdomain="pmc_article", licence=lic, licence_class=cls, licence_evidence=lic_url or f"https://europepmc.org/article/PMC/{pm.replace('PMC', '')}",
                    url=f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pm}/", title=title or r.get("title", ""),
                    attribution=f"{r.get('authorString', '')} ({r.get('pubYear', '')}). {title}. {r.get('journalTitle', '')}. {('doi:' + doi) if doi else pm}. {lic}",
                    date=(r.get("firstPublicationDate") or r.get("pubYear") or "")[:10], paragraphs=paras,
                    licence_note="licence read from the article XML; non-commercial terms: research / education use only"))
                if ok:
                    got[terms] += 1
        ctx.log("docs", ctx.pool.n, dict(ctx.pool.stats))


@source("who", "medicine", "CC BY-NC-SA 3.0 IGO (WHO, https://www.who.int/about/policies/publishing/copyright)", "share_alike")
def src_who(ctx: Ctx):
    """WHO fact sheets (English HTML pages under /news-room/fact-sheets/detail/)."""
    from lxml import html as LH
    a = ctx.args
    idx = ctx.http.text("https://www.who.int/news-room/fact-sheets")
    slugs = sorted(set(re.findall(r"/news-room/fact-sheets/detail/([a-z0-9\-]+)", idx or "")))
    random.Random(a.seed).shuffle(slugs)
    ctx.log("fact sheets", len(slugs))
    for sl in slugs:
        if ctx.pool.full:
            break
        url = f"https://www.who.int/news-room/fact-sheets/detail/{sl}"
        try:
            h = ctx.http.text(url, allow_404=True)
        except Exception:
            continue
        if not h:
            continue
        root = LH.fromstring(h)
        body = root.xpath("//*[contains(@class,'sf-detail-body-wrapper')]")
        if not body:
            continue
        paras = []
        for el in body[0].iter("p", "li"):
            tx = " ".join("".join(el.itertext()).split())
            if len(tx.split()) >= 7:
                paras.append(tx)
        t = root.xpath("//title/text()")
        title = (t[0].split("|")[0].strip() if t else sl).strip()
        date = ""
        m = re.search(r'"datePublished"\s*:\s*"(\d{4}-\d\d-\d\d)', h) or re.search(r"(\d{1,2} \w+ \d{4})", " ".join(body[0].itertext())[:400])
        if m:
            date = m.group(1)
        ctx.pool.add(dict(
            id=f"hard_who:{sl}", domain="medicine", subdomain="who_fact_sheet", licence="CC-BY-NC-SA-3.0-IGO", licence_class="noncommercial",
            licence_evidence="https://www.who.int/about/policies/publishing/copyright", url=url, title=title,
            attribution=f"World Health Organization, \"{title}\" (fact sheet), {url}, {date}; CC BY-NC-SA 3.0 IGO", date=date, paragraphs=paras,
            licence_note="WHO open-access publications: CC BY-NC-SA 3.0 IGO (non-commercial, share-alike); fact-sheet text only, no images"))
    ctx.log("docs", ctx.pool.n, dict(ctx.pool.stats))


# ------------------------------------------------------------------------------------------------------------------------ CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="source", required=True)
    sub.add_parser("list")
    for name, meta in SOURCES.items():
        sp = sub.add_parser(name, help=meta["doc"][:100])
        sp.add_argument("--out", required=True)
        sp.add_argument("--max-docs", type=int, default=200)
        sp.add_argument("--seed", type=int, default=20261020)
        sp.add_argument("--tokenizer", default=None)
        sp.add_argument("--min-interval", type=float, default=1.0)
        sp.add_argument("--scan-limit", type=int, default=4000, help="records read per Hub shard")
        sp.add_argument("--max-shards", type=int, default=6)
        sp.add_argument("--sample", type=float, default=1.0, help="keep this fraction of the candidate records (seeded hash)")
        sp.add_argument("--keep-title", action="store_true")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.source == "list":
        for n, m in SOURCES.items():
            print(f"{n:12s} {m['domain']:13s} {m['cls']:14s} {m['licence']}")
        return 0
    meta = SOURCES[args.source]
    tc = F.TokenCounter(args.tokenizer)
    if not tc.exact:
        print("[warn] no Gemma tokenizer: token counts are chars/4 estimates", file=sys.stderr)
    # api.php of Wikipedia is the sanctioned bot interface (UA with contact, >= 1 s between requests) although robots.txt disallows /w/ for web crawlers
    http = F.Http(min_interval=max(1.0, args.min_interval), respect_robots=(args.source != "wikipedia"))
    pool = HardPool(args.out, args.source, tc, args.max_docs)
    t0 = time.time()
    ok = False
    try:
        meta["fn"](Ctx(args, http, pool))
        ok = True
    finally:
        pool.close(complete=ok)
    print(json.dumps(dict(source=args.source, out=args.out, docs=pool.n, requests=http.n_requests, rejected=dict(pool.stats), seconds=round(time.time() - t0, 1)), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
