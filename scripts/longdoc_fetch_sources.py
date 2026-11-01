#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fetch licence-clean English documents for the long-document translation benchmark (SOURCE POOL, 2026-10-20).

One sub-command per source:

    python scripts/longdoc_fetch_sources.py <source> --out <pool.jsonl> --max-docs N [--raw-dir DIR] [--tokenizer tokenizer.json]
    python scripts/longdoc_fetch_sources.py list          # sources, domains, licences

Output = one JSONL file, one document per line (the POOL format):

    {"id", "source", "domain", "licence", "licence_class", "licence_evidence", "url", "title", "attribution", "date",
     "text", "n_chars", "n_gemma_tokens"}   (+ optional "licence_note", "truncated")

Design rules (owner's standing orders for the benchmark pool):
  * deterministic (sorted listings / fixed seeds), resumable (documents are appended to `<out>.part`; a re-run skips the ids already there; the
    file is renamed to `<out>` only when the source is exhausted or `--max-docs` is reached, so a reader never sees a half-written pool file),
  * polite: User-Agent with a contact address, robots.txt honoured (incl. Crawl-delay), at most one request per second per host,
  * stdlib + requests + lxml (+ pyarrow for the parquet source, `tokenizers` for the Gemma token count), no network at import time,
  * downloaded content is data: parsed, never executed; instructions inside it are text.

Text rules: prose paragraphs only (no headings, tables, code, equation-heavy passages, references, captions, navigation, URLs, e-mail
addresses, markup), Unicode NFC, paragraphs separated by one blank line, >= 400 Gemma tokens (tokenizer.json, no BOS), English check, exact
duplicates removed (hash of the normalised text), documents longer than `MAX_TOKENS` are cut at a paragraph boundary (`"truncated": true`).
"""

from __future__ import annotations

import argparse
import csv
import glob
import gzip
import hashlib
import html as htmlmod
import io
import json
import os
import random
import re
import sys
import time
import unicodedata
import urllib.robotparser
from collections import Counter, OrderedDict
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from urllib.parse import urlparse, urljoin, unquote

UA = "SabiYarn-longdoc-benchmark/0.1 (research; contact <email>)"
MIN_TOKENS = 400
MAX_TOKENS = 6500
MIN_PARAGRAPH_WORDS = 5

DOMAINS = ("news", "encyclopedia", "health", "law_government", "education", "stories", "travel", "science_explainer", "government_public", "opinion_magazine")
LICENCE_CLASSES = ("pd", "permissive", "share_alike", "noncommercial", "unclear")

# --------------------------------------------------------------------------------------------------------------- text cleaning
_URL_RE = re.compile(r"(?:https?://|ftp://|www\.)\S+", re.I)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_BARE_DOMAIN_RE = re.compile(r"\b[\w-]+\.(?:com|org|net|gov|edu|int|co\.uk|gov\.uk|info|io)(?:/\S*)?\b", re.I)
_CITE_RE = re.compile(r"\[(?:\d{1,3}(?:\s*[-,–]\s*\d{1,3})*|citation needed|clarification needed|note \d+|[a-z])\]", re.I)
_ZW_RE = re.compile("[\u200b\u200c\u200d\u2060\ufeff\u00ad]")
_WS_RE = re.compile(r"[ \t\r\f\v\u00a0\u2002\u2003\u2009\u202f]+")
_BOILER_RE = re.compile(
    r"^(?:read more|read also|related|share this|share on|follow us|subscribe|sign up|click here|photo:|image:|picture:|credit:|source:|"
    r"copyright|all rights reserved|advertisement|tags?:|filed under|posted in|print|email|tweet|like us|listen|watch|view all|"
    r"back to top|skip to|menu|search|cookies?|image source|image credit|photo credit|photograph:|caption:|\(?see also)(?![A-Za-z0-9])", re.I)
_STOP = frozenset("the of and to a in is that it for was on with as be by at this from are or an which have has had not but they his her their "
                  "we you he she were been will would can all there when who what if so more about than into its also one other some them these "
                  "those our may do did out up no only over such after most".split())


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def normalise_space(s: str) -> str:
    s = _ZW_RE.sub("", s)
    s = s.replace("\u2028", " ").replace("\u2029", " ").replace("\n", " ")
    s = _WS_RE.sub(" ", s)
    return s.strip()


def strip_urls(p: str) -> Optional[str]:
    """Remove URLs / e-mails / bare domains. A parenthesised URL is dropped with its brackets; a paragraph where a URL or e-mail is part of
    the running sentence (the remaining text would be truncated) is rejected (None)."""
    if not (_URL_RE.search(p) or _EMAIL_RE.search(p) or _BARE_DOMAIN_RE.search(p)):
        return p
    q = re.sub(r"\s*[(\[][^()\[\]]{0,12}(?:https?://|www\.)[^()\[\]]*[)\]]", "", p)
    q = re.sub(r"\s*[(\[][^()\[\]]*\b[\w-]+\.(?:com|org|net|gov|edu|int)\b[^()\[\]]*[)\]]", "", q)
    if _URL_RE.search(q) or _EMAIL_RE.search(q) or _BARE_DOMAIN_RE.search(q):
        return None
    return q


def clean_paragraph(p: str) -> str:
    p = htmlmod.unescape(p)
    p = nfc(p)
    p = normalise_space(p)
    p = _CITE_RE.sub("", p)
    p = re.sub(r"\s+([,.;:!?])", r"\1", p)
    p = re.sub(r"\(\s*\)", "", p)
    return normalise_space(p)


def _letters_ratio(p: str) -> float:
    nonspace = [c for c in p if not c.isspace()]
    if not nonspace:
        return 0.0
    return sum(c.isalpha() for c in nonspace) / len(nonspace)


_GAP_RE = re.compile(r"\b(?:the|a|an|of|to|by|with|in|for|and|from|at|on|that|than|as)\s+[\u2019']s\b|\s'\s|\bthe\s*[,;.)]|\b(?:a|an)\s*[,;)]|"
                     r"\b(?:of|by|with)\s*[,;]|\(\s*[,;]|[,;]\s*[,;]|\bof\s+(?:and|or)\b|\s[\u2019']\s*[,.]")


def has_gap(p: str) -> bool:
    """True when the text shows the signature of removed markup (a template / link whose text was dropped): 'the \u2019s representative', 'of,' ..."""
    return bool(_GAP_RE.search(p))


def abbreviation_density(p: str) -> float:
    toks = p.split()
    return sum(1 for t in toks if re.fullmatch(r"[A-Za-z]{1,4}\.", t)) / max(1, len(toks))


def keep_paragraph(p: str, min_words: int = MIN_PARAGRAPH_WORDS) -> bool:
    """True when `p` (already cleaned) looks like a prose paragraph or a list item with a complete statement."""
    if not p:
        return False
    words = p.split()
    if len(words) < min_words:
        return False
    if _BOILER_RE.match(p):
        return False
    if re.search(r"\bet al\.?[,.;]", p) and re.search(r"\b(?:19|20)\d\d\b", p) and len(words) < 70:        # bibliographic reference
        return False
    if has_gap(p) or abbreviation_density(p) > 0.10:
        return False
    if _letters_ratio(p) < 0.72:
        return False
    digits = sum(c.isdigit() for c in p)
    if digits / max(1, len(p)) > 0.12:
        return False
    if sum(1 for c in p if c in "=<>{}[]|\\^~_#$%*") > 3:
        return False
    # heading-like: short, no sentence punctuation, mostly capitalised words
    end = p.rstrip("\"”’')»]*")[-1:] if p.rstrip("\"”’')»]*") else ""
    if end not in ".!?:;…" and len(words) < 12:
        cap = sum(1 for w in words if w[:1].isupper())
        if cap / len(words) > 0.6 or len(words) < 8:
            return False
    stop = sum(1 for w in re.findall(r"[a-zA-Z']+", p.lower()) if w in _STOP)
    if len(words) >= 8 and stop / max(1, len(words)) < 0.12:
        return False
    return True


def english_ok(text: str) -> bool:
    toks = re.findall(r"[^\W\d_]+", text.lower())
    if len(toks) < 50:
        return False
    stop = sum(1 for t in toks if t in _STOP) / len(toks)
    ascii_letters = sum(1 for c in text if c.isascii() and c.isalpha()) / max(1, sum(1 for c in text if c.isalpha()))
    return stop >= 0.22 and ascii_letters >= 0.93


def short_english_ok(text: str) -> bool:
    """English check that also works for one-sentence items (verbatim benchmark text): mostly ASCII letters and, from 12 words on, a normal share of function words."""
    toks = re.findall(r"[^\W\d_]+", text.lower())
    if len(toks) < 3:
        return False
    letters = [c for c in text if c.isalpha()]
    ascii_letters = sum(1 for c in letters if c.isascii()) / max(1, len(letters))
    stop = sum(1 for t in toks if t in _STOP) / len(toks)
    return ascii_letters >= 0.93 and (len(toks) < 12 or stop >= 0.15)


def clean_paragraphs(raw: Iterable[str], min_words: int = MIN_PARAGRAPH_WORDS) -> List[str]:
    out: List[str] = []
    for p in raw:
        p = clean_paragraph(p)
        p = strip_urls(p) if p else None
        if p is None:
            continue
        p = clean_paragraph(p)
        if keep_paragraph(p, min_words):
            out.append(p)
    return out


# ------------------------------------------------------------------------------------------------------------------------ HTML
_DROP_TAGS = frozenset(
    "script style nav aside figure figcaption table form header footer noscript svg button select iframe math audio video canvas "
    "template menu dialog h1 h2 h3 h4 h5 h6 caption thead tbody tfoot pre code".split())
_BLOCK_TAGS = frozenset(
    "p div section article main ul ol li dl dt dd blockquote tr td th br hr body html details summary fieldset address center".split())


def _is_footnote_sup(el) -> bool:
    cls = (el.get("class") or "") + " " + (el.get("id") or "")
    if re.search(r"refer|footnote|fn|cite|note", cls, re.I):
        return True
    return any(c.tag == "a" for c in el.iterchildren() if isinstance(c.tag, str)) or bool(re.fullmatch(r"\[?[\d\s,\-–*†‡]{1,8}\]?", (el.text_content() or "").strip()))


def html_paragraphs(root, drop_classes: Sequence[str] = (), drop_ids: Sequence[str] = ()) -> List[str]:
    """Paragraph strings of an lxml element / list of elements. Headings, tables, figures, navigation, scripts are skipped."""
    roots = root if isinstance(root, (list, tuple)) else [root]
    out: List[str] = []
    buf: List[str] = []
    cls_re = re.compile(r"(?:^|[\s_-])(?:%s)(?:$|[\s_-])" % "|".join(re.escape(c) for c in drop_classes), re.I) if drop_classes else None

    def flush():
        if buf:
            s = "".join(buf)
            buf.clear()
            if s.strip():
                out.append(s)

    def walk(el):
        tag = el.tag
        if not isinstance(tag, str):          # comment / processing instruction
            return
        tag = tag.lower()
        if tag in _DROP_TAGS:
            return
        if tag == "sup" and _is_footnote_sup(el):
            return
        if tag in ("sub",) and not (el.text or "").strip():
            return
        cl = (el.get("class") or "")
        if cls_re is not None and cls_re.search(cl):
            return
        if drop_ids and (el.get("id") or "") in drop_ids:
            return
        if el.get("hidden") is not None or el.get("aria-hidden") == "true":
            return
        block = tag in _BLOCK_TAGS and tag != "br"
        if tag == "br":
            buf.append(" ")
            return
        if block:
            flush()
        if el.text:
            buf.append(el.text)
        for ch in el:
            walk(ch)
            if ch.tail:
                buf.append(ch.tail)
        if block:
            flush()

    for r in roots:
        walk(r)
        flush()
    return out


def parse_html(html_text: str):
    from lxml import html as lh
    if not html_text.strip():
        return None
    return lh.fromstring(html_text)


def html_to_paragraphs(html_text: str, **kw) -> List[str]:
    root = parse_html(html_text)
    return [] if root is None else html_paragraphs(root, **kw)


# ---------------------------------------------------------------------------------------------------------------------- tokens
class TokenCounter:
    """Gemma token count via the `tokenizers` library (tokenizer.json of google/gemma-3-270m-it, no BOS). Without a tokenizer file
    (offline tests) falls back to chars/4 and says so in `.exact`."""

    def __init__(self, path: Optional[str] = None):
        self.exact = False
        self._tok = None
        path = path or os.environ.get("LONGDOC_TOKENIZER") or ""
        if not path:
            hits = sorted(glob.glob("/workspace/sft_build/hf/hub/models--google--gemma-3-270m-it/snapshots/*/tokenizer.json"))
            hits += sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-3-270m-it/snapshots/*/tokenizer.json")))
            path = hits[0] if hits else ""
        if path and os.path.exists(path):
            try:
                from tokenizers import Tokenizer
                self._tok = Tokenizer.from_file(path)
                self.exact = True
            except Exception as e:      # pragma: no cover
                print(f"[tokens] cannot load {path}: {e}", file=sys.stderr)

    def count(self, text: str) -> int:
        if self._tok is None:
            return max(1, int(len(text) / 4.0))
        return len(self._tok.encode(text, add_special_tokens=False).ids)

    def count_many(self, texts: Sequence[str]) -> List[int]:
        if self._tok is None:
            return [self.count(t) for t in texts]
        return [len(e.ids) for e in self._tok.encode_batch(list(texts), add_special_tokens=False)]


def fingerprint(text: str) -> str:
    s = re.sub(r"[^0-9a-z]+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def cut_to_tokens(paragraphs: List[str], tc: TokenCounter, max_tokens: int = MAX_TOKENS) -> Tuple[List[str], bool]:
    """Keep the leading paragraphs while the document stays <= max_tokens (paragraph boundary); returns (paragraphs, truncated)."""
    counts = tc.count_many(paragraphs)
    total = 0
    kept: List[str] = []
    for p, n in zip(paragraphs, counts):
        n += 2
        if total + n > max_tokens and kept:
            return kept, True
        kept.append(p)
        total += n
    return kept, False


# ------------------------------------------------------------------------------------------------------------------------ HTTP
class Http:
    """requests wrapper: User-Agent, robots.txt (incl. Crawl-delay), >= `min_interval` seconds between requests to the same host, retries."""

    def __init__(self, min_interval: float = 1.0, timeout: float = 60.0, respect_robots: bool = True, retries: int = 4):
        import requests
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA, "Accept-Language": "en"})
        self.min_interval = min_interval
        self.timeout = timeout
        self.respect_robots = respect_robots
        self.retries = retries
        self._last: Dict[str, float] = {}
        self._robots: Dict[str, Optional[urllib.robotparser.RobotFileParser]] = {}
        self._delay: Dict[str, float] = {}
        self.n_requests = 0

    def _host(self, url: str) -> str:
        u = urlparse(url)
        return f"{u.scheme}://{u.netloc}"

    def _wait(self, host: str, extra: float = 0.0):
        gap = max(self.min_interval, self._delay.get(host, 0.0)) + extra
        now = time.time()
        last = self._last.get(host, 0.0)
        if now - last < gap:
            time.sleep(gap - (now - last))
        self._last[host] = time.time()

    def allowed(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        host = self._host(url)
        if host not in self._robots:
            rp = urllib.robotparser.RobotFileParser()
            try:
                self._wait(host)
                r = self.s.get(host + "/robots.txt", timeout=self.timeout)
                ct = r.headers.get("content-type", "")
                if r.status_code == 200 and "html" not in ct:
                    rp.parse(r.text.splitlines())
                    cd = rp.crawl_delay(UA) or rp.crawl_delay("*")
                    if cd:
                        self._delay[host] = float(cd)
                    self._robots[host] = rp
                else:
                    self._robots[host] = None       # no (usable) robots.txt: everything allowed
            except Exception:
                self._robots[host] = None
        rp = self._robots[host]
        return True if rp is None else rp.can_fetch(UA, url)

    def get(self, url: str, params=None, headers=None, stream: bool = False, allow_404: bool = False):
        if not self.allowed(url):
            raise PermissionError(f"robots.txt disallows {url}")
        host = self._host(url)
        last = None
        for attempt in range(self.retries):
            self._wait(host, extra=0.0 if attempt == 0 else 2.0 ** attempt)
            try:
                self.n_requests += 1
                r = self.s.get(url, params=params, headers=headers, timeout=self.timeout, stream=stream)
                if r.status_code in (429, 500, 502, 503, 504):
                    last = RuntimeError(f"HTTP {r.status_code} for {url}")
                    ra = r.headers.get("Retry-After")
                    if ra and ra.isdigit():
                        time.sleep(min(60, int(ra)))
                    continue
                if r.status_code == 404 and allow_404:
                    return None
                r.raise_for_status()
                return r
            except Exception as e:     # network errors, 4xx
                last = e
                if getattr(getattr(e, "response", None), "status_code", 0) in (400, 401, 403, 404, 410):
                    break
        raise RuntimeError(f"giving up on {url}: {last}")

    def text(self, url, **kw) -> Optional[str]:
        r = self.get(url, **kw)
        if r is None:
            return None
        try:
            return r.content.decode("utf-8")
        except UnicodeDecodeError:
            return r.content.decode(r.encoding or "latin-1", errors="replace")

    def json(self, url, **kw):
        r = self.get(url, **kw)
        return None if r is None else r.json()


# ------------------------------------------------------------------------------------------------------------------- pool writer
class Pool:
    """Collects accepted documents into `<out>.part` (append, flushed per document) and renames to `<out>` in `close()`."""

    def __init__(self, out: str, source: str, tc: TokenCounter, max_docs: int, min_tokens: int = MIN_TOKENS, max_tokens: int = MAX_TOKENS,
                 exclude_from: Sequence[str] = (), min_paragraphs: int = 3, verbatim: bool = False):
        self.out, self.part = out, out + ".part"
        # `verbatim`: curated benchmark text (WMT, NTREX, FLORES+ ...): the paragraphs are kept exactly as published (only Unicode NFC / whitespace), no
        # prose heuristics, repeated paragraphs kept, short documents allowed (`min_paragraphs`, `min_tokens`), a short-text English check.
        self.min_paragraphs, self.verbatim = min_paragraphs, verbatim
        self.source, self.tc, self.max_docs = source, tc, max_docs
        self.min_tokens, self.max_tokens = min_tokens, max_tokens
        self.publish_every = 0
        self.ids, self.hashes = set(), set()
        self.n = 0
        self.stats = Counter()
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        if os.path.exists(self.part):
            with open(self.part, encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    self.ids.add(d["id"])
                    self.hashes.add(fingerprint(d["text"]))
                    self.n += 1
        self.n_resumed = self.n
        for ef in exclude_from:                 # ids / texts of an earlier pool file: never fetched or written again (extends a pool without duplicates)
            if not ef or not os.path.exists(ef):
                continue
            with open(ef, encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    self.ids.add(d["id"])
                    self.hashes.add(fingerprint(d["text"]))
        self.fh = open(self.part, "a", encoding="utf-8")
        self.rej_path = out + ".rejects"
        self.n_rej = 0

    def _reject(self, reason: str, did: str, paras: Sequence[str]):
        self.stats[reason] += 1
        if self.n_rej < 300:                    # small sample of the rejected documents, for debugging the parsers
            self.n_rej += 1
            with open(self.rej_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"id": did, "reason": reason, "head": " | ".join(paras)[:300]}, ensure_ascii=False) + "\n")

    def seen(self, doc_id: str) -> bool:
        return doc_id in self.ids

    @property
    def full(self) -> bool:
        return self.n >= self.max_docs

    def add(self, d: dict) -> bool:
        """d: id, domain, licence, licence_class, licence_evidence, url, title, attribution, date, paragraphs|text (+ optional keys)."""
        did = d["id"]
        if did in self.ids:
            return False
        paras = d.pop("paragraphs", None)
        if paras is None:
            paras = [p for p in re.split(r"\n\s*\n", d.pop("text")) if p.strip()]
        if self.verbatim:
            paras = [q for q in (clean_paragraph(p) for p in paras) if q]
        else:
            paras = list(OrderedDict.fromkeys(clean_paragraphs(paras)))          # exact repeats of a paragraph inside one document (lead repeated as first paragraph) are dropped
        if len(paras) < self.min_paragraphs:
            self._reject("few_paragraphs", did, paras)
            return False
        paras, truncated = cut_to_tokens(paras, self.tc, self.max_tokens)
        text = "\n\n".join(paras)
        if not (short_english_ok(text) if self.verbatim else english_ok(text)):
            self._reject("not_english", did, paras)
            return False
        n_tok = self.tc.count(text)
        if n_tok < self.min_tokens:
            self.stats["too_short"] += 1
            return False
        h = fingerprint(text)
        if h in self.hashes:
            self.stats["duplicate"] += 1
            return False
        rec = OrderedDict()
        rec["id"] = did
        rec["source"] = d.get("source", self.source)
        for k in ("domain", "licence", "licence_class", "licence_evidence", "url", "title", "attribution", "date"):
            rec[k] = d.get(k, "") or ""
        assert rec["domain"] in DOMAINS, rec["domain"]
        assert rec["licence_class"] in LICENCE_CLASSES, rec["licence_class"]
        rec["text"] = text
        rec["n_chars"] = len(text)
        rec["n_gemma_tokens"] = n_tok
        if truncated:
            rec["truncated"] = True
        for k in ("licence_note", "extra"):
            if d.get(k):
                rec[k] = d[k]
        self.fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.fh.flush()
        self.ids.add(did)
        self.hashes.add(h)
        self.n += 1
        self.stats["accepted"] += 1
        if self.publish_every and self.n % self.publish_every == 0:
            self.publish()
        return True

    def publish(self):
        """Atomic snapshot of the documents so far (copy to a temp file, rename): a reader of `<out>` never sees a half-written file."""
        import shutil
        self.fh.flush()
        tmp = self.out + ".snap"
        shutil.copyfile(self.part, tmp)
        os.replace(tmp, self.out)

    def close(self, complete: bool = True):
        self.fh.close()
        if complete and os.path.exists(self.part):
            os.replace(self.part, self.out)


class Ctx:
    def __init__(self, args, http: Http, pool: Pool):
        self.args, self.http, self.pool = args, http, pool
        self.raw_dir = args.raw_dir

    def log(self, *a):
        print(f"[{self.pool.source}] ", *a, file=sys.stderr, flush=True)


# ============================================================================================================== source registry
SOURCES: Dict[str, dict] = {}


def source(name: str, domains: str, licence: str, cls: str, evidence: str, access: str):
    def deco(fn):
        SOURCES[name] = dict(fn=fn, domains=domains, licence=licence, cls=cls, evidence=evidence, access=access, doc=(fn.__doc__ or "").strip())
        return fn
    return deco


# =================================================================================================================== Project Gutenberg
_PG_START = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I | re.S)
_PG_END = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", re.I)
_PG_HEADING = re.compile(
    r"^(?:(?:CHAPTER|Chapter|CHAP\.|Chap\.|BOOK|Book|PART|Part|STAVE|Stave|SECTION|Section|LETTER|Letter|ACT|Act|STORY|Story)\b\s*[IVXLCDM\d]*[.:]?.{0,80}"
    r"|[IVXLCDM]{1,7}\.?|\d{1,3}\.?|[A-Z][A-Z .,;:'’\-!?]{2,70})$")


def gutenberg_paragraphs(raw: str) -> List[Tuple[str, bool]]:
    """Paragraphs of a Project Gutenberg plain-text file as (text, is_heading); licence header / footer stripped."""
    m = _PG_START.search(raw)
    if m:
        raw = raw[m.end():]
    e = _PG_END.search(raw)
    if e:
        raw = raw[:e.start()]
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    raw = re.sub(r"\[(?:Illustration|Footnote|Transcriber|Note|Sidenote|Image)[^\]]*\]", " ", raw, flags=re.S | re.I)
    out: List[Tuple[str, bool]] = []
    for blk in re.split(r"\n\s*\n", raw):
        lines = [l.strip() for l in blk.split("\n") if l.strip()]
        if not lines:
            continue
        joined = normalise_space(" ".join(lines))
        joined = re.sub(r"(?<![\w])_([^_\n]+)_(?![\w])", r"\1", joined)      # _italic_
        joined = joined.replace("_", "")
        avg = sum(len(l) for l in lines) / len(lines)
        heading = len(lines) <= 3 and len(joined) <= 90 and bool(_PG_HEADING.match(joined))
        if (len(lines) >= 3 and avg < 40) or (len(lines) >= 2 and avg < 26):          # verse / table of contents / index / tables
            continue
        out.append((joined, heading))
    return out


def gutenberg_segments(paras: List[Tuple[str, bool]], tc: TokenCounter, min_tokens: int = MIN_TOKENS) -> List[Tuple[str, List[str]]]:
    """Split a book at its headings into (heading, paragraphs) segments; segments below `min_tokens` are merged forward."""
    segs: List[Tuple[str, List[str]]] = []
    head, cur = "", []
    for text, is_head in paras:
        if is_head:
            if cur:
                segs.append((head, cur))
            head, cur = text, []
        else:
            cur.append(text)
    if cur:
        segs.append((head, cur))
    merged: List[Tuple[str, List[str]]] = []
    carry_head, carry = "", []
    for h, ps in segs:
        n_raw = len(ps)
        ps = clean_paragraphs(ps)
        if not ps or len(ps) < 0.7 * n_raw or sum(len(p.split()) for p in ps) / len(ps) < 25:      # prefer narrative prose: tables, lists, indexes, verse fail
            continue
        if not carry:
            carry_head = h
        carry = carry + ps
        if tc.count("\n\n".join(carry)) >= min_tokens + 100:
            merged.append((carry_head, carry))
            carry = []
    if carry and merged:
        pass          # a short tail is dropped on purpose (usually an epilogue / advert)
    return merged


def pg_death_year(authors: str) -> Optional[int]:
    years = []
    for a in (authors or "").split(";"):
        a = a.strip()
        if not a:
            continue
        m = re.search(r"(\d{4})\??\s*-\s*(\d{4})", a)
        if not m:
            return None
        years.append(int(m.group(2)))
    return max(years) if years else None


def pg_domain(subjects: str, locc: str, shelves: str) -> Optional[str]:
    s, l, b = (subjects or "").lower(), (locc or ""), (shelves or "").lower()
    first = l.split(";")[0].strip()
    if re.search(r"description and travel|voyages and travels|travel|africa -- (?:description|social life)", s) and not re.search(r"juvenile fiction|fiction|poetry", s):
        return "travel"
    if re.search(r"fiction|short stories|fairy tales|folklore|tales|legends|fables|myths", s) or first.startswith(("PZ", "PR", "PS")):
        if re.search(r"poetry|drama|plays", s) and not re.search(r"fiction|stories|tales", s):
            return None
        return "stories"
    if first.startswith(("AE",)) or "encyclopedia" in s or "cyclopedia" in s:
        return "encyclopedia"
    if first.startswith(("R",)) or re.search(r"hygiene|nursing|diseases|medicine|first aid|health|physiology", s):
        return "health"
    if first.startswith(("L",)) or re.search(r"education|textbooks|teaching|schools", s):
        return "education"
    if first.startswith(("J", "K")) or re.search(r"constitution|political science|law and legislation|government|civil rights", s):
        return "law_government"
    if first.startswith(("Q", "S", "T")) or re.search(r"natural history|agriculture|science|animals|plants|geology|astronomy|botany|zoology", s):
        return "science_explainer"
    if first.startswith(("G",)):
        return "travel"
    return None


PG_QUOTA = {"stories": 0.36, "travel": 0.16, "science_explainer": 0.14, "health": 0.10, "education": 0.10, "law_government": 0.10, "encyclopedia": 0.04}
PG_EVIDENCE = "https://www.gutenberg.org/policy/permission.html"


def pg_candidates(catalog_csv: str, max_death: int = 1954, seed: int = 20261020) -> Dict[str, List[dict]]:
    rows = list(csv.DictReader(io.StringIO(catalog_csv)))
    by_dom: Dict[str, List[dict]] = {d: [] for d in PG_QUOTA}
    per_author: Counter = Counter()
    rng = random.Random(seed)
    rng.shuffle(rows)
    for r in rows:
        if r.get("Language") != "en" or r.get("Type") != "Text":
            continue
        dy = pg_death_year(r.get("Authors", ""))
        if dy is None or dy > max_death:
            continue
        dom = pg_domain(r.get("Subjects", ""), r.get("LoCC", ""), r.get("Bookshelves", ""))
        if dom is None or dom not in by_dom:
            continue
        a = r.get("Authors", "").split(";")[0]
        if per_author[(a, dom)] >= 2:
            continue
        per_author[(a, dom)] += 1
        title = normalise_space(r.get("Title", "").replace("\n", " "))
        by_dom[dom].append(dict(id=r["Text#"], title=title, authors=r.get("Authors", ""), death=dy, dom=dom))
    return by_dom


@source("gutenberg", "stories,travel,science_explainer,health,education,law_government,encyclopedia", "Public domain (US; author died <= 1954)", "pd",
        "https://www.gutenberg.org/policy/permission.html", "gutenberg.org/cache/epub/<id>/pg<id>.txt + pg_catalog.csv (1 req/s)")
def src_gutenberg(ctx: Ctx):
    """Project Gutenberg English books by authors who died before 1955 (PD in the US and in life+70 countries); split at chapter headings."""
    a = ctx.args
    cat = ctx.http.text("https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv")
    by_dom = pg_candidates(cat, seed=a.seed)
    budget = {d: max(1, int(round(a.max_docs * q))) for d, q in PG_QUOTA.items()}
    per_book = a.per_book
    got = Counter()
    # round-robin over domains so that a small --max-docs still covers all of them
    iters = {d: iter(v) for d, v in by_dom.items()}
    active = list(iters)
    while active and not ctx.pool.full:
        for d in list(active):
            if ctx.pool.full:
                break
            if got[d] >= budget[d]:
                active.remove(d)
                continue
            try:
                book = next(iters[d])
            except StopIteration:
                active.remove(d)
                continue
            prefix = f"gutenberg:pg{book['id']}-"
            if any(i.startswith(prefix) for i in ctx.pool.ids):
                got[d] += sum(1 for i in ctx.pool.ids if i.startswith(prefix))
                continue
            try:
                txt = ctx.http.text(f"https://www.gutenberg.org/cache/epub/{book['id']}/pg{book['id']}.txt", allow_404=True)
            except Exception as e:
                ctx.log("skip", book["id"], e)
                continue
            if not txt or re.search(r"^\s*(?:Translator|Translated by)\s*:", txt[:6000], re.I | re.M):
                continue
            segs = gutenberg_segments(gutenberg_paragraphs(txt), ctx.pool.tc)
            if not segs:
                continue
            rng = random.Random(f"{a.seed}-{book['id']}")
            order = list(range(1 if len(segs) >= 3 else 0, len(segs)))     # the first segment is mostly front matter
            rng.shuffle(order)
            took = 0
            for k in sorted(order[:per_book * 3]):
                h, ps = segs[k]
                if took >= per_book:
                    break
                ok = ctx.pool.add(dict(
                    id=f"gutenberg:pg{book['id']}-{k:03d}", domain=d, licence="Public domain (US; author died %d)" % book["death"], licence_class="pd",
                    licence_evidence=PG_EVIDENCE, url=f"https://www.gutenberg.org/ebooks/{book['id']}", title=(book["title"] + (" - " + h if h else ""))[:200],
                    attribution=f"{book['authors'].split(';')[0]}; via Project Gutenberg (ebook #{book['id']})", date="", paragraphs=ps))
                if ok:
                    took += 1
                    got[d] += 1
    ctx.log("per-domain", dict(got))


# ======================================================================================================================== GOV.UK (OGL v3)
GOVUK_OGL = "https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/"
GOVUK_TERMS = "https://www.gov.uk/help/terms-conditions"
GOVUK_DROP = ("application-notice", "help-notice", "call-to-action", "contact", "address", "info-notice", "form-title", "attachment", "gem-c",
              "govuk-notification", "example", "summary", "govspeak-embed", "demoted", "important-notice")

_HEALTH = ["uk-health-security-agency", "department-of-health-and-social-care", "medicines-and-healthcare-products-regulatory-agency",
           "public-health-england", "nhs-england"]
_EDU = ["department-for-education", "ofsted", "ofqual"]
_SCI = ["department-for-environment-food-rural-affairs", "environment-agency", "natural-england", "forestry-commission",
        "animal-and-plant-health-agency", "department-for-energy-security-and-net-zero", "government-office-for-science", "met-office"]
_LAW = ["home-office", "ministry-of-justice", "hm-revenue-customs", "cabinet-office", "attorney-generals-office", "foreign-commonwealth-development-office"]

# (spec id, domain, search filters, share of --max-docs)
GOVUK_SPECS = [
    ("travel", "travel", {"filter_format": "travel_advice"}, 0.14),
    ("news-story", "news", {"filter_format": "news_story"}, 0.08),
    ("news-world", "news", {"filter_format": "world_news_story"}, 0.07),
    ("news-press", "news", {"filter_format": "press_release"}, 0.05),
    ("law-speech", "law_government", {"filter_format": "speech"}, 0.12),
    ("law-guide", "law_government", {"filter_format": "detailed_guide", "filter_organisations": _LAW}, 0.07),
    ("law-guidance", "law_government", {"filter_format": "guidance", "filter_organisations": _LAW}, 0.05),
    ("health-guide", "health", {"filter_format": "detailed_guide", "filter_organisations": _HEALTH}, 0.08),
    ("health-guidance", "health", {"filter_format": "guidance", "filter_organisations": _HEALTH}, 0.08),
    ("health-news", "health", {"filter_format": "news_story", "filter_organisations": _HEALTH}, 0.02),
    ("edu-guide", "education", {"filter_format": "detailed_guide", "filter_organisations": _EDU}, 0.05),
    ("edu-guidance", "education", {"filter_format": "guidance", "filter_organisations": _EDU}, 0.06),
    ("edu-speech", "education", {"filter_format": "speech", "filter_organisations": _EDU}, 0.03),
    ("sci-guide", "science_explainer", {"filter_format": "detailed_guide", "filter_organisations": _SCI}, 0.07),
    ("sci-guidance", "science_explainer", {"filter_format": "guidance", "filter_organisations": _SCI}, 0.05),
    ("sci-news", "science_explainer", {"filter_format": "news_story", "filter_organisations": _SCI}, 0.03),
]


def govuk_content_paragraphs(c: dict) -> List[str]:
    """Paragraphs of a GOV.UK content-API item (details.body / details.parts[].body / details.introduction)."""
    d = c.get("details") or {}
    htmls: List[str] = []
    if isinstance(d.get("introduction"), str):
        htmls.append(d["introduction"])
    if isinstance(d.get("body"), str):
        htmls.append(d["body"])
    for p in d.get("parts") or []:
        if isinstance(p, dict) and isinstance(p.get("body"), str):
            htmls.append(p["body"])
    out: List[str] = []
    for h in htmls:
        if not h.strip():
            continue
        root = parse_html("<div>" + h + "</div>")
        if root is not None:
            out.extend(html_paragraphs(root, drop_classes=GOVUK_DROP))
    return out


def govuk_listing(http: Http, filters: dict, seed: int, count: int = 100) -> Iterator[dict]:
    """Search-API result rows of one spec, in a seeded random page order (deterministic)."""
    base = {"count": count, "fields": "title,link,format,public_timestamp,organisations"}
    params = dict(base, **{k: v for k, v in filters.items()})
    j = http.json("https://www.gov.uk/api/search.json", params=dict(params, count=1))
    total = min(int(j["total"]), 9900)
    pages = list(range(0, total, count))
    random.Random(seed).shuffle(pages)
    for start in pages:
        j = http.json("https://www.gov.uk/api/search.json", params=dict(params, start=start, order="public_timestamp"))
        rows = j.get("results", [])
        random.Random(seed + start).shuffle(rows)
        for r in rows:
            yield r


@source("govuk", "travel,news,law_government,health,education,science_explainer", "OGL-3.0 (Open Government Licence v3.0)", "permissive", GOVUK_TERMS,
        "GOV.UK Search API + Content API (JSON, 1 req/s)")
def src_govuk(ctx: Ctx):
    """GOV.UK content (travel advice, news, speeches, guidance) under the Open Government Licence v3.0."""
    a = ctx.args
    only = set(a.only.split(",")) if a.only else None
    specs = [s for s in GOVUK_SPECS if (not only or s[1] in only or s[0] in only)]
    budget = {s[0]: max(1, int(round(a.max_docs * s[3]))) for s in specs}
    got = Counter()
    for i in ctx.pool.ids:                      # resume: count what is already there per spec
        sid = (i.split(":", 1)[1] if ":" in i else "").split("|")[0]
        got[sid] += 1
    iters = {s[0]: govuk_listing(ctx.http, s[2], a.seed + k) for k, s in enumerate(specs)}
    dom = {s[0]: s[1] for s in specs}
    tried = Counter()
    active = [s[0] for s in specs]
    while active and not ctx.pool.full:
        for sid in list(active):
            if ctx.pool.full:
                break
            if got[sid] >= budget[sid] or tried[sid] > budget[sid] * 14 + 60:
                active.remove(sid)
                continue
            try:
                row = next(iters[sid])
            except StopIteration:
                active.remove(sid)
                continue
            path = row["link"]
            doc_id = f"govuk:{sid}|{path.strip('/').replace('/', '~')}"
            if ctx.pool.seen(doc_id):
                continue
            tried[sid] += 1
            try:
                c = ctx.http.json("https://www.gov.uk/api/content" + path, allow_404=True)
            except Exception as e:
                ctx.log("skip", path, e)
                continue
            if not c or c.get("locale", "en") != "en":
                continue
            paras = govuk_content_paragraphs(c)
            orgs = [o.get("title", "") for o in (c.get("links", {}).get("organisations") or [])]
            ok = ctx.pool.add(dict(
                id=doc_id, domain=dom[sid], licence="OGL-3.0", licence_class="permissive", licence_evidence=GOVUK_TERMS,
                url="https://www.gov.uk" + path, title=c.get("title", ""),
                attribution="Crown copyright" + (", " + "; ".join(orgs) if orgs else "") + ", GOV.UK, licensed under the Open Government Licence v3.0",
                date=(c.get("first_published_at") or c.get("public_updated_at") or "")[:10], paragraphs=paras,
                licence_note="OGL v3 applies 'except where otherwise stated'; third-party quotes, logos and images are excluded from the licence"))
            if ok:
                got[sid] += 1
    ctx.log("per-spec", dict(got))


# ====================================================================================================================== MediaWiki dumps
_WT_COMMENT = re.compile(r"<!--.*?-->", re.S)
_WT_REF = re.compile(r"<ref\b[^>]*?/>|<ref\b[^>]*?>.*?</ref>", re.S | re.I)
_WT_BLOCKTAGS = re.compile(r"<(gallery|math|nowiki|timeline|imagemap|score|syntaxhighlight|source|pre|code|references|poem|noinclude|onlyinclude|includeonly|mapframe|maplink|ref)\b.*?</\1\s*>", re.S | re.I)
_WT_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_WT_FILE = re.compile(r"\[\[\s*:?(?:File|Image|Category|Media|Template|Help|Portal|Special|Talk|User|MediaWiki|Draft)\s*:[^\[\]]*(?:\[\[[^\[\]]*\]\][^\[\]]*)*\]\]", re.I)
_WT_INTERWIKI = re.compile(r"\[\[\s*:?(?:w|wp|wikipedia|wikt|wiktionary|wikinews|n|commons|c|s|wikisource|q|wikiquote|v|wikivoyage|b|wikibooks|d|wikidata|m|meta|en)\s*:\s*([^\[\]|]*)(?:\|([^\[\]]*))?\]\]", re.I)
_WT_LANGLINK = re.compile(r"^\s*\[\[[a-z]{2,3}(?:-[a-z]+)*:[^\[\]|]*\]\]\s*$", re.M)
_WT_LINK = re.compile(r"\[\[(?:[^\[\]|]*\|)?([^\[\]|]*)\]\]")
_WT_EXT = re.compile(r"\[(?:https?:)?//[^\s\]]+(?:\s+([^\]]*))?\]")
_WT_CUT_HEADING = re.compile(r"^\s*=+\s*(?:Sources?|Related news|Related articles?|External links?|See also|References?|Notes|Footnotes|Further reading|Bibliography|Related stories)\s*=+\s*$", re.I | re.M)
_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]


_INLINE_TPL = re.compile(r"\{\{\s*(?:w|wp|wikipedia|wikt|wiktionary|w2|lw|iw|interwiki|l|link|nowrap|lang|abbr|convert|ill)\s*\|([^{}]*)\}\}", re.I)


def _inline_template(m) -> str:
    args = [a.strip() for a in m.group(1).split("|")]
    args = [a.split("=", 1)[1] if re.match(r"^[a-z\-]+=", a) and False else a for a in args]
    pos = [a for a in args if a and not re.match(r"^[A-Za-z\-_]+=", a)]
    if not pos:
        return ""
    name = m.group(0).split("|", 1)[0].strip("{ ").lower()
    if name in ("convert",):
        return " ".join(pos[:2])
    if name in ("lang",):
        return pos[-1]
    return pos[-1] if len(pos) > 1 else pos[0]


def strip_templates(s: str) -> str:
    """Inline link templates ({{w|a|b}} -> b) are expanded; every other {{...}} (nested) and {|...|} table is removed."""
    prev = None
    while prev != s:
        prev = s
        s = _INLINE_TPL.sub(_inline_template, s)
    out, depth, i, n = [], 0, 0, len(s)
    while i < n:
        if s.startswith("{{", i):
            depth += 1
            i += 2
        elif s.startswith("}}", i) and depth:
            depth -= 1
            i += 2
        elif depth:
            i += 1
        else:
            out.append(s[i])
            i += 1
    s = "".join(out)
    prev = None
    while prev != s:                                    # tables, possibly nested
        prev = s
        s = re.sub(r"\{\|[^{}]*?\n\|\}", "", s, flags=re.S)
    return s


def wikitext_to_paragraphs(text: str, cut_sections: bool = True) -> List[str]:
    text = _WT_COMMENT.sub("", text)
    if cut_sections:
        m = _WT_CUT_HEADING.search(text)
        if m:
            text = text[:m.start()]
    text = _WT_REF.sub("", text)
    text = _WT_BLOCKTAGS.sub("", text)
    text = strip_templates(text)
    text = _WT_LANGLINK.sub("", text)
    for _ in range(3):
        text = _WT_FILE.sub("", text)
    text = _WT_INTERWIKI.sub(lambda m: (m.group(2) or m.group(1)).strip(), text)
    text = _WT_LINK.sub(lambda m: m.group(1), text)
    text = _WT_EXT.sub(lambda m: (m.group(1) or ""), text)
    text = _WT_TAG.sub(" ", text)
    text = text.replace("'''", "").replace("''", "")
    text = re.sub(r"^\s*=+.*?=+\s*$", "", text, flags=re.M)         # headings
    text = re.sub(r"^[ \t]*[:;]+[ \t]*", "", text, flags=re.M)      # indents / definition lists
    text = re.sub(r"^[ \t]*(?:\*+|#+)[ \t]*", "", text, flags=re.M)  # list bullets (items stay paragraphs)
    text = re.sub(r"^-{4,}\s*$", "", text, flags=re.M)
    text = htmlmod.unescape(text)
    return [p for p in re.split(r"\n\s*\n|\n", text) if p.strip()]


def strip_dateline(p: str) -> str:
    """'LONDON, England -- ' / 'Monday, May 3, 2010 --' at the start of a news paragraph."""
    p = re.sub(r"^(?:\(?[A-Z][A-Za-z.'’ -]{2,40}(?:,\s*[A-Z][A-Za-z.'’ -]{2,30}){0,2}\)?\s*[—–-]{1,2}\s+)", "", p)
    p = re.sub(r"^[\s:\-—–]+", "", p)
    p = re.sub(r"^(?:(?:Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)day,?\s+)?(?:%s)\s+\d{1,2}(?:,\s*\d{4})?\s*[—–-]{1,2}\s+" % "|".join(m.capitalize() for m in _MONTHS), "", p)
    return p


def wikinews_date(wikitext: str) -> str:
    m = re.search(r"\{\{\s*date\s*\|\s*([^}|]*)", wikitext, re.I)
    if m:
        d = m.group(1).strip()
        m2 = re.match(r"([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})", d)
        if m2 and m2.group(1).lower() in _MONTHS:
            return "%s-%02d-%02d" % (m2.group(3), _MONTHS.index(m2.group(1).lower()) + 1, int(m2.group(2)))
        m2 = re.match(r"(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})", d)
        if m2 and m2.group(2).lower() in _MONTHS:
            return "%s-%02d-%02d" % (m2.group(3), _MONTHS.index(m2.group(2).lower()) + 1, int(m2.group(1)))
    return ""


def wikinews_licence(date: str) -> Tuple[str, str]:
    """(licence, licence_class) by the publication date, following https://en.wikinews.org/wiki/Wikinews:Copyright (read 2026-10-20)."""
    if not date:
        return "", "unclear"
    if date < "2005-09-25":
        return "Public domain (published before 2005-09-25)", "pd"
    if date < "2024-12-16":
        return "CC-BY-2.5", "permissive"
    return "CC-BY-4.0", "permissive"


def iter_dump_pages(path: str, ns: str = "0") -> Iterator[Tuple[str, str, str, str]]:
    """(title, wikitext, timestamp, page id) of the pages of a MediaWiki pages-articles dump (.xml.bz2) in namespace `ns` (comma-separated ids allowed); redirects skipped."""
    import bz2
    nsset = set(ns.split(","))
    from lxml import etree
    f = bz2.open(path, "rb")
    title = pid = nsv = text = ts = None
    redirect = False
    for ev, el in etree.iterparse(f, events=("end",), huge_tree=True):
        tag = etree.QName(el).localname
        if tag == "title":
            title = el.text
        elif tag == "ns":
            nsv = el.text
        elif tag == "id" and pid is None and el.getparent() is not None and etree.QName(el.getparent()).localname == "page":
            pid = el.text
        elif tag == "redirect":
            redirect = True
        elif tag == "timestamp":
            ts = el.text
        elif tag == "text":
            text = el.text or ""
        elif tag == "page":
            if nsv in nsset and not redirect and text:
                yield title, text, ts or "", pid or ""
            title = pid = nsv = text = ts = None
            redirect = False
            el.clear()
            while el.getprevious() is not None:
                del el.getparent()[0]


def _download(ctx: Ctx, url: str, dest: str):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return
    r = ctx.http.get(url, stream=True)
    tmp = dest + ".tmp"
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    os.replace(tmp, dest)


_WN_SKIP_STATUS = re.compile(r"\{\{\s*(?:review|develop|developing|AAA|disputed|notpublished|not published|abandoned|stale|rejected|breaking|unreviewed)\b", re.I)
_WN_PUBLISHED = re.compile(r"\{\{\s*(?:publish|published|archive|archived)\b", re.I)


def wikinews_docs(path: str) -> Iterator[dict]:
    for title, wt, ts, pid in iter_dump_pages(path):
        if (title or "").startswith(("Digest/", "Wikinews", "Portal", "Main Page", "Spotlight")):
            continue
        if not _WN_PUBLISHED.search(wt) or _WN_SKIP_STATUS.search(wt):
            continue
        if re.search(r"\{\{\s*(?:disambig|interview|opinion|editorial|obituary)\b", wt, re.I):
            pass
        date = wikinews_date(wt)
        lic, cls = wikinews_licence(date)
        if cls == "unclear":
            continue
        paras = wikitext_to_paragraphs(wt)
        if paras:
            paras[0] = strip_dateline(paras[0])
        if any(has_gap(clean_paragraph(p)) for p in paras):          # dropped template / link text left a broken sentence: skip the article
            continue
        yield dict(id=f"wikinews:{pid}", domain="news", licence=lic, licence_class=cls, licence_evidence="https://en.wikinews.org/wiki/Wikinews:Copyright",
                   url="https://en.wikinews.org/wiki/" + (title or "").replace(" ", "_"), title=title or "",
                   attribution='Wikinews contributors, "%s", en.wikinews.org, %s' % (title, lic), date=date, paragraphs=paras,
                   licence_note="may quote third-party text / press releases under fair use; images excluded")


@source("wikinews", "news", "CC-BY-4.0 (>= 2024-12-16) / CC-BY-2.5 (2005-09-25..2024-12-15) / PD (older)", "permissive",
        "https://en.wikinews.org/wiki/Wikinews:Copyright", "dumps.wikimedia.org/enwikinews/latest/enwikinews-latest-pages-articles.xml.bz2")
def src_wikinews(ctx: Ctx):
    """English Wikinews published articles from the official XML dump; licence by publication date."""
    raw = ctx.raw_dir or os.path.join(os.path.dirname(os.path.abspath(ctx.args.out)), "..", "raw", "dumps")
    os.makedirs(raw, exist_ok=True)
    path = os.path.join(raw, "enwikinews-latest-pages-articles.xml.bz2")
    _download(ctx, "https://dumps.wikimedia.org/enwikinews/latest/enwikinews-latest-pages-articles.xml.bz2", path)
    only = set((ctx.args.only or "permissive,pd").split(","))      # one pool file per licence class: run twice with --only pd / --only permissive
    for d in wikinews_docs(path):
        if ctx.pool.seen(d["id"]) or d["licence_class"] not in only:
            continue
        yield d


_WV_SKIP = re.compile(r"^(?:Wikivoyage|Talk|Phrasebook|Travel topics)", re.I)


def wikivoyage_docs(path: str) -> Iterator[dict]:
    for title, wt, ts, pid in iter_dump_pages(path):
        if re.search(r"\{\{\s*(?:disamb|disambiguation|stub|outline|usabletravelguide|pagebanner\|[^}]*\}\}\s*\{\{\s*outline)", wt, re.I) and len(wt) < 3000:
            continue
        paras = wikitext_to_paragraphs(wt, cut_sections=False)
        yield dict(id=f"wikivoyage:{pid}", domain="travel", licence="CC-BY-SA-4.0", licence_class="share_alike",
                   licence_evidence="https://en.wikivoyage.org/wiki/Wikivoyage:Terms_of_use", url="https://en.wikivoyage.org/wiki/" + (title or "").replace(" ", "_"),
                   title=title or "", attribution='Wikivoyage contributors, "%s", en.wikivoyage.org, CC BY-SA' % title, date=(ts or "")[:10], paragraphs=paras,
                   licence_note="share-alike: derivative translations must be released under CC BY-SA; some text imported from CC BY-SA 3.0 sources")


@source("wikivoyage", "travel", "CC-BY-SA-3.0/4.0", "share_alike", "https://en.wikivoyage.org/wiki/Wikivoyage:Terms_of_use",
        "dumps.wikimedia.org/enwikivoyage/latest/enwikivoyage-latest-pages-articles.xml.bz2")
def src_wikivoyage(ctx: Ctx):
    """English Wikivoyage travel guides (share-alike) from the official dump."""
    raw = ctx.raw_dir or os.path.join(os.path.dirname(os.path.abspath(ctx.args.out)), "..", "raw", "dumps")
    os.makedirs(raw, exist_ok=True)
    path = os.path.join(raw, "enwikivoyage-latest-pages-articles.xml.bz2")
    _download(ctx, "https://dumps.wikimedia.org/enwikivoyage/latest/enwikivoyage-latest-pages-articles.xml.bz2", path)
    for d in wikivoyage_docs(path):
        if not ctx.pool.seen(d["id"]) and sampled(d["id"], ctx.args.seed, ctx.args.sample):
            yield d


# ================================================================================================================== Global Voices (CC-BY 3.0)
GV_EVIDENCE = "https://globalvoices.org/about/global-voices-attribution-policy/"
_GV_BOILER = re.compile(
    r"^(?:featured image|this post is part|this story (?:originally|was)|this article (?:originally|was)|originally published|written by|edited by|"
    r"republished|read more|read also|screenshot|photo|image|graph|video|illustration|source:|listen|support|follow|\*+|this is a|translated by|"
    r"this post was|global voices|cover image|header image|map |a version of this|the author of this|[\w .'-]+ contributed to (?:this|the) post|"
    r"for more|click|\u2192|\u2013 |@|\[all links|\[this|read the first|this article is part)", re.I)


def gv_paragraphs(html_text: str) -> List[str]:
    root = parse_html("<div>" + html_text + "</div>")
    if root is None:
        return []
    paras = html_paragraphs(root, drop_classes=("wp-caption", "wp-caption-text", "caption", "twitter-tweet", "instagram-media", "embed", "wp-block-embed",
                                                "gv-", "share", "related", "wp-block-gallery", "tiled-gallery", "post-gallery", "screen-reader"))
    return [p for p in paras if not _GV_BOILER.match(clean_paragraph(p))]


@source("globalvoices", "news", "CC-BY-3.0", "permissive", GV_EVIDENCE, "WordPress REST API https://globalvoices.org/wp-json/wp/v2/posts (robots.txt Crawl-delay 10)")
def src_globalvoices(ctx: Ctx):
    """Global Voices English posts (citizen-media news analysis), 'unless otherwise stated' CC BY 3.0."""
    a = ctx.args
    base = "https://globalvoices.org/wp-json/wp/v2/posts"
    params = {"per_page": 100, "_embed": "author", "_fields": "id,date,link,title,content,_embedded,_links"}
    for attempt in range(12):                   # globalvoices.org was unreachable from the rental for stretches of an hour: wait and retry
        try:
            r = ctx.http.get(base, params=dict(params, per_page=100, page=1))
            break
        except Exception as e:
            ctx.log("first page failed, retry in 120 s:", str(e)[:80])
            time.sleep(120)
    else:
        raise RuntimeError("globalvoices.org unreachable")
    pages_total = int(r.headers.get("X-WP-TotalPages", "1"))
    pages = list(range(1, pages_total + 1))
    random.Random(a.seed).shuffle(pages)
    for pg in pages:
        j = None
        for attempt in range(4):
            try:
                j = ctx.http.json(base, params=dict(params, page=pg))
                break
            except Exception as e:
                ctx.log("page", pg, str(e)[:80], "- waiting 60 s")
                time.sleep(60)
        if j is None:
            continue
        for p in j or []:
            did = f"globalvoices:{p['id']}"
            if ctx.pool.seen(did):
                continue
            auth = ""
            try:
                auth = p["_embedded"]["author"][0]["name"]
            except Exception:
                pass
            title = htmlmod.unescape(re.sub(r"<[^>]+>", "", p["title"]["rendered"]))
            yield dict(id=did, domain="news", licence="CC-BY-3.0", licence_class="permissive", licence_evidence=GV_EVIDENCE, url=p["link"], title=title,
                       attribution=f"{auth or 'Global Voices contributor'}, Global Voices, {p['link']}, CC BY 3.0", date=p["date"][:10],
                       paragraphs=gv_paragraphs(p["content"]["rendered"]),
                       licence_note="images and embedded third-party media are not under the same licence (removed); quoted tweets/posts removed")
        ctx.log("page", pg, "accepted", ctx.pool.n)


# ============================================================================================================== OpenStax (CC BY 4.0 snapshots)
import subprocess

OPENSTAX_REPOS = {
    # repo -> (domain, subject).  All OpenStax text books are 'education' unless they are clinical / health text books.
    "statistics": ("education", "statistics"), "physics": ("science_explainer", "physics"), "introduction-intellectual-property": ("law_government", "intellectual property"),
    "life-liberty-and-pursuit-happiness": ("education", "US history"), "introduction-business": ("education", "business"),
    "anatomy-physiology": ("health", "anatomy and physiology"), "microbiology": ("health", "microbiology"), "biology-bundle": ("science_explainer", "biology"),
    "astronomy": ("science_explainer", "astronomy"), "chemistry-bundle": ("science_explainer", "chemistry"), "psychology": ("education", "psychology"),
    "introduction-sociology": ("education", "sociology"), "american-government": ("law_government", "American government"),
    "us-history": ("education", "US history"), "world-history": ("education", "world history"), "introduction-anthropology": ("education", "anthropology"),
    "introduction-political-science": ("law_government", "political science"), "introduction-philosophy": ("education", "philosophy"),
    "principles-economics-bundle": ("education", "economics"), "principles-management-bundle": ("education", "management"), "entrepreneurship": ("education", "entrepreneurship"),
    "business-ethics": ("education", "business ethics"), "principles-marketing": ("education", "marketing"), "college-success-bundle": ("education", "college success"),
    "lifespan-development": ("health", "lifespan development"), "neuroscience": ("health", "neuroscience"), "nursing-external-bundle": ("health", "nursing"),
    "business-law": ("law_government", "business law"), "writing-guide": ("education", "writing"), "organic-chemistry": ("science_explainer", "organic chemistry"),
    "college-physics-bundle": ("science_explainer", "physics"), "university-physics-bundle": ("science_explainer", "physics"), "principles-finance": ("education", "finance"),
    "principles-accounting-bundle": ("education", "accounting"), "workplace-software-skills": ("education", "workplace skills"),
    "introduction-python-programming": ("education", "programming"), "principles-data-science": ("education", "data science"),
    "foundations-information-systems": ("education", "information systems"), "additive-manufacturing": ("education", "manufacturing"),
}
_CNX_DROP = {"figure", "table", "note", "exercise", "solution", "problem", "glossary", "equation", "media", "caption", "commentary", "example", "title",
             "metadata", "footnote", "code", "preformat", "definition", "rule", "proof", "statement", "image", "label", "abstract", "newline_", "quote_", "cnx-diagram"}
_CNX_MATH = {"math"}


def _local(tag) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


_CNX_BAD_START = re.compile(r"^(?:by the end of this (?:section|chapter)|key terms|learning objectives|for the following|use the following|answer the following|use the information|use the data|after completing this section|"
                            r"before reading this section|students will be able|please watch|watch the|click on|visit the|go to)", re.I)
_CNX_REF = re.compile(r"\b(?:Press|University|Publishers?|Books|Journal|Vol\.|pp\.|Routledge|Norton|Wiley|Oxford|Cambridge|Springer|Ed\.|eds?\.)\b")


def cnx_is_reference(p: str) -> bool:
    return bool(_CNX_REF.search(p)) and bool(re.search(r"\b(?:1[5-9]\d\d|20[0-2]\d)\b", p)) and p.count(",") >= 2 and len(p.split()) < 70 and not re.search(r"\b(?:is|was|were|are|has|have)\b", p[:200])


def cnxml_paragraphs(xml_bytes: bytes) -> Tuple[str, List[str]]:
    """(module title, prose paragraphs) of an OpenStax CNXML module; paragraphs with maths / media inside are dropped."""
    from lxml import etree
    root = etree.fromstring(xml_bytes, etree.XMLParser(recover=True, huge_tree=True))
    title = ""
    for t in root.iterchildren():
        if _local(t.tag) == "title":
            title = "".join(t.itertext()).strip()
    out: List[str] = []

    def text_of(el) -> Optional[str]:
        parts: List[str] = []

        def rec(e):
            n = _local(e.tag)
            if n in _CNX_MATH or n in ("media", "image", "equation", "footnote", "figure", "table", "code", "list"):
                parts.append("\u0000")            # poison marker -> the whole paragraph is dropped
                return
            if e.text:
                parts.append(e.text)
            for c in e:
                rec(c)
                if c.tail:
                    parts.append(c.tail)
        rec(el)
        s = "".join(parts)
        return None if "\u0000" in s else s

    def walk(el):
        n = _local(el.tag)
        if n in _CNX_DROP:
            return
        if n in ("para", "item") and not any(_local(c.tag) in ("list", "para") for c in el):
            s = text_of(el)
            if s and not _CNX_BAD_START.match(s.strip()) and not cnx_is_reference(s):
                if n == "item" and not re.search(r"[.!?][\"\u201d\u2019')]*\s*$", s.strip()):
                    return                                  # list item run-on / fragment without terminal punctuation
                out.append(s)
            return
        skip_list = False
        for c in el:
            if _local(c.tag) == "list" and skip_list:
                skip_list = False
                continue
            if _local(c.tag) == "para":
                t = "".join(c.itertext()).strip().lower()
                skip_list = t.startswith(("by the end of this", "after completing this", "learning objectives", "in this section", "you will be able to"))
            walk(c)

    for c in root.iterchildren():
        if _local(c.tag) == "content":
            walk(c)
    return title, out


def openstax_collection(xml_bytes: bytes) -> Tuple[str, List[Tuple[str, str, str]]]:
    """(book title, [(chapter title, module id, module title)]) in reading order from a *.collection.xml."""
    from lxml import etree
    root = etree.fromstring(xml_bytes, etree.XMLParser(recover=True))
    book = ""
    for el in root.iter():
        if _local(el.tag) == "title" and el.getparent() is not None and _local(el.getparent().tag) == "metadata":
            book = (el.text or "").strip()
            break
    mods: List[Tuple[str, str, str]] = []

    def rec(el, chapter: str):
        for c in el:
            n = _local(c.tag)
            if n == "subcollection":
                t = ""
                for tt in c.iter():
                    if _local(tt.tag) == "title":
                        t = (tt.text or "").strip()
                        break
                rec(c, t or chapter)
            elif n == "module":
                title = ""
                for tt in c.iter():
                    if _local(tt.tag) == "title":
                        title = (tt.text or "").strip()
                        break
                mods.append((chapter, c.get("document", ""), title))
            elif n == "content":
                rec(c, chapter)
    for c in root:
        if _local(c.tag) == "content":
            rec(c, "")
    return book, mods


def _git(args, cwd=None, check=True) -> str:
    r = subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True, env=dict(os.environ, GIT_TERMINAL_PROMPT="0"))
    if check and r.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr[:300]}")
    return r.stdout


@source("openstax", "education,health,science_explainer,law_government", "CC-BY-4.0 (git snapshot whose LICENSE file says Attribution 4.0)", "permissive",
        "https://github.com/openstax/ (LICENSE at the pinned commit)", "git clone --filter=blob:none (sparse: modules/ collections/) of github.com/openstax/osbooks-*")
def src_openstax(ctx: Ctx):
    """OpenStax textbook modules from the newest commit of each book repository whose LICENSE is CC BY 4.0 (OpenStax moved most books to
    CC BY-NC-SA 4.0 in 2025-26; the earlier text stays CC BY 4.0). --only current|snapshot selects books whose HEAD is CC BY / older snapshots."""
    a = ctx.args
    raw = os.path.join(ctx.raw_dir or os.path.join(os.path.dirname(os.path.abspath(a.out)), "..", "raw"), "openstax")
    os.makedirs(raw, exist_ok=True)
    want = (a.only or "current,snapshot").split(",")
    repos = sorted(OPENSTAX_REPOS)
    random.Random(a.seed).shuffle(repos)
    per_book = max(a.per_book, 1) if a.per_book != 3 else 40
    for repo in repos:
        if ctx.pool.full:
            break
        domain, subject = OPENSTAX_REPOS[repo]
        d = os.path.join(raw, repo)
        try:
            if not os.path.isdir(os.path.join(d, ".git")):
                time.sleep(1.0)
                _git(["clone", "--quiet", "--filter=blob:none", "--no-checkout", f"https://github.com/openstax/osbooks-{repo}.git", d])
            head = _git(["rev-parse", "HEAD"], cwd=d).strip()
            lic_head = _git(["show", "HEAD:LICENSE"], cwd=d, check=False).splitlines()[:1]
            kind, sha = None, None
            cc_by = lambda commit: (_git(["show", f"{commit}:LICENSE"], cwd=d, check=False).splitlines()[:1] or [""])[0].strip() == "Attribution 4.0 International"
            if cc_by("HEAD"):
                kind, sha = "current", head
            else:           # newest commit of the last period in which the LICENSE file said CC BY 4.0 (= the parent of the change to NC-SA)
                for c in _git(["log", "--format=%H", "--", "LICENSE"], cwd=d).split():
                    if not cc_by(c):
                        par = _git(["rev-parse", c + "^"], cwd=d, check=False).strip()
                        if par and cc_by(par):
                            kind, sha = "snapshot", par
                        break
            if not sha or kind not in want:
                ctx.log(repo, "skipped (no CC BY 4.0 snapshot)" if not sha else f"skipped ({kind} not selected)")
                continue
            _git(["sparse-checkout", "init", "--cone"], cwd=d, check=False)
            _git(["sparse-checkout", "set", "modules", "collections"], cwd=d)
            _git(["checkout", "--quiet", "-f", sha], cwd=d)
            when = _git(["show", "-s", "--format=%cs", sha], cwd=d).strip()
        except Exception as e:
            ctx.log(repo, "git failed:", e)
            continue
        cols = sorted(glob.glob(os.path.join(d, "collections", "*.collection.xml")))
        seen_mod = set()
        mods: List[Tuple[str, str, str, str]] = []
        for cpath in cols:
            btitle, ms = openstax_collection(open(cpath, "rb").read())
            for chap, mid, mtitle in ms:
                if mid and mid not in seen_mod:
                    seen_mod.add(mid)
                    mods.append((btitle, chap, mid, mtitle))
        rng = random.Random(f"{a.seed}-{repo}")
        rng.shuffle(mods)
        took = 0
        lic_url = f"https://github.com/openstax/osbooks-{repo}/blob/{sha}/LICENSE"
        for btitle, chap, mid, mtitle in mods:
            if took >= per_book or ctx.pool.full:
                break
            did = f"openstax:{repo}:{mid}"
            if ctx.pool.seen(did):
                took += 1
                continue
            p = os.path.join(d, "modules", mid, "index.cnxml")
            if not os.path.exists(p):
                continue
            title, paras = cnxml_paragraphs(open(p, "rb").read())
            note = ("OpenStax relicensed this book to CC BY-NC-SA 4.0 after this snapshot (LICENSE history); CC BY 4.0 is irrevocable for text distributed "
                    "under it, but the licensor's current offer is non-commercial: OWNER DECISION before release" if kind == "snapshot"
                    else "current OpenStax licence is CC BY 4.0; individual figures / third-party items may carry other licences (images are not included)")
            ok = ctx.pool.add(dict(
                id=did, domain=domain, licence="CC-BY-4.0", licence_class="permissive", licence_evidence=lic_url,
                url=f"https://github.com/openstax/osbooks-{repo}/blob/{sha}/modules/{mid}/index.cnxml",
                title=re.sub(r"^[^\w]+", "", f"{btitle}: {chap + ' - ' if chap else ''}{mtitle or title}")[:200],
                attribution=f"OpenStax, {btitle}, Rice University, CC BY 4.0 (https://openstax.org)", date=when, paragraphs=paras,
                licence_note=note, extra={"book": btitle, "subject": subject, "snapshot": kind, "commit": sha}))
            if ok:
                took += 1
        ctx.log(repo, kind, "docs", took)
        if not a.keep_raw:
            subprocess.run(["rm", "-rf", d])


# ===================================================================================================================== Our World in Data
OWID_EVIDENCE = "https://ourworldindata.org/faqs#can-i-use-our-content"
_OWID_END = re.compile(r"^(?:our articles and data visualizations rely on work|all visualizations, data, and code produced by our world in data|all visualizations, data|you have the permission to use|"
                       r"the data produced by third parties|all of our charts can be embedded|help us do this work|cite this work|reuse this work|"
                       r"we thank|we would like to thank|thanks to|i would like to thank|i thank|acknowledg)", re.I)
_OWID_HEALTH = re.compile(r"health|disease|death|mortality|malaria|cancer|vaccin|hiv|aids|child|infant|mental|life-expectancy|pandemic|covid|nutrition|hunger|smoking|"
                          r"alcohol|drug|maternal|birth|suicide|obesity|diabetes|tuberculosis|sanitation|water|pneumonia|diarrh", re.I)


def owid_paragraphs(html_text: str) -> Tuple[str, str, List[str]]:
    root = parse_html(html_text)
    if root is None:
        return "", "", []
    meta = lambda n: next(iter(root.xpath(f'//meta[@name="{n}" or @property="{n}"]/@content')), "") or ""
    title = htmlmod.unescape(meta("og:title")).replace(" - Our World in Data", "").replace(" | Our World in Data", "")
    date = (meta("article:published_time") or "")[:10]
    scope = root.xpath("//article") or root.xpath("//main") or root.xpath("//body")
    paras = html_paragraphs(scope, drop_classes=("chart", "grapher", "wp-block-owid", "footnote", "article-block__chart", "ref", "sources", "KeyChart", "key-insights",
                                                 "research-and-writing", "related", "cta", "newsletter", "article-footer", "site-footer", "toc", "sticky"))
    out: List[str] = []
    for p in paras:
        c = clean_paragraph(p)
        if _OWID_END.match(c):
            break
        out.append(p)
    return title, date, out


@source("owid", "science_explainer,health", "CC-BY-4.0", "permissive", OWID_EVIDENCE, "ourworldindata.org sitemap.xml + article HTML (1 req/s)")
def src_owid(ctx: Ctx):
    """Our World in Data articles / topic pages: text is CC BY 4.0 (third-party data excluded)."""
    sm = ctx.http.text("https://ourworldindata.org/sitemap.xml")
    urls = [u for u in re.findall(r"<loc>([^<]+)</loc>", sm) if u.count("/") == 3]
    skip = ("explorers", "search", "donate", "subscribe", "about", "team", "jobs", "hiring", "privacy", "funding", "faqs", "organization", "contact", "teaching",
            "charts", "blog", "latest", "data", "sdgs", "slideshows", "feedback", "cookies", "terms", "grapher", "api", "owid-grapher", "writing", "countries", "profile")
    urls = [u for u in sorted(urls) if u.rstrip("/").rsplit("/", 1)[-1] not in skip and "hiring" not in u and "we-published" not in u]
    random.Random(ctx.args.seed).shuffle(urls)
    for u in urls:
        did = "owid:" + u.rstrip("/").rsplit("/", 1)[-1]
        if ctx.pool.seen(did):
            continue
        try:
            h = ctx.http.text(u, allow_404=True)
        except Exception as e:
            ctx.log("skip", u, e)
            continue
        if not h:
            continue
        title, date, paras = owid_paragraphs(h)
        dom = "health" if _OWID_HEALTH.search(u) else "science_explainer"
        yield dict(id=did, domain=dom, licence="CC-BY-4.0", licence_class="permissive", licence_evidence=OWID_EVIDENCE, url=u, title=title,
                   attribution="Our World in Data (Max Roser, Hannah Ritchie and colleagues; see the page for the authors), CC BY 4.0, " + u, date=date, paragraphs=paras,
                   licence_note="text and charts CC BY 4.0; data produced by third parties follows its original licence (data and charts are not included)")


# ================================================================================================================================ NASA
NASA_EVIDENCE = "https://www.nasa.gov/nasa-brand-center/images-and-media/"
_NASA_BAD = re.compile(r"^(?:media contacts?|news media contact|contacts?:|for more information|for news media|-end-|explore more|related terms|share|details|last updated|"
                       r"credits?:|image credit|nasa earth observatory (?:image|map)|story by|caption by|images? (?:by|courtesy)|references?:|more information|learn more|visit|follow|read more|sign up|about nasa|nasa\.gov)(?![A-Za-z0-9])", re.I)


def nasa_paragraphs(html_text: str) -> List[str]:
    root = parse_html("<div>" + html_text + "</div>")
    if root is None:
        return []
    paras = html_paragraphs(root, drop_classes=("wp-block-nasa", "hds-", "caption", "credit", "gallery", "media-contact", "related", "wp-block-gallery",
                                                "wp-block-embed", "share", "button", "wp-block-button", "wp-element-caption", "wp-block-image"))
    out: List[str] = []
    for p in paras:
        c = clean_paragraph(p)
        if re.match(r"^(?:-end-|-30-|###)", c):
            break
        if re.match(r"^(?:media contacts?|news media contact)", c, re.I):
            break
        if _NASA_BAD.match(c) or re.search(r"\bCaption by\b|NASA Earth Observatory (?:image|map)", c):
            continue
        out.append(p)
    return out


@source("nasa", "science_explainer", "Public domain (US federal government work; third-party credits excluded)", "pd", NASA_EVIDENCE,
        "WordPress REST API of science.nasa.gov and www.nasa.gov (wp-json/wp/v2/posts)")
def src_nasa(ctx: Ctx):
    """NASA articles (science.nasa.gov, nasa.gov): generally not copyrighted (US government work), except where credited to third parties."""
    hosts = ["https://science.nasa.gov", "https://www.nasa.gov"]
    params = {"per_page": 100, "_fields": "id,date,link,title,content"}
    plans = []
    for h in hosts:
        r = ctx.http.get(h + "/wp-json/wp/v2/posts", params=dict(params, page=1))
        n = int(r.headers.get("X-WP-TotalPages", "1"))
        pages = list(range(1, n + 1))
        random.Random(ctx.args.seed + len(h)).shuffle(pages)
        plans.append((h, iter(pages)))
    alive = list(plans)
    while alive:
        for h, it in list(alive):
            try:
                pg = next(it)
            except StopIteration:
                alive.remove((h, it))
                continue
            try:
                j = ctx.http.json(h + "/wp-json/wp/v2/posts", params=dict(params, page=pg))
            except Exception as e:
                ctx.log("page", h, pg, e)
                continue
            for p in j or []:
                did = f"nasa:{urlparse(h).netloc.split('.')[0]}-{p['id']}"
                if ctx.pool.seen(did) or "/blog/" in p["link"]:
                    continue
                title = htmlmod.unescape(re.sub(r"<[^>]+>", "", p["title"]["rendered"]))
                yield dict(id=did, domain="science_explainer", licence="Public domain (US government work)", licence_class="pd", licence_evidence=NASA_EVIDENCE,
                           url=p["link"], title=title, attribution="NASA (" + p["link"] + ")", date=p["date"][:10], paragraphs=nasa_paragraphs(p["content"]["rendered"]),
                           licence_note="NASA text is generally not copyrighted; material credited to ESA, CSA, JAXA, universities or contractors may be; images are not included")


# ===================================================================================================== US federal health sites (sitemap crawl)
_SITE_STOP = re.compile(r"^(?:last reviewed|last updated|reviewed|this content is provided as a service|niddk would like to thank|clinical trials|publications?$|source:|"
                        r"for more information|related (?:topics|information|resources)|resources$|share this|print this|references?$|find a clinical|content last reviewed|"
                        r"the nimh is part of|nimh is part of|this publication is in the public domain|reprints|if you reuse|please cite)", re.I)
_SITE_SKIP = re.compile(r"^(?:the national institute of mental health archives|nimh archives|skip to|an official website|here's how you know|official websites use|secure \.gov|share sensitive|nih websites are changing|per executive order|"
                        r"español|translate|print|email|facebook|twitter|linkedin|related|back to top|on this page|jump to|table of contents|contact us)", re.I)

NPS_SITEMAPS = ["https://www.nps.gov/sitemap.xml"]          # an index: expanded by the fetcher
NPS_EVIDENCE = "https://www.nps.gov/aboutus/disclaimer.htm"
NPS_NOTE = "NPS: material created by the National Park Service is 'generally considered in the public domain' unless otherwise indicated; photographs and quoted material may be third-party"
FEDSITES = {
    # key: (domain, sitemap urls, include regex, exclude regex, content xpath, evidence url, attribution, licence note)
    "niddk": ("health", ["https://www.niddk.nih.gov/sitemap-sc.xml"], r"/health-information/(?:digestive-diseases|liver-disease|urologic-diseases|kidney-disease|diabetes|weight-management)/",
              r"/professionals|/health-statistics|/clinical-trials|/research-areas", "https://www.niddk.nih.gov/copyright",
              "National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK), National Institutes of Health",
              "NIDDK: 'majority of information on this site is copyright free'; documents co-sponsored with private organisations and some graphics may be third-party; logos must be removed from edited content; no implied endorsement"),
    "nimh-health": ("health", ["https://www.nimh.nih.gov/sites/default/files/sitemap/sitemap.xml"], r"/health/(?:topics|publications)/",
                    r"/health/publications/?$|/health/topics/?$", "https://www.nimh.nih.gov/about/copyright-guidelines",
                    "National Institute of Mental Health (NIMH), National Institutes of Health", "NIMH publications are in the public domain; some images / quoted material may be copyrighted"),
    "nimh-news": ("science_explainer", ["https://www.nimh.nih.gov/sites/default/files/sitemap/sitemap.xml"], r"/news/science-updates/", r"/index",
                  "https://www.nimh.nih.gov/about/copyright-guidelines", "National Institute of Mental Health (NIMH), National Institutes of Health",
                  "NIMH science updates are US government text; contributed images and quoted journal text may be third-party"),
    "nida": ("health", ["https://nida.nih.gov/sitemap.xml"], r"/publications/(?:research-reports|drugfacts)/|/research-topics/", r"/es/|/index$", "https://nida.nih.gov/about-nida/copyright-policy",
             "National Institute on Drug Abuse (NIDA), National Institutes of Health", "NIDA text is public domain unless noted; some graphics are third-party"),
    "nps-history": ("encyclopedia", NPS_SITEMAPS, r"nps\.gov/[a-z0-9]+/learn/historyculture/", r"/index\.htm$|/learn/historyculture/?$", NPS_EVIDENCE,
                    "National Park Service (NPS), U.S. Department of the Interior", NPS_NOTE),
    "nps-nature": ("science_explainer", NPS_SITEMAPS, r"nps\.gov/[a-z0-9]+/learn/nature/", r"/index\.htm$", NPS_EVIDENCE,
                   "National Park Service (NPS), U.S. Department of the Interior", NPS_NOTE),
    "nps-articles": ("encyclopedia", NPS_SITEMAPS, r"nps\.gov/articles/\d+/", r"/index\.htm$", NPS_EVIDENCE,
                     "National Park Service (NPS), U.S. Department of the Interior", NPS_NOTE),
}


def fed_paragraphs(html_text: str) -> Tuple[str, str, List[str]]:
    root = parse_html(html_text)
    if root is None:
        return "", "", []
    title = " ".join((root.xpath("string(//h1)") or root.xpath("string(//title)") or "").split())
    date = ""
    for q in ('//meta[@property="article:published_time"]/@content', '//meta[@name="dcterms.issued"]/@content', '//meta[@name="date"]/@content', "//time/@datetime"):
        v = root.xpath(q)
        if v:
            date = v[0][:10]
            break
    scope = root.xpath("//main") or root.xpath('//*[@role="main"]') or root.xpath("//article") or root.xpath("//body")
    paras = html_paragraphs(scope, drop_classes=("breadcrumb", "share", "social", "print", "related", "sidebar", "toc", "on-this-page", "feedback", "nav", "menu", "footer",
                                                  "callout", "alert", "banner", "usa-", "pager", "pagination", "contact", "newsletter", "subscribe", "language"))
    out: List[str] = []
    seen = set()
    for p in paras:
        c = clean_paragraph(p)
        if _SITE_STOP.match(c):
            break
        if _SITE_SKIP.match(c) or c in seen:
            continue
        seen.add(c)
        out.append(p)
    return title, date, out


@source("fedhealth", "health,science_explainer,encyclopedia", "Public domain (US federal government text)", "pd", "per site: see FEDSITES (copyright / reuse pages)",
        "sitemap.xml + page HTML of niddk.nih.gov, nimh.nih.gov, nida.nih.gov (1 req/s)")
def src_fedhealth(ctx: Ctx):
    """NIH institute health pages (NIDDK, NIMH, NIDA): US government text, public domain unless noted."""
    a = ctx.args
    keys = (a.only.split(",") if a.only else list(FEDSITES))
    queues: Dict[str, List[str]] = {}
    for k in keys:
        domain, sitemaps, inc, exc, ev, attr, note = FEDSITES[k]
        urls: List[str] = []
        for sm in sitemaps:
            t = ctx.http.text(sm) or ""
            locs = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", t)
            if "<sitemapindex" in t:                        # expand a sitemap index (one level)
                for child in locs:
                    try:
                        urls += re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", ctx.http.text(child, allow_404=True) or "")
                    except Exception as e:
                        ctx.log("sitemap", child, str(e)[:80])
            else:
                urls += locs
        urls = sorted(set(u for u in urls if re.search(inc, u) and not re.search(exc, u)))
        random.Random(a.seed + len(k)).shuffle(urls)
        queues[k] = urls
        ctx.log(k, len(urls), "candidate urls")
    active = list(queues)
    idx = Counter()
    while active:
        for k in list(active):
            if idx[k] >= len(queues[k]):
                active.remove(k)
                continue
            u = queues[k][idx[k]]
            idx[k] += 1
            domain, _, _, _, ev, attr, note = FEDSITES[k]
            did = f"{k}:" + re.sub(r"^https?://[^/]+/", "", u).strip("/").replace("/", "~")
            if ctx.pool.seen(did):
                continue
            try:
                h = ctx.http.text(u, allow_404=True)
            except Exception as e:
                ctx.log("skip", u, str(e)[:100])
                continue
            if not h:
                continue
            title, date, paras = fed_paragraphs(h)
            yield dict(id=did, domain=domain, licence="Public domain (US federal government work)", licence_class="pd", licence_evidence=ev, url=u, title=title,
                       attribution=attr + ", " + u, date=date, paragraphs=paras, licence_note=note)


# ============================================================================================================== law / government sources
def window_paragraphs(paras: List[str], tc: TokenCounter, start: int, target_tokens: int) -> List[str]:
    """Consecutive paragraphs from `start` until ~`target_tokens` Gemma tokens are reached."""
    out, tot = [], 0
    for p in paras[start:]:
        out.append(p)
        tot += tc.count(p) + 2
        if tot >= target_tokens:
            break
    return out


LEG_OGL = "https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/"


def legislation_note_paragraphs(html_text: str) -> List[str]:
    root = parse_html(html_text)
    if root is None:
        return []
    scope = root.xpath('//div[@id="viewLegContents"]') or root.xpath("//main") or root.xpath("//body")
    paras = html_paragraphs(scope, drop_classes=("LegNavigation", "breadcrumb", "pageTitle", "prevNext", "help", "tools", "print"))
    out = []
    for p in paras:
        c = re.sub(r"^\s*\d{1,4}\s*\.\s*", "", clean_paragraph(p))
        if re.match(r"^(?:section|schedule|part|article|regulation|paragraph) [\dA-Z]+[:\s]", c, re.I) and len(c) < 160:
            continue
        out.append(c)
    return out


@source("legislation", "law_government", "OGL-3.0 (Open Government Licence v3.0)", "permissive", "https://www.legislation.gov.uk/ (footer of every page) + " + LEG_OGL,
        "legislation.gov.uk sitemap-ukpga.xml + <act>/notes/data.htm (Explanatory Notes; robots Crawl-delay 5)")
def src_legislation(ctx: Ctx):
    """Explanatory Notes of UK Public General Acts (plain-English commentary on the sections), OGL v3."""
    a = ctx.args
    cand: List[Tuple[str, int]] = []
    for year in range(2000, 2027):        # Explanatory Notes exist for government Acts since 1999; public general acts are numbered 1..N in a year
        try:
            feed = ctx.http.text(f"https://www.legislation.gov.uk/ukpga/{year}/data.feed", allow_404=True) or ""
        except Exception as e:
            ctx.log("feed", year, str(e)[:80])
            continue
        m = re.search(r'facetType type="UnitedKingdomPublicGeneralAct"[^>]*value="(\d+)"', feed)
        for n in range(1, (int(m.group(1)) if m else 0) + 1):
            cand.append((f"ukpga/{year}/{n}", year))
    cand = sorted(set(cand))
    ctx.log("ukpga acts since 1999:", len(cand))
    random.Random(a.seed).shuffle(cand)
    tc = ctx.pool.tc
    for path, year in cand:
        base = f"legislation:{path.replace('/', '-')}"
        if ctx.pool.seen(base + "-a"):
            continue
        try:
            h = ctx.http.text(f"https://www.legislation.gov.uk/{path}/notes/data.htm", allow_404=True)
        except Exception as e:
            ctx.log("skip", path, str(e)[:80])
            continue
        if not h:
            continue
        title = ""
        m = re.search(r"<title>([^<]+)</title>", h)
        if m:
            title = htmlmod.unescape(m.group(1)).split(" - ")[0].replace("Explanatory Notes to the ", "").strip()
        paras = [p for p in legislation_note_paragraphs(h)]
        paras = clean_paragraphs(paras)
        if len(paras) < 12:
            continue
        rng = random.Random(f"{a.seed}-{path}")
        n = len(paras)
        starts = [min(6, n // 10), rng.randrange(n // 3, max(n // 3 + 1, 2 * n // 3))]
        for k, (label, st, target) in enumerate((("a", starts[0], 4500), ("b", starts[1], 2600))):
            win = window_paragraphs(paras, tc, st, target)
            yield dict(id=f"{base}-{label}", domain="law_government", licence="OGL-3.0", licence_class="permissive",
                       licence_evidence=f"https://www.legislation.gov.uk/{path}/notes ('All content is available under the Open Government Licence v3.0 except where otherwise stated')",
                       url=f"https://www.legislation.gov.uk/{path}/notes", title=f"Explanatory Notes: {title}", attribution=f"Crown copyright, legislation.gov.uk, {title}, Open Government Licence v3.0",
                       date=str(year), paragraphs=win,
                       licence_note="OGL v3 except where stated; the site also contains EUR-Lex derived content (EU-origin legislation) under Decision 2011/833/EU")


FR_EVIDENCE = "https://www.govinfo.gov/about/policies"
_FR_SKIP = re.compile(r"^(?:authority:|start printed page|dated:|billing code|fr doc|\[fr doc|addresses:?|dates:?|for further information|docket|rin\b|"
                      r"comments must be|list of subjects|accordingly|you may submit|we will post|electronic submission|written comments|public comments|"
                      r"submit comments|instructions:|supplementary information:?|summary:?|agency:?|action:?|\*\s*\*\s*\*)", re.I)


def fr_paragraphs(html_text: str) -> List[str]:
    root = parse_html(html_text)
    if root is None:
        return []
    paras = html_paragraphs(root, drop_classes=("document-headings", "fr-box", "fr-seal", "fr-seal-block", "printed-page", "footnote", "table", "gpotbl", "graphic", "img"))
    out = []
    for p in paras:
        c = clean_paragraph(p)
        if _FR_SKIP.match(c) or re.search(r"\b\d+ (?:FR|CFR) \d+\b.*\b\d+ (?:FR|CFR) \d+\b.*\b\d+ (?:FR|CFR)", c):
            continue
        out.append(p)
    return out


@source("federalregister", "law_government", "Public domain (US federal government work)", "pd", FR_EVIDENCE,
        "federalregister.gov API v1 (documents.json by random publication date) + full_text/html")
def src_federalregister(ctx: Ctx):
    """Federal Register rules, proposed rules and notices (preambles explain the regulation in prose); US government works."""
    a = ctx.args
    rng = random.Random(a.seed)
    import datetime
    d0, d1 = datetime.date(2003, 1, 2), datetime.date(2026, 9, 30)
    days = [d0 + datetime.timedelta(days=i) for i in range((d1 - d0).days) if (d0 + datetime.timedelta(days=i)).weekday() < 5]
    rng.shuffle(days)
    per_day = 3
    for day in days:
        try:
            j = ctx.http.json("https://www.federalregister.gov/api/v1/documents.json", params={
                "per_page": 100, "conditions[publication_date][is]": day.isoformat(), "conditions[type][]": ["RULE", "PRORULE", "NOTICE"],
                "fields[]": ["title", "body_html_url", "document_number", "publication_date", "agencies", "type", "page_length", "html_url"]})
        except Exception as e:
            ctx.log("day", day, str(e)[:80])
            continue
        rows = [r for r in j.get("results", []) if (r.get("page_length") or 0) >= 4 and r.get("body_html_url")
                and not re.match(r"(?:Self-Regulatory Organizations|Sunshine Act|Airworthiness Directives|Privacy Act of 1974|Agency Information Collection)", r.get("title", ""))]
        rows.sort(key=lambda r: r["document_number"])
        random.Random(f"{a.seed}-{day}").shuffle(rows)
        took = 0
        for r in rows:
            if took >= per_day:
                break
            did = "federalregister:" + r["document_number"]
            if ctx.pool.seen(did):
                continue
            try:
                h = ctx.http.text(r["body_html_url"], allow_404=True)
            except Exception as e:
                continue
            if not h:
                continue
            ags = "; ".join(x.get("name", x.get("raw_name", "")) for x in (r.get("agencies") or []))
            yield dict(id=did, domain="law_government", licence="Public domain (US federal government work)", licence_class="pd", licence_evidence=FR_EVIDENCE,
                       url=r.get("html_url", ""), title=r["title"], attribution=f"{ags or 'US federal agency'}, Federal Register {r['publication_date']} ({r['document_number']})",
                       date=r["publication_date"], paragraphs=fr_paragraphs(h),
                       licence_note="US government work; documents may quote third-party comments or incorporate copyrighted standards by reference")
            took += 1


# ======================================================================================================= Wikipedia (HF wikimedia/wikipedia)
WP_REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"       # wikimedia/wikipedia, 20231101.en (pinned)
WP_EVIDENCE = "https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use"


def sampled(key: str, seed: int, frac: float) -> bool:
    if frac >= 1.0:
        return True
    return int(hashlib.sha1(f"{seed}:{key}".encode()).hexdigest()[:8], 16) / 2 ** 32 < frac


@source("wikipedia", "encyclopedia", "CC-BY-SA-4.0 / CC-BY-SA-3.0 and GFDL (dual)", "share_alike", WP_EVIDENCE,
        "HF dataset wikimedia/wikipedia 20231101.en parquet shards (pinned revision), hash-sampled")
def src_wikipedia(ctx: Ctx):
    """English Wikipedia articles (share-alike; contamination-flagged later: Wikipedia is in the SFT/RL English source)."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    a = ctx.args
    shards = [int(x) for x in (a.only or "3,14,22,31").split(",")]
    raw = os.path.join(ctx.raw_dir or os.path.join(os.path.dirname(os.path.abspath(a.out)), "..", "raw"), "wikipedia")
    os.makedirs(raw, exist_ok=True)
    for sh in shards:
        fn = f"20231101.en/train-{sh:05d}-of-00041.parquet"
        path = hf_hub_download("wikimedia/wikipedia", fn, repo_type="dataset", revision=WP_REVISION, local_dir=raw)
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=2000, columns=["id", "url", "title", "text"]):
            for row in batch.to_pylist():
                title, text = row["title"] or "", row["text"] or ""
                if len(text) < 3200 or re.match(r"(?:List of|Lists of|Index of|Outline of|Timeline of)", title) or not sampled(row["id"], a.seed, a.sample):
                    continue
                did = f"wikipedia:{row['id']}"
                if ctx.pool.seen(did):
                    continue
                yield dict(id=did, domain="encyclopedia", licence="CC-BY-SA-4.0 / CC-BY-SA-3.0 + GFDL", licence_class="share_alike", licence_evidence=WP_EVIDENCE,
                           url=row["url"], title=title, attribution=f'Wikipedia contributors, "{title}", en.wikipedia.org (revision of the 2023-11-01 dump), CC BY-SA',
                           date="2023-11-01", paragraphs=[p for p in text.split("\n") if p.strip()],
                           licence_note="share-alike: derivatives (translations) must carry CC BY-SA; text may include CC BY-SA 3.0 imported material and non-free quotes")
        if not a.keep_raw:
            try:
                os.remove(path)
            except OSError:
                pass


# ======================================================================================================================= pool utilities
BINS = ((400, 800, "400-800"), (800, 1600, "800-1600"), (1600, 3200, "1600-3200"), (3200, 10 ** 9, ">=3200"))
# all length ranges incl. the short bins of the v3 pool files (Gemma tokens): 20-99 ~ b64/b256 items, 100-399 ~ b256/b512, then as BINS
RANGES = ((0, 20, "<20"), (20, 100, "20-99"), (100, 400, "100-399"), (400, 800, "400-799"), (800, 1600, "800-1599"), (1600, 3200, "1600-3199"), (3200, 10 ** 9, ">=3200"))


def quantiles(xs: List[int]) -> Dict[str, int]:
    if not xs:
        return {}
    ys = sorted(xs)
    q = lambda f: ys[min(len(ys) - 1, int(f * len(ys)))]
    return {"min": ys[0], "p10": q(0.1), "p25": q(0.25), "median": q(0.5), "p75": q(0.75), "p90": q(0.9), "max": ys[-1]}


def pool_stats(pool_dir: str, samples_dir: Optional[str] = None) -> dict:
    """Per-file statistics of the *.jsonl pool files (docs, token quantiles, docs per length bin, licence, class, domains) + domain x bin x class tables."""
    files = sorted(f for f in glob.glob(os.path.join(pool_dir, "*.jsonl")))
    out: dict = {"files": {}, "domain_bin_class": {}}
    table: Dict[str, Counter] = {}
    for f in files:
        toks, doms, lic, cls, srcs = [], Counter(), Counter(), Counter(), Counter()
        bins = Counter()
        rng = Counter()
        sample_docs = []
        n = 0
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                n += 1
                t = d["n_gemma_tokens"]
                toks.append(t)
                doms[d["domain"]] += 1
                lic[d["licence"]] += 1
                cls[d["licence_class"]] += 1
                srcs[d["source"]] += 1
                for lo, hi, name in RANGES:
                    if lo <= t < hi:
                        rng[name] += 1
                for lo, hi, name in BINS:
                    if lo <= t < hi:
                        bins[name] += 1
                        table.setdefault(f"{d['domain']}|{d['licence_class']}", Counter())[name] += 1
                if len(sample_docs) < 3 and 200 < len(d["text"]):
                    sample_docs.append(d)
        key = os.path.basename(f)
        out["files"][key] = {"docs": n, "tokens": quantiles(toks), "docs_per_bin": dict(bins), "bytes": os.path.getsize(f), "domains": dict(doms),
                             "licences": dict(lic), "licence_classes": dict(cls), "sources": dict(srcs),
                             "docs_per_range": {name: rng.get(name, 0) for _, _, name in RANGES}}
        if samples_dir and sample_docs:
            os.makedirs(samples_dir, exist_ok=True)
            with open(os.path.join(samples_dir, key.split("__")[0] + ".jsonl"), "w", encoding="utf-8") as sf:
                for d in sample_docs:
                    e = dict(d)
                    e["text"] = d["text"][:600]
                    sf.write(json.dumps(e, ensure_ascii=False) + "\n")
    for k, c in sorted(table.items()):
        out["domain_bin_class"][k] = {name: c.get(name, 0) for _, _, name in BINS}
    return out


def retokenize(inp: str, out: str, tokenizer: Optional[str]) -> int:
    """Re-run the paragraph cleaner over a pool file (picks up cleaner fixes made after the crawl), recount the Gemma tokens, drop the documents
    that fall below the limits and the exact duplicates, and write the result atomically."""
    tc = TokenCounter(tokenizer)
    if not tc.exact:
        print("[retokenize] no Gemma tokenizer found", file=sys.stderr)
        return 1
    n = kept = 0
    seen = set()
    with open(inp, encoding="utf-8") as f, open(out + ".part", "w", encoding="utf-8") as g:
        for line in f:
            d = json.loads(line)
            n += 1
            paras = clean_paragraphs(re.sub(r"(?<!\n)\n(?!\n)", " ", d["text"]).split("\n\n"))
            if len(paras) < 3:
                continue
            paras, trunc = cut_to_tokens(paras, tc)
            text = "\n\n".join(paras)
            fp = fingerprint(text)
            if fp in seen or not english_ok(text):
                continue
            seen.add(fp)
            d["text"] = text
            d["n_chars"] = len(text)
            d["n_gemma_tokens"] = tc.count(text)
            if trunc or d.get("truncated"):
                d["truncated"] = True
            if d["n_gemma_tokens"] >= MIN_TOKENS:
                g.write(json.dumps(d, ensure_ascii=False) + "\n")
                kept += 1
    os.replace(out + ".part", out)
    print(json.dumps({"in": inp, "out": out, "docs_in": n, "docs_out": kept}))
    return 0


# ====================================================================================================================== PMC open access (CC BY)
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
PMC_TOPICS = [
    # (domain, europe PMC query terms)
    ("health", "malaria AND Africa"), ("health", "maternal health AND Africa"), ("health", "HIV AND community AND Africa"), ("health", "tuberculosis"),
    ("health", "childhood vaccination"), ("health", "child malnutrition"), ("health", "mental health AND community"), ("health", "COVID-19 AND response AND Africa"),
    ("health", "primary health care"), ("health", "diabetes AND self-management"), ("health", "hypertension AND community"), ("health", "cervical cancer screening"),
    ("health", "antibiotic resistance"), ("health", "neglected tropical diseases"), ("health", "water sanitation hygiene"), ("health", "community health workers"),
    ("health", "health system financing"), ("health", "nursing AND patients"), ("health", "traditional medicine"), ("health", "breastfeeding"),
    ("science_explainer", "climate change AND agriculture AND Africa"), ("science_explainer", "smallholder farmers AND crop yield"), ("science_explainer", "soil fertility"),
    ("science_explainer", "biodiversity conservation"), ("science_explainer", "renewable energy"), ("science_explainer", "food security"), ("science_explainer", "drought AND adaptation"),
    ("science_explainer", "fisheries"), ("science_explainer", "forest AND communities"), ("science_explainer", "artificial intelligence AND society"),
    ("science_explainer", "mobile phones AND development"), ("science_explainer", "urbanization"), ("science_explainer", "air pollution"), ("science_explainer", "plastic waste"),
    ("science_explainer", "livestock AND pastoralists"), ("science_explainer", "wildlife AND tourism"),
    ("education", "education quality AND low-income countries"), ("education", "teacher training"), ("education", "school feeding"), ("education", "literacy AND children"),
    ("education", "university students AND learning"),
    ("law_government", "health policy AND legislation"), ("law_government", "human rights AND law AND Africa"), ("law_government", "governance AND corruption"),
    ("law_government", "refugees AND policy"), ("law_government", "land tenure"), ("law_government", "public administration"),
]
_PMC_SKIP_SEC = re.compile(r"method|materials|statistic|data (?:collection|analysis)|results?|supplement|ethic|acknowledg|funding|competing|conflict|author|abbreviation|reference|"
                           r"appendix|limitation|availability|consent|participants|measures|procedure|design|setting|search strategy|inclusion|exclusion|sampling|variables|instrument", re.I)


def jats_paragraphs(xml_text: str) -> Tuple[str, str, List[str], str]:
    """(title, licence url, paragraphs, doi) of a JATS full-text article: abstract + the body sections that are not methods / results / back matter."""
    from lxml import etree
    root = etree.fromstring(xml_text.encode("utf-8"), etree.XMLParser(recover=True, huge_tree=True))
    title = " ".join("".join(root.xpath("//article-meta/title-group/article-title")[0].itertext()).split()) if root.xpath("//article-meta/title-group/article-title") else ""
    lic = (root.xpath("//article-meta//*[local-name()='license_ref']/text()") or root.xpath("//article-meta//license/@*[local-name()='href']") or [""])[0].strip()
    doi = (root.xpath("//article-meta/article-id[@pub-id-type='doi']/text()") or [""])[0]

    def ptext(p) -> Optional[str]:
        parts: List[str] = []

        def rec(e):
            t = _local(e.tag)
            if t in ("xref", "inline-formula", "disp-formula", "table-wrap", "fig", "supplementary-material", "mml:math", "math", "sup"):
                if t in ("inline-formula", "disp-formula", "math"):
                    parts.append("\u0000")
                return
            if e.text:
                parts.append(e.text)
            for c in e:
                rec(c)
                if c.tail:
                    parts.append(c.tail)
        rec(p)
        s = "".join(parts)
        return None if "\u0000" in s else s

    out: List[str] = []
    for p in root.xpath("//article-meta/abstract[not(@abstract-type)]//p | //article-meta/abstract[not(@abstract-type)]//p"):
        s = ptext(p)
        if s:
            out.append(s)
    for sec in root.xpath("//body/sec"):
        st = " ".join("".join(sec.xpath("./title")[0].itertext()).split()) if sec.xpath("./title") else ""
        if _PMC_SKIP_SEC.search(st) or _PMC_SKIP_SEC.search(sec.get("sec-type") or ""):
            continue
        for p in sec.xpath(".//p[not(ancestor::table-wrap) and not(ancestor::fig) and not(ancestor::boxed-text)]"):
            s = ptext(p)
            if s:
                out.append(s)
    return title, lic, out, doi


def pmc_licence(url: str) -> Tuple[str, str]:
    m = re.search(r"creativecommons\.org/(licenses|publicdomain)/([a-z\-]+)/([\d.]+)", url or "")
    if not m:
        return "CC-BY (version not stated in the article)", "permissive"
    kind, ver = m.group(2), m.group(3)
    if kind == "by":
        return f"CC-BY-{ver}", "permissive"
    if kind in ("zero", "mark"):
        return "CC0-1.0", "pd"
    return f"CC-{kind.upper()}-{ver}", "noncommercial" if ("nc" in kind or "nd" in kind) else "share_alike"


@source("pmc", "health,science_explainer,education,law_government", "CC-BY (any version, per article) / CC0", "permissive", EPMC + " (licence URL inside every article's JATS <license>)",
        "Europe PMC REST search (LICENSE:'cc by') + fullTextXML (1 req/s)")
def src_pmc(ctx: Ctx):
    """Open-access research / review articles under CC BY from PubMed Central via Europe PMC (introduction / discussion / conclusion prose)."""
    a = ctx.args
    topics = PMC_TOPICS if not a.only else [t for t in PMC_TOPICS if t[0] in a.only.split(",")]
    per_topic = max(3, a.max_docs // len(topics) + 2)
    base = 'LICENSE:"cc by" AND OPEN_ACCESS:y AND SRC:MED AND HAS_FT:Y AND PUB_YEAR:[2012 TO 2025] AND LANG:eng AND ('
    iters = []
    for dom, terms in topics:
        iters.append((dom, terms, iter(()), 0))
    got = Counter()
    state = {}
    cursors = {terms: "*" for _, terms in topics}
    active = [t for t in topics]
    while active and not ctx.pool.full:
        for dom, terms in list(active):
            if ctx.pool.full:
                break
            if got[terms] >= per_topic:
                active.remove((dom, terms))
                continue
            q = base + terms + ")"
            try:
                j = ctx.http.json(EPMC + "/search", params={"query": q, "format": "json", "pageSize": 25, "resultType": "core", "cursorMark": cursors[terms]})
            except Exception as e:
                ctx.log("search", terms, str(e)[:100])
                active.remove((dom, terms))
                continue
            res = j.get("resultList", {}).get("result", [])
            nxt = j.get("nextCursorMark")
            if not res or not nxt or nxt == cursors[terms]:
                active.remove((dom, terms))
                continue
            cursors[terms] = nxt
            for r in res:
                pm = r.get("pmcid")
                if not pm or got[terms] >= per_topic or ctx.pool.full:
                    continue
                did = f"pmc:{pm}"
                if ctx.pool.seen(did):
                    got[terms] += 1
                    continue
                try:
                    x = ctx.http.text(f"{EPMC}/{pm}/fullTextXML", allow_404=True)
                except Exception as e:
                    continue
                if not x:
                    continue
                title, lic_url, paras, doi = jats_paragraphs(x)
                lic, cls = pmc_licence(lic_url)
                if cls != "permissive" and cls != "pd":
                    continue
                ok = ctx.pool.add(dict(
                    id=did, domain=dom, licence=lic, licence_class=cls, licence_evidence=lic_url or f"https://europepmc.org/article/PMC/{pm.replace('PMC', '')}",
                    url=f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pm}/", title=title or r.get("title", ""),
                    attribution=f"{r.get('authorString', '')} ({r.get('pubYear', '')}). {title}. {r.get('journalTitle', '')}. {('doi:' + doi) if doi else pm}. {lic}",
                    date=(r.get("firstPublicationDate") or r.get("pubYear") or "")[:10], paragraphs=paras,
                    licence_note="licence read from the article XML; individual figures / third-party material inside an article may carry other licences (not included)"))
                if ok:
                    got[terms] += 1


# ============================================================================================ Encyclopaedia Britannica 11th ed. (Gutenberg)
_EB_START = re.compile(r"^([A-Z][A-Z'’\-]{2,}(?:(?:,| of|\s)\s*[A-Z][A-Z'’\-]*)*)\s*[(,.—]")


def eb_articles(raw: str) -> List[Tuple[str, List[str]]]:
    """Split an Encyclopaedia Britannica (11th ed.) Gutenberg slice into (headword, paragraphs) articles."""
    out: List[Tuple[str, List[str]]] = []
    head, cur = "", []
    for text, _ in gutenberg_paragraphs(raw):
        m = _EB_START.match(text)
        if m and len(m.group(1)) <= 60 and sum(c.isupper() for c in text[:40]) >= 6:
            if head and cur:
                out.append((head, cur))
            head, cur = m.group(1).title(), [text]
        elif head:
            cur.append(text)
    if head and cur:
        out.append((head, cur))
    return out


@source("eb1911", "encyclopedia", "Public domain (published 1910-1911)", "pd", PG_EVIDENCE,
        "gutenberg.org/cache/epub/<id>/pg<id>.txt for the 'Encyclopaedia Britannica, 11th Edition' slices (1 req/s)")
def src_eb1911(ctx: Ctx):
    """Long articles of the Encyclopaedia Britannica, 11th edition (1910-11), public domain, from Project Gutenberg slices."""
    a = ctx.args
    cat = ctx.http.text("https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv")
    rows = [r for r in csv.DictReader(io.StringIO(cat)) if r.get("Language") == "en" and r.get("Title", "").startswith("Encyclopaedia Britannica, 11th Edition")
            and "Prependix" not in r["Title"]]
    random.Random(a.seed).shuffle(rows)
    per_slice = max(1, min(8, a.per_book if a.per_book != 3 else 6))
    for r in rows:
        if ctx.pool.full:
            break
        pid = r["Text#"]
        if any(i.startswith(f"eb1911:pg{pid}-") for i in ctx.pool.ids):
            continue
        try:
            txt = ctx.http.text(f"https://www.gutenberg.org/cache/epub/{pid}/pg{pid}.txt", allow_404=True)
        except Exception as e:
            ctx.log("skip", pid, str(e)[:80])
            continue
        if not txt:
            continue
        arts = [(h, clean_paragraphs(ps)) for h, ps in eb_articles(txt)]
        arts = [(h, ps) for h, ps in arts if len(ps) >= 4 and sum(len(p) for p in ps) > 3500]
        random.Random(f"{a.seed}-{pid}").shuffle(arts)
        for h, ps in arts[:per_slice]:
            ctx.pool.add(dict(id=f"eb1911:pg{pid}-{re.sub(r'[^a-z0-9]+', '-', h.lower()).strip('-')[:40]}", domain="encyclopedia", licence="Public domain (published 1910-1911)",
                              licence_class="pd", licence_evidence=PG_EVIDENCE, url=f"https://www.gutenberg.org/ebooks/{pid}", title=f"{h} (Encyclopaedia Britannica, 11th ed.)",
                              attribution=f"Encyclopaedia Britannica, 11th ed. (1910-11), article {h}; via Project Gutenberg ebook #{pid}", date="1911", paragraphs=ps,
                              licence_note="published 1910-11 (US public domain); individual contributors died long ago but a few after 1954: PD outside the US is likely, not verified per article"))


# ======================================================================================================================== Voice of America
VOA_EVIDENCE = "https://www.voanews.com/p/5338.html"
_VOA_WIRE = re.compile(r"\b(?:Associated Press|Reuters|Agence France|AFP|AP|dpa|Bloomberg|Getty|Xinhua|TASS|Anadolu|The New York Times|The Washington Post|CNN)\b")


def voa_paragraphs(html_text: str) -> Tuple[str, str, List[str]]:
    root = parse_html(html_text)
    if root is None:
        return "", "", []
    meta = lambda n: next(iter(root.xpath(f'//meta[@name="{n}" or @property="{n}"]/@content')), "") or ""
    title = htmlmod.unescape(meta("og:title"))
    date = (meta("article:published_time") or "")[:10]
    scope = root.xpath("//div[contains(@class,'wsw')]") or root.xpath("//article")
    paras = html_paragraphs(scope, drop_classes=("wsw__embed", "media-block", "caption", "share", "related", "c-mmp", "embed", "wsw__quote", "tweet"))
    out = [p for p in paras if not re.match(r"^\s*(?:editor.s note|this story was|with additional|this article|related:|some information|VOA)", clean_paragraph(p), re.I)]
    return title, date, out


@source("voa", "news", "Public domain (VOA-produced material; wire-service content excluded)", "pd", VOA_EVIDENCE,
        "voanews.com sitemap_415_*.xml.gz -> /a/ article HTML (1 req/s)")
def src_voa(ctx: Ctx):
    """Voice of America English news produced by VOA itself (copyright statement: public domain); articles mentioning AP / Reuters / AFP etc. are dropped."""
    import gzip
    idx = ctx.http.text("https://www.voanews.com/sitemap.xml") or ""
    children = [u for u in re.findall(r"<loc>([^<]+)</loc>", idx) if u.endswith(".xml.gz")]
    urls: List[str] = []
    for c in sorted(children):
        try:
            r = ctx.http.get(c)
            x = gzip.decompress(r.content).decode("utf-8", "ignore")
        except Exception as e:
            ctx.log("sitemap", c, str(e)[:80])
            continue
        urls += [u for u in re.findall(r"<loc>([^<]+)</loc>", x) if re.search(r"/a/[^/]+/\d+\.html$", u)]
    urls = sorted(set(urls))
    ctx.log(len(urls), "article urls")
    random.Random(ctx.args.seed).shuffle(urls)
    for u in urls:
        did = "voa:" + re.search(r"/(\d+)\.html$", u).group(1)
        if ctx.pool.seen(did):
            continue
        try:
            h = ctx.http.text(u, allow_404=True)
        except Exception as e:
            continue
        if not h:
            continue
        title, date, paras = voa_paragraphs(h)
        if any(_VOA_WIRE.search(p) for p in paras) or _VOA_WIRE.search(title):
            ctx.pool.stats["wire_service"] += 1
            continue
        yield dict(id=did, domain="news", licence="Public domain (VOA-produced material)", licence_class="pd", licence_evidence=VOA_EVIDENCE, url=u, title=title,
                   attribution="Voice of America (voanews.com), " + u, date=date, paragraphs=paras,
                   licence_note="VOA statement: material produced exclusively by VOA is public domain; licensed wire (AFP/AP/Reuters) and other third-party content is not (articles mentioning wire services were dropped)")


# =================================================================================================== v2 pool: established, citable sources
# (2026-10-20, second source pool: Common Pile v0.1 (Kandpal et al. 2025, arXiv:2506.05209), GovReport, BillSum, ...)
DOMAIN_KEYWORDS: List[Tuple[str, "re.Pattern"]] = [
    ("health", re.compile(r"\b(?:health|medic\w*|nurs\w*|patient|disease|clinical|hospital|drug|vaccin\w*|cancer|diabetes|mental|surgery|anatomy|physiology|pharmac\w*|epidemi\w*|public health|therap\w*|"
                          r"nutrition|medicare|medicaid|HIV|malaria|obesity|pandemic|dental|disabilit\w*)\b", re.I)),
    ("education", re.compile(r"\b(?:education\w*|school\w*|student\w*|teach\w*|learning|curricul\w*|college\w*|universit\w*|pedagog\w*|classroom|literacy|academic|skills|training|"
                             r"higher education|childhood|tutor\w*)\b", re.I)),
    ("travel", re.compile(r"\b(?:travel\w*|tourism|tourist\w*|hospitality|destination\w*|hotel\w*|airport\w*|national park\w*|itinerary|vacation|hospitality)\b", re.I)),
    ("stories", re.compile(r"\b(?:fiction|short stories|stories|tales?|novels?|folklore|fables?|folktales?)\b", re.I)),
    ("science_explainer", re.compile(r"\b(?:biolog\w*|chemi\w*|physic\w*|geolog\w*|ecolog\w*|astronom\w*|climate|environment\w*|engineering|energy|species|ocean\w*|space|NASA|earth|"
                                     r"technolog\w*|research|scien\w*|agricultur\w*|water|forest\w*|wildlife|weather|satellite\w*|mathematic\w*|statistic\w*|comput\w*|software|programming|internet|cyber\w*|"
                                     r"robot\w*|artificial intelligence|data science|electronics|network\w*)\b", re.I)),
    ("encyclopedia", re.compile(r"\b(?:histor\w*|geograph\w*|anthropolog\w*|philosoph\w*|religio\w*|civili[sz]ation\w*|sociolog\w*|psycholog\w*|mythology|archaeolog\w*|"
                                r"cultur\w*|music|art history|linguistic\w*|world)\b", re.I)),
    ("law_government", re.compile(r"\b(?:law\w*|legal|legislat\w*|government\w*|polic\w*|politic\w*|constitution\w*|court\w*|justice|criminal|regulat\w*|tax\w*|budget|defen[cs]e|"
                                  r"military|congress\w*|federal|public administration|rights|election\w*|security|veteran\w*|immigration|trade|appropriations|homeland)\b", re.I)),
]


def keyword_domain(text: str, default: Optional[str] = None, min_hits: int = 2, skip: Sequence[str] = ()) -> Optional[str]:
    """Domain of the label with most keyword hits in `text` (ties: first in DOMAIN_KEYWORDS order); `default` when fewer than `min_hits` hits."""
    best, best_n = default, 0
    for dom, rx in DOMAIN_KEYWORDS:
        if dom in skip:
            continue
        n = len(rx.findall(text))
        if n > best_n:
            best, best_n = dom, n
    return best if best_n >= min_hits else default


# ---------------------------------------------------------------------------------------------------------------- plain / markdown text
_MD_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_MD_ATTR = re.compile(r"\{[^{}]*\}")
_GOV_NOISE = re.compile(r"\[\[Page [^\]]*\]\]|<GRAPHIC>|\[GRAPHIC\][^\n]*|<DOC>|</?pre>|\[TIFF OMITTED\][^\n]*|\[Congressional [^\]]*\]|\[From the U\.S\. Government Publishing Office\]|"
                        r"\[Federal Register[^\]]*\]|\[Pages? [^\]]*\]|\[www\.gpo\.gov\]", re.I)


def plain_blocks(text: str, join_lines: bool = True) -> List[str]:
    """Blank-line separated blocks of a hard-wrapped plain text (lines of a block are joined by a space)."""
    text = _GOV_NOISE.sub(" ", text.replace("\r\n", "\n").replace("\r", "\n")).replace("``", "\u201c").replace("''", "\u201d")
    out: List[str] = []
    for blk in re.split(r"\n\s*\n", text):
        lines = [l.strip() for l in blk.split("\n") if l.strip() and not re.fullmatch(r"[-_=*\s]{4,}", l.strip())]
        if not lines:
            continue
        out.append(" ".join(lines) if join_lines else "\n".join(lines))
    return out


def markdown_paragraphs(text: str) -> List[str]:
    """Paragraph strings of a lightweight Markdown document (Common Pile html/pdf conversions): headings, tables, code, rules and image lines are dropped,
    emphasis markers and link targets removed. A blank-line separated block is one paragraph; list items are separate paragraphs."""
    out: List[str] = []
    for blk in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
        lines = [l.strip() for l in blk.split("\n") if l.strip()]
        cur: List[str] = []
        items: List[str] = []
        for l in lines:
            if re.match(r"^(?:#{1,6}\s|\||```|~~~|---+$|===+$|\*\s*\*\s*\*$|<|!\[)", l) or l.startswith(">") and len(l) < 60:
                continue
            m = re.match(r"^(?:[-*+•]|\d{1,3}[.)])\s+(.*)$", l)
            if m:
                if len(m.group(1).split()) >= 10 and m.group(1).rstrip()[-1:] in ".!?\"”’":      # keep only full-sentence list items
                    items.append(m.group(1))
                continue
            cur.append(l)
        for p in ([" ".join(cur)] if cur else []) + items:
            p = _MD_LINK.sub(r"\1", p)
            p = _MD_ATTR.sub("", p)
            p = re.sub(r"(\*\*|__|\*|`)", "", p)
            p = re.sub(r"^\s*>\s*", "", p)
            if re.search(r"Pressbooks|Learn more about how|Creative Commons|Attribution|All rights reserved|Download (?:this|the) ", p) or len(p) > 4500:
                continue
            if p.rstrip().endswith("?") and len(p.split()) <= 30:            # review / discussion questions are not running prose
                continue
            out.append(p)
    return out


# ------------------------------------------------------------------------------------------------------------------------ Common Pile
CP_PAPER = "https://arxiv.org/abs/2506.05209"
_CC_URL = re.compile(r"creativecommons\.org/(licenses|publicdomain)/([a-z\-]+)/([\d.]+)", re.I)


def cp_licence(s: str) -> Optional[Tuple[str, str, str]]:
    """(licence, class, evidence) of a Common Pile `metadata.license` string, or None when it is not public domain / CC0 / CC BY."""
    s = (s or "").strip()
    if not s:
        return None
    m = _CC_URL.search(s)
    if m:
        kind, ver = m.group(2).lower(), m.group(3)
        url = m.group(0)
        if kind == "by":
            return f"CC-BY-{ver}", "permissive", "https://" + url
        if kind in ("zero", "mark"):
            return ("CC0-1.0" if kind == "zero" else "Public domain mark 1.0"), "pd", "https://" + url
        return None
    if re.search(r"ShareAlike|Non-?Commercial|No-?Deriv|NonCommercial|GNU|GFDL|\bSA\b|\bNC\b|\bND\b", s, re.I):
        return None
    if re.fullmatch(r"(?:public domain|cc0.*|pd|us government work)", s, re.I):
        return "Public domain", "pd", s
    if re.search(r"Attribution", s, re.I):
        return "CC-BY (version not stated)", "permissive", s
    return None


def cp_stream(repo: str, path: str) -> Iterator[dict]:
    """Stream a gzip-JSONL shard of a Hugging Face dataset without downloading it to disk (huggingface_hub HfFileSystem)."""
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    with fs.open(f"datasets/{repo}/{path}", "rb") as raw:
        g = gzip.open(raw, "rt", encoding="utf-8")
        try:
            for line in g:
                try:
                    yield json.loads(line)
                except Exception:
                    continue
        except (EOFError, OSError):         # truncated stream: keep what was read
            return


def cp_shards(repo: str, subdir: str, suffix: str = ".gz") -> List[str]:
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    return sorted(x["name"].split(f"datasets/{repo}/", 1)[1] for x in fs.ls(f"datasets/{repo}/{subdir}".rstrip("/"), detail=True) if x["name"].endswith(suffix))


def pick_long_and_random(cands: List[Tuple[int, dict]], cap: int, seed: str, long_share: float = 0.4) -> List[dict]:
    """From (length, record) candidates choose `cap`: the longest `long_share` of the cap plus a seeded random sample of the rest (long documents feed the long bins)."""
    if len(cands) <= cap:
        return [c for _, c in cands]
    ordered = sorted(cands, key=lambda x: -x[0])
    n_long = int(cap * long_share)
    chosen = ordered[:n_long]
    rest = ordered[n_long:]
    random.Random(seed).shuffle(rest)
    chosen += rest[:cap - n_long]
    return [c for _, c in chosen]


CP_NEWS_OUTLETS = {      # shard stem -> (display name, homepage)
    "360info": ("360info", "https://360info.org/"), "altnews": ("Alt News", "https://www.altnews.in/"), "balkandiskurs": ("Balkan Diskurs", "https://balkandiskurs.com/en/"),
    "educeleb": ("EduCeleb", "https://educeleb.com/"), "factly": ("Factly", "https://factly.in/"), "freedom": ("Freedom of the Press Foundation", "https://freedom.press/"),
    "libertytvradio": ("Liberty TV Radio", ""), "mekongeye": ("Mekong Eye", "https://www.mekongeye.com/"), "milwaukeenns": ("Milwaukee Neighborhood News Service", "https://milwaukeenns.org/"),
    "minorityafrica": ("Minority Africa", "https://minorityafrica.org/"), "newcanadianmedia": ("New Canadian Media", "https://www.newcanadianmedia.ca/"),
    "oxpeckers": ("Oxpeckers", "https://oxpeckers.org"), "propastop": ("Propastop", "https://www.propastop.org/en/"), "solutionsjournalism": ("Solutions Journalism Exchange", ""),
    "thepublicrecord": ("The Public Record", "https://thepublicrecord.ca/"), "zimfact": ("ZimFact", "https://zimfact.org/"), "scidev": ("SciDev.Net", "https://www.scidev.net/"),
    "africasacountry": ("Africa Is a Country", "https://africasacountry.com/"), "meduza": ("Meduza", "https://meduza.io/en"), "fides": ("Agenzia Fides", "https://www.fides.org/en"),
    "tasnim": ("Tasnim News Agency", ""), "globalvoices": ("Global Voices", "https://globalvoices.org/"),
}


CP_OPINION_OUTLETS = frozenset({"360info", "africasacountry", "solutionsjournalism", "freedom", "thepublicrecord", "newcanadianmedia"})      # analysis / essays / commentary rather than reporting


def cp_news_doc(rec: dict, outlet: str) -> Optional[dict]:
    """Pool-ready dict from a Common Pile news record (lines: title - outlet / author / date / body...), or None (licence not CC BY / PD)."""
    md = rec.get("metadata") or {}
    lic = cp_licence(md.get("license", ""))
    if not lic:
        return None
    lines = [l.strip() for l in (rec.get("text") or "").split("\n") if l.strip()]
    if len(lines) < 4:
        return None
    name, _home = CP_NEWS_OUTLETS.get(outlet, (outlet, ""))
    title = re.sub(r"\s+[-|–]\s+" + re.escape(name) + r"\s*$", "", lines[0]).strip()
    author = (md.get("author") or "").strip()
    return dict(id=f"cp_news:{outlet}-{rec.get('id')}", domain="opinion_magazine" if outlet in CP_OPINION_OUTLETS else "news", licence=lic[0], licence_class=lic[1], licence_evidence=lic[2], url=md.get("url", ""), title=title[:200],
                attribution=f"{author + ', ' if author else ''}{name}, {md.get('url', '')}, {lic[0]}", date=(rec.get("created") or "")[:40], paragraphs=lines[1:],
                licence_note=f"Common Pile v0.1 news (opennewswire CC BY outlets), licence string read per document: {md.get('license', '')[:120]}",
                extra={"outlet": name, "author": author, "common_pile_source": rec.get("source", "")})


@source("cp_news", "news", "CC-BY (per document, from the Common Pile 'news' collection) / public domain", "permissive", CP_PAPER + " ; https://huggingface.co/datasets/common-pile/news",
        "Hugging Face common-pile/news v0/documents/*.jsonl.gz (streamed; no robots needed)")
def src_cp_news(ctx: Ctx):
    """CC BY news and reporting from the outlets listed by opennewswire.org (360info, Minority Africa, ZimFact, Oxpeckers, ...), via the Common Pile (Kandpal et al. 2025)."""
    a = ctx.args
    repo = "common-pile/news"
    shards = cp_shards(repo, "v0/documents")
    by_outlet: Dict[str, List[str]] = {}
    for sh in shards:
        stem = os.path.basename(sh).split("_", 1)[1].split(".")[0]
        by_outlet.setdefault(stem, []).append(sh)
    skip = set((a.only or "globalvoices").split(","))             # Global Voices is already fetched from its own API
    outlets = sorted(o for o in by_outlet if o not in skip)
    cap = max(5, a.max_docs // max(1, len(outlets)) + 5)
    cap_big = int(cap * 1.6)
    for o in outlets:
        if ctx.pool.full:
            break
        cands: List[Tuple[int, dict]] = []
        for sh in by_outlet[o]:
            for rec in cp_stream(repo, sh):
                if len(rec.get("text") or "") < 1700:
                    continue
                cands.append((len(rec["text"]), rec))
                if len(cands) >= 20000:
                    break
        chosen = pick_long_and_random(cands, cap_big, f"{a.seed}-{o}")
        n0 = ctx.pool.n
        for rec in chosen:
            d = cp_news_doc(rec, o)
            if d:
                ctx.pool.add(d)
            if ctx.pool.n - n0 >= cap or ctx.pool.full:
                break
        ctx.log(o, "candidates", len(cands), "accepted", ctx.pool.n - n0)


def gv_authors(pool_file: str, out_file: str) -> int:
    """Fill the author names missing from `globalvoices__permissive.jsonl` (attribution + extra.author) from the Common Pile 'globalvoices' shards
    (metadata.author, matched on the post URL). Writes `out_file` atomically; documents without a match keep their record unchanged."""
    urls = {}
    with open(pool_file, encoding="utf-8") as f:
        docs = [json.loads(l) for l in f if l.strip()]
    want = {d["url"].rstrip("/") for d in docs}
    repo = "common-pile/news"
    for sh in [s for s in cp_shards(repo, "v0/documents") if "globalvoices" in s]:
        for rec in cp_stream(repo, sh):
            md = rec.get("metadata") or {}
            u = (md.get("url") or "").rstrip("/")
            if u in want and (md.get("author") or "").strip():
                urls[u] = md["author"].strip()
    n = 0
    tmp = out_file + ".part"
    with open(tmp, "w", encoding="utf-8") as g:
        for d in docs:
            au = urls.get(d["url"].rstrip("/"))
            if au and not (d.get("extra") or {}).get("author"):
                d["extra"] = dict(d.get("extra") or {}, author=au)
                d["attribution"] = f"{au}, Global Voices, {d['url']}, CC BY 3.0"
                n += 1
            g.write(json.dumps(d, ensure_ascii=False) + "\n")
    os.replace(tmp, out_file)
    print(f"[gv-authors] {n} of {len(docs)} documents got an author name ({len(urls)} URLs found in the Common Pile shards)")
    return 0


# ---- books / chapters (Pressbooks, LibreTexts): CC BY chapters, domain from the title
CP_BOOK_SOURCES = {          # name -> (repo, file pattern dir, shard suffix)
    "pressbooks": ("common-pile/pressbooks", "", ".json.gz"),
    "libretexts": ("common-pile/libretexts", "", ".json.gz"),
    "pressbooks-stories": ("common-pile/pressbooks", "", ".json.gz"),       # same shards; only fiction / folktale / narrative books, domain forced to 'stories'
}
CP_STORY_TITLE = re.compile(r"african folktales|short stor(?:y|ies)|arabian nights|open stories|narrative of the life|interesting narrative|folk\s?tales?|fairy tales?|fables|"
                            r"anthology of (?:world|earlier american|american|medieval|early british)|open anthology|stories from", re.I)


def book_domain(title: str, url: str = "") -> Optional[str]:
    if "libretexts.org" in url:
        host = urlparse(url).netloc.split(".")[0]
        fixed = {"med": "health", "bio": "science_explainer", "chem": "science_explainer", "phys": "science_explainer", "geo": "science_explainer", "eng": "science_explainer",
                 "biz": "education", "human": None, "socialsci": None, "workforce": "education", "k12": "education", "ped": "education"}
        if fixed.get(host):
            return fixed[host]
    return keyword_domain(title, default=None, min_hits=1)


@source("cp_books", "education,health,science_explainer,law_government,stories,travel", "CC-BY / public domain (per chapter, from the Common Pile Pressbooks / LibreTexts collections)", "permissive",
        CP_PAPER, "Hugging Face common-pile/pressbooks and common-pile/libretexts (streamed); use --only pressbooks|libretexts")
def src_cp_books(ctx: Ctx):
    """CC BY open-textbook chapters (Pressbooks, LibreTexts) via the Common Pile; CC BY-SA / NC / ND chapters are skipped."""
    a = ctx.args
    which = (a.only or "pressbooks").split(",")
    per_book = max(1, a.per_book)
    for w in which:
        repo, sub, suf = CP_BOOK_SOURCES[w]
        best: Dict[str, List[Tuple[int, dict]]] = {}
        for sh in cp_shards(repo, sub, suf):
            for rec in cp_stream(repo, sh):
                md = rec.get("metadata") or {}
                text = rec.get("text") or ""
                if len(text) < 2200 or not cp_licence(md.get("license", "")):
                    continue
                if w.endswith("-stories") != bool(CP_STORY_TITLE.search(md.get("title") or "")) and (w.endswith("-stories") or CP_STORY_TITLE.search(md.get("title") or "")):
                    continue
                key = md.get("book_url") or md.get("title") or rec.get("id")
                lst = best.setdefault(key, [])
                lst.append((len(text), rec))
                if len(lst) > per_book * 4:
                    lst.sort(key=lambda x: -x[0])
                    del lst[per_book * 3:]
            ctx.log(w, os.path.basename(sh), "books so far", len(best))
            if len(best) > 20000:
                break
        books = sorted(best)
        random.Random(a.seed).shuffle(books)
        for bk in books:
            if ctx.pool.full:
                break
            lst = sorted(best[bk], key=lambda x: -x[0])[:per_book * 3]
            random.Random(f"{a.seed}-{bk}").shuffle(lst)
            took = 0
            for _, rec in lst:
                md = rec["metadata"]
                lic = cp_licence(md.get("license", ""))
                btitle = (md.get("title") or "").strip()
                if re.search(r"pressbook|template|directory|compendium|sandbox|test book|guide to open", btitle, re.I):
                    continue
                dom = "stories" if w.endswith("-stories") else book_domain(btitle, rec.get("id", ""))
                if not dom or took >= per_book:
                    continue
                paras = markdown_paragraphs(rec["text"])
                ok = ctx.pool.add(dict(
                    id=f"cp_{w}:{hashlib.sha1(str(rec.get('id')).encode()).hexdigest()[:14]}", domain=dom, licence=lic[0], licence_class=lic[1], licence_evidence=lic[2],
                    url=md.get("url") or rec.get("id", ""), title=btitle[:200],
                    attribution=f"{htmlmod.unescape(md.get('author') or 'various authors')}, {btitle}, {md.get('url') or rec.get('id', '')}, {lic[0]}", date=(rec.get("created") or "")[:20], paragraphs=paras,
                    licence_note=f"Common Pile v0.1 {w}: chapter licence string read per document ({md.get('license', '')[:100]}); third-party items inside a chapter may carry other licences",
                    extra={"book_url": md.get("book_url", ""), "common_pile_source": w}))
                if ok:
                    took += 1


# ---- US Government Publishing Office (GovInfo) via the Common Pile
def usgpo_domain(col: str, title: str, text: str) -> str:
    return keyword_domain(title + " " + text[:3000], default="law_government", min_hits=3, skip=("travel", "stories")) if col in ("CRPT", "CHRG") else "law_government"


USGPO_NOTE = ("US government publication (GovInfo) read from the Common Pile 'usgpo' collection (licence field 'Public Domain'); committee reports and hearings can quote "
              "third-party copyrighted statements")


@source("cp_usgpo", "law_government,health,education,science_explainer", "Public domain (US Government Publishing Office / GovInfo; Common Pile licence field)", "pd",
        "https://www.govinfo.gov/about/policies ; " + CP_PAPER, "Hugging Face common-pile/usgpo data/*.jsonl.gz (streamed; ~105 s per shard); --only CRPT,CHRG,DCPD; --shards i:j | n1,n2,...")
def src_cp_usgpo(ctx: Ctx):
    """Congressional committee reports (CRPT), hearings (CHRG) and presidential statements (DCPD) from GovInfo via the Common Pile; a hearing / report is cut at 6,500 tokens."""
    a = ctx.args
    cols = set((a.only or "CRPT,CHRG,DCPD").split(","))
    repo = "common-pile/usgpo"
    shards = [s for s in cp_shards(repo, "data", ".jsonl.gz")]
    if a.shards and ":" not in a.shards:                      # explicit shard numbers, e.g. --shards 10,13 (the shards are grouped by collection)
        want = {int(x) for x in a.shards.split(",") if x.strip()}
        shards = [s for s in shards if int(os.path.basename(s).split("_")[0]) in want]
        lo, hi = 0, len(shards)
    else:
        random.Random(a.seed).shuffle(shards)
        lo, hi = (int(x) for x in (a.shards or f"0:{len(shards)}").split(":"))
    caps = {"CRPT": 0.45, "CHRG": 0.35, "DCPD": 0.2}
    got: Counter = Counter()
    for sh in shards[lo:hi]:
        if ctx.pool.full:
            break
        n_scan = 0
        for rec in cp_stream(repo, sh):
            col = rec.get("collection")
            if col not in cols or ctx.pool.full:
                continue
            if got[col] >= a.max_docs * caps.get(col, 0.3):
                continue
            n_scan += 1
            text = rec.get("text") or ""
            if len(text) < 6000:
                continue
            md = rec.get("metadata") or {}
            title = re.sub(r"\s+", " ", rec.get("title") or "").strip()
            blocks = plain_blocks(text)
            if col == "CHRG":                      # a hearing starts with committee lists: skip the first part, keep the testimony
                blocks = blocks[max(0, len(blocks) // 12):]
            blocks = [b for b in blocks if not re.match(r"^(?:[A-Z][A-Z .,'\-]{3,60}|\(?\d+\)?|Mr\. Chairman\.?)$", b)]
            if len(blocks) < 8:
                continue
            ok = ctx.pool.add(dict(
                id="cp_usgpo:" + rec["id"], domain=usgpo_domain(col, title, text), licence="Public domain (US government work)", licence_class="pd",
                licence_evidence="https://www.govinfo.gov/about/policies (Public Domain & Copyright Notice); Common Pile license field: " + str(md.get("license")), url=md.get("url", ""),
                title=title[:200], attribution=f"{rec.get('author') or 'US Government'}, {title}, GovInfo {rec.get('id')}", date=rec.get("date", ""), paragraphs=blocks,
                licence_note=USGPO_NOTE, extra={"collection": col}))
            if ok:
                got[col] += 1
        ctx.log(os.path.basename(sh), dict(got))


# ---- GovReport (Huang et al. 2021) and BillSum (Kornilova & Eidelman 2019): HF parquet
GOVREPORT_PAPER = "https://aclanthology.org/2021.naacl-main.112/"      # Huang et al., "Efficient Attentions for Long Document Summarization", NAACL 2021


def hf_parquet_rows(repo: str, files: Sequence[str], columns: Sequence[str]) -> Iterator[dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    for f in files:
        p = hf_hub_download(repo, f, repo_type="dataset")
        pf = pq.ParquetFile(p)
        for batch in pf.iter_batches(batch_size=256, columns=list(columns)):
            for r in batch.to_pylist():
                yield r


def govreport_url(rid: str) -> Tuple[str, str]:
    """(agency, URL) of a SCROLLS GovReport id: 'crs_R46266' -> CRS product page, 'gao_GAO-18-123' -> GAO product page."""
    kind, _, num = rid.partition("_")
    if kind == "crs":
        m = re.match(r"([A-Z]+)(\d+)", num)
        return "CRS", f"https://crsreports.congress.gov/product/pdf/{m.group(1)}/{num}" if m else "https://crsreports.congress.gov/"
    return "GAO", f"https://www.gao.gov/products/{num}"


def govreport_paragraphs(inp: str) -> List[str]:
    """Paragraphs of a SCROLLS GovReport input: lines starting with a tab are section headings (dropped), the other lines are paragraphs."""
    out = []
    for l in inp.split("\n"):
        if not l.strip() or l.startswith("\t"):
            continue
        l = re.sub(r"\(\s+", "(", l)
        l = re.sub(r"\s+\)", ")", l)
        l = re.sub(r"\b(\d+) (st|nd|rd|th)\b", r"\1\2", l)
        l = re.sub(r"\s+([,.;:])", r"\1", l)
        out.append(l)
    return out


def zip_jsonl(path: str, member: str) -> Iterator[dict]:
    import zipfile
    with zipfile.ZipFile(path) as z, z.open(member) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except Exception:
                    continue


@source("govreport", "law_government,health,education,science_explainer", "Public domain (US federal work; CRS / GAO reports) - GovReport dataset", "pd",
        "https://www.gao.gov/about/copyright ; https://crsreports.congress.gov/ ; " + GOVREPORT_PAPER, "Hugging Face tau/scrolls gov_report.zip (SCROLLS, Shaham et al. 2022): train/validation/test jsonl")
def src_govreport(ctx: Ctx):
    """GovReport (CRS and GAO reports, Huang et al. NAACL 2021) in its SCROLLS packaging (paragraph structure kept); the reports are cut at 6,500 tokens at a paragraph boundary."""
    from huggingface_hub import hf_hub_download
    a = ctx.args
    zp = hf_hub_download("tau/scrolls", "gov_report.zip", repo_type="dataset")
    rows = []
    for member in ("gov_report/test.jsonl", "gov_report/validation.jsonl", "gov_report/train.jsonl"):
        for r in zip_jsonl(zp, member):
            if len(r.get("input", "")) >= 5000 and sampled(r["id"], a.seed, a.sample):
                rows.append(r)
        ctx.log(member, "rows kept so far", len(rows))
    rows.sort(key=lambda r: hashlib.sha1(f"{a.seed}{r['id']}".encode()).hexdigest())
    for r in rows:
        if ctx.pool.full:
            break
        kind, url = govreport_url(r["id"])
        paras = govreport_paragraphs(r["input"])
        out = " ".join((r.get("output") or "").split())
        dom = keyword_domain(out[:1500] + " " + " ".join(paras[:4])[:2000], default="law_government", min_hits=3, skip=("travel", "stories"))
        ctx.pool.add(dict(
            id=f"govreport:{r['id']}", domain="government_public", licence="Public domain (US federal government report)", licence_class="pd",
            licence_evidence=("https://www.gao.gov/about/copyright" if kind == "GAO" else "https://crsreports.congress.gov/"), url=url,
            title=(" ".join(re.sub(r"^(?:What GAO Found|Why GAO Did This Study|What GAO Recommends|Summary)\s*", "", out).split()[:14]) + " ...")[:140],
            attribution=("US Government Accountability Office" if kind == "GAO" else "Congressional Research Service") + f" report {r['id'].split('_', 1)[1]}, via GovReport (Huang et al., NAACL 2021)", date="",
            paragraphs=paras, licence_note="GAO and CRS products are US government works; both agencies state that some reports contain copyrighted third-party material (images, quotations); "
                                           "the dataset gives no report title / date (the beginning of the reference summary is used as title)", extra={"agency": kind, "scrolls_id": r["id"], "topic_guess": dom}))
    try:
        os.remove(os.path.realpath(zp))
    except OSError:
        pass


def billsum_paragraphs(text: str) -> List[str]:
    """Paragraphs of a US bill text: lines are re-flowed; a new paragraph starts at '(a)', '(1)', 'SEC.' ... ; `` '' quotes become typographic quotes."""
    t = text.replace("``", "“").replace("''", "”")
    out: List[str] = []
    cur: List[str] = []
    for line in t.split("\n"):
        s = line.strip()
        if not s:
            if cur:
                out.append(" ".join(cur))
                cur = []
            continue
        if re.match(r"^(?:SEC(?:TION|\.)\s*\d|TITLE\s+[IVXL\d]|\(\s*[a-zA-Z0-9]{1,4}\s*\))", s) and cur:
            out.append(" ".join(cur))
            cur = []
        cur.append(s)
    if cur:
        out.append(" ".join(cur))
    res = []
    for p in out:
        p = re.sub(r"^(?:SEC(?:TION|\.)\s*\d+[A-Za-z]?\.?\s*[A-Z][A-Z ,'\-]*\.?\s*)", "", p)          # 'SEC. 2. DEFINITIONS.' header glued to the text
        p = re.sub(r"^(?:\(\s*[a-zA-Z0-9]{1,4}\s*\)\s*)+", "", p)
        p = re.sub(r"^[A-Z][A-Za-z ,'\-]{2,60}\.--\s*", "", p)                                       # run-in heading 'In General.--'
        p = re.sub(r"(?<=\w)\.--(?=[A-Z])", ". ", p)
        if re.search(r"table of contents|^\W*SEC\.", p, re.I) or p.count("\u201c(") >= 3 or p.count("--") >= 3 or re.match(r"^[\W\d]", p):
            continue
        res.append(p)
    return res


@source("billsum", "law_government,health,education,science_explainer", "CC0-1.0 dataset; US Congressional bills are public domain", "pd",
        "https://huggingface.co/datasets/FiscalNote/billsum (license: cc0-1.0) ; https://www.govinfo.gov/about/policies", "Hugging Face FiscalNote/billsum data/train-*.parquet")
def src_billsum(ctx: Ctx):
    """BillSum (Kornilova & Eidelman 2019): text of US Congressional bills (5,000-20,000 characters), 103rd-115th Congress."""
    a = ctx.args
    rows = list(enumerate(hf_parquet_rows("FiscalNote/billsum", ["data/train-00000-of-00001.parquet", "data/test-00000-of-00001.parquet"], ["text", "title", "summary"])))
    random.Random(a.seed).shuffle(rows)
    for i, r in rows:
        if ctx.pool.full:
            break
        paras = billsum_paragraphs(r["text"])
        dom = keyword_domain(r["title"] + " " + r["summary"][:800], default="law_government", min_hits=2, skip=("travel", "stories"))
        ctx.pool.add(dict(
            id=f"billsum:{i:05d}", domain=dom, licence="Public domain (US Congressional bill; dataset CC0-1.0)", licence_class="pd",
            licence_evidence="https://huggingface.co/datasets/FiscalNote/billsum (card: license cc0-1.0); https://www.govinfo.gov/about/policies", url="", title=r["title"][:200],
            attribution=f"United States Congress, {r['title']}; via BillSum (Kornilova & Eidelman 2019)", date="", paragraphs=paras,
            licence_note="bill text is a US government work; BillSum gives no bill number / URL in the HF release", extra={"dataset": "BillSum"}))


# ---- DOAB (Directory of Open Access Books) via the Common Pile: CC BY books converted from PDF with Marker
DOAB_SKIP = ("health", "law_government", "stories", "science_explainer")          # research-register science books (MDPI / IntechOpen special issues) are not general reading


@source("cp_doab", "encyclopedia,education,science_explainer,government_public,travel", "CC-BY (per book, DOAB metadata + licence statement verified by the Common Pile)", "permissive",
        CP_PAPER + " ; https://www.doabooks.org/", "Hugging Face common-pile/doab (streamed *.json.gz shards); one or two ~4,500-token windows per book")
def src_cp_doab(ctx: Ctx):
    """Windows of CC BY open-access academic books (DOAB, Marker PDF-to-text) chosen from the middle of the book; domain from the book title."""
    a = ctx.args
    repo = "common-pile/doab"
    per_book = max(1, a.per_book)
    seen_books: Counter = Counter()
    shards = cp_shards(repo, "", ".json.gz")
    random.Random(a.seed).shuffle(shards)
    for sh in shards:
        if ctx.pool.full:
            break
        for rec in cp_stream(repo, sh):
            if ctx.pool.full:
                break
            md = rec.get("metadata") or {}
            lic = cp_licence(md.get("license", ""))
            text = rec.get("text") or ""
            book = md.get("book_id") or rec.get("id")
            if not lic or len(text) < 30000 or seen_books[book] >= per_book or re.search(r"MDPI|IntechOpen", md.get("publisher", ""), re.I):
                continue
            title = (md.get("title") or "").strip()
            dom = keyword_domain(title, default=None, min_hits=1, skip=DOAB_SKIP)
            if not dom:
                continue
            paras = clean_paragraphs(markdown_paragraphs(text))
            if len(paras) < 40:
                continue
            rng = random.Random(f"{a.seed}-{rec.get('id')}")
            start = rng.randrange(len(paras) // 6, max(len(paras) // 6 + 1, (3 * len(paras)) // 5))
            win = window_paragraphs(paras, ctx.pool.tc, start, 4600)
            while win and win[0][:1].islower():            # a window that starts mid-sentence (PDF page break) loses its first fragment
                win = win[1:]
            ok = ctx.pool.add(dict(
                id=f"cp_doab:{hashlib.sha1(str(rec.get('id')).encode()).hexdigest()[:14]}", domain=dom, licence=lic[0], licence_class=lic[1], licence_evidence=lic[2],
                url=md.get("url", ""), title=title[:200], attribution=f"{md.get('author') or 'various authors'}, {title}, {md.get('publisher', '')}, {lic[0]}", date="", paragraphs=win,
                licence_note="DOAB book licence read by the Common Pile (CC BY only); PDF-to-text conversion (Marker): residual running heads / footnote fragments possible; figures and third-party "
                             "quotations inside a book may carry other licences", extra={"book_id": book, "publisher": md.get("publisher", ""), "isbn": md.get("isbn", "")}))
            if ok:
                seen_books[book] += 1
        ctx.log(os.path.basename(sh), "docs", ctx.pool.n)


def capacity_table(pool_dir: str, exclude: Sequence[str] = ("openstax_ccby_snapshot", "share_alike")) -> dict:
    """Domain x length-range table of the pd + permissive pool files (files whose name contains an `exclude` string are skipped): documents per range
    (400-799 / 800-1599 / 1600-3199 / >=3200 Gemma tokens) and the 'usable' capacity per bin (a longer document can be cut down)."""
    ranges = [(400, 800), (800, 1600), (1600, 3200), (3200, 10 ** 9)]
    table: Dict[str, Counter] = {}
    per_file: Dict[str, Counter] = {}
    for f in sorted(glob.glob(os.path.join(pool_dir, "*.jsonl"))):
        base = os.path.basename(f)
        if any(x in base for x in exclude):
            continue
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d["licence_class"] not in ("pd", "permissive"):
                    continue
                t = d["n_gemma_tokens"]
                k = next((i for i, (lo, hi) in enumerate(ranges) if lo <= t < hi), None)
                if k is None:
                    continue
                table.setdefault(d["domain"], Counter())[k] += 1
                per_file.setdefault(base, Counter())[(d["domain"], k)] += 1
    out = {}
    for dom, c in sorted(table.items()):
        n = [c.get(i, 0) for i in range(4)]
        out[dom] = dict(r400=n[0], r800=n[1], r1600=n[2], r3200=n[3], total=sum(n), usable512=sum(n), usable1024=sum(n[1:]), usable2048=sum(n[2:]), usable4096=n[3])
    tot = {k: sum(v[k] for v in out.values()) for k in next(iter(out.values()))} if out else {}
    out["all"] = tot
    return {"domains": out, "files": {f: {f"{dom}|{k}": n for (dom, k), n in c.items()} for f, c in per_file.items()}}


# ============================================================================================================ curated benchmark text (v3)
# WMT news test sets, NTREX-128, FLORES+ and similar: human-curated, document / passage structured English, kept VERBATIM (no prose heuristics),
# any length from `V3_MIN_TOKENS` Gemma tokens up. Licence classes are recorded honestly (`unclear` for WMT, `share_alike` for NTREX / FLORES+).
V3_MIN_TOKENS = 20


def verbatim_mode(ctx: "Ctx", min_tokens: int = V3_MIN_TOKENS, min_paragraphs: int = 1):
    ctx.pool.verbatim = True
    ctx.pool.min_tokens = min_tokens
    ctx.pool.min_paragraphs = min_paragraphs


def _tar_members(path: str, pattern: str):
    """Yield (member name, file object) of the members of a (gz) tar archive whose name matches the regex `pattern`; sequential read, nothing is extracted to disk."""
    import tarfile
    rx = re.compile(pattern)
    with tarfile.open(path, "r|gz") as tf:
        for m in tf:
            if m.isfile() and rx.search(m.name):
                f = tf.extractfile(m)
                if f is not None:
                    yield m.name, f


def wmt_parse_sgm(text: str):
    """WMT <= 2020 SGML: yields (attrs, [[(seg id, text) ...] per <p>]) per <doc>."""
    for m in re.finditer(r"<doc\b([^>]*)>(.*?)</doc>", text, re.S):
        attrs = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
        paras = []
        for pm in re.finditer(r"<p>(.*?)</p>", m.group(2), re.S):
            paras.append([(int(i), htmlmod.unescape(t)) for i, t in re.findall(r'<seg id="(\d+)">(.*?)</seg>', pm.group(1), re.S)])
        yield attrs, paras


def wmt_parse_xml(fileobj):
    """WMT >= 2021 XML (`<doc origlang= id= domain=> <src> <p> <seg id=>`): yields (attrs, [[(seg id, text) ...] per <p>]); only the source side is read."""
    from lxml import etree
    for _, el in etree.iterparse(fileobj, events=("end",), tag="doc", huge_tree=True):
        src = el.find("src")
        paras = []
        if src is not None:
            for p in src.findall("p"):
                paras.append([(int(s.get("id") or 0), "".join(s.itertext())) for s in p.findall("seg")])
        yield dict(el.attrib), paras
        el.clear()


def wmt_mqm_docs(path: str) -> Iterator[dict]:
    """newstest2020 en-de source documents from the Google MQM release (https://github.com/google/wmt-mqm-human-evaluation, Apache-2.0 for the annotations): its TSV carries the
    WMT20 source segment of every (document, segment) with the document id, which is the only compliant copy of the WMT20 English sources (data.statmt.org disallows robots)."""
    import csv
    csv.field_size_limit(10 ** 9)
    docs: "OrderedDict[str, Dict[int, str]]" = OrderedDict()
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            docs.setdefault(r["doc"], {})[int(r["seg_id"])] = r["source"]
    for doc, segs in docs.items():
        yield dict(docid=doc, domain_label="news", pair_file=os.path.basename(path), paras=[[(i, segs[i]) for i in sorted(segs)]])


def wmt25_docs(path: str, domains: Sequence[str] = ("news",)) -> Iterator[dict]:
    """WMT25 General MT English-source documents (`wmt25-genmt.jsonl`: one record per language pair; the English source of a document is identical in all pairs, so it is deduplicated by the
    id after the language pair). Paragraphs = the lines of `src_text`."""
    seen = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("src_lang") != "en" or r.get("domain") not in domains:
                continue
            key = r["doc_id"].split("_#_", 1)[-1]
            if key in seen:
                continue
            seen.add(key)
            paras = [[(i + 1, t)] for i, t in enumerate(x for x in r["src_text"].split("\n") if x.strip())]
            yield dict(docid=key.replace("_#_", ":"), domain_label=r["domain"], pair_file=os.path.basename(path), paras=paras)


def looks_like_headline(seg: str) -> bool:
    seg = seg.strip()
    return 0 < len(seg) <= 160 and not seg.endswith(".")


def wmt_layout(paras: Sequence[Sequence[Tuple[int, str]]]) -> Tuple[List[str], List[List[int]], str]:
    """Paragraph texts of one WMT document + for each paragraph [first seg id, last seg id] + the layout name.
    Paragraph = the <p> element of the data (segments joined by a space). A document that is ONE <p> (WMT19 / WMT20 / most of WMT22 / WMT24) is one
    paragraph, except that a first segment that looks like a headline (<= 160 chars, no final full stop) becomes a paragraph of its own."""
    paras = [[(i, t.strip()) for i, t in p if t and t.strip()] for p in paras]
    paras = [p for p in paras if p]
    layout = "source_paragraphs"
    if len(paras) == 1 and len(paras[0]) >= 2:
        layout = "single_block"
        if looks_like_headline(paras[0][0][1]):
            paras = [paras[0][:1], paras[0][1:]]
            layout = "headline+single_block"
    elif len(paras) >= 2 and len(paras[0]) == 1 and looks_like_headline(paras[0][0][1]):
        layout = "source_paragraphs(headline_first)"
    texts = [" ".join(t for _, t in p) for p in paras]
    ranges = [[p[0][0], p[-1][0]] for p in paras]
    return texts, ranges, layout


WMT_TERMS_URL = "https://www.statmt.org/wmt{yy}/translation-task.html"
WMT_TERMS = ("The data released for the WMT{yy} {task} task can be freely used for research purposes, we {just}ask that you cite the WMT{yy} shared task overview "
             "paper, and respect any additional citation requirements on the individual data sets. For other uses of the data, you should consult with original owners of the data sets.")
WMT_SETS = OrderedDict([
    ("newstest2019", dict(year=2019, task="news translation", just="just ", kind="ntrex", url="", pattern="",
                          paper="Barrault et al. 2019, Findings of the 2019 Conference on Machine Translation (WMT19), https://aclanthology.org/W19-5301")),
    ("newstest2020", dict(year=2020, task="news translation", just="just ", kind="mqm", url="https://raw.githubusercontent.com/google/wmt-mqm-human-evaluation/main/newstest2020/ende/mqm_newstest2020_ende.no-postedits.tsv", pattern="",
                          paper="Barrault et al. 2020, Findings of the 2020 Conference on Machine Translation (WMT20), https://aclanthology.org/2020.wmt-1.1")),
    ("newstest2021", dict(year=2021, task="news translation", just="just ", kind="xml", url="https://github.com/wmt-conference/wmt21-news-systems/archive/refs/tags/v1.3.tar.gz", pattern=r"/xml/newstest2021\.en-[a-z]{2}\.all\.xml$",
                          paper="Akhbardeh et al. 2021, Findings of the 2021 Conference on Machine Translation (WMT21), https://aclanthology.org/2021.wmt-1.1")),
    ("wmttest2022", dict(year=2022, task="General MT", just="just ", kind="xml", url="https://github.com/wmt-conference/wmt22-news-systems/archive/refs/tags/v1.1.tar.gz", pattern=r"/xml/wmttest2022\.en-[a-z]{2,3}\.all\.xml$",
                         paper="Kocmi et al. 2022, Findings of the 2022 Conference on Machine Translation (WMT22), https://aclanthology.org/2022.wmt-1.1")),
    ("wmttest2023", dict(year=2023, task="General MT", just="", kind="xml", url="https://github.com/wmt-conference/wmt23-news-systems/archive/refs/tags/v.0.1.tar.gz", pattern=r"/xml/wmttest2023\.en-[a-z]{2,3}\.all\.xml$",
                         paper="Kocmi et al. 2023, Findings of the 2023 Conference on Machine Translation (WMT23): LLMs Are Here but Not Quite There Yet, https://aclanthology.org/2023.wmt-1.1")),
    ("wmttest2024", dict(year=2024, task="General MT", just="", kind="xml", seg_is_paragraph=True, url="https://github.com/wmt-conference/wmt24-news-systems/releases/download/v1.1/data_onlyxml.tar.gz", pattern=r"^xml/wmttest2024\.en-[a-z]{2,3}\.all\.xml$",
                         paper="Kocmi et al. 2024, Findings of the WMT24 General Machine Translation Shared Task: The LLM Era Is Here but MT Is Not Solved Yet, https://aclanthology.org/2024.wmt-1.1")),
    ("wmttest2025", dict(year=2025, task="General MT", just="", kind="jsonl25", url="https://raw.githubusercontent.com/wmt-conference/wmt25-general-mt/main/data/wmt25-genmt.jsonl", pattern="",
                         paper="Findings of the WMT25 General Machine Translation Shared Task: Time to Stop Evaluating on Easy Test Sets, https://aclanthology.org/2025.wmt-1.22")),
])


def wmt_terms(testset: str) -> Tuple[str, str]:
    """(licence evidence = the statmt.org page, the quoted terms) of a WMT test set (read 2026-10-20; the WMT23 / WMT24 pages say 'we ask' instead of 'we just ask')."""
    m = WMT_SETS[testset]
    yy = str(m["year"])[2:]
    q = WMT_TERMS.format(yy=yy, task=m["task"], just=m["just"])
    if m["year"] == 2024:
        q = q.replace("shared task overview paper", "shared task overview findings paper")
    if m["year"] == 2025:
        q = ("The data released for the General MT task can be freely used for research purposes, we ask that you cite the shared task overview findings paper, and respect any "
             "additional citation requirements on the individual data sets. For other uses of the data, you should consult with original owners of the data sets.")
        return "https://www2.statmt.org/wmt25/translation-task.html", q
    return WMT_TERMS_URL.format(yy=yy), q


def wmt_docs(path: str, testset: str, domains: Sequence[str] = ("news",), pattern: Optional[str] = None) -> Iterator[dict]:
    """Original-English documents of one WMT test set (the English SOURCE side of every en-xx pair, a document that is in several pairs is yielded once).
    Test-suite items (`testsuite=` attribute, single challenge sentences) and non-selected domains (social, speech, conversation, e-commerce, ...) are skipped."""
    meta = WMT_SETS[testset]
    seen = set()
    if meta["kind"] == "mqm":
        yield from wmt_mqm_docs(path)
        return
    if meta["kind"] == "jsonl25":
        yield from wmt25_docs(path, domains)
        return
    for name, f in _tar_members(path, pattern or meta["pattern"]):
        it = wmt_parse_sgm(f.read().decode("utf-8")) if meta["kind"] == "sgm" else wmt_parse_xml(f)
        for attrs, paras in it:
            if attrs.get("origlang", "en") != "en" or "testsuite" in attrs:
                continue
            dom = attrs.get("domain") or attrs.get("genre") or "news"
            if dom not in domains:
                continue
            docid = attrs.get("docid") or attrs.get("id") or ""
            if not docid or docid in seen:
                continue
            seen.add(docid)
            yield dict(docid=docid, domain_label=dom, pair_file=os.path.basename(name), paras=paras)


def wmt_outlet(docid: str) -> str:
    d = re.sub(r"^(?:test-en-news_|test-en-)", "", docid)
    d = re.sub(r"[._-]?\d+$", "", d)
    return d


def wmt_row(w: dict, testset: str, domain: str = "news") -> Optional[dict]:
    meta = WMT_SETS[testset]
    paras = w["paras"]
    if meta.get("seg_is_paragraph"):            # WMT24: the unit of the data is the paragraph (one <seg> = one paragraph; WMT24++ uses the same segments)
        paras = [[sg] for p in paras for sg in p]
    texts, ranges, layout = wmt_layout(paras)
    if not texts:
        return None
    if meta.get("seg_is_paragraph"):
        layout = "paragraph_segments"
    ev, quote = wmt_terms(testset)
    docid = w["docid"]
    has_head = layout.startswith(("headline", "source_paragraphs(headline"))
    return dict(id=f"wmt:{testset}:{docid}", source="wmt_news", domain=domain, licence=f"WMT {meta['year']} data terms (research use; publishers' copyright)", licence_class="unclear",
                licence_evidence=ev, url="", title=texts[0] if has_head else "",
                attribution=f"WMT {meta['year']} {'news test set' if meta['year'] < 2022 else 'General MT test set'} ({testset}), document {docid} (news outlet: {wmt_outlet(docid)}); cite {meta['paper']}",
                date="", paragraphs=texts,
                licence_note=f"{quote} ({ev}, read 2026-10-20). Not an open licence: the articles are third-party news text released by the WMT organisers for research; cite the overview paper.",
                extra={"testset": testset, "docid": docid, "outlet": wmt_outlet(docid), "origlang": "en", "wmt_domain": w["domain_label"], "layout": layout,
                       "paragraph_segments": ranges, "n_segments": sum(r[1] - r[0] + 1 for r in ranges), "pair_file": w["pair_file"], "citation": meta["paper"],
                       "obtained_from": {"mqm": "google/wmt-mqm-human-evaluation newstest2020 en-de source column (data.statmt.org disallows robots)", "jsonl25": "wmt-conference/wmt25-general-mt data/wmt25-genmt.jsonl"}.get(meta["kind"], meta["url"])})


def _wmt_archive(ctx: "Ctx", testset: str) -> str:
    raw = ctx.raw_dir or os.path.join(os.path.dirname(os.path.abspath(ctx.args.out)), "..", "raw", "wmt")
    os.makedirs(raw, exist_ok=True)
    meta = WMT_SETS[testset]
    ext = ".tsv" if meta["url"].endswith(".tsv") else ".jsonl" if meta["url"].endswith(".jsonl") else ".tgz" if meta["url"].endswith(".tgz") else ".tar.gz"
    path = os.path.join(raw, testset + ext)
    _download(ctx, meta["url"], path)
    return path


def _rm(path: str):
    try:
        os.remove(path)
    except OSError:
        pass


@source("wmt_news", "news", "WMT data terms (research use; third-party publisher text)", "unclear", "https://www.statmt.org/wmt22/translation-task.html",
        "statmt.org / GitHub wmt-conference test archives (newstest2019-2021, wmttest2022-2024), English SOURCE side of all en-xx pairs")
def src_wmt_news(ctx: Ctx):
    """WMT news-translation test sets 2019-2024, English source documents (original English only), one row per document, paragraphs as marked in the data."""
    verbatim_mode(ctx)
    sets = [x for x in (ctx.args.only or ",".join(k for k, m in WMT_SETS.items() if m["kind"] != "ntrex")).split(",") if x]       # newstest2019 = NTREX-128 (own pool file)
    for ts in sets:
        path = _wmt_archive(ctx, ts)
        n = 0
        for w in wmt_docs(path, ts):
            d = wmt_row(w, ts)
            if d is not None:
                n += 1
                yield d
        ctx.log(ts, "documents", n)
        if not ctx.args.keep_raw:
            _rm(path)


@source("wmt_literary", "stories", "WMT data terms (research use; third-party literary text)", "unclear", "https://www.statmt.org/wmt24/translation-task.html",
        "github.com/wmt-conference wmt24-news-systems xml + wmt25-general-mt jsonl (English literary source documents of WMT24 and WMT25)")
def src_wmt_literary(ctx: Ctx):
    """WMT24 / WMT25 General MT, domain `literary`: English fiction excerpts (WMT24: chunks of about 1,000 words; WMT25: two book chapters), the only literary text of the WMT campaign."""
    verbatim_mode(ctx)
    for ts in [x for x in (ctx.args.only or "wmttest2024,wmttest2025").split(",") if x]:
        path = _wmt_archive(ctx, ts)
        for w in wmt_docs(path, ts, domains=("literary",)):
            d = wmt_row(w, ts, domain="stories")
            if d is None:
                continue
            d["id"] = d["id"].replace("wmt:", "wmtlit:")
            d["source"] = "wmt_literary"
            d["attribution"] = d["attribution"].replace("(news outlet:", "(literary excerpt:")
            d["licence_note"] += " Literary excerpts: the copyright status of each excerpt is NOT stated in the data; see the findings paper."
            yield d
        if not ctx.args.keep_raw:
            _rm(path)


# ------------------------------------------------------------------------------------------------------------------------ NTREX-128
NTREX_RAW = "https://raw.githubusercontent.com/MicrosoftTranslator/NTREX/main/"
NTREX_EVIDENCE = "https://github.com/MicrosoftTranslator/NTREX/blob/main/LICENSE.md"


@source("ntrex128", "news", "CC-BY-SA-4.0 (NTREX repository README / LICENSE.md); underlying text = WMT19 news test set (WMT terms)", "share_alike", NTREX_EVIDENCE,
        "github.com/MicrosoftTranslator/NTREX (newstest2019-src.eng.txt + DOCUMENT_IDS.tsv) document boundaries from DOCUMENT_IDS.tsv")
def src_ntrex128(ctx: Ctx):
    """NTREX-128 English documents (= WMT19 en-de newstest2019 source, 123 documents, 1,997 sentences) with the line range of each document, so the 128 human reference translations can be attached."""
    verbatim_mode(ctx)
    raw = ctx.raw_dir or os.path.join(os.path.dirname(os.path.abspath(ctx.args.out)), "..", "raw", "ntrex")
    os.makedirs(raw, exist_ok=True)
    lines_path, ids_path = os.path.join(raw, "newstest2019-src.eng.txt"), os.path.join(raw, "DOCUMENT_IDS.tsv")
    _download(ctx, NTREX_RAW + "NTREX-128/newstest2019-src.eng.txt", lines_path)
    _download(ctx, NTREX_RAW + "DOCUMENT_IDS.tsv", ids_path)
    with open(lines_path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    with open(ids_path, encoding="utf-8") as f:
        docids = [x.strip() for x in f.read().split("\n") if x.strip()]
    assert len(docids) == len(lines), (len(docids), len(lines))
    spans: "OrderedDict[str, List[int]]" = OrderedDict()
    for i, d in enumerate(docids):
        spans.setdefault(d, [i, i + 1])[1] = i + 1
    ev, quote = wmt_terms("newstest2019")
    for docid, (a, b) in spans.items():
        # the WMT19 en-de source file is ONE <p> per document, so the layout is the same as for the other single-block WMT documents (headline + body)
        w = dict(docid=docid, domain_label="news", pair_file="newstest2019-ende-src.en.sgm (= NTREX-128/newstest2019-src.eng.txt)", paras=[[(i - a + 1, lines[i]) for i in range(a, b)]])
        row = wmt_row(w, "newstest2019")
        row.update(id=f"ntrex128:{docid}", source="ntrex128", licence="CC-BY-SA-4.0 (NTREX-128; underlying text WMT19 newstest2019 English source)", licence_class="share_alike",
                   licence_evidence=NTREX_EVIDENCE)
        row["attribution"] = (f"NTREX-128 (Federmann, Kocmi, Xin 2022, https://aclanthology.org/2022.sumeval-1.4) = WMT19 newstest2019 English source, document {docid} "
                              f"(news outlet: {wmt_outlet(docid)}); cite also Barrault et al. 2019, https://aclanthology.org/W19-5301")
        row["licence_note"] = ("NTREX README (read 2026-10-20): 'The NTREX-128 data set is released under the CC BY-SA 4.0 license' (https://github.com/MicrosoftTranslator/NTREX). "
                               "The English text is the WMT19 news test set: " + quote + " (" + ev + ")")
        row["extra"].update(ntrex_lines=[a, b], ntrex_file="NTREX-128/newstest2019-src.eng.txt", also_in="wmt newstest2019 en-de source (same text, deduplicated)",
                            ntrex_citation="Federmann, Kocmi, Xin 2022, NTREX-128, https://aclanthology.org/2022.sumeval-1.4")
        yield row


# ------------------------------------------------------------------------------------------------------------------------ FLORES+ English
FLORES_REPO = "openlanguagedata/flores_plus"
FLORES_EVIDENCE = "https://huggingface.co/datasets/openlanguagedata/flores_plus"
FLORES_DOMAIN = {"wikinews": "news", "wikivoyage": "travel", "wikibooks": "education"}
FLORES_CITES = ("Goyal et al. 2022, The Flores-101 Evaluation Benchmark for Low-Resource and Multilingual Machine Translation, TACL, https://aclanthology.org/2022.tacl-1.30 ; "
                "NLLB Team 2022, No Language Left Behind, https://arxiv.org/abs/2207.04672 ; FLORES+ (OLDI), https://huggingface.co/datasets/openlanguagedata/flores_plus")


def flores_passages(rows: Sequence[dict], split: str, max_sentences: int = 6) -> Iterator[dict]:
    """Group the FLORES+ English rows (sorted by id) into article passages: consecutive ids with the same `url`; a longer run is cut into chunks of at most `max_sentences`
    sentences (a trailing chunk of a single sentence is merged into the previous one). Yields pool documents (one paragraph = the sentences joined by a space)."""
    rows = sorted(rows, key=lambda r: int(r["id"]))
    groups: List[List[dict]] = []
    for r in rows:
        if groups and groups[-1][-1]["url"] == r["url"] and int(r["id"]) == int(groups[-1][-1]["id"]) + 1:
            groups[-1].append(r)
        else:
            groups.append([r])
    for g in groups:
        n = len(g)
        k = max(1, -(-n // max_sentences))                  # number of chunks
        size, extra = divmod(n, k)                          # near-equal chunks
        i = 0
        for c in range(k):
            m = size + (1 if c < extra else 0)
            chunk, i = g[i:i + m], i + m
            ids = [int(r["id"]) for r in chunk]
            r0 = chunk[0]
            dom = r0.get("domain", "")
            title = urlparse(r0["url"]).path.rsplit("/", 1)[-1].replace("_", " ")
            title = unquote(title)
            yield dict(id=f"flores_plus:{split}:{ids[0]}-{ids[-1]}", split=split, ids=ids, domain_flores=dom, url=r0["url"], title=title, topic=r0.get("topic", ""),
                       sentences=[r["text"] for r in chunk])


@source("flores_plus", "news,travel,education", "CC-BY-SA-4.0 (FLORES+ dataset card; English text from Wikinews / Wikivoyage / Wikibooks (Wikijunior))", "share_alike", FLORES_EVIDENCE,
        "HF openlanguagedata/flores_plus eng_Latn dev + devtest (or local copies via --raw-dir)")
def src_flores_plus(ctx: Ctx):
    """FLORES+ English dev + devtest, one row per 2-6 sentence article passage; human translations exist for 44 of our languages (`flores_plus_ids.json` maps row -> FLORES ids)."""
    verbatim_mode(ctx)
    for split in ("dev", "devtest"):
        path = None
        for cand in ([os.path.join(ctx.raw_dir, split, "eng_Latn.jsonl")] if ctx.raw_dir else []):
            if os.path.exists(cand):
                path = cand
        if path is None:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(FLORES_REPO, f"{split}/eng_Latn.jsonl", repo_type="dataset")
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        for p in flores_passages(rows, split):
            dom = FLORES_DOMAIN.get(p["domain_flores"], "education")
            yield dict(id=p["id"], domain=dom, licence="CC-BY-SA-4.0", licence_class="share_alike", licence_evidence=FLORES_EVIDENCE, url=p["url"], title=p["title"],
                       attribution=f'FLORES+ English ({p["domain_flores"]} article "{p["title"]}", {p["url"]}), CC BY-SA 4.0; cite ' + FLORES_CITES,
                       date="", paragraphs=[" ".join(p["sentences"])],
                       licence_note="FLORES+ card (read 2026-10-20): 'FLORES+ is a multilingual machine translation benchmark released under CC BY-SA 4.0'. The English sentences are taken from "
                                    "Wikinews, Wikivoyage and Wikibooks (Wikijunior) articles (CC BY-SA); human translations exist for many languages (attach via flores_plus_ids.json).",
                       extra={"flores_split": split, "ids": [p["ids"][0], p["ids"][-1]], "flores_ids": p["ids"], "n_sentences": len(p["ids"]), "domain": p["domain_flores"],
                              "topic": p["topic"], "citation": FLORES_CITES})


def flores_ids_map(pool_file: str) -> Dict[str, List[list]]:
    """row id -> [[split, [flores ids]]] for every row of a FLORES+ pool file (the coordinator attaches the human references with it)."""
    out: Dict[str, List[list]] = {}
    with open(pool_file, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            ex = d.get("extra") or {}
            if "flores_ids" in ex:
                out[d["id"]] = [[ex["flores_split"], ex["flores_ids"]]]
    return out


def ntrex_lines_map(pool_file: str) -> Dict[str, dict]:
    """row id -> {'file': NTREX source file, 'lines': [start, end) 0-based line range, valid for every NTREX-128 reference file}."""
    out: Dict[str, dict] = {}
    with open(pool_file, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            ex = d.get("extra") or {}
            if "ntrex_lines" in ex:
                out[d["id"]] = {"file": ex["ntrex_file"], "lines": ex["ntrex_lines"], "docid": ex["docid"]}
    return out



# ------------------------------------------------------------------------------------------------------------------ GlotStoryBook (English side)
GSB_REPO = "cis-lmu/GlotStoryBook"
GSB_REPOS = {"asp": ("global-asp/asp-source", "African Storybook Project (https://africanstorybook.org)"),
             "pb": ("global-asp/pb-source", "Pratham Books StoryWeaver (https://prathambooks.org/)"),
             "lida": ("global-asp/lida-source", "LIDA Stories (https://lidastories.net/)"),
             "lcb": ("global-asp/lcb-source", "Little Cree Books (http://littlecreebooks.com/)")}


def gsb_licence(raw: str) -> Tuple[str, str]:
    """GlotStoryBook `License` value -> (licence string, class)."""
    s = re.sub(r"[\s_]+", "-", raw.strip().upper())
    if s in ("PUBLIC-DOMAIN", "CC0", "CC0-1.0"):
        return "Public Domain", "pd"
    if "NC" in s.split("-"):
        return raw.strip().replace(" ", "-"), "noncommercial"
    if s in ("CC-BY", "CC-BY-4.0", "CC-BY-3.0"):
        return raw.strip().replace(" ", "-"), "permissive"
    return raw.strip(), "unclear"


def gsb_title(file_name: str) -> str:
    t = re.sub(r"\.md$", "", file_name)
    t = re.sub(r"^\d+[_-]", "", t)
    t = re.sub(r"_en$", "", t)
    return t.replace("-", " ").replace("_", " ").strip().capitalize()


def gsb_stories(rows: Iterable[dict]) -> Iterator[dict]:
    """English rows of GlotStoryBook.csv -> one story per (Source, File Name); one paragraph per `Text Number` row (page / sentence group), page order kept."""
    stories: "OrderedDict[Tuple[str, str], List[dict]]" = OrderedDict()
    for r in rows:
        if r.get("ISO639-3") == "eng":
            stories.setdefault((r["Source"], r["File Name"]), []).append(r)
    for (src, fn), rs in stories.items():
        rs.sort(key=lambda r: int(r["Text Number"] or 0))
        lic, cls = gsb_licence(rs[0]["License"])
        yield dict(source_repo=src, file_name=fn, rows=rs, licence=lic, cls=cls)


@source("glotstorybook_en", "stories", "per story: CC-BY / Public Domain / CC-BY-NC / CC-BY-NC-SA (License column of GlotStoryBook; original repositories github.com/global-asp)", "permissive",
        "https://huggingface.co/datasets/cis-lmu/GlotStoryBook", "HF cis-lmu/GlotStoryBook GlotStoryBook.csv, English rows (ISO639-3 = eng); `--only permissive|pd|noncommercial` selects the licence class of the output file")
def src_glotstorybook_en(ctx: Ctx):
    """English children's / language-learning stories of the African Storybook Project, Pratham Books (StoryWeaver), LIDA Stories and Little Cree Books, with the GlotStoryBook Parallel ID (human translations into African languages exist)."""
    import csv
    from huggingface_hub import hf_hub_download
    verbatim_mode(ctx, min_tokens=V3_MIN_TOKENS)
    want = (ctx.args.only or "permissive").split(",")
    path = hf_hub_download(GSB_REPO, "GlotStoryBook.csv", repo_type="dataset")
    csv.field_size_limit(10 ** 9)
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for st in gsb_stories(rows):
        if st["cls"] not in want:
            continue
        rs = st["rows"]
        repo, site = GSB_REPOS.get(rs[0]["Story Category"], (st["source_repo"], st["source_repo"]))
        authors = rs[0]["Text By"].strip()
        translators = rs[0]["Translation By"].strip()
        title = gsb_title(st["file_name"])
        yield dict(id=f"glotstorybook:{rs[0]['Parallel ID']}", source="glotstorybook_en", domain="stories", licence=st["licence"], licence_class=st["cls"],
                   licence_evidence="https://huggingface.co/datasets/cis-lmu/GlotStoryBook (License column) ; https://github.com/" + repo, url="https://github.com/" + repo, title=title,
                   attribution=f'{authors or "unknown author"}, "{title}", {site}, {st["licence"]}; collected in GlotStoryBook (cis-lmu) from github.com/{repo}',
                   date="", paragraphs=[r["Text"] for r in rs],
                   licence_note=("Per-story licence read from the License column of GlotStoryBook (the card itself says 'We do not own any of the text'); the stories come from github.com/" + repo +
                                 ". Used elsewhere in our project as a translation benchmark (GlotStoryBook): expect contamination flags. " +
                                 ("NC-SA: noncommercial AND share-alike. " if "SA" in st["licence"].upper() else "")),
                   extra={"parallel_id": rs[0]["Parallel ID"], "story_category": rs[0]["Story Category"], "file_name": st["file_name"], "text_numbers": [int(r["Text Number"] or 0) for r in rs],
                          "text_by": authors, "translation_by": translators, "source_repo": repo, "n_pages": len(rs)})


def nalibali_stories(rows: Iterable[dict]) -> Iterator[dict]:
    """English rows of GlotStoryBook `nalibali.csv` -> pool documents (paragraphs = non-empty lines of the story text)."""
    for r in rows:
        if r.get("iso639-3") != "eng":
            continue
        paras = [x for x in (normalise_space(l) for l in (r.get("story") or "").split("\n")) if x]
        author = re.sub(r"^\s*by\s+", "", (r.get("author") or "").strip(), flags=re.I)
        yield dict(parallel_id=r["parallel_id"], title=(r.get("title") or "").strip(), author=author, link=r.get("link", ""), paras=paras)


@source("nalibali_en", "stories", "Nal'ibali story library: terms page could not be read (JavaScript); GlotStoryBook card: 'We do not own any of the text'", "unclear",
        "https://nalibali.org/terms-of-use/", "HF cis-lmu/GlotStoryBook nalibali.csv, English rows")
def src_nalibali_en(ctx: Ctx):
    """English stories of the Nal'ibali (South African reading-for-enjoyment) story library, as collected in GlotStoryBook; multilingual parallel stories (parallel_id). Licence UNVERIFIED."""
    import csv
    from huggingface_hub import hf_hub_download
    verbatim_mode(ctx)
    path = hf_hub_download(GSB_REPO, "nalibali.csv", repo_type="dataset")
    csv.field_size_limit(10 ** 9)
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for st in nalibali_stories(rows):
        yield dict(id=f"nalibali:{st['parallel_id']}", domain="stories", licence="not stated / unverified", licence_class="unclear", licence_evidence="https://nalibali.org/terms-of-use/",
                   url=st["link"], title=st["title"], attribution=f'{st["author"] or "unknown author"}, "{st["title"]}", Nal\u2019ibali, {st["link"]}; collected in GlotStoryBook (cis-lmu)',
                   date="", paragraphs=st["paras"],
                   licence_note="Licence NOT verified: https://nalibali.org/terms-of-use/ returned no readable licence text without JavaScript on 2026-10-20 (footer: '(c) 2024 Nal\u2019ibali'); the GlotStoryBook card says "
                                "'We do not own any of the text' and carries no licence for this subset. Keep out of a public release unless Nal\u2019ibali confirms the terms.",
                   extra={"parallel_id": st["parallel_id"], "author": st["author"]})


# --------------------------------------------------------------------------------------------------------------------------- UN News
UN_TERMS_URL = "https://www.un.org/en/about-us/terms-of-use"
UN_TERMS_QUOTE = ("The United Nations grants permission to Users to visit the Site and to download and copy the information, documents and materials (collectively, \"Materials\") from the Site "
                  "for the User's personal, non-commercial use, without any right to resell or redistribute them or to compile or create derivative works therefrom, subject to the terms and "
                  "conditions outlined below, and also subject to more specific restrictions that may apply to specific Material within this Site.")
UN_TOPICS = ["africa", "climate-change", "health", "human-rights", "peace-and-security", "economic-development", "migrants-and-refugees", "women", "humanitarian-aid",
             "culture-and-education", "un-affairs", "sdgs", "law-and-crime-prevention", "global-perspective-human-stories", "middle-east", "asia-pacific", "americas", "europe"]
_UN_STORY = re.compile(r'href="(/en/story/\d{4}/\d{2}/\d+)"')


def un_story_paragraphs(html_text: str) -> Tuple[str, str, List[str]]:
    """(title, published date, paragraphs) of one UN News story page: the <p> of the main text blocks only (not captions, embeds, 'Read more' / 'Listen' lines)."""
    root = parse_html(html_text)
    if root is None:
        return "", "", []
    title = ""
    for el in root.xpath('//meta[@property="og:title"]/@content'):
        title = re.sub(r"\s*\|\s*UN News\s*$", "", el).strip()
        break
    date = ""
    for el in root.xpath('//meta[@property="article:published_time"]/@content'):
        date = el[:10]
        break
    paras: List[str] = []
    for div in root.xpath('//div[contains(@class,"field--name-field-text-column")]'):
        for p in div.xpath("./p"):
            t = normalise_space(p.text_content())
            if not t or re.match(r"^(?:read more|more on|listen|watch|see also|related|follow|click|download|►|►)", t, re.I) or t.startswith(("UN News", "©")):
                continue
            paras.append(t)
    return title, date, paras


@source("unnews", "news", "UN website terms of use: personal non-commercial use, no redistribution (https://www.un.org/en/about-us/terms-of-use)", "noncommercial", UN_TERMS_URL,
        "news.un.org topic listings (robots.txt honoured, <= 1 request/s) + story pages")
def src_unnews(ctx: Ctx):
    """UN News English stories (topic listing pages -> /en/story/ pages). Licence class noncommercial (UN terms of use grant personal non-commercial use only)."""
    a = ctx.args
    ctx.pool.min_tokens = 100
    ctx.pool.min_paragraphs = 2
    http = ctx.http
    max_pages = 30
    urls: "OrderedDict[str, None]" = OrderedDict()
    topics = [t for t in (a.only or ",".join(UN_TOPICS)).split(",") if t]
    # interleave topics page by page so that a small --max-docs still spans every topic
    for page in range(max_pages):
        for t in topics:
            u = f"https://news.un.org/en/news/topic/{t}" + (f"?page={page}" if page else "")
            try:
                html_text = http.text(u, allow_404=True)
            except Exception as e:
                ctx.log("listing", u, "failed", str(e)[:80])
                continue
            for m in _UN_STORY.finditer(html_text or ""):
                urls.setdefault("https://news.un.org" + m.group(1), None)
        ctx.log("listing page", page, "story urls", len(urls))
        if len(urls) >= a.max_docs * 1.6:
            break
    for u in urls:
        if ctx.pool.full:
            break
        did = "unnews:" + u.rsplit("/", 1)[-1]
        if ctx.pool.seen(did):
            continue
        try:
            html_text = http.text(u, allow_404=True)
        except Exception as e:
            ctx.log("story", u, "failed", str(e)[:80])
            continue
        if not html_text:
            continue
        title, date, paras = un_story_paragraphs(html_text)
        yield dict(id=did, domain="news", licence="UN website terms of use (personal non-commercial use)", licence_class="noncommercial", licence_evidence=UN_TERMS_URL, url=u, title=title,
                   attribution=f'UN News, "{title}", {u}, © United Nations', date=date, paragraphs=paras,
                   licence_note="UN terms of use (read 2026-10-20), " + UN_TERMS_URL + ": '" + UN_TERMS_QUOTE + "' Redistribution is not granted: permission from the UN is needed for a public release; page meta rights: 'COPYRIGHT (c) UNITED NATIONS'.")


# ------------------------------------------------------------------------------------------------------------------ Simple English Wikipedia
SW_EVIDENCE = "https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use"


def simplewiki_docs(path: str) -> Iterator[dict]:
    for title, wt, ts, pid in iter_dump_pages(path):
        if re.match(r"(?:List of|Lists of|Index of|Outline of|Timeline of|Wikipedia:|Category:)", title or "") or re.search(r"\{\{\s*(?:disambig|disambiguation|stub)\b", wt, re.I) and len(wt) < 800:
            continue
        paras = wikitext_to_paragraphs(wt, cut_sections=True)
        yield dict(id=f"simplewiki:{pid}", domain="encyclopedia", licence="CC-BY-SA-4.0 / CC-BY-SA-3.0 + GFDL", licence_class="share_alike", licence_evidence=SW_EVIDENCE,
                   url="https://simple.wikipedia.org/wiki/" + (title or "").replace(" ", "_"), title=title or "",
                   attribution='Simple English Wikipedia contributors, "%s", simple.wikipedia.org, CC BY-SA' % title, date=(ts or "")[:10], paragraphs=paras,
                   licence_note="share-alike: derivatives (translations) must carry CC BY-SA; text may include CC BY-SA 3.0 imported material")


@source("simplewiki", "encyclopedia", "CC-BY-SA-4.0 / CC-BY-SA-3.0 + GFDL", "share_alike", SW_EVIDENCE,
        "dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2")
def src_simplewiki(ctx: Ctx):
    """Simple English Wikipedia articles (short, plain English; encyclopedia): short and medium documents, from 150 Gemma tokens."""
    ctx.pool.min_tokens = 150
    ctx.pool.min_paragraphs = 2
    raw = ctx.raw_dir or os.path.join(os.path.dirname(os.path.abspath(ctx.args.out)), "..", "raw", "dumps")
    os.makedirs(raw, exist_ok=True)
    path = os.path.join(raw, "simplewiki-latest-pages-articles.xml.bz2")
    _download(ctx, "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2", path)
    for d in simplewiki_docs(path):
        if not ctx.pool.seen(d["id"]) and sampled(d["id"], ctx.args.seed, ctx.args.sample):
            yield d
    if not ctx.args.keep_raw:
        _rm(path)


# ------------------------------------------------------------------------------------------------------------------ Wikijunior (Wikibooks)
WB_EVIDENCE = "https://en.wikibooks.org/wiki/Wikibooks:Copyrights"


def wikibooks_docs(path: str, prefixes: Sequence[str], ns: str = "110") -> Iterator[dict]:       # namespace 110 = Wikijunior
    for title, wt, ts, pid in iter_dump_pages(path, ns=ns):
        if not title or not any(title.startswith(p) for p in prefixes):
            continue
        paras = wikitext_to_paragraphs(wt, cut_sections=True)
        top = title.split("/")[0]
        yield dict(id=f"wikibooks:{pid}", domain="education", licence="CC-BY-SA-4.0", licence_class="share_alike", licence_evidence=WB_EVIDENCE + " ; siteinfo rightsinfo: Creative Commons Attribution-Share Alike 4.0",
                   url="https://en.wikibooks.org/wiki/" + title.replace(" ", "_"), title=title, attribution='Wikibooks contributors, "%s", en.wikibooks.org, CC BY-SA' % title,
                   date=(ts or "")[:10], paragraphs=paras, extra={"book": top},
                   licence_note="share-alike: derivatives (translations) must carry CC BY-SA 4.0 (en.wikibooks.org siteinfo rightsinfo read 2026-10-20); older revisions may be CC BY-SA 3.0 / GFDL. Wikijunior = Wikibooks children's non-fiction series.")


@source("wikijunior", "education", "CC-BY-SA-4.0 (en.wikibooks.org siteinfo rightsinfo)", "share_alike", WB_EVIDENCE,
        "dumps.wikimedia.org/enwikibooks/latest/enwikibooks-latest-pages-articles.xml.bz2 (pages titled 'Wikijunior:...'; --only = comma-separated title prefixes)")
def src_wikijunior(ctx: Ctx):
    """Wikijunior children's non-fiction chapters from the English Wikibooks dump (explanatory, simple register; FLORES takes its education passages from the same series)."""
    ctx.pool.min_tokens = 100
    ctx.pool.min_paragraphs = 2
    raw = ctx.raw_dir or os.path.join(os.path.dirname(os.path.abspath(ctx.args.out)), "..", "raw", "dumps")
    os.makedirs(raw, exist_ok=True)
    path = os.path.join(raw, "enwikibooks-latest-pages-articles.xml.bz2")
    _download(ctx, "https://dumps.wikimedia.org/enwikibooks/latest/enwikibooks-latest-pages-articles.xml.bz2", path)
    for d in wikibooks_docs(path, [x for x in (ctx.args.only or "Wikijunior").split(",") if x]):
        if not ctx.pool.seen(d["id"]):
            yield d
    if not ctx.args.keep_raw:
        _rm(path)


# ------------------------------------------------------------------------------------------------------------------ MAFAND-MT (English news sentences)
MAFAND_RAW = "https://raw.githubusercontent.com/masakhane-io/lafand-mt/main/data/json_files/"
MAFAND_PAIRS = ("amh", "hau", "ibo", "kin", "lug", "luo", "nya", "pcm", "sna", "swa", "tsn", "twi", "xho", "yor", "zul")
MAFAND_PAPER = "Adelani et al. 2022, A Few Thousand Translations Go a Long Way! Leveraging Pre-trained Models for African News Translation, NAACL, https://aclanthology.org/2022.naacl-main.223"


def mafand_items(files: Dict[Tuple[str, str], str]) -> Iterator[dict]:
    """files: (language code, split) -> JSON-lines text of that en-xx file ({"translation": {"en": ..., "<lang>": ...}}). The same English text in several pairs becomes ONE item
    with all its human references (`parallel`: {lang: reference})."""
    items: "OrderedDict[str, dict]" = OrderedDict()
    for (lang, split), text in sorted(files.items()):
        for i, line in enumerate(text.splitlines()):
            if not line.strip():
                continue
            tr = json.loads(line).get("translation") or {}
            en, ref = (tr.get("en") or "").strip(), (tr.get(lang) or "").strip()
            if not en or not ref:
                continue
            key = fingerprint(en)
            it = items.setdefault(key, dict(en=en, parallel={}, where=[]))
            it["parallel"].setdefault(lang, ref)
            it["where"].append(f"{lang}:{split}:{i}")
    for it in items.values():
        yield it


@source("mafand_en", "news", "CC-BY-NC-4.0 (HF masakhane/mafand card; MAFAND-MT paper); GPL-3.0 LICENSE file in the code repository", "noncommercial", "https://huggingface.co/datasets/masakhane/mafand",
        "github.com/masakhane-io/lafand-mt data/json_files/en-xx/{test,dev}.json (--only = comma-separated language codes)")
def src_mafand_en(ctx: Ctx):
    """English side of the MAFAND-MT African news translation sets (test + dev), human translations in up to 15 of our languages; one item per English segment (sentence or short news item)."""
    verbatim_mode(ctx)
    langs = [x for x in (ctx.args.only or ",".join(MAFAND_PAIRS)).split(",") if x]
    files: Dict[Tuple[str, str], str] = {}
    for lang in langs:
        for split in ("test", "dev"):
            t = ctx.http.text(f"{MAFAND_RAW}en-{lang}/{split}.json", allow_404=True)
            if t:
                files[(lang, split)] = t
    for it in mafand_items(files):
        did = "mafand:" + fingerprint(it["en"])[:14]
        if ctx.pool.seen(did):
            continue
        yield dict(id=did, domain="news", licence="CC-BY-NC-4.0", licence_class="noncommercial", licence_evidence="https://huggingface.co/datasets/masakhane/mafand (card licence: cc-by-nc-4.0)",
                   url="https://github.com/masakhane-io/lafand-mt", title="", attribution="MAFAND-MT (masakhane-io/lafand-mt), " + MAFAND_PAPER + "; news sources named in the paper", date="",
                   paragraphs=[it["en"]],
                   licence_note="HF card of masakhane/mafand (read 2026-10-20): license cc-by-nc-4.0; the code repository LICENSE is GPL-3.0. Noncommercial. Widely used African MT benchmark: expect contamination flags "
                                "(training data of African MT systems).",
                   extra={"parallel": it["parallel"], "where": it["where"], "citation": MAFAND_PAPER})


# ------------------------------------------------------------------------------------------------------------------ WMT24++ human references
WMT24PP_REPO = "google/wmt24pp"
WMT24PP_CONFIGS = ("en-sw_KE", "en-sw_TZ", "en-zu_ZA")          # the African languages of WMT24++ that are in our 77 (Swahili, Zulu)


def wmt24pp_map(pool_files: Sequence[str], configs: Sequence[str] = WMT24PP_CONFIGS, hf_files: Optional[Dict[str, str]] = None) -> Dict[str, dict]:
    """Pool row id -> {'document_id', 'domain', 'segment_ids': {config: [ids]}, 'n_segments'} for the WMT24 documents that have WMT24++ human post-edited references
    (google/wmt24pp, HF card licence apache-2.0; Deutsch et al. 2025, https://arxiv.org/abs/2502.12404). `hf_files` (config -> local jsonl) overrides the Hub download."""
    segs: Dict[str, Dict[str, Dict[str, str]]] = {}          # config -> document_id -> {segment_id: source}
    for cfg in configs:
        path = (hf_files or {}).get(cfg)
        if path is None:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(WMT24PP_REPO, cfg + ".jsonl", repo_type="dataset")
        d: Dict[str, Dict[str, str]] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("is_bad_source"):
                    continue
                d.setdefault(r["document_id"], {})[str(r["segment_id"])] = r["source"]
        segs[cfg] = d
    out: Dict[str, dict] = {}
    for pf in pool_files:
        with open(pf, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                ex = row.get("extra") or {}
                if ex.get("testset") != "wmttest2024":
                    continue
                docid = ex["docid"]
                ent = {"document_id": docid, "hf": WMT24PP_REPO, "segment_ids": {}}
                for cfg in configs:
                    if docid in segs[cfg]:
                        ent["segment_ids"][cfg] = sorted(int(k) for k in segs[cfg][docid])
                if ent["segment_ids"]:
                    ent["n_segments"] = {c: len(v) for c, v in ent["segment_ids"].items()}
                    out[row["id"]] = ent
    return out


# ===================================================================================================================== CLI (keep last)
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="source", required=True)
    sub.add_parser("list", help="print the source table")
    st = sub.add_parser("stats", help="write POOL_STATS.json (+ samples/) for a directory of pool files")
    st.add_argument("--pool-dir", required=True)
    st.add_argument("--out", required=True)
    st.add_argument("--samples-dir", default=None)
    cp = sub.add_parser("capacity", help="domain x length-range capacity table of the pd + permissive pool files")
    cp.add_argument("--pool-dir", required=True)
    cp.add_argument("--exclude", default="openstax_ccby_snapshot,share_alike")
    w24 = sub.add_parser("wmt24pp-ids", help="map WMT24 pool rows to the WMT24++ human references (Swahili, Zulu) -> JSON")
    w24.add_argument("--in", dest="inp", required=True, help="comma-separated pool files")
    w24.add_argument("--out", required=True)
    gv = sub.add_parser("gv-authors", help="fill the Global Voices author names from the Common Pile globalvoices shards")
    gv.add_argument("--in", dest="inp", required=True)
    gv.add_argument("--out", required=True)
    rt = sub.add_parser("retokenize", help="recount the Gemma tokens of a pool file (e.g. fetched where no tokenizer exists)")
    rt.add_argument("--in", dest="inp", required=True)
    rt.add_argument("--out", required=True)
    rt.add_argument("--tokenizer", default=None)
    for name, meta in SOURCES.items():
        sp = sub.add_parser(name, help=meta["doc"].split("\n")[0][:120])
        sp.add_argument("--out", required=True, help="pool JSONL to write (atomically: <out>.part -> <out>)")
        sp.add_argument("--max-docs", type=int, default=1000)
        sp.add_argument("--raw-dir", default=None, help="scratch directory for downloaded dumps / archives")
        sp.add_argument("--tokenizer", default=None, help="Gemma tokenizer.json (default: search the HF cache)")
        sp.add_argument("--publish-every", type=int, default=150, help="atomic snapshot of <out> every N accepted documents (0 = only at the end)")
        sp.add_argument("--sample", type=float, default=1.0, help="dump sources: keep this fraction of the candidate pages (hash of the page id)")
        sp.add_argument("--seed", type=int, default=20261020)
        sp.add_argument("--min-interval", type=float, default=1.0, help="seconds between two requests to one host (>= 1.0)")
        sp.add_argument("--per-book", type=int, default=3, help="gutenberg / openstax: documents per book")
        sp.add_argument("--keep-raw", action="store_true", help="do not delete downloaded dumps afterwards")
        sp.add_argument("--shards", default=None, help="cp_usgpo: slice 'i:j' of the (seed-shuffled) shard list, to split the work over several processes")
        sp.add_argument("--exclude-from", default=None, help="comma-separated existing pool files whose ids / texts are skipped (extend a pool without duplicates)")
        sp.add_argument("--only", default=None, help="comma-separated sub-selection (source specific: domains / books / agencies / countries)")
    return ap


def run_source(name: str, args) -> int:
    meta = SOURCES[name]
    tc = TokenCounter(args.tokenizer)
    if not tc.exact:
        print("[warn] no Gemma tokenizer.json found: token counts are chars/4 estimates", file=sys.stderr)
    http = Http(min_interval=max(1.0, args.min_interval))
    pool = Pool(args.out, name, tc, args.max_docs, exclude_from=[x for x in (args.exclude_from or "").split(",") if x])
    pool.publish_every = args.publish_every
    ctx = Ctx(args, http, pool)
    if args.raw_dir:
        os.makedirs(args.raw_dir, exist_ok=True)
    t0 = time.time()
    ok = False
    try:
        r = meta["fn"](ctx)
        if r is not None:                       # generator style sources
            for d in r:
                if d is None:
                    continue
                pool.add(d)
                if pool.full:
                    break
        ok = True
    finally:
        pool.close(complete=ok)
    print(json.dumps(dict(source=name, out=args.out, docs=pool.n, requests=http.n_requests, rejected=dict(pool.stats), seconds=round(time.time() - t0, 1)),
                     ensure_ascii=False))
    return 0


def _atomic_json(path: str, obj) -> None:
    with open(path + ".part", "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=0, sort_keys=True)
    os.replace(path + ".part", path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.source == "stats":
        st = pool_stats(args.pool_dir, args.samples_dir)
        with open(args.out + ".part", "w", encoding="utf-8") as f:
            json.dump(st, f, ensure_ascii=False, indent=1, sort_keys=True)
        os.replace(args.out + ".part", args.out)
        print(f"{len(st['files'])} pool files -> {args.out}")
        return 0
    if args.source == "capacity":
        res = capacity_table(args.pool_dir, [x for x in args.exclude.split(",") if x])
        print(f"{'domain':20s} {'400-799':>8s} {'800-1599':>9s} {'1600-3199':>10s} {'>=3200':>7s} {'total':>7s} | usable 512 1024 2048 4096")
        for dom, v in res["domains"].items():
            print(f"{dom:20s} {v['r400']:8d} {v['r800']:9d} {v['r1600']:10d} {v['r3200']:7d} {v['total']:7d} | {v['usable512']:6d} {v['usable1024']:5d} {v['usable2048']:5d} {v['usable4096']:5d}")
        return 0
    if args.source == "wmt24pp-ids":
        m = wmt24pp_map([x for x in args.inp.split(",") if x])
        _atomic_json(args.out, m)
        print(f"{len(m)} WMT24 pool rows have WMT24++ references -> {args.out}")
        return 0
    if args.source == "gv-authors":
        return gv_authors(args.inp, args.out)
    if args.source == "retokenize":
        return retokenize(args.inp, args.out, args.tokenizer)
    if args.source == "list":
        for n, m in SOURCES.items():
            print(f"{n:24s} {m['cls']:13s} {m['licence']:40s} {m['domains']}")
        return 0
    rc = run_source(args.source, args)
    base = os.path.dirname(os.path.abspath(args.out))
    if rc == 0 and args.source == "flores_plus" and os.path.exists(args.out):
        _atomic_json(os.path.join(base, "flores_plus_ids.json"), flores_ids_map(args.out))
    if rc == 0 and args.source == "ntrex128" and os.path.exists(args.out):
        _atomic_json(os.path.join(base, "ntrex128_lines.json"), ntrex_lines_map(args.out))
    return rc


if __name__ == "__main__":
    sys.exit(main())
