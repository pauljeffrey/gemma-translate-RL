# SPDX-License-Identifier: AGPL-3.0-or-later
"""Human-reference material of the long-document benchmark and the sentence sets of the translator bake-off (D-44).

HUMAN DOCUMENT REFERENCES (policy (c) of the plan: human references wherever they exist, synthetic references only fill the gaps)
  SmolDoc        google/smol smoldoc (CC-BY-4.0): professionally translated documents, sentence aligned by the publisher (`srcs` / `trgs` lists of equal length,
                 `sep` = the whitespace between sentences). 45 of our 77 languages; 130 documents per low-resource language; median ~230 Gemma tokens of English.
  FLORES+        openlanguagedata/flores_plus (CC-BY-SA-4.0): 2-6 consecutive sentences of one Wikinews / Wikibooks / Wikivoyage article (median ~110 tokens).
  stitched       several SmolDoc documents (or FLORES+ passages) of one language concatenated to a 512 / 1024 / 2048 token document with its HUMAN reference:
                 not one coherent text, but a real long-context test (omission, loops, drift over thousands of tokens) with a perfect reference. `ref_kind` =
                 "human_stitched"; the plain ones are "human".
Every document carries `contaminated` (the English side was in the SFT training windows: exact OR >= 50% of the 8-grams, the D-42 flag files) so both the
clean-only and the all-documents numbers can be reported.

SENTENCE SETS OF THE BAKE-OFF (`load_refset`): FLORES+ dev (997 sentences, the translator selection set; downloaded once from the gated Hub repo with the box's read
access), FLORES+ devtest (local), SmolSent (863 sentences, local) and the MAFAND dev rows of bench_dev.jsonl, per language, rows {id, english, reference}.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import OrderedDict, defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from gemma_mt.eval import longdoc as L
from gemma_mt.languages import LANGUAGES, SUPPORTED_RL, canonical_iso

DEFAULT_BENCH_ROOT = "/workspace/sft_build/bench"
FLORES_REPO = "openlanguagedata/flores_plus"
REFSETS = ("flores_dev", "flores_devtest", "smol_sent", "mafand_dev", "ibom_mt", "flores_moz", "glot_story", "salt")
REFSET_LICENCE = {"flores_dev": "CC-BY-SA-4.0", "flores_devtest": "CC-BY-SA-4.0", "smol_sent": "CC-BY-4.0", "mafand_dev": "CC-BY-4.0 (MAFAND-MT)", "ibom_mt": "CC-BY-NC-4.0",
                  "flores_moz": "CC-BY-SA-4.0", "glot_story": "per row CC-BY / CC-BY-NC", "salt": "CC-BY-SA-4.0"}
FALLBACK_ORDER = ("flores_dev", "smol_sent", "flores_moz", "ibom_mt", "glot_story", "salt", "mafand_dev", "flores_devtest")   # which human set a language is bake-off'ed on, first available


def _read_jsonl_rows(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


# ------------------------------------------------------------------------------------------------ FLORES+
def fetch_flores_dev(codes: Sequence[str], cache_dir: str, token: Optional[str] = None, log: Callable[[str], None] = print) -> Dict[str, str]:
    """Download `dev/<code>.jsonl` of the gated FLORES+ repo into `cache_dir` (skips files that exist). The token is taken from the environment by
    huggingface_hub and is never printed. Returns {code: path}."""
    from huggingface_hub import hf_hub_download

    os.makedirs(cache_dir, exist_ok=True)
    out = {}
    for code in codes:
        dst = os.path.join(cache_dir, f"{code}.jsonl")
        if not os.path.isfile(dst):
            try:
                p = hf_hub_download(FLORES_REPO, f"dev/{code}.jsonl", repo_type="dataset", token=token)
            except Exception as e:                               # a missing file is a missing language, not a crash
                log(f"[refs] FLORES+ dev/{code}.jsonl: {type(e).__name__}")
                continue
            tmp = dst + ".part"
            with open(p, "rb") as a, open(tmp, "wb") as b:
                b.write(a.read())
            os.replace(tmp, dst)
        out[code] = dst
    return out


def flores_rows(iso: str, split: str, bench_root: str = DEFAULT_BENCH_ROOT, dev_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """FLORES+ sentence pairs of one language: [{id, english, reference, url, domain, topic}]. devtest comes from bench/test_sets/flores_plus/devtest, dev from
    `dev_dir` (fetch_flores_dev). [] when the language has no FLORES+ code or the file is missing."""
    code = LANGUAGES[canonical_iso(iso)].flores
    if not code:
        return []
    if split == "devtest":
        base = os.path.join(bench_root, "test_sets", "flores_plus", "devtest")
    elif split == "dev":
        base = dev_dir or os.path.join(os.path.dirname(bench_root.rstrip("/")), "longdoc_bench", "raw", "flores_dev")
    else:
        raise ValueError(split)
    pt, pe = os.path.join(base, f"{code}.jsonl"), os.path.join(base, "eng_Latn.jsonl")
    if not (os.path.isfile(pt) and os.path.isfile(pe)):
        return []
    en = {str(r["id"]): r for r in _read_jsonl_rows(pe)}
    out = []
    for r in _read_jsonl_rows(pt):
        e = en.get(str(r["id"]))
        if e and r.get("text") and e.get("text"):
            out.append({"id": str(r["id"]), "english": e["text"], "reference": r["text"], "url": e.get("url", ""), "domain": e.get("domain", ""), "topic": e.get("topic", "")})
    return out


def load_refset(refset: str, iso: str, bench_root: str = DEFAULT_BENCH_ROOT, dev_dir: Optional[str] = None, limit: int = 0) -> List[Dict[str, Any]]:
    """Human sentence pairs of one language for the bake-off: rows {id, english, reference} (+ url / domain for FLORES+). [] when not available."""
    iso = canonical_iso(iso)
    if refset in ("flores_dev", "flores_devtest"):
        rows = flores_rows(iso, "dev" if refset == "flores_dev" else "devtest", bench_root, dev_dir)
    elif refset == "smol_sent":
        code = LANGUAGES[iso].smol_sent
        p = os.path.join(bench_root, "test_sets", "smol_sent", f"en_{code}.jsonl") if code else ""
        rows = []
        if p and os.path.isfile(p):
            for k, r in enumerate(_read_jsonl_rows(p)):
                if r.get("sl", "en") == "en" and (r.get("src") or "").strip() and (r.get("trg") or "").strip():
                    rows.append({"id": str(r.get("id", k)), "english": r["src"].strip(), "reference": r["trg"].strip()})
    elif refset == "mafand_dev":
        p = os.path.join(bench_root, "bench_dev.jsonl")
        rows = []
        if os.path.isfile(p):
            for k, r in enumerate(_read_jsonl_rows(p)):
                if r.get("benchmark") == "mafand_dev" and canonical_iso(r.get("iso", "")) == iso:
                    rows.append({"id": str(r.get("id", k)), "english": r["english"], "reference": r["reference"]})
    elif refset in ("ibom_mt", "flores_moz", "glot_story", "salt"):
        from gemma_mt.eval import benchmarks as B

        path = os.path.join(bench_root, "test_sets", refset)
        if not os.path.isdir(path):
            return []
        bm = B.load_benchmark(refset, path=path, isos=[iso])
        rows = [{"id": str(r["id"]), "english": r["english"], "reference": r["reference"]} for r in bm]
    else:
        raise ValueError(f"unknown refset {refset!r}; choose from {REFSETS}")
    return rows[:limit] if limit else rows


def best_refset(iso: str, bench_root: str = DEFAULT_BENCH_ROOT, dev_dir: Optional[str] = None, min_rows: int = 100) -> Optional[str]:
    """The first set of FALLBACK_ORDER that has >= min_rows sentences for the language (None = no human sentence set at all)."""
    for rs in FALLBACK_ORDER:
        if len(load_refset(rs, iso, bench_root, dev_dir)) >= min_rows:
            return rs
    return None


def refset_coverage(bench_root: str = DEFAULT_BENCH_ROOT, dev_dir: Optional[str] = None, isos: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, int]]:
    """{iso: {refset: n sentences}} for the 77 languages (0 = none): which languages can be bake-off'ed on which human sentence set."""
    out: Dict[str, Dict[str, int]] = {}
    for iso in (isos or SUPPORTED_RL):
        out[iso] = {rs: len(load_refset(rs, iso, bench_root, dev_dir)) for rs in REFSETS}
    return out


# ------------------------------------------------------------------------------------------------ contamination flags (D-42 files)
class DocFlags:
    """Lookup of the D-42 contamination flag files: `contaminated(benchmark, iso, row_id)` -> True / False / None (unknown)."""

    def __init__(self, bench_root: str = DEFAULT_BENCH_ROOT):
        self.root = bench_root
        self._loaded: Dict[str, Dict[str, Any]] = {}

    def _file(self, name: str) -> Optional[Dict[str, Any]]:
        if name not in self._loaded:
            cands = [os.path.join(self.root, "test_sets", "_contamination", f"contamination_{name}.json"), os.path.join(self.root, f"contamination_{name}.json")]
            self._loaded[name] = None
            for p in cands:
                if os.path.isfile(p):
                    with open(p, encoding="utf-8") as f:
                        self._loaded[name] = json.load(f)
                    break
        return self._loaded[name]

    def contaminated(self, name: str, iso: str, row_id: Any) -> Optional[bool]:
        d = self._file(name)
        if d is None:
            return None
        v = d["exact"].get(f"{iso}:{row_id}")
        return None if v is None else bool(v)


# ------------------------------------------------------------------------------------------------ human documents
def _join_sentences_with_sep(parts: Sequence[str], seps: Optional[Sequence[str]]) -> Tuple[str, List[int]]:
    """Document text from SmolDoc sentences and their separators, plus the paragraph starts (a separator with a line break before a sentence starts a paragraph)."""
    if not seps or len(seps) != len(parts):
        seps = [" "] * len(parts)
    starts, text = [0], []
    for i, (p, s) in enumerate(zip(parts, seps)):
        text.append(p)
        text.append(s)
        if i + 1 < len(parts) and "\n" in s:
            starts.append(i + 1)
    return "".join(text).strip(), starts


def smoldoc_docs(iso: str, bench_root: str = DEFAULT_BENCH_ROOT, flags: Optional[DocFlags] = None, count_tokens: Optional[Callable[[str], int]] = None,
                 limit: int = 0) -> List[L.RefDoc]:
    """SmolDoc English -> `iso` as RefDocs (ref_kind "human"). The publisher's sentence alignment (`srcs` / `trgs`) is the alignment unit; the hypothesis of
    a system is aligned to it with the length model, so the English units need not equal `split_sentences` units."""
    iso = canonical_iso(iso)
    code = LANGUAGES[iso].smol_doc
    p = os.path.join(bench_root, "test_sets", "smol_doc", f"en_{code}.jsonl") if code else ""
    if not (p and os.path.isfile(p)):
        return []
    flags = flags or DocFlags(bench_root)
    out = []
    for k, r in enumerate(_read_jsonl_rows(p)):
        srcs, trgs = r.get("srcs") or [], r.get("trgs") or []
        if r.get("sl", "en") != "en" or not srcs or len(srcs) != len(trgs) or any(not (s or "").strip() for s in srcs) or any(not (t or "").strip() for t in trgs):
            continue
        rid = r.get("id", k)
        text, starts = _join_sentences_with_sep(srcs, r.get("sep"))
        d = L.RefDoc(doc_id=f"smoldoc:{rid}", iso=iso, source=text, src_sents=[L.norm_text(s) for s in srcs], ref_sents=[L.norm_text(t) for t in trgs],
                     ref_kind="human", ref_system="smoldoc", domain="smoldoc_" + str(rid).split("_")[0], para_starts=starts,
                     contaminated=flags.contaminated("smol_doc", iso, rid))
        if count_tokens:
            d.bin = L.bin_of(count_tokens(d.source))
        out.append(d)
        if limit and len(out) >= limit:
            break
    return out


def flores_passage_docs(iso: str, split: str = "devtest", bench_root: str = DEFAULT_BENCH_ROOT, dev_dir: Optional[str] = None, flags: Optional[DocFlags] = None,
                        count_tokens: Optional[Callable[[str], int]] = None) -> List[L.RefDoc]:
    """FLORES+ sentences of one article (same url, consecutive ids) as one document with its human reference (ref_kind "human")."""
    iso = canonical_iso(iso)
    rows = flores_rows(iso, split, bench_root, dev_dir)
    flags = flags or DocFlags(bench_root)
    groups: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    for r in rows:
        groups.setdefault(r["url"] or f"noid-{r['id']}", []).append(r)
    out = []
    for url, rs in groups.items():
        src = [L.norm_text(r["english"]) for r in rs]
        ref = [L.norm_text(r["reference"]) for r in rs]
        fl = [flags.contaminated("flores_plus" if split == "devtest" else "dev", iso, r["id"] if split == "devtest" else f"{iso}-{r['id']}") for r in rs]
        known = [f for f in fl if f is not None]
        d = L.RefDoc(doc_id=f"flores_{split}:{rs[0]['id']}-{rs[-1]['id']}", iso=iso, source=" ".join(src), src_sents=src, ref_sents=ref, ref_kind="human",
                     ref_system=f"flores_plus_{split}", domain="flores_" + rs[0].get("domain", ""), para_starts=[0],
                     contaminated=(any(known) if known else None))
        if count_tokens:
            d.bin = L.bin_of(count_tokens(d.source))
        out.append(d)
    return out


def _stable_order_key(doc_id: str, salt: str) -> str:
    return hashlib.sha1(f"{salt}|{doc_id}".encode("utf-8")).hexdigest()


def stitch_documents(docs: Sequence[L.RefDoc], target_tokens: int, count_tokens: Callable[[str], int], *, salt: str = "stitch", tolerance: float = 0.2,
                     prefix: str = "stitch") -> List[L.RefDoc]:
    """Concatenate whole documents of ONE language (human references) into documents of about `target_tokens` Gemma tokens (English side), in a stable pseudo-
    random order (sha1 of the doc id + salt, so languages that share the same English documents get the SAME stitched English documents). Each component
    starts a paragraph. A group is closed when it reaches target*(1-tolerance); the last incomplete group is dropped. Returns RefDocs with ref_kind
    "human_stitched" and `contaminated` = any component contaminated (None when no component is known)."""
    docs = sorted(docs, key=lambda d: _stable_order_key(d.doc_id, salt))
    out: List[L.RefDoc] = []
    cur: List[L.RefDoc] = []
    cur_tok = 0
    lo = target_tokens * (1 - tolerance)
    hi = target_tokens * (1 + tolerance)
    for d in docs:
        t = count_tokens(d.source)
        if cur and cur_tok + t > hi:                       # this document would overshoot: close the group if it is big enough, else start over with it
            if cur_tok >= lo:
                out.append(_merge(cur, prefix, target_tokens, count_tokens))
            cur, cur_tok = [], 0
        cur.append(d)
        cur_tok += t
        if cur_tok >= lo and cur_tok >= target_tokens * 0.95:
            out.append(_merge(cur, prefix, target_tokens, count_tokens))
            cur, cur_tok = [], 0
    return out


def _merge(parts: Sequence[L.RefDoc], prefix: str, target: int, count_tokens: Callable[[str], int]) -> L.RefDoc:
    src, ref, starts, flags = [], [], [], []
    for d in parts:
        base = len(src)
        starts.extend(base + s for s in (d.para_starts or [0]))
        src.extend(d.src_sents)
        ref.extend(d.ref_sents)
        flags.extend(d.flags or [""] * len(d.ref_sents))
    known = [p.contaminated for p in parts if p.contaminated is not None]
    did = f"{prefix}{target}:" + hashlib.sha1("+".join(p.doc_id for p in parts).encode()).hexdigest()[:12]
    text = L.join_sentences(src, starts)
    return L.RefDoc(doc_id=did, iso=parts[0].iso, source=text, src_sents=src, ref_sents=ref, ref_kind="human_stitched", ref_system=parts[0].ref_system,
                    domain="stitched", bin=L.bin_of(count_tokens(text)), para_starts=starts, flags=flags, contaminated=(any(known) if known else None))


def docs_to_files(docs: Sequence[L.RefDoc], out_dir: str, tag: str = "human") -> Dict[str, int]:
    """Write the benchmark file layout: docs_<tag>.jsonl (English, once per doc id) and refs_<tag>_<iso>.jsonl."""
    eng: Dict[str, Dict[str, Any]] = {}
    per_iso: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for d in docs:
        eng.setdefault(d.doc_id, {"doc_id": d.doc_id, "text": d.source_text, "sents": d.src_sents, "para_starts": d.para_starts, "domain": d.domain, "bin": d.bin})
        per_iso[d.iso].append({"doc_id": d.doc_id, "iso": d.iso, "ref_kind": d.ref_kind, "ref_system": d.ref_system, "ref_sents": d.ref_sents, "flags": d.flags,
                               "contaminated": d.contaminated})
    n = {"docs": L.write_jsonl(os.path.join(out_dir, f"docs_{tag}.jsonl"), eng.values())}
    for iso, rows in per_iso.items():
        n[iso] = L.write_jsonl(os.path.join(out_dir, f"refs_{tag}_{iso}.jsonl"), rows)
    return n


# ------------------------------------------------------------------------------------------------ CLI
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("human", help="build the human-reference documents (SmolDoc, FLORES+ passages, stitched) and print their statistics")
    h.add_argument("--bench-root", default=DEFAULT_BENCH_ROOT)
    h.add_argument("--out", required=True)
    h.add_argument("--isos", nargs="*", default=None)
    h.add_argument("--tokenizer", default=None)
    h.add_argument("--stitch", nargs="*", type=int, default=[512, 1024, 2048])
    h.add_argument("--clean-only-stitch", action="store_true", help="stitch only documents whose English was NOT in the training windows")
    c = sub.add_parser("coverage", help="sentences per language and bake-off sentence set")
    c.add_argument("--bench-root", default=DEFAULT_BENCH_ROOT)
    c.add_argument("--dev-dir", default=None)
    f = sub.add_parser("fetch-flores-dev", help="download the FLORES+ dev files of our languages (gated repo, read access of the box)")
    f.add_argument("--dest", required=True)
    f.add_argument("--isos", nargs="*", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "coverage":
        cov = refset_coverage(a.bench_root, a.dev_dir)
        print(json.dumps(cov, indent=0))
        return 0
    if a.cmd == "fetch-flores-dev":
        codes = ["eng_Latn"] + sorted({LANGUAGES[i].flores for i in (a.isos or SUPPORTED_RL) if LANGUAGES[canonical_iso(i)].flores})
        got = fetch_flores_dev(codes, a.dest)
        print(f"fetched {len(got)}/{len(codes)} dev files into {a.dest}")
        return 0
    count = L.load_token_counter(a.tokenizer)
    flags = DocFlags(a.bench_root)
    isos = [canonical_iso(i) for i in (a.isos or SUPPORTED_RL)]
    all_docs, stats = [], {}
    for iso in isos:
        sd = smoldoc_docs(iso, a.bench_root, flags, count)
        fd = flores_passage_docs(iso, "devtest", a.bench_root, None, flags, count)
        pool = [d for d in sd if not (a.clean_only_stitch and d.contaminated)]
        st = {t: stitch_documents(pool, t, count) for t in a.stitch} if pool else {}
        all_docs += sd + fd + [d for v in st.values() for d in v]
        stats[iso] = {"smoldoc": len(sd), "smoldoc_clean": sum(1 for d in sd if d.contaminated is False), "flores_devtest_passages": len(fd),
                      **{f"stitched_{t}": len(v) for t, v in st.items()}}
    print(json.dumps(stats, indent=0))
    print(json.dumps(docs_to_files(all_docs, a.out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
