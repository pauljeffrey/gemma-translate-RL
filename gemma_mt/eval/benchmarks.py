# SPDX-License-Identifier: AGPL-3.0-or-later
"""Benchmark adapters. Every benchmark is turned into ONE normalised schema:

    {"english": str, "reference": str, "iso": str, "benchmark": str, "id": str}

Supported (English -> African language only; the source is always English):
  flores_plus  FLORES+ (openlanguagedata/flores_plus, GATED: accept the terms on the Hub and set HF_TOKEN).
               One config per language (e.g. "yor_Latn"), splits dev / devtest, rows parallel by `id`; English is the
               "eng_Latn" config.
  ntrex        NTREX-128: plain text files `newstest2019-src.eng.txt` and `newstest2019-ref.<code>.txt` (one sentence
               per line, parallel). Source repo: https://github.com/MicrosoftTranslator/NTREX (use `fetch_ntrex`).
  mafand       MAFAND-MT news (masakhane/mafand): configs "en-<code>", splits train/validation/test, a
               `translation` dict {"en": ..., "<code>": ...}. Only the English-source (en-*) pairs are used; the
               fr-* pairs are skipped.

Each loader reads from LOCAL files when `path` is given and from the Hub otherwise (network is only touched at
runtime, on the machine that runs the evaluation). The per-language codes of every benchmark live in
gemma_mt/languages.py (`flores`, `ntrex`, `mafand`); a language without a code for a benchmark simply has no rows
there and is reported as "no data", never silently filled.

Local layouts understood:
  flores_plus  <dir>/<code>/<split>.{parquet,jsonl}  |  <dir>/<split>/<code>.{parquet,jsonl}  |  <dir>/<code>.<split>.{...}
               rows: id, text            (+ the English config under code eng_Latn)
  ntrex        <dir>/newstest2019-src.eng.txt, <dir>/newstest2019-ref.<code>.txt   (also under <dir>/NTREX-128/)
  mafand       <dir>/en-<code>/<split>/*.{parquet,jsonl,json}  |  <dir>/en-<code>/<split>.{jsonl,json}

Extra benchmarks (D-42, not part of the default set; select them with `--benchmarks smol_sent smol_doc gatitos`). The first three come from the
Hub dataset `google/smol` (CC-BY-4.0; Caswell et al., "SMOL: Professionally translated parallel data for 115 under-represented
languages", arXiv 2502.12301) as raw JSON-lines files `<dir>/en_<code>.jsonl`, English -> X only (the `X_en` files are the same pairs reversed):
  smol_sent    SmolSent: 863 English sentences (professional translations; some languages fewer). Row = {sl, tl, id, src, trg}.
  smol_doc     SmolDoc: documents of 1-41 sentences (median 12, about 190 English words), `srcs` / `trgs` lists of equal length and a `sep` list of
               the whitespace between sentences. One row = one document: English = srcs joined with `sep`, reference = trgs joined with the same `sep`.
  gatitos      GATITOS: about 4,000 English words / short phrases, each with ONE OR MORE accepted translations (`trgs`). Row = {src, trgs}. Word level:
               scored with exact match (any accepted translation) and chrF++ against the best accepted translation (row["references"]).
  ibom_mt, glot_story, salt   normalised by fetch_data to `<dir>/en_<code>.jsonl` with rows {"id", "src", "trg"} (+ "license" for glot_story).
  flores_moz   FLORES-MOZ in the FLORES+ layout: `<dir>/devtest/{eng_Latn,<code>}.jsonl` (rows id, text).
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

from gemma_mt.data.io import iter_rows
from gemma_mt.languages import ALL_ISOS, LANGUAGES, canonical_iso

BENCHMARKS = ("flores_plus", "ntrex", "mafand")                    # the default set of the post-SFT evaluation (unchanged)
SMOL_BENCHMARKS = ("smol_sent", "smol_doc", "gatitos")            # D-42: opt-in (Google SMOL / GATITOS)
OTHER_BENCHMARKS = ("ibom_mt", "flores_moz", "glot_story", "salt")  # D-42: opt-in, small public sets for languages without any test set
EXTRA_BENCHMARKS = SMOL_BENCHMARKS + OTHER_BENCHMARKS
ALL_BENCHMARKS = BENCHMARKS + EXTRA_BENCHMARKS
PAIR_FILE_BENCHMARKS = ("ibom_mt", "glot_story", "salt")           # normalised by fetch_data to <dir>/en_<code>.jsonl = {"id", "src", "trg", ...}
IBOM_REPO, IBOM_REVISION = "howard-nlp/ibom-mt", "90d73874e275a27857e6dd2fce6ed9197ff2afa4"                           # CC-BY-NC-4.0
FLORES_MOZ_REPO, FLORES_MOZ_REVISION = "MOZNLP/FLORES_MOZ", "751dd12a6bb2f2ebbc52c3e3a2653c2356d63589"                # CC-BY-SA-4.0
GLOT_STORY_REPO, GLOT_STORY_REVISION = "cis-lmu/GlotStoryBook", "ad913a24f4a912cc0cd49a5f69e2055eba483d1f"            # per-row CC-BY / CC-BY-NC / CC BY-NC-SA
SALT_REPO, SALT_REVISION = "Sunbird/salt", "a0ffb52b49cdf7bf69149e5d418655313ffc41f4"                                 # CC-BY-SA-4.0, gated (auto-accept)
LICENCES = {"smol_sent": "CC-BY-4.0", "smol_doc": "CC-BY-4.0", "gatitos": "CC-BY-4.0", "ibom_mt": "CC-BY-NC-4.0", "flores_moz": "CC-BY-SA-4.0",
            "glot_story": "mixed per row: CC-BY / CC-BY-NC / CC BY-NC-SA", "salt": "CC-BY-SA-4.0"}
WORD_LEVEL_BENCHMARKS = ("gatitos",)                              # scored with exact match; no spBLEU / BLEU / AfriCOMET
DOC_LEVEL_BENCHMARKS = ("smol_doc",)                              # documents: long generation budget, no AfriCOMET (its encoder truncates at 512 tokens)
SMOL_REPO = "google/smol"
SMOL_REVISION = "fdaff3a1a019f89fa30a562a85d7c1d3e9150444"       # pinned: the card changes (April 2026 update); lastModified 2026-08-06
SMOL_SUBDIR = {"smol_sent": "smolsent", "smol_doc": "smoldoc", "gatitos": "gatitos"}
FLORES_REPO = "openlanguagedata/flores_plus"
MAFAND_REPO = "masakhane/mafand"
NTREX_GIT = "https://github.com/MicrosoftTranslator/NTREX"


def parse_benchmark_names(names: Sequence[str]) -> List[str]:
    """`--benchmarks a b` / `--benchmarks a,b` / `smol` (= smol_sent smol_doc gatitos) / `other` (= ibom_mt flores_moz glot_story salt) / `extra` (both) / `all`;
    validated, order kept, no duplicates."""
    out: List[str] = []
    for tok in names:
        for n in str(tok).replace(";", ",").split(","):
            n = n.strip()
            if not n:
                continue
            group = {"smol": SMOL_BENCHMARKS, "other": OTHER_BENCHMARKS, "extra": EXTRA_BENCHMARKS, "all": ALL_BENCHMARKS}.get(n, (n,))
            for g in group:
                if g not in ALL_BENCHMARKS:
                    raise ValueError(f"unknown benchmark {g!r}; choose from {ALL_BENCHMARKS} (or smol / other / extra / all)")
                if g not in out:
                    out.append(g)
    return out


def code_for(benchmark: str, iso: str) -> Optional[str]:
    lang = LANGUAGES[canonical_iso(iso)]
    return {"flores_plus": lang.flores, "ntrex": lang.ntrex, "mafand": lang.mafand,
            "smol_sent": lang.smol_sent, "smol_doc": lang.smol_doc, "gatitos": lang.gatitos, "ibom_mt": lang.ibom_mt,
            "flores_moz": lang.flores_moz, "glot_story": lang.glot_story, "salt": lang.salt}[benchmark]


def languages_with_data(benchmark: str) -> List[str]:
    return [i for i in ALL_ISOS if code_for(benchmark, i)]


def coverage_table(benchmarks: Sequence[str] = BENCHMARKS) -> str:
    rows = ["| tag | name | " + " | ".join(benchmarks) + " |", "|---|---|" + "---|" * len(benchmarks)]
    for i in ALL_ISOS:
        l = LANGUAGES[i]
        rows.append(f"| `{i}` | {l.name} | " + " | ".join(code_for(b, i) or "-" for b in benchmarks) + " |")
    return "\n".join(rows)


def _row(en: str, ref: str, iso: str, bench: str, rid: Any) -> Dict[str, str]:
    return {"english": en.strip(), "reference": ref.strip(), "iso": iso, "benchmark": bench, "id": str(rid)}


def _files_matching(base: str, stems: Sequence[str]) -> Optional[str]:
    for stem in stems:
        for ext in (".parquet", ".jsonl", ".json", ".jsonl.gz"):
            p = stem + ext
            if os.path.exists(p):
                return p
        hits = sorted(glob.glob(stem + "/*.parquet") + glob.glob(stem + "/*.jsonl") + glob.glob(stem + "/*.json"))
        if hits:
            return hits[0] if len(hits) == 1 else stem
    return None


# ------------------------------------------------------------------------------------------------ FLORES+
def _flores_local(path: str, code: str, split: str) -> Optional[List[Dict]]:
    cands = [os.path.join(path, code, split), os.path.join(path, split, code), os.path.join(path, f"{code}.{split}")]
    hit = _files_matching(path, cands)
    if hit is None:
        return None
    return list(iter_rows(hit, columns=("id", "text")))


def _flores_hub(code: str, split: str, token: Optional[str]) -> List[Dict]:  # pragma: no cover - network
    from datasets import load_dataset

    ds = load_dataset(FLORES_REPO, code, split=split, token=token)
    return [dict(r) for r in ds]


def load_flores_plus(isos: Optional[Sequence[str]] = None, split: str = "devtest", path: Optional[str] = None,
                     token: Optional[str] = None, limit: int = 0, code_attr: str = "flores_plus") -> List[Dict[str, str]]:
    token = token or os.environ.get("HF_TOKEN")
    get = (lambda code: _flores_local(path, code, split)) if path else (lambda code: _flores_hub(code, split, token))
    en_rows = get("eng_Latn")
    if not en_rows:
        raise FileNotFoundError("FLORES+ English (eng_Latn) rows not found")
    en = {str(r["id"]): r["text"] for r in en_rows}
    out: List[Dict[str, str]] = []
    for iso in (isos or languages_with_data(code_attr)):
        code = code_for(code_attr, iso)
        if not code:
            continue
        rows = get(code)
        if not rows:
            continue
        n = 0
        for r in rows:
            rid = str(r["id"])
            if rid in en and r.get("text"):
                out.append(_row(en[rid], r["text"], canonical_iso(iso), "flores_plus", rid))
                n += 1
                if limit and n >= limit:
                    break
    return out


# ------------------------------------------------------------------------------------------------ NTREX-128
def _ntrex_dir(path: str) -> str:
    for d in (path, os.path.join(path, "NTREX-128"), os.path.join(path, "NTREX", "NTREX-128")):
        if os.path.exists(os.path.join(d, "newstest2019-src.eng.txt")):
            return d
    raise FileNotFoundError(f"newstest2019-src.eng.txt not found under {path}")


def fetch_ntrex(dest: str) -> str:  # pragma: no cover - network, explicit call only
    """git clone the NTREX repository (runtime, GPU box). Returns the NTREX-128 directory."""
    subprocess.run(["git", "clone", "--depth", "1", NTREX_GIT, dest], check=True)
    return os.path.join(dest, "NTREX-128")


def load_ntrex(isos: Optional[Sequence[str]] = None, path: str = "", limit: int = 0) -> List[Dict[str, str]]:
    d = _ntrex_dir(path)
    with open(os.path.join(d, "newstest2019-src.eng.txt"), encoding="utf-8") as f:
        src = f.read().split("\n")
    if src and src[-1] == "":
        src.pop()
    out: List[Dict[str, str]] = []
    for iso in (isos or languages_with_data("ntrex")):
        code = code_for("ntrex", iso)
        p = os.path.join(d, f"newstest2019-ref.{code}.txt") if code else None
        if not p or not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            ref = f.read().split("\n")
        if ref and ref[-1] == "":
            ref.pop()
        if len(ref) != len(src):
            raise ValueError(f"NTREX {code}: {len(ref)} reference lines vs {len(src)} source lines")
        n = 0
        for i, (s, r) in enumerate(zip(src, ref)):
            if s.strip() and r.strip():
                out.append(_row(s, r, canonical_iso(iso), "ntrex", i))
                n += 1
                if limit and n >= limit:
                    break
    return out


# ------------------------------------------------------------------------------------------------ MAFAND-MT
def _mafand_rows(rows: Iterable[Dict], code: str) -> Iterator[tuple]:
    for r in rows:
        tr = r.get("translation")
        if tr is None and "en" in r and code in r:
            tr = r
        if not tr or "en" not in tr or code not in tr:
            continue
        yield tr["en"], tr[code]


def _mafand_local(path: str, code: str, split: str) -> Optional[List[Dict]]:
    cfg = os.path.join(path, f"en-{code}")
    hit = _files_matching(path, [os.path.join(cfg, split)])
    if hit is None:
        return None
    return list(iter_rows(hit, columns=("translation",)))


def _mafand_hub(code: str, split: str, token: Optional[str]) -> List[Dict]:  # pragma: no cover - network
    """Uses the Hub's auto-converted parquet branch (the original repo is a loading script)."""
    from huggingface_hub import snapshot_download

    d = snapshot_download(MAFAND_REPO, repo_type="dataset", revision="refs/convert/parquet",
                          allow_patterns=[f"en-{code}/{split}/*.parquet"], token=token)
    return _mafand_local(d, code, split) or []


def load_mafand(isos: Optional[Sequence[str]] = None, split: str = "test", path: Optional[str] = None,
                token: Optional[str] = None, limit: int = 0) -> List[Dict[str, str]]:
    token = token or os.environ.get("HF_TOKEN")
    out: List[Dict[str, str]] = []
    for iso in (isos or languages_with_data("mafand")):
        code = code_for("mafand", iso)
        if not code:
            continue
        rows = _mafand_local(path, code, split) if path else _mafand_hub(code, split, token)
        if not rows:
            continue
        n = 0
        for i, (en, tgt) in enumerate(_mafand_rows(rows, code)):
            if en and en.strip() and tgt and tgt.strip():
                out.append(_row(en, tgt, canonical_iso(iso), "mafand", i))
                n += 1
                if limit and n >= limit:
                    break
    return out


# ------------------------------------------------------------------------------------------------ SMOL / GATITOS (google/smol)
def _smol_file(path: str, code: str) -> Optional[str]:
    p = os.path.join(path, f"en_{code}.jsonl")
    return p if os.path.isfile(p) and os.path.getsize(p) > 0 else None


def _smol_json_lines(p: str) -> Iterator[Dict[str, Any]]:
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def join_with_sep(parts: Sequence[str], seps: Optional[Sequence[str]]) -> str:
    """SmolDoc sentences joined with the document's own separators (` `, newline, blank line, list bullets); the `sep` list has one entry
    per sentence (the last one is empty). Without `sep` (older revisions) a single space is used. The result is stripped."""
    if not seps or len(seps) != len(parts):
        seps = [" "] * len(parts)
    return "".join(f"{p}{s}" for p, s in zip(parts, seps)).strip()


def load_smol_sent(isos: Optional[Sequence[str]] = None, path: str = "", limit: int = 0) -> List[Dict[str, str]]:
    """SmolSent English -> X rows (`<path>/en_<code>.jsonl`)."""
    out: List[Dict[str, str]] = []
    for iso in (isos or languages_with_data("smol_sent")):
        code = code_for("smol_sent", iso)
        p = _smol_file(path, code) if code else None
        if not p:
            continue
        n = 0
        for r in _smol_json_lines(p):
            if r.get("sl", "en") != "en" or not (r.get("src") or "").strip() or not (r.get("trg") or "").strip():
                continue
            out.append(_row(r["src"], r["trg"], canonical_iso(iso), "smol_sent", r.get("id", n)))
            n += 1
            if limit and n >= limit:
                break
    return out


def load_smol_doc(isos: Optional[Sequence[str]] = None, path: str = "", limit: int = 0) -> List[Dict[str, Any]]:
    """SmolDoc English -> X rows: one row per document; `n_sentences` and `factuality` (of the English text) are kept for the report."""
    out: List[Dict[str, Any]] = []
    for iso in (isos or languages_with_data("smol_doc")):
        code = code_for("smol_doc", iso)
        p = _smol_file(path, code) if code else None
        if not p:
            continue
        n = 0
        for r in _smol_json_lines(p):
            srcs, trgs = r.get("srcs") or [], r.get("trgs") or []
            if r.get("sl", "en") != "en" or not srcs or len(srcs) != len(trgs):
                continue
            en, ref = join_with_sep(srcs, r.get("sep")), join_with_sep(trgs, r.get("sep"))
            if not en or not ref:
                continue
            row = _row(en, ref, canonical_iso(iso), "smol_doc", r.get("id", n))
            row["n_sentences"] = len(srcs)
            row["factuality"] = str(r.get("factuality", ""))
            out.append(row)
            n += 1
            if limit and n >= limit:
                break
    return out


def load_gatitos(isos: Optional[Sequence[str]] = None, path: str = "", limit: int = 0) -> List[Dict[str, Any]]:
    """GATITOS English -> X rows: `reference` = the first accepted translation, `references` = all of them (exact match and chrF++ use the list).
    Entries that occur twice (same English string) are merged into one row with the union of their translations (order kept)."""
    out: List[Dict[str, Any]] = []
    for iso in (isos or languages_with_data("gatitos")):
        code = code_for("gatitos", iso)
        p = _smol_file(path, code) if code else None
        if not p:
            continue
        by_src: Dict[str, List[str]] = {}
        for r in _smol_json_lines(p):
            src = (r.get("src") or "").strip()
            if r.get("sl", "en") != "en" or not src:
                continue
            tr = by_src.setdefault(src, [])
            for t in r.get("trgs") or []:
                t = (t or "").strip()
                if t and t not in tr:
                    tr.append(t)
        n = 0
        for i, (src, trs) in enumerate(by_src.items()):
            if not trs:
                continue
            row = _row(src, trs[0], canonical_iso(iso), "gatitos", i)
            row["references"] = trs
            out.append(row)
            n += 1
            if limit and n >= limit:
                break
    return out


def load_pair_files(benchmark: str, isos: Optional[Sequence[str]] = None, path: str = "", limit: int = 0) -> List[Dict[str, Any]]:
    """ibom_mt / glot_story / salt: `<path>/en_<code>.jsonl`, rows {"id", "src", "trg", ["license"]} (written by fetch_data)."""
    out: List[Dict[str, Any]] = []
    for iso in (isos or languages_with_data(benchmark)):
        code = code_for(benchmark, iso)
        p = _smol_file(path, code) if code else None
        if not p:
            continue
        n = 0
        for r in _smol_json_lines(p):
            if not (r.get("src") or "").strip() or not (r.get("trg") or "").strip():
                continue
            row = _row(r["src"], r["trg"], canonical_iso(iso), benchmark, r.get("id", n))
            if r.get("license"):
                row["license"] = r["license"]
            out.append(row)
            n += 1
            if limit and n >= limit:
                break
    return out


def load_flores_moz(isos: Optional[Sequence[str]] = None, path: str = "", limit: int = 0) -> List[Dict[str, Any]]:
    """FLORES-MOZ devtest (FLORES+ layout, English = eng_Latn)."""
    rows = load_flores_plus([i for i in (isos or languages_with_data("flores_moz"))], "devtest", path, None, limit, code_attr="flores_moz")
    for r in rows:
        r["benchmark"] = "flores_moz"
    return rows


# ------------------------------------------------------------------------------------------------ front door
def load_benchmark(name: str, *, path: Optional[str] = None, isos: Optional[Sequence[str]] = None,
                   split: Optional[str] = None, limit: int = 0, token: Optional[str] = None) -> List[Dict[str, str]]:
    """Rows of one benchmark in the normalised schema. `split` defaults: flores_plus devtest, mafand test."""
    if name == "flores_plus":
        return load_flores_plus(isos, split or "devtest", path, token, limit)
    if name == "ntrex":
        if not path:
            raise ValueError("ntrex needs --path to a local NTREX checkout (use fetch_ntrex to clone it)")
        return load_ntrex(isos, path, limit)
    if name == "mafand":
        return load_mafand(isos, split or "test", path, token, limit)
    if name == "flores_moz":
        if not path:
            raise ValueError("flores_moz needs --path to the directory written by gemma_mt.eval.fetch_data")
        return load_flores_moz(isos, path, limit)
    if name in PAIR_FILE_BENCHMARKS:
        if not path:
            raise ValueError(f"{name} needs --path to the directory written by gemma_mt.eval.fetch_data")
        return load_pair_files(name, isos, path, limit)
    if name in SMOL_BENCHMARKS:
        if not path:
            raise ValueError(f"{name} needs --path to the directory written by gemma_mt.eval.fetch_data (en_<code>.jsonl files of google/smol)")
        return {"smol_sent": load_smol_sent, "smol_doc": load_smol_doc, "gatitos": load_gatitos}[name](isos, path, limit)
    raise ValueError(f"unknown benchmark {name!r}; choose from {ALL_BENCHMARKS}")


def load_normalised_file(path: str, benchmark: str = "custom") -> List[Dict[str, str]]:
    """Any parquet/jsonl that already has english, reference, iso."""
    out = []
    for r in iter_rows(path, columns=("english", "reference", "iso", "id", "benchmark")):
        iso = canonical_iso(r.get("iso"))
        if iso is None or not r.get("english") or not r.get("reference"):
            continue
        out.append(_row(r["english"], r["reference"], iso, str(r.get("benchmark") or benchmark), r.get("id", len(out))))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Show which of our languages each benchmark covers / count local rows.")
    ap.add_argument("--list", action="store_true", help="print the coverage table (no data needed)")
    ap.add_argument("--benchmark", choices=ALL_BENCHMARKS)
    ap.add_argument("--extra", action="store_true", help="with --list: also print the smol_sent / smol_doc / gatitos codes")
    ap.add_argument("--path")
    ap.add_argument("--split")
    a = ap.parse_args(argv)
    if a.list or not a.benchmark:
        print(coverage_table(ALL_BENCHMARKS if a.extra else BENCHMARKS))
        return 0
    rows = load_benchmark(a.benchmark, path=a.path, split=a.split)
    from collections import Counter

    for iso, n in sorted(Counter(r["iso"] for r in rows).items()):
        print(f"{iso}\t{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
