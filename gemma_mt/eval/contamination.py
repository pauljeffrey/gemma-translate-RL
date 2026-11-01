# SPDX-License-Identifier: AGPL-3.0-or-later
"""Benchmark contamination: which benchmark ROWS have their English sentence in the SFT training text.

The Gemma SFT windows (`gemma/windows_sft`, the exact input of `prepare_sft` -> `gemma/packed_fwd`) contain many English sentences of
FLORES+ and MAFAND (found 2026-10-14 for the DEV sets: only 10,043 of 30,418 dev rows had an English sentence absent from the windows,
`bench/bench_dev_clean.jsonl`). This module reproduces that filter for the TEST sets and adds two cheaper-to-ignore-but-useful signals:

    exact   the normalised English sentence of the row equals a normalised sentence of some training window (the rule of the dev
            filter: windows split at sentence ends `(?<=[.!?])\\s+`), OR occurs inside a window as a contiguous token run (catches
            benchmark sentences that contain several sentences and windows whose sentence split differs)
    near    not exact, but >= `near_threshold` (0.5) of the sentence's 8-token n-grams occur somewhere in the training English
    pair    exact AND the row's reference (normalised) occurs in the TARGET side of a window of the same language: the model saw
            the very pair (a stronger form of contamination than the English side alone)

Normalisation = NFKC, case-fold, every run of non-alphanumeric characters (punctuation, whitespace, `_`) -> one space. Rows with fewer than
8 normalised tokens have no n-grams: only the sentence-level `exact` rule applies to them.

The headline "clean" subset removes `exact` rows; `clean_strict` also removes `near` rows (sensitivity check).

Extra benchmarks (D-42). `smol_sent`: as above. `smol_doc`: a document is a long text, so "its English sentence occurs in a window" is replaced by
`exact OR near` (the document occurs as a contiguous run, or >= 50% of its 8-token n-grams occur in the training English: a document of which
half was seen is not a fair test); the `near` flag itself is kept. `gatitos`: the English side is a word or a short phrase (< 8 tokens), for which
only the sentence rule exists (the entry equals a whole sentence of a window); the `pair` flag (and the translation occurs in the target side of a
window of the same language) is the one that says the model has seen the entry.

Flag files: `contamination_<benchmark>.json` = {"meta": {...}, "exact": {"<iso>:<row id>": bool}, "near": {...}, "pair": {...}} for every row of the
benchmark (`<iso>:<id>` is unique across a benchmark: FLORES+/NTREX ids are shared by all languages, MAFAND ids are per language).
"""

from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import re
import sys
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

NGRAM = 8
NEAR_THRESHOLD = 0.5
_NON_ALNUM = re.compile(r"[\W_]+", re.UNICODE)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def normalise(text: str) -> str:
    """NFKC, case-fold, punctuation / whitespace runs -> one space, stripped."""
    return _NON_ALNUM.sub(" ", unicodedata.normalize("NFKC", text or "").casefold()).strip()


def row_key(iso: str, rid: Any) -> str:
    return f"{iso}:{rid}"


def _grams(tokens: Sequence[str], n: int = NGRAM) -> List[int]:
    return [hash(tuple(tokens[i: i + n])) for i in range(len(tokens) - n + 1)]


class BenchIndex:
    """Index of the distinct (normalised) English sentences of the benchmark rows to look for."""

    def __init__(self, rows: Iterable[Dict[str, str]], n: int = NGRAM):
        self.n = n
        self.ids: Dict[str, int] = {}                  # normalised english -> string id
        self.texts: List[str] = []
        self.refs: List[Dict[str, Set[str]]] = []      # string id -> {iso: {normalised references}}
        for r in rows:
            s = normalise(r["english"])
            if not s:
                continue
            i = self.ids.get(s)
            if i is None:
                i = self.ids[s] = len(self.texts)
                self.texts.append(s)
                self.refs.append({})
            for ref in [r.get("reference", "")] + list(r.get("references") or []):        # GATITOS rows have several accepted translations
                self.refs[i].setdefault(r["iso"], set()).add(normalise(ref))
        self.gram_to_strings: Dict[int, List[int]] = defaultdict(list)
        self.first_gram: Dict[int, List[int]] = defaultdict(list)
        self.gram_count: List[int] = []
        for i, s in enumerate(self.texts):
            toks = s.split()
            gs = _grams(toks, n)
            for g in set(gs):
                self.gram_to_strings[g].append(i)
            if gs:
                self.first_gram[gs[0]].append(i)
            self.gram_count.append(len(set(gs)))


@dataclass
class ScanResult:
    exact: Set[int] = field(default_factory=set)               # string ids found as a sentence / contiguous run
    sentence: Set[int] = field(default_factory=set)            # ... as a whole normalised SENTENCE of a window (the rule of the dev filter)
    grams: Set[int] = field(default_factory=set)               # benchmark n-gram hashes seen in the training English
    pair: Set[Tuple[int, str]] = field(default_factory=set)    # (string id, iso) whose reference is in a window's target too
    windows: int = 0
    unique_english: int = 0

    def merge(self, other: "ScanResult") -> None:
        self.exact |= other.exact
        self.sentence |= other.sentence
        self.grams |= other.grams
        self.pair |= other.pair
        self.windows += other.windows
        self.unique_english += other.unique_english


def scan_windows(index: BenchIndex, windows: Iterable[Tuple[str, str, str]]) -> ScanResult:
    """windows: (english, target, iso) triples of the training text. Equal English texts are analysed once (the pair check still
    runs for every window)."""
    res = ScanResult()
    cache: Dict[str, Tuple[int, ...]] = {}
    n = index.n
    for en, target, iso in windows:
        res.windows += 1
        hits = cache.get(en)
        if hits is None:
            res.unique_english += 1
            found: Set[int] = set()
            sent_hits: Set[int] = set()
            toks: List[str] = []
            for sent in _SENT_SPLIT.split(en):
                ns = normalise(sent)
                if not ns:
                    continue
                i = index.ids.get(ns)
                if i is not None:
                    found.add(i)
                    sent_hits.add(i)
                toks.extend(ns.split())
            if len(toks) >= n:
                gs = _grams(toks, n)
                joined = None
                for pos, g in enumerate(gs):
                    cand = index.gram_to_strings.get(g)
                    if cand is None:
                        continue
                    res.grams.add(g)
                    fg = index.first_gram.get(g)
                    if fg:
                        if joined is None:
                            joined = " " + " ".join(toks) + " "
                        for i in fg:
                            if i not in found and (" " + index.texts[i] + " ") in joined:
                                found.add(i)
            hits = cache[en] = tuple(sorted(found))
            res.exact.update(found)
            res.sentence.update(sent_hits)
        if hits and target:
            tnorm = " " + " ".join(normalise(s) for s in _SENT_SPLIT.split(target)) + " "
            for i in hits:
                for ref in index.refs[i].get(iso, ()):
                    if ref and (" " + ref + " ") in tnorm:
                        res.pair.add((i, iso))
    return res


# ------------------------------------------------------------------------------------------------ flags
def flags_for_rows(rows: Sequence[Dict[str, str]], index: BenchIndex, scan: ScanResult,
                   near_threshold: float = NEAR_THRESHOLD) -> Dict[str, Dict[str, bool]]:
    """{"exact": {key: bool}, "near": {...}, "pair": {...}} for every row (key = "<iso>:<id>")."""
    overlap: Dict[int, float] = {}
    for i, s in enumerate(index.texts):
        toks = s.split()
        gs = set(_grams(toks, index.n))
        overlap[i] = (sum(1 for g in gs if g in scan.grams) / len(gs)) if gs else 0.0
    out: Dict[str, Dict[str, bool]] = {"exact": {}, "sentence": {}, "near": {}, "pair": {}}
    for r in rows:
        k = row_key(r["iso"], r["id"])
        i = index.ids.get(normalise(r["english"]))
        ex = i is not None and i in scan.exact
        out["exact"][k] = bool(ex)
        out["sentence"][k] = bool(i is not None and i in scan.sentence)
        out["near"][k] = bool(i is not None and not ex and overlap[i] >= near_threshold)
        out["pair"][k] = bool(i is not None and (i, r["iso"]) in scan.pair)
    return out


def doc_level_flags(flags: Dict[str, Dict[str, bool]]) -> Dict[str, Dict[str, bool]]:
    """Documents: contaminated = exact OR near (see the module docstring). The sentence / near / pair flags stay as computed."""
    out = {k: dict(v) for k, v in flags.items()}
    for key in out["exact"]:
        out["exact"][key] = bool(flags["exact"][key] or flags["near"][key])
    return out


def summarise(rows: Sequence[Dict[str, str]], flags: Dict[str, Dict[str, bool]]) -> Dict[str, Any]:
    by: Dict[str, Dict[str, int]] = {}
    for r in rows:
        k = row_key(r["iso"], r["id"])
        d = by.setdefault(r["iso"], {"n": 0, "exact": 0, "sentence": 0, "near": 0, "pair": 0})
        d["n"] += 1
        for f in ("exact", "sentence", "near", "pair"):
            d[f] += int(flags[f][k])
    tot = {f: sum(d[f] for d in by.values()) for f in ("n", "exact", "sentence", "near", "pair")}
    return {"total": tot, "by_language": by}


def write_flag_file(path: str, benchmark: str, rows: Sequence[Dict[str, str]], flags: Dict[str, Dict[str, bool]],
                    extra_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    summ = summarise(rows, flags)
    meta = {"benchmark": benchmark, "ngram": NGRAM, "near_threshold": NEAR_THRESHOLD, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "normalisation": "NFKC, casefold, [\\W_]+ -> ' '", "key": "<iso>:<row id>", **(extra_meta or {}), **summ}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, **flags}, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, path)
    return meta


def load_flag_file(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def apply_flags(rows: Sequence[Dict[str, Any]], flag_files: Dict[str, str]) -> Dict[str, int]:
    """Set `contaminated` (exact), `contaminated_sentence` (dev-filter rule), `near_dup`, `pair_seen` on every row whose benchmark has a flag file (`flag_files`: benchmark ->
    path). A benchmark without a file, or a row missing from its file, gets None (never silently "clean"). Returns {benchmark: n rows
    without a flag}."""
    loaded: Dict[str, Dict[str, Any]] = {}
    missing: Dict[str, int] = defaultdict(int)
    for r in rows:
        b = r["benchmark"]
        if b not in flag_files:
            r["contaminated"] = r["contaminated_sentence"] = r["near_dup"] = r["pair_seen"] = None
            missing[b] += 1
            continue
        if b not in loaded:
            loaded[b] = load_flag_file(flag_files[b])
        d = loaded[b]
        k = row_key(r["iso"], r["id"])
        if k not in d["exact"]:
            r["contaminated"] = r["contaminated_sentence"] = r["near_dup"] = r["pair_seen"] = None
            missing[b] += 1
            continue
        r["contaminated"] = bool(d["exact"][k])
        r["contaminated_sentence"] = bool(d.get("sentence", d["exact"]).get(k, d["exact"][k]))
        r["near_dup"] = bool(d["near"].get(k, False))
        r["pair_seen"] = bool(d["pair"].get(k, False))
    return dict(missing)


def default_flag_files(bench_dir: str, benchmarks: Sequence[str]) -> Dict[str, str]:
    out = {}
    for b in benchmarks:
        p = os.path.join(bench_dir, f"contamination_{b}.json")
        if os.path.isfile(p):
            out[b] = p
    return out


# ------------------------------------------------------------------------------------------------ scanning the parquet windows
_INDEX: Optional[BenchIndex] = None


def _iter_parquet_windows(path: str) -> Iterator[Tuple[str, str, str]]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=100_000, columns=["english", "target", "iso"]):
        d = batch.to_pydict()
        yield from zip(d["english"], d["target"], d["iso"])


def _scan_file(path: str) -> Tuple[str, ScanResult, float]:
    t0 = time.time()
    return path, scan_windows(_INDEX, _iter_parquet_windows(path)), time.time() - t0


def scan_parquet_dir(index: BenchIndex, windows_dir: str, processes: int = 8, log=print) -> ScanResult:
    """Scan every `*.parquet` of `windows_dir` (columns english, target, iso) in `processes` forked workers."""
    global _INDEX
    files = sorted(glob.glob(os.path.join(windows_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet files in {windows_dir}")
    _INDEX = index
    total = ScanResult()
    ctx = mp.get_context("fork")        # children inherit the index and its hash seed (n-gram hashes are only compared inside this run)
    with ctx.Pool(max(1, min(processes, len(files)))) as pool:
        for k, (path, res, dt) in enumerate(pool.imap_unordered(_scan_file, files), 1):
            total.merge(res)
            log(f"[contam] {k}/{len(files)} {os.path.basename(path)}: {res.windows} windows ({res.unique_english} distinct english) "
                f"{dt:.0f}s | exact so far {len(total.exact)} / {len(index.texts)}")
    return total


def main(argv: Optional[List[str]] = None) -> int:
    from gemma_mt.eval import benchmarks as B

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windows", required=True, help="directory of the SFT training windows (parquet: english, target, iso)")
    ap.add_argument("--bench-root", required=True, help="directory written by gemma_mt.eval.fetch_data (flores_plus/, ntrex/, mafand/)")
    ap.add_argument("--benchmarks", nargs="*", default=list(B.BENCHMARKS), help="default flores_plus ntrex mafand; also smol_sent smol_doc gatitos (or smol)")
    ap.add_argument("--out-dir", required=True, help="where contamination_<benchmark>.json are written")
    ap.add_argument("--dev-file", default=None, help="the dev benchmark jsonl (bench_dev.jsonl): scanned in the same pass, written as "
                                                     "contamination_dev.json (validation of this scanner against bench_dev_clean.jsonl)")
    ap.add_argument("--dev-clean-file", default=None, help="bench_dev_clean.jsonl: compared with the dev rows this scanner leaves clean")
    ap.add_argument("--processes", type=int, default=8)
    a = ap.parse_args(argv)

    sets: Dict[str, List[Dict[str, str]]] = {}
    a.benchmarks = B.parse_benchmark_names(a.benchmarks)
    for name in a.benchmarks:
        sets[name] = B.load_benchmark(name, path=os.path.join(a.bench_root, name))
        print(f"[contam] {name}: {len(sets[name])} rows", flush=True)
    if a.dev_file:
        sets["dev"] = B.load_normalised_file(a.dev_file, "dev")
        print(f"[contam] dev: {len(sets['dev'])} rows", flush=True)
    all_rows = [r for rows in sets.values() for r in rows]
    index = BenchIndex(all_rows)
    print(f"[contam] {len(index.texts)} distinct english sentences, {len(index.gram_to_strings)} distinct {NGRAM}-grams", flush=True)
    t0 = time.time()
    scan = scan_parquet_dir(index, a.windows, a.processes, log=lambda m: print(m, flush=True))
    print(f"[contam] scanned {scan.windows} windows in {time.time() - t0:.0f}s", flush=True)
    for name, rows in sets.items():
        flags = flags_for_rows(rows, index, scan)
        extra = {"training_text": a.windows, "training_windows": scan.windows}
        if name in B.DOC_LEVEL_BENCHMARKS:
            flags = doc_level_flags(flags)
            extra["rule"] = "document: exact OR near (>= 50% of the 8-token n-grams in the training English)"
        elif name in B.WORD_LEVEL_BENCHMARKS:
            extra["rule"] = "word level: sentence rule only (entry equals a whole window sentence); `pair` = entry and translation seen together"
        meta = write_flag_file(os.path.join(a.out_dir, f"contamination_{name}.json"), name, rows, flags, extra)
        t = meta["total"]
        print(f"[contam] {name}: {t['exact']}/{t['n']} exact ({100 * t['exact'] / max(t['n'], 1):.1f}%), sentence-rule {t['sentence']}, "
              f"near {t['near']}, pair {t['pair']}", flush=True)
        if name == "dev" and a.dev_clean_file:
            ref = {(r["iso"], str(r["id"])) for r in B.load_normalised_file(a.dev_clean_file, "dev")}
            for rule in ("sentence", "exact"):
                mine = {(r["iso"], r["id"]) for r in rows if not flags[rule][row_key(r["iso"], r["id"])]}
                print(f"[contam] dev check, rule {rule!r}: scanner clean {len(mine)}, bench_dev_clean {len(ref)}, only in mine {len(mine - ref)}, "
                      f"only in bench_dev_clean {len(ref - mine)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
