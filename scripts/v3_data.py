# SPDX-License-Identifier: AGPL-3.0-or-later
"""Human-style reference sets per language for the anchors v3 (D-68): religious / pivot-style text (the only text some languages have) and out-of-domain human text (to measure the domain shift).

Sources (all read-only; paths are the box layout of 2026-10-22, override with the arguments):
  pivot_held   R3 English-pivot corpus, `held` rows (300-600 sentences per language, mostly Bible / JW)           /workspace/jobs/b3/corpus/<iso>.parquet
  r15_val      R15 probe tests, set `val` (JW multi-sentence windows, never trained on)                           /workspace/jobs/r15/probe/tests/<iso>.jsonl
  r15_seen     R15 probe tests, set `seen`
  r15_<set>    R15 probe tests, other sets: smol_sent, glot_story, wiki, salt, ibom_mt, flores_plus (out of domain)
  hu_<set>     scripts' human_units.jsonl sets (FLORES+ dev, SmolSent, GlotStoryBook, SALT, IbomMT, NTREX ...)
  r10_<ds>     R10 cleaned pools: finetranslations (web text; the English side is machine-made, the target is human-written), opus wikimedia / Tatoeba, opus bible-uedin, salt, ibom
A unit is {src, ref, set, dom}; `dom` = "rel" (religious / JW / Bible style) or "ood" (everything else).
"""
from __future__ import annotations

import glob
import json
import os
import zlib
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

REL_SETS = {"pivot_held", "r15_val", "r15_seen", "r10_opus_bible-uedin"}
SKIP_SETS = {"smol_doc"}


def _fold(iso: str, src: str) -> int:
    return zlib.crc32((iso + "|" + " ".join(src.split()).casefold()).encode("utf-8")) & 1


def _ok(src: str, ref: str) -> bool:
    return bool(src and ref and src.strip() and ref.strip() and len(src.split()) >= 3)


def load_sets(iso: str, human: Optional[str] = "/workspace/jobs/b2/human_units.jsonl", corpus: str = "/workspace/jobs/b3/corpus", r15: str = "/workspace/jobs/r15/probe/tests",
              r10: str = "/workspace/jobs/r10/clean", max_per_set: int = 1500) -> Dict[str, List[Dict[str, str]]]:
    """{set name -> units} for one language; every unit deduplicated on (src, ref)."""
    out: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    seen = set()

    def add(name: str, src: str, ref: str) -> None:
        k = (" ".join(src.split()).casefold(), " ".join(ref.split()).casefold())
        if _ok(src, ref) and k not in seen and len(out[name]) < max_per_set:
            seen.add(k)
            out[name].append({"src": src.strip(), "ref": ref.strip(), "set": name, "dom": "rel" if name in REL_SETS else "ood"})

    fn = os.path.join(corpus, iso + ".parquet")
    if os.path.exists(fn):
        import pyarrow.parquet as pq

        t = pq.read_table(fn, columns=["en", "x", "held"]).to_pydict()
        for en, x, h in zip(t["en"], t["x"], t["held"]):
            if h:
                add("pivot_held", en or "", x or "")
    f = os.path.join(r15, iso + ".jsonl")
    if os.path.exists(f):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            s = r.get("set", "?")
            if s == "gold":
                continue
            add("r15_" + s, r.get("english", ""), r.get("reference") or r.get("target") or "")
    if human and os.path.exists(human):
        for l in open(human, encoding="utf-8"):
            r = json.loads(l)
            if r["iso"] == iso and r["set"] not in SKIP_SETS and not r["set"].endswith("_doc.jsonl"):
                add("hu_" + r["set"], r["src"], r["ref"])
    for fp in glob.glob(os.path.join(r10, f"*__{iso}.jsonl")):
        ds = os.path.basename(fp)[: -len(f"__{iso}.jsonl")]
        if ds.startswith(("salt_train", "afridocmt", "ibom_train")):       # big or duplicated pools: dev / validation style sets only
            continue
        for l in open(fp, encoding="utf-8"):
            r = json.loads(l)
            add("r10_" + ds, r.get("english", ""), r.get("target", ""))
    return dict(out)


def split_dom(sets: Dict[str, List[Dict[str, str]]]):
    rel = [u for n, us in sets.items() for u in us if u["dom"] == "rel"]
    ood = [u for n, us in sets.items() for u in us if u["dom"] == "ood"]
    return rel, ood
