# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pseudo references of the long-document benchmark (D-44): every English sentence of every document translated ON ITS OWN by the language's chosen open system, then joined.

    python scripts/longdoc_translate.py --docs source_set/core/docs.jsonl --tiers bakeoff/tiers.json --backend hf --device cuda --out source_set/core/refs
    python scripts/longdoc_translate.py --docs ... --system madlad400-7b-mt --system nllb200-3.3b --langs yor hau --out ...      # two systems = two reference sets + agreement

Output per (system, language): `<out>/refs_pseudo_<system>_<iso>.jsonl`, rows {doc_id, iso, ref_kind "pseudo", ref_system, ref_sents (aligned 1:1 to the document's
canonical sentences), flags, flagged_share}. `<out>/pseudo_summary.json` has the per-language flag rates, the number of valid documents and, with two systems, the
agreement (document-level chrF++ of one system's output against the other's). A document whose reference has more than `max_flagged` (5%) flagged sentences
(empty / copied / looping / absurd length) is `valid=false` and dropped by `longdoc.load_refdocs(only_valid=True)`.
Resumable: a finished (system, language) file is skipped; sentence translations are cached in `<out>/cache_<system>.jsonl`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

from gemma_mt.eval import longdoc as L
from gemma_mt.eval.longdoc_backends import Translator, target_code
from gemma_mt.languages import SUPPORTED_RL, canonical_iso


def load_english_docs(path: str) -> List[Dict[str, Any]]:
    """docs.jsonl rows {doc_id, text, ...}; the sentence units are recomputed with `canonical_sentences` (the file's `sents` must agree, checked)."""
    docs = L.read_jsonl(path)
    for d in docs:
        sents = L.canonical_sentences(d["text"])
        if "sents" in d and d["sents"] != sents:
            raise ValueError(f"{d['doc_id']}: stored sentences differ from canonical_sentences(text)")
    return docs


def pseudo_refs_for_language(docs: Sequence[Dict[str, Any]], translator: Translator, iso: str, system: str, out_path: str, *, batch_size: int = 128,
                             max_flagged: float = 0.05, retry: Optional[Translator] = None, log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Translate `docs` into `iso` sentence by sentence and write the reference rows to `out_path`. Returns the summary of the language."""
    t0 = time.time()
    retry_fn = None
    if retry is not None:
        retry_fn = retry.fn_for(iso)
    td = L.translate_documents(list(docs), translator.fn_for(iso), iso=iso, system=system, batch_size=batch_size, retry_fn=retry_fn)
    rows = []
    for t in td:
        r = t.to_ref_row()
        r["valid"] = t.flagged_share <= max_flagged
        rows.append(r)
    L.write_jsonl(out_path, rows)
    n_sent = sum(len(t.tgt_sents) for t in td)
    flags: Dict[str, int] = {}
    for t in td:
        for f in t.flags:
            if f:
                flags[f] = flags.get(f, 0) + 1
    s = {"iso": iso, "system": system, "docs": len(td), "valid_docs": sum(1 for r in rows if r["valid"]), "sentences": n_sent, "flags": flags,
         "flagged_share": sum(flags.values()) / max(n_sent, 1), "seconds": round(time.time() - t0, 1)}
    log(f"[pseudo] {system} {iso}: {s['valid_docs']}/{s['docs']} valid docs, {n_sent} sentences, {sum(flags.values())} flagged ({s['seconds']}s)")
    return s


def agreement(refs_a: Dict[str, List[str]], refs_b: Dict[str, List[str]]) -> Optional[Dict[str, float]]:
    """Document-level agreement of two pseudo-reference sets of one language ({doc_id: sentences}): corpus chrF++ of A against B and of B against A, and their mean."""
    ids = [i for i in refs_a if i in refs_b]
    if len(ids) < 3:
        return None
    from gemma_mt.eval.metrics import chrf_pp

    a = [" ".join(refs_a[i]) for i in ids]
    b = [" ".join(refs_b[i]) for i in ids]
    ab, ba = chrf_pp(a, b), chrf_pp(b, a)
    return {"n_docs": len(ids), "chrf_a_vs_b": ab, "chrf_b_vs_a": ba, "chrf_mean": 0.5 * (ab + ba)}


def main(argv: Optional[List[str]] = None) -> int:
    from gemma_mt.eval import longdoc_bakeoff as BO
    from gemma_mt.eval.longdoc_backends import make_translator

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--langs", nargs="*", default=None)
    ap.add_argument("--system", action="append", default=None, help="repeat for several systems; default: the system the tiers file chose per language")
    ap.add_argument("--tiers", default=None)
    ap.add_argument("--min-tier", default="C", help="skip languages below this tier (A > B > C > N) when systems come from the tiers file")
    ap.add_argument("--backend", default="hf", choices=["hf", "vllm", "ctranslate2", "stub"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--batch-items", type=int, default=64)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--allow-proxy", action="store_true")
    ap.add_argument("--limit-docs", type=int, default=0)
    a = ap.parse_args(argv)
    docs = load_english_docs(a.docs)
    if a.limit_docs:
        docs = docs[: a.limit_docs]
    tiers = L.load_tiers(a.tiers)
    order = {"A": 0, "B": 1, "C": 2, "N": 3}
    langs = [canonical_iso(x) for x in (a.langs or SUPPORTED_RL)]
    os.makedirs(a.out, exist_ok=True)
    translators: Dict[str, Translator] = {}
    summary: Dict[str, Any] = {"languages": {}, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "docs": len(docs)}
    done: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
    for iso in langs:
        systems = a.system or ([tiers[iso]["system"]] if tiers.get(iso, {}).get("system") and order.get(tiers[iso]["tier"], 3) <= order[a.min_tier] else [])
        for name in systems:
            spec = BO.get_spec(name)
            if a.backend != "stub" and target_code(spec, iso, a.allow_proxy) is None:
                summary["languages"].setdefault(iso, {})[name] = {"status": "unsupported"}
                continue
            out_path = os.path.join(a.out, f"refs_pseudo_{name}_{iso}.jsonl")
            if name not in translators:
                kw = dict(device=a.device, dtype=a.dtype, num_beams=a.num_beams, batch_items=a.batch_items, allow_proxy=a.allow_proxy,
                          cache_path=os.path.join(a.out, f"cache_{name}.jsonl"))
                if a.backend == "ctranslate2":
                    kw = dict(model_dir=a.model_dir, device=a.device, allow_proxy=a.allow_proxy, cache_path=kw["cache_path"])
                elif a.backend == "vllm":
                    kw = dict(allow_proxy=a.allow_proxy, cache_path=kw["cache_path"])
                elif a.backend == "stub":
                    kw = dict(cache_path=kw["cache_path"])
                translators[name] = make_translator(spec, a.backend, **kw)
            if os.path.isfile(out_path):
                summary["languages"].setdefault(iso, {})[name] = {"status": "already_done"}
            else:
                summary["languages"].setdefault(iso, {})[name] = pseudo_refs_for_language(docs, translators[name], iso, name, out_path)
            done.setdefault(iso, {})[name] = {r["doc_id"]: r["ref_sents"] for r in L.read_jsonl(out_path)}
        if len(done.get(iso, {})) >= 2:
            (n1, r1), (n2, r2) = list(done[iso].items())[:2]
            summary["languages"][iso]["agreement"] = {"systems": [n1, n2], **(agreement(r1, r2) or {})}
    for t in translators.values():
        t.close()
    with open(os.path.join(a.out, "pseudo_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
