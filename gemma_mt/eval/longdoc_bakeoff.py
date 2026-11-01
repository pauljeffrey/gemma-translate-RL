# SPDX-License-Identifier: AGPL-3.0-or-later
"""Translator bake-off on human sentence sets (D-44): which open system makes the best pseudo-reference translator of each language, and how much to trust it.

    python scripts/longdoc_bakeoff.py run --system madlad400-7b-mt --backend hf --device cuda --dtype bf16 --refset flores_dev --langs yor hau ... --out DIR
    python scripts/longdoc_bakeoff.py run --system stub --backend stub --langs yor --limit 20 --out DIR           # plumbing check, no model
    python scripts/longdoc_bakeoff.py run --hf-id Helsinki-NLP/opus-mt-en-ig --prompt-style marian --code ig --langs ibo --limit 8 --out DIR   # unregistered model
    python scripts/longdoc_bakeoff.py compare --dirs DIR_A DIR_B ... --out tiers.json --report report.md

`run` translates the English sentences of FLORES+ dev (default; `--refset flores_devtest | smol_sent | mafand_dev` for languages FLORES+ does not cover) with ONE
system, scores chrF++ (word order 2), BLEU (13a) and, where the language is covered by AfriCOMET, AfriCOMET-STL-1.1 (reference based, `--comet`), with sentence-
level bootstrap CIs, plus the rates of empty / copied / looping outputs. Translations are cached (resume after a crash) and written per language.
`compare` merges several run directories: per language the best system by chrF++ (a permissive-licence system wins when it is within `--licence-margin` chrF++ of
the best non-permissive one), the confidence TIER (A / B / C / N, `longdoc.assign_tier`), the pairwise agreement (chrF++ of one system's output against another's)
and a markdown table. The tiers file is the input of `longdoc.py score --tiers`.
GPU use: only through the backend (`--device cuda` / vLLM / CTranslate2); the default is CPU. Nothing is uploaded anywhere.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence

from gemma_mt.eval import longdoc as L
from gemma_mt.eval import longdoc_refs as R
from gemma_mt.eval.longdoc_backends import StubTranslator, Translator, make_translator, target_code
from gemma_mt.languages import LANGUAGES, SUPPORTED_RL, canonical_iso

PERMISSIVE = ("permissive",)


def get_spec(name: Optional[str] = None, hf_id: str = "", prompt_style: str = "marian", codes: Optional[Dict[str, str]] = None, template: str = "",
             src_code: str = "en", family: str = "other") -> Any:
    """A registered `longdoc_systems.SystemSpec` by name, or an ad-hoc spec for an unregistered model (`codes`: iso -> system code)."""
    if name and name != "adhoc":
        if name == "stub":
            return SimpleNamespace(name="stub", hf_id="", prompt_style="nllb", template="", codes={}, proxy_codes={}, src_code="eng", licence_class="permissive", family="stub",
                                   weights_licence="n/a")
        try:
            from gemma_mt.eval import longdoc_systems as S
        except ImportError as e:
            raise KeyError(f"unknown system {name!r} (the registry gemma_mt.eval.longdoc_systems is not available: {e})")
        if name not in S.SYSTEMS:
            raise KeyError(f"unknown system {name!r}; registered: {', '.join(sorted(S.SYSTEMS))}")
        return S.SYSTEMS[name]
    return SimpleNamespace(name=hf_id or "adhoc", hf_id=hf_id, hf_id_template="", revision="", family=family, prompt_style=prompt_style, template=template, src_code=src_code,
                           codes=dict(codes or {}), proxy_codes={}, licence_class="unknown", weights_licence="unknown")


def comet_applicable(iso: str) -> bool:
    l = LANGUAGES[canonical_iso(iso)]
    return l.africomet_code is not None and l.africomet_status in ("verified", "listed")


def _score_sentences(hyps: List[str], refs: List[str], srcs: List[str], scorer_ref: Optional[Callable], scorer_qe: Optional[Callable], n_boot: int, seed: int) -> Dict[str, Any]:
    cs, bs = L.doc_stat_vectors(hyps, refs)
    boot = L.bootstrap_scores(cs, bs, n_boot, seed)
    res: Dict[str, Any] = {"n": len(hyps), "chrf": boot["chrf"], "bleu": boot["bleu"]}
    if scorer_ref is not None:
        sc = scorer_ref(srcs, hyps, refs)
        res["africomet"] = L.bootstrap_mean(sc, n_boot, seed + 1)
    if scorer_qe is not None:
        res["africomet_qe"] = L.bootstrap_mean(scorer_qe(srcs, hyps), n_boot, seed + 2)
    flags = [L.sentence_flag(s, h) for s, h in zip(srcs, hyps)]
    for k in ("empty", "copy", "loop", "short", "long"):
        res[f"rate_{k}"] = sum(1 for f in flags if f == k) / max(len(flags), 1)
    res["length_ratio"] = sum(len(h) for h in hyps) / max(sum(len(r) for r in refs), 1)
    return res


def run_bakeoff(spec: Any, translator: Translator, langs: Sequence[str], refset: str, out_dir: str, *, bench_root: str = R.DEFAULT_BENCH_ROOT, dev_dir: Optional[str] = None,
                limit: int = 0, scorer_ref: Optional[Callable] = None, scorer_qe: Optional[Callable] = None, comet_all: bool = False, n_boot: int = 1000, seed: int = 0,
                allow_proxy: bool = False, batch_sentences: int = 256, log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Translate + score every language of `langs` with `translator`; write translations_<iso>.jsonl and results.json into `out_dir`. Returns the results dict."""
    os.makedirs(out_dir, exist_ok=True)
    results: Dict[str, Any] = {"system": getattr(spec, "name", "?"), "hf_id": getattr(spec, "hf_id", ""), "weights_licence": getattr(spec, "weights_licence", ""),
                               "licence_class": getattr(spec, "licence_class", ""), "refset": refset, "limit": limit, "allow_proxy": allow_proxy,
                               "created": time.strftime("%Y-%m-%d %H:%M:%S"), "per_language": {}}
    for raw in langs:
        iso = canonical_iso(raw)
        entry: Dict[str, Any] = {"iso": iso}
        is_stub = getattr(spec, "family", "") == "stub"
        code = target_code(spec, iso, allow_proxy) if not is_stub else "stub"
        rows = R.load_refset(refset, iso, bench_root, dev_dir, limit)
        if code is None:
            entry["status"] = "unsupported"
        elif not rows:
            entry["status"] = f"no_{refset}_data"
        else:
            entry["status"] = "ok"
            entry["code"] = code
            entry["proxy"] = bool(allow_proxy and iso not in (getattr(spec, "codes", {}) or {}) and not is_stub)
            srcs = [r["english"] for r in rows]
            refs = [r["reference"] for r in rows]
            t0 = time.time()
            hyps: List[str] = []
            for i in range(0, len(srcs), batch_sentences):
                hyps += translator.translate(srcs[i: i + batch_sentences], iso)
            sec = time.time() - t0
            with open(os.path.join(out_dir, f"translations_{iso}.jsonl"), "w", encoding="utf-8") as f:
                for r, h in zip(rows, hyps):
                    f.write(json.dumps({"id": r["id"], "english": r["english"], "reference": r["reference"], "hyp": h}, ensure_ascii=False) + "\n")
            use_comet = comet_all or comet_applicable(iso)
            entry.update(_score_sentences(hyps, refs, srcs, scorer_ref if use_comet else None, scorer_qe if use_comet else None, n_boot, seed))
            entry["seconds"] = round(sec, 2)
            entry["sentences_per_second"] = round(len(srcs) / max(sec, 1e-9), 2)
            log(f"[bakeoff] {results['system']} {iso}: n={len(srcs)} chrF++ {entry['chrf']['value']:.1f} BLEU {entry['bleu']['value']:.1f}"
                + (f" AfriCOMET {entry['africomet']['value']:.3f}" if "africomet" in entry else "") + f" ({sec:.0f}s)")
        results["per_language"][iso] = entry
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(run_report(results))
    return results


def run_report(results: Dict[str, Any]) -> str:
    out = [f"# {results['system']} on {results['refset']}", "", f"licence: {results.get('weights_licence', '')} ({results.get('licence_class', '')}); proxies allowed: {results.get('allow_proxy')}", "",
           "| lang | status | n | chrF++ [95% CI] | BLEU | AfriCOMET | empty | copy | loop | len ratio | sent/s |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for iso, e in sorted(results["per_language"].items()):
        if e["status"] != "ok":
            out.append(f"| `{iso}` | {e['status']} | | | | | | | | | |")
            continue
        c = e["chrf"]
        out.append(f"| `{iso}` | ok{' (proxy)' if e.get('proxy') else ''} | {e['n']} | {c['value']:.1f} [{c['lo']:.1f}, {c['hi']:.1f}] | {e['bleu']['value']:.1f} | "
                   f"{('%.3f' % e['africomet']['value']) if 'africomet' in e else '-'} | {e['rate_empty']:.3f} | {e['rate_copy']:.3f} | {e['rate_loop']:.3f} | "
                   f"{e['length_ratio']:.2f} | {e.get('sentences_per_second', '-')} |")
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------------------------------------ compare
def load_runs(dirs: Sequence[str]) -> List[Dict[str, Any]]:
    runs = []
    for d in dirs:
        with open(os.path.join(d, "results.json"), encoding="utf-8") as f:
            r = json.load(f)
        r["_dir"] = d
        runs.append(r)
    return runs


def pairwise_agreement(dir_a: str, dir_b: str, iso: str) -> Optional[float]:
    """chrF++ of system A's translations scored against system B's (symmetrised mean of both directions) on the sentences they share; None without both files."""
    pa, pb = os.path.join(dir_a, f"translations_{iso}.jsonl"), os.path.join(dir_b, f"translations_{iso}.jsonl")
    if not (os.path.isfile(pa) and os.path.isfile(pb)):
        return None
    a = {r["id"]: r["hyp"] for r in L.read_jsonl(pa)}
    b = {r["id"]: r["hyp"] for r in L.read_jsonl(pb)}
    ids = [i for i in a if i in b]
    if len(ids) < 5:
        return None
    from gemma_mt.eval.metrics import chrf_pp

    return 0.5 * (chrf_pp([a[i] for i in ids], [b[i] for i in ids]) + chrf_pp([b[i] for i in ids], [a[i] for i in ids]))


def compare_runs(runs: Sequence[Dict[str, Any]], thresholds: Optional[Dict[str, Dict[str, float]]] = None, licence_margin: float = 1.5, use_lower_bound: bool = False,
                 isos: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Per language: all systems' scores, the chosen pseudo-reference system and the tier."""
    isos = list(isos or SUPPORTED_RL)
    tiers: Dict[str, Any] = {}
    for iso in isos:
        cands = []
        for r in runs:
            e = r["per_language"].get(iso)
            if e and e.get("status") == "ok":
                cands.append({"system": r["system"], "dir": r["_dir"], "refset": r["refset"], "licence_class": r.get("licence_class", ""), "chrf": e["chrf"]["value"],
                              "chrf_lo": e["chrf"]["lo"], "bleu": e["bleu"]["value"], "africomet": (e.get("africomet") or {}).get("value"), "n": e["n"], "proxy": e.get("proxy", False),
                              "rate_loop": e.get("rate_loop"), "rate_empty": e.get("rate_empty")})
        if not cands:
            tiers[iso] = {"tier": "N", "system": None, "reason": "no system evaluated this language", "candidates": []}
            continue
        cands.sort(key=lambda c: -c["chrf"])
        best = cands[0]
        if best["licence_class"] not in PERMISSIVE:                      # prefer an openly licensed system when it is almost as good
            perm = [c for c in cands if c["licence_class"] in PERMISSIVE and best["chrf"] - c["chrf"] <= licence_margin]
            if perm:
                best = perm[0]
        t = L.assign_tier(best, thresholds, use_lower_bound=use_lower_bound)
        t["licence_class"] = best["licence_class"]
        t["proxy"] = best["proxy"]
        t["refset"] = best["refset"]
        t["candidates"] = cands
        others = [c for c in cands if c["system"] != best["system"]]
        if others:
            t["runner_up"] = others[0]["system"]
            t["runner_up_chrf"] = others[0]["chrf"]
            ag = pairwise_agreement(best["dir"], others[0]["dir"], iso)
            if ag is not None:
                t["agreement_chrf"] = ag
        tiers[iso] = t
    counts: Dict[str, int] = {}
    for t in tiers.values():
        counts[t["tier"]] = counts.get(t["tier"], 0) + 1
    return {"meta": {"thresholds": thresholds or L.DEFAULT_TIER_THRESHOLDS, "licence_margin": licence_margin, "use_lower_bound": use_lower_bound,
                     "systems": [r["system"] for r in runs], "created": time.strftime("%Y-%m-%d %H:%M:%S"), "counts": counts}, "tiers": tiers}


def compare_report(cmp: Dict[str, Any]) -> str:
    systems = cmp["meta"]["systems"]
    out = ["| lang | " + " | ".join(systems) + " | best | tier | agreement |", "|---|" + "---|" * (len(systems) + 3)]
    for iso, t in cmp["tiers"].items():
        by = {c["system"]: c for c in t.get("candidates", [])}
        cells = []
        for s in systems:
            c = by.get(s)
            cells.append("-" if c is None else f"{c['chrf']:.1f}" + ("" if c["africomet"] is None else f" / {c['africomet']:.2f}"))
        out.append(f"| `{iso}` | " + " | ".join(cells) + f" | {t.get('system') or '-'} | {t['tier']} | {('%.1f' % t['agreement_chrf']) if 'agreement_chrf' in t else '-'} |")
    out.append("")
    out.append("tier counts: " + ", ".join(f"{k}={v}" for k, v in sorted(cmp["meta"]["counts"].items())))
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------------------------------------ CLI
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="translate + score one system")
    r.add_argument("--system", default="adhoc", help="registered system name (longdoc_systems), 'stub', or 'adhoc' with --hf-id")
    r.add_argument("--backend", default="hf", choices=["hf", "vllm", "ctranslate2", "stub"])
    r.add_argument("--hf-id", default="")
    r.add_argument("--prompt-style", default="marian")
    r.add_argument("--code", default=None, help="adhoc: the system code used for every language given (one language) ")
    r.add_argument("--template", default="")
    r.add_argument("--langs", nargs="*", default=None, help="our iso tags (default: the 77)")
    r.add_argument("--refset", default="flores_dev", choices=list(R.REFSETS))
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--out", required=True)
    r.add_argument("--bench-root", default=R.DEFAULT_BENCH_ROOT)
    r.add_argument("--dev-dir", default=None)
    r.add_argument("--device", default="cpu")
    r.add_argument("--dtype", default="fp32")
    r.add_argument("--num-beams", type=int, default=4)
    r.add_argument("--batch-items", type=int, default=32)
    r.add_argument("--allow-proxy", action="store_true")
    r.add_argument("--model-dir", default=None, help="ctranslate2 converted model directory")
    r.add_argument("--comet", default="none", choices=["none", "fake", "inproc", "worker"], help="AfriCOMET-STL-1.1 (reference based) on the covered languages")
    r.add_argument("--comet-qe", default="none", choices=["none", "fake", "inproc", "worker"])
    r.add_argument("--python", default=None)
    r.add_argument("--pythonpath", default=None)
    r.add_argument("--comet-batch", type=int, default=32)
    r.add_argument("--n-boot", type=int, default=1000)
    r.add_argument("--fetch-dev", action="store_true", help="download the FLORES+ dev files needed (gated repo) before running")
    c = sub.add_parser("compare", help="merge run directories: best system and tier per language")
    c.add_argument("--dirs", nargs="+", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--report", default=None)
    c.add_argument("--licence-margin", type=float, default=1.5)
    c.add_argument("--lower-bound", action="store_true")
    c.add_argument("--thresholds", default=None, help="json {'A': {'chrf':..,'comet':..}, 'B': {...}}")
    a = ap.parse_args(argv)
    if a.cmd == "compare":
        th = json.load(open(a.thresholds)) if a.thresholds else None
        cmp = compare_runs(load_runs(a.dirs), th, a.licence_margin, a.lower_bound)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(cmp, f, ensure_ascii=False, indent=1)
        md = compare_report(cmp)
        if a.report:
            with open(a.report, "w", encoding="utf-8") as f:
                f.write(md)
        print(md)
        return 0
    langs = [canonical_iso(x) for x in (a.langs or SUPPORTED_RL)]
    codes = {langs[0]: a.code} if (a.code and a.system == "adhoc") else None
    spec = get_spec(a.system, a.hf_id, a.prompt_style, codes, a.template)
    dev_dir = a.dev_dir or os.path.join(os.path.dirname(os.path.abspath(a.out.rstrip("/"))), "flores_dev_raw")
    if a.refset == "flores_dev" and a.fetch_dev:
        R.fetch_flores_dev(["eng_Latn"] + sorted({LANGUAGES[i].flores for i in langs if LANGUAGES[i].flores}), dev_dir)
    if a.backend == "stub":
        tr = StubTranslator(allow_proxy=a.allow_proxy, cache_path=os.path.join(a.out, "cache.jsonl"))
    elif a.backend == "hf":
        tr = make_translator(spec, "hf", device=a.device, dtype=a.dtype, num_beams=a.num_beams, batch_items=a.batch_items, allow_proxy=a.allow_proxy,
                             cache_path=os.path.join(a.out, "cache.jsonl"))
    elif a.backend == "ctranslate2":
        tr = make_translator(spec, "ctranslate2", model_dir=a.model_dir, device=a.device, allow_proxy=a.allow_proxy, cache_path=os.path.join(a.out, "cache.jsonl"))
    else:
        tr = make_translator(spec, "vllm", allow_proxy=a.allow_proxy, cache_path=os.path.join(a.out, "cache.jsonl"))
    sref = L._make_scorer(a.comet, a.python, a.pythonpath, True, a.comet_batch)
    sqe = L._make_scorer(a.comet_qe, a.python, a.pythonpath, False, a.comet_batch)
    res = run_bakeoff(spec, tr, langs, a.refset, a.out, bench_root=a.bench_root, dev_dir=dev_dir, limit=a.limit, scorer_ref=sref, scorer_qe=sqe, n_boot=a.n_boot,
                      allow_proxy=a.allow_proxy)
    tr.close()
    print(run_report(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
