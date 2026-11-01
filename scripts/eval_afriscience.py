#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Evaluate a checkpoint on AfriScience-MT (dsfsi/afriscience_mt, Apache-2.0; English -> amh hau lug nso yor zul), DECISIONS.md D-52.

    python scripts/eval_afriscience.py --model <HF_ORG_LEGACY>/gemma-3-270m-translate-rl2c --revision latest --splits dev,test --out DIR \\
        --vllm-python /workspace/venv_vllm/bin/python [--decoding greedy | sample:temperature=0.3,top_p=0.95 | @decoding.json] [--levels sentence,document]

Two levels, both from the dataset's `corpus` config (the English side is identical in the six lang_pairs; checked):
  sentence   every English sentence is translated on its own (what the paper's Table 2/3 score; 970 test sentences per language)
  document   a whole paper (its sentences joined by single spaces, 28 test papers = 600-1,500 AfriXLM-R tokens) is translated in ONE prompt (the paper's document-level setting)
Metrics per language and macro: chrF (sacrebleu defaults: char 6, word 0, beta 2: the paper's chrF), chrF++ (word order 2), BLEU (13a), spBLEU (flores200 tokenizer), mean sentence chrF
(sentence level only; the average of per-sentence chrF, the other possible reading of the paper's "metrics on individual sentence pairs"), finished rate (generation stopped at its own
end token), prefix-ok rate, length ratio (hyp / reference characters). Hypotheses and references are NFC-normalised. NOT computed here: SSA-COMET, the paper's primary metric.
`--published CSV` puts the paper's own numbers (dataset config `metrics`, test split) beside the result.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple

LANGS = ("amh", "hau", "lug", "nso", "yor", "zul")
NAMES = {"amh": "Amharic", "hau": "Hausa", "lug": "Luganda", "nso": "Northern Sotho", "yor": "Yoruba", "zul": "isiZulu"}


# ---------------------------------------------------------------------------------------------------------------------------- data
def load_corpus(corpus_dir: str, split: str, languages: Sequence[str] = LANGS) -> Dict[str, Dict[str, Any]]:
    """{iso: {"sentences": [{paper, sid, domain, src, ref}], "documents": [{paper, domain, src, ref, n_sentences}]}} from `<corpus_dir>/<split>-*.parquet`."""
    import glob

    import pandas as pd

    files = sorted(glob.glob(os.path.join(corpus_dir, f"{split}-*.parquet")))
    if not files:
        raise FileNotFoundError(f"no {split}-*.parquet in {corpus_dir}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    out: Dict[str, Dict[str, Any]] = {}
    for iso in languages:
        d = df[df.lang_pair == f"eng-{iso}"].sort_values(["paper_id", "sentence_id"])
        if d.empty:
            raise ValueError(f"{split}: no rows for eng-{iso}")
        nf = unicodedata.normalize
        sents = [{"paper": str(r.paper_id), "sid": int(r.sentence_id), "domain": r.domain, "src": nf("NFC", r.source), "ref": nf("NFC", r.target)} for r in d.itertuples()]
        docs = []
        for pid, g in d.groupby("paper_id", sort=True):
            docs.append({"paper": str(pid), "domain": g.domain.iloc[0], "src": " ".join(nf("NFC", s) for s in g.source), "ref": " ".join(nf("NFC", s) for s in g.target), "n_sentences": len(g)})
        out[iso] = {"sentences": sents, "documents": docs}
    return out


# ---------------------------------------------------------------------------------------------------------------------------- decoding
def parse_decoding(spec: str) -> Dict[str, Any]:
    """'greedy' (default) | 'sample[:k=v,...]' with temperature / top_p / top_k / seed | '@file.json' (the same keys plus `greedy`)."""
    d: Dict[str, Any] = {"greedy": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0, "seed": 0}
    if spec in ("", "greedy"):
        return d
    if spec.startswith("@"):
        d.update(json.load(open(spec[1:])))
        d["greedy"] = bool(d.get("greedy", d.get("temperature", 1.0) in (0, 0.0)))
        return d
    kind, _, rest = spec.partition(":")
    if kind != "sample":
        raise ValueError(f"unknown decoding {spec!r}: greedy | sample:k=v,... | @file.json")
    d["greedy"] = False
    for kv in filter(None, rest.split(",")):
        k, _, v = kv.partition("=")
        if k not in ("temperature", "top_p", "top_k", "seed"):
            raise ValueError(f"unknown decoding key {k!r}")
        d[k] = int(v) if k in ("top_k", "seed") else float(v)
    return d


# ---------------------------------------------------------------------------------------------------------------------------- metrics
def _nfc(xs: Sequence[str]) -> List[str]:
    return [unicodedata.normalize("NFC", x) for x in xs]


def text_metrics(hyps: Sequence[str], refs: Sequence[str], sentence_mean: bool = False) -> Dict[str, float]:
    from sacrebleu.metrics import CHRF

    from gemma_mt.eval import metrics as M

    h, r = _nfc(hyps), _nfc(refs)
    out = {"chrf": float(CHRF(word_order=0).corpus_score(h, [r]).score), "chrf_pp": float(CHRF(word_order=2).corpus_score(h, [r]).score),
           "bleu": M.bleu(h, r), "spbleu": M.spbleu(h, r), "n": len(h)}
    if sentence_mean:
        c = CHRF(word_order=0)
        out["chrf_sentence_mean"] = sum(c.sentence_score(a, [b]).score for a, b in zip(h, r)) / max(len(h), 1)
    out["length_ratio"] = sum(len(a) for a in h) / max(sum(len(b) for b in r), 1)
    return out


def macro(per_lang: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    keys = {k for d in per_lang.values() for k, v in d.items() if isinstance(v, (int, float)) and k != "n"}
    return {k: sum(d[k] for d in per_lang.values() if k in d) / max(sum(1 for d in per_lang.values() if k in d), 1) for k in sorted(keys)}


def score_level(rows_by_lang: Dict[str, List[Dict[str, Any]]], outputs: Dict[str, List[Dict[str, Any]]], sentence_mean: bool) -> Dict[str, Any]:
    per, domains = {}, {}
    for iso, rows in rows_by_lang.items():
        outs = outputs[iso]
        m = text_metrics([o["hyp"] for o in outs], [r["ref"] for r in rows], sentence_mean)
        m["finished_rate"] = sum(bool(o.get("finished", True)) for o in outs) / max(len(outs), 1)
        m["prefix_ok_rate"] = sum(o.get("prefix_status", "ok") in ("ok", "present", "stripped", "n/a") for o in outs) / max(len(outs), 1)
        m["empty_rate"] = sum(not o["hyp"].strip() for o in outs) / max(len(outs), 1)
        per[iso] = m
        bydom: Dict[str, Tuple[List[str], List[str]]] = {}
        for r, o in zip(rows, outs):
            h, rf = bydom.setdefault(r["domain"], ([], []))
            h.append(o["hyp"]); rf.append(r["ref"])
        domains[iso] = {d: {"chrf": text_metrics(h, rf)["chrf"], "n": len(h)} for d, (h, rf) in sorted(bydom.items())}
    return {"per_language": per, "macro": macro(per), "per_domain_chrf": domains}


# ---------------------------------------------------------------------------------------------------------------------------- generation
def generate(sampler, tok, stop_ids, pad_id, prompts_src: Sequence[Tuple[str, str]], dec: Dict[str, Any], max_new_tokens: int, chunk: int = 4096, log=print) -> List[Dict[str, Any]]:
    """prompts_src = [(english, iso)] -> [{raw, hyp, prefix_status, finished}] through the vLLM sampler (greedy or sampled, one completion per prompt)."""
    from gemma_mt import format as fmt
    from gemma_mt.rl.text import decode_completion

    prompts = [fmt.build_prompt(e, iso, tok) for e, iso in prompts_src]
    res: List[Dict[str, Any]] = []
    for s in range(0, len(prompts), chunk):
        roll = sampler.generate(prompts[s: s + chunk], group_size=1, max_new_tokens=max_new_tokens, temperature=dec["temperature"], top_p=dec["top_p"], top_k=dec["top_k"],
                                stop_ids=stop_ids, pad_id=pad_id, greedy=dec["greedy"], key=int(dec["seed"]) + s)
        for (e, iso), ids in zip(prompts_src[s: s + chunk], roll.completions):
            d = decode_completion(tok, ids, stop_ids)
            st = fmt.strip_iso_prefix(d.text, iso)
            res.append({"raw": d.text, "hyp": st.text, "prefix_status": st.status, "finished": d.finished})
        log(f"  generated {min(s + chunk, len(prompts))}/{len(prompts)}")
    return res


def run_split(corpus: Dict[str, Dict[str, Any]], levels: Sequence[str], gen_fn, args) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    preds: Dict[str, Any] = {}
    for level in levels:
        key = "sentences" if level == "sentence" else "documents"
        order = [(iso, i) for iso in corpus for i in range(len(corpus[iso][key]))]
        outs = gen_fn([(corpus[iso][key][i]["src"], iso) for iso, i in order], args.max_new_tokens_sentence if level == "sentence" else args.max_new_tokens_document)
        by: Dict[str, List[Dict[str, Any]]] = {iso: [] for iso in corpus}
        for (iso, i), o in zip(order, outs):
            by[iso].append(o)
        out[level] = score_level({iso: corpus[iso][key] for iso in corpus}, by, sentence_mean=(level == "sentence"))
        preds[level] = {iso: [{"paper": r["paper"], "hyp": o["hyp"], "finished": o["finished"], "ref": r["ref"]} for r, o in zip(corpus[iso][key], by[iso])] for iso in corpus}
    return {"scores": out, "predictions": preds}


# ---------------------------------------------------------------------------------------------------------------------------- report
def published_table(path: Optional[str]) -> Dict[str, Dict[str, float]]:
    """{system label: {iso: chrF}} from the CSV written by scripts/prepare_afriscience.py (the paper's released per-run metrics, English -> X, test split)."""
    if not path or not os.path.isfile(path):
        return {}
    import csv

    out: Dict[str, Dict[str, float]] = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        out.setdefault(r["system"], {})[r["iso"]] = float(r["chrf"])
    return out


def markdown(name: str, split: str, res: Dict[str, Any], published: Dict[str, Dict[str, float]]) -> str:
    L = []
    for level, blk in res["scores"].items():
        isos = list(blk["per_language"])
        L.append(f"### {name} - {split} - {level} level (greedy unless stated)\n")
        L.append("| metric | " + " | ".join(isos) + " | macro |")
        L.append("|---|" + "---|" * (len(isos) + 1))
        for m in ("chrf", "chrf_pp", "bleu", "spbleu", "chrf_sentence_mean", "finished_rate", "length_ratio"):
            if not all(m in blk["per_language"][i] for i in isos):
                continue
            L.append(f"| {m} | " + " | ".join(f"{blk['per_language'][i][m]:.2f}" for i in isos) + f" | {blk['macro'][m]:.2f} |")
        if split == "test" and published:
            tag = "sentence" if level == "sentence" else "document"
            for sysname, v in published.items():
                if (tag == "document") != sysname.startswith("doc:"):
                    continue
                vals = [v.get(i) for i in isos]
                L.append(f"| published chrF: {sysname} | " + " | ".join("-" if x is None else f"{x:.1f}" for x in vals) + f" | {sum(x for x in vals if x is not None) / max(sum(x is not None for x in vals), 1):.1f} |")
        L.append("")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Hub id or local dir")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--name", default=None, help="label in the output (default: model@revision)")
    ap.add_argument("--splits", default="dev,test")
    ap.add_argument("--levels", default="sentence,document")
    ap.add_argument("--languages", default=",".join(LANGS))
    ap.add_argument("--corpus-dir", default=None, help="dir with {train,dev,test}-*.parquet (default: downloaded from the Hub dataset dsfsi/afriscience_mt, corpus/ only)")
    ap.add_argument("--decoding", default="greedy")
    ap.add_argument("--max-new-tokens-sentence", type=int, default=512)
    ap.add_argument("--max-new-tokens-document", type=int, default=7168)
    ap.add_argument("--sampler", choices=("vllm", "hf"), default="vllm")
    ap.add_argument("--vllm-python", default=None)
    ap.add_argument("--vllm-gpu-util", type=float, default=0.15)
    ap.add_argument("--vllm-max-model-len", type=int, default=10240)
    ap.add_argument("--vllm-max-num-seqs", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16, help="hf sampler")
    ap.add_argument("--extra-libs", default="/workspace/sft_build/pylibs", help="appended to sys.path (sacrebleu)")
    ap.add_argument("--published", default=None, help="CSV (system, iso, chrf) of the paper's released metrics for the test split")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-predictions", action="store_true")
    a = ap.parse_args(argv)

    from gemma_mt.eval import runtime as RT

    RT.add_extra_libs(a.extra_libs)
    os.makedirs(a.out, exist_ok=True)
    langs = [x for x in a.languages.split(",") if x]
    levels = [x for x in a.levels.split(",") if x]
    dec = parse_decoding(a.decoding)
    corpus_dir = a.corpus_dir
    if not corpus_dir:
        from huggingface_hub import snapshot_download

        corpus_dir = os.path.join(snapshot_download("dsfsi/afriscience_mt", repo_type="dataset", allow_patterns=["corpus/*", "README.md"], token=os.environ.get("HF_TOKEN")), "corpus")
    data = {sp: load_corpus(corpus_dir, sp, langs) for sp in a.splits.split(",") if sp}
    ref = RT.resolve_model(a.model, revision=a.revision)
    tok, stop, pad = RT.load_tokenizer_and_stops(a.tokenizer or ref.path)
    name = a.name or (f"{a.model}@{a.revision}" if a.revision else a.model)
    RT.log(f"{name}: sha {ref.sha}; splits {list(data)}; levels {levels}; decoding {dec}")

    t0 = time.time()
    sampler = None
    model = None
    if a.sampler == "vllm":
        if not a.vllm_python:
            ap.error("--sampler vllm needs --vllm-python")
        RT.wait_for_gpu(a.vllm_gpu_util * 96.0 * 0.9, 600)
        sampler = RT.make_vllm_sampler(ref.path, python=a.vllm_python, gpu_util=a.vllm_gpu_util, max_model_len=a.vllm_max_model_len, max_num_seqs=a.vllm_max_num_seqs,
                                       log_path=os.path.join(a.out, "vllm_worker.log"))

        def gen_fn(items, mnt):
            return generate(sampler, tok, stop.ids, pad, items, dec, mnt, log=RT.log)
    else:
        import torch

        from gemma_mt.eval.eval_translation import generate_translations
        from gemma_mt.modeling import load_model

        if not dec["greedy"]:
            ap.error("the hf sampler decodes greedily only; use --sampler vllm for sampling")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_model(ref.path, attn_implementation="sdpa", dtype=torch.bfloat16 if device == "cuda" else torch.float32, device=device)

        def gen_fn(items, mnt):
            rows = [{"english": e, "iso": iso} for e, iso in items]
            return generate_translations(model, tok, rows, stop_ids=stop.ids, pad_id=pad, device=device, batch_size=a.batch_size, max_new_tokens=mnt, log=RT.log)

    results: Dict[str, Any] = {"model": a.model, "revision": a.revision, "sha": ref.sha, "name": name, "decoding": dec, "max_new_tokens": {"sentence": a.max_new_tokens_sentence, "document": a.max_new_tokens_document},
                               "metrics_note": "chrF = sacrebleu CHRF(word_order=0); chrF++ = word_order 2; BLEU 13a; spBLEU flores200 (falls back to 13a offline); SSA-COMET not computed", "splits": {}}
    try:
        for sp, corpus in data.items():
            r = run_split(corpus, levels, gen_fn, a)
            results["splits"][sp] = r["scores"]
            if not a.no_predictions:
                with open(os.path.join(a.out, f"predictions_{sp}.json"), "w", encoding="utf-8") as f:
                    json.dump(r["predictions"], f, ensure_ascii=False)
            RT.log(f"{sp} done ({time.time() - t0:.0f}s)")
    finally:
        if sampler is not None:
            sampler.close()
    results["seconds"] = round(time.time() - t0, 1)
    results["spbleu_tokenizer"] = __import__("gemma_mt.eval.metrics", fromlist=["x"]).spbleu_tokenizer()
    with open(os.path.join(a.out, "afriscience_eval.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    pub = published_table(a.published)
    md = "\n".join(markdown(name, sp, {"scores": s}, pub) for sp, s in results["splits"].items())
    with open(os.path.join(a.out, "afriscience_eval.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
