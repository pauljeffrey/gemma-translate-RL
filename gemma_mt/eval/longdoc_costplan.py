# SPDX-License-Identifier: AGPL-3.0-or-later
"""Size and cost planning of the long-document benchmark (D-44): measured inputs + explicit assumptions -> documents x languages x tokens -> GPU hours.

    python -m gemma_mt.eval.longdoc_costplan fertility --human-dir source_set/human --out sysinfo/fertility.json
    python -m gemma_mt.eval.longdoc_costplan se --predictions /workspace/sft_build/eval/sft_best/predictions.jsonl --out sysinfo/se_analysis.json
    python -m gemma_mt.eval.longdoc_costplan plan --fertility sysinfo/fertility.json --selection source_set/SELECTION_REPORT.json --out sysinfo/cost_plan.json

`fertility`  MEASURED: tokens of the human reference per token of the English source, per language, in (a) Gemma tokens (what OUR model generates and what the
             context holds) and (b) AfroXLM-R tokens (what AfriCOMET reads); from the SmolDoc / FLORES+ human documents of `longdoc_refs human`.
`se`         MEASURED on real model output: sentence-level predictions of our SFT model on FLORES+ devtest are concatenated into pseudo-documents of k consecutive
             sentences (k = 4 ... 64, ~110 ... 1,800 English tokens); per language the document-level chrF++ spread and the 95% bootstrap half-width of the corpus
             chrF++ at n = 100 / 200 / 400 documents; paired-difference detectability from an assumed correlation (NOT measured: only one system is available).
             A LOWER BOUND for real documents: it contains no loop / omission failures and the documents are stitched, not coherent.
`plan`       Pure arithmetic with explicit assumptions (every throughput is an assumption, flagged; measured anchors from earlier reports are quoted next to them):
             documents x bin mix x tokens per document -> English tokens -> sentences -> translator output tokens per system, our model's output tokens, AfriCOMET
             triples -> GPU hours on one RTX PRO 6000 Blackwell 96 GB (roofline for decoding: tokens/s <= batch x memory bandwidth / weight bytes, times an
             efficiency factor).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

from gemma_mt.eval import longdoc as L

# ------------------------------------------------------------------------------------------------ hardware and throughput assumptions
GPU = {"name": "NVIDIA RTX PRO 6000 Blackwell 96 GB", "mem_bw_gbps": 1792.0, "note": "memory bandwidth from the NVIDIA datasheet (GDDR7, ~1.8 TB/s); not measured by us"}
# efficiency = fraction of the roofline a real engine reaches at the given batch (assumption; CTranslate2 / vLLM typically 25-50% of the memory roofline
# for the large batches used here; HF transformers `generate` 10-20%)
EFFICIENCY = {"ctranslate2": (0.25, 0.45), "vllm": (0.25, 0.45), "hf": (0.08, 0.18)}
# weight bytes per parameter by precision
BYTES = {"bf16": 2.0, "fp16": 2.0, "int8": 1.0, "int8_float16": 1.0}
# systems of the plan: (params in billions, backend, precision, decode batch)
SYSTEMS = {
    "nllb200-1.3b-distilled": (1.3, "ctranslate2", "int8_float16", 256),
    "nllb200-3.3b": (3.3, "ctranslate2", "int8_float16", 192),
    "madlad400-3b-mt": (3.0, "ctranslate2", "int8_float16", 192),
    "madlad400-7b-mt": (7.2, "ctranslate2", "int8_float16", 128),
    "madlad400-10b-mt": (10.7, "ctranslate2", "int8_float16", 96),
    "translategemma-12b-it": (13.2, "vllm", "bf16", 128),
    "sunflower-qwen3.8-27b": (27.4, "vllm", "bf16", 128),
    "afriqueqwen3.5-9b-50langs-instruct": (9.0, "vllm", "bf16", 192),
    "opus-mt-tc-bible-big-eng-mul": (0.24, "hf", "bf16", 64),
    "sunflower-qwen3.5-9b": (9.4, "vllm", "bf16", 192),
    "madlad400-3b-mt": (3.0, "ctranslate2", "int8_float16", 192),
}
# languages each system names (longdoc_systems.coverage_counts(); copied here so that the plan runs without the registry)
COVERED = {"nllb200-1.3b-distilled": 44, "nllb200-3.3b": 44, "madlad400-3b-mt": 44, "madlad400-7b-mt": 44, "madlad400-10b-mt": 44, "translategemma-12b-it": 47,
           "sunflower-qwen3.8-27b": 31, "afriqueqwen3.5-9b-50langs-instruct": 41, "opus-mt-tc-bible-big-eng-mul": 59, "sunflower-qwen3.5-9b": 31}
COVERED["nllb200-1.3b-distilled"] = 44
# the two translator bundles of the 500-document plan (owner decision 2026-10-20: one shared English set of 500 documents into all 77 languages)
# owner decision 2026-10-20 (second message): TWO systems only - NLLB-200 3.3B (internal referee, CC-BY-NC) and MADLAD-400 (released pseudo-references, Apache-2.0); OPUS-MT tc-bible is an optional zero-cost supplement
BUNDLE_FULL = ("nllb200-3.3b", "madlad400-7b-mt")                        # 17.6 GB + 33.2 GB of weights, one system at a time
BUNDLE_MIN = ("nllb200-3.3b", "madlad400-3b-mt")                         # 17.6 GB + 11.8 GB
OWN_MODEL = {"name": "gemma-3-270m SabiYarn-MT (vLLM)", "tok_s_low": 10_000.0, "tok_s_high": 40_000.0,
             "anchor": "MEASURED 5,347 tok/s with 4 sequences of b1024 on a card at 100% load from another job (reports/gemma_rl_vllm_2026-10-17.md); 5-10k tok/s sustained there. "
                       "ASSUMPTION for a dedicated 96 GB card with 256-512 concurrent sequences: 10-40k tok/s."}
COMET = {"pairs_s_5070": (290.0, 350.0), "anchor": "MEASURED on the RTX 5070 (AfriCOMET-QE fp16, 87.5 AfroXLM tokens per (src, mt) pair): 290-350 pairs/s single process, 480 pairs/s with three "
                                                  "(reports/gemma_rl_cost_estimate_2026-10-17.md)", "speedup_vs_5070": (1.5, 3.0), "ref_token_factor": 1.5}


def decode_tok_s(params_b: float, backend: str, precision: str, batch: int, eff: float) -> float:
    """Roofline decode throughput: each step reads all weights once and emits `batch` tokens."""
    weight_gb = params_b * BYTES[precision]
    return batch * GPU["mem_bw_gbps"] / weight_gb * eff


# ------------------------------------------------------------------------------------------------ fertility (measured)
def _afroxlm_tokenizer():
    from tokenizers import Tokenizer

    hits = sorted(glob.glob(os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub", "models--Davlan--afro-xlmr-large-76L", "snapshots", "*", "tokenizer.json")))
    if not hits:
        return None
    t = Tokenizer.from_file(hits[-1])
    return lambda s: len(t.encode(s, add_special_tokens=False).ids)


def measure_fertility(human_dir: str, tokenizer_json: Optional[str] = None, max_docs: int = 400) -> Dict[str, Any]:
    """Per language: Gemma tokens and AfroXLM tokens of the human reference per English token (corpus ratio), SmolDoc + FLORES+ documents."""
    count = L.load_token_counter(tokenizer_json)
    count_x = _afroxlm_tokenizer()
    docs = {d["doc_id"]: d for d in L.read_jsonl(os.path.join(human_dir, "docs_human.jsonl"))}
    out: Dict[str, Any] = {}
    for p in sorted(glob.glob(os.path.join(human_dir, "refs_human_*.jsonl"))):
        iso = os.path.basename(p)[len("refs_human_"):-len(".jsonl")]
        src_g = ref_g = src_x = ref_x = sent_n = n = 0
        for r in L.read_jsonl(p):
            if r.get("ref_kind") == "human_stitched" or n >= max_docs:
                continue
            d = docs.get(r["doc_id"])
            if not d:
                continue
            s, t = " ".join(d["sents"]), " ".join(r["ref_sents"])
            src_g += count(s)
            ref_g += count(t)
            if count_x:
                src_x += count_x(s)
                ref_x += count_x(t)
            sent_n += len(d["sents"])
            n += 1
        if n:
            out[iso] = {"docs": n, "sentences": sent_n, "gemma_ref_per_src": ref_g / max(src_g, 1), "afroxlm_ref_per_src": (ref_x / max(src_x, 1)) if count_x else None,
                        "gemma_tokens_per_src_sentence": src_g / max(sent_n, 1), "afroxlm_src_per_sentence": (src_x / max(sent_n, 1)) if count_x else None}
    ratios = sorted(v["gemma_ref_per_src"] for v in out.values())
    summ = {"languages": len(out), "gemma_ref_per_src": {"min": ratios[0], "median": ratios[len(ratios) // 2], "mean": sum(ratios) / len(ratios), "p90": ratios[int(0.9 * (len(ratios) - 1))],
                                                       "max": ratios[-1]}} if ratios else {}
    return {"per_language": out, "summary": summ}


# ------------------------------------------------------------------------------------------------ standard errors from real predictions (measured)
def se_analysis(predictions: str, ks: Sequence[int] = (4, 8, 16, 32, 64), ns: Sequence[int] = (100, 200, 400), n_rep: int = 300, seed: int = 0, benchmark: str = "flores_plus",
                tokenizer_json: Optional[str] = None, max_langs: int = 0) -> Dict[str, Any]:
    """See the module docstring ('se')."""
    import numpy as np

    count = L.load_token_counter(tokenizer_json)
    by: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in L.read_jsonl(predictions):
        if r.get("benchmark") == benchmark and r.get("hyp") is not None:
            by[r["iso"]].append(r)
    rng = np.random.RandomState(seed)
    per_lang: Dict[str, Any] = {}
    for iso in sorted(by)[: max_langs or None]:
        rows = sorted(by[iso], key=lambda r: int(r["id"]) if str(r["id"]).lstrip("-").isdigit() else 0)
        res: Dict[str, Any] = {"n_sentences": len(rows)}
        for k in ks:
            docs = [rows[i: i + k] for i in range(0, len(rows) - k + 1, k)]
            if len(docs) < 8:
                continue
            hyp = [" ".join(r["hyp"] for r in d) for d in docs]
            ref = [" ".join(r["reference"] for r in d) for d in docs]
            cs, bs = L.doc_stat_vectors(hyp, ref)
            doc_chrf = np.array([L.scores_from_stats(cs, bs, [i])[0] for i in range(len(docs))])
            corpus = L.scores_from_stats(cs, bs)[0]
            tok = float(np.mean([count(" ".join(r["english"] for r in d)) for d in docs]))
            half = {}
            for n in ns:
                vals = []
                for _ in range(n_rep):
                    idx = rng.randint(0, len(docs), size=n)
                    vals.append(L.scores_from_stats(cs, bs, idx)[0])
                lo, hi = np.percentile(vals, [2.5, 97.5])
                half[str(n)] = float((hi - lo) / 2)
            res[str(k)] = {"n_docs": len(docs), "english_tokens": tok, "corpus_chrf": corpus, "doc_chrf_mean": float(doc_chrf.mean()), "doc_chrf_sd": float(doc_chrf.std(ddof=1)), "half_width": half}
        per_lang[iso] = res
    agg: Dict[str, Any] = {}
    for k in ks:
        rs = [v[str(k)] for v in per_lang.values() if str(k) in v]
        if not rs:
            continue
        sds = sorted(r["doc_chrf_sd"] for r in rs)
        agg[str(k)] = {"languages": len(rs), "english_tokens": float(np.mean([r["english_tokens"] for r in rs])), "mean_corpus_chrf": float(np.mean([r["corpus_chrf"] for r in rs])),
                       "doc_sd_median": sds[len(sds) // 2], "doc_sd_p90": sds[int(0.9 * (len(sds) - 1))],
                       "half_width_median": {str(n): float(np.median([r["half_width"][str(n)] for r in rs])) for n in ns},
                       "half_width_p90": {str(n): float(np.percentile([r["half_width"][str(n)] for r in rs], 90)) for n in ns}}
    # extrapolation of the document SD to longer documents: log-log fit of the median SD against the mean token length
    xs = [math.log(a["english_tokens"]) for a in agg.values()]
    ys = [math.log(a["doc_sd_median"]) for a in agg.values()]
    ext = {}
    if len(xs) >= 3:
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / max(sum((x - mx) ** 2 for x in xs), 1e-12)
        icpt = my - slope * mx
        ext = {"slope": slope, "intercept": icpt, "predicted_doc_sd": {str(t): math.exp(icpt + slope * math.log(t)) for t in (512, 1024, 2048, 4096)},
               "note": "log-log fit of the median document chrF++ SD vs English tokens over the measured lengths; 2048 / 4096 are EXTRAPOLATIONS"}
    return {"per_language": per_lang, "aggregate": agg, "extrapolation": ext, "n_rep": n_rep, "ns": list(ns), "ks": list(ks),
            "caveat": "sentence-level SFT predictions on FLORES+ devtest concatenated into pseudo-documents: no loop / omission failures, stitched articles; a lower bound of the spread of real long-document scores"}


def detectable_difference(sd_doc: float, n: int, rho: float, z_alpha: float = 1.96, z_power: float = 0.84) -> float:
    """Smallest true chrF++ difference between two systems detectable with 80% power at 5% (two-sided), documents resampled in pairs: SD of the paired per-document
    difference = sd_doc * sqrt(2 (1 - rho)); rho = correlation of the two systems' document scores (ASSUMED)."""
    return (z_alpha + z_power) * sd_doc * math.sqrt(2 * (1 - rho)) / math.sqrt(n)


def half_width(sd: float, n: int, z: float = 1.96) -> float:
    """Half-width of the 95% CI of a mean of n independent documents with standard deviation sd (the bootstrap CI converges to it)."""
    return z * sd / math.sqrt(n)


def mixture_sd(sd_quality: float, p_fail: float, delta: float) -> float:
    """SD of the document score when a share p_fail of the documents FAILS (loop / omission / not finished) and loses `delta` chrF++ on average (two-component
    mixture; the failure-free spread comes from `se_analysis`). ASSUMED p_fail and delta: the earlier SmolDoc runs finished 53-63% of the documents of a rare language."""
    return math.sqrt(sd_quality ** 2 + p_fail * (1 - p_fail) * delta ** 2)


# ------------------------------------------------------------------------------------------------ the plan (arithmetic)
def plan(n_docs: int, n_langs: int, bin_shares: Dict[int, float], tokens_per_bin: Dict[int, float], eng_tokens_per_sentence: float, gemma_fertility: float,
         translator_token_factor: float = 0.75, systems: Sequence[str] = ("nllb200-3.3b", "madlad400-7b-mt", "sunflower-qwen3.8-27b", "opus-mt-tc-bible-big-eng-mul"),
         own_models: int = 3, covered: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """GPU hours for one document set of `n_docs` English documents translated into `n_langs` languages.
    gemma_fertility = Gemma tokens of the target per Gemma token of the English; translator_token_factor = translator (SentencePiece / BPE) tokens per Gemma token of the SAME
    text (assumption: the NLLB / MADLAD vocabularies are denser on African languages than Gemma's); every translator system translates the languages it names (`covered`);
    own_models = how many of our model checkpoints are evaluated on the full documents."""
    covered = covered or COVERED
    mean_tok = sum(bin_shares[b] * tokens_per_bin[b] for b in bin_shares) / sum(bin_shares.values())
    eng_tokens = n_docs * mean_tok
    sentences = eng_tokens / eng_tokens_per_sentence
    out_tok_gemma = eng_tokens * gemma_fertility
    out: Dict[str, Any] = {"docs": n_docs, "languages": n_langs, "mean_english_tokens_per_doc": mean_tok, "english_tokens": eng_tokens, "sentences_per_language": sentences,
                           "target_gemma_tokens_per_language": out_tok_gemma, "translator": {}, "assumptions": {}}
    for name in systems:
        pb, be, prec, batch = SYSTEMS[name]
        lo_eff, hi_eff = EFFICIENCY[be]
        tok_lo, tok_hi = decode_tok_s(pb, be, prec, batch, lo_eff), decode_tok_s(pb, be, prec, batch, hi_eff)
        nl = min(n_langs, covered.get(name, n_langs))
        gen = out_tok_gemma * translator_token_factor * nl
        out["translator"][name] = {"languages": nl, "output_tokens_total": gen, "tok_s_range": [tok_lo, tok_hi], "gpu_hours": [gen / tok_hi / 3600, gen / tok_lo / 3600], "backend": be,
                                   "precision": prec, "batch": batch, "roofline_tok_s": decode_tok_s(pb, be, prec, batch, 1.0)}
    gen_own = out_tok_gemma * n_langs * own_models
    out["own_model"] = {"output_tokens_total": gen_own, "gpu_hours": [gen_own / OWN_MODEL["tok_s_high"] / 3600, gen_own / OWN_MODEL["tok_s_low"] / 3600], "models": own_models, **OWN_MODEL}
    # AfriCOMET: (source sentence, mt sentence, reference sentence) per sentence and per scored system output; the reference doubles as the pseudo-reference check
    triples = sentences * n_langs * own_models
    lo = COMET["pairs_s_5070"][0] * COMET["speedup_vs_5070"][0] / COMET["ref_token_factor"]
    hi = COMET["pairs_s_5070"][1] * COMET["speedup_vs_5070"][1] / COMET["ref_token_factor"]
    out["africomet"] = {"triples_total": triples, "triples_per_s_range": [lo, hi], "gpu_hours": [triples / hi / 3600, triples / lo / 3600], "anchor": COMET["anchor"],
                        "assumption": "x1.5-3 vs the RTX 5070 and x1.5 encoder tokens for the extra reference segment"}
    out["assumptions"] = {"gpu": GPU, "efficiency": EFFICIENCY, "translator_token_factor": translator_token_factor, "covered": {k: covered.get(k) for k in systems}}
    return out


def bakeoff_plan(systems: Sequence[str], gemma_fertility: float, sentences: int = 997, eng_tokens_per_sentence: float = 28.0, translator_token_factor: float = 0.75,
                 covered: Optional[Dict[str, int]] = None, comet_languages: int = 55) -> Dict[str, Any]:
    """GPU hours of the bake-off itself: `sentences` FLORES+ dev sentences per language the system names, AfriCOMET (reference based) on the languages it covers."""
    covered = covered or COVERED
    out: Dict[str, Any] = {"sentences_per_language": sentences, "systems": {}}
    tot_lo = tot_hi = 0.0
    for name in systems:
        pb, be, prec, batch = SYSTEMS[name]
        lo_eff, hi_eff = EFFICIENCY[be]
        nl = covered.get(name, 44)
        tokens = sentences * eng_tokens_per_sentence * gemma_fertility * translator_token_factor * nl
        t_lo, t_hi = tokens / decode_tok_s(pb, be, prec, batch, hi_eff) / 3600, tokens / decode_tok_s(pb, be, prec, batch, lo_eff) / 3600
        out["systems"][name] = {"languages": nl, "output_tokens": tokens, "gpu_hours": [t_lo, t_hi], "weights_gb_bf16": pb * 2.0}
        tot_lo += t_lo
        tot_hi += t_hi
    triples = sentences * min(comet_languages, 44) * len(systems)
    lo = COMET["pairs_s_5070"][0] * COMET["speedup_vs_5070"][0] / COMET["ref_token_factor"]
    hi = COMET["pairs_s_5070"][1] * COMET["speedup_vs_5070"][1] / COMET["ref_token_factor"]
    out["africomet"] = {"triples": triples, "gpu_hours": [triples / hi / 3600, triples / lo / 3600]}
    out["total_gpu_hours"] = [tot_lo + out["africomet"]["gpu_hours"][0], tot_hi + out["africomet"]["gpu_hours"][1]]
    out["note"] = "decoding only plus AfriCOMET; add model download / load time (weights of the 27B system alone are 55 GB) and 2-4x if HF generate is used instead of CTranslate2 / vLLM"
    return out


def plan_release(n_docs: int, n_langs: int, mean_tokens_per_doc: float, eng_tokens_per_sentence: float, gemma_fertility: float, *, own_models: int = 3,
                 human_rows: Optional[Dict[str, Dict[str, float]]] = None, translator_token_factor: float = 0.75) -> Dict[str, Any]:
    """GPU hours of the 500-document release: translators (two bundles), our model, AfriCOMET-ref, AfriCOMET-QE and the RL reward (QE per chunk: the same sentence pairs as
    the QE metric, so a pair cache halves it), and the human-reference subset (SmolDoc: `human_rows` = {name: {rows, tokens_per_doc, sents_per_doc}}).
    Every translator system is charged for ALL the languages it names (upper bound: the real run needs the best + runner-up per language)."""
    bins = {512: 0.35, 1024: 0.30, 2048: 0.22, 4096: 0.13}
    tpb = {512: mean_tokens_per_doc, 1024: mean_tokens_per_doc, 2048: mean_tokens_per_doc, 4096: mean_tokens_per_doc}      # mean tokens/doc is given directly
    out: Dict[str, Any] = {"n_docs": n_docs, "languages": n_langs}
    for name, systems in (("bundle_full", BUNDLE_FULL), ("bundle_min", BUNDLE_MIN), ("opus_supplement", ("opus-mt-tc-bible-big-eng-mul",))):
        p = plan(n_docs, n_langs, bins, tpb, eng_tokens_per_sentence, gemma_fertility, translator_token_factor=translator_token_factor, systems=systems, own_models=own_models)
        out[name] = {"translator": p["translator"], "translator_hours": [sum(v["gpu_hours"][i] for v in p["translator"].values()) for i in (0, 1)]}
    base = plan(n_docs, n_langs, bins, tpb, eng_tokens_per_sentence, gemma_fertility, systems=("nllb200-3.3b",), own_models=own_models)
    out["english_tokens"], out["sentences_per_language"] = base["english_tokens"], base["sentences_per_language"]
    out["target_gemma_tokens_all_languages"] = base["target_gemma_tokens_per_language"] * n_langs
    out["own_model"], out["africomet_ref"] = base["own_model"], base["africomet"]
    q = COMET["pairs_s_5070"][0] * COMET["speedup_vs_5070"][0], COMET["pairs_s_5070"][1] * COMET["speedup_vs_5070"][1]      # QE: no reference segment
    pairs = base["africomet"]["triples_total"]
    out["africomet_qe"] = {"pairs_total": pairs, "gpu_hours": [pairs / q[1] / 3600, pairs / q[0] / 3600]}
    out["reward_qe"] = {"pairs_total": pairs, "gpu_hours": [pairs / q[1] / 3600, pairs / q[0] / 3600], "cpu_note": "GlotLID, chunk planning, consensus (group mode) and the shaping run on the CPU: about 1 core-second per document (assumption)"}
    out["reward_qe"]["cpu_core_hours"] = n_docs * n_langs * own_models / 3600.0
    hum: Dict[str, Any] = {}
    for name, h in (human_rows or {}).items():
        toks = h["rows"] * h["tokens_per_doc"]
        gen = toks * gemma_fertility * own_models
        pairs_h = h["rows"] * h["sents_per_doc"] * own_models
        ref_lo = COMET["pairs_s_5070"][0] * COMET["speedup_vs_5070"][0] / COMET["ref_token_factor"]
        ref_hi = COMET["pairs_s_5070"][1] * COMET["speedup_vs_5070"][1] / COMET["ref_token_factor"]
        hum[name] = {"rows": h["rows"], "english_tokens": toks, "own_model_tokens": gen, "own_model_hours": [gen / OWN_MODEL["tok_s_high"] / 3600, gen / OWN_MODEL["tok_s_low"] / 3600],
                     "pairs": pairs_h, "africomet_ref_hours": [pairs_h / ref_hi / 3600, pairs_h / ref_lo / 3600], "qe_plus_reward_hours": [2 * pairs_h / q[1] / 3600, 2 * pairs_h / q[0] / 3600]}
    out["human_subset"] = hum
    def tot(bundle):
        lo = out[bundle]["translator_hours"][0] + out["own_model"]["gpu_hours"][0] + out["africomet_ref"]["gpu_hours"][0] + out["africomet_qe"]["gpu_hours"][0] + out["reward_qe"]["gpu_hours"][0]
        hi = out[bundle]["translator_hours"][1] + out["own_model"]["gpu_hours"][1] + out["africomet_ref"]["gpu_hours"][1] + out["africomet_qe"]["gpu_hours"][1] + out["reward_qe"]["gpu_hours"][1]
        for h in hum.values():
            lo += h["own_model_hours"][0] + h["africomet_ref_hours"][0] + h["qe_plus_reward_hours"][0]
            hi += h["own_model_hours"][1] + h["africomet_ref_hours"][1] + h["qe_plus_reward_hours"][1]
        return [lo, hi]
    out["total_hours_full"], out["total_hours_min"] = tot("bundle_full"), tot("bundle_min")
    out["total_hours_full_dedup"] = [out["total_hours_full"][0] - out["reward_qe"]["gpu_hours"][0] * 0.5 - out["africomet_qe"]["gpu_hours"][0] * 0.5,
                                     out["total_hours_full"][1] - out["reward_qe"]["gpu_hours"][1] * 0.5 - out["africomet_qe"]["gpu_hours"][1] * 0.5]
    return out


def plan_release_markdown(r: Dict[str, Any]) -> str:
    f = lambda x: f"{x[0]:.1f} - {x[1]:.1f}"
    L_ = ["| item | volume | GPU hours (estimate) |", "|---|---|---|",
          f"| English set | {r['n_docs']} documents x {r['languages']} languages; {r['english_tokens'] / 1e3:.0f} k English tokens, {r['sentences_per_language']:.0f} sentences per language | |"]
    for key, label in (("bundle_full", "NLLB-3.3B + MADLAD-7B"), ("bundle_min", "NLLB-3.3B + MADLAD-3B"), ("opus_supplement", "optional OPUS tc-bible supplement")):
        for name, v in r[key]["translator"].items():
            L_.append(f"| pseudo-references, {label}: {name} ({v['languages']} languages) | {v['output_tokens_total'] / 1e6:.1f} M tokens | {f(v['gpu_hours'])} |")
        L_.append(f"| **translators, {label}** | | **{f(r[key]['translator_hours'])}** |")
    L_ += [f"| our model, {r['own_model']['models']} checkpoints | {r['own_model']['output_tokens_total'] / 1e6:.0f} M generated tokens | {f(r['own_model']['gpu_hours'])} |",
           f"| AfriCOMET reference based (chunked) | {r['africomet_ref']['triples_total'] / 1e6:.2f} M triples | {f(r['africomet_ref']['gpu_hours'])} |",
           f"| AfriCOMET QE (metric) | {r['africomet_qe']['pairs_total'] / 1e6:.2f} M pairs | {f(r['africomet_qe']['gpu_hours'])} |",
           f"| RL reward (QE per chunk + CPU rules) | {r['reward_qe']['pairs_total'] / 1e6:.2f} M pairs, {r['reward_qe']['cpu_core_hours']:.0f} CPU core-hours | {f(r['reward_qe']['gpu_hours'])} (identical pairs as the QE metric: a pair cache removes one of the two) |"]
    for name, h in r["human_subset"].items():
        L_.append(f"| human subset {name} | {h['rows']:,} (document, language) rows, {h['english_tokens'] / 1e6:.2f} M English tokens | model {f(h['own_model_hours'])} + ref {f(h['africomet_ref_hours'])} + QE and reward {f(h['qe_plus_reward_hours'])} |")
    L_ += [f"| **TOTAL, NLLB-3.3B + MADLAD-7B** | | **{f(r['total_hours_full'])}** ({f(r['total_hours_full_dedup'])} with the QE pair cache) |", f"| **TOTAL, NLLB-3.3B + MADLAD-3B** | | **{f(r['total_hours_min'])}** |"]
    return "\n".join(L_) + "\n"


def plan_markdown(core: Dict[str, Any], ext: Dict[str, Any]) -> str:
    def row(label, p, key):
        lo, hi = p[key]["gpu_hours"]
        return f"{lo:.1f} - {hi:.1f}"
    lines = ["| item | core | extended |", "|---|---|---|",
             f"| documents x languages | {core['docs']} x {core['languages']} | {ext['docs']} x {ext['languages']} |",
             f"| English tokens per language | {core['english_tokens'] / 1e3:.0f} k | {ext['english_tokens'] / 1e3:.0f} k |",
             f"| English sentences per language | {core['sentences_per_language']:.0f} | {ext['sentences_per_language']:.0f} |",
             f"| target Gemma tokens, all languages | {core['target_gemma_tokens_per_language'] * core['languages'] / 1e6:.1f} M | {ext['target_gemma_tokens_per_language'] * ext['languages'] / 1e6:.1f} M |"]
    for name in core["translator"]:
        lines.append(f"| translator {name} ({core['translator'][name]['languages']} languages): output tokens / GPU hours | {core['translator'][name]['output_tokens_total'] / 1e6:.1f} M / "
                     f"{row(name, core['translator'], name)} | {ext['translator'][name]['output_tokens_total'] / 1e6:.1f} M / {row(name, ext['translator'], name)} |")
    lines.append(f"| our model ({core['own_model']['models']} checkpoints): output tokens / GPU hours | {core['own_model']['output_tokens_total'] / 1e6:.1f} M / "
                 f"{core['own_model']['gpu_hours'][0]:.1f} - {core['own_model']['gpu_hours'][1]:.1f} | {ext['own_model']['output_tokens_total'] / 1e6:.1f} M / "
                 f"{ext['own_model']['gpu_hours'][0]:.1f} - {ext['own_model']['gpu_hours'][1]:.1f} |")
    lines.append(f"| AfriCOMET (ref-based) triples / GPU hours | {core['africomet']['triples_total'] / 1e6:.2f} M / {core['africomet']['gpu_hours'][0]:.1f} - {core['africomet']['gpu_hours'][1]:.1f} | "
                 f"{ext['africomet']['triples_total'] / 1e6:.2f} M / {ext['africomet']['gpu_hours'][0]:.1f} - {ext['africomet']['gpu_hours'][1]:.1f} |")
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fertility")
    f.add_argument("--human-dir", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--tokenizer", default=None)
    s = sub.add_parser("se")
    s.add_argument("--predictions", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--n-rep", type=int, default=300)
    s.add_argument("--max-langs", type=int, default=0)
    p = sub.add_parser("plan")
    p.add_argument("--fertility", required=True)
    p.add_argument("--selection", default=None, help="SELECTION_REPORT.json of the source set (mean tokens per sentence, bin mix)")
    p.add_argument("--core", type=int, default=200)
    p.add_argument("--extended", type=int, default=1000)
    p.add_argument("--languages", type=int, default=77)
    p.add_argument("--out", required=True)
    r5 = sub.add_parser("plan500", help="the 500-document release plan (D-44 addendum)")
    r5.add_argument("--fertility", required=True)
    r5.add_argument("--selection", required=True, help="SELECTION_REPORT.json of the 500-document set")
    r5.add_argument("--human-stats", required=True, help="human_doc_stats.json (SmolDoc rows per language)")
    r5.add_argument("--smoldoc-tokens", type=float, default=292.1, help="mean Gemma tokens of a SmolDoc document (measured 292.1)")
    r5.add_argument("--smoldoc-sents", type=float, default=9.32)
    r5.add_argument("--languages", type=int, default=77)
    r5.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "plan500":
        fert = json.load(open(a.fertility))
        sel = json.load(open(a.selection))["set"]
        hs = json.load(open(a.human_stats))
        rows = sum(v["smoldoc"] for v in hs.values())
        stitched = sum(v.get("stitched_1024", 0) for v in hs.values())
        res = plan_release(sel["n"], a.languages, sel["tokens_mean"], sel["tokens_total"] / sel["sentences_total"], fert["summary"]["gemma_ref_per_src"]["mean"],
                           human_rows={"SmolDoc": {"rows": rows, "tokens_per_doc": a.smoldoc_tokens, "sents_per_doc": a.smoldoc_sents},
                                       "SmolDoc stitched to 1024 tokens": {"rows": stitched, "tokens_per_doc": 1072.6, "sents_per_doc": 49.4}})
        res["markdown"] = plan_release_markdown(res)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(res["markdown"])
        return 0
    if a.cmd == "fertility":
        res = measure_fertility(a.human_dir, a.tokenizer)
    elif a.cmd == "se":
        res = se_analysis(a.predictions, n_rep=a.n_rep, max_langs=a.max_langs)
    else:
        fert = json.load(open(a.fertility))
        gf = fert["summary"]["gemma_ref_per_src"]["mean"] if fert.get("summary") else 2.2
        tps = 27.0
        bin_shares = {512: 0.35, 1024: 0.30, 2048: 0.22, 4096: 0.13}
        tokens_per_bin = {512: 520.0, 1024: 1030.0, 2048: 2050.0, 4096: 4000.0}
        if a.selection and os.path.isfile(a.selection):
            sel = json.load(open(a.selection))
            ext_s = sel.get("extended", {})
            if ext_s.get("n") and ext_s.get("sentences_total"):
                tps = ext_s["tokens_total"] / ext_s["sentences_total"]
                bd = ext_s.get("by_bin", {})
                if bd:
                    bin_shares = {int(k): v / ext_s["n"] for k, v in bd.items()}
        core = plan(a.core, a.languages, bin_shares, tokens_per_bin, tps, gf)
        ext = plan(a.extended, a.languages, bin_shares, tokens_per_bin, tps, gf)
        bo = bakeoff_plan(("nllb200-3.3b", "madlad400-7b-mt", "sunflower-qwen3.8-27b", "opus-mt-tc-bible-big-eng-mul", "translategemma-12b-it", "afriqueqwen3.5-9b-50langs-instruct"), gf)
        res = {"core": core, "extended": ext, "bakeoff": bo, "markdown": plan_markdown(core, ext), "inputs": {"gemma_fertility": gf, "english_tokens_per_sentence": tps, "bin_shares": bin_shares}}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps(res.get("summary", res.get("aggregate", res.get("markdown", ""))), indent=1) if isinstance(res.get("summary", res.get("aggregate", "")), dict) else res.get("markdown", ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
