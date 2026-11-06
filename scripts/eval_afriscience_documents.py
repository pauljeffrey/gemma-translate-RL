#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""FULL document-level evaluation on AfriScience-MT test (owner 2026-11-04): 28 papers x 6 languages = 168 documents, a document = one paper's sentences joined by single spaces in sentence_id order
(scripts/eval_afriscience.load_corpus). Two modes per model:

  whole     the whole document in ONE generation (greedy, --max-new-doc new tokens; the paper's document-level setting)
  chunked   every sentence translated on its own (greedy); each UNFINISHED sentence (no end token) re-decoded by best-of-4 (T 0.7, pick = highest mean sentence chrF++ to the other FINISHED candidates,
            the greedy output is kept when no candidate finishes); the sentences of a paper are joined by single spaces  (D-50 of the first research: reports/longdoc_eval_final_model_2026-10-21.md)

    python scripts/eval_afriscience_documents.py --model DIR --tokenizer DIR --name NAME --out OUT --vllm-python /venv/main/bin/python [--vllm-gpu-util 0.09] [--limit-docs N]

Writes OUT/documents_eval.json (scores) and OUT/documents_predictions.json (per language and mode: paper, src, hyp, ref, finished, and for `chunked` the sentence lists). Text metrics here: corpus chrF (word order 0, the
paper's chrF), chrF++, BLEU, spBLEU, length ratio (characters), finished rate, loop rate (rl/guards repetition detectors), tail coverage (chrF recall of the LAST third of the reference against the hypothesis:
does the translation reach the end of the paper?), sentence coverage (chunked: share of sentences with a non-empty hypothesis). The neural metrics (SSA-COMET-MTL, AfriCOMET, reference-based) are
computed afterwards by scripts/eval_afriscience_documents_neural.py from the predictions file.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_afriscience as EA  # noqa: E402


def loop_flag(text: str) -> bool:
    from gemma_mt.rl import guards

    return bool(guards.is_repetitive(text) or guards.is_self_repetitive(text))


def tail_coverage(hyp: str, ref: str) -> float:
    from gemma_mt.rl import science_terms as st

    n = len(ref)
    return st.chrf_recall_precision(hyp, ref[2 * n // 3:])[0] if n > 30 else 1.0


def mode_scores(rows_by_lang: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    per = {}
    for iso, rows in rows_by_lang.items():
        m = EA.text_metrics([r["hyp"] for r in rows], [r["ref"] for r in rows])
        m["finished_rate"] = sum(bool(r["finished"]) for r in rows) / len(rows)
        m["loop_rate"] = sum(loop_flag(r["hyp"]) for r in rows) / len(rows)
        m["tail_coverage"] = sum(tail_coverage(r["hyp"], r["ref"]) for r in rows) / len(rows)
        m["empty_rate"] = sum(not r["hyp"].strip() for r in rows) / len(rows)
        if "sent_hyps" in rows[0]:
            m["sentence_coverage"] = sum(sum(bool(h.strip()) for h in r["sent_hyps"]) / max(len(r["sent_hyps"]), 1) for r in rows) / len(rows)
            m["repaired_sentences"] = sum(r["repaired"] for r in rows)
            m["unfinished_after_repair"] = sum(r["unfinished_after"] for r in rows)
        per[iso] = m
    return {"per_language": per, "macro": EA.macro(per)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True); ap.add_argument("--tokenizer", default=None); ap.add_argument("--name", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--vllm-python", default=None); ap.add_argument("--vllm-gpu-util", type=float, default=0.09); ap.add_argument("--vllm-max-model-len", type=int, default=9216)
    ap.add_argument("--vllm-max-num-seqs", type=int, default=24); ap.add_argument("--max-new-doc", type=int, default=4096); ap.add_argument("--max-new-sentence", type=int, default=512)
    ap.add_argument("--sampler", choices=("vllm", "hf"), default="vllm"); ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--corpus-dir", default=None); ap.add_argument("--languages", default=",".join(EA.LANGS)); ap.add_argument("--split", default="test"); ap.add_argument("--limit-docs", type=int, default=0)
    ap.add_argument("--best-of", type=int, default=4); ap.add_argument("--repair-temperature", type=float, default=0.7)
    ap.add_argument("--extra-libs", default="/workspace/eval_385m/pylibs"); a = ap.parse_args(argv)

    from gemma_mt.eval import runtime as RT

    RT.add_extra_libs(a.extra_libs)
    os.makedirs(a.out, exist_ok=True)
    corpus_dir = a.corpus_dir
    if not corpus_dir:
        from huggingface_hub import snapshot_download

        corpus_dir = os.path.join(snapshot_download("dsfsi/afriscience_mt", repo_type="dataset", allow_patterns=["corpus/*", "README.md"]), "corpus")
    corpus = EA.load_corpus(corpus_dir, a.split, [x for x in a.languages.split(",") if x])
    if a.limit_docs:
        for iso in corpus:
            keep = {d["paper"] for d in corpus[iso]["documents"][: a.limit_docs]}
            corpus[iso]["documents"] = corpus[iso]["documents"][: a.limit_docs]
            corpus[iso]["sentences"] = [s for s in corpus[iso]["sentences"] if s["paper"] in keep]
    ref = RT.resolve_model(a.model)
    tok, stop, pad = RT.load_tokenizer_and_stops(a.tokenizer or ref.path)
    dec = EA.parse_decoding("greedy")
    RT.log(f"{a.name}: documents {sum(len(c['documents']) for c in corpus.values())}, sentences {sum(len(c['sentences']) for c in corpus.values())}")
    t0 = time.time(); sampler = None; model = None
    if a.sampler == "vllm":
        RT.wait_for_gpu(a.vllm_gpu_util * 96.0 * 0.9, 600)
        sampler = RT.make_vllm_sampler(ref.path, python=a.vllm_python, gpu_util=a.vllm_gpu_util, max_model_len=a.vllm_max_model_len, max_num_seqs=a.vllm_max_num_seqs,
                                       log_path=os.path.join(a.out, "vllm_worker.log"))

        def gen(items, mnt):
            return EA.generate(sampler, tok, stop.ids, pad, items, dec, mnt, chunk=256, log=RT.log)
    else:
        import torch

        from gemma_mt.eval.eval_translation import generate_translations
        from gemma_mt.modeling import load_model

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_model(ref.path, attn_implementation="sdpa", dtype=torch.bfloat16 if device == "cuda" else torch.float32, device=device)

        def gen(items, mnt):
            rows = [{"english": e, "iso": iso} for e, iso in items]
            return generate_translations(model, tok, rows, stop_ids=stop.ids, pad_id=pad, device=device, batch_size=a.batch_size, max_new_tokens=mnt, log=RT.log)

    preds: Dict[str, Any] = {"whole": {}, "chunked": {}}
    try:
        # ---- mode whole
        order = [(iso, i) for iso in corpus for i in range(len(corpus[iso]["documents"]))]
        outs = gen([(corpus[iso]["documents"][i]["src"], iso) for iso, i in order], a.max_new_doc)
        for (iso, i), o in zip(order, outs):
            d = corpus[iso]["documents"][i]
            preds["whole"].setdefault(iso, []).append({"paper": d["paper"], "src": d["src"], "ref": d["ref"], "hyp": o["hyp"], "finished": bool(o["finished"])})
        RT.log(f"whole documents done ({time.time() - t0:.0f}s)")
        # ---- mode chunked
        sorder = [(iso, i) for iso in corpus for i in range(len(corpus[iso]["sentences"]))]
        souts = gen([(corpus[iso]["sentences"][i]["src"], iso) for iso, i in sorder], a.max_new_sentence)
        unfinished = [k for k, o in enumerate(souts) if not o["finished"]]
        repaired = 0
        if unfinished and sampler is not None:
            from gemma_mt import format as fmt
            from gemma_mt.rl import consensus as cons
            from gemma_mt.rl.text import decode_completion

            prompts = [fmt.build_prompt(corpus[sorder[k][0]]["sentences"][sorder[k][1]]["src"], sorder[k][0], tok) for k in unfinished]
            roll = sampler.generate(prompts, group_size=a.best_of, max_new_tokens=a.max_new_sentence, temperature=a.repair_temperature, top_p=1.0, top_k=0, stop_ids=stop.ids, pad_id=pad, greedy=False, key=7)
            for j, k in enumerate(unfinished):
                cands = []
                for ids in roll.completions[j * a.best_of: (j + 1) * a.best_of]:
                    d = decode_completion(tok, ids, stop.ids)
                    if d.finished:
                        cands.append(fmt.strip_iso_prefix(d.text, sorder[k][0]).text)
                if cands:
                    best = max(range(len(cands)), key=lambda x: sum(cons.chrf_sentence(cands[x], cands[y]) for y in range(len(cands)) if y != x) / max(len(cands) - 1, 1)) if len(cands) > 1 else 0
                    souts[k] = dict(souts[k], hyp=cands[best], finished=True); repaired += 1
        RT.log(f"chunked sentences done: {len(souts)} sentences, {len(unfinished)} unfinished, {repaired} repaired ({time.time() - t0:.0f}s)")
        by_paper: Dict[str, Dict[str, Any]] = {}
        for (iso, i), o in zip(sorder, souts):
            s = corpus[iso]["sentences"][i]
            e = by_paper.setdefault(f"{iso}|{s['paper']}", {"src_sents": [], "ref_sents": [], "sent_hyps": [], "unfinished_after": 0, "repaired": 0})
            e["src_sents"].append(s["src"]); e["ref_sents"].append(s["ref"]); e["sent_hyps"].append(o["hyp"]); e["unfinished_after"] += int(not o["finished"])
        rep_set = set(unfinished)
        for k in unfinished:
            iso, i = sorder[k]
            if souts[k]["finished"]: by_paper[f"{iso}|{corpus[iso]['sentences'][i]['paper']}"]["repaired"] += 1
        for iso in corpus:
            for d in corpus[iso]["documents"]:
                e = by_paper[f"{iso}|{d['paper']}"]
                preds["chunked"].setdefault(iso, []).append({"paper": d["paper"], "src": d["src"], "ref": d["ref"], "hyp": " ".join(e["sent_hyps"]), "finished": e["unfinished_after"] == 0, **e})
    finally:
        if sampler is not None:
            sampler.close()
    res = {"model": a.model, "name": a.name, "split": a.split, "max_new_doc": a.max_new_doc, "documents": sum(len(v) for v in preds["whole"].values()),
           "scores": {m: mode_scores(preds[m]) for m in ("whole", "chunked")}, "seconds": round(time.time() - t0, 1)}
    json.dump(res, open(os.path.join(a.out, "documents_eval.json"), "w"), indent=1, ensure_ascii=False)
    json.dump(preds, open(os.path.join(a.out, "documents_predictions.json"), "w"), ensure_ascii=False)
    for m in ("whole", "chunked"):
        mac = res["scores"][m]["macro"]
        print(m, {k: round(v, 3) for k, v in mac.items() if k in ("chrf", "chrf_pp", "bleu", "finished_rate", "loop_rate", "length_ratio", "tail_coverage")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
