#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fetch AfriScience-MT (dsfsi/afriscience_mt, Apache-2.0) and derive what the science RL run needs (DECISIONS.md D-52).

    python scripts/prepare_afriscience.py --out DIR

Downloads `corpus/*` (train / dev / test, 4.6 MB) and `predictions/metrics-*` (the paper's released per-run metrics, 0.14 MB) - NOT the 260 MB of per-sentence predictions - then writes
  DIR/bench_afrisci_dev_sent.jsonl   dev sentences: english, reference, iso, benchmark, id (the format of `benchmark_eval_path`: selection of the best checkpoint by chrF++ on the DEV split)
  DIR/bench_afrisci_dev_doc.jsonl    dev papers (sentences joined by spaces), the same format (monitor-only `benchmark_aux`)
  DIR/published_chrf_test_eng2x.csv  system, iso, chrf, bleu, ssa_comet: English -> X, test split, main (non-ablation) runs of the paper's systems
The TEST split is never written to a benchmark file used during training.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

# (label, model_short, experiment_type, prompt_strategy, lora_rank) of the systems shown beside our numbers
SYSTEMS = [
    ("NLLB-200-1.3B fine-tuned", "nllb_200_1_3b", "seq2seq_finetune", None, None),
    ("NLLB-200-distilled-600M fine-tuned", "nllb_200_distilled_600m", "seq2seq_finetune", None, None),
    ("M2M100-1.2B fine-tuned", "m2m100_1_2b", "seq2seq_finetune", None, None),
    ("NLLB-200-1.3B zero-shot", "nllb_200_1_3b", "zero_shot", None, None),
    ("Gemma2-9B LoRA r64 fine-tuned", "gemma2_9b_it", "lora_finetune", None, 64),
    ("GPT-5.4 10-shot", "gpt54", "llm_icl", "template5_10shot", None),
    ("Gemini-3.1-Flash-Lite 10-shot", "gemini_31_flash_lite", "llm_icl", "template5_10shot", None),
    ("doc: GPT-5.4 1-shot", "gpt54", "llm_doc", "doc_1shot", None),
    ("doc: GPT-5.4 0-shot", "gpt54", "llm_doc", "doc_0shot", None),
    ("doc: Gemini-3.1-Flash-Lite 1-shot", "gemini_31_flash_lite", "llm_doc", "doc_1shot", None),
    ("doc: TranslateGemma-12B 1-shot", "translategemma_12b", "llm_doc", "doc_1shot", None),
    ("doc: Gemma2-9B-IT 1-shot", "gemma2_9b_it", "llm_doc", "doc_1shot", None),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-download", action="store_true", help="DIR already holds corpus/ and predictions/metrics-*.parquet")
    a = ap.parse_args(argv)
    import pandas as pd

    os.makedirs(a.out, exist_ok=True)
    if not a.no_download:
        from huggingface_hub import snapshot_download

        snapshot_download("dsfsi/afriscience_mt", repo_type="dataset", allow_patterns=["corpus/*", "predictions/metrics-*", "glossary/*", "README.md"], local_dir=a.out, token=os.environ.get("HF_TOKEN"))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import eval_afriscience as E

    corpus = E.load_corpus(os.path.join(a.out, "corpus"), "dev")
    for kind, key in (("sent", "sentences"), ("doc", "documents")):
        rows = []
        for iso, blk in corpus.items():
            for i, r in enumerate(blk[key]):
                rows.append({"english": r["src"], "reference": r["ref"], "iso": iso, "benchmark": f"afrisci_dev_{kind}", "id": f"{r['paper']}-{r.get('sid', 'doc')}"})
        p = os.path.join(a.out, f"bench_afrisci_dev_{kind}.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{p}: {len(rows)} rows")
    m = pd.read_parquet(os.path.join(a.out, "predictions", "metrics-00000.parquet"))
    base = m[(m.dataset == "afriscience_mt") & (m.split == "test") & (m.source_lang == "eng")]
    out = []
    for label, ms, et, ps, rank in SYSTEMS:
        x = base[(base.model_short == ms) & (base.experiment_type == et)]
        x = x[x.prompt_strategy.isna()] if ps is None else x[x.prompt_strategy == ps]
        if rank is not None:
            x = x[x.lora_rank == rank]
        x = x[~x.temp_setting.isin(["t06", "t6"])]
        # the main run of a system is the non-ablation one; the sentence-level ICL runs are flagged as ablation in the release (the paper's Table 3 uses them): take non-ablation if it exists
        y = x[~x.is_ablation]
        x = y if len(y) else x
        for iso, g in x.groupby("target_lang"):
            g = g.sort_values("timestamp").iloc[-1]
            out.append({"system": label, "iso": iso, "chrf": round(float(g.chrf), 3), "bleu": round(float(g.bleu), 3), "ssa_comet": round(float(g.ssa_comet), 4)})
    p = os.path.join(a.out, "published_chrf_test_eng2x.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["system", "iso", "chrf", "bleu", "ssa_comet"])
        w.writeheader()
        w.writerows(out)
    print(f"{p}: {len(out)} rows, systems {sorted({r['system'] for r in out})}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
