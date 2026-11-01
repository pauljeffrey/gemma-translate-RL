#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""CPU smoke test of the whole long-document pipeline with a SMALL public model (D-44): stitched FLORES+ dev passages with human references are translated
(a) sentence by sentence (the pseudo-reference protocol) and (b) as one whole document (what a long-context system does), both by the same model, and (b) is scored
against the human reference AND against (a). Prints chrF++/BLEU/AfriCOMET (optional), coverage, loops, length ratio. Default model: Helsinki-NLP/opus-mt-en-ig (Igbo, 75 M).

    python scripts/longdoc_smoke_cpu.py --iso ibo --n-docs 10 --target-tokens 512 --out /workspace/sft_build/longdoc_bench/bakeoff/smoke_ibo
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gemma_mt.eval import longdoc as L  # noqa: E402
from gemma_mt.eval import longdoc_backends as BE  # noqa: E402
from gemma_mt.eval import longdoc_bakeoff as BO  # noqa: E402
from gemma_mt.eval import longdoc_refs as R  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iso", default="ibo")
    ap.add_argument("--hf-id", default="Helsinki-NLP/opus-mt-en-ig")
    ap.add_argument("--code", default="ig")
    ap.add_argument("--prompt-style", default="marian")
    ap.add_argument("--n-docs", type=int, default=10)
    ap.add_argument("--target-tokens", type=int, default=512)
    ap.add_argument("--dev-dir", default="/workspace/sft_build/longdoc_bench/raw/flores_dev")
    ap.add_argument("--bench-root", default=R.DEFAULT_BENCH_ROOT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--comet", default="none", choices=["none", "fake", "worker"])
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--pythonpath", default=None)
    ap.add_argument("--n-boot", type=int, default=200)
    a = ap.parse_args(argv)
    count = L.load_token_counter()
    parts = R.flores_passage_docs(a.iso, "dev", a.bench_root, a.dev_dir, count_tokens=count)
    docs = R.stitch_documents(parts, a.target_tokens, count, salt="smoke")[: a.n_docs]
    spec = BO.get_spec("adhoc", a.hf_id, a.prompt_style, {a.iso: a.code})
    tr = BE.make_translator(spec, "hf", device="cpu", dtype="fp32", num_beams=4, max_new_tokens_cap=512)
    # (a) sentence by sentence = the pseudo reference; (b) the whole document in one call
    pseudo = L.translate_documents([{"doc_id": d.doc_id, "text": d.source_text, "sents": d.src_sents, "para_starts": d.para_starts} for d in docs], tr.fn_for(a.iso), iso=a.iso, system=a.hf_id, batch_size=16)
    whole = {d.doc_id: L.Hyp(d.doc_id, a.iso, tr.translate([d.source_text], a.iso)[0], None, a.hf_id) for d in docs}
    scorer = None if a.comet == "none" else L._make_scorer(a.comet, a.python, a.pythonpath, True, 16)
    res_h = L.evaluate_language(docs, whole, iso=a.iso, scorer_ref=scorer, n_boot=a.n_boot, keep_rows=True)
    pseudo_docs = [L.RefDoc(d.doc_id, a.iso, d.source, d.src_sents, p.tgt_sents, "pseudo", a.hf_id, d.domain, d.bin, d.para_starts, p.flags) for d, p in zip(docs, pseudo)]
    res_p = L.evaluate_language(pseudo_docs, whole, iso=a.iso, scorer_ref=scorer, n_boot=a.n_boot, keep_rows=True)
    # how good is the sentence-by-sentence translation itself against the HUMAN reference (the pseudo-reference ceiling)
    seg = {d.doc_id: L.Hyp(d.doc_id, a.iso, p.text, True) for d, p in zip(docs, pseudo)}
    res_s = L.evaluate_language(docs, seg, iso=a.iso, scorer_ref=scorer, n_boot=a.n_boot, keep_rows=False)
    brief = lambda r: {k: (r[k]["value"] if isinstance(r.get(k), dict) and "value" in r[k] else r.get(k)) for k in
                       ("n_docs", "chrf", "bleu", "africomet_ref", "coverage_chars", "loop_rate", "finished_rate", "ends_cleanly_rate", "docs_with_omission")} | {"length_ratio_mean": r["length_ratio"]["mean"]}
    out = {"model": a.hf_id, "iso": a.iso, "docs": len(docs), "mean_source_tokens": sum(count(d.source) for d in docs) / max(len(docs), 1),
           "A_sentence_by_sentence_vs_human": brief(res_s), "B_whole_doc_vs_human": brief(res_h), "C_whole_doc_vs_sentence_by_sentence_pseudo": brief(res_p)}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "smoke.json"), "w", encoding="utf-8") as f:
        json.dump({**out, "rows_B": res_h["rows"], "rows_C": res_p["rows"]}, f, ensure_ascii=False, indent=1)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
