#!/usr/bin/env python
"""Step 1: per-sentence sacrebleu statistics for every stored prediction file (our models), recomputed from the hypotheses.

For every row (benchmark, iso, id) of a predictions.jsonl we store
  chrf_st[24]  CHRF(word_order=2) segment statistics of NFC(hyp) vs NFC(reference): 6 char orders then 2 word orders, [hyp, ref, match] each
               chrF++ = f_score(sum of 24), plain chrF = f_score(sum of the first 18)
  bleu_st[10]  BLEU(tokenize='flores200') segment statistics (spBLEU)
  finished     the `finished` flag written by the generation harness
The first row of a duplicated key is kept (the 385M SFT file has one duplicated key, SmolSent afr).
Usage: python s01_rescore.py CACHE_DIR [SYSTEM ...]      (python with sacrebleu 2.6.0 + sentencepiece; CPU only; all systems when none is named)
"""
import json, os, sys, unicodedata, time
from multiprocessing import Pool
import numpy as np
from sacrebleu.metrics import CHRF, BLEU

ROOT = "/path/to/repo/research_archive_2026-11-03/"
SYSTEMS = {
    "sft385":  ROOT + "eval_385m/out_m385/predictions.jsonl",
    "stageA":  [ROOT + "eval_385m_rl_predictions_2026-11-04/out_stageA/predictions.jsonl",      # FLORES+ devtest
                ROOT + "eval_385m_rl_predictions_2026-11-04/out_stageA_x/predictions.jsonl"],   # NTREX + MAFAND
    "b1best":  [ROOT + "eval_385m_rl_predictions_2026-11-04/out_b1best/predictions.jsonl",
                ROOT + "eval_385m_rl_predictions_2026-11-04/out_b1best_x/predictions.jsonl"],
    "mixed":   ROOT + "eval_385m_rl_predictions_2026-11-04/out_wk_mixedbest/predictions.jsonl",
    "e03c1":   ROOT + "box_final_2026-11-04/e03_control/run_records_and_eval/c1_s6000_predictions.jsonl",
    "old_sft": ROOT + "box_outputs_2026-11-04/out_old_sft/predictions.jsonl",
    "old_rl":  ROOT + "box_outputs_2026-11-04/out_old_rl2c/predictions.jsonl",
}
nfc = lambda s: unicodedata.normalize("NFC", s)


def run(args):
    name, path, cache = args
    out = os.path.join(cache, name + ".npz")
    if os.path.exists(out):
        return name, "cached"
    t0 = time.time()
    seen, rows = set(), []
    for pth in ([path] if isinstance(path, str) else path):
        for line in open(pth, encoding="utf-8"):
            r = json.loads(line)
            k = (r["benchmark"], r["iso"], str(r["id"]))
            if k in seen:
                continue
            seen.add(k)
            rows.append((k, nfc(r["hyp"]), nfc(r["reference"]), str(r.get("finished")).lower() == "true"))
    chrf, bleu = CHRF(word_order=2), BLEU(tokenize="flores200")
    hyps = [x[1] for x in rows]
    refs = [x[2] for x in rows]
    cs = np.array(chrf._extract_corpus_statistics(hyps, [refs]), dtype=np.int32)
    bs = np.array(bleu._extract_corpus_statistics(hyps, [refs]), dtype=np.int32)
    np.savez_compressed(out, system=name, bench=np.array([x[0][0] for x in rows]), iso=np.array([x[0][1] for x in rows]),
                        rid=np.array([x[0][2] for x in rows]), chrf_st=cs, bleu_st=bs, finished=np.array([x[3] for x in rows]))
    return name, "%d rows, %.0f s" % (len(rows), time.time() - t0)


if __name__ == "__main__":
    cache = sys.argv[1]
    os.makedirs(cache, exist_ok=True)
    todo = {n: p_ for n, p_ in SYSTEMS.items() if len(sys.argv) < 3 or n in sys.argv[2:]}
    with Pool(len(todo)) as p:
        for res in p.imap_unordered(run, [(n, pth, cache) for n, pth in todo.items()]):
            print(res, flush=True)
