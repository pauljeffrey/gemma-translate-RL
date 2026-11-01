# SPDX-License-Identifier: AGPL-3.0-or-later
"""Does the long-document failure metric see what it should? (D-44) Controlled trials on documents WITH human references: the human translation is damaged in known ways
and the metric (`longdoc.plan_document` + `repetition_loop` + chrF++) must notice.

    python -m gemma_mt.eval.longdoc_validate --human-dir source_set/human --langs hau swh zul kin amh yor --min-sentences 25 --trials 3 --out sysinfo/metric_validation.json

Trials, per document (reference translation = the hypothesis, so the intact document must score perfectly):
  intact      the human translation as it is                                     -> false-alarm rate: coverage < 1, omitted sentences, loop
  drop-k      k = 1, 2, 4 random sentences removed (k <= n/8)                    -> document detection, omitted SHARE (estimated 1 - coverage vs true), omitted count; the
                                                                                    index of the omitted sentence is reported too, but a length-only alignment can place an
                                                                                    omission bead on a neighbour of similar length: it localises worse than it measures
  truncate    the translation cut after 60% / 85% of its sentences (generation budget hit) -> measured coverage vs the true share
  loop        the translation cut at 70% and continued with a repeated 3-word unit (the 'muna muna' failure) -> loop detected, coverage
  copy        3 random sentences replaced by the English source                  -> copied sentences found
  swap        two adjacent sentences exchanged (order error)                      -> reported only (coverage stays 1; chrF++ drops a little)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

from gemma_mt.eval import longdoc as L


def _damage(d: L.RefDoc, rng: random.Random, kind: str, k: int = 0, frac: float = 0.0):
    sents = list(d.ref_sents)
    n = len(sents)
    truth: Dict[str, Any] = {"dropped": [], "copied": [], "loop": False, "kept_share": 1.0}
    if kind == "intact":
        pass
    elif kind == "drop":
        drop = sorted(rng.sample(range(1, n - 1), k))                  # not the first / last sentence: an omitted edge is easier and is a separate case
        truth["dropped"] = drop
        sents = [s for i, s in enumerate(sents) if i not in set(drop)]
    elif kind == "truncate":
        cut = max(1, int(n * frac))
        sents = sents[:cut]
        truth["kept_share"] = sum(len(s) for s in d.ref_sents[:cut]) / max(sum(len(s) for s in d.ref_sents), 1)
    elif kind == "loop":
        cut = max(1, int(n * 0.7))
        unit = " ".join(sents[cut - 1].split()[:3]) or "muna muna muna"
        sents = sents[:cut] + [(unit + " ") * 25]
        truth["loop"] = True
        truth["kept_share"] = sum(len(s) for s in d.ref_sents[:cut]) / max(sum(len(s) for s in d.ref_sents), 1)
    elif kind == "copy":
        idx = sorted(rng.sample(range(1, n - 1), 3))
        for i in idx:
            sents[i] = d.src_sents[i]
        truth["copied"] = idx
    elif kind == "swap":
        i = rng.randrange(1, n - 2)
        sents[i], sents[i + 1] = sents[i + 1], sents[i]
    else:
        raise ValueError(kind)
    return " ".join(sents), truth


def run_trials(docs: Sequence[L.RefDoc], trials: int = 3, seed: int = 0, min_sentences: int = 25) -> Dict[str, Any]:
    by_lang: Dict[str, List[L.RefDoc]] = defaultdict(list)
    for d in docs:
        if len(d.src_sents) >= min_sentences:
            by_lang[d.iso].append(d)
    stats: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    rng = random.Random(seed)
    for iso, ds in sorted(by_lang.items()):
        lm = L.make_length_model([s for d in ds for s in d.src_sents], [s for d in ds for s in d.ref_sents])
        for d in ds:
            for t in range(trials):
                specs = [("intact", 0, 0.0)] if t == 0 else []
                specs += [("drop", k, 0.0) for k in (1, 2, 4) if k <= len(d.src_sents) // 8] + [("truncate", 0, 0.6), ("truncate", 0, 0.85), ("loop", 0, 0.0), ("copy", 0, 0.0), ("swap", 0, 0.0)]
                for kind, k, frac in specs:
                    name = kind if kind in ("intact", "loop", "copy", "swap") else (f"drop{k}" if kind == "drop" else f"truncate{int(frac * 100)}")
                    text, truth = _damage(d, rng, kind, k, frac)
                    p = L.plan_document(d.src_sents, d.ref_sents, text, lm)
                    lp = L.repetition_loop(text)
                    r = stats[name]
                    r["n"] += 1
                    r["coverage_sum"] += p.coverage_chars
                    r["kept_share_sum"] += truth["kept_share"]
                    r["flag_doc"] += int(p.coverage_chars < 0.999 or bool(p.omitted_idx) or bool(p.short_idx))
                    r["loop_detected"] += int(lp["loop"])
                    if truth["dropped"]:
                        got, ref_ = set(p.omitted_idx), set(truth["dropped"])
                        near = set(p.omitted_idx) | set(p.short_idx)
                        true_share = sum(len(d.src_sents[i]) for i in ref_) / max(sum(len(x) for x in d.src_sents), 1)
                        est_share = 1.0 - p.coverage_chars
                        r["drop_true"] += len(ref_)
                        r["drop_found_exact"] += len(got & ref_)
                        r["drop_found_or_short"] += len(ref_ & near)
                        r["flag_pred"] += len(got)
                        r["flag_pred_correct"] += len(got & ref_)
                        r["true_share_sum"] += true_share
                        r["est_share_sum"] += est_share
                        r["share_abs_err_sum"] += abs(est_share - true_share)
                        r["count_est_sum"] += p.omitted_sents + p.short_beads
                    if truth["copied"]:
                        r["copy_true"] += len(truth["copied"])
                        r["copy_found"] += len(set(p.omitted_idx) & set(truth["copied"]))
    out: Dict[str, Any] = {"languages": sorted(by_lang), "docs": sum(len(v) for v in by_lang.values()), "trials_per_doc": trials, "min_sentences": min_sentences, "cases": {}}
    for name, r in sorted(stats.items()):
        n = r["n"]
        c = {"n": int(n), "doc_flagged_rate": r["flag_doc"] / n, "mean_coverage_chars": r["coverage_sum"] / n, "loop_detected_rate": r["loop_detected"] / n}
        if name.startswith("truncate") or name == "loop":
            c["mean_true_kept_share"] = r["kept_share_sum"] / n
        if r["drop_true"]:
            c.update({"true_omitted_share": r["true_share_sum"] / n, "estimated_omitted_share": r["est_share_sum"] / n, "omitted_share_abs_error": r["share_abs_err_sum"] / n,
                      "omitted_count_estimated_per_doc": r["count_est_sum"] / n, "omitted_count_true_per_doc": r["drop_true"] / n,
                      "sentence_index_recall_exact": r["drop_found_exact"] / r["drop_true"], "sentence_index_recall_with_shortfall": r["drop_found_or_short"] / r["drop_true"],
                      "sentence_index_precision": (r["flag_pred_correct"] / r["flag_pred"]) if r["flag_pred"] else None})
        if r["copy_true"]:
            c["copy_recall"] = r["copy_found"] / r["copy_true"]
        out["cases"][name] = c
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--human-dir", required=True)
    ap.add_argument("--langs", nargs="*", default=None)
    ap.add_argument("--min-sentences", type=int, default=25)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--max-docs-per-lang", type=int, default=60)
    ap.add_argument("--kinds", nargs="*", default=["human_stitched", "human"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    docs_p = os.path.join(a.human_dir, "docs_human.jsonl")
    docs: List[L.RefDoc] = []
    for fn in sorted(os.listdir(a.human_dir)):
        if not fn.startswith("refs_human_"):
            continue
        iso = fn[len("refs_human_"):-len(".jsonl")]
        if a.langs and iso not in a.langs:
            continue
        rds = [d for d in L.load_refdocs(docs_p, os.path.join(a.human_dir, fn)) if d.ref_kind in a.kinds and len(d.src_sents) >= a.min_sentences]
        docs += rds[: a.max_docs_per_lang]
    res = run_trials(docs, a.trials, a.seed, a.min_sentences)
    res["token_stats"] = {"mean_sentences": sum(len(d.src_sents) for d in docs) / max(len(docs), 1)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res["cases"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
