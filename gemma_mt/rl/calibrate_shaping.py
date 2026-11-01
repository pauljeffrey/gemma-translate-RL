# SPDX-License-Identifier: AGPL-3.0-or-later
"""Calibrate the per-language shaping anchors with AfriCOMET-QE on HUMAN translations (run once, on the GPU box).

    python -m gemma_mt.rl.calibrate_shaping --bench flores_plus:/data/flores_plus --split dev \\
        --out calibration/shaping.json [--model runs/sft/best] [--max-per-language 500]

Ported from the repo's rlhf/scripts/calibrate_shaping.py, keyed by target language. For every language it scores a
sample (<= --max-per-language) of (english source, human reference) with the reference-free QE model and writes

  anchors      [t_fail, t_mid, t_great]   t_mid = P50, t_great = P90 of QE(source, human reference);
               t_fail = max(0.46, P90 of QE on MISMATCHED references)   (a fluent wrong sentence must not pass)
  human_quantiles_p10_p50_p90
  auc_vs_mismatched / auc_vs_copy / auc_vs_half_omitted   how well QE separates the human reference from a
               reference of another sentence / a copy of the source / the reference with half its words removed
  usable       auc_vs_mismatched >= --min-auc (default 0.7): otherwise QE cannot see quality in this language and
               the RL stage gives the language advantage weight 0
  reliability  linear in AUC between 0.65 (0) and 0.90 (1): the advantage weight
  len_ratio    p01 / p02 / p05 / p50 / p95 of len(reference) / len(source) in characters, plus `corpus` (total reference chars / total source chars)
               and `s2` (Gale-Church variance of the length difference per source character): the length window of the penalties and the
               language-aware length model of the omission detector (D-30)
  diacritics   (rl/diacritics.py, D-31) the human diacritic density distribution (median / P10 / P99), whether the language uses diacritics
               (median >= 0.008), the effective density threshold, and the inventory of marks / special letters / (letter, marks) clusters the
               human references use (widened by other dev sets of the language, e.g. MAFAND dev)
  lexical      per feature (numbers / names / content words shared with the source, rl/lexical.py) the human P10 / P50, kept only
               where human references carry the overlap; the reward penalises falling below P10 (reward.lexical_penalty)
  model        (with --model) the same statistics for the SFT model's greedy translations and the share below t_fail

Use the file with  reward.calibration_path  in configs/rl.yaml.

FLORES+ DEV (D-32): `--bench-file calibration/flores_dev_all.jsonl` (scripts/build_flores_dev_all.py; rows carry `benchmark` = flores_plus_dev or
mafand_dev) with `--max-per-language 997 --fail-floor 0.0`. Rows of benchmark `mafand_dev` of a language that also has FLORES+ rows only widen its
diacritic inventory; a language with MAFAND rows only (pcm) is calibrated on them. A RL language with no rows at all (efi) gets an entry
{usable: false, reason: no reference data}: default anchors, advantage weight 0. `--resume` skips the languages already in --out (the file is
rewritten after every language); `--gpu-mem-gb` caps this process's GPU memory (a shared card). Calibration uses DEV data only (never devtest).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence

import unicodedata

from gemma_mt.rl import diacritics, lexical
from gemma_mt.rl.omission_eval import length_stats
from gemma_mt.rl.reward_shaping import FAIL_FLOOR, auc, compute_anchors, pair_anchors, reliability_from_auc, shape


def calibrate(rows: Sequence[Dict[str, str]], scorer: Callable, translate: Optional[Callable] = None,
              max_per_language: int = 500, min_auc: float = 0.7, seed: int = 0, fail_floor: float = FAIL_FLOOR,
              no_data_languages: Sequence[str] = (), done: Optional[Dict[str, Dict[str, Any]]] = None,
              on_language: Optional[Callable[[str, Dict[str, Any]], None]] = None) -> Dict[str, Dict[str, Any]]:
    """rows: english, reference, iso (+ benchmark). scorer(srcs, mts) -> QE scores. translate(rows) -> model translations.
    Text is normalised to NFC first (the reward path does the same, D-29). `done`: entries to keep (resume). `no_data_languages`: RL languages
    without any reference set: written as unusable (default anchors, weight 0)."""
    rng = random.Random(seed)
    by: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    inv_extra: Dict[str, List[str]] = defaultdict(list)
    for r in rows:
        if r.get("english") and r.get("reference") and r.get("iso"):
            r = dict(r, english=unicodedata.normalize("NFC", r["english"]), reference=unicodedata.normalize("NFC", r["reference"]))
            by[r["iso"]].append(r)
    for iso in list(by):                                           # other dev sets only widen the diacritic inventory when FLORES+ is there
        main = [r for r in by[iso] if r.get("benchmark") != "mafand_dev"]
        if main and len(main) != len(by[iso]):
            inv_extra[iso] = [r["reference"] for r in by[iso] if r.get("benchmark") == "mafand_dev"]
            by[iso] = main
    out: Dict[str, Dict[str, Any]] = dict(done or {})
    for iso in no_data_languages:
        if iso not in by and iso not in out:
            out[iso] = {"usable": False, "reliability": 0.0, "reliability_same_source": 0.0, "n": 0, "anchors": [fail_floor, 0.60, 0.75],
                        "reason": "no human reference set (not in FLORES+ / MAFAND dev): default anchors, advantage weight 0"}
    for iso, rs in sorted(by.items()):
        if iso in out:
            continue
        rng.shuffle(rs)
        rs = rs[:max_per_language]
        if len(rs) < 3:
            continue
        src = [r["english"] for r in rs]
        ref = [r["reference"] for r in rs]
        human = scorer(src, ref)
        mism = scorer(src, ref[1:] + ref[:1])                       # the reference of ANOTHER sentence
        copy = scorer(src, src)                                     # "translating" by copying the source
        half = scorer(src, [" ".join(r.split()[: max(1, len(r.split()) // 2)]) for r in ref])   # omission
        append_txt = [r + " " + ref[(k + 1) % len(ref)] for k, r in enumerate(ref)]
        append = scorer(src, append_txt)                              # hallucinated / appended unrelated sentence
        ratios = sorted(len(r) / max(len(s), 1) for s, r in zip(src, ref))
        ls = length_stats(src, ref)
        acc = lambda bad: float(sum(h > b for h, b in zip(human, bad)) / len(human))
        ss = {"mismatch": acc(mism), "copy": acc(copy), "half_omitted": acc(half), "append_unrelated": acc(append)}
        e: Dict[str, Any] = {
            "len_ratio": {"p05": ratios[int(0.05 * (len(ratios) - 1))], "p50": ratios[len(ratios) // 2], "p95": ratios[int(0.95 * (len(ratios) - 1))],
                          "p01": ls["p01"], "p02": ls["p02"], "corpus": ls["corpus"], "s2": ls["s2"]},
            "same_source_accuracy": ss,
            "n": len(rs), "anchors": list(pair_anchors(human, mism, fail_floor)), "fail_floor": fail_floor,
            "anchors_at_floor_0.46": list(pair_anchors(human, mism, FAIL_FLOOR)),
            "human_quantiles_p10_p50_p90": list(compute_anchors(human)),
            "mismatched_p90": float(sorted(mism)[int(0.9 * (len(mism) - 1))]),
            "auc_vs_mismatched": auc(human, mism), "auc_vs_copy": auc(human, copy), "auc_vs_half_omitted": auc(human, half),
            "mean_human": sum(human) / len(human), "mean_mismatched": sum(mism) / len(mism),
            "mean_copy": sum(copy) / len(copy), "mean_half_omitted": sum(half) / len(half),
            "share_human_below_0.46": sum(h < 0.46 for h in human) / len(human),
        }
        if translate is not None:
            mt = translate(rs)
            ms = scorer(src, mt)
            e["model"] = {"anchors": list(compute_anchors(ms)), "mean": sum(ms) / len(ms),
                          "below_t_fail": sum(s < e["anchors"][0] for s in ms) / len(ms)}
        # lexical fidelity (rl/lexical.py): thresholds from the human references, and what the penalty adds on top of QE in the same-source test
        lex = lexical.summarise(zip(src, ref))
        if lex:
            th = {k: d["p10"] for k, d in lex.items()}
            anc = e["anchors"]
            sh = lambda c: shape(c, anc)
            hum_r = [sh(c) - lexical.deficit(lexical.features(s_, r_), th) for c, s_, r_ in zip(human, src, ref)]
            corrupt = {"mismatch": ref[1:] + ref[:1],
                       "append_unrelated": [r + " " + ref[(k + 1) % len(ref)] for k, r in enumerate(ref)]}
            gain = {}
            for name, bad_txt in corrupt.items():
                bad_qe = (append if name == "append_unrelated" else mism)
                bad_r = [sh(c) - lexical.deficit(lexical.features(s_, b_), th) for c, s_, b_ in zip(bad_qe, src, bad_txt)]
                gain[name] = {"qe_only": ss[name], "qe_plus_lexical": float(sum(h > b for h, b in zip(hum_r, bad_r)) / len(hum_r))}
            e["lexical"] = lex
            e["lexical_same_source_accuracy"] = gain
        e["diacritics"] = diacritics.summarise(ref, extra_refs=inv_extra.get(iso, ()))
        e["usable"] = bool(e["auc_vs_mismatched"] >= min_auc)
        e["reliability"] = reliability_from_auc(e["auc_vs_mismatched"]) if e["usable"] else 0.0
        # same-source ranking accuracy on the errors QE can actually be asked about (copies and appended text are handled by guards / the length
        # window): 0.5 = chance -> 0, 0.9 -> 1
        hard = (ss["mismatch"] + ss["half_omitted"]) / 2
        e["reliability_same_source"] = float(min(1.0, max(0.0, (hard - 0.5) / 0.4))) if e["usable"] else 0.0
        out[iso] = e
        if on_language:
            on_language(iso, e)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", action="append", default=[], help="name[:local_path] (flores_plus|ntrex|mafand)")
    ap.add_argument("--bench-file", action="append", default=[], help="parquet/jsonl with english,reference,iso")
    ap.add_argument("--split", default=None, help="benchmark split (use the DEV split, not the one you report on)")
    ap.add_argument("--languages", nargs="*", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None, help="optional: also score this model's greedy translations (generated with vLLM since 2026-10-29: needs --vllm-python)")
    ap.add_argument("--vllm-python", default=None, help="interpreter of the environment that has vllm (required with --model)")
    ap.add_argument("--qe-model", default="masakhane/africomet-qe-stl-1.1")
    ap.add_argument("--max-per-language", type=int, default=500)
    ap.add_argument("--min-auc", type=float, default=0.7)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--fail-floor", type=float, default=FAIL_FLOOR,
                    help="t_fail = max(floor, P90 of QE on mismatched references). 0.46 = the old AfriCOMET-scale floor; 0.0 lets each language's own mismatched "
                         "distribution decide (needed for languages whose human references score below 0.46, D-32)")
    ap.add_argument("--resume", action="store_true", help="keep the languages already present in --out, calibrate the rest (file rewritten after each language)")
    ap.add_argument("--gpu-mem-gb", type=float, default=0.0, help="cap this process's GPU memory (a shared card); 0 = no cap")
    ap.add_argument("--no-data-languages", nargs="*", default=["efi"], help="RL languages without any reference set: written as unusable")
    a = ap.parse_args(argv)

    from gemma_mt.eval import benchmarks as B
    from gemma_mt.rl.comet import CometScorer

    rows: List[Dict[str, str]] = []
    for spec in a.bench:
        name, _, path = spec.partition(":")
        rows += B.load_benchmark(name, path=path or None, isos=a.languages, split=a.split)
    for p in a.bench_file:
        rows += B.load_normalised_file(p)
    if not rows:
        print("no rows", file=sys.stderr)
        return 2
    from gemma_mt.languages import QE_LANGUAGES

    rows = [r for r in rows if r["iso"] in QE_LANGUAGES]
    if a.gpu_mem_gb > 0:
        import torch

        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory / 2**30
            torch.cuda.set_per_process_memory_fraction(min(1.0, a.gpu_mem_gb / total))
    scorer_obj = CometScorer(a.qe_model, batch_size=a.batch_size)
    scorer = lambda s, m: scorer_obj(s, m)
    translate = None
    if a.model:                                    # 2026-10-29 (Amendment 4): evaluation generation goes through vLLM, no HF generate
        if not a.vllm_python:
            ap.error("--model needs --vllm-python (all evaluation generation uses vLLM)")
        from gemma_mt.eval import runtime as RT
        from gemma_mt.eval.eval_translation import generate_translations

        tok, stop, pad = RT.load_tokenizer_and_stops(a.model)
        sampler = RT.make_vllm_sampler(a.model, python=a.vllm_python)

        def translate(rs):
            outs = generate_translations(None, tok, rs, stop_ids=stop.ids, pad_id=pad, device=None, sampler=sampler, require_vllm=True)
            return [o["hyp"] for o in outs]

    done: Dict[str, Dict[str, Any]] = {}
    if a.resume and os.path.exists(a.out):
        with open(a.out, encoding="utf-8") as f:
            done = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
        print(f"resuming: {len(done)} languages already in {a.out}", flush=True)
    meta = {"_meta": {"fail_floor": a.fail_floor, "max_per_language": a.max_per_language, "min_auc": a.min_auc, "qe_model": a.qe_model,
                      "bench_files": a.bench_file, "bench": a.bench, "split": a.split or "dev (bench files must contain DEV rows only)"}}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    def line(iso, e):
        m = e.get("model")
        if "auc_vs_mismatched" not in e:
            return f"{iso:5} n={e['n']:4} UNUSABLE ({e.get('reason')})"
        d = (e.get("diacritics") or {})
        return (f"{iso:5} n={e['n']:4} t_fail/t_mid/t_great={[round(x, 3) for x in e['anchors']]} "
                f"rel={e['reliability']:.2f} auc: mismatched={e['auc_vs_mismatched']:.3f} copy={e['auc_vs_copy']:.3f} "
                f"half-omitted={e['auc_vs_half_omitted']:.3f}"
                + (f" dia={'Y' if d.get('applies') else 'n'}" if d else "")
                + (f" | model P50={m['anchors'][1]:.3f} below_t_fail={m['below_t_fail']:.0%}" if m else "")
                + ("" if e["usable"] else "  <-- QE does not separate real from mismatched here; excluded"))

    def save(iso, e, _res={}):
        _res[iso] = e
        print(line(iso, e), flush=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({**meta, **done, **_res}, f, indent=1)

    res = calibrate(rows, scorer, translate, a.max_per_language, a.min_auc, fail_floor=a.fail_floor,
                    no_data_languages=[i for i in a.no_data_languages if i in QE_LANGUAGES], done=done, on_language=save)
    for iso, e in res.items():
        if "auc_vs_mismatched" not in e:
            print(line(iso, e))
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({**meta, **res}, f, indent=1)
    n_ok = sum(v["usable"] for v in res.values())
    print(f"wrote {len(res)} languages ({n_ok} usable, {len(res) - n_ok} excluded) to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
