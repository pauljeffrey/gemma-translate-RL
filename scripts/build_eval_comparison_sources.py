#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Collect, from files already in the repo, the numbers the post-SFT report of the 385M model is compared with (2026-11-02). Nothing is re-run, nothing is invented:
every output row carries the file it was read from. Stdlib only, CPU.

    python scripts/build_eval_comparison_sources.py [--out reports/data/eval_385m_plan_2026-11-02]

Outputs (CSV) and what they hold:
  own_270m_macro_by_benchmark.csv        our first-research 270M SFT-final and RL-final, macro per benchmark and subset (all / no_pair / clean), chrF++ and spBLEU, finished %      (paper/tables/t_main.md)
  own_270m_per_language.csv              the same per language x benchmark x subset, SFT | RL | delta with 95% intervals                                                     (reports/data/final_model_per_language_wide.csv)
  own_270m_exposure_by_benchmark.csv     how much of each test set the OLD models saw (English in SFT, pair in SFT, rows of the clean subset, test split used in fitting)         (paper/tables/t_exposure_bench.md)
  own_270m_vs_systems_flores_devtest.csv our models next to systems RE-RUN by us on the same sentences and next to PUBLISHED rows, one table                                     (paper/tables/t_model_comparison_main.md)
  published_per_language.csv             published per-language en->xx numbers (NLLB-200 family, AfriqueLLM family, Gemini low-n, ...) with URL and location                          (reports/data/published_baselines.csv)
  published_aggregates_bidirectional.csv published AGGREGATES of the TranslatePsy paper (both directions averaged: NOT comparable with en->xx)                                        (paper/tables/t_model_comparison_published_aggregates.md)
  afriquellm_ssa_comet_perlang.csv       AfriqueLLM Table 20, SSA-COMET per language (checkpoint unnamed in the paper; scale 0-100)                                              (reports/data/afriquellm_table20_ssa_comet_enxx.csv)
  afriscience_mt_published_chrf.csv      AfriScience-MT test chrF (no word order) per language: our checkpoints and the numbers printed by the dataset paper                         (paper/tables/t_final2_afriscience_pub.md)
  SOURCES.csv                            manifest: file, content, original reference, metric, decoding / harness, comparable with a new chrF++ run on FLORES+ devtest?, caveat
"""
from __future__ import annotations

import argparse
import csv
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
T = os.path.join(ROOT, "paper", "tables")
D = os.path.join(ROOT, "reports", "data")


def clean(s: str) -> str:
    s = s.strip().replace("\\_", "_").replace("$^\\dagger$", "†").replace("\\to", "->")
    s = re.sub(r"\$\\?\\?Delta\$", "delta", s).replace("$\\Delta$", "delta").replace("$", "").replace("`", "")
    return s.strip()


def md_table(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.startswith("|"):
            continue
        cells = [clean(c) for c in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", c) or c == "" for c in cells) and any(c for c in cells):
            continue
        rows.append(cells)
    return rows[0], rows[1:]


def write(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"{os.path.relpath(path, ROOT)}: {len(rows)} rows")


def num(x):
    m = re.match(r"^\s*([+-]?\d+(?:\.\d+)?)", x or "")
    return m.group(1) if m else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(D, "eval_385m_plan_2026-11-02"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    src = lambda p: os.path.relpath(p, ROOT)
    manifest = []

    # 1. macro per benchmark ------------------------------------------------------------------------------------------------------------------------------------------------
    p = os.path.join(T, "t_main.md")
    h, rows = md_table(p)
    out, bench = [], ""
    for r in rows:
        bench = r[0] or bench
        b = bench.replace("†", "")
        subset, n = r[1], r[2]
        fin = [x.strip() for x in r[9].split("/")]
        for metric, i_s, i_r, i_d in (("chrF++", 3, 4, 5), ("spBLEU", 6, 7, 8)):
            out.append([b, subset, n, "SFT-final (step 27638)", metric, num(r[i_s]), "", fin[0], src(p)])
            out.append([b, subset, n, "RL-final (GRPO step 1433)", metric, num(r[i_r]), r[i_d], fin[1] if len(fin) > 1 else "", src(p)])
    write(os.path.join(a.out, "own_270m_macro_by_benchmark.csv"),
          ["benchmark", "subset", "n_languages", "model", "metric", "value_macro", "delta_vs_SFT_with_95ci_(RL_rows)", "finished_pct", "source_file"], out)
    manifest.append(["own_270m_macro_by_benchmark.csv", "270M SFT-final and RL-final, macro per benchmark and subset", "paper section 6.1 / t_main", "chrF++, spBLEU (greedy, 768 new tokens)",
                     "our harness, greedy, English->X", "yes for the `clean` rows (re-run the new model on the same sentences)", "`all`/`no_pair` rows are inflated by exposure of the old SFT text (95% of FLORES+ devtest English); the Efik reward thresholds were fitted on IbomMT"])

    # 2. per language ---------------------------------------------------------------------------------------------------------------------------------------------------
    p = os.path.join(D, "final_model_per_language_wide.csv")
    rows = list(csv.reader(open(p, encoding="utf-8")))
    write(os.path.join(a.out, "own_270m_per_language.csv"), rows[0] + ["source_file"], [r + [src(p)] for r in rows[1:]])
    manifest.append(["own_270m_per_language.csv", "per language x benchmark x subset (all/no_pair/clean): SFT-final, RL-final, delta with intervals, finished, LID", "reports/final_model_per_language_2026-10-21.md",
                     "chrF++, spBLEU, finished, LID-ok", "our harness, greedy", "yes (`clean` and `no_pair` columns)", "n per subset is small for `clean` (48 FLORES+ sentences per language)"])

    # 3. exposure ---------------------------------------------------------------------------------------------------------------------------------------------------------
    p = os.path.join(T, "t_exposure_bench.md")
    h, rows = md_table(p)
    write(os.path.join(a.out, "own_270m_exposure_by_benchmark.csv"), h + ["source_file"], [r + [src(p)] for r in rows])
    manifest.append(["own_270m_exposure_by_benchmark.csv", "how much of each test set the OLD SFT/RL pipeline saw", "paper section 6.9a", "share of rows", "contamination audit",
                     "context only", "applies to the OLD 270M models; the 385M run uses the clean-rerun data (protected index) and needs its own audit before any `clean` claim"])

    # 4. one table with re-run and published rows -----------------------------------------------------------------------------------------------------------------------------------
    p = os.path.join(T, "t_model_comparison_main.md")
    lines = [l for l in open(p, encoding="utf-8")]
    start = next(i for i, l in enumerate(lines) if l.startswith("| Model"))
    tab = [[clean(c) for c in l.strip().strip("|").split("|")] for l in lines[start:] if l.startswith("|") and not re.match(r"^\|\s*:?-", l)]
    h, rows = tab[0], tab[1:]
    out = []
    for r in rows:
        origin = "re-run by us (same sentences)" if r[-1].startswith("computed") else ("published by the system's authors" if r[-1].startswith("published") else r[-1])
        out.append(r + [origin, src(p)])
    write(os.path.join(a.out, "own_270m_vs_systems_flores_devtest.csv"), h + ["origin_class", "source_file"], out)
    manifest.append(["own_270m_vs_systems_flores_devtest.csv", "our 270M checkpoints, NLLB/TranslatePsy re-runs by us, published rows (NLLB, AfriqueLLM family) in one table", "paper section 6.8 / t_model_comparison_main",
                     "chrF++ (NLLB-43 / Afrique-21 / TranslatePsy-19 language sets), AfriCOMET-STL-1.1 ref-based, SSA-COMET",
                     "re-run rows: FLORES+ devtest `no_pair` subset, NLLB CT2 beam 4, TranslatePsy zero-shot greedy; published rows: the authors' own pipeline, whole devtest",
                     "re-run rows: yes for the same language sets; published rows: indicative only", "published rows are few-shot base models or beam search on FLORES-200 devtest: not our decoding; do not mix with `clean` rows"])

    # 5. published per language -------------------------------------------------------------------------------------------------------------------------------------------------
    p = os.path.join(D, "published_baselines.csv")
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    out = []
    for r in rows:
        if r["flag"] == "no_number":
            note = "no number published (figure only)"
        elif r["flag"] == "low_n":
            note = "10 sentences per language, chrF not chrF++: do NOT compare"
        elif r["flag"] == "no_chrf":
            note = "WMT24++ (other test set); the paper prints MetricX / COMET-22 only, no chrF++ or BLEU: do NOT compare"
        elif r["benchmark"].startswith("AfroBench MT"):
            note = "AfroBench pooled score (FLORES-200 + MAFAND + NTREX-128 mixed, best of 3 prompts, 0-shot, en/fr->xx): different sentence mix and prompt selection, do NOT compare"
        elif r["iso"].startswith("MACRO"):
            note = "macro of a fixed language set that the table does not list per language (see the language set in the source), direction inferred as the AVERAGE of en->xx and xx->en: do NOT compare with an en->xx run"
        elif r["model"].startswith("NLLB"):
            note = "chrF++ (word order 2) per language, FLORES-200 devtest, beam search: comparable with a chrF++ run on FLORES+ devtest (same sentences, some references corrected)"
        else:
            note = "chrF++ per language, AfroBench-Lite FLORES-200 en->xx, split not stated, few-shot base model: indicative"
        out.append([r["model"], r["version"], r["iso"], r["benchmark"], r["direction"], r["chrF++"], r["BLEU_spBLEU"], r["bleu_kind"], r["flag"], note, r["source_url"], r["source_loc"], src(p)])
    write(os.path.join(a.out, "published_per_language.csv"),
          ["model", "version", "iso", "benchmark", "direction", "chrF++", "BLEU_or_spBLEU", "bleu_kind", "flag", "comparability_note", "source_url", "source_loc", "source_file"], out)
    manifest.append(["published_per_language.csv", "published per-language en->xx numbers (NLLB-200 1.3B/3.3B/distilled/MoE, AfriqueLLM family, Gemini 2.5 Flash low-n, TranslatePsy no_number)", "reports/published_baselines_2026-10-21.md (agent A3)",
                     "chrF++ (NLLB: sacreBLEU word order 2, 1 decimal; Afrique: Table 22)", "NLLB: beam search (default); Afrique: 5-shot base models through AfroBench-Lite", "NLLB yes (44 of our 77 languages); Afrique indicative (21 languages)",
                     "NO published per-language BLEU/spBLEU for NLLB 1.3B/3.3B; NO per-language TranslatePsy; TranslateGemma publishes MetricX/COMET-22 only"])

    # 6. aggregates ---------------------------------------------------------------------------------------------------------------------------------------------------------------
    p = os.path.join(T, "t_model_comparison_published_aggregates.md")
    lines = [l for l in open(p, encoding="utf-8")]
    start = next(i for i, l in enumerate(lines) if l.startswith("| model"))
    tab = [[clean(c) for c in l.strip().strip("|").split("|")] for l in lines[start:] if l.startswith("|") and not re.match(r"^\|\s*:?-", l)]
    h, rows = tab[0], tab[1:]
    write(os.path.join(a.out, "published_aggregates_bidirectional.csv"), h + ["comparability", "source_file"],
          [r + ["NOT comparable with en->xx macros (both directions averaged, 19 languages)", src(p)] for r in rows])
    manifest.append(["published_aggregates_bidirectional.csv", "TranslatePsy-AfriSLM paper Tables 24/25 aggregates, FLORES-200 devtest, 19 languages", "TranslatePsy-AfriSLM paper (Tables 24/25)",
                     "chrF++, spBLEU, SSA-COMET 0-1, COMET-22, MetricX-24", "paper's pipeline", "no", "bidirectional averages (verified by recomputation, reports/published_baselines_2026-10-21.md section 4)"])

    # 7. AfriqueLLM SSA-COMET ------------------------------------------------------------------------------------------------------------------------------------------------------
    p = os.path.join(D, "afriquellm_table20_ssa_comet_enxx.csv")
    rows = list(csv.reader(open(p, encoding="utf-8")))
    write(os.path.join(a.out, "afriquellm_ssa_comet_perlang.csv"), rows[0] + ["source_file"], [r + [src(p)] for r in rows[1:]])
    manifest.append(["afriquellm_ssa_comet_perlang.csv", "AfriqueLLM arXiv 2601.06395 Table 20: SSA-COMET per language, FLORES-200 en->xx", "arXiv 2601.06395", "SSA-COMET 0-100 (checkpoint not named)",
                     "paper's pipeline", "only if we score our output with the SAME SSA-COMET checkpoint (unknown): treat as indicative", "our reward-side SSA-COMET-MTL numbers use a named checkpoint and a different scale handling"])

    # 8. AfriScience-MT --------------------------------------------------------------------------------------------------------------------------------------------------------------
    p = os.path.join(T, "t_final2_afriscience_pub.md")
    h, rows = md_table(p)
    write(os.path.join(a.out, "afriscience_mt_published_chrf.csv"), h + ["source_file"], [r + [src(p)] for r in rows])
    manifest.append(["afriscience_mt_published_chrf.csv", "AfriScience-MT test, chrF per language: our 270M checkpoints and the dataset paper's systems", "arXiv abdulmumin2026afriscience; reports/afriscience_mt_2026-10-21.md",
                     "chrF WITHOUT word n-grams (the published metric); sentence level and whole-paper level", "paper's pipelines (10-shot / fine-tuned; document level 0/1-shot)",
                     "chrF (word order 0) yes if we compute plain chrF; never compare with chrF++", "our old models were trained on the AfriScience-MT TRAIN split (in-domain); the 385M clean run keeps test AND dev out of training and selection (dev = selection only)"])

    write(os.path.join(a.out, "SOURCES.csv"), ["file", "content", "original reference", "metric", "decoding / harness", "comparable with a new run?", "caveat"], manifest)


if __name__ == "__main__":
    main()
