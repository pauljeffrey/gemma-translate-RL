#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build the final-model per-language master table (D-49) from the benchmark runs of the SFT-final and RL-final models.

Inputs  (reports/data/final_model_runs/, written by the box job `eval_sft_benchmarks.sh` + `score_bench.py` + `bench_boot.py` / `macro_boot.py`):
  bench_metrics_sft.csv, bench_metrics_rl.csv      max_new_tokens 768  (system, benchmark, iso, subset, n, chrf, spbleu, finished, prefix_ok, empty, lid_n, lid_ok, lid_ok_sib)
  bench_metrics_sft320.csv, bench_metrics_rl320.csv max_new_tokens 320 (optional)
  bench_boot.csv                                   paired per-language bootstrap RL - SFT (500 resamples)
  macro_boot.json                                  macro bootstrap of RL - SFT per (benchmark, subset, group) (1,000 resamples, produced by macro_summary.py)
Outputs: reports/data/final_model_per_language.csv (long), reports/data/final_model_per_language_wide.csv (one row per language x benchmark x subset, SFT | RL | delta),
         reports/data/final_model_per_language_max320.csv, reports/final_model_per_language_2026-10-21.md
Stdlib only.
"""
from __future__ import annotations
import os

import csv
import json
import statistics
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "reports" / "data"
RUNS = DATA / "final_model_runs"
sys.path.insert(0, str(ROOT / "gemma_mt"))
from gemma_mt import languages as L  # noqa: E402

SFT_NAME, RL_NAME = "sft_step27638", "rl_final_step1433"
BENCH_ORDER = ["flores_plus", "ntrex", "mafand", "smol_sent", "ibom_mt", "flores_moz", "salt", "glot_story"]
BENCH_LABEL = {"flores_plus": "FLORES+ devtest", "ntrex": "NTREX-128", "mafand": "MAFAND-MT test", "smol_sent": "SmolSent", "ibom_mt": "IbomMT",
               "flores_moz": "FLORES-MOZ", "salt": "SALT", "glot_story": "GlotStoryBook"}
SUBSETS = ["all", "no_pair", "clean"]
GROUPS = OrderedDict([("qe_validated", "QE-validated (42)"), ("blend", "QE-blend (12)"), ("consensus_only", "consensus-only (23)"), ("sft_only", "SFT-only, no RL (8)")])

FAMILY = {}
for fam, isos in {
    "Bantu": "swh swc kin run lug zul xho nbl nde ssw sna nya sot tsn nso ven tso lin tum bem umb kmb kik kam lua kqn lun loz kua her yao ndc vmw tsc seh nyn luy kon dua bas".split(),
    "Atlantic-Congo (non-Bantu)": "yor ibo efi ibb ish bin iso oke urh twi fat ewe fon gaa tiv bbj kbp mos dag sag wol ful".split(),
    "Mande": "bam dyu men".split(),
    "Afroasiatic": "hau amh tir som orm sid wal kab tzm shi".split(),
    "Nilo-Saharan": "luo ach nus knc".split(),
    "Creole / Pidgin": "pcm wes kri kea".split(),
    "Other (Indo-European, Austronesian)": "afr mlg".split(),
}.items():
    for i in isos:
        FAMILY[i] = fam


def group_of(iso):
    l = L.LANGUAGES[iso]
    if iso in L.QE_VALIDATED:
        return "qe_validated"
    if l.rl_consensus:
        return "consensus_only"
    if l.rl_eligible:
        return "blend"
    return "sft_only"


def fnum(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def read_metrics(path):
    out = {}
    if not Path(path).exists():
        return out
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[(r["benchmark"], r["iso"], r["subset"])] = {k: (fnum(r[k]) if k != "system" and k != "benchmark" and k != "iso" and k != "subset" else r[k]) for k in r}
    return out


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def f1(x, d=1):
    return "-" if x is None else f"{x:.{d}f}"


def sg(x, d=1):
    return "-" if x is None else f"{x:+.{d}f}"


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}"


def main():
    sft, rl = read_metrics(RUNS / "bench_metrics_sft.csv"), read_metrics(RUNS / "bench_metrics_rl.csv")
    sft3, rl3 = read_metrics(RUNS / "bench_metrics_sft320.csv"), read_metrics(RUNS / "bench_metrics_rl320.csv")
    boot = {}
    with open(RUNS / "bench_boot.csv", newline="") as f:
        for r in csv.DictReader(f):
            boot[(r["benchmark"], r["iso"], r["subset"])] = {k: fnum(v) for k, v in r.items() if k not in ("benchmark", "iso", "subset")}
    mb = json.load(open(RUNS / "macro_boot.json")) if (RUNS / "macro_boot.json").exists() else {}
    names = {i: l.name for i, l in L.LANGUAGES.items()}
    isos = list(L.LANGUAGES)

    # ---------------- long CSV (one row per language x benchmark x checkpoint x subset)
    cols = ["iso", "language", "group", "family", "benchmark", "checkpoint", "subset", "n", "chrf", "spbleu", "finished", "prefix_ok", "empty", "lid_ok", "lid_ok_sibling", "max_new_tokens"]
    def long_rows(sets, budget):
        rows = []
        for ck, M in sets:
            for (b, i, s), r in sorted(M.items(), key=lambda kv: (BENCH_ORDER.index(kv[0][0]), isos.index(kv[0][1]), SUBSETS.index(kv[0][2]))):
                rows.append([i, names[i], group_of(i), FAMILY[i], b, ck, s, int(r["n"]), r["chrf"], r["spbleu"], r["finished"], r["prefix_ok"], r["empty"], r["lid_ok"], r["lid_ok_sib"], budget])
        return rows
    def write(path, rows, header):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(header)
            for r in rows:
                w.writerow(["" if v is None else (f"{v:.4f}" if isinstance(v, float) else v) for v in r])
    write(DATA / "final_model_per_language.csv", long_rows([(SFT_NAME, sft), (RL_NAME, rl)], 768), cols)
    if sft3 or rl3:
        write(DATA / "final_model_per_language_max320.csv", long_rows([(SFT_NAME + "_max320", sft3), (RL_NAME + "_max320", rl3)], 320), cols)

    # ---------------- wide CSV (deltas)
    wcols = ["iso", "language", "group", "family", "benchmark", "subset", "n", "primary", "sft_chrf", "rl_chrf", "d_chrf", "d_chrf_lo", "d_chrf_hi", "sft_spbleu", "rl_spbleu", "d_spbleu", "d_spbleu_lo", "d_spbleu_hi",
             "sft_finished", "rl_finished", "sft_lid_ok", "rl_lid_ok", "sft_lid_ok_sibling", "rl_lid_ok_sibling"]
    # primary benchmark per language (no_pair rows must exist and have n >= 1)
    primary = {}
    for i in isos:
        for b in BENCH_ORDER:
            if (b, i, "no_pair") in rl and (b, i, "no_pair") in sft:
                primary[i] = b
                break
    wide = []
    for i in isos:
        for b in BENCH_ORDER:
            for s in SUBSETS:
                k = (b, i, s)
                if k not in rl or k not in sft:
                    continue
                a, c, bt = sft[k], rl[k], boot.get(k, {})
                wide.append([i, names[i], group_of(i), FAMILY[i], b, s, int(c["n"]), int(primary.get(i) == b), a["chrf"], c["chrf"], c["chrf"] - a["chrf"], bt.get("d_chrf_lo"), bt.get("d_chrf_hi"),
                             a["spbleu"], c["spbleu"], c["spbleu"] - a["spbleu"], bt.get("d_spbleu_lo"), bt.get("d_spbleu_hi"), a["finished"], c["finished"], a["lid_ok"], c["lid_ok"], a["lid_ok_sib"], c["lid_ok_sib"]])
    write(DATA / "final_model_per_language_wide.csv", wide, wcols)

    # ---------------- aggregation helpers
    def cell(M, b, i, s, key):
        r = M.get((b, i, s))
        return None if r is None else r[key]
    def macro_of(M, b, s, members, key):
        return mean([cell(M, b, i, s, key) for i in members])
    def members_for(b, s, pred=lambda i: True):
        return [i for i in isos if (b, i, s) in rl and (b, i, s) in sft and pred(i)]
    def prim_macro(M, s, members, key):
        vals = []
        for i in members:
            b = primary.get(i)
            if b is not None and s == "no_pair":
                vals.append(cell(M, b, i, s, key))
        return mean(vals)
    def mbrow(b, s, g, metric):
        return mb.get(f"{b}|{s}|{g}", {}).get(metric)
    def ci_txt(r):
        return "-" if not r else f"{r['d']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]"

    S_BUD = 6 if rl3 else None
    S_NOB = 7 if rl3 else 6
    out = []
    P = out.append
    P("# Final model, per language: SFT-final vs RL-final on every human-reference benchmark (2026-10-21)\n")
    P("Task D-49 (2). Raw numbers: `reports/data/final_model_per_language.csv` (long: language x benchmark x checkpoint x subset, the format `gemma_mt/scripts/join_published_vs_ours.py` reads), "
      "`reports/data/final_model_per_language_wide.csv` (SFT | RL | delta with paired-bootstrap CI), `reports/data/final_model_per_language_max320.csv` (same at the older 320-token budget), run files in `reports/data/final_model_runs/`. "
      "Generated by `gemma_mt/scripts/build_final_per_language.py`. Decoding study (sentence level, 20 languages): `reports/final_model_decoding_study_2026-10-21.md`.\n")

    # ---- 0 headline
    fl = "flores_plus"
    P("## 0. Headline\n")
    m_all = members_for(fl, "no_pair")
    P(f"* **FLORES+ devtest, English -> X, greedy, {len(m_all)} languages, rows whose reference pair was not in the SFT text (`no_pair`, 96% of the rows)**: "
      f"chrF++ SFT-final **{f1(macro_of(sft, fl, 'no_pair', m_all, 'chrf'))}** -> RL-final **{f1(macro_of(rl, fl, 'no_pair', m_all, 'chrf'))}** "
      f"(delta {ci_txt(mbrow(fl, 'no_pair', 'all', 'chrf'))}); spBLEU {f1(macro_of(sft, fl, 'no_pair', m_all, 'spbleu'))} -> {f1(macro_of(rl, fl, 'no_pair', m_all, 'spbleu'))} "
      f"(delta {ci_txt(mbrow(fl, 'no_pair', 'all', 'spbleu'))}). Brackets are 95% paired-bootstrap intervals of the macro delta (1,000 resamples of the sentences of every language).")
    m_cl = members_for(fl, "clean")
    P(f"* Same on the `clean` subset (English sentence not in the SFT text; only 48 sentences per language, {len(m_cl)} languages): chrF++ {f1(macro_of(sft, fl, 'clean', m_cl, 'chrf'))} -> {f1(macro_of(rl, fl, 'clean', m_cl, 'chrf'))} "
      f"(delta {ci_txt(mbrow(fl, 'clean', 'all', 'chrf'))}); on all 1,012 rows {f1(macro_of(sft, fl, 'all', members_for(fl, 'all'), 'chrf'))} -> {f1(macro_of(rl, fl, 'all', members_for(fl, 'all'), 'chrf'))}.")
    if rl3:
        m3 = [i for i in members_for(fl, "no_pair") if (fl, i, "no_pair") in rl3 and (fl, i, "no_pair") in sft3]
        P(f"* **At the 320-token budget of the 2026-10-18/19 reports** the same comparison is SFT {f1(macro_of(sft3, fl, 'no_pair', m3, 'chrf'))} -> RL {f1(macro_of(rl3, fl, 'no_pair', m3, 'chrf'))} chrF++ (delta {sg(macro_of(rl3, fl, 'no_pair', m3, 'chrf') - macro_of(sft3, fl, 'no_pair', m3, 'chrf'), 2)}), "
          f"because the SFT model loops (finished {pct(macro_of(sft3, fl, 'all', members_for(fl, 'all'), 'finished'))}% on FLORES+) and the longer budget lengthens its runaway completions; the RL model finishes 100% at both budgets (section {S_BUD}). Use the 320 figure when comparing with the older reports and the 768 figure for the model as it would behave with a generous limit.")
    P(f"* Finished rate (completion ends inside the 768-token budget) on FLORES+ `all`: SFT {pct(macro_of(sft, fl, 'all', members_for(fl, 'all'), 'finished'))}% -> RL {pct(macro_of(rl, fl, 'all', members_for(fl, 'all'), 'finished'))}%; "
      f"GlotLID right-language rate: SFT {pct(macro_of(sft, fl, 'all', members_for(fl, 'all'), 'lid_ok'))}% -> RL {pct(macro_of(rl, fl, 'all', members_for(fl, 'all'), 'lid_ok'))}% (sibling-tolerant {pct(macro_of(sft, fl, 'all', members_for(fl, 'all'), 'lid_ok_sib'))}% -> {pct(macro_of(rl, fl, 'all', members_for(fl, 'all'), 'lid_ok_sib'))}%).")
    n_with = len(primary)
    nob = [i for i in isos if i not in primary]
    nob_rl = [i for i in nob if L.LANGUAGES[i].rl_eligible]
    P(f"* **Coverage.** {n_with} of the 85 language codes (of them {len(primary) - len([i for i in primary if not L.LANGUAGES[i].rl_eligible])} of the 77 RL languages) have at least one human-reference benchmark row; "
      f"**{len(nob)} have none** ({len(nob_rl)} of them RL languages): " + ", ".join(f"`{i}`" for i in nob) + ". Their quality is only judged by the reward models (AfriCOMET-QE / consensus), never against a human reference; see section " + str(S_NOB) + ".")
    P("")

    # ---- 1 setup
    P("## 1. Systems, data, method\n")
    P(f"* **SFT-final** = `{os.environ['HF_ORG']}/gemma3-270m-translate` tag `step-27638`. **RL-final** = `<HF_ORG_LEGACY>/gemma-3-270m-translate-rl2c` `latest` (GRPO step 1433, `runs/final_1433`; the Hub sha used by the decoding study is `80baefff3b16`). Both: greedy, prompt `Translate the text to {{language}}:  {{english}}`, "
      "the language tag written by the model is stripped before scoring, metrics on NFC text, `max_new_tokens` 768 (the headline; section " + str(S_BUD) + " gives FLORES+ at the 320 budget of the 2026-10-18/19 reports).")
    P("* **Pipeline**: the repo's `gemma_mt.eval.sft_benchmarks` through `scripts/eval_sft_benchmarks.sh` (vLLM worker, HF fallback), no AfriCOMET, no base model. Benchmarks: FLORES+ devtest (1,012 sentences, 45 languages), NTREX-128 (1,997, 24), MAFAND-MT test (15), SmolSent (863, 45), "
      "IbomMT (2), FLORES-MOZ (1), SALT (5), GlotStoryBook (8). Test splits only. GATITOS (words) and SmolDoc (documents) are not part of this table.")
    P("* **Metrics**: chrF++ (sacrebleu `word_order=2`), **spBLEU** = sacrebleu BLEU with `tokenize=\"flores200\"` (no 13a BLEU is reported: it is not meaningful for these scripts), corpus-level per language and subset; macro = unweighted mean over the languages of the row set; "
      "`finished` = the completion ends with the stop token inside the budget; `LID` = GlotLID v3 top-1 label of the hypothesis is one of the language's labels (`LID~` also accepts sibling labels such as `swc` for `swh`).")
    P("* **Subsets** (contamination flags in `bench/test_sets/_contamination`): `all` = every row; `no_pair` = rows whose *reference pair* was not in the SFT text (the model never saw this translation; headline); `clean` = rows whose *English sentence* is not in the SFT text "
      "(for FLORES+ only 48 of 1,012 sentences, for SmolSent 76 of 863, so per-language clean numbers are noisy). The model has seen 95% of the FLORES+ English sentences in other language pairs, which is why `clean` is lower than `all`.")
    P("* **Groups** (from `languages.py`): `qe_validated` 42 languages (AfriCOMET-QE validated against human references), `blend` 12 (named on the AfriCOMET card, QE blended with the consensus signal), `consensus_only` 23 (no QE call, consensus reward only), `sft_only` 8 (SFT targets that were not RL-trained: nde fat knc gaa men luy tzm ful; per `languages.py` `nde`, `knc`, `ful`, `gaa` have no SFT data at all, so `nde` and `knc` rows are zero-shot and the model does not write their tag, `prefix_ok` = 0 and the tag stays in the scored text). The 77 RL languages are the first three groups.")
    P("* **Cross-check against the existing reports**: the new SFT-final numbers reproduce the 2026-10-19 rare-language report (same checkpoint, step 27638, 320 tokens, HF sampler there vs vLLM here) with a mean absolute difference of 0.1 chrF++ on IbomMT / FLORES-MOZ / GlotStoryBook / SALT / the 9 gain languages (largest +-0.9); "
      "the 2026-10-18 FLORES+ table is for the older step-20500 checkpoint, so the SFT side is regenerated here (all benchmarks, same pipeline as the RL side) instead of copied.")
    P("")

    # ---- 2 macro tables
    P("## 2. Macro averages\n")
    P("### 2.1 Per benchmark and subset (all languages of the benchmark)\n")
    P("| benchmark | subset | langs | rows/lang | chrF++ SFT | chrF++ RL | delta chrF++ [95% CI] | spBLEU SFT | spBLEU RL | delta spBLEU [95% CI] | finished SFT / RL % | LID SFT / RL % |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for b in BENCH_ORDER:
        for s in SUBSETS:
            mem = members_for(b, s)
            if not mem:
                continue
            n_med = int(statistics.median([rl[(b, i, s)]["n"] for i in mem]))
            P(f"| {BENCH_LABEL[b]} | {s} | {len(mem)} | {n_med} | {f1(macro_of(sft, b, s, mem, 'chrf'))} | {f1(macro_of(rl, b, s, mem, 'chrf'))} | {ci_txt(mbrow(b, s, 'all', 'chrf'))} | "
              f"{f1(macro_of(sft, b, s, mem, 'spbleu'))} | {f1(macro_of(rl, b, s, mem, 'spbleu'))} | {ci_txt(mbrow(b, s, 'all', 'spbleu'))} | "
              f"{pct(macro_of(sft, b, s, mem, 'finished'))} / {pct(macro_of(rl, b, s, mem, 'finished'))} | {pct(macro_of(sft, b, s, mem, 'lid_ok'))} / {pct(macro_of(rl, b, s, mem, 'lid_ok'))} |")
    P("")
    P("### 2.2 FLORES+ devtest `no_pair`, by training group and by family\n")
    P("| set | langs | chrF++ SFT | chrF++ RL | delta chrF++ [95% CI] | spBLEU SFT | spBLEU RL | delta spBLEU [95% CI] | finished SFT / RL % | LID SFT / RL % |")
    P("|---|---|---|---|---|---|---|---|---|---|")
    sets = [("all FLORES+ languages", "all", m_all)]
    for g, lab in GROUPS.items():
        sets.append((lab, g, [i for i in m_all if group_of(i) == g]))
    sets.append(("RL languages (qe + blend + consensus)", "rl77", [i for i in m_all if group_of(i) != "sft_only"]))
    for fam in OrderedDict.fromkeys(FAMILY.values()):
        sets.append((fam, "fam:" + fam, [i for i in m_all if FAMILY[i] == fam]))
    for lab, key, mem in sets:
        if not mem:
            continue
        P(f"| {lab} | {len(mem)} | {f1(macro_of(sft, fl, 'no_pair', mem, 'chrf'))} | {f1(macro_of(rl, fl, 'no_pair', mem, 'chrf'))} | {ci_txt(mbrow(fl, 'no_pair', key, 'chrf'))} | "
          f"{f1(macro_of(sft, fl, 'no_pair', mem, 'spbleu'))} | {f1(macro_of(rl, fl, 'no_pair', mem, 'spbleu'))} | {ci_txt(mbrow(fl, 'no_pair', key, 'spbleu'))} | "
          f"{pct(macro_of(sft, fl, 'no_pair', mem, 'finished'))} / {pct(macro_of(rl, fl, 'no_pair', mem, 'finished'))} | {pct(macro_of(sft, fl, 'no_pair', mem, 'lid_ok'))} / {pct(macro_of(rl, fl, 'no_pair', mem, 'lid_ok'))} |")
    P("")
    P("### 2.3 Primary benchmark of every language (`no_pair`), by group\n")
    P("Each language is scored on the first benchmark it has in the order FLORES+, NTREX, MAFAND, SmolSent, IbomMT, FLORES-MOZ, SALT, GlotStoryBook. Different benchmarks have different difficulty, so the macro over this mix is a coverage summary, not a like-for-like score; use section 2.2 for the comparison.\n")
    P("| group | langs with a benchmark | langs in group | chrF++ SFT | chrF++ RL | delta chrF++ [95% CI] | spBLEU SFT | spBLEU RL | delta spBLEU [95% CI] |")
    P("|---|---|---|---|---|---|---|---|---|")
    for g, lab in list(GROUPS.items()) + [("rl77", "RL languages (77)"), ("all", "all 85")]:
        tot = [i for i in isos if (g == "all" or (g == "rl77" and group_of(i) != "sft_only") or group_of(i) == g)]
        mem = [i for i in tot if i in primary]
        P(f"| {lab} | {len(mem)} | {len(tot)} | {f1(prim_macro(sft, 'no_pair', mem, 'chrf'))} | {f1(prim_macro(rl, 'no_pair', mem, 'chrf'))} | {ci_txt(mbrow('primary', 'no_pair', g, 'chrf'))} | "
          f"{f1(prim_macro(sft, 'no_pair', mem, 'spbleu'))} | {f1(prim_macro(rl, 'no_pair', mem, 'spbleu'))} | {ci_txt(mbrow('primary', 'no_pair', g, 'spbleu'))} |")
    P("")

    # ---- 3 per-language master table
    P("## 3. Master table: every language on its primary benchmark (`no_pair`)\n")
    P("Delta = RL-final minus SFT-final; the bracket is the 95% paired-bootstrap interval of the language's delta (500 resamples), marked `*` when it excludes 0. `n` = sentences scored. `fin` = finished %, `LID` = GlotLID right-language %. "
      "Languages with **no human-reference benchmark** are listed at the end of the table with `-`.\n")
    P("| iso | language | group | family | benchmark | n | chrF++ SFT | chrF++ RL | delta chrF++ | spBLEU SFT | spBLEU RL | delta spBLEU | fin SFT / RL | LID SFT / RL |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    def lang_row(i):
        b = primary.get(i)
        base = f"| `{i}` | {names[i]} | {GROUPS[group_of(i)].split(' (')[0]} | {FAMILY[i].split(' (')[0]} |"
        if b is None:
            return base + " **none** | - | - | - | - | - | - | - | - | - |"
        k = (b, i, "no_pair"); a, c, bt = sft[k], rl[k], boot.get(k, {})
        def dtxt(d, lo, hi):
            if lo is None:
                return sg(d)
            star = "*" if (lo > 0 or hi < 0) else ""
            return f"{d:+.1f} [{lo:+.1f}, {hi:+.1f}]{star}"
        return (base + f" {BENCH_LABEL[b]} | {int(c['n'])} | {f1(a['chrf'])} | {f1(c['chrf'])} | {dtxt(c['chrf'] - a['chrf'], bt.get('d_chrf_lo'), bt.get('d_chrf_hi'))} | "
                f"{f1(a['spbleu'])} | {f1(c['spbleu'])} | {dtxt(c['spbleu'] - a['spbleu'], bt.get('d_spbleu_lo'), bt.get('d_spbleu_hi'))} | "
                f"{pct(a['finished'])} / {pct(c['finished'])} | {pct(a['lid_ok'])} / {pct(c['lid_ok'])} |")
    order = sorted(isos, key=lambda i: (list(GROUPS).index(group_of(i)), primary.get(i) is None, -(rl[(primary[i], i, "no_pair")]["chrf"] if i in primary else 0)))
    for i in order:
        P(lang_row(i))
    P("")

    # ---- 4 per benchmark per language wide table
    P("## 4. All benchmarks side by side (chrF++ SFT -> RL, `no_pair`; `all` in section 4.2)\n")
    P("Cells are `SFT -> RL`. A language appears in a column only if the benchmark covers it.\n")
    def side_table(sub):
        P("| iso | language | " + " | ".join(BENCH_LABEL[b] for b in BENCH_ORDER) + " |")
        P("|---|---|" + "---|" * len(BENCH_ORDER))
        for i in order:
            cells = []
            for b in BENCH_ORDER:
                k = (b, i, sub)
                cells.append(f"{sft[k]['chrf']:.1f} -> {rl[k]['chrf']:.1f}" if (k in sft and k in rl) else "")
            if any(cells):
                P(f"| `{i}` | {names[i]} | " + " | ".join(cells) + " |")
        P("")
    P("### 4.1 `no_pair`\n"); side_table("no_pair")
    P("### 4.2 `all`\n"); side_table("all")

    # ---- 5 gains and losses
    P("## 5. Largest and smallest gains; where RL is not better\n")
    P("RL-final is better than SFT-final on all 45 FLORES+ languages, so the right-hand columns are the *smallest gains*, not losses; the cells where RL is actually below SFT are listed in 5.3.\n")
    def gl(b, s, label, top=10):
        mem = members_for(b, s)
        d = sorted(((rl[(b, i, s)]["chrf"] - sft[(b, i, s)]["chrf"], i) for i in mem), reverse=True)
        P(f"**{label}** ({len(mem)} languages)\n")
        P("| rank | largest gains: language | SFT -> RL chrF++ | delta | | smallest gains: language | SFT -> RL chrF++ | delta |")
        P("|---|---|---|---|---|---|---|---|")
        for r in range(top):
            g, i = d[r]; l, j = d[-1 - r]
            bt_g, bt_l = boot.get((b, i, s), {}), boot.get((b, j, s), {})
            P(f"| {r + 1} | {names[i]} (`{i}`) | {sft[(b, i, s)]['chrf']:.1f} -> {rl[(b, i, s)]['chrf']:.1f} | {g:+.1f} [{bt_g.get('d_chrf_lo', 0):+.1f}, {bt_g.get('d_chrf_hi', 0):+.1f}] | | "
              f"{names[j]} (`{j}`) | {sft[(b, j, s)]['chrf']:.1f} -> {rl[(b, j, s)]['chrf']:.1f} | {l:+.1f} [{bt_l.get('d_chrf_lo', 0):+.1f}, {bt_l.get('d_chrf_hi', 0):+.1f}] |")
        P("")
        neg = [i for g, i in d if g < 0]
        P(f"{len([1 for g, _ in d if g > 0])} of {len(d)} languages improve, {len(neg)} drop. Languages with a negative delta: " + (", ".join(f"`{i}` ({rl[(b, i, s)]['chrf'] - sft[(b, i, s)]['chrf']:+.1f})" for i in neg) or "none") + ".\n")
    gl(fl, "no_pair", "FLORES+ devtest, `no_pair`, chrF++")
    # across all other benchmarks
    cells = []
    for (b, i, s), r in rl.items():
        if b != fl and s == "no_pair" and (b, i, s) in sft and r["n"] >= 100:
            cells.append((r["chrf"] - sft[(b, i, s)]["chrf"], b, i))
    cells.sort(reverse=True)
    P(f"**Other benchmarks, `no_pair`, cells with >= 100 sentences ({len(cells)} language x benchmark cells)**\n")
    P("| rank | largest gains | SFT -> RL | delta | | smallest gains | SFT -> RL | delta |")
    P("|---|---|---|---|---|---|---|---|")
    for r in range(10):
        g, b, i = cells[r]; l, b2, j = cells[-1 - r]
        P(f"| {r + 1} | {names[i]} on {BENCH_LABEL[b]} | {sft[(b, i, 'no_pair')]['chrf']:.1f} -> {rl[(b, i, 'no_pair')]['chrf']:.1f} | {g:+.1f} | | {names[j]} on {BENCH_LABEL[b2]} | "
          f"{sft[(b2, j, 'no_pair')]['chrf']:.1f} -> {rl[(b2, j, 'no_pair')]['chrf']:.1f} | {l:+.1f} |")
    P("")

    P("### 5.3 Cells where RL-final is below SFT-final\n")
    negs = sorted(((rl[k]["chrf"] - sft[k]["chrf"], k) for k in rl if k in sft and k[0] in BENCH_ORDER and rl[k]["chrf"] < sft[k]["chrf"]))
    P("chrF++ (any subset):\n")
    P("| language | benchmark | subset | n | SFT -> RL chrF++ | delta [95% CI] |")
    P("|---|---|---|---|---|---|")
    for d_, k in negs:
        bt_ = boot.get(k, {})
        P(f"| {names[k[1]]} (`{k[1]}`) | {BENCH_LABEL[k[0]]} | {k[2]} | {int(rl[k]['n'])} | {sft[k]['chrf']:.1f} -> {rl[k]['chrf']:.1f} | {d_:+.2f} [{bt_.get('d_chrf_lo', 0):+.2f}, {bt_.get('d_chrf_hi', 0):+.2f}] |")
    sp = sorted(((rl[k]["spbleu"] - sft[k]["spbleu"], k) for k in rl if k in sft and k[2] == "no_pair" and rl[k]["spbleu"] < sft[k]["spbleu"]))
    P("\nspBLEU, `no_pair` cells:\n")
    P("| language | benchmark | n | SFT -> RL spBLEU | delta [95% CI] |")
    P("|---|---|---|---|---|")
    for d_, k in sp:
        bt_ = boot.get(k, {})
        P(f"| {names[k[1]]} (`{k[1]}`) | {BENCH_LABEL[k[0]]} | {int(rl[k]['n'])} | {sft[k]['spbleu']:.1f} -> {rl[k]['spbleu']:.1f} | {d_:+.2f} [{bt_.get('d_spbleu_lo', 0):+.2f}, {bt_.get('d_spbleu_hi', 0):+.2f}] |")
    P("\nThe only chrF++ regression with an interval that excludes 0 on a `no_pair` subset is Luganda on SALT (81 sentences; SALT Luganda translations are in the SFT text, so the other 419 rows are `pair_seen`); chrF++ and spBLEU also fall slightly on Sesotho (FLORES+ spBLEU) and on Hausa (MAFAND, spBLEU) although chrF++ rises there. "
      "Swahili / Zulu / Xhosa `clean` deltas below 0 are within noise (48 sentences).\n")

    # ---- 6 max320
    if rl3:
        P("## 6. Effect of the token budget: FLORES+ at 320 vs 768 new tokens\n")
        P("The 2026-10-18/19 reports used 320 new tokens; this report's headline uses 768. A looping completion is scored as it stands, so the larger budget lets the looping SFT model append more repeated text (precision falls: SFT-final 30.6 at 320 -> 28.9 at 768 chrF++), while the RL model finishes every sentence inside either budget (34.3 at both). "
      "The RL-minus-SFT gain is therefore **+3.7 chrF++ at the 320 budget of the older reports and +5.4 at 768**; about 1.7 points of the 768 gain are the SFT model's runaway tail, the remaining +3.7 still contains the 10% of SFT sentences that loop inside 320 tokens. Quote the 320 number when comparing with the earlier reports.\n")
        P("| subset | langs | SFT 320 | SFT 768 | RL 320 | RL 768 | RL-SFT at 320 | RL-SFT at 768 | finished SFT 320 / 768 | finished RL 320 / 768 |")
        P("|---|---|---|---|---|---|---|---|---|---|")
        for s in SUBSETS:
            mem = [i for i in members_for(fl, s) if (fl, i, s) in rl3 and (fl, i, s) in sft3]
            if not mem:
                continue
            a3, a7, r3, r7 = (macro_of(M, fl, s, mem, "chrf") for M in (sft3, sft, rl3, rl))
            P(f"| {s} | {len(mem)} | {f1(a3, 2)} | {f1(a7, 2)} | {f1(r3, 2)} | {f1(r7, 2)} | {sg(r3 - a3, 2)} | {sg(r7 - a7, 2)} | "
              f"{pct(macro_of(sft3, fl, s, mem, 'finished'))} / {pct(macro_of(sft, fl, s, mem, 'finished'))} | {pct(macro_of(rl3, fl, s, mem, 'finished'))} / {pct(macro_of(rl, fl, s, mem, 'finished'))} |")
        P("")
        # SFT-only 320 on other benchmarks
        P("SFT-final at 320 tokens on all benchmarks is in `final_model_per_language_max320.csv` (RL at 320 was run on FLORES+ only).\n")
        sec = 7
    else:
        sec = 6
    # ---- no-benchmark languages
    P(f"## {sec}. Languages without a human-reference benchmark\n")
    P("| iso | language | group | RL-trained | why no benchmark |")
    P("|---|---|---|---|---|")
    for i in nob:
        why = "only word-level GATITOS (not scored here)" if L.LANGUAGES[i].gatitos else "no FLORES+ / NTREX / MAFAND / SmolSent / IbomMT / FLORES-MOZ / GlotStoryBook / SALT split in our mapping"
        P(f"| `{i}` | {names[i]} | {GROUPS[group_of(i)].split(' (')[0]} | {'yes' if L.LANGUAGES[i].rl_eligible else 'no'} | {why} |")
    P("\nThese rows are absent from the long CSV (no scores) and are marked `none` in section 3. For them the reward-model scores of the RL run are the only quality signal.\n")
    P(f"## {sec + 1}. Caveats\n")
    P("* **Non-comparable rows.** `shi` (Tachelhit) on NTREX has an Arabic-script reference while the model writes Latin transliteration: chrF++ 0.3 -> 1.0 is a script mismatch, not a quality measure (it is still inside the primary-benchmark macros of section 2.3; drop it for a clean average). `knc` and `nde` are zero-shot (no SFT data, see section 1). "
      "GlotLID right-language rates are not meaningful for `knc`, `nde` (0-3%), `kon` (no label) and are low for Pidgin / Nuer-like cases where GlotLID has no good class; compare SFT vs RL, not absolute values.")
    P("* FLORES+ `clean` rests on 48 sentences per language; per-language `clean` deltas are noisy (bootstrap CIs in the wide CSV), the macro is stable.")
    P("* All benchmarks are test splits that are never trained on, but the English side of 95% of FLORES+ devtest (91% of SmolSent) appears in the SFT text through other language pairs; `no_pair` removes only the rows whose own reference was seen, `clean` removes every row whose English sentence was seen. This audit covers the SFT text; the RL prompt pools were not re-audited against these test sets here.")
    P("* Macro averages weight a 48-sentence language like a 1,012-sentence one; the bootstrap resamples sentences within each language and averages languages with equal weight, i.e. it measures sentence-sampling noise, not language-sampling noise.")
    P("* Different benchmarks are not comparable with each other (different domains, lengths, reference quality); never read a difference between two benchmark columns as a difference between languages.")
    (ROOT / "reports" / "final_model_per_language_2026-10-21.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("wrote report; languages with benchmark:", len(primary), "without:", nob)


if __name__ == "__main__":
    main()
