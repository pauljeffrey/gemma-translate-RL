#!/usr/bin/env python
"""Step 3: wide per-language CSVs, markdown tables, LaTeX (booktabs) snippets, the figure, the inventory and the missing list.

usage: python s03_build.py DATA_DIR LATEX_DIR      (python with numpy + matplotlib; reads the CSVs of s02_compute.py; writes DATA_DIR/*.csv|md and LATEX_DIR/*.tex|pdf|png)
Nothing is computed here except formatting and the AfriScience-MT block (read from stored report data).
"""
import csv
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from frc_lib import LABEL, SHORT, REPO  # noqa

D, TEX = sys.argv[1], sys.argv[2]
os.makedirs(TEX, exist_ok=True)


def rd(name):
    return list(csv.DictReader(open(os.path.join(D, name))))


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


macro = {}
for r in rd("macro_table.csv"):
    macro[(r["group"], r["system"])] = r
perl = {}
for r in rd("per_language_long.csv"):
    perl[(r["group"], r["system"], r["iso"])] = r
checks = json.load(open(os.path.join(D, "checks.json")))
pubblock = rd("published_block.csv")
ssa = rd("ssa_reward_family.csv")
NAME = dict(re.findall(r'Language\("([a-z]{3})", "([^"]+)"', open(REPO + "gemma_mt/gemma_mt/languages.py").read()))
for _r in csv.DictReader(open(REPO + "reports/data/nllb600m_per_language_benchmark.csv")):
    NAME.setdefault(_r["iso"], _r["language"])
NAME.update({"knc": "Kanuri", "afr": "Afrikaans"})


def gl(g):
    return checks["groups"][g]["langs"]


def glc(g):
    return checks["groups"][g]["comet_langs"]


def fmt(x, d=1):
    x = fl(x)
    return "--" if np.isnan(x) else f"{x:.{d}f}"


def mrow(g, s):
    return macro.get((g, s))


# ------------------------------------------------------------------------------------------------ wide per-language CSVs
WIDE = [("F_NP", ["sft385", "e03c1", "stageA", "b1best", "mixed", "nllb600m", "nllb3b3", "tpsy08"]),
        ("F_ALL", ["sft385", "e03c1", "stageA", "b1best", "mixed", "nllb600m", "nllb3b3", "tpsy08"]),
        ("F_C270", ["sft385", "e03c1", "stageA", "b1best", "mixed", "old_sft", "old_rl", "nllb600m", "nllb3b3", "tpsy08"]),
        ("N_ALL", ["sft385", "e03c1", "stageA", "b1best", "mixed", "nllb3b3"]),
        ("N_C270", ["sft385", "e03c1", "stageA", "b1best", "mixed", "old_sft", "old_rl", "nllb3b3"]),
        ("M_ALL", ["sft385", "e03c1", "stageA", "b1best", "mixed", "nllb3b3"]),
        ("M_C270", ["sft385", "e03c1", "stageA", "b1best", "mixed", "old_sft", "old_rl", "nllb3b3"]),
        ("S_ALL", ["sft385", "nllb3b3"]), ("SALT_ALL", ["sft385", "nllb3b3"]), ("G_ALL", ["sft385"]), ("I_ALL", ["sft385"]), ("Z_ALL", ["sft385"])]
# TranslatePsy-0.8B is scored on 19 languages of the same sentences: its per-language cells are valid next to the other systems on the same rows
tp_rows = {r["iso"]: r for r in rd("per_language_long.csv") if r["group"] == "F_NP_T19" and r["system"] == "tpsy08"}
tp_rows_c = {r["iso"]: r for r in rd("per_language_long.csv") if r["group"] == "F_C270_T19" and r["system"] == "tpsy08"}
tp_rows_all = {r["iso"]: r for r in rd("per_language_long.csv") if r["group"] == "F_ALL_T19" and r["system"] == "tpsy08"}
tp_for = {"F_NP": tp_rows, "F_C270": tp_rows_c, "F_ALL": tp_rows_all}
for g, systems in WIDE:
    rows = []
    for iso in gl(g):
        row = {"iso": iso, "language": NAME.get(iso, iso)}
        first = perl[(g, systems[0], iso)]
        row["n_rows"] = first["n_rows"]
        for s in systems:
            r = perl.get((g, s, iso)) or (tp_for.get(g, {}).get(iso) if s == "tpsy08" else None)
            for m in ("chrfpp", "chrf", "spbleu", "africomet"):
                row[f"{s}_{m}"] = "" if r is None else r[m]
        rows.append(row)
    with open(os.path.join(D, f"per_language_wide_{g}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

# ------------------------------------------------------------------------------------------------ LaTeX helpers
TEXSYS = {"sft385": "SFT", "e03c1": r"SFT-ctl\textsuperscript{a}", "stageA": "RL-A", "b1best": "RL-B1", "mixed": "RL",
          "old_sft": r"270M SFT", "old_rl": r"270M RL", "nllb600m": "NLLB-600M", "nllb3b3": "NLLB-3.3B", "tpsy08": "TPsy-0.8B", "tpsy2b": "TPsy-2B"}


def tex_esc(s):
    return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


def best_marks(vals, higher=True):
    ok = [v for v in vals if not np.isnan(v)]
    if not ok:
        return [False] * len(vals)
    b = max(ok) if higher else min(ok)
    return [(not np.isnan(v)) and abs(v - b) < 1e-9 for v in vals]


def core_table(fname, caption, label, panels, systems, resize=True, star=False, notes="", sep_after=(), require=("nllb3b3",)):
    """panels: list of (header, group, [(metric, header, decimals)]).  Bold best per column among the listed systems."""
    cols = []
    for head, g, mets in panels:
        for met, mh, dec in mets:
            cols.append((head, g, met, mh, dec))
    vals = {}
    for s in systems:
        vals[s] = []
        for head, g, met, mh, dec in cols:
            r = mrow(g, s)
            v = np.nan
            if r is not None:
                v = fl(r[{"chrfpp": "chrfpp", "chrf": "chrf", "spbleu": "spbleu", "comet": "africomet"}[met]])
            vals[s].append(v)
    marks = []
    for j in range(len(cols)):
        complete = all(not np.isnan(vals[s][j]) for s in systems if s in require)
        marks.append(best_marks([vals[s][j] for s in systems]) if complete else [False] * len(systems))
    lines = []
    ncol = len(cols)
    spec = "@{}l" + "r" * ncol + "@{}"
    lines.append(r"\begin{%s}[t]" % ("table*" if star else "table"))
    lines.append(r"\centering\footnotesize")
    if resize:
        lines.append(r"\resizebox{%s}{!}{%%" % (r"\textwidth" if star else r"\columnwidth"))
    lines.append(r"\begin{tabular}{%s}" % spec)
    lines.append(r"\toprule")
    # header row 1: panel names with language counts
    hdr = [""]
    cm = []
    pos = 2
    for head, g, mets in panels:
        n1 = macro[(g, systems[0])]["n_lang"] if (g, systems[0]) in macro else ""
        nc = macro[(g, systems[0])]["n_lang_comet"] if (g, systems[0]) in macro else ""
        # comet count: from any per-sentence system of the group
        ncs = [macro[(g, s)]["n_lang_comet"] for s in systems if (g, s) in macro and macro[(g, s)]["n_lang_comet"] not in ("0", "")]
        ncc = ncs[0] if ncs else "0"
        txt = f"{head} ({n1}/{ncc})" if any(m == "comet" for m, _, _ in mets) and any(m != "comet" for m, _, _ in mets) else f"{head} ({n1})"
        hdr.append(r"\multicolumn{%d}{c}{%s}" % (len(mets), txt))
        cm.append(r"\cmidrule(lr){%d-%d}" % (pos, pos + len(mets) - 1))
        pos += len(mets)
    lines.append(" & ".join(hdr) + r" \\")
    lines.append("".join(cm))
    lines.append("System & " + " & ".join(mh for _, _, _, mh, _ in cols) + r" \\")
    lines.append(r"\midrule")
    for k, s in enumerate(systems):
        cells = []
        for j, (head, g, met, mh, dec) in enumerate(cols):
            t = fmt(vals[s][j], dec)
            if marks[j][k] and t != "--":
                t = r"\textbf{%s}" % t
            cells.append(t)
        lines.append(TEXSYS[s] + " & " + " & ".join(cells) + r" \\")
        if s in sep_after:
            lines.append(r"\midrule")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    if resize:
        lines.append("}")
    lines.append(r"\caption{%s}" % caption)
    lines.append(r"\label{%s}" % label)
    lines.append(r"\end{%s}" % ("table*" if star else "table"))
    open(os.path.join(TEX, fname), "w").write("\n".join(lines) + "\n")
    return vals


M = lambda g, s, k: fl(macro[(g, s)][k]) if (g, s) in macro else np.nan  # noqa
nsent = lambda g: f"{int(macro[(g, 'mixed')]['n_sentences']):,}"  # noqa

# ---- Table A (core): 385M models against NLLB-200, three benchmarks
capA = (r"English$\to$African translation, language-macro scores on identical sentences. "
        r"F+ = FLORES+ devtest sentences whose reference pair the earlier 270M SFT did not see (%s sentences, %s languages; 385M: 0 flagged rows); "
        r"NT = NTREX-128 and MF = MAFAND-MT test, all rows (385M contamination flags: 0.25\%% / 0.47\%% of rows, effect $<$0.01 chrF++). "
        r"Counts in parentheses: languages shared by all systems for chrF++ / for COMET. chrF++ = sacreBLEU word order 2; COMET = held-out reference-based AfriCOMET-STL-1.1 (0--1). "
        r"SFT, RL = our 385M models (RL = final model); SFT-ctl = SFT continued for the compute of the RL chain on TranslatePsy en$\to$xx plus replay (E-03). "
        r"Ours decode greedily, NLLB with beam 4. Bold = best in column (not set where NLLB-3.3B has no score). ``--'': not available (see text). "
        r"NLLB-3.3B on NT/MF is a per-language corpus-level aggregate on the same rows (no per-sentence output archived; no COMET). "
        r"\textsuperscript{a}Trained on NLLB-3.3B translations for 19 languages, never released.") % (nsent("F_NP"), macro[("F_NP", "mixed")]["n_lang"])
core_table("tabA_core_main_comparison.tex", capA, "tab:full-core",
           [("F+", "F_NP", [("chrfpp", "chrF++", 1), ("comet", "COMET", 3)]),
            ("NT", "N_ALL", [("chrfpp", "chrF++", 1), ("comet", "COMET", 3)]),
            ("MF", "M_ALL", [("chrfpp", "chrF++", 1), ("comet", "COMET", 3)])],
           ["sft385", "e03c1", "mixed", "nllb600m", "nllb3b3"], sep_after=("mixed",))

# ---- Table A2: TranslatePsy language set (19 languages), four metrics
capA2 = (r"FLORES+ devtest, the 19 languages of TranslatePsy-AfriSLM-0.8B (afr amh hau ibo kin lin lug mlg nya orm sna som sot swh tsn wol xho yor zul), same %s sentences for every system "
         r"(reference pair not seen by the earlier 270M SFT). Language macro; chrF++, chrF, spBLEU (flores200 tokenizer) and held-out reference-based AfriCOMET-STL-1.1. "
         r"TPsy-0.8B was distilled from NLLB-3.3B outputs on these languages. SFT-ctl\textsuperscript{a} as in Table~\ref{tab:full-core}. Bold = best in column.") % nsent("F_NP_T19")
core_table("tabA2_translatepsy19.tex", capA2, "tab:full-tpsy19",
           [("FLORES+ TPsy-19 set", "F_NP_T19", [("chrfpp", "chrF++", 1), ("chrf", "chrF", 1), ("spbleu", "spBLEU", 1), ("comet", "COMET", 3)])],
           ["sft385", "e03c1", "mixed", "nllb600m", "nllb3b3", "tpsy08"], sep_after=("mixed", "nllb3b3"))

# ---- Table A3: rows clean for the earlier 270M too, with the earlier 270M SFT / RL
capA3 = (r"The same comparison on the only rows that are clean for every system including the earlier 270M models (English side not in the 270M SFT text): "
         r"FLORES+ 48 sentences per language (%s sentences), NTREX and MAFAND %s and %s sentences. 270M = the first research's SFT / RL-final models regenerated in the 385M harness (E-13). "
         r"Language macro, counts in parentheses (chrF++ / COMET languages). Bold = best in column. NLLB-3.3B on NT/MF as in Table~\ref{tab:full-core}.") % (nsent("F_C270"), nsent("N_C270"), nsent("M_C270"))
core_table("tabA3_with_earlier_270M_clean_rows.tex", capA3, "tab:full-270m",
           [("F+", "F_C270", [("chrfpp", "chrF++", 1), ("comet", "COMET", 3)]),
            ("NT", "N_C270", [("chrfpp", "chrF++", 1), ("comet", "COMET", 3)]),
            ("MF", "M_C270", [("chrfpp", "chrF++", 1), ("comet", "COMET", 3)])],
           ["old_sft", "old_rl", "sft385", "e03c1", "mixed", "nllb600m", "nllb3b3"], sep_after=("old_rl", "mixed"))

# ---- all-systems macro table (appendix): stages included
capC = (r"All systems, language macro, chrF++ and held-out AfriCOMET-STL-1.1 (per-sentence rows on identical sentences within a panel). "
        r"RL-A = stage A final, RL-B1 = B1 best, RL = final (mixed best). Bold = best in column; ``--'' = not available.")
core_table("tabC_all_systems_macro.tex", capC, "tab:full-all",
           [("F+ no\\_pair", "F_NP", [("chrfpp", "chrF++", 1), ("comet", "COMET", 3)]),
            ("F+ clean", "F_C270", [("chrfpp", "chrF++", 1), ("comet", "COMET", 3)]),
            ("NT all", "N_ALL", [("chrfpp", "chrF++", 1), ("comet", "COMET", 3)]),
            ("MF all", "M_ALL", [("chrfpp", "chrF++", 1), ("comet", "COMET", 3)])],
           ["sft385", "e03c1", "stageA", "b1best", "mixed", "old_sft", "old_rl", "nllb600m", "nllb3b3"], star=True, sep_after=("mixed", "old_rl"))
# the old 270M rows are only valid in the clean panel: blank them in the other panels (they are not in those groups, so cells are already '--')


# ---- paired differences table (RL minus system with 95% paired bootstrap intervals)
def paired_table():
    rows = [("F_NP", "F+ no\\_pair"), ("F_NP_T19", "F+ TPsy-19"), ("N_ALL", "NT all"), ("M_ALL", "MF all"), ("F_C270", "F+ clean")]
    sysl = ["sft385", "e03c1", "stageA", "b1best", "old_sft", "old_rl", "nllb600m", "nllb3b3", "tpsy08"]
    lines = [r"\begin{table*}[t]", r"\centering\footnotesize", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{@{}ll" + "r" * 2 + "rr@{}}", r"\toprule",
             r"Row set (langs) & RL $-$ system & $\Delta$chrF++ & 95\% CI & $\Delta$COMET & 95\% CI \\", r"\midrule"]
    for g, nm in rows:
        first = True
        for s in sysl:
            m = macro.get((g, s))
            if m is None or m.get("d_chrfpp_RL_minus_sys", "") == "":
                continue
            n1 = m["n_lang"]
            nc = m["n_lang_comet"]
            dc = fl(m.get("d_africomet_RL_minus_sys", ""))
            ci1 = f"[{fmt(m['d_chrfpp_lo'], 2)}, {fmt(m['d_chrfpp_hi'], 2)}]" + ("$^{\\ell}$" if m.get("d_ci_kind", "").startswith("language") else "")
            ci2 = "--" if np.isnan(dc) else f"[{fmt(m['d_africomet_lo'], 4)}, {fmt(m['d_africomet_hi'], 4)}]"
            lines.append(f"{(nm + ' (' + n1 + '/' + nc + ')') if first else ''} & {TEXSYS[s]} & {fmt(m['d_chrfpp_RL_minus_sys'], 2)} & {ci1} & {'--' if np.isnan(dc) else fmt(dc, 4)} & {ci2} \\\\")
            first = False
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}%", "}",
              r"\caption{Final RL model minus each other system, language-macro difference with 95\% paired bootstrap interval (1,000 resamples, Poisson weights per source sentence shared across systems and, for FLORES+ and NTREX, across languages; seed 20261104). "
              r"Positive = RL better. Counts: languages for chrF++ / COMET. $^{\ell}$: no per-sentence output of NLLB-3.3B on NTREX/MAFAND is archived; the interval is a bootstrap over languages (language sampling only). "
              r"SFT-ctl\textsuperscript{a}: see Table~\ref{tab:full-core}.}", r"\label{tab:full-paired}", r"\end{table*}"]
    open(os.path.join(TEX, "tabD_paired_differences.tex"), "w").write("\n".join(lines) + "\n")


paired_table()


# ---- per-language appendix tables
def perlang_table(fname, caption, label, g, systems, metric, dec, nrows_col=True, with_n=True):
    isos = gl(g)
    cols = systems
    vals = {s: [fl((perl.get((g, s, i)) or (tp_for.get(g, {}).get(i) if s == "tpsy08" else None) or {}).get(metric, "")) for i in isos] for s in cols}
    lines = [r"\begin{table*}[t]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}ll" + ("r" if with_n else "") + "r" * len(cols) + "@{}}", r"\toprule",
             "ISO & Language & " + ("$n$ & " if with_n else "") + " & ".join(TEXSYS[s] for s in cols) + r" \\", r"\midrule"]
    for k, i in enumerate(isos):
        cells = []
        bests = best_marks([vals[s][k] for s in cols])
        for j, s in enumerate(cols):
            t = fmt(vals[s][k], dec)
            if bests[j] and t != "--":
                t = r"\textbf{%s}" % t
            cells.append(t)
        n = perl[(g, systems[0], i)]["n_rows"]
        lines.append(f"{i} & {tex_esc(NAME.get(i, i))} & " + (f"{n} & " if with_n else "") + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    mac = []
    for s in cols:
        v = np.array(vals[s])
        mac.append("--" if np.isnan(v).any() else fmt(np.mean(v), dec))
    # macro over the languages of the table where the system has a value (TranslatePsy: 19)
    mac2 = []
    for s in cols:
        v = np.array(vals[s])
        mac2.append(fmt(np.nanmean(v), dec) if (~np.isnan(v)).sum() > 0 else "--")
    lines.append(r"Macro (languages with a value) & & " + ("& " if with_n else "") + " & ".join(mac2) + r" \\")
    lines.append(r"Languages with a value & & " + ("& " if with_n else "") + " & ".join(str(int((~np.isnan(np.array(vals[s]))).sum())) for s in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\caption{%s}" % caption, r"\label{%s}" % label, r"\end{table*}"]
    open(os.path.join(TEX, fname), "w").write("\n".join(lines) + "\n")


sysF = ["sft385", "e03c1", "mixed", "nllb600m", "nllb3b3", "tpsy08"]
nF = nsent("F_NP")
perlang_table("tabB1_flores_perlang_chrfpp.tex",
              r"FLORES+ devtest, chrF++ per language on the no\_pair rows ($n$ sentences per language; identical for every system). ``--'': language not covered by the system. "
              r"The last rows are macros over the languages with a value, so columns have different language sets (use the main tables for comparisons).",
              "tab:full-flores-chrfpp", "F_NP", sysF, "chrfpp", 1)
perlang_table("tabB2_flores_perlang_africomet.tex",
              r"FLORES+ devtest, held-out reference-based AfriCOMET-STL-1.1 per language, same sentences as the chrF++ table (languages outside the AfriCOMET encoder card are ``--'').",
              "tab:full-flores-comet", "F_NP", sysF, "africomet", 3)
perlang_table("tabB3_flores_perlang_chrf.tex", r"FLORES+ devtest, plain chrF (word order 0) per language, same sentences as Table~\ref{tab:full-flores-chrfpp}.",
              "tab:full-flores-chrf", "F_NP", sysF, "chrf", 1)
perlang_table("tabB4_flores_perlang_spbleu.tex", r"FLORES+ devtest, spBLEU (flores200 SentencePiece tokenizer) per language, same sentences as Table~\ref{tab:full-flores-chrfpp}.",
              "tab:full-flores-spbleu", "F_NP", sysF, "spbleu", 1)
sysNM = ["sft385", "e03c1", "mixed", "nllb3b3"]
perlang_table("tabB5_ntrex_perlang_chrfpp.tex", r"NTREX-128, chrF++ per language, all rows. NLLB-3.3B is a corpus-level aggregate on the same rows.", "tab:full-ntrex-chrfpp", "N_ALL", sysNM, "chrfpp", 1)
perlang_table("tabB6_ntrex_perlang_africomet.tex", r"NTREX-128, held-out AfriCOMET-STL-1.1 per language (no NLLB score exists on these rows).", "tab:full-ntrex-comet", "N_ALL", ["sft385", "e03c1", "mixed"], "africomet", 3)
perlang_table("tabB7_mafand_perlang_chrfpp.tex", r"MAFAND-MT test, chrF++ per language, all rows. NLLB-3.3B is a corpus-level aggregate on the same rows.", "tab:full-mafand-chrfpp", "M_ALL", sysNM, "chrfpp", 1)
perlang_table("tabB8_mafand_perlang_africomet.tex", r"MAFAND-MT test, held-out AfriCOMET-STL-1.1 per language (no NLLB score exists on these rows).", "tab:full-mafand-comet", "M_ALL", ["sft385", "e03c1", "mixed"], "africomet", 3)
perlang_table("tabB9_smolsent_perlang.tex", r"SmolSent, chrF++ per language, all rows: 385M SFT and NLLB-3.3B (corpus-level aggregate). No RL model was evaluated on SmolSent.", "tab:full-smol-chrfpp", "S_ALL", ["sft385", "nllb3b3"], "chrfpp", 1)
perlang_table("tabB10_salt_perlang.tex", r"SALT, chrF++ per language, all rows: 385M SFT and NLLB-3.3B (corpus-level aggregate). No RL model was evaluated on SALT.", "tab:full-salt-chrfpp", "SALT_ALL", ["sft385", "nllb3b3"], "chrfpp", 1)

# ------------------------------------------------------------------------------------------------ AfriScience-MT block (stored values; not recomputed)
sn = json.load(open(os.path.join(REPO, "research_archive_2026-11-03/box_final_2026-11-04/e03_control/run_records_and_eval/sent_neural_c1.json")))
afs_rows = []
SN = {"sft": "385M SFT", "stageA": "385M RL stage A", "b1best": "385M RL B1 best", "mixedbest": "385M RL (mixed best = Hub main)", "c1_s6000": "385M SFT-only control (E-03)"}
LANGS6 = ["amh", "hau", "lug", "nso", "yor", "zul"]
for k, lab in SN.items():
    t = sn[k]["test"]
    afs_rows.append(dict(system=lab, kind="our run, same 5,820 test sentences, our scoring", chrF_wo0={l: t[l]["chrf"] for l in LANGS6}, chrfpp=float(np.mean([t[l]["chrfpp"] for l in LANGS6])),
                         africomet=float(np.mean([t[l]["africomet_stl"] for l in LANGS6])), ssa_qe=float(np.mean([t[l]["ssa_qe"] for l in LANGS6]))))
pub_afs = []
for r in csv.DictReader(open(os.path.join(REPO, "reports/data/eval_385m_results_2026-11-03/afriscience_sentence.csv"))):
    if r["split"] != "test":
        continue
    if "old 270M" in r["system"] or "published" in r["system"]:
        pub_afs.append(r)
with open(os.path.join(D, "afriscience_block.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["system", "kind", "amh", "hau", "lug", "nso", "yor", "zul", "chrF_macro6", "chrFpp_macro6", "AfriCOMET_STL_macro6", "SSA_COMET_QE_macro6_REWARD_FAMILY"])
    for r in afs_rows:
        w.writerow([r["system"], r["kind"]] + [round(r["chrF_wo0"][l], 2) for l in LANGS6] + [round(float(np.mean(list(r["chrF_wo0"].values()))), 2), round(r["chrfpp"], 2), round(r["africomet"], 4), round(r["ssa_qe"], 4)])
    for r in pub_afs:
        kind = "published number (the AfriScience-MT paper's scoring)" if "published" in r["system"] else "first research's own run, stored value (earlier code state, not rerun)"
        w.writerow([r["system"], kind] + [round(float(r[l]), 2) for l in LANGS6] + [round(float(r["macro"]), 2), "", "", ""])

# ------------------------------------------------------------------------------------------------ markdown tables
md = []


def mdtable(title, g, systems, extra_note=""):
    md.append(f"#### {title}\n")
    md.append(f"Row set `{macro[(g, systems[0])]['row_set']}`, {macro[(g, systems[0])]['n_sentences']} sentences, {macro[(g, systems[0])]['n_lang']} languages (AfriCOMET: {max([int(macro[(g, s)]['n_lang_comet'] or 0) for s in systems])}). {extra_note}\n")
    md.append("| system | chrF++ [95% CI] | chrF | spBLEU | AfriCOMET [95% CI] | RL minus system, chrF++ [CI] | RL minus system, AfriCOMET [CI] | langs RL above |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for s in systems:
        m = macro.get((g, s))
        if m is None:
            continue
        a = f"{fmt(m['chrfpp'], 2)}" + (f" [{fmt(m['chrfpp_lo'], 2)}, {fmt(m['chrfpp_hi'], 2)}]" if m.get("chrfpp_lo") not in ("", None) else "")
        c = fmt(m["africomet"], 4) + (f" [{fmt(m['africomet_lo'], 4)}, {fmt(m['africomet_hi'], 4)}]" if m.get("africomet_lo") not in ("", None) else "")
        d1 = "" if m.get("d_chrfpp_RL_minus_sys", "") == "" else f"{fmt(m['d_chrfpp_RL_minus_sys'], 2)} [{fmt(m['d_chrfpp_lo'], 2)}, {fmt(m['d_chrfpp_hi'], 2)}]" + (" (language bootstrap)" if m.get("d_ci_kind", "").startswith("language") else "")
        d2 = "" if m.get("d_africomet_RL_minus_sys", "") == "" else f"{fmt(m['d_africomet_RL_minus_sys'], 4)} [{fmt(m['d_africomet_lo'], 4)}, {fmt(m['d_africomet_hi'], 4)}]"
        md.append(f"| {LABEL[s]} | {a} | {fmt(m['chrf'], 2)} | {fmt(m['spbleu'], 2)} | {c} | {d1} | {d2} | {m.get('n_lang_RL_above', '')} |")
    md.append("")


S5 = ["sft385", "e03c1", "stageA", "b1best", "mixed"]
mdtable("T1. FLORES+ devtest, no_pair rows, NLLB-covered languages (headline)", "F_NP", S5 + ["nllb600m", "nllb3b3"])
mdtable("T2. FLORES+ devtest, no_pair rows, TranslatePsy-0.8B language set", "F_NP_T19", S5 + ["nllb600m", "nllb3b3", "tpsy08"])
mdtable("T2b. FLORES+ devtest, no_pair rows, TranslatePsy-2B language set (9 languages)", "F_NP_T9", S5 + ["nllb600m", "nllb3b3", "tpsy08", "tpsy2b"])
mdtable("T3. NTREX-128, all rows", "N_ALL", S5 + ["nllb3b3"], "NLLB-3.3B: corpus-level per-language aggregate on the same rows (no per-sentence output, no AfriCOMET).")
mdtable("T4. MAFAND-MT test, all rows", "M_ALL", S5 + ["nllb3b3"], "NLLB-3.3B: corpus-level per-language aggregate on the same rows.")
mdtable("T5. FLORES+ devtest, rows clean for the earlier 270M too", "F_C270", S5 + ["old_sft", "old_rl", "nllb600m", "nllb3b3"])
mdtable("T5b. FLORES+ clean-for-270M rows, TranslatePsy-0.8B language set", "F_C270_T19", S5 + ["old_sft", "old_rl", "nllb600m", "nllb3b3", "tpsy08"])
mdtable("T6. NTREX-128, rows clean for the earlier 270M", "N_C270", S5 + ["old_sft", "old_rl", "nllb3b3"])
mdtable("T7. MAFAND-MT, rows clean for the earlier 270M", "M_C270", S5 + ["old_sft", "old_rl", "nllb3b3"])
mdtable("T8. FLORES+ devtest, all 1,012 rows (chrF-family only; AfriCOMET of NLLB exists only on the no_pair rows)", "F_ALL", S5 + ["nllb600m", "nllb3b3"])
mdtable("T9. SmolSent, all rows", "S_ALL", ["sft385", "nllb3b3"])
mdtable("T10. SALT, all rows", "SALT_ALL", ["sft385", "nllb3b3"])
mdtable("T11a. GlotStoryBook (385M SFT only)", "G_ALL", ["sft385"])
mdtable("T11b. IbomMT (385M SFT only)", "I_ALL", ["sft385"])
mdtable("T11c. FLORES-MOZ (385M SFT only)", "Z_ALL", ["sft385"])
md.append("#### T12. Published whole-devtest block (FLORES-200 devtest chrF++ of the NLLB release; our systems rescored on FLORES+ all 1,012 rows; not sentence-paired)\n")
md.append("| system | what | languages | chrF++ macro | RL minus system | 95% CI (language bootstrap, 10,000) | languages RL above |")
md.append("|---|---|---:|---:|---:|---|---:|")
for r in pubblock:
    md.append(f"| {r['system']} | {r['kind']} | {r['n_lang']} | {fmt(r['chrfpp_macro'], 2)} | {fmt(r.get('RL_minus_system', ''), 2)} | " + (f"[{fmt(r['lang_boot_lo'], 2)}, {fmt(r['lang_boot_hi'], 2)}]" if r.get("lang_boot_lo") else "") + f" | {r.get('n_lang_RL_above', '')} |")
md.append("")
md.append("#### T13. SSA-COMET-MTL reference-based x100 (REWARD FAMILY, NOT INDEPENDENT; FLORES+ no_pair rows; none exists for the 385M models)\n")
md.append("| system | languages | macro on its own languages | macro on the 19 TranslatePsy-0.8B languages | macro on the 9 TranslatePsy-2B languages |")
md.append("|---|---:|---:|---:|---:|")
for r in ssa:
    md.append(f"| {r['system']} | {r['n_lang']} | {fmt(r['ssa_comet_mtl_ref_x100_macro'], 2)} | {fmt(r.get('macro19_x100', ''), 2)} | {fmt(r.get('macro9_x100', ''), 2)} |")
md.append("")
md.append("#### T14. AfriScience-MT test sentences, 6 languages (amh hau lug nso yor zul), plain chrF (word order 0)\n")
md.append("| system | kind | chrF macro | chrF++ | AfriCOMET-STL | SSA-COMET-QE (reward family) |")
md.append("|---|---|---:|---:|---:|---:|")
for r in rd("afriscience_block.csv"):
    md.append(f"| {r['system']} | {r['kind']} | {r['chrF_macro6']} | {r['chrFpp_macro6']} | {r['AfriCOMET_STL_macro6']} | {r['SSA_COMET_QE_macro6_REWARD_FAMILY']} |")
md.append("")
open(os.path.join(D, "tables_generated.md"), "w").write("\n".join(md) + "\n")

# ------------------------------------------------------------------------------------------------ figure
import matplotlib  # noqa
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa

BLUE, ORANGE, GREEN, VERM = "#0072B2", "#E69F00", "#009E73", "#D55E00"
SKY, PURPLE, YELLOW, GREY, BLACK = "#56B4E9", "#CC79A7", "#F0E442", "#8C8C8C", "#000000"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5, "legend.frameon": False, "legend.fontsize": 8, "figure.dpi": 100, "savefig.dpi": 200, "pdf.fonttype": 42})
COL = {"sft385": GREY, "e03c1": PURPLE, "mixed": BLUE, "nllb600m": ORANGE, "nllb3b3": VERM, "tpsy08": GREEN, "stageA": SKY}
LEG = {"sft385": "385M SFT", "e03c1": "385M SFT-only control (E-03)", "mixed": "385M RL (final)", "nllb600m": "NLLB-200-distilled-600M", "nllb3b3": "NLLB-200-3.3B", "tpsy08": "TranslatePsy-0.8B"}


def panel(ax, groups, titles, systems, metric, lo_key, hi_key, ylabel, yl=None):
    w = 0.8 / len(systems)
    for gi, g in enumerate(groups):
        for si, s in enumerate(systems):
            m = macro.get((g, s))
            v = np.nan if m is None else fl(m[metric])
            x = gi + (si - (len(systems) - 1) / 2) * w
            if np.isnan(v):
                ax.text(x, (yl[0] if yl else 0) + 0.3 * ((yl[1] - yl[0]) / 20 if yl else 1), "n/a", rotation=90, ha="center", va="bottom", fontsize=6.5, color=GREY)
                continue
            err = None
            if m.get(lo_key) not in ("", None):
                err = [[v - fl(m[lo_key])], [fl(m[hi_key]) - v]]
            ax.bar(x, v, w * 0.92, color=COL[s], yerr=err, error_kw=dict(lw=0.8, capsize=1.5), label=LEG[s] if gi == 0 else None)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(titles)
    ax.set_ylabel(ylabel)
    if yl:
        ax.set_ylim(*yl)


main_groups = ["F_NP", "N_ALL", "M_ALL"]
titles = [f"FLORES+ no_pair\n({macro[('F_NP','mixed')]['n_lang']}/{macro[('F_NP','mixed')]['n_lang_comet']} lang.)", f"NTREX all\n({macro[('N_ALL','mixed')]['n_lang']}/{macro[('N_ALL','mixed')]['n_lang_comet']})", f"MAFAND all\n({macro[('M_ALL','mixed')]['n_lang']}/{macro[('M_ALL','mixed')]['n_lang_comet']})"]
fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4), gridspec_kw=dict(width_ratios=[3.1, 3.1, 3.1]))
sys_main = ["sft385", "e03c1", "mixed", "nllb600m", "nllb3b3"]
panel(axes[0], main_groups, titles, sys_main, "chrfpp", "chrfpp_lo", "chrfpp_hi", "chrF++ (language macro)", (20, 48))
axes[0].set_title("(a) chrF++")
panel(axes[1], main_groups, titles, sys_main, "africomet", "africomet_lo", "africomet_hi", "AfriCOMET-STL-1.1 (held out, ref.-based)", (0.5, 0.72))
axes[1].set_title("(b) AfriCOMET (NLLB: not scored on NTREX/MAFAND)")
sys_t = ["sft385", "e03c1", "mixed", "nllb600m", "nllb3b3", "tpsy08"]
# panel (c): TranslatePsy-19 set, chrF++ (left axis) -- AfriCOMET shown as text above bars
ax = axes[2]
w = 0.8 / len(sys_t)
for si, s in enumerate(sys_t):
    m = macro[("F_NP_T19", s)]
    v = fl(m["chrfpp"])
    err = [[v - fl(m["chrfpp_lo"])], [fl(m["chrfpp_hi"]) - v]]
    ax.bar(si * w - 0.4 + w / 2, v, w * 0.92, color=COL[s], yerr=err, error_kw=dict(lw=0.8, capsize=1.5), label=LEG[s] if s == "tpsy08" else None)
    ax.text(si * w - 0.4 + w / 2, 24.3, f"{fl(m['africomet']):.3f}", rotation=90, ha="center", va="bottom", fontsize=7, color="white")
ax.set_xticks([0])
ax.set_xticklabels([f"FLORES+ no_pair, TPsy set\n({macro[('F_NP_T19','mixed')]['n_lang']} lang.; white = AfriCOMET)"])
ax.set_ylim(24, 50)
ax.set_title("(c) the 19 TranslatePsy languages")
ax.set_ylabel("chrF++ (language macro)")
h, l = [], []
for a in axes:
    hh, ll = a.get_legend_handles_labels()
    for x_, y_ in zip(hh, ll):
        if y_ not in l:
            h.append(x_); l.append(y_)
fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12))
fig.suptitle("Our 385M models against other systems, English$\\to$African, identical sentences per panel (error bars: 95% paired sentence bootstrap; none for NLLB-3.3B on NTREX/MAFAND)", fontsize=8.5, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(TEX, "fig_full_results.pdf"), bbox_inches="tight", metadata={"Creator": None, "Producer": None, "CreationDate": None})
fig.savefig(os.path.join(TEX, "fig_full_results.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------------------------------------ inventory and missing list
inv = []


def add(system, bench, metric, status, level, rowset, langs, src):
    inv.append(dict(system=system, benchmark=bench, metric=metric, status=status, level=level, row_set=rowset, languages=langs, source=src))


A = "research_archive_2026-11-03/"
for s, lab in [("sft385", "385M SFT"), ("e03c1", "385M SFT-only control (E-03)")]:
    for b, nl, nlc in [("FLORES+ devtest", gl("F_ALL").__len__(), 40), ("NTREX-128", 21, 21), ("MAFAND-MT", 14, 14)]:
        add(lab, b, "chrF++/chrF/spBLEU", "per-sentence", "sentence", "all, no_pair, clean270", f"{nl} (shared with NLLB)", "predictions.jsonl rescored (s01_rescore.py)")
        add(lab, b, "AfriCOMET-STL-1.1 ref.", "per-sentence", "sentence", "all rows (AfriCOMET-card languages)", f"{nlc}", "evidence npz comet column")
for b in ["SmolSent", "SALT", "GlotStoryBook", "IbomMT", "FLORES-MOZ"]:
    add("385M SFT", b, "chrF++/chrF/spBLEU", "per-sentence", "sentence", "all", "", A + "eval_385m/out_m385/predictions.jsonl")
    add("385M SFT-only control (E-03)", b, "all", "MISSING", "", "", "", "control was evaluated on FLORES+, NTREX, MAFAND only")
for s, lab in [("stageA", "385M RL stage A"), ("b1best", "385M RL B1 best"), ("mixed", "385M RL (mixed best = Hub main)")]:
    for b in ["FLORES+ devtest", "NTREX-128", "MAFAND-MT"]:
        add(lab, b, "chrF++/chrF/spBLEU/AfriCOMET", "per-sentence", "sentence", "all, no_pair, clean270", "", A + "eval_385m_rl_predictions_2026-11-04/ (+ evidence npz for AfriCOMET)")
    for b in ["SmolSent", "SALT", "GlotStoryBook", "IbomMT", "FLORES-MOZ"]:
        add(lab, b, "all", "MISSING", "", "", "", "RL models were evaluated on FLORES+, NTREX, MAFAND only")
for s, lab in [("old_sft", "270M SFT (earlier, E-13)"), ("old_rl", "270M RL-final (earlier, E-13)")]:
    for b in ["FLORES+ devtest", "NTREX-128", "MAFAND-MT"]:
        add(lab, b, "chrF++/chrF/spBLEU/AfriCOMET", "per-sentence", "sentence", "clean270 (all rows exist but are exposure-inflated)", "", A + "box_outputs_2026-11-04/out_old_*/predictions.jsonl")
    for b in ["SmolSent", "SALT", "GlotStoryBook", "IbomMT", "FLORES-MOZ"]:
        add(lab, b, "all", "MISSING (E-13 reran 3 benchmarks only)", "", "", "", "")
add("NLLB-200-distilled-600M", "FLORES+ devtest", "chrF++/chrF/spBLEU", "per-sentence (beam 4, CT2 fp16, 1,012 rows x 44 languages)", "sentence", "all, no_pair, clean270", "44", "research_archive_2026-10-23/box/jobs/e2/scores/nllb600m.pkl.gz")
add("NLLB-200-distilled-600M", "FLORES+ devtest", "AfriCOMET-STL-1.1 ref.", "per-sentence", "sentence", "no_pair rows only (pair-seen rows never scored)", "42 (40 shared with ours)", "e2/scores/nllb600m.comet.pkl")
add("NLLB-200-distilled-600M", "FLORES+ devtest", "SSA-COMET-MTL ref. (reward family)", "per-sentence", "sentence", "no_pair rows", "22", "e2/scores/nllb600m.ssa.pkl")
for b in ["NTREX-128", "MAFAND-MT", "SmolSent", "SALT"]:
    add("NLLB-200-distilled-600M", b, "all", "MISSING", "", "", "", "R14 ran 600M on FLORES+ only (its columns are empty in nllb600m_per_language_benchmark.csv)")
add("NLLB-200-3.3B", "FLORES+ devtest", "chrF++/chrF/spBLEU", "per-sentence (beam 4, CT2 int8)", "sentence", "all, no_pair, clean270", "44", "e2/scores/nllb3b3.pkl.gz")
add("NLLB-200-3.3B", "FLORES+ devtest", "AfriCOMET-STL-1.1 ref.", "per-sentence", "sentence", "no_pair rows only", "42 (40 shared)", "e2/scores/nllb3b3.comet.pkl")
add("NLLB-200-3.3B", "FLORES+ devtest", "SSA-COMET-MTL ref. (reward family)", "per-sentence", "sentence", "no_pair rows", "22", "e2/scores/nllb3b3.ssa.pkl")
for b, nl, src in [("NTREX-128", 21, "all, no_pair, clean"), ("MAFAND-MT", 14, "all, no_pair, clean"), ("SmolSent", 36, "all, no_pair, clean"), ("SALT", 3, "all, no_pair, clean")]:
    add("NLLB-200-3.3B", b, "chrF++ and spBLEU", "aggregate only (corpus-level per language)", "language", src, str(nl), "reports/data/nllb600m_per_language_benchmark.csv (per-sentence outputs not archived)")
    add("NLLB-200-3.3B", b, "chrF (plain), AfriCOMET, SSA-COMET", "MISSING", "", "", "", "")
for lab in ["NLLB-200-1.3B", "NLLB-200-distilled-1.3B", "NLLB-200-MoE-54.5B"]:
    add(lab, "FLORES-200 devtest", "chrF++", "published only", "language", "whole devtest (NLLB release metrics file)", "43", "reports/data/eval_385m_results_2026-11-03/published_nllb_flores_per_language.csv")
add("NLLB-200-3.3B (published)", "FLORES-200 devtest", "chrF++", "published only", "language", "whole devtest", "43", "same file")
add("TranslatePsy-AfriSLM-0.8B", "FLORES+ devtest", "chrF++/chrF/spBLEU/AfriCOMET/SSA-COMET", "per-sentence (vLLM greedy, 256 tokens)", "sentence", "all rows (chrF); no_pair rows (AfriCOMET, SSA)", "19", "e2/scores/tpsy08.*")
add("TranslatePsy-AfriSLM-2B", "FLORES+ devtest", "chrF++/chrF/spBLEU/AfriCOMET/SSA-COMET", "per-sentence", "sentence", "no_pair rows", "9 of 19 (stopped by the GPU cap)", "e2/scores/tpsy2b.*")
add("TranslatePsy-AfriSLM-4B", "FLORES+ devtest", "all", "MISSING (not run; published numbers are bidirectional 19-language aggregates, not comparable)", "", "", "", "")
for b in ["NTREX-128", "MAFAND-MT", "SmolSent", "SALT"]:
    add("TranslatePsy-AfriSLM-0.8B / 2B / 4B", b, "all", "MISSING", "", "", "", "")
add("AfriqueLLM family (AfriqueGemma/Llama/Qwen, 4-14B), base LLMs", "FLORES-200 (AfroBench-Lite, 5-shot)", "chrF++ (their implementation)", "published only (21 languages)", "language", "split not stated", "21", "reports/data/published_baselines.csv; reports/data/afriquellm_table20_ssa_comet_enxx.csv (SSA-COMET, unnamed checkpoint)")
add("AfriNLLB-600M", "FLORES-200 devtest", "AfriCOMET-mtl v1.0 (NOT stl-1.1; not comparable)", "published only (9 languages)", "language", "", "9", "AfriNLLB paper Table 7 via reports/model_comparison_2026-10-23.md")
add("Fine-tuned NLLB-1.3B / 600M, M2M100, Gemma2-9B, GPT-5.4, Gemini-3.1-Flash-Lite", "AfriScience-MT test", "chrF (word order 0)", "published only", "language", "test split, 6 languages", "6", "reports/data/eval_385m_results_2026-11-03/afriscience_sentence.csv")
with open(os.path.join(D, "inventory.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["system", "benchmark", "metric", "status", "level", "row_set", "languages", "source"])
    w.writeheader()
    w.writerows(inv)

missing = [
    dict(gap="NLLB-200-3.3B per-sentence outputs on NTREX-128 (21 languages) and MAFAND-MT (14 languages)", why="only corpus-level aggregates exist, so no paired bootstrap, no AfriCOMET, no plain chrF on these benchmarks", run="R14-style CT2 int8 beam-4 translation, config nllb3b3_beam4 on ntrex,mafand (script research_archive_2026-10-23/box/jobs/r14/code/nllb_run.py)", a100_hours="1.2 (ESTIMATE: R14 logged 25,961 NTREX sentences in 1,532 s; 41.9k NTREX + 21k MAFAND sentences)"),
    dict(gap="NLLB-200-distilled-600M on NTREX-128, MAFAND-MT, SmolSent, SALT (any form)", why="R14 ran the 600M model on FLORES+ only", run="nllb_run.py with NLLB_MODEL=600m, beam4, benches ntrex,mafand,smol_sent,salt", a100_hours="0.5 (ESTIMATE: the 600M model translated 44.5k FLORES+ sentences greedily in 353 s; beam 4 about 3x)"),
    dict(gap="NLLB-200-3.3B per-sentence outputs on SmolSent (36 languages) and SALT (3)", why="aggregates only; no AfriCOMET", run="nllb_run.py NLLB_MODEL=3b3 beam4 on smol_sent,salt", a100_hours="0.6 (ESTIMATE, 32k sentences)"),
    dict(gap="AfriCOMET-STL-1.1 (ref.-based) for the NLLB outputs above and for the 1.8k FLORES+ pair-seen rows", why="NLLB / TranslatePsy were scored on the no_pair rows only", run="e2_comet.py-style scoring (masakhane/africomet-stl-1.1, fp16) of the new hypothesis files", a100_hours="0.3 (ESTIMATE, about 150k sentence triples at about 150/s)"),
    dict(gap="TranslatePsy-AfriSLM-0.8B on NTREX-128 (languages shared with its 19) and MAFAND-MT", why="only FLORES+ was run", run="e2_gen_chat.py tpsy08 (vLLM greedy, 256 tokens, model-card prompt) on ntrex, mafand; then AfriCOMET", a100_hours="0.4 (ESTIMATE incl. scoring; FLORES+ 19 x 1,012 sentences took about 5 min)"),
    dict(gap="TranslatePsy-AfriSLM-2B: remaining 10 of 19 languages on FLORES+; NTREX and MAFAND", why="stopped after 9 languages by the 6 GB GPU cap", run="e2_gen_chat.py tpsy2b ...", a100_hours="0.8 (ESTIMATE; 5.4 GB weights)"),
    dict(gap="TranslatePsy-AfriSLM-4B (any benchmark)", why="never run; 8.6 GB weights exceeded the cap; published numbers are bidirectional aggregates", run="e2_gen_chat.py tpsy4b on FLORES+ (19 languages), NTREX, MAFAND, then AfriCOMET", a100_hours="1.5 (ESTIMATE)"),
    dict(gap="385M RL stages, SFT-only control and the 270M models on SmolSent, SALT, GlotStoryBook, IbomMT, FLORES-MOZ", why="the RL models were evaluated on FLORES+, NTREX, MAFAND only; the five other benchmarks exist for the 385M SFT (and A0) only", run="eval_sft_benchmarks (385M harness, greedy, 768 tokens) with benchmarks smol_sent,salt,glot_story,ibom_mt,flores_moz for mixed best, stage A, B1 best, control C1 and the two 270M models, then AfriCOMET", a100_hours="0.8 (ESTIMATE: the 122,585-row 3-benchmark run took 456 s; 48.8k rows per model; 6 models)"),
    dict(gap="SSA-COMET-MTL (reference-based) for the 385M models on FLORES+ (reward family; column would be empty for ours)", why="E-05 scored AfriScience-MT only", run="e2_ssa.py on the 22 languages for SFT, control, RL (no_pair rows)", a100_hours="0.3 (ESTIMATE, 60k pairs)"),
    dict(gap="Beam-4 / decoder-matched row for the RL model on full FLORES+ (ours greedy vs NLLB beam 4)", why="E-02 measured beam-4 on 20 languages x 200 sentences only (RL beam-4: -0.77 chrF++ vs greedy, MBR +0.92)", run="E-19: vLLM beam search (eos = end_of_turn) for mixed best on 44 languages x 1,012", a100_hours="2.5 (ESTIMATE: 4,000 prompts took 827 s -> 44.5k prompts)"),
    dict(gap="Published AfriqueLLM, NLLB-1.3B / MoE etc. on the same sentences with AfriCOMET", why="no per-sentence outputs and larger weights; only published chrF++ / SSA-COMET", run="not recommended (weights 4-54 GB); keep as published-only rows", a100_hours="n/a"),
]
with open(os.path.join(D, "missing.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(missing[0].keys()))
    w.writeheader()
    w.writerows(missing)
print("done")
