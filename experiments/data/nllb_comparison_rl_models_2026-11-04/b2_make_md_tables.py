#!/usr/bin/env python
"""Render the CSVs of b1_compare.py as markdown (tables.md) so the report is copied, not retyped."""
import pandas as pd
OUT = "/path/to/repo/reports/data/nllb_comparison_rl_models_2026-11-04"
def md(df, fmt=None):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append(f"{v:.2f}" if abs(v) >= 1 or v == 0 else f"{v:.4f}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)
t1 = pd.read_csv(OUT + "/table1_macro_43_languages.csv")
a = t1[t1.system == "NLLB-3.3B"][["model", "n_languages", "ours_macro", "system_macro", "diff", "diff_lo", "diff_hi", "macro_ratio", "median_lang_ratio", "n_ours_ge", "n_ours_ge_minus1", "n_ours_ge_80pct"]]
b = t1.pivot(index="model", columns="system", values="diff").reset_index()
t2 = pd.read_csv(OUT + "/table2_per_language_43.csv")
t2s = t2[["iso", "nllb_tertile", "nllb_3p3b", "nllb_moe_54p5b", "sft", "stageA", "b1best", "mixedbest", "weakfinal", "mixed_minus_nllb3p3b", "mixed_over_nllb3p3b", "mixed_finished", "mixed_repeat"]]
t3 = pd.read_csv(OUT + "/table3_by_nllb_tertile.csv")
t3s = t3[["tertile", "n", "nllb_3p3b", "nllb_moe_54p5b", "sft", "stageA", "b1best", "mixedbest", "weakfinal", "sft_pct_of_nllb3p3b", "stageA_pct_of_nllb3p3b", "mixedbest_pct_of_nllb3p3b"]]
t4 = pd.read_csv(OUT + "/table4_clean_language_subsets.csv")
t5 = pd.read_csv(OUT + "/table5_afrique_21_languages.csv")[["system", "n_languages", "system_macro", "sft", "mixedbest", "mixed_minus_system", "n_lang_mixed_ge_system"]]
t6 = pd.read_csv(OUT + "/table6b_same_sentence_summary.csv")
t6c = pd.read_csv(OUT + "/table6c_same_sentence_43_languages.csv")
t7 = pd.read_csv(OUT + "/table7_africomet_same_100_sentences.csv")
with open(OUT + "/tables.md", "w") as f:
    for name, df in [("Table 1 (NLLB-3.3B row of each model)", a), ("Table 1b: difference to the other NLLB variants (chrF++)", b), ("Table 2 per language", t2s),
                     ("Table 3 NLLB-3.3B tertiles", t3s), ("Table 4 language subsets", t4), ("Table 5 Afrique 5-shot, 21 languages", t5),
                     ("Table 6 same-sentence (post-editor run), all languages both have", t6), ("Table 6c same-sentence, 43 shared languages", t6c),
                     ("Table 7 AfriCOMET same 100 sentences", t7)]:
        f.write(f"## {name}\n\n{md(df)}\n\n")
print(open(OUT + "/tables.md").read()[:3000])
