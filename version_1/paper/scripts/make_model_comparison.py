#!/usr/bin/env python3
"""Model-comparison master table, paper tables and figures (agent E2, 2026-10-23).

    python paper/scripts/make_model_comparison.py [master|tables|figs]      (default: all three)

Inputs (all in the repository, no GPU, no network):
  reports/data/model_comparison_computed/{per_language,macro,paired_vs}.csv   scores WE computed on the shared box with ONE metric stack
        (chrF++ sacrebleu word_order 2 on NFC text, spBLEU flores200, AfriCOMET reference-based masakhane/africomet-stl-1.1) on FLORES+ devtest en->xx;
        produced by gemma_mt/scripts/e2_*.py (see reports/model_comparison_2026-10-23.md for the exact commands)
  reports/data/published_baselines.csv                     numbers copied from papers / model cards (chrF++, spBLEU)
  reports/data/afriquellm_table20_ssa_comet_enxx.csv       AfriqueLLM Table 20 (SSA-COMET en->xx, FLORES-200, 0-100; checkpoint not named by the paper)
  reports/data/published_comet_audit.csv                   which COMET-family numbers are published, and their protocol
  reports/data/language_coverage_claim_check.csv           African-language coverage of other systems
  reports/data/model_comparison_language_sets.json         language sets used for the macros
Outputs: reports/data/model_comparison_master.csv, paper/tables/t_model_comparison_*.md, paper/figures/fig12_model_size.{png,pdf}, fig13_language_coverage.{png,pdf}

Mixing rule (owner brief): published numbers enter the same column as computed ones ONLY when metric version and test set match; otherwise they are
in their own column (published chrF++ = whole FLORES-200 devtest from the paper's own pipeline; published SSA-COMET = a different metric; AfriNLLB's
AfriCOMET = africomet-mtl v1.0 on 9 languages). Macros are given only when the system covers every language of the set (else n/a with k/K).
"""
import csv
import json
import os
import re
import sys
from collections import OrderedDict, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import REPO, RDATA, TAB, FIG, OI  # noqa: E402

COMP = os.path.join(RDATA, "model_comparison_computed")
MASTER = os.path.join(RDATA, "model_comparison_master.csv")

LANG = dict(afr="Afrikaans", amh="Amharic", bam="Bambara", bem="Bemba", dyu="Dyula", ewe="Ewe", fon="Fon", hau="Hausa", ibo="Igbo", kab="Kabyle", kam="Kamba",
            kbp="Kabiye", kea="Kabuverdianu", kik="Kikuyu", kin="Kinyarwanda", kmb="Kimbundu", knc="Kanuri", lin="Lingala", lua="Luba-Kasai", lug="Luganda",
            luo="Luo", mlg="Malagasy", mos="Mossi", nso="Northern Sotho", nus="Nuer", nya="Chichewa", orm="Oromo", run="Kirundi", sag="Sango", sna="Shona",
            som="Somali", sot="Southern Sotho", ssw="Swati", swh="Swahili", tir="Tigrinya", tsn="Tswana", tso="Tsonga", tum="Tumbuka", twi="Twi",
            umb="Umbundu", vmw="Makhuwa", wol="Wolof", xho="Xhosa", yor="Yoruba", zul="Zulu")

# model key -> (display name, params in billions (None = closed / unknown), type, short group)
OURS = OrderedDict([
    ("rl", ("SabiYarn-270M RL-final (GRPO step 1433)", 0.27, "ours; 270M decoder LM, 77 languages", "ours")),
    ("sft", ("SabiYarn-270M SFT-final (step 27638)", 0.27, "ours; 270M decoder LM, 77 languages", "ours")),
    ("a2w", ("SabiYarn-270M science-stage start (A2W alpha 0.7)", 0.27, "ours; 270M decoder LM, 77 languages", "ours")),
    ("live250", ("SabiYarn-270M live science run (step 250, 2026-10-23)", 0.27, "ours; 270M decoder LM, 77 languages", "ours")),
    ("final2", ("SabiYarn-270M final run (final2 step 401)", 0.27, "ours; 270M decoder LM, 77 languages", "ours")),
])
COMPUTED_EXT = OrderedDict([
    ("nllb600m", ("NLLB-200-distilled-600M (re-run: CT2 fp16, beam 4)", 0.615, "encoder-decoder MT", "nllb")),
    ("nllb3b3", ("NLLB-200-3.3B (re-run: CT2 int8, beam 4)", 3.3, "encoder-decoder MT", "nllb")),
    ("tpsy08", ("TranslatePsy-AfriSLM-0.8B (re-run: zero-shot, greedy)", 0.8, "instruction-tuned LLM (Qwen3.5), 19 languages", "tpsy")),
    ("tpsy2b", ("TranslatePsy-AfriSLM-2B (re-run: zero-shot, greedy)", 2.0, "instruction-tuned LLM (Qwen3.5), 19 languages", "tpsy")),
    ("gemma3_4b", ("Gemma3-4B-it (re-run: zero-shot, greedy)", 4.3, "instruction-tuned LLM", "gemma")),
])
# published models -> (params B, type)
PUB = {
    "NLLB-600M": (0.6, "encoder-decoder MT"), "AfriNLLB-600M": (0.548, "encoder-decoder MT (pruned NLLB)"),
    "NLLB-200-1.3B": (1.3, "encoder-decoder MT"), "NLLB-200-distilled-1.3B": (1.3, "encoder-decoder MT"), "NLLB-200-3.3B": (3.3, "encoder-decoder MT"),
    "NLLB-200-MoE-54.5B": (54.5, "encoder-decoder MT (MoE)"),
    "Llama3.1-8B": (8.0, "base LM, 5-shot"), "Lugha-Llama-8B-wura": (8.0, "base LM, 5-shot"), "AfriqueLlama-8B": (8.0, "African-adapted LM, 5-shot"),
    "Gemma3-4B": (4.3, "base LM, 5-shot"), "AfriqueGemma-4B": (4.3, "African-adapted LM, 5-shot"),
    "Gemma3-12B": (12.0, "base LM, 5-shot"), "AfriqueGemma-12B": (12.0, "African-adapted LM, 5-shot"),
    "Qwen3-4B": (4.0, "base LM, 5-shot"), "AfriqueQwen-4B": (4.0, "African-adapted LM, 5-shot"),
    "Qwen3.5-4B": (4.0, "base LM, 5-shot"), "AfriqueQwen3.5-4B": (4.0, "African-adapted LM, 5-shot"),
    "AfriqueQwen3.5-4B-ExtendedCM": (4.0, "African-adapted LM, 5-shot"), "AfriqueQwen3.5-4B-50Langs": (4.0, "African-adapted LM, 5-shot"),
    "Qwen3-8B": (8.2, "base LM, 5-shot"), "AfriqueQwen-8B": (8.2, "African-adapted LM, 5-shot"),
    "Qwen3-14B": (14.8, "base LM, 5-shot"), "AfriqueQwen-14B": (14.8, "African-adapted LM, 5-shot"),
    "Gemini-2.5-Flash": (None, "closed API"), "Gemini 1.5 pro": (None, "closed API"), "GPT-4o (Aug)": (None, "closed API"),
    "TranslatePsy-AfriSLM-0.8B": (0.8, "instruction-tuned LLM (Qwen3.5)"), "TranslatePsy-AfriSLM-2B": (2.0, "instruction-tuned LLM (Qwen3.5)"),
    "TranslatePsy-AfriSLM-4B": (4.0, "instruction-tuned LLM (Qwen3.5)"),
    "TranslateGemma-4B": (4.3, "translation LLM (Gemma 3)"), "TranslateGemma-12B": (12.0, "translation LLM (Gemma 3)"), "TranslateGemma-27B": (27.0, "translation LLM (Gemma 3)"),
    "Hunyuan-MT-7B": (7.0, "translation LLM"), "Apertus-8B": (8.0, "general LLM"),
    "Qwen3.5-122B-A10B": (122.0, "general LLM (MoE)"), "Qwen3.5-27B": (27.0, "general LLM"), "Qwen3.5-9B": (9.0, "general LLM"), "Qwen3.5-2B": (2.0, "general LLM"),
    "Qwen3.5-0.8B": (0.8, "general LLM"), "NLLB 3.3B": (3.3, "encoder-decoder MT"), "Aya-101 13B": (13.0, "multilingual LLM"), "Gemma 2 27B": (27.0, "general LLM"),
    "LLaMa 3.1 70B": (70.0, "general LLM"),
}
# the published row of a re-run model (for the "re-run vs published" check)
RERUN_PUBLISHED = {"nllb600m": "NLLB-600M", "nllb3b3": "NLLB-200-3.3B"}


def fnum(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_sets():
    return json.load(open(os.path.join(RDATA, "model_comparison_language_sets.json")))


def size_str(b):
    if b is None:
        return "closed"
    return f"{b * 1000:.0f}M" if b < 1 else (f"{b:g}B")


# ----------------------------------------------------------------------------------------------------------------- loaders
def _comp(name):
    """P5 (2026-10-23): the `*_with_final2.csv` files are the E2 files re-aggregated with the final-run checkpoint (final2, step 401); every older row is identical
    (checked row by row), paired_vs has the references rl, sft, a2w, live250. The old files are kept; the new ones are preferred when present."""
    new = os.path.join(COMP, name + "_with_final2.csv")
    return new if os.path.exists(new) else os.path.join(COMP, name + ".csv")


def load_computed():
    per = read_csv(_comp("per_language"))
    mac = read_csv(_comp("macro"))
    pair = read_csv(_comp("paired_vs")) if os.path.exists(_comp("paired_vs")) else []
    return per, mac, pair


def load_published():
    """{model: {iso: {'chrf':x,'spbleu':y,'flag':..,'note':..,'bench':..}}} for per-language en->xx rows, plus aggregate rows."""
    rows = read_csv(os.path.join(RDATA, "published_baselines.csv"))
    per = defaultdict(dict)
    agg = []
    for r in rows:
        if r["flag"] == "aggregate" or r["iso"] in ("MACRO19", "MACRO"):
            agg.append(r)
            continue
        if r["direction"] != "en->xx" or fnum(r["chrF++"]) is None or r["flag"] in ("no_chrf", "no_number"):
            continue
        iso = "swh" if r["iso"] == "swa" else r["iso"]
        per[r["model"]][iso] = dict(chrf=fnum(r["chrF++"]), spbleu=fnum(r["BLEU_spBLEU"]), flag=r["flag"], bench=r["benchmark"], url=r["source_url"], kind=r["bleu_kind"])
    return per, agg


def load_ssa_comet_pub():
    out = defaultdict(dict)
    path = os.path.join(RDATA, "afriquellm_table20_ssa_comet_enxx.csv")
    for r in read_csv(path):
        iso = "swh" if r["iso_paper"] == "swa" else r["iso_paper"]
        if iso in ("avg", "aeb", "ary", "arz"):
            continue
        out[r["model"]][iso] = float(r["ssa_comet_0_100"])
    return out


def load_afrinllb_africomet():
    """AfriNLLB paper Table 7 (africomet-mtl v1.0, FLORES-200 devtest en->xx, 0-100), parsed from the audit csv text."""
    out = defaultdict(dict)
    for r in read_csv(os.path.join(RDATA, "published_comet_audit.csv")):
        if r["metric_family"] == "AfriCOMET" and r["model"] in ("NLLB-600M", "AfriNLLB-600M"):
            txt = r["value_or_location"].split("en->xx:")[-1]
            txt = txt.split(";")[0] if ";" in txt else txt
            for iso, v in re.findall(r"\b([a-z]{3}) (\d+\.\d+)", txt):
                if iso in LANG or iso == "arz":
                    out[r["model"]][iso] = float(v)
    return out


# ----------------------------------------------------------------------------------------------------------------- macros
def macro_pub(per_lang, iso_list, key="chrf"):
    vals = [per_lang[i][key] if isinstance(per_lang[i], dict) else per_lang[i] for i in iso_list if i in per_lang and per_lang[i] is not None]
    vals = [v for v in vals if v is not None]
    return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)


def macro_index(mac):
    idx = {}
    for r in mac:
        idx[(r["model"], r["set"], r["subset"], r["metric"])] = r
    return idx


def fmt_ci(r, nd=1, scale=1.0):
    if r is None:
        return "n/a"
    v, lo, hi = float(r["value"]) * scale, float(r["lo"]) * scale, float(r["hi"]) * scale
    s = f"{v:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"
    if r["full"] != "1":
        s = f"({s}) {r['k_covered']}/{r['k_set']}"
    return s


def fmt_cell(r, nd=1, scale=1.0, ci=True):
    """Strict rule: a macro is shown only when the system covers every language of the set; otherwise 'n/a (k/K)'."""
    if r is None:
        return "n/a"
    if r["full"] != "1":
        return f"n/a ({r['k_covered']}/{r['k_set']})"
    v = float(r["value"]) * scale
    if not ci:
        return f"{v:.{nd}f}"
    return f"{v:.{nd}f} [{float(r['lo']) * scale:.{nd}f}, {float(r['hi']) * scale:.{nd}f}]"


# ----------------------------------------------------------------------------------------------------------------- master csv
MASTER_COLS = ["model", "size", "type", "language", "benchmark", "subset", "n", "chrF++", "chrF++_ci_lo", "chrF++_ci_hi", "spBLEU", "AfriCOMET_ref", "AfriCOMET_ci_lo",
               "AfriCOMET_ci_hi", "SSA_COMET_ref", "SSA_COMET_ci_lo", "SSA_COMET_ci_hi", "SSA_COMET_published", "AfriCOMET_published_other_version", "source", "notes"]


def build_master():
    per, mac, _ = load_computed()
    pub, agg = load_published()
    ssa = load_ssa_comet_pub()
    afn = load_afrinllb_africomet()
    rows = []
    allmeta = {**{k: v for k, v in OURS.items()}, **COMPUTED_EXT}
    for r in per:
        if r["model"] not in allmeta:
            continue
        name, params, typ, _g = allmeta[r["model"]]
        sub = r["subset"]
        note = {"no_pair": "headline subset: reference pair never seen in SFT", "clean": "English sentence not in the SFT text (48 sentences)",
                "all": "all 1,012 devtest sentences (the SFT data contains most of them for our models; not a fair test for our models)"}[sub]
        if r["model"] in ("tpsy08", "tpsy2b", "gemma3_4b"):
            note += "; zero-shot with the model's own chat template; run on the no_pair sentences only" if sub != "all" else ""
        rows.append({"model": name, "size": size_str(params), "type": typ, "language": r["iso"], "benchmark": "FLORES+ devtest en->xx", "subset": sub, "n": r["n"],
                     "chrF++": r["chrf"], "chrF++_ci_lo": r["chrf_lo"], "chrF++_ci_hi": r["chrf_hi"], "spBLEU": r["spbleu"],
                     "AfriCOMET_ref": r.get("africomet", ""), "AfriCOMET_ci_lo": r.get("africomet_lo", ""), "AfriCOMET_ci_hi": r.get("africomet_hi", ""),
                     "SSA_COMET_ref": r.get("ssa_comet", ""), "SSA_COMET_ci_lo": r.get("ssa_comet_lo", ""), "SSA_COMET_ci_hi": r.get("ssa_comet_hi", ""),
                     "source": "computed", "notes": note + ("" if r.get("africomet") not in (None, "") else "; AfriCOMET n/a (language not listed by AfriCOMET-STL-1.1, or subset all)")})
    for m, d in pub.items():
        if m not in PUB:
            continue
        params, typ = PUB[m]
        for iso, v in sorted(d.items()):
            if iso not in LANG and iso not in ("arz", "ary", "aeb", "kon"):
                continue
            ss = ssa.get(m, {}).get(iso, "")
            if not ss and m.startswith("NLLB") and False:
                ss = ""
            af = afn.get(m, {}).get(iso, "")
            rows.append({"model": m, "size": size_str(params), "type": typ, "language": iso, "benchmark": v["bench"][:80], "subset": "published (paper's own devtest pipeline)", "n": "",
                         "chrF++": v["chrf"], "spBLEU": v["spbleu"] if v["spbleu"] is not None else "", "AfriCOMET_ref": "", "SSA_COMET_published": ss,
                         "AfriCOMET_published_other_version": af, "source": "published",
                         "notes": ("flag=" + v["flag"] + "; " if v["flag"] else "") + "spBLEU kind=" + (v["kind"] or "NA") + "; source " + v["url"][:90]})
    for m, d in afn.items():
        for iso, v in sorted(d.items()):
            rows.append({"model": m, "size": size_str(PUB[m][0]), "type": PUB[m][1], "language": iso, "benchmark": "FLORES-200 devtest en->xx", "subset": "published (AfriNLLB paper Table 7)",
                         "AfriCOMET_published_other_version": v, "source": "published",
                         "notes": "AfriCOMET = masakhane/africomet-mtl v1.0 (NOT stl-1.1; scale 0-100; zero-shot for most African languages; Wolof not covered), arXiv 2602.09373; shown in its own column because metric version differs; arz is Egyptian Arabic (not ours)"})
    for r in agg:
        m = r["model"]
        if m not in PUB:
            continue
        params, typ = PUB[m]
        rows.append({"model": m, "size": size_str(params), "type": typ, "language": "MACRO19 (both directions averaged, inferred)", "benchmark": r["benchmark"][:80], "subset": "published aggregate",
                     "chrF++": r["chrF++"], "spBLEU": r["BLEU_spBLEU"] if r["BLEU_spBLEU"] != "NA" else "", "source": "published",
                     "notes": "aggregate only; NOT comparable with en->xx per-language numbers; " + r["metric_note"][:100]})
    with open(MASTER, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MASTER_COLS)
        w.writeheader()
        for r in rows:
            out = {}
            for c in MASTER_COLS:
                v = r.get(c, "")
                out[c] = round(float(v), 4) if isinstance(v, (int, float)) or (isinstance(v, str) and re.fullmatch(r"-?\d+\.\d{5,}", v)) else v
            w.writerow(out)
    print("master:", len(rows), "rows ->", MASTER)
    return rows


# ----------------------------------------------------------------------------------------------------------------- tables
def md_table(header, rows, align=None):
    align = align or (["l"] + ["r"] * (len(header) - 1))
    sep = ["---:" if a == "r" else ":---" for a in align]
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def build_tables():
    sets = load_sets()
    per, mac, pair = load_computed()
    pub, agg = load_published()
    ssa = load_ssa_comet_pub()
    afn = load_afrinllb_africomet()
    M = macro_index(mac)
    allmeta = {**OURS, **COMPUTED_EXT}
    present = [k for k in allmeta if any(r["model"] == k for r in per)]
    K = {s: len(v) for s, v in sets.items()}
    out = {}

    # ---- main table: one row per model
    hdr = ["Model", "Params", "chrF++ NLLB-43", "chrF++ Afrique-21", "chrF++ TPsy-19", "AfriCOMET (ours) 42 lang.", "AfriCOMET (ours) TPsy-19", "SSA-COMET Afrique-21, ours-scored x100", "SSA-COMET Afrique-21, paper's", "Source"]
    rows = []

    def computed_row(k):
        name, params, typ, g = allmeta[k]
        c = lambda s, m, nd=1, scale=1.0: fmt_cell(M.get((k, s, "no_pair", m)), nd, scale)
        return [name, size_str(params), c("nllb43", "chrf"), c("afrique21", "chrf"), c("tpsy19", "chrf"), c("comet_scored42", "africomet", 3), c("tpsy19", "africomet", 3), c("afrique21", "ssa_comet", 1, 100.0), "n/a", "computed (same sentences, no_pair)"]

    for k in OURS:
        if k in present:
            rows.append(computed_row(k))
    for k in COMPUTED_EXT:
        if k in present:
            rows.append(computed_row(k))
    # published rows, sorted by params
    pr = []
    for m, d in pub.items():
        if m not in PUB or m in ("NLLB 3.3B",):
            continue
        params, typ = PUB[m]
        cells = []
        for s in ("nllb43", "afrique21", "tpsy19"):
            v, kk = macro_pub(d, sets[s])
            cells.append(f"{v:.1f}" if v is not None and kk == K[s] else (f"n/a ({kk}/{K[s]})" if kk else "n/a"))
        sv, sk = macro_pub(ssa.get(m, {}), sets["afrique21"], key=None) if False else (None, 0)
        sd = ssa.get(m, {})
        sv, sk = (sum(sd[i] for i in sets["afrique21"] if i in sd) / len([i for i in sets["afrique21"] if i in sd]), len([i for i in sets["afrique21"] if i in sd])) if sd else (None, 0)
        scell = f"{sv:.1f}" if sv is not None and sk == K["afrique21"] else "n/a"
        flag = ";".join(sorted({v["flag"] for v in d.values() if v["flag"]}))
        pr.append((params if params is not None else 1e9, [m + ("  (n=10 sentences/language, chrF)" if flag == "low_n" else ""), size_str(params), cells[0], cells[1], cells[2], "n/a", "n/a", "n/a", scell,
                                                           "published (paper's own pipeline, whole devtest)" + (f"; flag {flag}" if flag else "")]))
    for _, r in sorted(pr, key=lambda t: t[0]):
        if r[2].startswith("n/a") and r[3].startswith("n/a") and r[4].startswith("n/a") and r[8] == "n/a":
            continue
        rows.append(r)
    note = (
        "Rows 'computed': our own scoring on identical sentences (FLORES+ devtest `no_pair` subset: the sentences whose reference pair our SFT never saw; "
        "macro over the language set, 95 % bootstrap CI over sentences, no language-sampling noise; AfriCOMET = reference-based `masakhane/africomet-stl-1.1`, scale 0-1). "
        "Rows 'published': the paper's own chrF++ on the whole FLORES-200 devtest (5-shot base models and encoder-decoder MT; different prompts, decoding and, for the Afrique tables, an unspecified split), "
        "SSA-COMET = a different metric (paper's checkpoint unnamed; 0-100), so published and computed cells are never mixed in one column of a macro. "
        "'n/a (k/K)': the system does not cover all K languages of the set (k covered); we do not average over a different language set. "
        "NLLB-43 = the 43 FLORES+ languages with a published NLLB number; Afrique-21 = the 21 languages of the AfriqueLLM tables that are in FLORES+; TPsy-19 = the 19 TranslatePsy-AfriSLM languages. "
        "AfriCOMET n/a for kea, knc, vmw (not on the AfriCOMET-STL-1.1 card), so 42 of the 45 FLORES+ languages are scored.")
    out["t_model_comparison_main.md"] = "### Table M1. One table: chrF++ and AfriCOMET, English -> X, FLORES+ devtest\n\n" + md_table(hdr, rows, ["l"] + ["r"] * 8 + ["l"]) + "\n" + note + "\n"

    # ---- computed macros per set (chrF++, spBLEU, AfriCOMET) with CIs, three subsets
    for sub in ("no_pair", "clean"):
        h = ["Model"] + [f"{s}" for s in ("all45", "nllb43", "afrique21", "tpsy19", "tpsy2b9", "comet_scored42", "africomet_encoder16", "afrimte9")]
        for metric, nd, title in (("chrf", 1, "chrF++"), ("spbleu", 1, "spBLEU"), ("africomet", 3, "AfriCOMET (ref-based, stl-1.1)"), ("ssa_comet", 3, "SSA-COMET-MTL (ref-based, McGill-NLP/ssa-comet-mtl, 0-1; only the 22 languages of Afrique-21 and TPsy-19 were scored)")):
            rr = []
            for k in present:
                rr.append([allmeta[k][0]] + [fmt_cell(M.get((k, s, sub, metric)), nd) for s in h[1:]])
            out[f"t_model_comparison_computed_{sub}_{metric}.md"] = (f"### Table M2 ({sub}). {title}, macro over language sets, computed on identical sentences ({sub})\n\n" +
                                                                   md_table(h, rr, ["l"] + ["r"] * 8) + "\nset sizes: " + ", ".join(f"{s}={K[s]}" for s in h[1:]) + "\n")

    # ---- paired differences to RL-final
    if pair:
        rr = []
        for r in pair:
            if r.get("vs", "rl") != "rl":
                continue
            if r["subset"] != "no_pair" or r["full"] != "1" or r["metric"] not in ("chrf", "africomet") or r["set"] not in ("nllb43", "afrique21", "tpsy19", "comet_scored42"):
                continue
            rr.append([allmeta.get(r["model"], (r["model"],))[0], r["set"], r["metric"], r["k"], f"{float(r['delta']):+.3f} [{float(r['lo']):+.3f}, {float(r['hi']):+.3f}]"])
        out["t_model_comparison_paired.md"] = ("### Table M3. Paired difference to RL-final (system minus RL-final, macro over the set; same sentences; 95 % CI)\n\n" +
                                               md_table(["system", "set", "metric", "k", "difference"], rr, ["l", "l", "l", "r", "r"]))

    # ---- P5: the final run (final2 step 401) next to the other checkpoints of ours, macro + paired differences (no_pair, all four metrics)
    if "final2" in present:
        own = [k for k in ("sft", "rl", "a2w", "live250", "final2") if k in present]
        sel = [("nllb43", "chrf", "chrF++ NLLB-43", 2, 1.0), ("nllb43", "spbleu", "spBLEU NLLB-43", 2, 1.0), ("comet_scored42", "africomet", "AfriCOMET 42 lang.", 4, 1.0),
               ("afrique21", "ssa_comet", "SSA-COMET Afrique-21 (x100)", 2, 100.0), ("tpsy19", "chrf", "chrF++ TPsy-19", 2, 1.0), ("all45", "chrf", "chrF++ 45 lang.", 2, 1.0)]
        rr = []
        for k in own:
            rr.append([allmeta[k][0].replace("SabiYarn-270M ", "")] + [fmt_cell(M.get((k, s, "no_pair", m)), nd, sc) for s, m, _, nd, sc in sel])
        out["t_final2_flores_macro.md"] = ("### Table P5-1. The final run (final2, step 401) next to our other checkpoints, FLORES+ devtest `no_pair` rows, macro with 95% sentence-bootstrap CI (same sentences)\n\n" +
                                          md_table(["checkpoint"] + [x[2] for x in sel], rr, ["l"] + ["r"] * len(sel)))
        P = {(r["model"], r["vs"], r["set"], r["subset"], r["metric"]): r for r in pair}
        rr = []
        for vs, lab in (("rl", "RL-final (phase 2c, step 1,433)"), ("sft", "SFT-final"), ("a2w", "science-stage start A2W alpha 0.7"), ("live250", "step-250 checkpoint of the run")):
            row = [lab]
            for s, m, _, nd, sc in sel:
                r = P.get(("final2", vs, s, "no_pair", m))
                row.append(f"{float(r['delta']) * sc:+.{nd}f} [{float(r['lo']) * sc:+.{nd}f}, {float(r['hi']) * sc:+.{nd}f}]" if r else "n/a")
            rr.append(row)
        out["t_final2_flores_paired.md"] = ("### Table P5-2. Paired difference, final2 minus the checkpoint named, FLORES+ devtest `no_pair`, macro over the set, 95% sentence-bootstrap CI (language-sampling noise not included)\n\n" +
                                           md_table(["final2 minus"] + [x[2] for x in sel], rr, ["l"] + ["r"] * len(sel)))

    # ---- per-language appendix: chrF++ and AfriCOMET (no_pair)
    pl = {(r["model"], r["iso"], r["subset"]): r for r in per}
    pubcols = [("NLLB-200-1.3B", "NLLB-1.3B pub."), ("NLLB-200-3.3B", "NLLB-3.3B pub."), ("NLLB-200-MoE-54.5B", "NLLB-MoE pub."), ("AfriqueGemma-12B", "AfriqueGemma-12B pub."),
               ("AfriqueQwen3.5-4B-50Langs", "AfriqueQwen3.5-4B-50L pub.")]
    h = ["Language", "n"] + [allmeta[k][0].split(" (")[0].replace("SabiYarn-270M ", "SY ") for k in present] + [b for _, b in pubcols]
    rr = []
    for iso in sets["all45"]:
        n = next((pl[(k, iso, "no_pair")]["n"] for k in present if (k, iso, "no_pair") in pl), "")
        row = [f"{LANG[iso]} ({iso})", n]
        for k in present:
            r = pl.get((k, iso, "no_pair"))
            row.append(f"{float(r['chrf']):.1f}" if r else "n/a")
        for m, _ in pubcols:
            v = pub.get(m, {}).get(iso)
            row.append(f"{v['chrf']:.1f}" if v else "n/a")
        rr.append(row)
    out["t_model_comparison_perlang_chrf.md"] = ("### Table A-M1. Per-language chrF++ (FLORES+ devtest en->xx). Columns 3..: computed on the `no_pair` sentences (n in column 2); "
                                                 "'pub.' columns: the paper's number on the whole devtest (not the same sentences; NLLB-MoE and Afrique rows from the papers).\n\n" + md_table(h, rr))
    h2 = ["Language", "n"] + [allmeta[k][0].split(" (")[0].replace("SabiYarn-270M ", "SY ") for k in present] + ["AfriNLLB pub. NLLB-600M (mtl v1.0)", "AfriNLLB pub. AfriNLLB"]
    rr = []
    for iso in sets["all45"]:
        n = next((pl[(k, iso, "no_pair")]["n"] for k in present if (k, iso, "no_pair") in pl), "")
        row = [f"{LANG[iso]} ({iso})", n]
        for k in present:
            r = pl.get((k, iso, "no_pair"))
            row.append(f"{float(r['africomet']):.3f}" if r and r.get("africomet") else "n/a")
        for m in ("NLLB-600M", "AfriNLLB-600M"):
            v = afn.get(m, {}).get(iso)
            row.append(f"{v / 100:.3f}" if v is not None else "n/a")
        rr.append(row)
    out["t_model_comparison_perlang_africomet.md"] = ("### Table A-M2. Per-language AfriCOMET (reference-based `africomet-stl-1.1`, 0-1; `no_pair` sentences). n/a = language not listed by the AfriCOMET card, "
                                                      "or the system does not translate it. The last two columns are the AfriNLLB paper's own values (africomet-mtl v1.0, 0-100 scaled to 0-1; Wolof unreliable "
                                                      "because v1.0 does not cover it) and are NOT comparable with the other columns.\n\n" + md_table(h2, rr))

    # ---- re-run vs published (validation of our metric stack)
    rr = []
    for k, pm in RERUN_PUBLISHED.items():
        if k not in present or pm not in pub:
            continue
        diffs = []
        for iso, v in pub[pm].items():
            r = pl.get((k, iso, "all"))
            if r:
                diffs.append(float(r["chrf"]) - v["chrf"])
        if diffs:
            rr.append([allmeta[k][0], pm, len(diffs), f"{sum(diffs) / len(diffs):+.2f}", f"{sum(abs(d) for d in diffs) / len(diffs):.2f}", f"{max(diffs, key=abs):+.2f}"])
    out["t_model_comparison_validation.md"] = ("### Table M4. Metric-stack check: our chrF++ on the whole devtest minus the published chrF++ of the same checkpoint family (per language)\n\n" +
                                               md_table(["re-run", "published row", "languages", "mean diff", "mean |diff|", "largest diff"], rr, ["l", "l", "r", "r", "r", "r"]) +
                                               "\nint8 CTranslate2 inference and sentence-level beam 4 explain differences below 1 chrF++; the Yoruba difference comes from NFC normalisation (28 % of Yoruba references are decomposed).\n")

    # ---- published aggregates (cannot be put next to per-language en->xx numbers)
    audit0 = read_csv(os.path.join(RDATA, "published_comet_audit.csv"))
    first_num = lambda s: (re.match(r"\s*(-?\d+(?:\.\d+)?)", s) or [None, None])[1]
    agg_comet = defaultdict(dict)
    for r in audit0:
        if r["test_set"].startswith("FLORES-200 devtest (1,012") and r["metric_family"] in ("SSA-COMET", "COMET-22", "MetricX"):
            agg_comet[r["model"]][r["metric_family"]] = first_num(r["value_or_location"])
    rr = []
    for r in agg:
        if "FLORES" not in r["benchmark"] or r["model"] not in PUB:
            continue
        a = agg_comet.get(r["model"], {})
        rr.append((PUB[r["model"]][0] or 1e9, [r["model"], size_str(PUB[r["model"]][0]), r["chrF++"], r["BLEU_spBLEU"], a.get("SSA-COMET", ""), a.get("COMET-22", ""), a.get("MetricX", "")]))
    ab = [[r["model"], size_str(PUB.get(r["model"], (None,))[0]), r["chrF++"]] for r in agg if r["iso"] == "MACRO_AFROBENCH"]
    out["t_model_comparison_published_aggregates.md"] = (
        "### Table A-M5a. Published AGGREGATES on FLORES-200 devtest, 19 African languages (TranslatePsy-AfriSLM paper, Tables 24/25; direction not stated in the captions, "
        "both directions averaged by inference). NOT comparable with the en->xx per-language numbers of Table M1; our own en->xx macros over the same 19 languages are in Table M2.\n\n" +
        md_table(["model", "params", "chrF++", "spBLEU", "SSA-COMET (0-1)", "COMET-22 (0-1)", "MetricX-24 (lower better)"], [x for _, x in sorted(rr, key=lambda t: t[0])], ["l"] + ["r"] * 6) +
        "\n### Table A-M5b. AfroBench pooled MT chrF (Flores-200 + MAFAND + NTREX-128, en/fr -> many African languages, best of 3 prompts, 0-shot; AfroBench paper Table 4)\n\n" +
        md_table(["model", "params", "chrF (pooled)"], ab, ["l", "r", "r"]))

    # ---- COMET audit summary
    audit = read_csv(os.path.join(RDATA, "published_comet_audit.csv"))
    fam = defaultdict(lambda: defaultdict(set))
    for r in audit:
        fam[r["model"]][r["metric_family"]].add(r["test_set"][:40])
    rr = []
    for m in sorted(fam, key=lambda x: (PUB.get(x, (1e9,))[0] or 1e9, x)):
        mf = fam[m]
        has = lambda f: "yes" if f in mf else ""
        rr.append([m, "yes (v1.0 mtl, 9 lang.)" if "AfriCOMET" in mf else "no", has("SSA-COMET"), has("COMET-22"), has("MetricX"), "yes" if any(k.startswith("other") or k == "COMET-Kiwi" for k in mf) else "",
                   "none" if set(mf) == {"none"} else ""])
    out["t_model_comparison_comet_audit.md"] = ("### Table A-M3. Which COMET-family numbers are PUBLISHED for the compared models (details and sources: reports/data/published_comet_audit.csv)\n\n" +
                                                md_table(["model", "AfriCOMET published", "SSA-COMET", "COMET-22", "MetricX", "other COMET family", "no COMET-family number"], rr, ["l", "l", "c", "c", "c", "c", "c"]))

    # ---- coverage evidence
    cov = read_csv(os.path.join(RDATA, "language_coverage_claim_check.csv"))
    rr = [[r["model"][:60], r["params"][:40], r["open_weights"][:30], r["african_langs_supported"][:80], r["african_langs_evaluated_enx"][:70], r["verified"]] for r in cov]
    out["t_coverage_claim.md"] = ("### Table A-M4. African-language coverage of other systems (evidence; reports/data/language_coverage_claim_check.md)\n\n" +
                                  md_table(["system", "params", "open", "African languages supported (claimed/trained)", "evaluated en->X", "verified"], rr, ["l"] * 6))

    for fn, txt in out.items():
        with open(os.path.join(TAB, fn), "w", encoding="utf-8") as f:
            f.write(txt)
    print("tables:", ", ".join(out))


# ----------------------------------------------------------------------------------------------------------------- figures
def build_figs():
    from common import setup_matplotlib
    plt = setup_matplotlib()
    import numpy as np
    sets = load_sets()
    per, mac, pair = load_computed()
    pub, _ = load_published()
    ssa = load_ssa_comet_pub()
    M = macro_index(mac)
    allmeta = {**OURS, **COMPUTED_EXT}
    present = [k for k in allmeta if any(r["model"] == k for r in per)]
    fam_color = dict(ours=OI["verm"], nllb=OI["blue"], tpsy=OI["green"], afrique=OI["orange"], base=OI["grey"], other=OI["sky"], gemma=OI["pink"])
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9))
    # (a) chrF++ on Afrique-21 vs params: computed (filled) + published (hollow)
    ax = axs[0]
    for k in present:
        name, params, typ, g = allmeta[k]
        r = M.get((k, "afrique21", "no_pair", "chrf"))
        if r and r["full"] == "1":
            v, lo, hi = float(r["value"]), float(r["lo"]), float(r["hi"])
            big = g == "ours"
            ax.errorbar(params, v, yerr=[[v - lo], [hi - v]], fmt="*" if big else "o", ms=9 if big else 5, color=fam_color[g], mec="black" if big else None, mew=0.6, capsize=1.5, zorder=5)
            if k in ("rl", "nllb3b3", "tpsy2b", "nllb600m"):
                ax.annotate({"rl": "RL-final", "nllb3b3": "NLLB-3.3B", "tpsy2b": "TPsy-2B", "nllb600m": "NLLB-600M"}[k], (params, v), textcoords="offset points", xytext=(5, -9 if k != "rl" else 9), fontsize=5.5)
    for m, d in pub.items():
        if m not in PUB or PUB[m][0] is None or m == "NLLB 3.3B":
            continue
        v, kk = macro_pub(d, sets["afrique21"])
        if v is None or kk != 21:
            continue
        g = "nllb" if m.startswith(("NLLB", "AfriNLLB")) else ("afrique" if m.startswith("Afrique") else "base")
        ax.plot(PUB[m][0], v, "o", ms=4, mfc="none", mec=fam_color[g], mew=1.0)
    ax.set_xscale("log")
    ax.set_xticks([0.3, 1, 3, 10, 30]); ax.set_xticklabels(["0.3", "1", "3", "10", "30"]); ax.minorticks_off()
    ax.set_xlabel("parameters (B)")
    ax.set_ylabel("chrF++, macro over 21 languages")
    ax.set_title("(a) chrF++ vs size", loc="left")
    # (b) AfriCOMET on TPsy-19 vs params (computed only)
    ax = axs[1]
    for k in present:
        name, params, typ, g = allmeta[k]
        r = M.get((k, "tpsy19", "no_pair", "africomet"))
        if r and r["full"] == "1":
            v, lo, hi = float(r["value"]), float(r["lo"]), float(r["hi"])
            big = g == "ours"
            ax.errorbar(params, v, yerr=[[v - lo], [hi - v]], fmt="*" if big else "o", ms=9 if big else 5, color=fam_color[g], mec="black" if big else None, mew=0.6, capsize=1.5, zorder=5)
            if k in ("rl", "nllb3b3", "tpsy2b", "nllb600m", "tpsy08"):
                ax.annotate({"rl": "RL-final", "nllb3b3": "NLLB-3.3B", "tpsy2b": "TPsy-2B", "nllb600m": "NLLB-600M", "tpsy08": "TPsy-0.8B"}[k], (params, v), textcoords="offset points", xytext=(5, -8 if k != "rl" else 5), fontsize=5.5)
    ax.set_xscale("log")
    ax.set_xticks([0.3, 1, 3, 10, 30]); ax.set_xticklabels(["0.3", "1", "3", "10", "30"]); ax.minorticks_off()
    ax.set_xlabel("parameters (B)")
    ax.set_ylabel("AfriCOMET (ref), macro over 19 languages")
    ax.set_title("(b) AfriCOMET vs size", loc="left")
    # (c) published SSA-COMET Afrique-21 vs params (paper's numbers)
    ax = axs[2]
    for m, d in ssa.items():
        if m not in PUB or PUB[m][0] is None:
            continue
        vals = [d[i] for i in sets["afrique21"] if i in d]
        if len(vals) != 21:
            continue
        g = "afrique" if m.startswith("Afrique") else "base"
        ax.plot(PUB[m][0], sum(vals) / 21, "o", ms=4, mfc="none", mec=fam_color[g], mew=1.0)
    ax.set_xscale("log")
    ax.set_xticks([0.3, 1, 3, 10, 30]); ax.set_xticklabels(["0.3", "1", "3", "10", "30"]); ax.minorticks_off()
    ax.set_xlabel("parameters (B)")
    ax.set_ylabel("SSA-COMET (published), macro 21")
    ax.set_title("(c) published SSA-COMET", loc="left")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="*", ls="", color=fam_color["ours"], mec="black", ms=8, label="ours (270M)"), Line2D([], [], marker="o", ls="", color=fam_color["nllb"], ms=5, label="NLLB, re-run"),
         Line2D([], [], marker="o", ls="", color=fam_color["tpsy"], ms=5, label="TranslatePsy, re-run"), Line2D([], [], marker="o", ls="", mfc="none", mec=fam_color["nllb"], label="NLLB / AfriNLLB, published"),
         Line2D([], [], marker="o", ls="", mfc="none", mec=fam_color["afrique"], label="Afrique models, published"), Line2D([], [], marker="o", ls="", mfc="none", mec=fam_color["base"], label="base LMs, published")]
    fig.legend(handles=h, loc="lower center", ncol=3, fontsize=6, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"fig12_model_size.{ext}"), bbox_inches="tight")
    plt.close(fig)

    # ---- coverage vs size
    pts = [  # (label, params B, African target languages supported, kind) kind: ours | eval (per-language results published) | claim (support claimed, no per-language en->X evaluation) | zero
        ("SabiYarn-270M (ours)", 0.27, 77, "ours"),
        ("NLLB-200 600M", 0.6, 55, "eval"), ("NLLB-200 1.3B", 1.3, 55, "eval"), ("NLLB-200 3.3B", 3.3, 55, "eval"), ("NLLB-200 MoE 54.5B", 54.5, 55, "eval"),
        ("MADLAD-400 3B", 3.0, 85, "claim"), ("MADLAD-400 10.7B", 10.7, 85, "claim"),
        ("TranslateGemma 4B", 4.3, 60, "claim"), ("TranslateGemma 27B", 27.0, 60, "claim"),
        ("AfriqueLLM 4-14B", 8.0, 20, "eval"), ("TranslatePsy 0.8B", 0.8, 19, "eval"), ("TranslatePsy 4B", 4.0, 19, "eval"),
        ("Aya-101", 13.0, 13, "claim"), ("Tiny Aya", 3.35, 10, "eval"), ("AfriNLLB", 0.548, 10, "eval"), ("InkubaLM", 0.4, 5, "claim"), ("mBART-50", 0.61, 3, "claim"),
        ("Lugha-Llama", 8.0, 16, "eval"), ("Hunyuan-MT-7B", 7.0, 0, "zero"), ("Llama 3.1 8B", 8.0, 0, "zero"), ("Aya Expanse 8B", 8.0, 0, "zero"),
    ]
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    for lab, p, n, kind in pts:
        if kind == "ours":
            ax.plot(p, n, "*", ms=13, color=OI["verm"], mec="black", mew=0.6, zorder=6)
        elif kind == "eval":
            ax.plot(p, n, "o", ms=4.5, color=OI["blue"])
        elif kind == "claim":
            ax.plot(p, n, "o", ms=4.5, mfc="none", mec=OI["blue"], mew=1.0)
        else:
            ax.plot(p, n, "x", ms=4.5, color=OI["grey"])
    for lab, p, n, kind in pts:
        if lab in ("SabiYarn-270M (ours)", "NLLB-200 600M", "NLLB-200 MoE 54.5B", "MADLAD-400 10.7B", "TranslateGemma 27B", "AfriqueLLM 4-14B", "TranslatePsy 0.8B", "Aya-101", "InkubaLM", "AfriNLLB", "Tiny Aya"):
            ax.annotate(lab, (p, n), textcoords="offset points", xytext=(6, -3) if "ours" in lab else (4, 3), fontsize=5.5, fontweight="bold" if "ours" in lab else "normal")
    ax.set_xscale("log")
    ax.set_xticks([0.3, 1, 3, 10, 30]); ax.set_xticklabels(["0.3", "1", "3", "10", "30"]); ax.minorticks_off()
    ax.set_xlabel("parameters (B)")
    ax.set_ylabel("African languages supported (as claimed / trained)")
    ax.set_ylim(-4, 100)
    h = [Line2D([], [], marker="*", ls="", color=OI["verm"], mec="black", ms=9, label="ours"), Line2D([], [], marker="o", ls="", color=OI["blue"], ms=5, label="per-language en->X results"),
         Line2D([], [], marker="o", ls="", mfc="none", mec=OI["blue"], label="claimed, no per-language results"), Line2D([], [], marker="x", ls="", color=OI["grey"], label="no African target")]
    ax.legend(handles=h, fontsize=5.5, loc="center right", bbox_to_anchor=(1.0, 0.36), frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"fig13_language_coverage.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    what = sys.argv[1:] or ["master", "tables", "figs"]
    if "master" in what:
        build_master()
    if "tables" in what:
        build_tables()
    if "figs" in what:
        build_figs()
