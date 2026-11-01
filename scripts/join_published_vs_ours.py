#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Join OUR per-language scores with the PUBLISHED baselines (D-51) and write a comparison report.

Inputs (both CSV, stdlib only, no GPU):
  --published  reports/data/published_baselines.csv      (long format, see reports/published_baselines_2026-10-21.md)
  --ours       reports/data/final_model_per_language.csv  (one row per language x benchmark x checkpoint [x subset])
Output:
  --out        reports/comparison_with_published_2026-10-21.md

Our file is read tolerantly. Column names are matched case-insensitively and ignoring punctuation:
  iso                       required  (`iso`, `lang`, `language_iso`)
  chrF++                    required  (`chrf++`, `chrf`, `chrF++`)
  BLEU                      optional  (`bleu`) = sacreBLEU 13a, never compared with spBLEU
  spBLEU                    optional  (`spbleu`) = sacreBLEU flores200 tokenizer
  benchmark                 optional  (`benchmark`, `bench`); default selection `flores_plus`
  checkpoint                optional  (`checkpoint`, `system`, `model`, `ckpt`); `base` rows are ignored
  subset                    optional  (`subset`, `split`); default `all` (every row); use `--subset clean` for rows whose
                                      English sentence is not in the SFT text (the model has seen 95% of FLORES devtest English)
A file in the format of reports/gemma_sft_benchmarks_2026-10-18/full_flores/results.csv works as is.

What is compared: English -> X, FLORES (our FLORES+ devtest vs the benchmark the other system reports), chrF++ per language;
spBLEU only against published spBLEU (NLLB-200 54.5B MoE is the only per-language source). Published rows are used only when
direction == en->xx, a number exists, and the row is not flagged (`low_n` Gemini rows with 10 sentences, `aggregate`
rows, `no_number`, `no_chrf`) unless --include-low-n. Aggregates (macro over 19 languages, averaged over BOTH directions) are shown
in a separate table next to OUR macro over the same languages and are NOT to be read as a like-for-like comparison.

Usage:
  python gemma_mt/scripts/join_published_vs_ours.py                       # defaults
  python gemma_mt/scripts/join_published_vs_ours.py --ours some.csv --checkpoint step3000 --subset clean --models all
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]                     # <repo>/gemma_mt/scripts/this -> <repo>
DEFAULT_MODELS = ["NLLB-200-1.3B", "NLLB-200-3.3B", "AfriqueLlama-8B", "AfriqueGemma-12B", "AfriqueQwen-14B"]
NOT_COMPARABLE_FLAGS = {"low_n", "aggregate", "no_number", "no_chrf"}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9+]", "", (s or "").lower())


def find_col(header, aliases):
    idx = {norm(h): h for h in header}
    for a in aliases:
        if norm(a) in idx:
            return idx[norm(a)]
    return None


def num(x):
    try:
        v = float(str(x).strip())
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def load_names():
    try:
        sys.path.insert(0, str(ROOT / "gemma_mt"))
        from gemma_mt.languages import LANGUAGES                  # type: ignore
        return {k: v.name for k, v in LANGUAGES.items()}
    except Exception:
        return {}


def read_ours(path, benchmark, checkpoint, subset):
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        hdr = rd.fieldnames or []
        rows = list(rd)
    c_iso = find_col(hdr, ["iso", "lang", "language_iso", "language"])
    c_chrf = find_col(hdr, ["chrF++", "chrf", "chrf_pp"])
    c_bleu = find_col(hdr, ["BLEU", "bleu_13a"])
    c_sp = find_col(hdr, ["spBLEU", "sp_bleu"])
    c_bm = find_col(hdr, ["benchmark", "bench", "dataset"])
    c_ck = find_col(hdr, ["checkpoint", "system", "model", "ckpt"])
    c_sub = find_col(hdr, ["subset", "split"])
    if not (c_iso and c_chrf):
        sys.exit(f"{path}: need an iso column and a chrF++ column, found {hdr}")
    info = {"columns": hdr, "benchmark_col": c_bm, "checkpoint_col": c_ck, "subset_col": c_sub, "spbleu_col": c_sp, "bleu_col": c_bleu}
    if c_bm and benchmark != "any":
        rows = [r for r in rows if benchmark.lower() in (r[c_bm] or "").lower()]
    if c_ck:
        cks = list(OrderedDict.fromkeys(r[c_ck] for r in rows))
        cks = [c for c in cks if norm(c) not in ("base", "gemma3270mit", "baseline")]
        if checkpoint:
            cks = [c for c in cks if c == checkpoint] or sys.exit(f"checkpoint {checkpoint!r} not in {cks}")
        elif len(cks) > 1:
            fin = [c for c in cks if "final" in c.lower()]
            print(f"[join] several checkpoints {cks}; using {(fin[-1] if fin else cks[-1])!r} (pass --checkpoint to choose)", file=sys.stderr)
            cks = [fin[-1] if fin else cks[-1]]
        rows = [r for r in rows if r[c_ck] in cks]
        info["checkpoint"] = cks[0] if cks else None
    else:
        info["checkpoint"] = None
    if c_sub and subset:
        rows = [r for r in rows if (r[c_sub] or "").lower() == subset.lower()]
    out = {}
    for r in rows:
        iso = (r[c_iso] or "").strip()
        v = num(r[c_chrf])
        if not iso or v is None or norm(iso) in ("macro", "all", "avg", "mean", "total") or iso.upper().startswith("MACRO"):
            continue
        out.setdefault(iso, {"chrf": v, "spbleu": num(r[c_sp]) if c_sp else None, "bleu": num(r[c_bleu]) if c_bleu else None,
                             "n": num(r.get("n")) if "n" in r else None})
    return out, info


def read_published(path, include_low_n):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    per = defaultdict(dict)        # model -> iso -> dict
    agg = []
    for r in rows:
        chrf = num(r.get("chrF++"))
        flag = (r.get("flag") or "").strip()
        iso = r["iso"]
        if iso.startswith("MACRO"):
            agg.append(r)
            continue
        if not r["direction"].startswith("en->") or chrf is None:
            continue
        if flag in NOT_COMPARABLE_FLAGS and not (flag == "low_n" and include_low_n):
            continue
        d = per[r["model"]]
        if iso not in d:
            d[iso] = {"chrf": chrf, "spbleu": num(r.get("BLEU_spBLEU")) if (r.get("bleu_kind") or "").lower().startswith("spbleu") else None,
                      "flag": flag, "bench": r.get("benchmark", ""), "src": r.get("source_url", "")}
    return per, agg, rows


def f1(x):
    return "NA" if x is None else f"{x:.1f}"


def macro(vals):
    vals = [v for v in vals if v is not None]
    return statistics.fmean(vals) if vals else None


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--published", default=str(ROOT / "reports/data/published_baselines.csv"))
    ap.add_argument("--ours", default=str(ROOT / "reports/data/final_model_per_language.csv"))
    ap.add_argument("--out", default=str(ROOT / "reports/comparison_with_published_2026-10-21.md"))
    ap.add_argument("--benchmark", default="flores_plus", help="substring of OUR benchmark column (or 'any'); default flores_plus")
    ap.add_argument("--checkpoint", default=None, help="OUR checkpoint/system name; default: the only non-base one, else the last 'final*'")
    ap.add_argument("--subset", default="all", help="OUR subset column value (all | clean | ...); default all")
    ap.add_argument("--models", default="default", help="comma list of published model names, 'default' (brief models) or 'all' (every per-language model)")
    ap.add_argument("--include-low-n", action="store_true", help="also use Gemini rows with 10 sentences per language (very noisy)")
    ap.add_argument("--label", default=None, help="name of OUR system in the tables")
    a = ap.parse_args(argv)

    for p in (a.published, a.ours):
        if not Path(p).exists():
            sys.exit(f"missing input {p}" + ("  (our per-language csv is written by another agent; pass --ours)" if p == a.ours else ""))
    names = load_names()
    ours, info = read_ours(a.ours, a.benchmark, a.checkpoint, a.subset)
    per, agg, _ = read_published(a.published, a.include_low_n)
    if not ours:
        sys.exit("no rows of ours left after the benchmark/checkpoint/subset filter")
    label = a.label or f"ours ({info['checkpoint'] or Path(a.ours).stem})"
    if a.models == "all":
        models = sorted(per)
    elif a.models == "default":
        models = [m for m in DEFAULT_MODELS if m in per]
        if a.include_low_n and "Gemini-2.5-Flash" in per:
            models.append("Gemini-2.5-Flash")
    else:
        models = [m.strip() for m in a.models.split(",") if m.strip()]
        miss = [m for m in models if m not in per]
        if miss:
            print(f"[join] models without usable rows: {miss}", file=sys.stderr)
        models = [m for m in models if m in per]
    langs_pub = sorted({i for m in models for i in per[m]})
    langs = [i for i in ours if i in langs_pub]
    lines = []
    w = lines.append
    w("# Our model vs published results, English -> African (generated)\n")
    w(f"Generated by `gemma_mt/scripts/join_published_vs_ours.py`. Ours: `{a.ours}`, benchmark filter `{a.benchmark}`, checkpoint `{info['checkpoint']}`, subset `{a.subset}`; "
      f"{len(ours)} languages with a chrF++. Published: `{a.published}` ({len(models)} systems: {', '.join(models)}).\n")
    w("Read `reports/published_baselines_2026-10-21.md` first: benchmark versions differ (FLORES+ devtest here vs FLORES-200 devtest or an unstated split there), "
      "decoding/prompting differ (NLLB beam search; Afrique base models 5-shot), and OUR model has seen the English side of 95% of the FLORES devtest sentences in SFT "
      "(the reference pair only for 7 languages): use `--subset clean` (48 sentences per language, noisy) as a cross-check. Differences below ~2 chrF++ points are not meaningful.\n")

    # ---- coverage
    w("## 1. Coverage\n")
    rows = []
    for m in models:
        shared = [i for i in per[m] if i in ours]
        rows.append([m, len(per[m]), len(shared)])
    w(md_table(["system", "our-77 languages with a published en->X chrF++", "of which we have a number"], rows) + "\n")
    w(f"Languages with at least one published number AND ours: **{len(langs)}**. Our languages without any: {', '.join(i for i in ours if i not in langs_pub) or 'none'}.\n")

    # ---- per language
    w("## 2. Per language, chrF++ (English -> X)\n")
    hdr = ["iso", "language", label] + models + ["best published", "worst published", "ours rank", "ours - best published"]
    rows = []
    wins = []
    deficits = []
    for i in langs:
        o = ours[i]["chrf"]
        vals = {m: per[m][i]["chrf"] for m in models if i in per[m]}
        allv = [(o, label)] + [(v, m) for m, v in vals.items()]
        pub = [(v, m) for m, v in vals.items()]
        best_v, best_m = max(pub, key=lambda t: t[0])
        worst_v, worst_m = min(pub, key=lambda t: t[0])
        rank = 1 + sum(1 for v, _ in allv if v > o)
        rows.append([i, names.get(i, i), f1(o)] + [f1(vals.get(m)) for m in models]
                    + [f"{best_m} {best_v:.1f}", f"{worst_m} {worst_v:.1f}", f"{rank}/{len(allv)}", f"{o - best_v:+.1f}"])
        if o > best_v:
            wins.append(i)
        deficits.append((o - best_v, i, best_m))
    w(md_table(hdr, rows) + "\n")
    w(f"Rank = position of ours among the systems that have a number for the language (1 = best); `ours - best published` is positive when we lead. "
      f"Ours is best on {len(wins)} of {len(langs)} languages" + (f" ({', '.join(wins)})" if wins else "") + ".\n")
    if deficits:
        deficits.sort()
        w("Largest deficits to the best published system: " + "; ".join(f"{i} {d:+.1f} ({m})" for d, i, m in deficits[:8]) + ".\n")
        w("Largest leads / smallest deficits: " + "; ".join(f"{i} {d:+.1f} ({m})" for d, i, m in deficits[::-1][:5]) + ".\n")

    # ---- macro on shared languages
    w("## 3. Macro chrF++ on the shared languages\n")
    rows = []
    for m in models:
        sh = [i for i in per[m] if i in ours]
        if not sh:
            continue
        mo, mm = macro([ours[i]["chrf"] for i in sh]), macro([per[m][i]["chrf"] for i in sh])
        win = sum(1 for i in sh if ours[i]["chrf"] > per[m][i]["chrf"])
        rows.append([m, len(sh), f1(mo), f1(mm), f"{mo - mm:+.1f}", f"{win}/{len(sh)}"])
    w(md_table(["system", "shared languages", f"macro {label}", "macro system", "difference", "ours better on"], rows) + "\n")
    common = [i for i in ours if all(i in per[m] for m in models)] if models else []
    if len(models) > 1 and common:
        rows = [[m, f1(macro([per[m][i]["chrf"] for i in common]))] for m in models]
        rows.append([label, f1(macro([ours[i]["chrf"] for i in common]))])
        w(f"Languages covered by ALL {len(models)} systems and by us ({len(common)}: {', '.join(common)}):\n")
        w(md_table(["system", "macro chrF++ on the common set"], rows) + "\n")
    else:
        w(f"No language is covered by all {len(models)} selected systems at once, so no common-set macro is shown.\n")

    # ---- spBLEU
    sp_models = [m for m in per if any(v["spbleu"] is not None for v in per[m].values())]
    if sp_models and any(v["spbleu"] is not None for v in ours.values()):
        w("## 4. spBLEU (flores200 tokenizer) on the shared languages\n")
        rows = []
        for m in sp_models:
            sh = [i for i in per[m] if i in ours and ours[i]["spbleu"] is not None and per[m][i]["spbleu"] is not None]
            if sh:
                mo, mm = macro([ours[i]["spbleu"] for i in sh]), macro([per[m][i]["spbleu"] for i in sh])
                rows.append([m, len(sh), f1(mo), f1(mm), f"{mo - mm:+.1f}"])
        w(md_table(["system", "shared languages", "macro ours", "macro system", "difference"], rows) + "\n")
    else:
        w("## 4. spBLEU\n")
        w("Not shown: either our file has no spBLEU column or no selected system publishes per-language spBLEU "
          "(only NLLB-200 54.5B MoE does; use `--models NLLB-200-MoE-54.5B`). Our `BLEU` (13a) is never compared with published spBLEU.\n")

    # ---- aggregates
    w("## 5. Published macros that are NOT per-language en->X (read the caveat)\n")
    rows = []
    for r in agg:
        if not r["iso"].startswith("MACRO19") or "FLORES" not in r["benchmark"].upper():
            continue
        if a.models != "all" and not (r["model"] in models or r["model"].startswith(("TranslatePsy", "TranslateGemma"))):
            continue
        ls = [x for x in (r.get("lang_set") or "").split(",") if x]
        have = [x for x in ls if x in ours]
        if not have:
            continue
        rows.append([r["model"], f1(num(r["chrF++"])), f1(num(r["BLEU_spBLEU"])), f"{len(have)}/{len(ls)}", f1(macro([ours[x]["chrf"] for x in have]))])
    if rows:
        w("The 19-language macros of the TranslatePsy-AfriSLM paper (Table 25) are, by our check, averages of en->X and X->en (see the baselines report); "
          "X->en chrF++ is ~9 points higher than en->X, so these values sit roughly 4-5 points above the en->X macro of the same system (NLLB-200-3.3B: 48.9 published vs 45.2 from its release csv on the same languages). The last column is OUR en->X macro over the same languages.\n")
        w(md_table(["system", "published chrF++ (both directions, inferred)", "spBLEU", "of our languages in the set", "ours en->X chrF++ macro"], rows) + "\n")
    else:
        w("No aggregate row overlaps with our languages.\n")

    w("## 6. Caveats\n")
    for t in [
        "Our benchmark is FLORES+ devtest (1,012 sentences); the published numbers are FLORES-200 devtest (NLLB; the AfroBench task of the Afrique paper does not state the split). The sentences are the same set, with FLORES+ corrections of some references.",
        "NLLB numbers come from the model release (beam search); Afrique numbers are 5-shot greedy with the AfroBench chrF++ implementation; ours is greedy with sacreBLEU chrF++ (word_order 2). Metric implementations differ by up to ~1 point.",
        "Our model was trained on the English side of most FLORES devtest sentences; published systems were not. Compare with `--subset clean` as a sanity check.",
        "Twi: ours is Akuapem (twi_Latn_akua1239); published rows use the generic `twi_Latn`/`twi`. Oromo = gaz_Latn, Malagasy = plt_Latn, Swahili = swh_Latn everywhere.",
        "Languages without a number in a system are NA, never zero. Ranks and macros only use the languages both sides have.",
    ]:
        w(f"* {t}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[join] wrote {a.out}: {len(langs)} languages compared, {len(models)} systems")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
