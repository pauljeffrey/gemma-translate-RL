#!/usr/bin/env python
"""Step 2: every comparison table of the full results comparison, from per-sentence data (CPU only, ~minutes).

usage: python s02_compute.py CACHE_DIR OUT_DIR           (CACHE_DIR from s01_rescore.py; python with numpy + sacrebleu 2.6.0 + sentencepiece)

Outputs in OUT_DIR:
  per_language_long.csv   one row per (group, system, language): n rows, chrF++, chrF, spBLEU, AfriCOMET (+ rows with a score), kind of source
  macro_table.csv         one row per (group, system): language counts, macros, 95% paired-bootstrap intervals of the macro, difference RL minus system
  published_block.csv     whole-devtest published NLLB numbers (FLORES-200) next to our FLORES+ numbers on all rows (separate block, not paired)
  ssa_reward_family.csv   SSA-COMET-MTL reference-based (reward family, NOT independent) where it exists
  checks.json             integrity checks (alignment of sentences, evidence statistics, NLLB aggregate vs per-sentence, row counts)
Row sets (see report): all | np270 (reference pair not seen by the 270M SFT) | clean270 (English not in the 270M SFT text) | clean270 intersect clean385.
Bootstrap: Poisson(1) weights per source sentence (shared across languages for FLORES+/NTREX), 1,000 resamples, seed 20261104, 95% percentile intervals,
language macro of the replicates, the SAME weights for every system (paired).
"""
import csv
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frc_lib import *  # noqa

CACHE, OUT = sys.argv[1], sys.argv[2]
B, SEED = 1000, 20261104
os.makedirs(OUT, exist_ok=True)
checks = {}

# ------------------------------------------------------------------------------------------------ load
ours = {n: load_ours(n, CACHE) for n in OURS}
e2 = {n: load_e2(n) for n in E2_SYSTEMS}
SYS = {**ours, **e2}
checks["evidence_stats_check"] = {n: {"rows_checked": s.check[0], "mismatching_rows": s.check[1]} for n, s in ours.items()}
agg = nllb_aggregates()
BENCH = ["flores_plus", "ntrex", "mafand", "smol_sent", "salt", "glot_story", "ibom_mt", "flores_moz"]
FOLD = {b: load_flags(b, "old") for b in ["flores_plus", "ntrex", "mafand", "smol_sent", "salt"]}
F385 = {b: load_flags(b, "385") for b in ["flores_plus", "ntrex", "mafand", "smol_sent", "salt"]}


def canon(bench, iso):
    return list(SYS["sft385"].d[(bench, iso)]["ids"])


def rowset(bench, iso, rs):
    ids = canon(bench, iso)
    if rs == "all":
        return ids
    if rs == "clean270":
        return [i for i in ids if FOLD[bench][0].get((iso, i), False)]
    if rs == "np270":
        return [i for i in ids if FOLD[bench][1].get((iso, i), False)]
    if rs == "clean270_385":
        return [i for i in ids if FOLD[bench][0].get((iso, i), False) and F385[bench][0].get((iso, i), False)]
    raise ValueError(rs)


# ------------------------------------------------------------------------------------------------ checks: sentence alignment ours vs e2 (FLORES+), NLLB aggregate vs per-sentence
al = {"flores_plus_text_mismatch": 0, "flores_plus_rows": 0}
pred = {}
for line in open(ARC + "eval_385m_rl_predictions_2026-11-04/out_wk_mixedbest/predictions.jsonl"):
    d = json.loads(line)
    if d["benchmark"] == "flores_plus":
        pred[(d["iso"], str(d["id"]))] = (d["english"], d["reference"])
for iso, m in e2["nllb3b3"].meta.items():
    for i, s, r in zip(e2["nllb3b3"].d[("flores_plus", iso)]["ids"], m["src"], m["ref"]):
        al["flores_plus_rows"] += 1
        al["flores_plus_text_mismatch"] += int(pred.get((iso, i)) != (s, r))
checks["alignment_ours_vs_nllb_flores"] = al
# NLLB aggregate vs per-sentence recomputation (same stack?)
diffs = []
for (b, sub, iso), a in agg.items():
    if b != "flores_plus" or not e2["nllb3b3"].has(b, iso) or np.isnan(a["n33_chrfpp"]):
        continue
    ids = rowset(b, iso, {"all": "all", "no_pair": "np270", "clean": "clean270"}[sub])
    st = e2["nllb3b3"].get(b, iso, ids)
    if st is None or len(ids) != a["n"]:
        continue
    diffs.append(float(chrfpp(st[0].sum(0)) - a["n33_chrfpp"]))
checks["nllb_aggregate_vs_persentence_chrfpp_3.3B"] = {"cells": len(diffs), "max_abs_diff": float(np.max(np.abs(diffs))) if diffs else None}
diffs = []
for (b, sub, iso), a in agg.items():
    if b != "flores_plus" or not e2["nllb3b3"].has(b, iso) or np.isnan(a["n33_spbleu"]):
        continue
    ids = rowset(b, iso, {"all": "all", "no_pair": "np270", "clean": "clean270"}[sub])
    st = e2["nllb3b3"].get(b, iso, ids)
    if st is None or len(ids) != a["n"]:
        continue
    diffs.append(float(spbleu_from_stats(st[1].sum(0)) - a["n33_spbleu"]))
checks["nllb_aggregate_vs_persentence_spbleu_3.3B"] = {"cells": len(diffs), "max_abs_diff": float(np.max(np.abs(diffs))) if diffs else None}
# row counts of the aggregate vs the reconstructed row sets
cnt_bad = []
for (b, sub, iso), a in agg.items():
    if b not in FOLD or not SYS["sft385"].has(b, iso):
        continue
    ids = rowset(b, iso, {"all": "all", "no_pair": "np270", "clean": "clean270"}[sub])
    if len(ids) != a["n"]:
        cnt_bad.append((b, sub, iso, len(ids), a["n"]))
checks["nllb_aggregate_row_count_mismatch"] = cnt_bad
# TranslatePsy-2B: usable rows
tp2 = e2["tpsy2b"]
checks["tpsy2b_empty_hyp_rows_np270"] = {iso: int(sum(1 for i, h in zip(tp2.d[("flores_plus", iso)]["ids"], tp2.meta[iso]["hyp"]) if FOLD["flores_plus"][1].get((iso, i), False) and not h.strip()))
                                         for iso in tp2.langs("flores_plus")}

# ------------------------------------------------------------------------------------------------ group definitions
OURS5 = ["sft385", "e03c1", "stageA", "b1best", "mixed"]
OLD = ["old_sft", "old_rl"]
GROUPS = [
    # id, title, bench, rowset, systems
    ("F_NP", "FLORES+ devtest, rows whose reference pair the 270M SFT never saw (no_pair)", "flores_plus", "np270", OURS5 + ["nllb600m", "nllb3b3"]),
    ("F_NP_T19", "FLORES+ devtest, no_pair rows, TranslatePsy-0.8B language set", "flores_plus", "np270", OURS5 + ["nllb600m", "nllb3b3", "tpsy08"]),
    ("F_NP_T9", "FLORES+ devtest, no_pair rows, TranslatePsy-2B language set", "flores_plus", "np270", OURS5 + ["nllb600m", "nllb3b3", "tpsy08", "tpsy2b"]),
    ("F_C270", "FLORES+ devtest, rows clean for the earlier 270M too (48 per language)", "flores_plus", "clean270", OURS5 + OLD + ["nllb600m", "nllb3b3"]),
    ("F_C270_T19", "FLORES+ devtest, clean-for-270M rows, TranslatePsy-0.8B language set", "flores_plus", "clean270", OURS5 + OLD + ["nllb600m", "nllb3b3", "tpsy08"]),
    ("F_ALL", "FLORES+ devtest, all 1,012 rows (385M: 0 flagged rows)", "flores_plus", "all", OURS5 + ["nllb600m", "nllb3b3"]),
    ("F_ALL_T19", "FLORES+ devtest, all rows, TranslatePsy-0.8B language set", "flores_plus", "all", OURS5 + ["nllb600m", "nllb3b3", "tpsy08"]),
    ("N_ALL", "NTREX-128, all rows (NLLB-3.3B aggregate only)", "ntrex", "all", OURS5 + ["nllb3b3"]),
    ("N_C270", "NTREX-128, rows clean for the earlier 270M (NLLB-3.3B aggregate only)", "ntrex", "clean270", OURS5 + OLD + ["nllb3b3"]),
    ("M_ALL", "MAFAND-MT test, all rows (NLLB-3.3B aggregate only)", "mafand", "all", OURS5 + ["nllb3b3"]),
    ("M_C270", "MAFAND-MT test, rows clean for the earlier 270M (NLLB-3.3B aggregate only)", "mafand", "clean270", OURS5 + OLD + ["nllb3b3"]),
    ("S_ALL", "SmolSent, all rows (385M SFT and NLLB-3.3B aggregate only; no RL model was evaluated)", "smol_sent", "all", ["sft385", "nllb3b3"]),
    ("SALT_ALL", "SALT, all rows (385M SFT and NLLB-3.3B aggregate only; no RL model was evaluated)", "salt", "all", ["sft385", "nllb3b3"]),
    ("G_ALL", "GlotStoryBook, all rows (385M SFT only; no comparator, no RL model)", "glot_story", "all", ["sft385"]),
    ("I_ALL", "IbomMT, all rows (385M SFT only)", "ibom_mt", "all", ["sft385"]),
    ("Z_ALL", "FLORES-MOZ, all rows (385M SFT only)", "flores_moz", "all", ["sft385"]),
]
AGG_SYS = {"nllb3b3": "n33"}                      # systems that exist as per-language aggregates on non-FLORES+ benchmarks
AGG_SUBSET = {"all": "all", "np270": "no_pair", "clean270": "clean"}


def persentence(sysname, bench):
    return not (bench != "flores_plus" and sysname in AGG_SYS)


def ci(x):
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def run_group(gid, title, bench, rs, systems):
    # --- languages: where every system has data on all rows of the row set
    cand = None
    for s in systems:
        if persentence(s, bench):
            L = {iso for (b, iso) in SYS[s].d if b == bench}
        else:
            L = {iso for (b, sub, iso) in agg if b == bench and sub == AGG_SUBSET[rs] and not np.isnan(agg[(b, sub, iso)]["n33_chrfpp"])}
        cand = L if cand is None else cand & L
    cand = sorted(i for i in cand if SYS["sft385"].has(bench, i))
    langs, dropped = [], []
    rows = {}
    for iso in cand:
        ids = rowset(bench, iso, rs)
        ok = len(ids) > 0
        for s in systems:
            if persentence(s, bench):
                ok = ok and SYS[s].get(bench, iso, ids) is not None
            else:
                a = agg[(bench, AGG_SUBSET[rs], iso)]
                ok = ok and a["n"] == len(ids)
        (langs if ok else dropped).append(iso)
        rows[iso] = ids
    # --- AfriCOMET availability: rows scored for every per-sentence system with any score in the language
    comet_mask, comet_langs = {}, []
    cs = [s for s in systems if persentence(s, bench)]
    for iso in langs:
        masks = []
        anyscore = []
        for s in cs:
            c = SYS[s].get(bench, iso, rows[iso])[2]
            anyscore.append(np.isfinite(c).any())
            masks.append(np.isfinite(c))
        # a language counts for AfriCOMET only if every per-sentence system has scores on it
        if all(anyscore):
            m = np.logical_and.reduce(masks)
            if m.sum() > 0:
                comet_mask[iso] = m
                comet_langs.append(iso)
    # AfriCOMET of NLLB / TranslatePsy exists only on the no_pair rows: on the all-rows FLORES+ groups the AfriCOMET cells would be on a DIFFERENT row set, so they are left empty
    if bench == "flores_plus" and rs == "all":
        comet_mask, comet_langs = {}, []
    # --- per system / language point values and bootstrap replicates
    res = {}
    W = {iso: weights(bench, iso, rows[iso], B, SEED).astype(np.float32) for iso in langs}
    long_rows = []
    for s in systems:
        res[s] = {"chrfpp": {}, "chrf": {}, "spbleu": {}, "comet": {}, "n_comet": {}, "rep_chrfpp": {}, "rep_comet": {}}
        for iso in langs:
            ids = rows[iso]
            if persentence(s, bench):
                chrf_st, bleu_st, comet = SYS[s].get(bench, iso, ids)
                tot = chrf_st.sum(0).astype(np.float64)
                res[s]["chrfpp"][iso] = float(chrfpp(tot))
                res[s]["chrf"][iso] = float(chrf(tot))
                res[s]["spbleu"][iso] = float(spbleu_from_stats(bleu_st.sum(0)))
                res[s]["rep_chrfpp"][iso] = chrfpp(W[iso].T.astype(np.float64) @ chrf_st.astype(np.float64))
                if iso in comet_mask:
                    m = comet_mask[iso]
                    res[s]["comet"][iso] = float(comet[m].mean())
                    res[s]["n_comet"][iso] = int(m.sum())
                    wm = W[iso][m].astype(np.float64)
                    res[s]["rep_comet"][iso] = (wm.T @ comet[m]) / wm.sum(0)
                src = "per-sentence"
            else:
                a = agg[(bench, AGG_SUBSET[rs], iso)]
                res[s]["chrfpp"][iso] = a["n33_chrfpp"]
                res[s]["spbleu"][iso] = a["n33_spbleu"]
                src = "aggregate (corpus-level per language, 270M-era run)"
            long_rows.append(dict(group=gid, benchmark=bench, row_set=rs, system=s, label=LABEL[s], iso=iso, n_rows=len(ids), source=src,
                                  chrfpp=res[s]["chrfpp"][iso], chrf=res[s]["chrf"].get(iso, np.nan), spbleu=res[s]["spbleu"][iso],
                                  africomet=res[s]["comet"].get(iso, np.nan), n_comet_rows=res[s]["n_comet"].get(iso, 0)))
    # --- macro
    macro_rows = []
    rl = "mixed" if "mixed" in systems else None
    for s in systems:
        r = res[s]
        mac = {k: (float(np.mean([r[k][i] for i in langs])) if r[k] else np.nan) for k in ("chrfpp", "chrf", "spbleu")}
        mc = float(np.mean([r["comet"][i] for i in comet_langs])) if r["comet"] and comet_langs else np.nan
        mcf = float(np.mean([r["chrfpp"][i] for i in comet_langs])) if comet_langs else np.nan
        row = dict(group=gid, title=title, benchmark=bench, row_set=rs, system=s, label=LABEL[s], n_lang=len(langs), n_lang_comet=len(comet_langs) if r["comet"] else 0,
                   n_sentences=int(sum(len(rows[i]) for i in langs)), chrfpp=mac["chrfpp"], chrf=mac["chrf"], spbleu=mac["spbleu"], africomet=mc,
                   chrfpp_on_comet_langs=mcf if r["comet"] else np.nan, source="per-sentence" if persentence(s, bench) else "aggregate")
        if persentence(s, bench):
            rep = np.mean([r["rep_chrfpp"][i] for i in langs], axis=0)
            row["chrfpp_lo"], row["chrfpp_hi"] = ci(rep)
            if r["rep_comet"] and comet_langs:
                repc = np.mean([r["rep_comet"][i] for i in comet_langs], axis=0)
                row["africomet_lo"], row["africomet_hi"] = ci(repc)
        # difference RL - system
        if rl and s != rl:
            a, b_ = res[rl], res[s]
            row["d_chrfpp_RL_minus_sys"] = a_m = float(np.mean([a["chrfpp"][i] for i in langs]) - mac["chrfpp"])
            if persentence(s, bench):
                rep = np.mean([a["rep_chrfpp"][i] - b_["rep_chrfpp"][i] for i in langs], axis=0)
                row["d_chrfpp_lo"], row["d_chrfpp_hi"] = ci(rep)
                row["d_ci_kind"] = "paired sentence bootstrap"
                if b_["rep_comet"] and comet_langs:
                    row["d_africomet_RL_minus_sys"] = float(np.mean([a["comet"][i] for i in comet_langs]) - np.mean([b_["comet"][i] for i in comet_langs]))
                    rep = np.mean([a["rep_comet"][i] - b_["rep_comet"][i] for i in comet_langs], axis=0)
                    row["d_africomet_lo"], row["d_africomet_hi"] = ci(rep)
            else:
                # no per-sentence output: language bootstrap over the common languages (language sampling noise only)
                d = np.array([a["chrfpp"][i] - b_["chrfpp"][i] for i in langs])
                rng = np.random.default_rng(SEED)
                bm = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
                row["d_chrfpp_lo"], row["d_chrfpp_hi"] = ci(bm)
                row["d_ci_kind"] = "language bootstrap only (no per-sentence output archived)"
            wins = sum(1 for i in langs if a["chrfpp"][i] > b_["chrfpp"][i])
            row["n_lang_RL_above"] = wins
        macro_rows.append(row)
    return long_rows, macro_rows, dict(group=gid, langs=langs, dropped=dropped, comet_langs=comet_langs,
                                       comet_row_coverage={i: float(comet_mask[i].mean()) for i in comet_langs})


all_long, all_macro, gmeta = [], [], {}
for gid, title, bench, rs, systems in GROUPS:
    if rs == "all" and bench == "flores_plus":
        pass
    lr, mr, meta = run_group(gid, title, bench, rs, systems)
    all_long += lr
    all_macro += mr
    gmeta[gid] = meta
    m0 = {r["system"]: r for r in mr}
    print(gid, "langs", len(meta["langs"]), "comet langs", len(meta["comet_langs"]), "dropped", meta["dropped"],
          {s: round(r["chrfpp"], 2) for s, r in m0.items()}, flush=True)

checks["groups"] = gmeta


def write_csv(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if (isinstance(v, float) and np.isnan(v)) else (round(v, 5) if isinstance(v, float) else v)) for k, v in r.items()})


write_csv(os.path.join(OUT, "per_language_long.csv"), all_long)
write_csv(os.path.join(OUT, "macro_table.csv"), all_macro)

# ------------------------------------------------------------------------------------------------ block (a): published whole-devtest numbers
pub = {}
for r in csv.DictReader(open(REPO + "reports/data/eval_385m_results_2026-11-03/published_nllb_flores_per_language.csv")):
    pub[r["iso"]] = r
PUBCOL = {"nllb_3p3b": "NLLB-200-3.3B (published)", "nllb_1p3b": "NLLB-200-1.3B (published)", "nllb_distilled_1p3b": "NLLB-200-distilled-1.3B (published)",
          "nllb_moe_54p5b": "NLLB-200-MoE-54.5B (published)"}
plangs = [iso for iso, r in pub.items() if r["nllb_3p3b"] not in ("", "nan") and all(SYS["mixed"].has("flores_plus", iso) for _ in [0])]
plangs = [i for i in plangs if all(SYS[s].has("flores_plus", i) for s in OURS5)]
pub_rows = []
rng = np.random.default_rng(SEED)
vals = {}
for s in OURS5 + ["nllb600m", "nllb3b3"]:
    vals[s] = {}
    for iso in plangs:
        ids = rowset("flores_plus", iso, "all")
        st = SYS[s].get("flores_plus", iso, ids)
        if st is not None:
            vals[s][iso] = float(chrfpp(st[0].sum(0)))
for col, lab in PUBCOL.items():
    vals[lab] = {iso: float(pub[iso][col]) for iso in plangs if pub[iso][col] not in ("", "nan")}
langs_pub = [i for i in plangs if all(i in v for v in vals.values())]
rl = vals["mixed"]
for key, v in vals.items():
    row = dict(system=LABEL.get(key, key), kind="our rescoring, FLORES+ all 1,012 rows" if key in LABEL else "published FLORES-200 devtest chrF++ (NLLB release metrics file, beam, 1 decimal)",
               n_lang=len(langs_pub), chrfpp_macro=float(np.mean([v[i] for i in langs_pub])))
    if key != "mixed":
        d = np.array([rl[i] - v[i] for i in langs_pub])
        bm = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
        row["RL_minus_system"] = float(d.mean())
        row["lang_boot_lo"], row["lang_boot_hi"] = ci(bm)
        row["n_lang_RL_above"] = int((d > 0).sum())
    pub_rows.append(row)
write_csv(os.path.join(OUT, "published_block.csv"), pub_rows)
pl = []
for iso in langs_pub:
    pl.append(dict(iso=iso, **{(LABEL.get(k, k)): round(vals[k][iso], 3) for k in vals}))
write_csv(os.path.join(OUT, "published_block_per_language.csv"), pl)
checks["published_block_languages"] = langs_pub

# ------------------------------------------------------------------------------------------------ SSA-COMET (reward family): where it exists (e2 only, 22 languages, no_pair rows, 270M-era scoring run)
ssa_rows = []
ssa_sys = ["nllb600m", "nllb3b3", "tpsy08", "tpsy2b"]
SSA = {s: e2[s].ssa for s in ssa_sys}
for s, lab in (("rl", "270M RL-final (first research's own run, earlier code state; NOT the E-13 rerun)"), ("sft", "270M SFT-final (first research's own run, earlier code state; NOT the E-13 rerun)")):
    SSA[s] = pickle.load(open(E2 + s + ".ssa.pkl", "rb"))
    LABEL[s] = lab
for s, d in SSA.items():
    vals_ = []
    for iso in sorted(d):
        ids = rowset("flores_plus", iso, "np270")
        sc = np.array([d[iso].get(i, np.nan) for i in ids])
        if np.isfinite(sc).any():
            vals_.append((iso, float(np.nanmean(sc)) * 100, int(np.isfinite(sc).sum())))
    ssa_rows.append(dict(system=LABEL[s], n_lang=len(vals_), n_sentences=int(sum(v[2] for v in vals_)), ssa_comet_mtl_ref_x100_macro=float(np.mean([v[1] for v in vals_])),
                         row_set="FLORES+ no_pair rows (270M flag)", languages=" ".join(v[0] for v in vals_)))
# the same macro restricted to the 19 TranslatePsy-0.8B languages (systems with all 19) and to the 9 TranslatePsy-2B languages (every system)
for tag, sset in (("macro19_x100", set(e2["tpsy08"].ssa)), ("macro9_x100", set(e2["tpsy2b"].ssa))):
    for s_, d in SSA.items():
        if not sset <= set(d):
            continue
        v = []
        for iso in sorted(sset):
            ids = rowset("flores_plus", iso, "np270")
            sc = np.array([d[iso].get(i, np.nan) for i in ids])
            v.append(float(np.nanmean(sc)) * 100)
        for r in ssa_rows:
            if r["system"] == LABEL[s_]:
                r[tag] = float(np.mean(v))
write_csv(os.path.join(OUT, "ssa_reward_family.csv"), ssa_rows)

json.dump(checks, open(os.path.join(OUT, "checks.json"), "w"), indent=1, default=str)
print("checks:", json.dumps({k: v for k, v in checks.items() if k != "groups"}, default=str)[:3000])
