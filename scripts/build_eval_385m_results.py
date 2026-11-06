#!/usr/bin/env python3
"""Build the result tables of the post-SFT evaluation of the 385M model (reports/eval_385m_results_2026-11-03.md) from the files that the A100 produced.

    python scripts/build_eval_385m_results.py --box DIR --out reports/data/eval_385m_results_2026-11-03 [--score score_g8] [--score-clean score_clean] [--comet comet_all] [--comet-clean comet_clean]

DIR is a copy of /workspace/eval_385m (only the small files are needed): out_m385/results.csv, out_a0/results.csv, <score>/{metrics_by_language,macro,bootstrap_pairs,bootstrap_by_language}.csv
(scripts/eval_385m_score.py), <comet>/{comet_macro,comet_by_language,comet_pair_by_language}.csv (scripts/comet_rows.py), decode_an/*.csv (scripts/decode_analyze.py), out_afsci_*/afriscience_eval.json.
Reference tables of the OLD 270M models and the published systems: reports/data/eval_385m_plan_2026-11-02/ (read from the repo).
Needs pandas (and tabulate for the markdown fragments). Every table is written as csv; `_fragments.md` holds the same tables as markdown for the report.
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
PLAN = os.path.join(REPO, "reports", "data", "eval_385m_plan_2026-11-02")
BENCH_ORDER = ["flores_plus", "ntrex", "mafand", "smol_sent", "salt", "glot_story", "ibom_mt", "flores_moz"]
BENCH_NAME = {"flores_plus": "FLORES+ devtest", "ntrex": "NTREX-128", "mafand": "MAFAND-MT test", "smol_sent": "SmolSent", "salt": "SALT", "glot_story": "GlotStoryBook", "ibom_mt": "IbomMT", "flores_moz": "FLORES-MOZ"}
GROUP_ORDER = ["Nguni", "Sotho-Tswana", "Other Bantu", "West African", "Cushitic / Ge'ez / Ethiopic", "Nilotic", "Berber / Arabic", "Creoles", "Other (Indo-European, Austronesian)"]
WEAK_CHRF, WEAK_FIN = 14.0, 0.90
FRAG = []


def md(df, title, fmt=None, index=False):
    FRAG.append(f"<!-- {title} -->\n" + df.to_markdown(index=index, floatfmt=fmt or ".2f") + "\n")
    return df


def save(df, out, name, title=None, fmt=None):
    df.to_csv(os.path.join(out, name), index=False)
    if title: md(df, title, fmt)
    print(f"{name}: {len(df)} rows")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--box", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--score", default="score_g8"); ap.add_argument("--comet", default="comet_all")
    ap.add_argument("--score-clean", default=None, help="scores with subsets all,clean,no_pair (needs the contamination flags attached to the predictions)")
    ap.add_argument("--comet-clean", default=None)
    ap.add_argument("--score-oldclean", default=None, help="scores of the 385M on the sentences that the OLD 270M models had not seen (old contamination flags attached to the 385M predictions)")
    ap.add_argument("--groups", default=None, help="json iso -> group (default <box>/groups_lang8.json)")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    B = a.box; P = lambda *x: os.path.join(B, *x)
    groups = json.load(open(a.groups or P("groups_lang8.json")))
    names = pd.read_csv(P("out_m385", "results.csv")); names = names[names.iso != "MACRO"][["iso", "language"]].drop_duplicates().set_index("iso")["language"].to_dict()
    names.update(pd.read_csv(P("out_a0", "results.csv")).query("iso != 'MACRO'")[["iso", "language"]].drop_duplicates().set_index("iso")["language"].to_dict())
    sc = P(a.score)
    met = pd.read_csv(os.path.join(sc, "metrics_by_language.csv")); mac = pd.read_csv(os.path.join(sc, "macro.csv"))
    bp = pd.read_csv(os.path.join(sc, "bootstrap_pairs.csv")); bl = pd.read_csv(os.path.join(sc, "bootstrap_by_language.csv"))
    cm = pd.read_csv(P(a.comet, "comet_macro.csv")); cl = pd.read_csv(P(a.comet, "comet_by_language.csv")); cpl = pd.read_csv(P(a.comet, "comet_pair_by_language.csv"))
    met["group"] = met.iso.map(groups).fillna("ungrouped"); met["language"] = met.iso.map(names)
    # ------------------------------------------------------------------ 1. greedy 385M vs A0, macro per benchmark (common languages), paired bootstrap
    rows = []
    for sub in sorted(met.subset.unique(), key=["all", "clean", "no_pair", "clean_strict"].index):
        for b in BENCH_ORDER:
            m = mac[(mac.benchmark == b) & (mac.subset == sub) & (mac.group == "all") & (mac.scope == "common")]
            p = bp[(bp.benchmark == b) & (bp.subset == sub) & (bp.group == "all")]
            if m.empty or p.empty: continue
            x, y = m[m.system == "m385"].iloc[0], m[m.system == "a0"].iloc[0]; q = p.iloc[0]
            c = cm[(cm.benchmark == b) & (cm["subset"] == sub)]
            r = dict(benchmark=BENCH_NAME[b], subset=sub, n_languages=int(x.n_languages), n_sentences=int(x.n_sentences),
                     chrf_m385=x.chrf, chrf_a0=y.chrf, d_chrf=q.d_chrf, d_chrf_lo=q.d_chrf_lo, d_chrf_hi=q.d_chrf_hi,
                     spbleu_m385=x.spbleu, spbleu_a0=y.spbleu, d_spbleu=q.d_spbleu, d_spbleu_lo=q.d_spbleu_lo, d_spbleu_hi=q.d_spbleu_hi,
                     finished_m385=x.finished, finished_a0=y.finished, lid_ok_m385=x.lid_ok, lid_ok_a0=y.lid_ok, repeat_m385=x["repeat"], repeat_a0=y["repeat"])
            if not c.empty:
                c = c.iloc[0]; r.update(africomet_n_languages=int(c.n_languages), africomet_m385=c.comet_b, africomet_a0=c.comet_a, d_africomet=c.d_comet, d_africomet_lo=c.d_lo, d_africomet_hi=c.d_hi,
                                        africomet_qe_m385=c.qe_b, africomet_qe_a0=c.qe_a)
            rows.append(r)
    t1 = pd.DataFrame(rows); save(t1, a.out, "greedy_macro_385m_vs_a0.csv")
    t1a = t1[t1.subset == "all"]
    md(pd.DataFrame({"benchmark": t1a.benchmark, "langs": t1a.n_languages, "chrF++ 385M": t1a.chrf_m385, "chrF++ A0": t1a.chrf_a0,
                     "delta [95% CI]": [f"{d:+.2f} [{l:+.2f}, {h:+.2f}]" for d, l, h in zip(t1a.d_chrf, t1a.d_chrf_lo, t1a.d_chrf_hi)],
                     "spBLEU 385M": t1a.spbleu_m385, "spBLEU A0": t1a.spbleu_a0, "delta spBLEU [95% CI]": [f"{d:+.2f} [{l:+.2f}, {h:+.2f}]" for d, l, h in zip(t1a.d_spbleu, t1a.d_spbleu_lo, t1a.d_spbleu_hi)],
                     "AfriCOMET 385M / A0 (langs)": [f"{x:.3f} / {y:.3f} ({int(n)})" if x == x else "-" for x, y, n in zip(t1a.get("africomet_m385", np.nan), t1a.get("africomet_a0", np.nan), t1a.get("africomet_n_languages", np.nan))],
                     "finished 385M / A0": [f"{x:.3f} / {y:.3f}" for x, y in zip(t1a.finished_m385, t1a.finished_a0)]}), "T1 greedy macro 385M vs A0 (all rows, common languages)")
    # the 385M on its own language set
    own = mac[(mac.system == "m385") & (mac.scope == "own") & (mac.group == "all") & (mac["subset"] == "all")].copy()
    own["benchmark"] = own.benchmark.map(BENCH_NAME); own = own.sort_values("benchmark")
    save(own[["benchmark", "n_languages", "n_sentences", "chrf", "spbleu", "finished", "empty", "copy", "repeat", "lid_ok"]], a.out, "greedy_macro_385m_own_language_set.csv", "T1b 385M own language set")
    # ------------------------------------------------------------------ 2. per language, all benchmarks
    w = met.pivot_table(index=["benchmark", "subset", "iso"], columns="system", values=["n", "chrf", "spbleu", "finished", "empty", "copy", "repeat", "lid_ok", "lid_ok_sib"], aggfunc="first")
    w.columns = [f"{v}_{s}" for v, s in w.columns]; w = w.reset_index()
    w = w.merge(bl[["benchmark", "subset", "iso", "d_chrf", "d_chrf_lo", "d_chrf_hi", "d_spbleu", "d_spbleu_lo", "d_spbleu_hi"]], on=["benchmark", "subset", "iso"], how="left")
    cc = cl.pivot_table(index=["benchmark", "subset", "iso"], columns="system", values=["comet", "comet_qe"], aggfunc="first"); cc.columns = [f"{v}_{s}" for v, s in cc.columns]
    w = w.merge(cc.reset_index(), on=["benchmark", "subset", "iso"], how="left")
    w = w.merge(cpl[["benchmark", "subset", "iso", "d_comet", "d_lo", "d_hi"]].rename(columns={"d_lo": "d_comet_lo", "d_hi": "d_comet_hi"}), on=["benchmark", "subset", "iso"], how="left")
    w["language"] = w.iso.map(names); w["group"] = w.iso.map(groups).fillna("ungrouped"); w["in_a0"] = w.chrf_a0.notna()
    w["benchmark"] = pd.Categorical(w.benchmark, BENCH_ORDER, ordered=True); w = w.sort_values(["benchmark", "subset", "group", "iso"]).reset_index(drop=True); w["benchmark"] = w.benchmark.astype(str)
    cols = ["benchmark", "subset", "iso", "language", "group", "in_a0", "n_m385", "chrf_m385", "spbleu_m385", "comet_m385", "comet_qe_m385", "finished_m385", "empty_m385", "lid_ok_m385", "lid_ok_sib_m385", "copy_m385", "repeat_m385",
            "n_a0", "chrf_a0", "spbleu_a0", "comet_a0", "finished_a0", "lid_ok_a0", "repeat_a0", "d_chrf", "d_chrf_lo", "d_chrf_hi", "d_spbleu", "d_spbleu_lo", "d_spbleu_hi", "d_comet", "d_comet_lo", "d_comet_hi"]
    save(w[[c for c in cols if c in w.columns]], a.out, "greedy_per_language_all_benchmarks.csv")
    # ------------------------------------------------------------------ 3. old 270M (clean / no_pair / all), published (NLLB)
    old = pd.read_csv(os.path.join(PLAN, "own_270m_per_language.csv"))
    ow = old.pivot_table(index=["benchmark", "iso"], columns="subset", values=["sft_chrf", "rl_chrf", "sft_finished", "rl_finished", "n"], aggfunc="first")
    ow.columns = [f"old_{v}_{s}".replace("old_sft_chrf", "old_sft_chrf").replace("old_n_", "old_n_") for v, s in ow.columns]; ow = ow.reset_index()
    pub = pd.read_csv(os.path.join(PLAN, "published_per_language.csv"))
    nl = pub[(pub.direction == "en->xx") & pub.model.isin(["NLLB-200-3.3B", "NLLB-200-1.3B", "NLLB-200-distilled-1.3B", "NLLB-200-MoE-54.5B"])].pivot_table(index="iso", columns="model", values="chrF++", aggfunc="first")
    nl.columns = ["nllb_" + c.replace("NLLB-200-", "").replace(".", "p").replace("-", "_").lower() for c in nl.columns]; nl = nl.reset_index()
    fl = w[(w.benchmark == "flores_plus") & (w.subset == "all")].merge(ow[ow.benchmark == "flores_plus"].drop(columns="benchmark"), on="iso", how="left").merge(nl, on="iso", how="left")
    fl["d_vs_old_sft_clean"] = fl.chrf_m385 - fl.old_sft_chrf_clean; fl["d_vs_old_sft_no_pair"] = fl.chrf_m385 - fl.old_sft_chrf_no_pair
    fl["d_vs_old_rl_clean"] = fl.chrf_m385 - fl.old_rl_chrf_clean
    fl["d_vs_nllb_3p3b"] = fl.chrf_m385 - fl.get("nllb_3p3b", np.nan)
    fl["weak_chrf"] = fl.chrf_m385 < WEAK_CHRF; fl["weak_finished"] = fl.finished_m385 < WEAK_FIN
    save(fl, a.out, "flores_per_language_385m_full.csv")
    # ---- 3a. 385M vs old 270M, macro on the common languages, per benchmark
    rows = []
    wa = w[w.subset == "all"]
    for b in BENCH_ORDER:
        x = wa[(wa.benchmark == b) & (wa.n_m385 >= 30)].merge(ow[ow.benchmark == b].drop(columns="benchmark"), on="iso", how="inner")
        if x.empty: continue
        for sub in ("all", "no_pair", "clean"):
            c_sft, c_rl = f"old_sft_chrf_{sub}", f"old_rl_chrf_{sub}"
            y = x.dropna(subset=[c_sft, c_rl])
            if y.empty: continue
            rows.append(dict(benchmark=BENCH_NAME[b], old_subset=sub, n_languages=len(y), chrf_m385_all_rows=y.chrf_m385.mean(), old_sft_chrf=y[c_sft].mean(), old_rl_chrf=y[c_rl].mean(),
                             d_m385_minus_old_sft=(y.chrf_m385 - y[c_sft]).mean(), d_m385_minus_old_rl=(y.chrf_m385 - y[c_rl]).mean(),
                             finished_m385=y.finished_m385.mean(), old_sft_finished=y[f"old_sft_finished_{sub}"].mean(), old_rl_finished=y[f"old_rl_finished_{sub}"].mean(),
                             n_languages_m385_beats_old_sft=int((y.chrf_m385 > y[c_sft]).sum())))
    t3 = pd.DataFrame(rows); save(t3, a.out, "vs_old270m_macro.csv")
    nar = lambda t, c385: pd.DataFrame({"benchmark": t.benchmark, "old rows": t.old_subset, "langs": t.n_languages, "chrF++ 385M": t[c385], "old SFT": t.old_sft_chrf, "old RL": t.old_rl_chrf,
                                        "385M - old SFT": t.d_m385_minus_old_sft, "385M - old RL": t.d_m385_minus_old_rl, "langs where 385M > old SFT": t.n_languages_m385_beats_old_sft,
                                        "finished 385M / SFT / RL": [f"{x:.2f} / {y:.2f} / {z:.2f}" for x, y, z in zip(t.finished_m385, t.old_sft_finished, t.old_rl_finished)]})
    md(nar(t3, "chrf_m385_all_rows"), "T3 385M (all its rows) vs old 270M rows (mixed sentence sets)")
    # per language table (all benchmarks, long)
    ol = wa.merge(ow, on=["benchmark", "iso"], how="inner")[["benchmark", "iso", "language", "group", "n_m385", "chrf_m385", "old_n_all", "old_n_clean", "old_sft_chrf_all", "old_sft_chrf_no_pair", "old_sft_chrf_clean", "old_rl_chrf_all", "old_rl_chrf_no_pair", "old_rl_chrf_clean"]]
    ol["d_m385_minus_old_sft_clean"] = ol.chrf_m385 - ol.old_sft_chrf_clean; ol["d_m385_minus_old_sft_all"] = ol.chrf_m385 - ol.old_sft_chrf_all
    save(ol, a.out, "vs_old270m_per_language.csv")
    # ------------------------------------------------------------------ 3b. like-for-like with the old 270M: the 385M scored on the old models' own clean / no_pair sentences
    if a.score_oldclean:
        oc = pd.read_csv(os.path.join(P(a.score_oldclean), "metrics_by_language.csv"))
        rows = []; per = []
        for sub in ("clean", "no_pair"):
            for b in BENCH_ORDER:
                x = oc[(oc.benchmark == b) & (oc["subset"] == sub) & (oc.system == "m385")][["iso", "n", "chrf", "finished"]].rename(columns={"n": "n_m385", "chrf": "chrf_m385_oldsubset", "finished": "finished_m385_oldsubset"})
                o = old[(old.benchmark == b) & (old["subset"] == sub)][["iso", "n", "sft_chrf", "rl_chrf", "sft_finished", "rl_finished"]].rename(columns={"n": "n_old"})
                y = x.merge(o, on="iso", how="inner"); y = y[y.n_m385 >= 30]
                if y.empty: continue
                y = y.assign(benchmark=BENCH_NAME[b], subset=sub, group=y.iso.map(groups)); per.append(y)
                rows.append(dict(benchmark=BENCH_NAME[b], old_subset=sub, n_languages=len(y), median_sentences_per_language=int(y.n_m385.median()), chrf_m385=y.chrf_m385_oldsubset.mean(), old_sft_chrf=y.sft_chrf.mean(), old_rl_chrf=y.rl_chrf.mean(),
                                 d_m385_minus_old_sft=(y.chrf_m385_oldsubset - y.sft_chrf).mean(), d_m385_minus_old_rl=(y.chrf_m385_oldsubset - y.rl_chrf).mean(), n_languages_m385_beats_old_sft=int((y.chrf_m385_oldsubset > y.sft_chrf).sum()),
                                 finished_m385=y.finished_m385_oldsubset.mean(), old_sft_finished=y.sft_finished.mean(), old_rl_finished=y.rl_finished.mean()))
        tt = pd.DataFrame(rows); save(tt, a.out, "vs_old270m_same_sentences_macro.csv")
        md(nar(tt, "chrf_m385").assign(sentences=tt.median_sentences_per_language.values).rename(columns={"sentences": "median sentences / language"}), "T3b 385M vs old 270M on the same sentences")
        pp = pd.concat(per); pp["d_m385_minus_old_sft"] = pp.chrf_m385_oldsubset - pp.sft_chrf; save(pp, a.out, "vs_old270m_same_sentences_per_language.csv")
    # ------------------------------------------------------------------ 4. groups
    g = mac[(mac.benchmark == "flores_plus") & (mac["subset"] == "all") & (mac.group != "all")].copy()
    gm = g[g.scope == "own"].pivot_table(index="group", columns="system", values=["n_languages", "chrf", "spbleu", "finished", "lid_ok", "repeat"], aggfunc="first"); gm.columns = [f"{v}_{s}" for v, s in gm.columns]; gm = gm.reset_index()
    gc = g[g.scope == "common"].pivot_table(index="group", columns="system", values=["n_languages", "chrf", "spbleu"], aggfunc="first"); gc.columns = [f"common_{v}_{s}" for v, s in gc.columns]; gc = gc.reset_index()
    gp = bp[(bp.benchmark == "flores_plus") & (bp["subset"] == "all") & (bp.group != "all")][["group", "n_languages", "d_chrf", "d_chrf_lo", "d_chrf_hi", "d_spbleu", "d_spbleu_lo", "d_spbleu_hi"]].rename(columns={"n_languages": "n_languages_paired"})
    gt = gm.merge(gc, on="group", how="left").merge(gp, on="group", how="left")
    gt["group"] = pd.Categorical(gt.group, GROUP_ORDER, ordered=True); gt = gt.sort_values("group"); gt["group"] = gt.group.astype(str)
    save(gt, a.out, "groups_flores_macro.csv")
    gv = gt[["group", "n_languages_m385", "chrf_m385", "finished_m385", "lid_ok_m385", "repeat_m385", "n_languages_paired", "common_chrf_m385", "common_chrf_a0", "d_chrf", "d_chrf_lo", "d_chrf_hi"]].copy()
    md(pd.DataFrame({"group": gv.group, "langs (385M)": gv.n_languages_m385.astype(int), "chrF++ 385M": gv.chrf_m385, "finished": gv.finished_m385, "LID ok": gv.lid_ok_m385, "repeat": gv.repeat_m385,
                     "langs paired with A0": gv.n_languages_paired.fillna(0).astype(int), "chrF++ 385M (paired set)": gv.common_chrf_m385, "chrF++ A0 (paired set)": gv.common_chrf_a0,
                     "delta [95% CI]": [f"{d:+.2f} [{l:+.2f}, {h:+.2f}]" if d == d else "-" for d, l, h in zip(gv.d_chrf, gv.d_chrf_lo, gv.d_chrf_hi)]}), "T4 groups, FLORES+ devtest")
    # groups for the other benchmarks (385M minus A0, paired)
    gb = bp[(bp["subset"] == "all") & (bp.group != "all")].copy(); gb["benchmark"] = gb.benchmark.map(BENCH_NAME)
    save(gb, a.out, "groups_all_benchmarks_paired_vs_a0.csv")
    # ------------------------------------------------------------------ 5. weak languages (any benchmark; primary = FLORES+ if present)
    prim = {}
    ma = met[met["subset"] == "all"]
    for iso, d in ma.groupby("iso"):
        d = d[d.n >= 30]
        for b in BENCH_ORDER:
            if b in set(d.benchmark): prim[iso] = b; break
    rows = []
    for iso, b in sorted(prim.items()):
        d = ma[(ma.iso == iso) & (ma.benchmark == b) & (ma.system == "m385")].iloc[0]
        da = ma[(ma.iso == iso) & (ma.benchmark == b) & (ma.system == "a0")]
        allb = ma[(ma.iso == iso) & (ma.system == "m385") & (ma.n >= 30)]
        rows.append(dict(iso=iso, language=names.get(iso), group=groups.get(iso, "ungrouped"), primary_benchmark=BENCH_NAME[b], n=int(d.n), chrf=d.chrf, finished=d["finished"], empty=d["empty"], lid_ok=d["lid_ok"], repeat=d["repeat"], copy=d["copy"],
                         a0_chrf=(da.chrf.iloc[0] if len(da) else np.nan), a0_finished=(da.finished.iloc[0] if len(da) else np.nan),
                         weak_chrf=d.chrf < WEAK_CHRF, weak_finished=d.finished < WEAK_FIN, benchmarks_available=",".join(sorted(set(allb.benchmark))),
                         min_chrf_any_benchmark=allb.chrf.min(), min_finished_any_benchmark=allb.finished.min()))
    wk = pd.DataFrame(rows); wk["weak"] = wk.weak_chrf | wk.weak_finished
    wk["reason"] = np.where(wk.weak_chrf & wk.weak_finished, "chrF++ < 14 and finished < 0.9", np.where(wk.weak_chrf, "chrF++ < 14", np.where(wk.weak_finished, "finished < 0.9", "")))
    save(wk.sort_values(["weak", "chrf"], ascending=[False, True]), a.out, "weak_languages.csv")
    ww = wk[wk.weak].sort_values("chrf")
    md(pd.DataFrame({"language": ww.iso + " " + ww.language, "group": ww.group, "benchmark": ww.primary_benchmark, "chrF++": ww.chrf, "finished": ww.finished, "LID ok": ww.lid_ok, "repeat": ww["repeat"], "A0 chrF++": ww.a0_chrf, "A0 finished": ww.a0_finished, "why": ww.reason}), "T5 weak languages")
    # all weak cells, any benchmark
    cell = ma[(ma.system == "m385") & (ma.n >= 30) & ((ma.chrf < WEAK_CHRF) | (ma.finished < WEAK_FIN))].copy(); cell["benchmark"] = cell.benchmark.map(BENCH_NAME); cell["language"] = cell.iso.map(names)
    save(cell[["benchmark", "iso", "language", "n", "chrf", "finished", "lid_ok", "repeat"]].sort_values(["benchmark", "chrf"]), a.out, "weak_cells_any_benchmark.csv")
    # ------------------------------------------------------------------ 6. NLLB (comparable) and indicative rows on the FLORES+ language set
    flx = fl.copy()
    rows = []
    for nm, col in (("NLLB-200 3.3B", "nllb_3p3b"), ("NLLB-200 1.3B", "nllb_1p3b"), ("NLLB-200 distilled 1.3B", "nllb_distilled_1p3b"), ("NLLB-200 MoE 54.5B", "nllb_moe_54p5b")):
        if col not in flx.columns: continue
        x = flx.dropna(subset=[col, "chrf_m385"]); y = x.dropna(subset=["chrf_a0"])
        rows.append(dict(system=nm, n_languages=len(x), chrf=x[col].mean(), m385_chrf=x.chrf_m385.mean(), d_m385_minus_system=(x.chrf_m385 - x[col]).mean(), n_m385_beats=int((x.chrf_m385 > x[col]).sum()),
                         n_languages_with_a0=len(y), a0_chrf_same_languages=y.chrf_a0.mean(), old_sft_clean_chrf=x.old_sft_chrf_clean.mean(), old_rl_clean_chrf=x.old_rl_chrf_clean.mean()))
    save(pd.DataFrame(rows), a.out, "published_nllb_flores_comparison.csv", "T6 NLLB comparison")
    save(flx[["iso", "language", "group", "chrf_m385", "chrf_a0", "old_sft_chrf_clean", "old_rl_chrf_clean", "nllb_3p3b", "nllb_1p3b", "nllb_distilled_1p3b", "nllb_moe_54p5b", "d_vs_nllb_3p3b"]].dropna(subset=["nllb_3p3b"]), a.out, "published_nllb_flores_per_language.csv")
    # indicative: AfriqueLLM family (5-shot base, AfroBench-Lite harness)
    af = pub[(pub.direction == "en->xx") & pub.benchmark.str.contains("AfroBench-Lite")].pivot_table(index="iso", columns="model", values="chrF++", aggfunc="first").reset_index()
    xa = flx[["iso", "chrf_m385"]].merge(af, on="iso", how="inner")
    rows = [dict(system=c, n_languages=len(xa.dropna(subset=[c])), chrf=xa[c].mean(), m385_chrf_same_languages=xa.dropna(subset=[c]).chrf_m385.mean()) for c in af.columns if c != "iso"]
    save(pd.DataFrame(rows).sort_values("chrf", ascending=False), a.out, "published_afrique_indicative.csv")
    # ------------------------------------------------------------------ 7. AfriScience sentence level
    def afs(name, split):
        d = json.load(open(P(f"out_afsci_{name}", "afriscience_eval.json")))["splits"][split]["sentence"]
        return d
    pubaf = pd.read_csv(os.path.join(PLAN, "afriscience_mt_published_chrf.csv")).set_index(["system", "level"])
    rows = []
    for split in ("test", "dev"):
        for nm in ("m385", "a0"):
            d = afs(nm, split); r = {"system": {"m385": "385M SFT (this run)", "a0": "A0 270M (this run)"}[nm], "split": split, "metric": "chrF (word order 0)"}
            for l in ("amh", "hau", "lug", "nso", "yor", "zul"): r[l] = d["per_language"][l]["chrf"]
            r["macro"] = d["macro"]["chrf"]; r["finished_macro"] = d["macro"]["finished_rate"]; r["chrf_pp_macro"] = d["macro"]["chrf_pp"]; r["spbleu_macro"] = d["macro"]["spbleu"]; rows.append(r)
    for (sy, lv), r in pubaf.iterrows():
        if lv.endswith("(sentence)"): rows.append({"system": sy + (" (old 270M)" if lv.startswith("ours") else " (published)"), "split": "test", "metric": "chrF (word order 0)", **{l: r[l] for l in ("amh", "hau", "lug", "nso", "yor", "zul")}, "macro": r["macro"]})
    t7 = pd.DataFrame(rows); save(t7, a.out, "afriscience_sentence.csv", "T7 AfriScience-MT sentence level")
    # ------------------------------------------------------------------ 8. decoding
    for tag in ("study20", "beam16"):
        f = P("decode_an", f"{tag}_macro.csv")
        if os.path.exists(f):
            m_ = pd.read_csv(f).drop(columns=["config"]); save(m_, a.out, f"decoding_{tag}_macro.csv")
            save(pd.read_csv(P("decode_an", f"{tag}_per_language.csv")), a.out, f"decoding_{tag}_per_language.csv")
    # ------------------------------------------------------------------ 9. optional clean tables
    if a.score_clean:
        sc2 = P(a.score_clean)
        m2 = pd.read_csv(os.path.join(sc2, "macro.csv")); b2 = pd.read_csv(os.path.join(sc2, "bootstrap_pairs.csv"))
        x = m2[(m2.group == "all") & (m2.scope == "common")].copy(); x["benchmark"] = x.benchmark.map(BENCH_NAME)
        save(x, a.out, "clean_macro_common.csv"); b2 = b2[b2.group == "all"].copy(); b2["benchmark"] = b2.benchmark.map(BENCH_NAME); save(b2, a.out, "clean_paired_385m_vs_a0.csv")
        save(pd.read_csv(os.path.join(sc2, "metrics_by_language.csv")), a.out, "clean_metrics_by_language.csv")
        if a.comet_clean and os.path.exists(P(a.comet_clean, "comet_macro.csv")): save(pd.read_csv(P(a.comet_clean, "comet_macro.csv")), a.out, "clean_comet_macro.csv")
    # ------------------------------------------------------------------ 10. compact formatted tables for the report
    ci = lambda d, l, h: f"{d:+.2f} [{l:+.2f}, {h:+.2f}]" if d == d else "-"
    if os.path.exists(P("decode_an", "study20_macro.csv")):
        dm = pd.read_csv(P("decode_an", "study20_macro.csv")); nm = {"greedy": "greedy", "beam4": "beam 4", "mbr16": "MBR-16 (+ greedy as 17th candidate)"}
        md(pd.DataFrame({"decoding": dm.system.map(nm), "chrF++ macro": dm.chrf_macro, "delta chrF++ vs greedy [95% CI]": [ci(*x) if x[0] == x[0] else "-" for x in zip(dm.d_chrf_vs_greedy, dm.lo, dm.hi)],
                         "spBLEU macro": dm.spbleu_macro, "delta spBLEU [95% CI]": [ci(*x) if x[0] == x[0] else "-" for x in zip(dm.d_spbleu_vs_greedy, dm.slo, dm.shi)],
                         "truncated at 256 tokens (ESTIMATED)": dm.trunc_est, "repetitive": dm.repeat}), "T8 decoding study (20 languages x 200 FLORES+ devtest sentences)")
    if os.path.exists(P("decode_an", "beam16_macro.csv")):
        dm = pd.read_csv(P("decode_an", "beam16_macro.csv"))
        md(pd.DataFrame({"decoding": dm.system, "chrF++ macro (16 languages x 100 sentences)": dm.chrf_macro, "delta [95% CI]": [ci(*x) if x[0] == x[0] else "-" for x in zip(dm.d_chrf_vs_greedy, dm.lo, dm.hi)], "spBLEU": dm.spbleu_macro}), "T8b beam parts")
    if os.path.exists(P("decode_an", "study20_per_language.csv")):
        d = pd.read_csv(P("decode_an", "study20_per_language.csv"))
        wd = d.pivot(index="language", columns="system", values=["chrf", "d_chrf_vs_greedy", "lo", "hi", "trunc_est"])
        o = pd.DataFrame({"language": wd.index, "greedy": wd[("chrf", "greedy")].values, "beam 4": wd[("chrf", "beam4")].values, "beam delta [95% CI]": [ci(*x) for x in zip(wd[("d_chrf_vs_greedy", "beam4")], wd[("lo", "beam4")], wd[("hi", "beam4")])],
                          "MBR-16": wd[("chrf", "mbr16")].values, "MBR delta [95% CI]": [ci(*x) for x in zip(wd[("d_chrf_vs_greedy", "mbr16")], wd[("lo", "mbr16")], wd[("hi", "mbr16")])], "greedy truncated (est.)": wd[("trunc_est", "greedy")].values})
        md(o.sort_values("greedy"), "T8c decoding per language")
    c = fl.copy(); c["old_sft_same"] = np.nan
    ps = pd.read_csv(os.path.join(a.out, "vs_old270m_same_sentences_per_language.csv")) if os.path.exists(os.path.join(a.out, "vs_old270m_same_sentences_per_language.csv")) else None
    if ps is not None:
        q = ps[(ps.benchmark == "FLORES+ devtest") & (ps["subset"] == "clean")][["iso", "chrf_m385_oldsubset", "sft_chrf", "rl_chrf"]]
        c = c.drop(columns="old_sft_same").merge(q, on="iso", how="left")
    tab = pd.DataFrame({"iso": c.iso, "group": c.group, "chrF++ 385M": c.chrf_m385, "chrF++ A0": c.chrf_a0, "delta vs A0 [95% CI]": [ci(*x) if x[0] == x[0] else "-" for x in zip(c.d_chrf, c.d_chrf_lo, c.d_chrf_hi)],
                        "AfriCOMET 385M": c.comet_m385, "finished": c.finished_m385, "LID ok": c.lid_ok_m385, "repeat": c["repeat_m385"],
                        "385M / old SFT / old RL on the 48 old-unseen sentences": [f"{x:.1f} / {y:.1f} / {z:.1f}" if x == x and y == y else "-" for x, y, z in zip(c.get("chrf_m385_oldsubset", np.nan), c.get("sft_chrf", np.nan), c.get("rl_chrf", np.nan))],
                        "NLLB-3.3B": c.nllb_3p3b})
    tab["_g"] = pd.Categorical(c.group, GROUP_ORDER, ordered=True); tab = tab.sort_values(["_g", "chrF++ 385M"], ascending=[True, False]).drop(columns="_g")
    md(tab, "T10 FLORES+ devtest per language")
    open(os.path.join(a.out, "_fragments.md"), "w").write("\n".join(FRAG)); print("fragments written")


if __name__ == "__main__":
    main()
