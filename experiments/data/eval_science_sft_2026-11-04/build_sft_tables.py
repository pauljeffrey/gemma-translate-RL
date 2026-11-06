#!/usr/bin/env python3
"""Tables of reports/eval_science_sft_385m_2026-11-04.md (science SFT, arm 2).

    python build_sft_tables.py --dir FILES_DIR > tables.md
FILES_DIR holds: choice_alpha.json, start_choice.json, afsci_a<alpha>.json (afriscience_eval.json per alpha), results_a<alpha>_x.csv (NTREX / MAFAND results.csv per alpha),
results_chosen_comet_x.csv (the chosen alpha's NTREX / MAFAND with AfriCOMET), sft_curve.txt (grep of the trainer log).
Old-270M columns: AfriScience test (stored: RL-final 1,433 / science-stage start A2W / final2 step 401, plain chrF; published fine-tuned systems) - FAIR from this stage on, both have had a science stage;
NTREX / MAFAND clean rows of the old SFT / RL (own_270m_per_language.csv).
"""
import argparse, csv, json, os, statistics as st

OLD_AFS = {"old 270M RL-final (1,433)": [25.3, 59.4, 46.2, 55.9, 37.4, 63.4], "old 270M science-stage start (A2W)": [26.2, 62.0, 48.9, 58.3, 39.8, 64.7],
           "old 270M final2 (step 401)": [26.6, 61.9, 49.3, 58.5, 40.1, 64.7], "NLLB-200-1.3B fine-tuned (published)": [45.5, 67.7, 51.5, 64.3, 17.7, 67.2],
           "NLLB-200-distilled-600M fine-tuned (published)": [44.4, 66.9, 50.6, 64.2, 17.6, 66.3], "M2M100-1.2B fine-tuned (published)": [39.0, 63.1, 49.5, 62.4, 16.6, 58.4]}
LANGS = ["amh", "hau", "lug", "nso", "yor", "zul"]
OLD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval_385m_plan_2026-11-02", "own_270m_per_language.csv")


def rd(p):
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); a = ap.parse_args(); D = a.dir
    ch = json.load(open(f"{D}/choice_alpha.json")); sc = json.load(open(f"{D}/start_choice.json")); alphas = list(ch["rows"].keys()); best = str(ch["chosen_alpha"])
    out = []; P = out.append
    P(f"Start checkpoint: **{sc['chosen']}** ({sc['reason']}); `{sc['model_dir']}`. Candidates: " + "; ".join(f"{n}: dev chrF++ {v['common_chrf']:.2f}, AfriCOMET {v['common_comet'] if v['common_comet'] is None else round(v['common_comet'], 4)} (step {v['step']})" for n, v in sc["candidates"].items()) + "\n")
    P("## Table 1. Interpolation sweep: theta(alpha) = (1 - alpha) theta_start + alpha theta_science-SFT (alpha 0 = the start checkpoint)\n")
    P("| alpha | AfriScience dev chrF (macro 6) | AfriScience test chrF | NTREX macro chrF++ | d vs alpha 0 | MAFAND macro chrF++ | d vs alpha 0 | passes non-regression (>= start - 0.5) |")
    P("|---:|---:|---:|---:|---:|---:|---:|---|")
    base = ch["rows"][alphas[0]]
    for al in alphas:
        r = ch["rows"][al]
        af = json.load(open(f"{D}/afsci_a{al}.json"))["splits"]["test"]["sentence"]["macro"]["chrf"] if os.path.exists(f"{D}/afsci_a{al}.json") else float("nan")
        ok = r["ntrex"] >= base["ntrex"] - 0.5 and r["mafand"] >= base["mafand"] - 0.5
        P(f"| {al}{' (chosen)' if al == best else ''} | {r['afsci_dev_chrf']:.2f} | {af:.2f} | {r['ntrex']:.2f} | {r['ntrex'] - base['ntrex']:+.2f} | {r['mafand']:.2f} | {r['mafand'] - base['mafand']:+.2f} | {'yes' if ok else 'no'} |")
    P(f"\nChosen alpha: **{best}** (rule: best AfriScience dev chrF subject to NTREX and MAFAND not below the start by more than 0.5).\n")
    P("## Table 2. AfriScience-MT test, sentence level, plain chrF (word order 0), greedy: now a FAIR comparison with the old 270M (both have had a science stage)\n")
    P("| system | amh | hau | lug | nso | yor | zul | macro |")
    P("|---|---:|---:|---:|---:|---:|---:|---:|")
    for al in (alphas[0], best):
        p = f"{D}/afsci_a{al}.json"
        if not os.path.exists(p): continue
        pl = json.load(open(p))["splits"]["test"]["sentence"]["per_language"]; v = [pl[l]["chrf"] for l in LANGS]
        P(f"| 385M alpha {al}{' (start, no science SFT)' if al == alphas[0] else ' (chosen)'} | " + " | ".join(f"{x:.1f}" for x in v) + f" | {st.mean(v):.1f} |")
    for n, v in OLD_AFS.items():
        P(f"| {n} | " + " | ".join(f"{x:.1f}" for x in v) + f" | {st.mean(v):.1f} |")
    P("")
    P("## Table 3. NTREX / MAFAND non-regression on the clean rows of the old models (chrF++, languages both cover)\n")
    old = {}
    for r in rd(OLD):
        if r["subset"] == "clean": old[(r["benchmark"], r["iso"])] = r
    P("| benchmark | languages | old SFT (clean) | old RL (clean) | 385M start (alpha 0) | 385M chosen alpha |")
    P("|---|---:|---:|---:|---:|---:|")
    for b in ("ntrex", "mafand"):
        res = {al: {r["iso"]: float(r["chrf"]) for r in rd(f"{D}/results_a{al}_x.csv") if r["benchmark"] == b and r["subset"] == "all" and r["chrf"]} for al in (alphas[0], best)}
        langs = [i for i in res[alphas[0]] if (b, i) in old and i in res[best]]
        if not langs: continue
        P(f"| {b.upper()} | {len(langs)} | {st.mean(float(old[(b, i)]['sft_chrf']) for i in langs):.1f} | {st.mean(float(old[(b, i)]['rl_chrf']) for i in langs):.1f} | {st.mean(res[alphas[0]][i] for i in langs):.1f} | {st.mean(res[best][i] for i in langs):.1f} |")
    P("")
    cm = [r for r in rd(f"{D}/results_chosen_comet_x.csv") if r["subset"] == "all" and r["comet"]]
    if cm:
        P("Held-out AfriCOMET of the chosen model: " + ", ".join(f"{b} {st.mean(float(r['comet']) for r in cm if r['benchmark'] == b):.3f} ({sum(1 for r in cm if r['benchmark'] == b)} languages)" for b in ("ntrex", "mafand") if any(r['benchmark'] == b for r in cm)) + ".\n")
    if os.path.exists(f"{D}/sft_curve.txt"):
        P("## SFT validation curve (AfriScience dev loss = early-stopping split; replay loss = forgetting monitor)\n")
        P("```\n" + open(f"{D}/sft_curve.txt").read().strip() + "\n```\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
