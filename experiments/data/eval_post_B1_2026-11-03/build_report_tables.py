#!/usr/bin/env python3
"""Builds the tables of reports/eval_post_B1_385m_2026-11-03.md from the files next to it.

    python build_report_tables.py > tables.md

Inputs (this directory): weak/ and family/ (eval_385m_score.py outputs: macro.csv, bootstrap_pairs.csv, bootstrap_by_language.csv), results/results_{sft,stageA,b1best}.csv
(+ results_{stageA,b1best}_x.csv for NTREX / MAFAND when present), and the old 270M per-language table ../eval_385m_plan_2026-11-02/own_270m_per_language.csv.
Systems: sft = 385M SFT-final (like the earlier evaluation), stageA = stage A final (step 400), b1best = stage B1 best (step 50, dev chrF++ 40.74).
"""
import csv, os, statistics as st, sys
H = os.path.dirname(os.path.abspath(__file__))
OLD = os.path.join(H, "..", "eval_385m_plan_2026-11-02", "own_270m_per_language.csv")
SYS = ["sft", "stageA", "b1best"]
NAME = {"sft": "385M SFT", "stageA": "stage A", "b1best": "B1 best"}


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def fm(x, spec):
    v = f(x)
    return '-' if v is None else format(v, spec)


def rd(p):
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def results(name, extra=False):
    out = {}
    for fn in ([f"results_{name}.csv"] + ([f"results_{name}_x.csv"] if extra else [])):
        for r in rd(os.path.join(H, "results", fn)):
            if r["subset"] == "all":
                out[(r["benchmark"], r["iso"])] = r
    return out


R = {s: results(s, extra=True) for s in SYS}
mean = lambda v: sum(v) / len(v) if v else float("nan")
bl = {}
for r in rd(os.path.join(H, "weak", "bootstrap_by_language.csv")):
    if r["subset"] == "all":
        bl[(r["pair"], r["benchmark"], r["iso"])] = r
bp = {}
for g in ("weak", "family"):
    for r in rd(os.path.join(H, g, "bootstrap_pairs.csv")):
        if r["subset"] == "all":
            bp[(r["pair"], r["benchmark"], r["group"])] = r
mac = {}
for g in ("weak", "family"):
    for r in rd(os.path.join(H, g, "macro.csv")):
        if r["subset"] == "all" and r["scope"] == "own":
            mac[(r["system"], r["benchmark"], r["group"])] = r
old = {}
for r in rd(OLD):
    old[(r["benchmark"], r["iso"], r["subset"])] = r


def fl_isos():
    return sorted({k[1] for k in R["b1best"] if k[0] == "flores_plus"})


def sign(r):
    if r is None:
        return "?"
    lo, hi = f(r["d_chrf_lo"]), f(r["d_chrf_hi"])
    return "+" if lo > 0 else "-" if hi < 0 else "="


out = []
P = out.append
# --- 1. FLORES+ macro
P("## Table 1. FLORES+ devtest, English to X, greedy, 54 languages (all rows)\n")
P("| system | chrF++ | spBLEU | finished | AfriCOMET (languages) |")
P("|---|---:|---:|---:|---:|")
for s in SYS:
    m = mac.get((s, "flores_plus", "all"))
    cs = [f(r["comet"]) for k, r in R[s].items() if k[0] == "flores_plus" and f(r["comet"]) is not None]
    P(f"| {NAME[s]} | {f(m['chrf']):.2f} | {f(m['spbleu']):.2f} | {f(m['finished']):.3f} | {mean(cs):.3f} ({len(cs)}) |")
P("")
P("Paired differences (macro over languages, 95% paired sentence-bootstrap, 1,000 resamples):\n")
P("| pair | group | languages | d chrF++ [95% CI] | d spBLEU [95% CI] |")
P("|---|---|---:|---:|---:|")
for pair in ("stageA:sft", "b1best:sft", "b1best:stageA"):
    for g in ("all", "other", "weak (x2.5 RL list)"):
        r = bp.get((pair, "flores_plus", g))
        if r:
            P(f"| {pair.replace(':', ' - ')} | {g} | {r['n_languages']} | {f(r['d_chrf']):+.2f} [{f(r['d_chrf_lo']):+.2f}, {f(r['d_chrf_hi']):+.2f}] | {f(r['d_spbleu']):+.2f} [{f(r['d_spbleu_lo']):+.2f}, {f(r['d_spbleu_hi']):+.2f}] |")
P("")
P("Group means (chrF++): weak = the 20 FLORES languages of the x2.5 RL list (ach amh bam bin dyu efi ewe fon ish kab kam kbp kea kmb kon luo mos nus nyn pcm seh shi tir tiv tum umb vmw wol yor that FLORES+ covers), other = the remaining 34.\n")
P("| group | n | " + " | ".join(NAME[s] for s in SYS) + " |")
P("|---|---:|" + "---:|" * len(SYS))
for g in ("all", "other", "weak (x2.5 RL list)"):
    row = [mac.get((s, "flores_plus", g)) for s in SYS]
    P(f"| {g} | {row[0]['n_languages']} | " + " | ".join(f"{f(r['chrf']):.2f}" for r in row) + " |")
P("")
# --- 2. language families
P("## Table 2. By language family (chrF++ macro, FLORES+ devtest)\n")
P("| family | n | " + " | ".join(NAME[s] for s in SYS) + " | B1 - SFT [CI] |")
P("|---|---:|" + "---:|" * len(SYS) + "---:|")
fams = sorted({k[2] for k in mac if k[1] == "flores_plus" and k[0] == "sft" and k[2] not in ("all", "other", "weak (x2.5 RL list)")})
for g in fams:
    row = [mac.get((s, "flores_plus", g)) for s in SYS]
    if None in row:
        continue
    r = bp.get(("b1best:sft", "flores_plus", g))
    ci = f"{f(r['d_chrf']):+.2f} [{f(r['d_chrf_lo']):+.2f}, {f(r['d_chrf_hi']):+.2f}]" if r else ""
    P(f"| {g} | {row[0]['n_languages']} | " + " | ".join(f"{f(r_['chrf']):.2f}" for r_ in row) + f" | {ci} |")
P("")
# --- 3. per language
P("## Table 3. Per language, FLORES+ devtest (chrF++; AfriCOMET where the held-out model covers the language)\n")
P("old 270M columns: the first research version (contaminated: FLORES+ English was in its SFT text, see the caveats), all rows, same harness; 'n/a' = language not covered by it.\n")
P("| iso | language | SFT | A | B1 | B1-SFT [CI] | B1-A [CI] | COMET SFT / A / B1 | old SFT | old RL | finished B1 |")
P("|---|---|---:|---:|---:|---|---|---|---:|---:|---:|")
cnt = {"stageA:sft": [0, 0, 0], "b1best:sft": [0, 0, 0], "b1best:stageA": [0, 0, 0]}
rows = []
for iso in fl_isos():
    g = {s: R[s].get(("flores_plus", iso)) for s in SYS}
    if any(v is None for v in g.values()):
        continue
    d1 = bl.get(("b1best:sft", "flores_plus", iso)); d2 = bl.get(("b1best:stageA", "flores_plus", iso))
    for pr in cnt:
        r = bl.get((pr, "flores_plus", iso))
        if r:
            cnt[pr][{"+": 0, "=": 1, "-": 2}[sign(r)]] += 1
    o = old.get(("flores_plus", iso, "all"))
    cm = " / ".join((f"{f(g[s]['comet']):.3f}" if f(g[s]["comet"]) is not None else "-") for s in SYS)
    ci = lambda r: f"{f(r['d_chrf']):+.1f} [{f(r['d_chrf_lo']):+.1f}, {f(r['d_chrf_hi']):+.1f}]" if r else ""
    rows.append((f(g["b1best"]["chrf"]) - f(g["sft"]["chrf"]),
                 f"| {iso} | {g['b1best']['language']} | {f(g['sft']['chrf']):.1f} | {f(g['stageA']['chrf']):.1f} | {f(g['b1best']['chrf']):.1f} | {ci(d1)} | {ci(d2)} | {cm} | "
                 + (f"{f(o['sft_chrf']):.1f} | {f(o['rl_chrf']):.1f}" if o else "n/a | n/a") + f" | {fm(g['b1best']['finished_rate'],'.2f')} |"))
for _, line in sorted(rows, key=lambda x: -x[0]):
    P(line)
P("")
P("Wins / ties / losses over the 54 languages (95% paired-bootstrap CI of the per-language chrF++ difference excludes 0 = win or loss):\n")
P("| pair | better | no clear difference | worse |")
P("|---|---:|---:|---:|")
for pr, c in cnt.items():
    P(f"| {pr.replace(':', ' - ')} | {c[0]} | {c[1]} | {c[2]} |")
P("")
# --- 4. old research version
P("## Table 4. First research version (old 270M SFT and RL) vs the 385M systems, same languages, greedy\n")
P("Old numbers come from the earlier harness runs (`own_270m_per_language.csv`; SFT-final step 27,638 and RL-final GRPO step 1,433). They saw FLORES+ English in training (95% of the devtest English in the SFT text) and NTREX/MAFAND/SmolSent partly (see `reports/eval_385m_results_2026-11-03.md` section 4); `clean` = rows whose English and pair were not in the old SFT text. The 385M systems are the clean rerun (protected index); their rows are all rows (contamination audit of the new SFT text pending). Lead with the clean-subset columns of NTREX/MAFAND; FLORES is marked.\n")
P("| benchmark | subset of old rows | languages | old SFT | old RL | 385M SFT | stage A | B1 best |")
P("|---|---|---:|---:|---:|---:|---:|---:|")
for bench, label in (("ntrex", "NTREX"), ("mafand", "MAFAND"), ("flores_plus", "FLORES+ (old models saw it)")):
    for sub in ("clean", "no_pair", "all"):
        ks = [iso for (b, iso, s_) in old if b == bench and s_ == sub and all((bench, iso) in R[s] for s in SYS) and f(old[(b, iso, s_)]["sft_chrf"]) is not None]
        if not ks:
            continue
        v = lambda s: mean([f(R[s][(bench, i)]["chrf"]) for i in ks])
        P(f"| {label} | {sub} | {len(ks)} | {mean([f(old[(bench, i, sub)]['sft_chrf']) for i in ks]):.1f} | {mean([f(old[(bench, i, sub)]['rl_chrf']) for i in ks]):.1f} | {v('sft'):.1f} | {v('stageA'):.1f} | {v('b1best'):.1f} |")
P("")
sys.stdout.write("\n".join(out) + "\n")
