#!/usr/bin/env python3
"""Tables of reports/eval_weak_group_385m_2026-11-03.md (owner request 2026-11-03: does the weak West-African group run help?).

    python build_weak_tables.py --scores DIR --results RESULTS_DIR [--start mixedbest --best weakbest --final weakfinal --ref b1best] [--run RUN_DIR_OF_THE_WEAK_RUN] > tables.md

--scores   output of eval_385m_score.py (macro.csv, bootstrap_pairs.csv, bootstrap_by_language.csv) with groups target8 / other, pairs `best:start`, `final:start`
--results  directory with results_<system>.csv (the results.csv of each system's eval dir; per language chrf, spbleu, comet, finished_rate)
Pre-registered readout (owner): mean chrF++ and AfriCOMET change on the 8 target languages (mos fon kbp bam wol ewe twi dyu) vs the mixed best (the weak run's start), paired bootstrap CIs over sentences
(chrF++) and over languages (AfriCOMET: only language means are kept, 95% bootstrap over the languages that have a score), per-language wins / losses; non-regression: mean change on the other languages and
the number of other languages worse than -0.5. VERDICT 'helps' only if: target mean gain >= +1.0 chrF++ with the CI excluding 0, AfriCOMET (target mean change) not worse than -0.005, other-language mean change >= -0.3.
"""
import argparse, csv, json, os, random, statistics as st

T8 = "mos fon kbp bam wol ewe twi dyu".split()
OLD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval_385m_plan_2026-11-02", "own_270m_per_language.csv")


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def rd(p):
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scores", required=True); ap.add_argument("--results", required=True)
    ap.add_argument("--start", default="mixedbest"); ap.add_argument("--best", default="weakbest"); ap.add_argument("--final", default="weakfinal"); ap.add_argument("--ref", default="b1best")
    ap.add_argument("--systems", nargs="*", default=None); ap.add_argument("--run", default=None); a = ap.parse_args()
    systems = a.systems or ["sft", "stageA", a.ref, a.start, a.best, a.final]
    R = {s: {} for s in systems}
    for s in systems:
        for r in rd(os.path.join(a.results, f"results_{s}.csv")):
            if r["subset"] == "all" and r["chrf"]: R[s][(r["benchmark"], r["iso"])] = r
    mac = {(r["system"], r["benchmark"], r["group"]): r for r in rd(os.path.join(a.scores, "macro.csv")) if r["subset"] == "all" and r["scope"] == "own"}
    bp = {(r["pair"], r["benchmark"], r["group"]): r for r in rd(os.path.join(a.scores, "bootstrap_pairs.csv")) if r["subset"] == "all"}
    bl = {(r["pair"], r["benchmark"], r["iso"]): r for r in rd(os.path.join(a.scores, "bootstrap_by_language.csv")) if r["subset"] == "all"}
    out = []; P = out.append
    isos = sorted({k[1] for k in R[a.start] if k[0] == "flores_plus"})
    other = [i for i in isos if i not in T8]
    # ---- 1 verdict inputs
    def boot_mean_ci(vals, n=2000, seed=1):
        rng = random.Random(seed); m = [st.mean(rng.choices(vals, k=len(vals))) for _ in range(n)]; m.sort()
        return st.mean(vals), m[int(0.025 * n)], m[int(0.975 * n)]
    P("## Table 1. Weak-group run versus its start point (mixed best), FLORES+ devtest, greedy\n")
    P("| model vs start | group | languages | d chrF++ [95% CI, sentence bootstrap] | d spBLEU | d AfriCOMET (mean over languages with a score) [95% CI over languages] |")
    P("|---|---|---:|---|---:|---|")
    verdicts = {}
    for sysname in dict.fromkeys((a.best, a.final)):
        pair = f"{sysname}:{a.start}"
        for g, langs in (("target8", [i for i in T8 if i in isos]), ("other", other)):
            r = bp.get((pair, "flores_plus", g))
            dc = [(f(R[sysname][("flores_plus", i)]["comet"]) - f(R[a.start][("flores_plus", i)]["comet"])) for i in langs
                  if f(R[sysname].get(("flores_plus", i), {}).get("comet")) is not None and f(R[a.start].get(("flores_plus", i), {}).get("comet")) is not None]
            ci = boot_mean_ci(dc) if len(dc) >= 3 else (st.mean(dc) if dc else None, None, None)
            cs = f"{ci[0]:+.4f} [{ci[1]:+.4f}, {ci[2]:+.4f}] (n={len(dc)})" if ci[1] is not None else (f"{ci[0]:+.4f} (n={len(dc)})" if dc else "-")
            if r:
                P(f"| {sysname} - {a.start} | {g} | {r['n_languages']} | {f(r['d_chrf']):+.2f} [{f(r['d_chrf_lo']):+.2f}, {f(r['d_chrf_hi']):+.2f}] | {f(r['d_spbleu']):+.2f} | {cs} |")
                verdicts[(sysname, g)] = (f(r["d_chrf"]), f(r["d_chrf_lo"]), f(r["d_chrf_hi"]), ci[0])
    P("")
    # ---- 2 per language table of the target languages
    P("## Table 2. The 8 target languages, FLORES+ devtest (chrF++ / AfriCOMET)\n")
    P("| iso | start (mixed best) chrF++ | weak best | weak final | best - start [CI] | final - start [CI] | COMET start / best / final | old 270M SFT / RL (stored, FLORES contaminated) |")
    P("|---|---:|---:|---:|---|---|---|---|")
    old = {r["iso"]: r for r in rd(OLD) if r["benchmark"] == "flores_plus" and r["subset"] == "all"}
    wins = {m: [0, 0, 0] for m in dict.fromkeys((a.best, a.final))}
    for i in T8:
        k = ("flores_plus", i)
        if any(k not in R[s] for s in (a.start, a.best, a.final)): continue
        row = [R[s][k] for s in (a.start, a.best, a.final)]
        ci = lambda sysname: (lambda r: (f"{f(r['d_chrf']):+.1f} [{f(r['d_chrf_lo']):+.1f}, {f(r['d_chrf_hi']):+.1f}]" if r else ""))(bl.get((f"{sysname}:{a.start}", "flores_plus", i)))
        for sysname in dict.fromkeys((a.best, a.final)):
            r = bl.get((f"{sysname}:{a.start}", "flores_plus", i))
            if r: wins[sysname][0 if f(r["d_chrf_lo"]) > 0 else 2 if f(r["d_chrf_hi"]) < 0 else 1] += 1
        cm = " / ".join((f"{f(r['comet']):.3f}" if f(r["comet"]) is not None else "-") for r in row)
        o = old.get(i)
        P(f"| {i} | {f(row[0]['chrf']):.1f} | {f(row[1]['chrf']):.1f} | {f(row[2]['chrf']):.1f} | {ci(a.best)} | {ci(a.final)} | {cm} | " + (f"{f(o['sft_chrf']):.1f} / {f(o['rl_chrf']):.1f}" if o else "n/a") + " |")
    P("")
    P("Per-language wins / ties / losses on the 8 targets (CI of the chrF++ difference excludes 0): " + "; ".join(f"{s}: {w[0]} better, {w[1]} no clear difference, {w[2]} worse" for s, w in wins.items()) + "\n")
    # ---- 3 non-regression on the other languages
    P("## Table 3. Non-regression on the other languages (FLORES+, chrF++, per-language difference to the start)\n")
    P("| model | languages | mean change | languages worse than -0.5 | worst 5 |")
    P("|---|---:|---:|---:|---|")
    for sysname in dict.fromkeys((a.best, a.final)):
        d = [(f(R[sysname][("flores_plus", i)]["chrf"]) - f(R[a.start][("flores_plus", i)]["chrf"]), i) for i in other if ("flores_plus", i) in R[sysname] and ("flores_plus", i) in R[a.start]]
        P(f"| {sysname} | {len(d)} | {st.mean(x for x, _ in d):+.2f} | {sum(1 for x, _ in d if x < -0.5)} | " + ", ".join(f"{i} {x:+.1f}" for x, i in sorted(d)[:5]) + " |")
    P("")
    # ---- 4 other benchmarks
    P("## Table 4. NTREX, MAFAND (macro chrF++ over each benchmark's languages; target languages inside them separately), AfriScience test chrF (within-model series only)\n")
    P("| system | NTREX all | NTREX target langs | MAFAND all | MAFAND target langs | AfriScience test chrF |")
    P("|---|---:|---:|---:|---:|---:|")
    for s in systems:
        cells = []
        for b in ("ntrex", "mafand"):
            v = [f(r["chrf"]) for (bb, i), r in R[s].items() if bb == b]
            vt = [f(r["chrf"]) for (bb, i), r in R[s].items() if bb == b and i in T8]
            cells += [f"{st.mean(v):.2f} ({len(v)})" if v else "-", f"{st.mean(vt):.2f} ({len(vt)})" if vt else "-"]
        af = os.path.join(a.results, f"afsci_{s}.json")
        cells.append(f"{json.load(open(af))['splits']['test']['sentence']['macro']['chrf']:.2f}" if os.path.exists(af) else "-")
        P(f"| {s} | " + " | ".join(cells) + " |")
    P("\nThe old-270M columns are not shown for AfriScience (not comparable until the 385M's science stage).\n")
    # ---- 5 weak run diagnostics
    if a.run and os.path.isdir(a.run):
        recs = [json.loads(l) for l in open(os.path.join(a.run, "metrics.jsonl"))]
        ev = [r for r in recs if "bench/chrf_macro" in r]
        P("## Table 5. The weak run's own dev series (8 languages' dev sentences)\n")
        P("| step | dev chrF++ macro | weak_dev chrF++ | held-out AfriCOMET | finished |")
        P("|---:|---:|---:|---:|---:|")
        for r in ev:
            P(f"| {r['step']} | {r['bench/chrf_macro']:.2f} | {r.get('bench_aux/weak_dev/chrf_macro', float('nan')):.2f} | {r.get('bench/comet_macro', float('nan')):.4f} | {r.get('bench/finished_rate', float('nan')):.3f} |")
        tr = [r for r in recs if "grad_norm" in r]
        gm = lambda k: [r[k] for r in tr if k in r]
        P("")
        P(f"Steps {len(tr)}; guide overhead median {st.median(gm('guide/overhead_frac')):.2f}; guide better than the other members in {st.mean(gm('guide/better_frac')):.2f} of groups on average, best member in {st.mean(gm('guide/best_frac')):.2f}; guide advantage share mean {st.mean(gm('guide/adv_share')):.3f}; MBR skipped in {st.mean(gm('guide/mbr_skipped')) if gm('guide/mbr_skipped') else 0:.2f} of steps.\n")
        sm = json.load(open(os.path.join(a.run, "summary.json"))) if os.path.exists(os.path.join(a.run, "summary.json")) else {}
        P(f"Stop reason: {sm.get('stop_reason', 'max steps')}; early_stop_dev: {json.dumps(sm.get('early_stop_dev'))[:300]}; hacking alarm checks file: {'present' if os.path.exists(os.path.join(a.run, 'hacking_alarm_checks.jsonl')) else 'none'}.\n")
        gb = os.path.join(a.run, "guide_by_language.json")
        if os.path.exists(gb):
            g = json.load(open(gb))
            P("Guide by language (MBR guide, every guided prompt): n, win rate (guide above the group mean), best rate, guide reward vs group mean\n")
            P("| iso | mode | n | win rate | best rate | reward guide | reward group mean | src counts |")
            P("|---|---|---:|---:|---:|---:|---:|---|")
            for i in T8:
                v = g.get(i)
                if v: P(f"| {i} | {v.get('mode')} | {v.get('n')} | {v.get('win_rate', 0):.2f} | {v.get('best_rate', 0):.2f} | {v.get('reward_guide', 0):.2f} | {v.get('reward_group_mean', 0):.2f} | {v.get('src')} |")
            P("")
    # ---- 6 verdict
    P("## Verdict (pre-registered rule)\n")
    for sysname in dict.fromkeys((a.best, a.final)):
        t = verdicts.get((sysname, "target8")); o = verdicts.get((sysname, "other"))
        if not t or not o: continue
        c1 = t[0] >= 1.0 and t[1] > 0
        c2 = (t[3] is None) or t[3] >= -0.005
        c3 = o[0] >= -0.3
        P(f"* **{sysname} vs {a.start}: {'HELPS' if (c1 and c2 and c3) else 'DOES NOT MEET THE RULE'}**: target-8 mean chrF++ {t[0]:+.2f} (CI {t[1]:+.2f} to {t[2]:+.2f}; needs >= +1.0 with the CI above 0: {'yes' if c1 else 'no'}); target AfriCOMET mean change {('%+.4f' % t[3]) if t[3] is not None else 'n/a'} (needs >= -0.005: {'yes' if c2 else 'no'}); other-language mean chrF++ change {o[0]:+.2f} (needs >= -0.3: {'yes' if c3 else 'no'}).")
    sys_out = "\n".join(out) + "\n"
    print(sys_out)


if __name__ == "__main__":
    main()
