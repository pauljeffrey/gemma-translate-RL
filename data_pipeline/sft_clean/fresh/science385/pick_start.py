#!/usr/bin/env python3
"""Start checkpoint of the science SFT (owner rule 2026-11-03): best by dev chrF++ with the held-out AfriCOMET not worse than the other candidate beyond its noise; the choice and the reason are written to JSON.

    python pick_start.py --candidate mixed=/workspace/clean_rerun/runs_385m/mixed --candidate weak=/workspace/clean_rerun/runs_385m_weak/mixed --out choice.json

Each candidate run dir has metrics.jsonl (eval records with bench/chrf_macro, bench/comet_macro, bench/comet_macro_sd, per-language bench/chrf/<iso> and bench/comet/<iso>) and best/ . Candidates are compared on the
LANGUAGES THEIR DEV SETS SHARE (the weak run's dev set only holds its 8 languages): mean chrF++ and mean held-out AfriCOMET over the common languages at each candidate's best evaluation (best = highest dev chrF++ macro of its own set).
The candidate with the higher common-language chrF++ wins unless its AfriCOMET (common languages) is lower than the other's by more than COMET_NOISE (0.005); then the other wins. A run without any evaluation after step 0, or without best/, is skipped.
LIMIT (recorded in the output): the weak run trains 8 languages only, its dev set cannot show regression elsewhere; that is checked afterwards by the NTREX / MAFAND non-regression of the SFT evaluation.
"""
import argparse, json, os
COMET_NOISE = 0.005


def best_record(run):
    p = os.path.join(run, "metrics.jsonl")
    if not os.path.exists(p) or not os.path.exists(os.path.join(run, "best", "config.json")): return None
    recs = [json.loads(l) for l in open(p)]
    ev = [r for r in recs if "bench/chrf_macro" in r and r.get("step", 0) > 0]
    return max(ev, key=lambda r: r["bench/chrf_macro"]) if ev else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--candidate", action="append", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    cands = {}
    for c in a.candidate:
        name, run = c.split("=", 1); r = best_record(run)
        if r: cands[name] = (run, r)
        else: print("skipped", name, "(no evaluation after step 0 or no best/)")
    if not cands: raise SystemExit("no usable candidate")
    langs = None
    for run, r in cands.values():
        s = {k.split("/")[-1] for k in r if k.startswith("bench/chrf/")}; langs = s if langs is None else langs & s
    def stat(r, pre):
        v = [r[f"bench/{pre}/{l}"] for l in sorted(langs) if f"bench/{pre}/{l}" in r]
        return sum(v) / len(v) if v else None
    table = {n: dict(step=r["step"], dev_macro=r["bench/chrf_macro"], common_chrf=stat(r, "chrf"), common_comet=stat(r, "comet"), run=run) for n, (run, r) in cands.items()}
    order = sorted(table, key=lambda n: -table[n]["common_chrf"]); win, why = order[0], "highest common-language dev chrF++"
    for other in order[1:]:
        cw, co = table[win]["common_comet"], table[other]["common_comet"]
        if cw is not None and co is not None and cw < co - COMET_NOISE:
            win, why = other, f"{win} has the higher chrF++ but its held-out AfriCOMET is lower by {co - cw:.4f} (> noise {COMET_NOISE}) than {other}"
    out = dict(chosen=win, reason=why, model_dir=os.path.join(table[win]["run"], "best"), common_languages=sorted(langs), candidates=table,
               limit="dev sets of different runs differ; regression on languages outside a candidate's dev set is not visible here")
    json.dump(out, open(a.out, "w"), indent=1); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
