#!/usr/bin/env python3
"""Score the saved dev predictions of an RL run offline with the REFERENCE-BASED AfriCOMET and replay the dev monitors (patch 3, owner 2026-11-03).

    python scripts/dev_comet_retro.py RUN_DIR [--model masakhane/africomet-stl-1.1] [--batch-size 64] [--fake] [--k 3 --eps 0.15 --eps-comet 0.004 --alarm-eps-chrf 0.3 --alarm-eps-comet 0.005]

Needs `<RUN_DIR>/dev_predictions/step_<N>.jsonl` (config `save_dev_predictions`, on from stage B1 on; stage A and the B1 steps before the patch did NOT save them: only the dev chrF++ numbers of
metrics.jsonl exist for them, no translations, so no AfriCOMET can be computed retroactively for stage A). For every saved dev evaluation it computes per language chrF++ and AfriCOMET (AfriCOMET languages only:
gemma_mt.languages.QE_LANGUAGES), the macro over languages for the bench set and for every aux set, then replays on that series (with the hold-out QE `eval/qe_mean_scored` and the finished rates of
metrics.jsonl when present) the rules of the trainer: the dev patience of patch 2 (chrF++ only), the combined stop of patch 3 (also needs the AfriCOMET macro to be flat) and the reward-hacking alarm, and
prints the first step at which each would have stopped the stage. Output: a table on stdout, `<RUN_DIR>/dev_comet_retro.json`, and the per-step scores are cached in `<RUN_DIR>/dev_comet_retro_scores.json`.
chrF++ is sacrebleu CHRF(word_order=2) over the language's sentences (corpus level, as the trainer's dev evaluation) when sacrebleu is importable, else the mean sentence chrF++ (marked in the output)."""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from gemma_mt.languages import QE_LANGUAGES                        # noqa: E402
from gemma_mt.rl import consensus as cons                          # noqa: E402
from gemma_mt.rl.run_controls import DevPatience, HackingAlarm     # noqa: E402


def chrf(hyps, refs):
    try:
        from gemma_mt.eval import metrics as M

        return float(M.chrf_pp(hyps, refs)), "sacrebleu corpus chrF++"
    except Exception:                                              # noqa: BLE001
        return float(np.mean([cons.chrf_sentence(h, r) for h, r in zip(hyps, refs)])), "mean sentence chrF++ (sacrebleu not importable)"


def load_steps(run):
    files = sorted(glob.glob(os.path.join(run, "dev_predictions", "step_*.jsonl")), key=lambda f: int(re.search(r"step_(\d+)", f).group(1)))
    return {int(re.search(r"step_(\d+)", f).group(1)): [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()] for f in files}


def macro(per_lang):
    return float(np.mean(list(per_lang.values()))) if per_lang else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--model", default="masakhane/africomet-stl-1.1")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--fake", action="store_true", help="FakeScorer instead of the model (tests)")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--eps", type=float, default=0.15)
    ap.add_argument("--eps-comet", type=float, default=0.004)
    ap.add_argument("--min-steps", type=int, default=75)
    ap.add_argument("--alarm-eps-chrf", type=float, default=0.3)
    ap.add_argument("--alarm-eps-comet", type=float, default=0.005)
    ap.add_argument("--aux", default="weak_dev")
    a = ap.parse_args()
    steps = load_steps(a.run)
    if not steps:
        print(f"no {a.run}/dev_predictions/step_*.jsonl: this run did not save its dev predictions (config save_dev_predictions; stage A and the early B1 evaluations did not). "
              "Only the dev chrF++ of metrics.jsonl exists for them; AfriCOMET cannot be computed retroactively without the translations.")
        return 1
    cache_path = os.path.join(a.run, "dev_comet_retro_scores.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    scorer = None
    qe_series, fin_series = {}, {}
    mp = os.path.join(a.run, "metrics.jsonl")
    if os.path.exists(mp):
        for l in open(mp):
            r = json.loads(l)
            if "eval/qe_mean_scored" in r:
                qe_series[int(r["step"])] = r["eval/qe_mean_scored"]
                fin_series[int(r["step"])] = (r.get("bench/finished_rate"), r.get("eval/greedy_finished_rate"))
            elif "bench/finished_rate" in r:
                fin_series.setdefault(int(r["step"]), (r.get("bench/finished_rate"), None))
    table, how = [], ""
    for step, rows in steps.items():
        key = str(step)
        if key not in cache:
            if scorer is None:
                if a.fake:
                    from gemma_mt.rl.comet import FakeScorer

                    scorer = FakeScorer()
                else:
                    from gemma_mt.rl.comet import CometScorer

                    scorer = CometScorer(a.model, batch_size=a.batch_size, reference_based=True)
            sel = [r for r in rows if r["iso"] in QE_LANGUAGES and r["ref"].strip()]
            sc = scorer([r["src"] for r in sel], [r["hyp"] for r in sel], [r["ref"] for r in sel])
            cache[key] = {f"{r['set']}|{r['iso']}|{r['id']}": float(s) for r, s in zip(sel, sc)}
            json.dump(cache, open(cache_path, "w"))
        by = {}
        for r in rows:
            by.setdefault(r["set"], {}).setdefault(r["iso"], {"h": [], "r": [], "c": []})
            d = by[r["set"]][r["iso"]]
            d["h"].append(r["hyp"]); d["r"].append(r["ref"])
            c = cache[key].get(f"{r['set']}|{r['iso']}|{r['id']}")
            if c is not None:
                d["c"].append(c)
        ent = {"step": step}
        for name, langs in by.items():
            ch = {}
            for iso, d in langs.items():
                ch[iso], how = chrf(d["h"], d["r"])
            cm = {iso: float(np.mean(d["c"])) for iso, d in langs.items() if d["c"]}
            ent[name] = {"chrf_macro": macro(ch), "comet_macro": macro(cm), "n_comet_languages": len(cm), "chrf": ch, "comet": cm}
        table.append(ent)
    print(f"{'step':>6} {'dev chrF++':>10} {a.aux + ' chrF++':>14} {'AfriCOMET':>10} {a.aux + ' AfriCOMET':>16} {'hold-out QE':>12}")
    for e in table:
        b, w = e.get("bench", {}), e.get(a.aux, {})
        print(f"{e['step']:6d} {b.get('chrf_macro', float('nan')):10.2f} {w.get('chrf_macro', float('nan')):14.2f} {b.get('comet_macro', float('nan')):10.4f} {w.get('comet_macro', float('nan')):16.4f} "
              f"{qe_series.get(e['step'], float('nan')):12.4f}")
    print("(" + how + ")")
    # ---- replay
    res = {}
    for label, require in (("dev_patience (patch 2, chrF++ only)", False), ("combined stop (patch 3, + AfriCOMET flat)", True)):
        d = DevPatience(a.k, a.eps, 0.98, a.eps_comet, require)
        fired = None
        for e in table:
            b, w = e.get("bench", {}), e.get(a.aux, {})
            d.observe(e["step"], b["chrf_macro"], w.get("chrf_macro"), "run", b.get("comet_macro") if b.get("n_comet_languages") else None)
            fb = fin_series.get(e["step"], (None, None))
            dec = d.check(e["step"], "run", 0, a.min_steps, True, fb[0], fb[1])
            if dec["fire"] and fired is None:
                fired = e["step"]
        res[label] = fired
    h = HackingAlarm(a.k, a.alarm_eps_chrf, a.alarm_eps_comet)
    alarm = None
    for e in table:
        b = e.get("bench", {})
        h.observe(e["step"], b["chrf_macro"], b.get("comet_macro") if b.get("n_comet_languages") else None, qe_series.get(e["step"]), "run")
        if h.check(e["step"])["fire"] and alarm is None:
            alarm = e["step"]
    res["reward-hacking alarm"] = alarm
    for k_, v_ in res.items():
        print(f"{k_}: " + (f"would have stopped at step {v_}" if v_ is not None else "would not have stopped within the saved evaluations"))
    json.dump({"table": table, "replay": res, "params": vars(a)}, open(os.path.join(a.run, "dev_comet_retro.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
