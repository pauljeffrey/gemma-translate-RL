#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""AB1 analysis: does RL keep its gain when it starts from a much weaker SFT checkpoint?  Paired sentence bootstrap on FLORES+ devtest (no_pair rows), per language and per group.

    python scripts/ab1_analyze.py --eval-dir /workspace/jobs/ab1/eval --runs-dir /workspace/jobs/ab1/runs --out /workspace/jobs/ab1/report \
        [--sys F0=F0 F_mid=F_mid F_end=F_end W0=W0 W_mid=W_mid W_end=W_end] [--boot 1000]
    python scripts/ab1_analyze.py --power-from reports/data/final_model_per_language_wide.csv         # the a-priori power table, no predictions needed

Inputs: <eval-dir>/<name>/predictions.jsonl of `scripts/launch_ab1.sh eval` (one directory per model: sft-final F0, RL snapshots, weak SFT W0 ...). Systems and their roles:
    F0  SFT-final (step 27,638)            F_mid / F_end   arm F after 200 / 400 RL steps
    W0  weak SFT (12.7% of the tokens)      W_mid / W_end   arm W after 200 / 400 RL steps
Every system is scored on the SAME sentences (the intersection of the rows of all systems, per language), chrF++ = sacrebleu CHRF(word_order=2) on NFC text, corpus level per language;
the bootstrap resamples the sentences of a language ONCE per replicate and applies that resample to every system (paired), B = 1000 replicates, 95% percentile intervals.
The intervals describe the test-set sampling only; one RL run per arm means the run-to-run (seed) variance of RL is NOT in them (see the report's caveats).

Language sets (fixed a priori from the SFT-final scores of reports/final_model_per_language_2026-10-21.md, FLORES+ no_pair): WEAK = chrF++ < 25 (19 languages), MID = 25-40 (15), STRONG >= 40 (11), GEEZ = amh tir.
Contrasts (per language and per set; a set value is the unweighted mean over its languages):
    gain_F = F_end - F0        gain_W = W_end - W0        DiD = gain_W - gain_F
    W_end_vs_F0 = W_end - F0   (does weak SFT + RL reach the level SFT-final started RL from?)          W_end_vs_F_end = W_end - F_end   (what RL does not close)
    recovery = (W_end - W0) / (F0 - W0)   (share of the SFT-data gap that RL closed; only where F0 - W0 >= 3 chrF++ for a language, always for a set)
    survive = gain_W / gain_F   (share of the RL gain that survives a weak start; only where gain_F >= 1 for a language)
Also reported: finished rate within 768 tokens, repetition rate (a word 4-gram occurring >= 4 times), the dev curve and training curves of both arms from runs/*/metrics.jsonl.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import unicodedata
from collections import Counter, OrderedDict, defaultdict

import numpy as np

WEAK = "amh bam dyu fon kab kam kbp kea kik kmb knc luo mos nus orm tir umb vmw yor".split()
MID = "bem ewe ibo kin lin lua lug run sag sna som ssw tum twi wol".split()
STRONG = "afr hau mlg nso nya sot swh tsn tso xho zul".split()
GEEZ = ["amh", "tir"]
SETS = OrderedDict([("ALL45", WEAK + MID + STRONG), ("WEAK19", WEAK), ("MID15", MID), ("STRONG11", STRONG), ("GEEZ2", GEEZ)])
DEFAULT_SYS = ["F0=F0", "F_mid=F_mid", "F_end=F_end", "W0=W0", "W_mid=W_mid", "W_end=W_end"]


def nfc(s):
    return unicodedata.normalize("NFC", s)


def rep4(text):
    w = text.split()
    if len(w) < 8:
        return False
    c = Counter(tuple(w[i:i + 4]) for i in range(len(w) - 3))
    return max(c.values()) >= 4


def power_table(path):
    rows = [r for r in csv.DictReader(open(path)) if r["benchmark"] == "flores_plus" and r["subset"] == "no_pair"]
    se = {r["iso"]: (float(r["d_chrf_hi"]) - float(r["d_chrf_lo"])) / 3.92 for r in rows}
    print("A-PRIORI POWER (paired-bootstrap SEs of the SFT-final -> RL-final comparison, FLORES+ no_pair, 1,012 sentences per language; DiD = difference of two such gains, independent runs)")
    print("set      langs  SE(one gain)  SE(DiD)  MDE 80% power (alpha .05, two-sided) = 2.8 x SE(DiD)")
    for name, g in SETS.items():
        g = [i for i in g if i in se]
        s = math.sqrt(sum(se[i] ** 2 for i in g)) / len(g)
        print("%-8s %3d    %.3f        %.3f    %.2f chrF++" % (name, len(g), s, s * math.sqrt(2), 2.8 * s * math.sqrt(2)))
    v = sorted(se.values())
    med = v[len(v) // 2]
    print("one language: median SE %.2f (range %.2f-%.2f) -> MDE %.2f chrF++ for the DiD of a single language" % (med, v[0], v[-1], 2.8 * med * math.sqrt(2)))


def load_pred(path):
    d = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("benchmark") != "flores_plus" or r.get("pair_seen") is not False:
            continue
        d[(r["iso"], str(r["id"]))] = r
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir")
    ap.add_argument("--runs-dir")
    ap.add_argument("--out")
    ap.add_argument("--sys", nargs="*", default=DEFAULT_SYS, help="ROLE=eval-subdir; roles F0 F_mid F_end W0 W_mid W_end (a role may be missing: its contrasts are skipped)")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pylibs", default="/workspace/sft_build/pylibs")
    ap.add_argument("--power-from")
    ap.add_argument("--common-finished", action="store_true", help="only sentences that EVERY system finished within the budget (separates loop repair from translation quality); written with the suffix _cf")
    a = ap.parse_args(argv)
    if a.power_from:
        power_table(a.power_from)
        if not a.eval_dir:
            return 0
    sys.path.insert(0, a.pylibs)
    from sacrebleu.metrics import CHRF

    chrf = CHRF(word_order=2)
    roles = OrderedDict(tuple(x.split("=", 1)) for x in a.sys)
    pred = {}
    for role, sub in roles.items():
        p = os.path.join(a.eval_dir, sub, "predictions.jsonl")
        if os.path.exists(p):
            pred[role] = load_pred(p)
        else:
            print(f"[ab1] {role}: {p} not found, skipped")
    if not pred:
        raise SystemExit("no predictions found")
    langs = sorted({k[0] for k in next(iter(pred.values()))})
    keys = {l: sorted(set.intersection(*[{k for k in d if k[0] == l} for d in pred.values()])) for l in langs}
    if a.common_finished:
        keys = {l: [k for k in ks if all(bool(d[k].get("finished", True)) for d in pred.values())] for l, ks in keys.items()}
    rng = np.random.default_rng(a.seed)
    B = a.boot
    score = {r: {} for r in pred}                                   # role -> iso -> array(B+1): index 0 = point estimate
    fin, rep = {r: {} for r in pred}, {r: {} for r in pred}
    nrows = {}
    for l in langs:
        ks = keys[l]
        n = len(ks)
        if n < 30:
            continue
        nrows[l] = n
        ix = np.concatenate([np.arange(n)[None, :], rng.integers(0, n, size=(B, n))], axis=0)
        for r, d in pred.items():
            hyp = [nfc(d[k]["hyp"]) for k in ks]
            ref = [nfc(d[k]["reference"]) for k in ks]
            st = np.array(chrf._extract_corpus_statistics(hyp, [ref]), dtype=float)
            score[r][l] = np.array([chrf._compute_score_from_stats(list(st[i].sum(0))).score for i in ix])
            fin[r][l] = float(np.mean([bool(d[k].get("finished", True)) for k in ks]))
            rep[r][l] = float(np.mean([rep4(d[k]["hyp"]) for k in ks]))
    langs = [l for l in langs if l in nrows]

    def setvals(name, fn):                                          # fn(role-scores, iso) -> array(B+1); set mean
        g = [i for i in SETS[name] if i in nrows]
        return np.mean([fn(i) for i in g], axis=0) if g else None

    def ci(v):
        return float(v[0]), float(np.percentile(v[1:], 2.5)), float(np.percentile(v[1:], 97.5))

    S = score
    has = lambda *rs: all(r in S for r in rs)
    contrasts = OrderedDict()
    if has("F0", "F_end"):
        contrasts["gain_F = F_end - F0"] = lambda i: S["F_end"][i] - S["F0"][i]
    if has("F0", "F_mid"):
        contrasts["gain_F_mid = F_mid - F0"] = lambda i: S["F_mid"][i] - S["F0"][i]
    if has("W0", "W_end"):
        contrasts["gain_W = W_end - W0"] = lambda i: S["W_end"][i] - S["W0"][i]
    if has("W0", "W_mid"):
        contrasts["gain_W_mid = W_mid - W0"] = lambda i: S["W_mid"][i] - S["W0"][i]
    if has("F0", "F_end", "W0", "W_end"):
        contrasts["DiD = gain_W - gain_F"] = lambda i: (S["W_end"][i] - S["W0"][i]) - (S["F_end"][i] - S["F0"][i])
    if has("F0", "W_end"):
        contrasts["W_end - F0"] = lambda i: S["W_end"][i] - S["F0"][i]
    if has("F_end", "W_end"):
        contrasts["W_end - F_end"] = lambda i: S["W_end"][i] - S["F_end"][i]
    if has("F0", "W0"):
        contrasts["SFT gap F0 - W0"] = lambda i: S["F0"][i] - S["W0"][i]
    out = {"n_rows": nrows, "boot": B, "systems": {r: {"macro_chrf": {s: ci(setvals(s, lambda i, r=r: S[r][i])) for s in SETS},
                                                        "finished": {s: float(np.mean([fin[r][i] for i in SETS[s] if i in nrows])) for s in SETS},
                                                        "rep4": {s: float(np.mean([rep[r][i] for i in SETS[s] if i in nrows])) for s in SETS}} for r in pred},
           "contrasts": {}, "recovery": {}, "survive": {}, "per_language": {}}
    for cn, fn in contrasts.items():
        out["contrasts"][cn] = {s: ci(setvals(s, fn)) for s in SETS}
    if has("F0", "W0", "W_end"):
        for s in SETS:
            num = setvals(s, lambda i: S["W_end"][i] - S["W0"][i])
            den = setvals(s, lambda i: S["F0"][i] - S["W0"][i])
            out["recovery"][s] = ci(num / np.where(np.abs(den) < 1e-9, np.nan, den))
    if has("F0", "F_end", "W0", "W_end"):
        for s in SETS:
            num = setvals(s, lambda i: S["W_end"][i] - S["W0"][i])
            den = setvals(s, lambda i: S["F_end"][i] - S["F0"][i])
            out["survive"][s] = ci(num / np.where(np.abs(den) < 1e-9, np.nan, den))
    for l in langs:
        row = {"n": nrows[l]}
        for r in pred:
            row[r] = ci(S[r][l])
            row[r + "_finished"] = fin[r][l]
        for cn, fn in contrasts.items():
            row[cn] = ci(fn(l))
        out["per_language"][l] = row
    # dev curves + training curves of the two arms
    curves = {}
    for arm, sub in (("F", "ab1_full"), ("W", "ab1_weak")):
        p = os.path.join(a.runs_dir or "", sub, "metrics.jsonl")
        if a.runs_dir and os.path.exists(p):
            rows = [json.loads(x) for x in open(p) if x.strip()]
            curves[arm] = {"dev": [(r["step"], r["bench/chrf_macro"], r.get("bench/finished_rate")) for r in rows if "bench/chrf_macro" in r],
                           "train50": [(r["step"], r.get("reward"), r.get("kl"), r.get("finished_rate"), r.get("completion_len"), r.get("t_step")) for r in rows if "reward" in r and r["step"] % 50 == 0]}
    out["curves"] = curves
    if a.out:
        sfx = "_cf" if a.common_finished else ""
        os.makedirs(a.out, exist_ok=True)
        json.dump(out, open(os.path.join(a.out, "ab1_results" + sfx + ".json"), "w"), indent=1, default=float)
        with open(os.path.join(a.out, "ab1_per_language" + sfx + ".csv"), "w", newline="") as f:
            w = csv.writer(f)
            head = ["iso", "n"] + [f"{r}_chrf" for r in pred] + [f"{r}_finished" for r in pred] + [c for c in contrasts] + [c + "_lo" for c in contrasts] + [c + "_hi" for c in contrasts]
            w.writerow(head)
            for l in langs:
                row = out["per_language"][l]
                w.writerow([l, row["n"]] + [round(row[r][0], 2) for r in pred] + [round(row[r + "_finished"], 3) for r in pred] + [round(row[c][0], 2) for c in contrasts]
                           + [round(row[c][1], 2) for c in contrasts] + [round(row[c][2], 2) for c in contrasts])
        md = [f"# AB1 results (FLORES+ devtest no_pair, greedy 768 tokens, B={B} paired bootstrap{', ONLY sentences finished by every system' if a.common_finished else ''})\n"]
        md.append("## Macro chrF++ per system (mean over the languages of the set; 95% CI)\n\n| system | " + " | ".join(SETS) + " | finished (ALL45) | rep4 (ALL45) |\n|---|" + "---|" * (len(SETS) + 2))
        for r in pred:
            m = out["systems"][r]
            md.append(f"| {r} | " + " | ".join("%.2f [%.2f, %.2f]" % m["macro_chrf"][s] for s in SETS) + " | %.3f | %.3f |" % (m["finished"]["ALL45"], m["rep4"]["ALL45"]))
        md.append("\n## Contrasts (chrF++ points; 95% CI)\n\n| contrast | " + " | ".join(SETS) + " |\n|---|" + "---|" * len(SETS))
        for cn in contrasts:
            md.append(f"| {cn} | " + " | ".join("%+.2f [%+.2f, %+.2f]" % out["contrasts"][cn][s] for s in SETS) + " |")
        if out["recovery"]:
            md.append("| recovery = gain_W / (F0 - W0) | " + " | ".join("%.2f [%.2f, %.2f]" % out["recovery"][s] for s in SETS) + " |")
        if out["survive"]:
            md.append("| survive = gain_W / gain_F | " + " | ".join("%.2f [%.2f, %.2f]" % out["survive"][s] for s in SETS) + " |")
        if curves:
            md.append("\n## Dev chrF++ during training (19 languages x 40 sentences, 256 tokens, from metrics.jsonl)\n")
            for arm in curves:
                md.append(f"* arm {arm}: " + ", ".join("%d:%.2f(fin %.3f)" % (s, c, f or float("nan")) for s, c, f in curves[arm]["dev"]))
        md.append("\n## Per language (point estimates; CIs in ab1_per_language.csv)\n\n| iso | n | " + " | ".join(pred) + " | gain_F | gain_W | DiD |\n|---|---|" + "---|" * (len(pred) + 3))
        for l in sorted(langs, key=lambda i: -out["per_language"][i].get("F0", (0,))[0] if "F0" in pred else 0):
            row = out["per_language"][l]
            g = lambda c: ("%+.1f" % row[c][0]) if c in row else ""
            md.append(f"| {l} | {row['n']} | " + " | ".join("%.1f" % row[r][0] for r in pred) + f" | {g('gain_F = F_end - F0')} | {g('gain_W = W_end - W0')} | {g('DiD = gain_W - gain_F')} |")
        open(os.path.join(a.out, "ab1_report" + sfx + ".md"), "w").write("\n".join(md) + "\n")
        print("wrote", os.path.join(a.out, "ab1_report" + sfx + ".md"))
    else:
        print(json.dumps(out["contrasts"], indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
