#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""AB2 analysis: RL from a strong vs a weak SFT start (arms F, W), and RL against two continued-SFT controls (S1, S1P) from the weak start.  CPU only.

    python scripts/ab2_analyze.py --eval-dir /workspace/jobs/ab2/eval --runs-dir /workspace/jobs/ab2/runs --out /workspace/jobs/ab2/results [--boot 1000] [--langs yor kin orm lin]
    python scripts/ab2_analyze.py --power-from reports/data/final_model_per_language_wide.csv          # the a-priori power table (no predictions needed)

Systems (one directory per system under --eval-dir, written by scripts/ab2_eval.sh: sys.jsonl.gz = per-sentence surface metrics, neural/<metric>.jsonl = AfriCOMET / SSA-COMET):
    F0 SFT-final | W0 weak SFT | F_RL = F0 + RL | W_RL = W0 + RL | W_S1 = W0 + continued SFT, GPU-time matched | W_S1P = W0 + continued SFT, pair-matched (3,136 samples)
Units: FLORES+ devtest (no_pair = the reference pair is not in the SFT text: the primary set; all rows), NTREX, MAFAND, SMOL-sent (no_pair) and their pool (POOLED, no_pair).
Metrics: chrF++ (sacrebleu CHRF word_order 2, NFC, corpus level per language), spBLEU (BLEU tokenize flores200), AfriCOMET-ref, SSA-COMET-ref (mean of the three views), AfriCOMET-QE and
SSA-COMET-QE (AfriCOMET-QE is the RL reward model: NOT independent evidence, reported but never used in the decision), finished rate (stop token inside 768 tokens), GlotLID ok.
Every system is scored on the SAME sentences (intersection per unit and language); the bootstrap resamples the sentences of a (unit, language) ONCE per replicate and applies it to every
system (paired), B = 1000, 95% percentile intervals; a macro value is the unweighted mean over the languages. The intervals describe test-set sampling only: ONE run per arm.
The decision rule (DECISION_RULE.txt, written before any result existed) is applied by `decide()` below and written to decision.json.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import sys
import unicodedata
from collections import OrderedDict, defaultdict

import numpy as np

ROLES = ["F0", "W0", "F_RL", "W_RL", "W_S1", "W_S1P"]
BENCH = OrderedDict([("flores_plus", "FLORES+"), ("ntrex", "NTREX"), ("mafand", "MAFAND"), ("smol_sent", "SMOL")])
NEURAL = ["afri_ref", "ssa_ref", "afri_qe", "ssa_qe"]
TEXT = ["chrf", "spbleu"]
RATES = ["fin", "lid"]
DEFAULT_LANGS = ["yor", "kin", "orm", "lin"]


def nfc(s):
    return unicodedata.normalize("NFC", s or "")


def load_sys(path):
    d = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            d[(r["b"], r["iso"], str(r["id"]))] = r
    return d


def load_neural(dirpath):
    out = {}
    for m in NEURAL:
        p = os.path.join(dirpath, m + ".jsonl")
        if not os.path.exists(p):
            continue
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            b, iso = r["chunk"].split("|")
            for i, v in r["scores"].items():
                val = v.get("s", next(iter(v.values())))
                out[(m, b, iso, str(i))] = float(val)
    return out


def power_table(path, langs):
    rows = [r for r in csv.DictReader(open(path)) if r["benchmark"] == "flores_plus" and r["subset"] == "no_pair" and r["iso"] in langs]
    se = {r["iso"]: (float(r["d_chrf_hi"]) - float(r["d_chrf_lo"])) / 3.92 for r in rows}
    n = {r["iso"]: int(r["n"]) for r in rows}
    print("A-PRIORI POWER (paired-bootstrap SEs of the SFT-final -> RL-final comparison of the main run, FLORES+ no_pair)")
    for l in langs:
        print(f"  {l}: n={n.get(l)}  SE of one RL gain = {se.get(l, float('nan')):.2f}")
    s = math.sqrt(sum(v ** 2 for v in se.values())) / len(se)
    print(f"macro over {len(se)} languages: SE(one gain) {s:.2f}; SE(gain_W - gain_F) {s * math.sqrt(2):.2f} -> MDE (80% power, two-sided 5%, 2.8 x SE) = {2.8 * s * math.sqrt(2):.2f} chrF++;"
          f" SE(W_RL - W_S1) ~ {s:.2f} (paired, same sentences) -> MDE {2.8 * s:.2f} chrF++; single language: SE(DiD) {math.sqrt(2) * np.median(list(se.values())):.2f} -> MDE {2.8 * math.sqrt(2) * np.median(list(se.values())):.2f}")
    return s


class Boot:
    def __init__(self, B, seed):
        self.B, self.seed, self.ix = B, seed, {}

    def index(self, key, n):
        k = (key, n)
        if k not in self.ix:
            h = int(hashlib.sha1(f"{self.seed}|{key}".encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(h)
            self.ix[k] = np.concatenate([np.arange(n)[None, :], rng.integers(0, n, size=(self.B, n))], axis=0)
        return self.ix[k]


def ci(v):
    v = np.asarray(v)
    return [float(v[0]), float(np.percentile(v[1:], 2.5)), float(np.percentile(v[1:], 97.5))]


class StatCache:
    """Per-sentence sufficient statistics of chrF++ / BLEU, computed ONCE per (system, metric, row) and reused by every unit and bootstrap."""

    def __init__(self, chrf, bleu):
        self.chrf, self.bleu, self.c = chrf, bleu, {}

    def get(self, role, metric, d, keys):
        store = self.c.setdefault((role, metric), {})
        miss = [k for k in keys if k not in store]
        if miss:
            hyp, ref = [nfc(d[k]["hyp"]) for k in miss], [nfc(d[k]["ref"]) for k in miss]
            st = np.array((self.chrf if metric == "chrf" else self.bleu)._extract_corpus_statistics(hyp, [ref]), dtype=float)
            for k, v in zip(miss, st):
                store[k] = v
        return np.stack([store[k] for k in keys])


def score_from_stats(metric, st, ix, chrf, bleu):
    """st: [n, ...] per-sentence statistics; ix: [B+1, n] resample indices (row 0 = identity). Multiplicity matrix times statistics, then the metric's own formula."""
    f = (chrf if metric == "chrf" else bleu)._compute_score_from_stats
    n = st.shape[0]
    flat = st.reshape(n, -1)
    cnt = np.stack([np.bincount(r, minlength=n) for r in ix]).astype(float)
    tot = cnt @ flat
    shape = st.shape[1:]
    return np.array([f(list(t.reshape(shape).tolist()) if len(shape) == 1 else t.reshape(shape).tolist()).score for t in tot])


def build(args):
    sys.path.insert(0, args.pylibs)
    from sacrebleu.metrics import BLEU, CHRF

    chrf, bleu = CHRF(word_order=2), BLEU(tokenize="flores200")
    systems, neural = OrderedDict(), {}
    for role in ROLES:
        sub = args.sys.get(role, role)
        p = os.path.join(args.eval_dir, sub, "sys.jsonl.gz")
        if os.path.exists(p):
            systems[role] = load_sys(p)
            neural[role] = load_neural(os.path.join(args.eval_dir, sub, "neural"))
        else:
            print(f"[ab2] {role}: {p} not found, skipped")
    if not systems:
        raise SystemExit("no system found")
    boot = Boot(args.boot, args.seed)
    SC = StatCache(chrf, bleu)
    have_m = {r: {k[0] for k in neural[r]} for r in neural}
    langs = args.langs
    units = OrderedDict()                       # unit -> {lang: [keys]}
    first = next(iter(systems.values()))
    def ok(r, subset):
        return True if subset == "all" else (r.get("pair_seen") is False)
    defs = [("FLORES+ no_pair", ["flores_plus"], "no_pair"), ("FLORES+ all", ["flores_plus"], "all"), ("NTREX no_pair", ["ntrex"], "no_pair"), ("MAFAND no_pair", ["mafand"], "no_pair"),
            ("SMOL no_pair", ["smol_sent"], "no_pair"), ("POOLED no_pair", list(BENCH), "no_pair")]
    for uname, bs, sub in defs:
        per = {}
        for l in langs:
            ks = None
            for role, d in systems.items():
                s = {k for k, r in d.items() if k[0] in bs and k[1] == l and ok(r, sub)}
                ks = s if ks is None else ks & s
            if ks:
                per[l] = sorted(ks)
        if per:
            units[uname] = per
    res = {"units": {}, "systems": list(systems)}
    cache = {}
    for uname, per in units.items():
        U = {"n": {l: len(k) for l, k in per.items()}, "metrics": {}, "cf": {}}
        for cf in (False, True):
            tag = "cf" if cf else "all"
            Mv = {}                                             # metric -> role -> lang -> array(B+1)
            nrows = {}
            nrows_n = {}
            for l, ks in per.items():
                if cf:
                    ks = [k for k in ks if all(bool(d[k].get("fin", True)) for d in systems.values())]
                if len(ks) < 30:
                    continue
                nrows[l] = len(ks)
                ix = boot.index((uname, l, tag), len(ks))
                nks = {m: [k for k in ks if all((m, k[0], l, k[2]) in neural[r] for r in systems)] for m in NEURAL if all(m in have_m[r] for r in systems)}
                for role, d in systems.items():
                    for m in TEXT:
                        st = SC.get(role, m, d, ks)
                        Mv.setdefault(m, {}).setdefault(role, {})[l] = score_from_stats(m, st, ix, chrf, bleu)
                    for m in RATES:
                        vals = np.array([float(bool(d[k].get(m))) if d[k].get(m) is not None else np.nan for k in ks])
                        if not np.isnan(vals).all():
                            vv = np.nan_to_num(vals, nan=float(np.nanmean(vals)))
                            Mv.setdefault(m, {}).setdefault(role, {})[l] = vv[ix].mean(1) * 100
                    for m, kn in nks.items():                      # neural metrics: the rows scored for EVERY system (all rows when the scorer ran with --cap 0)
                        if len(kn) < 30:
                            continue
                        vals = [neural[role][(m, k[0], l, k[2])] for k in kn]
                        Mv.setdefault(m, {}).setdefault(role, {})[l] = np.asarray(vals)[boot.index((uname, l, tag, m), len(kn))].mean(1) * 100
                        nrows_n.setdefault(m, {})[l] = len(kn)
            # drop neural metrics not available for every role
            for m in list(Mv):
                if len(Mv[m]) != len(systems) or any(set(Mv[m][r]) != set(Mv[m][next(iter(Mv[m]))]) for r in Mv[m]):
                    del Mv[m]
            out = {"n": nrows, "n_neural": nrows_n, "macro": {}, "contrasts": {}, "per_language": {}}
            ls = sorted(nrows)
            for m, byrole in Mv.items():
                out["macro"][m] = {r: ci(np.mean([byrole[r][l] for l in ls], axis=0)) for r in byrole}
                out["per_language"][m] = {l: {r: ci(byrole[r][l]) for r in byrole} for l in ls}
                con = contrasts(byrole, ls)
                out["contrasts"][m] = {c: ci(v) for c, v in con["macro"].items()}
                out.setdefault("contrasts_pl", {})[m] = {l: {c: ci(v) for c, v in con["lang"][l].items()} for l in ls}
                out.setdefault("ratios", {})[m] = {c: ci(v) for c, v in con["ratio"].items()}
            U[tag] = out
        res["units"][uname] = U
    return res


def contrasts(byrole, ls):
    """Replicate arrays for every contrast, per language and as the language macro."""
    have = lambda *r: all(x in byrole for x in r)
    defs = OrderedDict()
    if have("F0", "F_RL"): defs["gain_F = F_RL - F0"] = lambda r, l: r["F_RL"][l] - r["F0"][l]
    if have("W0", "W_RL"): defs["gain_W = W_RL - W0"] = lambda r, l: r["W_RL"][l] - r["W0"][l]
    if have("F0", "F_RL", "W0", "W_RL"): defs["DiD = gain_W - gain_F"] = lambda r, l: (r["W_RL"][l] - r["W0"][l]) - (r["F_RL"][l] - r["F0"][l])
    if have("F0", "W0"): defs["gap0 = F0 - W0"] = lambda r, l: r["F0"][l] - r["W0"][l]
    if have("F0", "W_RL"): defs["W_RL - F0"] = lambda r, l: r["W_RL"][l] - r["F0"][l]
    if have("F_RL", "W_RL"): defs["W_RL - F_RL"] = lambda r, l: r["W_RL"][l] - r["F_RL"][l]
    if have("W0", "W_S1"): defs["gain_S1 = W_S1 - W0"] = lambda r, l: r["W_S1"][l] - r["W0"][l]
    if have("W0", "W_S1P"): defs["gain_S1P = W_S1P - W0"] = lambda r, l: r["W_S1P"][l] - r["W0"][l]
    if have("W_RL", "W_S1"): defs["RL - S1 = W_RL - W_S1"] = lambda r, l: r["W_RL"][l] - r["W_S1"][l]
    if have("W_RL", "W_S1P"): defs["RL - S1P = W_RL - W_S1P"] = lambda r, l: r["W_RL"][l] - r["W_S1P"][l]
    lang = {l: {c: f(byrole, l) for c, f in defs.items()} for l in ls}
    macro = {c: np.mean([lang[l][c] for l in ls], axis=0) for c in defs}
    ratio = {}
    def rat(n, d):
        d = np.where(np.abs(d) < 1e-9, np.nan, d)
        return n / d
    if "gain_W = W_RL - W0" in macro and "gain_F = F_RL - F0" in macro:
        ratio["survival = gain_W / gain_F"] = rat(macro["gain_W = W_RL - W0"], macro["gain_F = F_RL - F0"])
    if "gain_W = W_RL - W0" in macro and "gap0 = F0 - W0" in macro:
        ratio["gap closed by RL = gain_W / (F0 - W0)"] = rat(macro["gain_W = W_RL - W0"], macro["gap0 = F0 - W0"])
    if "gain_S1 = W_S1 - W0" in macro and "gap0 = F0 - W0" in macro:
        ratio["gap closed by S1"] = rat(macro["gain_S1 = W_S1 - W0"], macro["gap0 = F0 - W0"])
    if "gain_S1P = W_S1P - W0" in macro and "gap0 = F0 - W0" in macro:
        ratio["gap closed by S1P"] = rat(macro["gain_S1P = W_S1P - W0"], macro["gap0 = F0 - W0"])
    return {"lang": lang, "macro": macro, "ratio": ratio}


def decide(res, mde):
    """Apply DECISION_RULE.txt. Primary endpoint: FLORES+ no_pair language-macro chrF++ (all sentences); everything else is confirmation."""
    P = res["units"].get("FLORES+ no_pair", {}).get("all")
    out = {"mde_chrf": mde, "preconditions": {}, "verdict_A": None, "verdict_B": None, "notes": []}
    if not P or "chrf" not in P["contrasts"]:
        out["notes"].append("primary unit not available")
        return out
    c = P["contrasts"]["chrf"]; r = P["ratios"]["chrf"]; mac = P["macro"]["chrf"]
    g = lambda k: c.get(k)
    lo = lambda k: g(k)[1] if g(k) else float("nan")
    hi = lambda k: g(k)[2] if g(k) else float("nan")
    pt = lambda k: g(k)[0] if g(k) else float("nan")
    fin_w0 = res["units"]["FLORES+ no_pair"]["all"]["macro"].get("fin", {}).get("W0", [float("nan")])[0]
    pre = out["preconditions"]
    pre["P1_gap0>=3"] = pt("gap0 = F0 - W0") >= 3.0
    pre["P2_gain_F_CI_above_0"] = lo("gain_F = F_RL - F0") > 0
    pre["P3_W0_finished>=60%"] = fin_w0 >= 60.0
    # replication on the other benchmarks (sign and CI of gain_W) and agreement of the reference-based neural metrics
    rep = {}
    for u in ("NTREX no_pair", "MAFAND no_pair", "SMOL no_pair"):
        U = res["units"].get(u, {}).get("all")
        if U and "chrf" in U["contrasts"] and "gain_W = W_RL - W0" in U["contrasts"]["chrf"]:
            rep[u] = U["contrasts"]["chrf"]["gain_W = W_RL - W0"][1] > 0
    out["replication_gain_W_CI_above_0"] = rep
    neural_ok = {}
    for m in ("ssa_ref", "afri_ref"):
        if m in P["contrasts"] and "gain_W = W_RL - W0" in P["contrasts"][m]:
            neural_ok[m] = P["contrasts"][m]["gain_W = W_RL - W0"][1] > 0
    out["neural_ref_gain_W_CI_above_0"] = neural_ok
    cf = res["units"]["FLORES+ no_pair"].get("cf", {})
    cf_ok = bool(cf and "chrf" in cf.get("contrasts", {}) and cf["contrasts"]["chrf"].get("gain_W = W_RL - W0", [0, -1, 0])[1] > 0)
    out["common_finished_gain_W_CI_above_0"] = cf_ok
    pl = P.get("contrasts_pl", {}).get("chrf", {})
    n_lang_pos = sum(1 for l, d in pl.items() if d.get("gain_W = W_RL - W0", [0, -1, 0])[1] > 0)
    out["languages_with_gain_W_CI_above_0"] = f"{n_lang_pos}/{len(pl)}"
    surv = r.get("survival = gain_W / gain_F")
    out["survival"] = surv
    if not (pre["P1_gap0>=3"] and pre["P2_gain_F_CI_above_0"] and pre["P3_W0_finished>=60%"]):
        out["verdict_A"] = "INCONCLUSIVE (a precondition failed: " + ", ".join(k for k, v in pre.items() if not v) + ")"
    else:
        gW_pos = lo("gain_W = W_RL - W0") > 0
        if not gW_pos:
            out["verdict_A"] = "REFUTE" if hi("gain_W = W_RL - W0") < pt("gain_F = F_RL - F0") - mde else "INCONCLUSIVE (gain_W not distinguishable from 0 but also not clearly below gain_F)"
        else:
            sup = (surv is not None and surv[0] >= 0.5 and sum(rep.values()) >= 2 and cf_ok and n_lang_pos >= 3 and all(neural_ok.values()) and bool(neural_ok))
            out["verdict_A"] = "SUPPORT" if sup else "PARTIAL SUPPORT"
        out["may_say_RL_replaces_SFT"] = bool(lo("W_RL - F_RL") <= 0 <= hi("W_RL - F_RL"))
        out["may_say_W_RL_reaches_F0"] = bool(hi("W_RL - F0") >= 0)
    # verdict B: RL against continued SFT at equal cost / equal data
    if "RL - S1 = W_RL - W_S1" in c or "RL - S1P = W_RL - W_S1P" in c:
        parts = {}
        for k in ("RL - S1 = W_RL - W_S1", "RL - S1P = W_RL - W_S1P"):
            if k in c:
                parts[k] = "RL better" if lo(k) > 0 else ("SFT better" if hi(k) < 0 else "no difference detectable")
        out["verdict_B_parts"] = parts
        vals = set(parts.values())
        out["verdict_B"] = "RL BETTER than continued SFT (both controls)" if vals == {"RL better"} else ("CONTINUED SFT BETTER (at least one control)" if "SFT better" in vals else "COMPARABLE / MIXED: " + str(parts))
    return out


def paper_json(res, runs_dir, steps, completions, s1_steps):
    P = res["units"].get("FLORES+ no_pair", {}).get("all", {})
    mac = P.get("macro", {}).get("chrf", {})
    fin = P.get("macro", {}).get("fin", {})
    arms = {}
    def arm(name, start, end, curve_run):
        a = {}
        if start in mac and end in mac:
            a.update(start_flores=round(mac[start][0], 2), end_flores=round(mac[end][0], 2), gain_flores=round(mac[end][0] - mac[start][0], 2))
        if start in fin and end in fin:
            a["finished"] = f"{fin[start][0] / 100:.3f} / {fin[end][0] / 100:.3f}"
        a["geez_gain"] = "n/a (Ge'ez languages not in this design)"
        a["weak_gain"] = a.get("gain_flores", "TODO(ablation)")
        a["completions"] = completions if curve_run else f"0 (continued SFT, {s1_steps} steps)" if name == "W_S1" else "0 (continued SFT, pair-matched)"
        if curve_run and "gain_flores" in a:
            a["gain_per_m"] = round(a["gain_flores"] / (completions / 1e6), 2)
        a["curve"] = []
        p = os.path.join(runs_dir or "", curve_run or "x", "metrics.jsonl")
        if curve_run and os.path.exists(p):
            rows = [json.loads(x) for x in open(p) if x.strip()]
            a["curve"] = [[r["step"], r["bench/chrf_macro"]] for r in rows if "bench/chrf_macro" in r]
        return a
    arms["S"] = arm("S", "F0", "F_RL", "ab2_full")
    arms["W"] = arm("W", "W0", "W_RL", "ab2_weak")
    arms["W_S1"] = arm("W_S1", "W0", "W_S1", None)
    arms["W_S1P"] = arm("W_S1P", "W0", "W_S1P", None)
    return {"arms": arms, "units": "RL steps", "note": "S = RL from SFT-final (arm F); W = RL from the weak SFT; W_S1 / W_S1P = continued SFT controls from the weak SFT (GPU-time matched / pair-matched). "
            "Values: FLORES+ devtest no_pair, language macro over yor kin orm lin (not the 45-language macro)."}


def fmt(v, p=2):
    return f"{v[0]:.{p}f} [{v[1]:.{p}f}, {v[2]:.{p}f}]"


def write_report(res, dec, path, args, mde):
    md = [f"# AB2 results (languages {', '.join(args.langs)}; greedy 768 tokens; B={args.boot} paired bootstrap; ONE run per arm)\n"]
    md.append(f"Systems found: {', '.join(res['systems'])}. MDE (a priori, 80% power) for the DiD of the macro: {mde[0]:.2f} chrF++; for RL vs a control: {mde[1]:.2f}.\n")
    for u, U in res["units"].items():
        for tag in ("all", "cf"):
            if tag not in U or not U[tag]["macro"]:
                continue
            T = U[tag]
            md.append(f"\n## {u}{' (ONLY sentences finished by every system)' if tag == 'cf' else ''}   n = {T['n']}\n")
            ms = list(T["macro"])
            md.append("| system | " + " | ".join(ms) + " |\n|---|" + "---|" * len(ms))
            for r in res["systems"]:
                md.append(f"| {r} | " + " | ".join(fmt(T["macro"][m][r], 2 if m in ("chrf", "spbleu") else 1) for m in ms) + " |")
            md.append("\n| contrast | " + " | ".join(ms) + " |\n|---|" + "---|" * len(ms))
            for cn in T["contrasts"].get("chrf", {}):
                md.append(f"| {cn} | " + " | ".join((fmt(T["contrasts"][m][cn], 2) if cn in T["contrasts"][m] else "") for m in ms) + " |")
            for cn in T["ratios"].get("chrf", {}):
                md.append(f"| {cn} (chrF++) | " + fmt(T["ratios"]["chrf"][cn], 2) + " |")
    md.append("\n## Decision (DECISION_RULE.txt applied by decide())\n\n```\n" + json.dumps(dec, indent=1, default=float) + "\n```\n")
    open(path, "w").write("\n".join(md) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir")
    ap.add_argument("--runs-dir")
    ap.add_argument("--out")
    ap.add_argument("--sys", nargs="*", default=[], help="ROLE=eval-subdir (default: the role name)")
    ap.add_argument("--langs", nargs="*", default=DEFAULT_LANGS)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pylibs", default="/workspace/sft_build/pylibs")
    ap.add_argument("--power-from")
    ap.add_argument("--steps", type=int, default=49)
    ap.add_argument("--s1-steps", type=int, default=0)
    a = ap.parse_args(argv)
    a.sys = dict(x.split("=", 1) for x in a.sys)
    s = None
    if a.power_from:
        s = power_table(a.power_from, a.langs)
        if not a.eval_dir:
            return 0
    mde = (2.8 * (s or 0.30) * math.sqrt(2), 2.8 * (s or 0.30))
    res = build(a)
    dec = decide(res, mde[0])
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        json.dump({"results": res, "decision": dec}, open(os.path.join(a.out, "ab2_results.json"), "w"), indent=1, default=float)
        json.dump(dec, open(os.path.join(a.out, "decision.json"), "w"), indent=1, default=float)
        completions = a.steps * 64 * 16
        if not a.s1_steps:                                      # N of S1 as written by ab2_set_n.py into the S1 config
            import re
            cfgp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "ab2_sft_s1.yaml")
            m = re.search(r"^max_steps:\s*(\d+)", open(cfgp).read(), re.M) if os.path.exists(cfgp) else None
            a.s1_steps = int(m.group(1)) if m else 0
        json.dump(paper_json(res, a.runs_dir, a.steps, completions, a.s1_steps), open(os.path.join(a.out, "ablation_results.json"), "w"), indent=1, default=float)
        write_report(res, dec, os.path.join(a.out, "ab2_report.md"), a, mde)
        print("wrote", os.path.join(a.out, "ablation_results.json"), "ab2_results.json decision.json ab2_report.md")
    else:
        print(json.dumps(dec, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
