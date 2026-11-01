# SPDX-License-Identifier: AGPL-3.0-or-later
"""D-65 evidence: does the number of RL samples a language saw explain its SFT -> RL chrF++ gain? (observational; read-only on finished-run files)

Inputs
  --draws   lang_draws_by_phase.json  (scripts/replay_schedule_draws.py: prompts per language per RL phase, re-derived from the stateless PromptSchedule)
  --wide    reports/data/final_model_per_language_wide.csv  (SFT-final vs RL-final chrF++ per language x benchmark x subset, paired-bootstrap CI of the delta)
  --calib   calibration/shaping_flores_dev_v8.json (content mode qe / blend / consensus, anchors t_fail / t_mid / t_great)
  --metrics runs/rl2c/metrics.jsonl (per-language dev chrF++ trajectory `bench/chrf/<L>`, greedy hold-out reward `eval/reward/<L>`, `reward/consensus/chrf/<L>`)
Outputs (in --out): per_language.csv, regression.json, saturation.csv, noise.json  + a short text summary on stdout.

Pure numpy (no scipy / pandas) so that it runs on the laptop and on the box.
    PYTHONPATH=gemma_mt python gemma_mt/scripts/lang_weight_evidence.py --draws ... --wide ... --calib ... --metrics ... --out reports/data/adaptive_language_weights_2026-10-22
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

STRONG = ["afr", "swh", "zul", "xho", "sot", "som", "ibo"]          # the owner's "strong languages with small gains"
W_FINAL = {"tir": 2.5, "amh": 2.5, "kbp": 2.5, "nus": 2.5, "kam": 2.5, "bam": 2.5, "mos": 2.5, "umb": 2.5, "kmb": 2.5, "dyu": 2.5, "kea": 2.5, "luo": 2.5, "kik": 2.5, "orm": 2.5,
           "fon": 2.5, "yor": 2.5, "vmw": 2.5, "efi": 2.5}
RNG = np.random.default_rng(20261022)


# ------------------------------------------------------------------------------------------------------------------------------ data
def load_draws(path: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    d = json.load(open(path))["summary"]
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for ph, s in d.items():
        for iso, n in s["prompts"].items():
            out.setdefault(iso, {})[ph] = {"prompts": n, "completions": n * s["group"]}
    return out


def load_gain_rows(path: str, languages: Sequence[str]):
    rows = []
    for r in csv.DictReader(open(path)):
        if r["subset"] != "no_pair" or r["iso"] not in languages:
            continue
        lo, hi = float(r["d_chrf_lo"]), float(r["d_chrf_hi"])
        rows.append({"iso": r["iso"], "bench": r["benchmark"], "n": int(r["n"]), "sft": float(r["sft_chrf"]), "rl": float(r["rl_chrf"]),
                     "d": float(r["d_chrf"]), "se": max((hi - lo) / 3.92, 0.05)})
    return rows


def script_of(iso: str) -> str:
    from gemma_mt.languages import LANGUAGES

    L = LANGUAGES[iso]
    code = L.flores or (L.glotlid[0] if L.glotlid else "")
    m = re.search(r"_([A-Z][a-z]{3})", code or "")
    return m.group(1) if m else "Latn"


def language_table(draws, rows, calib) -> List[dict]:
    by = defaultdict(list)
    for r in rows:
        by[r["iso"]].append(r)
    table = []
    for iso, rs in sorted(by.items()):
        w = np.array([1.0 / r["se"] ** 2 for r in rs])
        gain = float(np.sum(w * [r["d"] for r in rs]) / w.sum())
        se = float(1.0 / math.sqrt(w.sum()))
        base = float(np.sum(w * [r["sft"] for r in rs]) / w.sum())
        ph = draws.get(iso, {})
        c2c = ph.get("2c", {}).get("completions", 0)
        cum = c2c + ph.get("2", {}).get("completions", 0) + ph.get("2b_b64", {}).get("completions", 0)
        cal = calib.get(iso, {})
        mode = (cal.get("content") or {}).get("mode", "qe")
        a = cal.get("anchors") or [None, None, None]
        table.append({"iso": iso, "mode": mode, "script": script_of(iso), "w2c": W_FINAL.get(iso, 1.0), "n_bench": len(rs), "sft": base, "gain": gain, "se": se,
                      "c_2c": c2c, "c_cum": cum, "p_2c": ph.get("2c", {}).get("prompts", 0), "t_fail": a[0], "t_mid": a[1], "t_great": a[2],
                      "benches": ",".join(sorted(r["bench"] for r in rs))})
    return table


# ------------------------------------------------------------------------------------------------------------------------------ regression
def design(table: List[dict], spec: str):
    X, names = [], []
    cols = {"const": lambda t: 1.0, "sft10": lambda t: t["sft"] / 10.0, "sft10sq": lambda t: (t["sft"] / 10.0) ** 2, "blend": lambda t: float(t["mode"] == "blend"),
            "cons": lambda t: float(t["mode"] == "consensus"), "nonlatin": lambda t: float(t["script"] != "Latn"),
            "log2_c2c": lambda t: math.log2(max(t["c_2c"], 1)), "log2_cum": lambda t: math.log2(max(t["c_cum"], 1)), "w25": lambda t: float(t["w2c"] > 1.0)}
    for c in spec.split("+"):
        names.append(c)
    for t in table:
        X.append([cols[c](t) for c in names])
    return np.array(X), names


def ols(X, y, w=None):
    if w is None:
        w = np.ones(len(y))
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    return beta


def boot(table: List[dict], spec: str, weighted: bool, B: int = 4000):
    X, names = design(table, spec)
    y = np.array([t["gain"] for t in table])
    w = np.array([1.0 / (t["se"] ** 2 + 1.0) for t in table]) if weighted else None      # +1.0 = between-language noise floor (tau^2 ~ 1 chrF^2)
    b0 = ols(X, y, w)
    n = len(y)
    bs = np.empty((B, len(names)))
    for i in range(B):
        ix = RNG.integers(0, n, n)
        try:
            bs[i] = ols(X[ix], y[ix], None if w is None else w[ix])
        except np.linalg.LinAlgError:
            bs[i] = np.nan
    lo, hi = np.nanpercentile(bs, [2.5, 97.5], axis=0)
    res = X @ b0
    r2 = 1 - float(np.sum((y - res) ** 2) / np.sum((y - y.mean()) ** 2))
    return {"spec": spec, "weighted": weighted, "n": n, "r2": r2, "coef": {nm: {"est": float(b0[i]), "lo": float(lo[i]), "hi": float(hi[i])} for i, nm in enumerate(names)}}


def vif_note(table: List[dict]):
    lc = np.array([math.log2(max(t["c_2c"], 1)) for t in table])
    w = np.array([float(t["w2c"] > 1.0) for t in table])
    r = float(np.corrcoef(lc, w)[0, 1])
    within = {k: float(np.std(lc[w == k])) for k in (0.0, 1.0) if np.sum(w == k) > 2}
    return {"corr_log2_samples_vs_weight_class": r, "sd_log2_samples_within_class": within, "sd_log2_samples_total": float(np.std(lc)),
            "mean_log2_gap_between_classes": float(lc[w == 1].mean() - lc[w == 0].mean()) if (w == 1).any() and (w == 0).any() else None}


def row_level(rows, table, B: int = 3000):
    """Row-level model: d(language, benchmark) ~ benchmark FE + sft level + mode + nonlatin + log2 samples, language-cluster bootstrap, each language weighted 1/its rows."""
    info = {t["iso"]: t for t in table}
    benches = sorted({r["bench"] for r in rows})
    isos = sorted({r["iso"] for r in rows})

    def build(sel_isos):
        X, y, w = [], [], []
        for iso in sel_isos:
            rs = [r for r in rows if r["iso"] == iso]
            for r in rs:
                t = info[iso]
                X.append([1.0 if r["bench"] == b else 0.0 for b in benches] + [r["sft"] / 10.0, float(t["mode"] == "blend"), float(t["mode"] == "cons"), float(t["script"] != "Latn"),
                                                                              math.log2(max(t["c_2c"], 1))])
                y.append(r["d"])
                w.append(1.0 / len(rs))
        return np.array(X), np.array(y), np.array(w)

    names = benches + ["sft10", "blend", "cons", "nonlatin", "log2_c2c"]
    for i, nm in enumerate(names):
        if nm == "cons":
            names[i] = "consensus"
    X, y, w = build(isos)
    keep = [j for j in range(X.shape[1]) if np.any(X[:, j] != 0)]
    b0 = ols(X[:, keep], y, w)
    bs = []
    for _ in range(B):
        sel = [isos[i] for i in RNG.integers(0, len(isos), len(isos))]
        Xb, yb, wb = build(sel)
        try:
            bs.append(ols(Xb[:, keep], yb, wb))
        except np.linalg.LinAlgError:
            pass
    bs = np.array(bs)
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    return {"rows": len(y), "languages": len(isos), "coef": {names[j]: {"est": float(b0[i]), "lo": float(lo[i]), "hi": float(hi[i])} for i, j in enumerate(keep)}}


def stratum_compare(table: List[dict], thr: float = 22.0, B: int = 5000):
    """Model-free look at the confounded comparison: among languages with a LOW SFT baseline (< thr chrF++), the 2.5x-weight ones (2.4x the samples) vs the 1.0-weight ones. Difference of mean gains,
    bootstrap CI over languages; same for the mean-matched sub-stratum (nearest baseline)."""
    lo = [t for t in table if t["sft"] < thr]
    a = np.array([t["gain"] for t in lo if t["w2c"] > 1.0])
    b = np.array([t["gain"] for t in lo if t["w2c"] <= 1.0])
    if len(a) < 3 or len(b) < 3:
        return None
    d = float(a.mean() - b.mean())
    bs = [float(RNG.choice(a, len(a)).mean() - RNG.choice(b, len(b)).mean()) for _ in range(B)]
    return {"threshold": thr, "n_weight_2.5": int(len(a)), "n_weight_1.0": int(len(b)), "mean_gain_2.5": float(a.mean()), "mean_gain_1.0": float(b.mean()), "diff": d,
            "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "mean_sft_2.5": float(np.mean([t["sft"] for t in lo if t["w2c"] > 1.0])),
            "mean_sft_1.0": float(np.mean([t["sft"] for t in lo if t["w2c"] <= 1.0])), "samples_ratio": float(np.mean([t["c_2c"] for t in lo if t["w2c"] > 1.0]) / np.mean([t["c_2c"] for t in lo if t["w2c"] <= 1.0]))}


# ------------------------------------------------------------------------------------------------------------------------------ trajectories (saturation)
def trajectories(path: str):
    bench: Dict[str, Dict[int, float]] = defaultdict(dict)
    ev: Dict[str, Dict[int, float]] = defaultdict(dict)
    cons: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    base = {}
    for line in open(path):
        r = json.loads(line)
        s = r["step"]
        for k, v in r.items():
            m = re.match(r"bench/chrf/([a-z]{3})$", k)
            if m:
                bench[m.group(1)][s] = v
            m = re.match(r"eval/reward/([a-z]{3})$", k)
            if m:
                ev[m.group(1)][s] = v
            m = re.match(r"reward/consensus/chrf/([a-z]{3})$", k)
            if m and "reward" in r:
                cons[m.group(1)].append((s, v))
    return bench, ev, cons, base


def plateau_stats(series: Dict[int, float]):
    """Smooth with a 3-point mean; first third vs last third of the evaluations; step by which 80 % of the total (last-3 - first-3) change was reached."""
    st = sorted(series)
    v = np.array([series[s] for s in st])
    if len(v) < 9:
        return None
    sm = np.convolve(v, np.ones(3) / 3, mode="valid")
    tot = sm[-1] - sm[0]
    t80 = None
    if abs(tot) > 1e-9:
        frac = (sm - sm[0]) / tot
        for i, f in enumerate(frac):
            if f >= 0.8:
                t80 = st[i + 1]
                break
    third = max(len(v) // 3, 1)
    return {"first": float(v[:3].mean()), "last": float(v[-3:].mean()), "change": float(v[-3:].mean() - v[:3].mean()), "slope_first_third": float((v[third - 1] - v[0]) / max(st[third - 1] - st[0], 1) * 500),
            "slope_last_third": float((v[-1] - v[-third]) / max(st[-1] - st[-third], 1) * 500), "t80": t80}


# ------------------------------------------------------------------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", required=True)
    ap.add_argument("--wide", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    calib = json.load(open(a.calib))
    draws = load_draws(a.draws)
    rl_langs = [k for k in calib if len(k) == 3]
    rows = load_gain_rows(a.wide, rl_langs)
    table = language_table(draws, rows, calib)
    with open(os.path.join(a.out, "per_language.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    print(f"{len(table)} RL languages with human-reference rows; {len(rows)} (language, benchmark) rows")
    out: Dict[str, object] = {"n_languages": len(table), "collinearity": vif_note(table)}
    specs = {"A_baseline": "const+sft10", "B_baseline_mode": "const+sft10+blend+cons", "C_full_2c": "const+sft10+blend+cons+nonlatin+log2_c2c",
             "D_full_cum": "const+sft10+blend+cons+nonlatin+log2_cum", "E_weightclass": "const+sft10+blend+cons+nonlatin+w25",
             "F_quadratic_2c": "const+sft10+sft10sq+blend+cons+nonlatin+log2_c2c"}
    out["language_level"] = {k: {"ols": boot(table, s, False), "wls": boot(table, s, True)} for k, s in specs.items()}
    out["row_level"] = row_level(rows, table)
    # dose-response WITHIN a weight class (no weight confound): languages drawn at weight 1.0 only, and the 2.5x class only
    within = {}
    for cls, name in ((1.0, "weight_1.0"), (2.5, "weight_2.5")):
        sub = [t for t in table if t["w2c"] == cls]
        if len(sub) > 8:
            within[name] = {"n": len(sub), "log2_c2c": boot(sub, "const+sft10+blend+cons+log2_c2c", False, B=2000)["coef"]["log2_c2c"],
                            "sd_log2_samples": float(np.std([math.log2(max(t["c_2c"], 1)) for t in sub]))}
    out["within_class_dose"] = within
    out["low_baseline_stratum"] = {str(thr): stratum_compare(table, thr) for thr in (22.0, 26.0)}
    # strong languages: residual of the baseline-only model
    X, names = design(table, "const+sft10")
    y = np.array([t["gain"] for t in table])
    b = ols(X, y)
    resid = {t["iso"]: float(y[i] - X[i] @ b) for i, t in enumerate(table)}
    out["strong_residuals_baseline_model"] = {k: resid.get(k) for k in STRONG}
    # trajectories
    bench, ev, cons, base = trajectories(a.metrics)
    sat = []
    for iso in sorted(set(bench) | set(ev) | set(cons)):
        t = next((x for x in table if x["iso"] == iso), None)
        row = {"iso": iso, "strong": iso in STRONG, "mode": t["mode"] if t else "", "gain_rl_vs_sft": t["gain"] if t else None, "sft_chrf": t["sft"] if t else None,
               "t_great_reward": 2.0}
        p = plateau_stats(bench[iso]) if iso in bench else None
        if p:
            row.update({f"dev_{k}": v for k, v in p.items()})
        if iso in ev:
            vals = np.array([ev[iso][s] for s in sorted(ev[iso])])
            row.update({"eval_n_points": len(vals), "eval_reward_mean": float(vals.mean()), "eval_reward_first10": float(vals[:10].mean()), "eval_reward_last10": float(vals[-10:].mean()),
                        "eval_reward_frac_at_floor": float(np.mean(vals <= -1.99))})
        if iso in cons and len(cons[iso]) > 40:
            cv = np.array([v for _, v in sorted(cons[iso])])
            q = len(cv) // 4
            row.update({"consensus_chrf_first_q": float(cv[:q].mean()), "consensus_chrf_last_q": float(cv[-q:].mean()), "consensus_n_steps": len(cv)})
        sat.append(row)
    keys = sorted({k for r in sat for k in r})
    with open(os.path.join(a.out, "saturation.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(sat)
    # noise of a corpus chrF++ DELTA (paired) per sentence count: se * sqrt(n) from the paired bootstrap CIs of the wide csv
    sds = [r["se"] * math.sqrt(r["n"]) for r in rows if r["n"] >= 300]
    out["paired_delta_sd_per_sentence"] = {"median": float(np.median(sds)), "p10": float(np.percentile(sds, 10)), "p90": float(np.percentile(sds, 90)), "n_cells": len(sds)}
    sd = float(np.median(sds))
    out["mde_chrf_paired_80pct_power"] = {str(n): round(2.8 * sd / math.sqrt(n), 2) for n in (40, 60, 100, 200, 300, 843, 1012, 1997)}
    json.dump(out, open(os.path.join(a.out, "regression.json"), "w"), indent=1)
    # stdout summary
    print(json.dumps(out["collinearity"], indent=1))
    for k, v in out["language_level"].items():
        cf = v["ols"]["coef"]
        ks = [c for c in cf if c.startswith("log2") or c == "w25"]
        print(k, "R2 %.2f" % v["ols"]["r2"], {c: "%.2f [%.2f, %.2f]" % (cf[c]["est"], cf[c]["lo"], cf[c]["hi"]) for c in ks + ["sft10"] if c in cf})
    print("row-level", {c: "%.2f [%.2f, %.2f]" % (v["est"], v["lo"], v["hi"]) for c, v in out["row_level"]["coef"].items() if c in ("log2_c2c", "sft10", "blend", "consensus", "nonlatin")})
    print("within class", json.dumps(within, indent=0)[:600])
    print("low-baseline stratum", json.dumps(out["low_baseline_stratum"]))
    print("strong residuals", {k: (None if v is None else round(v, 2)) for k, v in out["strong_residuals_baseline_model"].items()})
    print("noise", out["paired_delta_sd_per_sentence"], out["mde_chrf_paired_80pct_power"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
