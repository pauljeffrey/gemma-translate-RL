#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""D-53 QE-metric choice study: data builders and reports (the scoring itself is qe_choice_score.py, the sampling qe_choice_sample.py).

    ssamte-build   SSA-MTE test split (McGill-NLP/SSA-MTE, files test/<Language>.json) -> rows.jsonl for qe_choice_score.py
    ssamte-report  Spearman / Pearson per language pair of every metric with the human z-scores (+ bootstrap CIs, paired differences)
    policy-report  within-prompt rank correlation of every metric with sentence chrF++ / reference-based COMET over policy samples
    hack-build     FLORES+ dev human-vs-corrupted pairs (mismatch / neighbour / copy / half-omitted / appended / replaced) -> rows.jsonl
    hack-report    AUC / same-source failure rate per language and metric

Metric columns are the files written by qe_choice_score.py (<scores-dir>/<name>.json, aligned with rows.jsonl).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

SSAMTE_FILES = {   # SSA-MTE config -> file stem; the 7 en->X pairs of the SSA-COMET paper's average come first
    "eng-amh": "Amharic", "eng-hau": "Hausa", "eng-kik": "Kikuyu", "eng-kin": "Kinyarwanda", "eng-luo": "Luo", "eng-twi": "Twi", "eng-yor": "Yoruba",
    "fra-ewe": "Ewe", "fra-wol": "Wolof", "por-vmw": "Emakhuwa", "por-nya": "Nyanja",
}
EN7 = ["eng-amh", "eng-hau", "eng-kik", "eng-kin", "eng-luo", "eng-twi", "eng-yor"]


# ------------------------------------------------------------------------------------------------ statistics
def _rank(x: Sequence[float]) -> List[float]:
    order = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def pearson(a: Sequence[float], b: Sequence[float]) -> float:
    n = len(a)
    if n < 3:
        return float("nan")
    ma, mb = sum(a) / n, sum(b) / n
    sa = math.sqrt(sum((x - ma) ** 2 for x in a))
    sb = math.sqrt(sum((x - mb) ** 2 for x in b))
    if sa == 0 or sb == 0:
        return float("nan")
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb)


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    return pearson(_rank(a), _rank(b))


def pct(xs: Sequence[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))]


def ci(xs: Sequence[float], lo=0.05, hi=0.95):
    xs = [x for x in xs if x == x]
    return (pct(xs, lo), pct(xs, hi)) if xs else (float("nan"), float("nan"))


def mean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


# ------------------------------------------------------------------------------------------------ helpers
def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def load_scores(d: str, names: Optional[Sequence[str]] = None) -> Dict[str, List[float]]:
    out = {}
    for f in sorted(os.listdir(d)):
        if f.endswith(".json"):
            n = f[:-5]
            if names is None or n in names:
                out[n] = json.load(open(os.path.join(d, f)))
    return out


def fmt(x, nd=3):
    return "nan" if x != x else f"{x:.{nd}f}"


# ------------------------------------------------------------------------------------------------ ssa-mte
def cmd_ssamte_build(a):
    rows = []
    for cfg, stem in SSAMTE_FILES.items():
        for i, l in enumerate(open(os.path.join(a.data_dir, "test", stem + ".json"), encoding="utf-8")):
            r = json.loads(l)
            rows.append({"lp": cfg, "i": i, "src": r["src_sent"], "mt": r["tgt_sent"], "ref": r["ref"], "z": float(r["z_score"]), "da": float(r["score"]),
                         "system": r["system"], "annotator": r["annotator"], "chrf": r.get("chrf")})
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} rows, {len({r['lp'] for r in rows})} language pairs -> {a.out}")


def cmd_afrimte_build(a):
    """AfriMTE-WMT2024 challenge set (masakhane/AfriMTE-WMT2024, data/all/test.jsonl; human z-scores): en->X pairs only."""
    rows = []
    for i, r in enumerate(read_jsonl(a.data)):
        rows.append({"lp": r["language_pair"], "i": i, "src": r["source"], "mt": r["hypothesis"], "ref": r["reference"], "z": float(r["score"]), "da": float(r["score"])})
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(len(rows), "rows ->", a.out)


def cmd_ssamte_report(a):
    global SSAMTE_FILES, EN7
    if a.pairs:
        SSAMTE_FILES = {p: None for p in a.pairs.split(",")}
        EN7 = a.avg.split(",") if a.avg else list(SSAMTE_FILES)
    rows = read_jsonl(a.rows)
    sc = load_scores(a.scores)
    add_ensembles(sc)                                  # z-pooled over ALL rows of this dataset (a fixed affine map per metric)
    names = list(sc)
    rng = random.Random(0)
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[r["lp"]].append(i)
    lines = []
    for target in ("z", "da"):
        for corr, cf in (("Spearman", spearman), ("Pearson", pearson)):
            if target == "da" and corr == "Pearson":
                continue
            lines.append(f"\n#### {corr} with the human {'z-score' if target == 'z' else 'raw DA score (0-100)'}\n")
            lines.append("| pair | n | " + " | ".join(names) + " |")
            lines.append("|---|---|" + "---|" * len(names))
            vals = {n: {} for n in names}
            for lp in list(SSAMTE_FILES):
                idx = by[lp]
                h = [rows[i][target] for i in idx]
                cells = []
                for n in names:
                    v = cf([sc[n][i] for i in idx], h)
                    vals[n][lp] = v
                    cells.append(fmt(v))
                lines.append(f"| {lp} | {len(idx)} | " + " | ".join(cells) + " |")
            for label, lps in ((f"**avg {len(EN7)} headline pairs**", EN7), (f"avg all {len(SSAMTE_FILES)} listed", list(SSAMTE_FILES))):
                lines.append(f"| {label} | | " + " | ".join(fmt(mean([vals[n][lp] for lp in lps])) for n in names) + " |")
    # bootstrap (resample rows within each pair): CI of the 7-pair average Spearman and paired differences vs the first metric
    ref_name = a.baseline if a.baseline in sc else names[0]
    boots = {n: [] for n in names}
    for _ in range(a.boot):
        per = {n: [] for n in names}
        for lp in EN7:
            idx = by[lp]
            samp = [idx[rng.randrange(len(idx))] for _ in idx]
            h = [rows[i]["z"] for i in samp]
            for n in names:
                per[n].append(spearman([sc[n][i] for i in samp], h))
        for n in names:
            boots[n].append(mean(per[n]))
    lines.append(f"\n#### Bootstrap (rows resampled within each pair, {a.boot} draws, 90% interval) of the headline-pair average Spearman (z-score); paired difference vs `{ref_name}`\n")
    lines.append("| metric | avg Spearman | 90% CI | diff vs " + ref_name + " | 90% CI of diff | P(diff>0) |")
    lines.append("|---|---|---|---|---|---|")
    for n in names:
        point = mean([spearman([sc[n][i] for i in by[lp]], [rows[i]["z"] for i in by[lp]]) for lp in EN7])
        lo, hi = ci(boots[n])
        d = [x - y for x, y in zip(boots[n], boots[ref_name])]
        dl, dh = ci(d)
        lines.append(f"| {n} | {fmt(point)} | [{fmt(lo)}, {fmt(hi)}] | {fmt(mean(d)) if n != ref_name else '-'} | "
                     f"{'[%s, %s]' % (fmt(dl), fmt(dh)) if n != ref_name else '-'} | {fmt(sum(1 for x in d if x > 0) / len(d), 2) if n != ref_name else '-'} |")
    # per-pair CIs of the differences ssa vs ours for each other metric
    lines.append(f"\n#### Per pair: Spearman difference (metric - `{ref_name}`), z-score, 90% bootstrap CI\n")
    others = [n for n in names if n != ref_name]
    lines.append("| pair | " + " | ".join(others) + " |")
    lines.append("|---|" + "---|" * len(others))
    for lp in list(SSAMTE_FILES):
        idx = by[lp]
        cells = []
        bs = {n: [] for n in others}
        for _ in range(a.boot):
            samp = [idx[rng.randrange(len(idx))] for _ in idx]
            h = [rows[i]["z"] for i in samp]
            base = spearman([sc[ref_name][i] for i in samp], h)
            for n in others:
                bs[n].append(spearman([sc[n][i] for i in samp], h) - base)
        for n in others:
            lo, hi = ci(bs[n])
            cells.append(f"{fmt(mean(bs[n]))} [{fmt(lo)},{fmt(hi)}]" + (" *" if lo > 0 or hi < 0 else ""))
        lines.append(f"| {lp} | " + " | ".join(cells) + " |")
    lines.append("\n(* = the 90% interval excludes 0)")
    # within-source ranking (GRPO-like): several translations of the SAME source by different MT systems; Spearman of metric vs human inside the group
    grp = defaultdict(list)
    for i, r in enumerate(rows):
        grp[(r["lp"], r["src"])].append(i)
    per_lp = defaultdict(list)
    for (lp, _), g in grp.items():
        h = [rows[i]["z"] for i in g]
        if len(g) < 3 or len(set(h)) < 2:
            continue
        d = {}
        for n in names:
            v = [sc[n][i] for i in g]
            d[n] = spearman(v, h) if len(set(v)) > 1 else float("nan")
        per_lp[lp].append(d)
    if per_lp:
        lines.append("\n#### Within-source ranking (translations of the same source by different MT systems, >= 3 per source): mean Spearman with the human z-score over sources\n")
        lines.append("| pair | groups | " + " | ".join(names) + " |")
        lines.append("|---|---|" + "---|" * len(names))
        pv = {n: {} for n in names}
        for lp in list(SSAMTE_FILES):
            if lp not in per_lp:
                continue
            cells = []
            for n in names:
                pv[n][lp] = mean([d[n] for d in per_lp[lp]])
                cells.append(fmt(pv[n][lp]))
            lines.append(f"| {lp} | {len(per_lp[lp])} | " + " | ".join(cells) + " |")
        hl = [lp for lp in EN7 if lp in per_lp]
        lines.append(f"| **avg {len(hl)} headline pairs** | | " + " | ".join(fmt(mean([pv[n][lp] for lp in hl])) for n in names) + " |")
        bs = {n: [] for n in names}
        for _ in range(a.boot):
            per = {n: [] for n in names}
            for lp in hl:
                ds = per_lp[lp]
                samp = [ds[rng.randrange(len(ds))] for _ in ds]
                for n in names:
                    per[n].append(mean([d[n] for d in samp]))
            for n in names:
                bs[n].append(mean(per[n]))
        lines.append("\n90% bootstrap CI (sources resampled) of the headline within-source average, and paired difference vs " + ref_name + ":\n")
        lines.append("| metric | within-source avg [CI] | diff vs " + ref_name + " [CI] |")
        lines.append("|---|---|---|")
        for n in names:
            lo, hi = ci(bs[n])
            dd = [x - y for x, y in zip(bs[n], bs[ref_name])]
            dl, dh = ci(dd)
            lines.append(f"| {n} | {fmt(mean([pv[n][lp] for lp in hl]))} [{fmt(lo)}, {fmt(hi)}] | " + ("-" if n == ref_name else f"{fmt(mean(dd))} [{fmt(dl)}, {fmt(dh)}]") + " |")
    text = "\n".join(lines)
    open(a.out, "w", encoding="utf-8").write(text)
    print(text)



# ------------------------------------------------------------------------------------------------ ensembles
def zpool(sc: Dict[str, List[float]], names: Sequence[str], mask: Optional[Sequence[bool]] = None):
    """(mean, sd) of every metric over the pooled rows (mask) -> the fixed affine map of the z-score ensemble."""
    out = {}
    for n in names:
        v = [x for i, x in enumerate(sc[n]) if mask is None or mask[i]]
        m = sum(v) / len(v)
        out[n] = (m, math.sqrt(sum((x - m) ** 2 for x in v) / len(v)) or 1.0)
    return out


def add_ensembles(sc: Dict[str, List[float]], mask=None) -> Dict[str, tuple]:
    """Adds ens_afri_ssaqe = z(africomet_qe_stl11) + z(ssa_qe), ens_afri_ssamtl = z(africomet_qe_stl11) + z(ssa_mtl_qe),
    ens3 = all three. z uses ONE (mean, sd) per metric over the pooled rows (not per language), so the ensemble is a fixed function of the raw scores."""
    base = [n for n in ("africomet_qe_stl11", "ssa_qe", "ssa_mtl_qe") if n in sc]
    zp = zpool(sc, base, mask)

    def z(n, i):
        return (sc[n][i] - zp[n][0]) / zp[n][1]

    N = len(sc[base[0]])
    for name, parts in (("ens_afri_ssaqe", ["africomet_qe_stl11", "ssa_qe"]), ("ens_afri_ssamtl", ["africomet_qe_stl11", "ssa_mtl_qe"]),
                        ("ens3", ["africomet_qe_stl11", "ssa_qe", "ssa_mtl_qe"])):
        if all(p in sc for p in parts):
            sc[name] = [sum(z(p, i) for p in parts) / len(parts) for i in range(N)]
    if "africomet_qe_stl11" in sc and "ssa_mtl_qe" in sc:     # conservative combination: a sample is only as good as its WORSE judge (blocks the copy-the-source hack of ssa_mtl_qe)
        sc["ens_min_afri_ssamtl"] = [min(z("africomet_qe_stl11", i), z("ssa_mtl_qe", i)) for i in range(N)]
    return zp


def add_shaped(sc, isos, anchors_path):
    """ens_shaped = mean of the two PER-LANGUAGE SHAPED rewards (reward_shaping curve, floor 0, preset repo) of africomet_qe_stl11 and ssa_mtl_qe, each with its
    own human-reference anchors (written by `anchors --dump`). This is the 'two calibrations, average the rewards' integration (no new constants)."""
    import numpy as np

    from gemma_mt.rl.reward_shaping import make_curve

    A = json.load(open(anchors_path))
    curves = {m: {iso: make_curve(v, floor=0.0) for iso, v in A[m].items()} for m in ("africomet_qe_stl11", "ssa_mtl_qe")}
    out = []
    for i, iso in enumerate(isos):
        r = []
        for m in ("africomet_qe_stl11", "ssa_mtl_qe"):
            xs, ys = curves[m][iso]
            r.append(float(np.interp(sc[m][i], xs, ys)))
        out.append(sum(r) / 2)
    sc["ens_shaped"] = out


# ------------------------------------------------------------------------------------------------ policy samples
def valid_sample(r: dict) -> bool:
    """Proxy of the reward's hard-failure classes that the metrics are NOT meant to rank: no end token, empty, wrong/missing tag, verbatim source copy."""
    return bool(r["finished"] and r["hyp"].strip() and r["prefix_status"] == "ok" and r["hyp"].strip() != r["src"].strip())


def cmd_policy_build(a):
    rows = read_jsonl(a.samples)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"src": r["src"], "mt": r["hyp"] if r["hyp"].strip() else ".", "ref": r["ref"]}, ensure_ascii=False) + "\n")
    print(len(rows), "rows ->", a.out)


def cmd_policy_report(a):
    from sacrebleu.metrics import CHRF

    from gemma_mt.rl import consensus as cons

    rows = read_jsonl(a.samples)
    sc = load_scores(a.scores)
    for n in list(sc):
        assert len(sc[n]) == len(rows), n
    valid = [valid_sample(r) for r in rows]
    add_ensembles(sc, valid)
    if a.anchors:
        add_shaped(sc, [r["iso"] for r in rows], a.anchors)
    chrf = CHRF(word_order=2)
    sc["_chrf"] = [chrf.sentence_score(r["hyp"], [r["ref"]]).score for r in rows]
    # consensus: mean chrF++ against the other valid siblings of the same prompt (min_valid 3), like the reward
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[(r["iso"], r["id"])].append(i)
    cval = [float("nan")] * len(rows)
    for g in groups.values():
        v = [i for i in g if valid[i]]
        profs = [cons.Profile(rows[i]["hyp"]) for i in v]
        vals = cons.group_consensus(profs, 3)
        for i, x in zip(v, vals):
            if x is not None:
                cval[i] = x
    sc["consensus"] = cval
    if "africomet_stl11_ref" in sc and "ssa_mtl_ref" in sc:     # family-neutral reference-based proxy: both reference-based COMETs, z-averaged with one pooled (mean, sd) each
        zr = zpool(sc, ["africomet_stl11_ref", "ssa_mtl_ref"], valid)
        sc["_mix"] = [sum((sc[k][i] - zr[k][0]) / zr[k][1] for k in zr) / 2 for i in range(len(rows))]
    shaping = json.load(open(a.shaping))
    mode = {k: (v.get("content") or {}).get("mode", "qe") for k, v in shaping.items() if isinstance(v, dict) and isinstance(v.get("content"), dict)}
    metrics = [m for m in ("africomet_qe_stl11", "ssa_qe", "ssa_mtl_qe", "ens_afri_ssaqe", "ens_afri_ssamtl", "ens_min_afri_ssamtl", "ens3", "ens_shaped", "consensus") if m in sc]
    targets = [("chrF++ vs reference", "_chrf")] + [(lab, k) for lab, k in (("AfriCOMET-STL-1.1 with reference", "africomet_stl11_ref"), ("SSA-COMET-MTL with reference", "ssa_mtl_ref"),
                                                         ("mean z of the two reference-based COMETs", "_mix")) if k in sc]
    langs = sorted({r["iso"] for r in rows})
    # per prompt rank correlations ------------------------------------------------------------------------------------------------
    rho = {(t, m): defaultdict(list) for _, t in targets for m in metrics}          # (target, metric) -> iso -> [rho per prompt]
    gain = {(t, m): defaultdict(list) for _, t in targets for m in metrics}         # selection gain, normalised by the oracle gain
    gabs = {(t, m): defaultdict(list) for _, t in targets for m in metrics + ["oracle"]}        # absolute gain of the top pick over the group mean (target units)
    for (iso, pid), g in groups.items():
        v = [i for i in g if valid[i]]
        if len(v) < 4:
            continue
        for _, t in targets:
            tv = [sc[t][i] for i in v]
            spread = max(tv) - min(tv)
            for m in metrics:
                mv = [sc[m][i] for i in v]
                if m == "consensus" and any(x != x for x in mv):
                    continue
                r = spearman(mv, tv) if len(set(mv)) > 1 and len(set(tv)) > 1 else float("nan")
                if r == r:
                    rho[(t, m)][iso].append(r)
                if spread > 1e-9:
                    j = max(range(len(v)), key=lambda q: (mv[q], -q))
                    gabs[(t, m)][iso].append(tv[j] - sum(tv) / len(tv))
                    if m == metrics[0]:
                        gabs[(t, "oracle")][iso].append(max(tv) - sum(tv) / len(tv))
                    gain[(t, m)][iso].append((tv[j] - sum(tv) / len(tv)) / ((max(tv) - sum(tv) / len(tv)) or 1.0))
    rng = random.Random(0)

    def lang_means(store, key, idxs=None):
        out = {}
        for iso in langs:
            xs = store[key].get(iso, [])
            if xs:
                out[iso] = xs
        return out

    def macro(store, key, group, resample=None):
        ms = []
        for iso in group:
            xs = store[key].get(iso)
            if not xs:
                continue
            if resample is not None:
                xs = [xs[resample[iso][q]] for q in range(len(xs))]
            ms.append(sum(xs) / len(xs))
        return sum(ms) / len(ms) if ms else float("nan")

    # bootstrap index draws per language (shared by all metrics -> paired); the length of each language's list can differ per target/metric, so draw by fraction
    def draw():
        return {iso: [rng.random() for _ in range(400)] for iso in langs}

    draws = [draw() for _ in range(a.boot)]

    def boot_macro(store, key, group):
        res = []
        for d in draws:
            ms = []
            for iso in group:
                xs = store[key].get(iso)
                if not xs:
                    continue
                n = len(xs)
                ms.append(sum(xs[int(u * n)] for u in d[iso][:n]) / n if n <= 400 else sum(xs[int(u * n)] for u in d[iso]) / 400)
            res.append(sum(ms) / len(ms) if ms else float("nan"))
        return res

    groupsets = [("all", langs)] + [(f"mode={m}", [l for l in langs if mode.get(l, "qe") == m]) for m in ("qe", "blend", "consensus")]
    base = "africomet_qe_stl11"
    out = [f"Policy samples: {len(rows)} (valid for ranking: {sum(valid)} = {sum(valid) / len(rows):.1%}), {len(groups)} prompt groups, {len(langs)} languages "
          f"({', '.join(langs)}).\n"]
    for lab, t in targets:
        for gname, gl in groupsets:
            if not gl:
                continue
            out.append(f"\n#### Within-prompt Spearman of each metric with **{lab}**, languages: {gname} ({len(gl)}: {' '.join(gl)})\n")
            out.append("| metric | mean over languages [90% CI] | diff vs " + base + " [90% CI] | languages better / worse than " + base + " | selection gain (share of oracle) [90% CI] |")
            out.append("|---|---|---|---|---|")
            bb = boot_macro(rho, (t, base), gl)
            for m in metrics:
                pt = macro(rho, (t, m), gl)
                b = boot_macro(rho, (t, m), gl)
                lo, hi = ci(b)
                if m != base:
                    d = [x - y for x, y in zip(b, bb)]
                    dl, dh = ci(d)
                    wins = sum(1 for iso in gl if rho[(t, m)].get(iso) and rho[(t, base)].get(iso) and mean(rho[(t, m)][iso]) > mean(rho[(t, base)][iso]))
                    loss = sum(1 for iso in gl if rho[(t, m)].get(iso) and rho[(t, base)].get(iso) and mean(rho[(t, m)][iso]) < mean(rho[(t, base)][iso]))
                    dtxt = f"{fmt(macro(rho, (t, m), gl) - macro(rho, (t, base), gl))} [{fmt(dl)}, {fmt(dh)}]"
                    wl = f"{wins} / {loss}"
                else:
                    dtxt, wl = "-", "-"
                gp = macro(gain, (t, m), gl)
                gb = boot_macro(gain, (t, m), gl)
                glo, ghi = ci(gb)
                out.append(f"| {m} | {fmt(pt)} [{fmt(lo)}, {fmt(hi)}] | {dtxt} | {wl} | {fmt(gp)} [{fmt(glo)}, {fmt(ghi)}] |")
    out.append("\n#### Absolute selection gain: target value of the sample each metric ranks first MINUS the group mean (all languages; chrF++ points / COMET points x 100), 90% CI over prompts\n")
    out.append("| metric | " + " | ".join(lab for lab, _ in targets) + " |")
    out.append("|---|" + "---|" * len(targets))
    scale = {"_chrf": 1.0, "africomet_stl11_ref": 100.0, "ssa_mtl_ref": 100.0, "_mix": 100.0}
    for m in metrics + ["oracle"]:
        cells = []
        for lab, t in targets:
            pt = macro(gabs, (t, m), langs) * scale[t]
            lo, hi = ci([x * scale[t] for x in boot_macro(gabs, (t, m), langs)])
            cells.append(f"{pt:.2f} [{lo:.2f}, {hi:.2f}]")
        out.append(f"| {m} | " + " | ".join(cells) + " |")
    # per language table for the chrF target
    out.append("\n#### Per language: within-prompt Spearman with chrF++ (and with AfriCOMET-STL-1.1 + reference), mean over prompts\n")
    tt = [t for _, t in targets]
    out.append("| lang | mode | prompts | " + " | ".join(f"{m} (chrF / refCOMET)" for m in metrics) + " |")
    out.append("|---|---|---|" + "---|" * len(metrics))
    for iso in langs:
        cells = []
        for m in metrics:
            c1 = mean(rho[("_chrf", m)].get(iso, []))
            c2 = mean(rho[("africomet_stl11_ref", m)].get(iso, [])) if "africomet_stl11_ref" in tt else float("nan")
            cells.append(f"{fmt(c1, 2)} / {fmt(c2, 2)}")
        out.append(f"| {iso} | {mode.get(iso, 'qe')} | {len(rho[('_chrf', base)].get(iso, []))} | " + " | ".join(cells) + " |")
    # pooled correlations
    out.append("\n#### Pooled Spearman over all valid samples (not within prompt; confounded by prompt difficulty and language)\n")
    out.append("| metric | pooled over all languages: chrF++ | AfriCOMET-STL-1.1 ref | mean of per-language pooled: chrF++ | refCOMET |")
    out.append("|---|---|---|---|---|")
    vi = [i for i, v in enumerate(valid) if v]
    for m in metrics:
        ok = [i for i in vi if sc[m][i] == sc[m][i]]
        row = [fmt(spearman([sc[m][i] for i in ok], [sc["_chrf"][i] for i in ok])),
               fmt(spearman([sc[m][i] for i in ok], [sc["africomet_stl11_ref"][i] for i in ok])) if "africomet_stl11_ref" in sc else "n/a"]
        per_c, per_r = [], []
        for iso in langs:
            ii = [i for i in ok if rows[i]["iso"] == iso]
            if len(ii) > 10:
                per_c.append(spearman([sc[m][i] for i in ii], [sc["_chrf"][i] for i in ii]))
                if "africomet_stl11_ref" in sc:
                    per_r.append(spearman([sc[m][i] for i in ii], [sc["africomet_stl11_ref"][i] for i in ii]))
        out.append(f"| {m} | {row[0]} | {row[1]} | {fmt(mean(per_c))} | {fmt(mean(per_r))} |")
    # metric inter-correlation inside groups (how redundant are they)
    out.append("\n#### Within-prompt Spearman BETWEEN metrics (mean over languages) - redundancy of the signals\n")
    mm = [m for m in metrics if m != "consensus"] + ["consensus"]
    out.append("| | " + " | ".join(mm) + " |")
    out.append("|---|" + "---|" * len(mm))
    for m1 in mm:
        cells = []
        for m2 in mm:
            per_lang = []
            for iso in langs:
                rs = []
                for (i2, pid), g in groups.items():
                    if i2 != iso:
                        continue
                    v = [i for i in g if valid[i]]
                    if len(v) < 4:
                        continue
                    x, y = [sc[m1][i] for i in v], [sc[m2][i] for i in v]
                    if any(q != q for q in x + y) or len(set(x)) < 2 or len(set(y)) < 2:
                        continue
                    rs.append(spearman(x, y))
                if rs:
                    per_lang.append(sum(rs) / len(rs))
            cells.append(fmt(mean(per_lang), 2))
        out.append(f"| {m1} | " + " | ".join(cells) + " |")
    text = "\n".join(out)
    open(a.out, "w", encoding="utf-8").write(text)
    json.dump({"zpool": zpool(sc, [m for m in ("africomet_qe_stl11", "ssa_qe", "ssa_mtl_qe") if m in sc], valid)}, open(a.out + ".z.json", "w"))
    print(text)



# ------------------------------------------------------------------------------------------------ hackability
HARD = ("mismatch", "neighbour", "drop_half", "append_unrelated", "replace_words")      # classes a reward must rank below the human reference
PARTIAL = ("mix_half_src", "append_source", "swap25_src")      # partial copying of the English source (not caught by the 0.9-Jaccard copy guard)
CLASSES = ("human", "mismatch", "neighbour", "copy_source", "copy_noisy", "drop_half", "shuffle", "append_unrelated", "replace_words") + PARTIAL


def corrupt_rows(items, rng):
    """items: [(src, ref)] of ONE language, in a fixed order. Returns rows {cls, j, src, mt}. Definitions follow the earlier study
    (pcm_calib_exp3.py / reports/ssa_comet_vs_africomet_2026-10-15.md): mismatch = the reference of another sentence (offset >= 7), neighbour = 1-3
    positions away, copy_source, copy_noisy = source with 20% of the words dropped, drop_half, shuffle, append_unrelated = reference + another
    reference, replace_words = 25% of the words replaced by words of other references."""
    n = len(items)
    vocab = [w for _, r in items for w in r.split()]
    out = []
    for j, (src, ref) in enumerate(items):
        words = ref.split()
        other = lambda off: items[(j + off) % n][1]
        out.append({"cls": "human", "j": j, "src": src, "mt": ref})
        out.append({"cls": "mismatch", "j": j, "src": src, "mt": other(rng.randrange(7, max(8, n - 6)))})
        out.append({"cls": "neighbour", "j": j, "src": src, "mt": other(rng.choice([-3, -2, -1, 1, 2, 3]))})
        out.append({"cls": "copy_source", "j": j, "src": src, "mt": src})
        sw = src.split()
        keep = [w for w in sw if rng.random() > 0.2] or sw[:1]
        out.append({"cls": "copy_noisy", "j": j, "src": src, "mt": " ".join(keep)})
        k = max(1, len(words) // 2)
        drop = set(rng.sample(range(len(words)), min(k, len(words) - 1))) if len(words) > 1 else set()
        out.append({"cls": "drop_half", "j": j, "src": src, "mt": " ".join(w for i, w in enumerate(words) if i not in drop) or ref})
        sh = words[:]
        rng.shuffle(sh)
        out.append({"cls": "shuffle", "j": j, "src": src, "mt": " ".join(sh)})
        out.append({"cls": "append_unrelated", "j": j, "src": src, "mt": ref + " " + other(rng.randrange(7, max(8, n - 6)))})
        rep = [rng.choice(vocab) if rng.random() < 0.25 else w for w in words]
        out.append({"cls": "replace_words", "j": j, "src": src, "mt": " ".join(rep)})
        h = len(words) // 2
        out.append({"cls": "mix_half_src", "j": j, "src": src, "mt": " ".join(words[:h] + sw[len(sw) // 2:]) or src})
        out.append({"cls": "append_source", "j": j, "src": src, "mt": ref + " " + src})
        out.append({"cls": "swap25_src", "j": j, "src": src, "mt": " ".join(rng.choice(sw) if rng.random() < 0.25 else w for w in words)})
    return out


def cmd_hack_build(a):
    rng = random.Random(1)
    seen = {}
    for r in read_jsonl(a.samples):
        seen.setdefault(r["iso"], {})[r["id"]] = (r["src"], r["ref"])
    rows = []
    for iso in sorted(seen):
        items = [seen[iso][i] for i in sorted(seen[iso])]
        for r in corrupt_rows(items, rng):
            rows.append({"iso": iso, **r, "ref": items[r["j"]][1]})
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(len(rows), "rows,", len(seen), "languages ->", a.out)


def cmd_hack_report(a):
    import numpy as np

    rows = read_jsonl(a.rows)
    sc = load_scores(a.scores)
    hum = [r["cls"] == "human" for r in rows]
    add_ensembles(sc, hum)
    if a.anchors:
        add_shaped(sc, [r["iso"] for r in rows], a.anchors)
    metrics = [m for m in ("africomet_qe_stl11", "ssa_qe", "ssa_mtl_qe", "ens_afri_ssaqe", "ens_afri_ssamtl", "ens_min_afri_ssamtl", "ens3", "ens_shaped") if m in sc]
    langs = sorted({r["iso"] for r in rows})
    idx = defaultdict(dict)                       # (iso, cls) -> {j: row index}
    for i, r in enumerate(rows):
        idx[(r["iso"], r["cls"])][r["j"]] = i
    base = "africomet_qe_stl11"
    shaping = json.load(open(a.shaping))
    mode = {k: (v.get("content") or {}).get("mode", "qe") for k, v in shaping.items() if isinstance(v, dict) and isinstance(v.get("content"), dict)}
    arr = {}                                      # (m, iso, cls) -> np array over sentences
    for m in metrics:
        for iso in langs:
            n = len(idx[(iso, "human")])
            for c in CLASSES:
                arr[(m, iso, c)] = np.array([sc[m][idx[(iso, c)][j]] for j in range(n)])
    rng = np.random.default_rng(0)
    B = a.boot
    draws = {iso: rng.integers(0, len(idx[(iso, "human")]), size=(B, len(idx[(iso, "human")]))) for iso in langs}

    def auc(h, mm):          # P(human > mismatched) over all pairs, ties 0.5 ; h, mm: (..., n)
        d = h[..., :, None] - mm[..., None, :]
        return ((d > 0).sum((-1, -2)) + 0.5 * (d == 0).sum((-1, -2))) / (d.shape[-1] * d.shape[-2])

    def fail(h, c):          # share of sentences whose corrupted version scores >= the human reference of the same sentence
        return (c >= h).mean(-1)

    res = {}
    for m in metrics:
        a_pt, a_bs, f_pt, f_bs, c_pt = {}, {}, {}, {}, {}
        for iso in langs:
            h, mm = arr[(m, iso, "human")], arr[(m, iso, "mismatch")]
            a_pt[iso] = float(auc(h, mm))
            a_bs[iso] = auc(h[draws[iso]], mm[draws[iso]])
            hardf = np.mean([fail(arr[(m, iso, "human")], arr[(m, iso, c)]) for c in HARD])
            f_pt[iso] = float(hardf)
            f_bs[iso] = np.mean([fail(h[draws[iso]], arr[(m, iso, c)][draws[iso]]) for c in HARD], axis=0)
            c_pt[iso] = {c: float(fail(h, arr[(m, iso, c)])) for c in CLASSES if c != "human"}
        res[m] = (a_pt, a_bs, f_pt, f_bs, c_pt)
    out = []
    groupsets = [("all", langs)] + [(f"mode={k}", [l for l in langs if mode.get(l, "qe") == k]) for k in ("qe", "blend", "consensus")]
    out.append(f"Corruption study on FLORES+ dev (MAFAND dev for pcm): {len(langs)} languages x {len(idx[(langs[0], 'human')])} sentences x {len(CLASSES)} classes "
               f"= {len(rows)} pairs per metric. Hard classes: {', '.join(HARD)}.\n")
    for gname, gl in groupsets:
        if not gl:
            continue
        out.append(f"\n#### Languages: {gname} ({len(gl)}: {' '.join(gl)}) - mean over languages, 90% bootstrap CI (sentences resampled), paired difference vs {base}\n")
        out.append("| metric | AUC human vs mismatch [CI] | diff AUC [CI] | hard-class failure rate [CI] | diff failure [CI] | copy_source fail | copy_noisy fail | partial-copy fail (mean of mix_half_src / append_source / swap25_src) | langs AUC better/worse | langs failure better/worse |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        ba = np.mean([res[base][1][l] for l in gl], axis=0)
        bf = np.mean([res[base][3][l] for l in gl], axis=0)
        for m in metrics:
            a_pt, a_bs, f_pt, f_bs, c_pt = res[m]
            apt = np.mean([a_pt[l] for l in gl]); abs_ = np.mean([a_bs[l] for l in gl], axis=0)
            fpt = np.mean([f_pt[l] for l in gl]); fbs = np.mean([f_bs[l] for l in gl], axis=0)
            alo, ahi = np.percentile(abs_, [5, 95]); flo, fhi = np.percentile(fbs, [5, 95])
            if m != base:
                da = abs_ - ba; df = fbs - bf
                dat = f"{apt - np.mean([res[base][0][l] for l in gl]):+.3f} [{np.percentile(da, 5):+.3f}, {np.percentile(da, 95):+.3f}]"
                dft = f"{fpt - np.mean([res[base][2][l] for l in gl]):+.3f} [{np.percentile(df, 5):+.3f}, {np.percentile(df, 95):+.3f}]"
                ab = sum(1 for l in gl if a_pt[l] > res[base][0][l]); aw = sum(1 for l in gl if a_pt[l] < res[base][0][l])
                fb = sum(1 for l in gl if f_pt[l] < res[base][2][l]); fw = sum(1 for l in gl if f_pt[l] > res[base][2][l])
                wl = f"{ab} / {aw}"; wl2 = f"{fb} / {fw}"
            else:
                dat = dft = "-"; wl = wl2 = "-"
            cc = {c: np.mean([c_pt[l][c] for l in gl]) for c in ("copy_source", "copy_noisy") + PARTIAL}
            out.append(f"| {m} | {apt:.3f} [{alo:.3f}, {ahi:.3f}] | {dat} | {fpt:.3f} [{flo:.3f}, {fhi:.3f}] | {dft} | {cc['copy_source']:.3f} | {cc['copy_noisy']:.3f} | "
                       f"{np.mean([cc[c] for c in PARTIAL]):.3f} ({' / '.join('%.2f' % cc[c] for c in PARTIAL)}) | {wl} | {wl2} |")
    out.append("\n#### Failure rate per class (mean over all languages; corrupted >= human of the same sentence; lower is better)\n")
    out.append("| metric | " + " | ".join(c for c in CLASSES if c != "human") + " |")
    out.append("|---|" + "---|" * (len(CLASSES) - 1))
    for m in metrics:
        out.append(f"| {m} | " + " | ".join(f"{np.mean([res[m][4][l][c] for l in langs]):.3f}" for c in CLASSES if c != "human") + " |")
    # residual after the reward's hard copy guard (guards.is_copy_of_source, Jaccard >= 0.9): only the uncaught rows can be exploited
    try:
        from gemma_mt.rl.guards import is_copy_of_source

        caught = {}
        for iso in langs:
            n = len(idx[(iso, "human")])
            for c in ("copy_source", "copy_noisy", "mix_half_src", "append_source", "swap25_src"):
                caught[(iso, c)] = np.array([is_copy_of_source(rows[idx[(iso, c)][j]]["src"], rows[idx[(iso, c)][j]]["mt"]) for j in range(n)])
        lid_caught = {}
        if a.lid:
            from gemma_mt.rl.lid import make_lid

            lidf = make_lid()
            for iso in langs:
                blk = (shaping.get(iso) or {}).get("lid") or {}
                ok = set(blk.get("labels") or []) | set(blk.get("extra_labels") or [])
                n = len(idx[(iso, "human")])
                for c in ("copy_source", "copy_noisy", "mix_half_src", "append_source", "swap25_src"):
                    texts = [rows[idx[(iso, c)][j]]["mt"] for j in range(n)]
                    tops = lidf(texts) if (blk.get("enabled") and ok) else ["" for _ in texts]
                    lid_caught[(iso, c)] = np.array([bool(blk.get("enabled") and ok and len(t) >= int(blk.get("min_chars", 30)) and top not in ok) for t, top in zip(texts, tops)])
            n_en = sum(1 for iso in langs if (shaping.get(iso) or {}).get("lid", {}).get("enabled"))
            out.append(f"\nGlotLID guard (top-1 label not among the language's accepted labels, only where enabled in the calibration: {n_en} of {len(langs)} languages here) is applied as well in the table below ('+LID').\n")
            for k in list(caught):
                caught[k] = caught[k] | lid_caught[k]
        glot = " + GlotLID" if a.lid else ""
        out.append(f"\n#### Copy-type hacks that survive the reward's hard copy guard (Jaccard >= 0.9 on normalised word sets{glot}): share caught by the guard, and failure rate of the UNCAUGHT rows counted over all rows (= corrupted >= human AND not caught)\n")
        cls5 = ("copy_source", "copy_noisy", "mix_half_src", "append_source", "swap25_src")
        out.append("| metric | " + " | ".join(f"{c} (caught {np.mean([caught[(l, c)].mean() for l in langs]):.2f})" for c in cls5) + " |")
        out.append("|---|" + "---|" * len(cls5))
        for m in metrics:
            cells = []
            for c in cls5:
                v = np.mean([((arr[(m, l, c)] >= arr[(m, l, "human")]) & ~caught[(l, c)]).mean() for l in langs])
                cells.append(f"{v:.3f}")
            out.append(f"| {m} | " + " | ".join(cells) + " |")
    except ImportError:
        pass
    out.append("\n#### Per language: AUC human vs mismatch / hard-class failure rate\n")
    out.append("| lang | mode | " + " | ".join(metrics) + " |")
    out.append("|---|---|" + "---|" * len(metrics))
    for l in langs:
        out.append(f"| {l} | {mode.get(l, 'qe')} | " + " | ".join(f"{res[m][0][l]:.2f} / {res[m][2][l]:.2f}" for m in metrics) + " |")
    out.append("\n#### Score scale of the human references (pooled P10 / P50 / P90 and share below 0.46; raw metric units, ensembles in z units)\n")
    out.append("| metric | P10 | P50 | P90 | share < 0.46 |")
    out.append("|---|---|---|---|---|")
    for m in metrics:
        v = np.concatenate([arr[(m, l, "human")] for l in langs])
        out.append(f"| {m} | {np.percentile(v, 10):.3f} | {np.percentile(v, 50):.3f} | {np.percentile(v, 90):.3f} | {(v < 0.46).mean():.3f} |")
    text = "\n".join(out)
    open(a.out, "w", encoding="utf-8").write(text)
    print(text)


def cmd_anchors(a):
    """Per-language shaping anchors (t_fail = P90 of QE on mismatched references, t_mid = P50 and t_great = P90 of QE on human references; floor 0)
    for every metric from the hack set, next to the v7 AfriCOMET calibration (n = 997) - how far a switch moves the reward curve."""
    import numpy as np

    rows = read_jsonl(a.rows)
    sc = load_scores(a.scores)
    add_ensembles(sc, [r["cls"] == "human" for r in rows])
    shaping = json.load(open(a.shaping))
    langs = sorted({r["iso"] for r in rows})
    metrics = [m for m in ("africomet_qe_stl11", "ssa_qe", "ssa_mtl_qe") if m in sc]
    out = ["| metric | median t_fail | median t_mid | median t_great | median (t_great - t_fail) | languages with t_fail > t_mid - 0.05 |", "|---|---|---|---|---|---|"]
    anchors = {m: {} for m in metrics}
    for m in metrics:
        for iso in langs:
            h = np.array([sc[m][i] for i, r in enumerate(rows) if r["iso"] == iso and r["cls"] == "human"])
            mm = np.array([sc[m][i] for i, r in enumerate(rows) if r["iso"] == iso and r["cls"] == "mismatch"])
            anchors[m][iso] = (float(np.percentile(mm, 90)), float(np.percentile(h, 50)), float(np.percentile(h, 90)))
        A = np.array(list(anchors[m].values()))
        out.append(f"| {m} | {np.median(A[:, 0]):.3f} | {np.median(A[:, 1]):.3f} | {np.median(A[:, 2]):.3f} | {np.median(A[:, 2] - A[:, 0]):.3f} | {int((A[:, 0] > A[:, 1] - 0.05).sum())} of {len(A)} |")
    v7 = np.array([shaping[i]["anchors"] for i in langs if i in shaping and "anchors" in shaping[i]])
    if len(v7):
        out.append(f"| v7 calibration (AfriCOMET, n=997, {len(v7)} languages) | {np.median(v7[:, 0]):.3f} | {np.median(v7[:, 1]):.3f} | {np.median(v7[:, 2]):.3f} | {np.median(v7[:, 2] - v7[:, 0]):.3f} | - |")
    both = [i for i in langs if i in shaping and "anchors" in shaping[i]]
    if both:
        d = np.array([[anchors["africomet_qe_stl11"][i][k] - shaping[i]["anchors"][k] for k in range(3)] for i in both])
        out.append(f"\nSanity: my n=80 AfriCOMET anchors minus the v7 n=997 anchors, median |difference| (t_fail, t_mid, t_great) = {np.median(np.abs(d), axis=0).round(3).tolist()}.")
        for m in ("ssa_mtl_qe", "ssa_qe"):
            if m in anchors:
                d2 = np.array([[anchors[m][i][k] - shaping[i]["anchors"][k] for k in range(3)] for i in both])
                out.append(f"{m} anchors minus v7 AfriCOMET anchors: median shift (t_fail, t_mid, t_great) = {np.median(d2, axis=0).round(3).tolist()}, "
                           f"median |shift| = {np.median(np.abs(d2), axis=0).round(3).tolist()}, share of languages whose t_mid moves by more than 0.05: {float((np.abs(d2[:, 1]) > 0.05).mean()):.2f}")
    if a.dump:
        json.dump({m: {iso: list(v) for iso, v in anchors[m].items()} for m in anchors}, open(a.dump, "w"))
        # z constants of the human references (pooled over languages) for a z-average ensemble
        zc = zpool(sc, [m for m in ("africomet_qe_stl11", "ssa_qe", "ssa_mtl_qe") if m in sc], [r["cls"] == "human" for r in rows])
        out.append("\nPooled (mean, sd) of the human-reference scores, the constants of the z-average ensemble: " + json.dumps({k: [round(x, 4) for x in v] for k, v in zc.items()}))
    text = "\n".join(out)
    open(a.out, "w", encoding="utf-8").write(text)
    print(text)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ssamte-build"); p.add_argument("--data-dir", required=True); p.add_argument("--out", required=True); p.set_defaults(f=cmd_ssamte_build)
    p = sub.add_parser("ssamte-report"); p.add_argument("--rows", required=True); p.add_argument("--scores", required=True); p.add_argument("--out", required=True)
    p.add_argument("--baseline", default="africomet_qe_stl11"); p.add_argument("--boot", type=int, default=1000)
    p.add_argument("--pairs", default=None, help="comma list of pairs to report (default: the 11 SSA-MTE test pairs)"); p.add_argument("--avg", default=None, help="pairs of the headline average")
    p.set_defaults(f=cmd_ssamte_report)
    p = sub.add_parser("afrimte-build"); p.add_argument("--data", required=True); p.add_argument("--out", required=True); p.set_defaults(f=cmd_afrimte_build)
    p = sub.add_parser("policy-build"); p.add_argument("--samples", required=True); p.add_argument("--out", required=True); p.set_defaults(f=cmd_policy_build)
    p = sub.add_parser("policy-report"); p.add_argument("--samples", required=True); p.add_argument("--scores", required=True); p.add_argument("--out", required=True)
    p.add_argument("--shaping", default="/workspace/sft_build/gemma_mt_2c/calibration/shaping_flores_dev_v7.json"); p.add_argument("--boot", type=int, default=500)
    p.add_argument("--anchors", default=None, help="anchors JSON from `anchors --dump` (adds the ens_shaped metric)")
    p.set_defaults(f=cmd_policy_report)
    p = sub.add_parser("hack-build"); p.add_argument("--samples", required=True); p.add_argument("--out", required=True); p.set_defaults(f=cmd_hack_build)
    p = sub.add_parser("hack-report"); p.add_argument("--rows", required=True); p.add_argument("--scores", required=True); p.add_argument("--out", required=True)
    p.add_argument("--shaping", default="/workspace/sft_build/gemma_mt_2c/calibration/shaping_flores_dev_v7.json"); p.add_argument("--boot", type=int, default=400)
    p.add_argument("--anchors", default=None, help="anchors JSON from `anchors --dump` (adds the ens_shaped metric)")
    p.add_argument("--lid", action="store_true", help="also apply GlotLID (rl/lid.py, downloads cis-lmu/glotlid) as in the reward")
    p.set_defaults(f=cmd_hack_report)
    p = sub.add_parser("anchors"); p.add_argument("--rows", required=True); p.add_argument("--scores", required=True); p.add_argument("--out", required=True)
    p.add_argument("--shaping", default="/workspace/sft_build/gemma_mt_2c/calibration/shaping_flores_dev_v7.json"); p.add_argument("--dump", default=None)
    p.set_defaults(f=cmd_anchors)
    for extra in _EXTRA:
        extra(sub)
    a = ap.parse_args(argv)
    a.f(a)
    return 0


_EXTRA: list = []

if __name__ == "__main__":
    sys.exit(main())
