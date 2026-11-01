# SPDX-License-Identifier: AGPL-3.0-or-later
"""Measure how well the chunk planner notices DROPPED and EXTRA sentences, on real parallel documents (D-30; finding F2 of the reward report).

    python -m gemma_mt.rl.omission_eval --data calibration/flores_dev_all.jsonl --languages yor hau ibo swh zul amh kin ewe \\
        --variants legacy new --tilings 6

Data: the jsonl of `scripts/build_flores_dev_all.py` (FLORES+ DEV, English -> language, rows {iso, id, english, reference}).
Documents are windows of 3-12 consecutive sentence pairs (random tilings of the id-ordered dev set). Tests, per document:

  untouched   the human translation as it is                                            -> false alarm = any omission-type flag
  drop        k random sentences removed from the target side, k = 1 .. max(1, n // 2)   -> sentence recall / precision, document detection
  extra       1-2 sentences of another document (target language) appended to the target -> detection, and false omission flags

"Omission-type flag" = the plan has an omission bead or a shortfall bead (chunk_plan: `n_failed` / `n_short` / `flagged_src`). Only documents whose
source and human target split into the SAME number of sentences (n >= 3) are used for the drop / extra tests (the dropped sentence is then
known); the false-alarm rate is measured on ALL documents, natural sentence-count differences of the human translation included.
Length statistics (ratio, s2, p05, p95) of each language come from the same dev sentences (2-4 numbers per language; not tuned per document).

Variants (`--variants`): legacy | prior0.1 (raised omission/extra priors) | short0.7 (shortfall check only) | dp4 (language-aware Gale-Church, omission/extra
cost 4) | dp4+short0.7 | dpcal (per-language cost from `choose_omission_cost`, other tilings than the evaluation).
`--calibrate-omission shaping.json --languages ...` writes the per-language block `omission` {cost, false_alarm, ...} into a calibration file.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from gemma_mt.rl import segments
from gemma_mt.rl.chunk_plan import plan_chunks
from gemma_mt.rl.segments import LengthModel, split_sentences


def length_stats(src: Sequence[str], ref: Sequence[str]) -> Dict[str, float]:
    """Human length statistics of one language (characters): corpus ratio, Gale-Church variance per source character, per-sentence quantiles."""
    ls = [len(s) for s in src]
    lr = [len(r) for r in ref]
    c = sum(lr) / max(sum(ls), 1)
    s2 = sum((o - c * s) ** 2 / max(s, 1) for s, o in zip(ls, lr)) / max(len(ls), 1)
    ratios = sorted(o / max(s, 1) for s, o in zip(ls, lr))
    q = lambda p: ratios[int(p * (len(ratios) - 1))]
    return {"corpus": c, "s2": s2, "p01": q(0.01), "p02": q(0.02), "p05": q(0.05), "p50": q(0.5), "p95": q(0.95)}


def make_docs(rows: Sequence[dict], rng: random.Random, lo: int = 3, hi: int = 12) -> List[List[dict]]:
    """One random tiling of the id-ordered rows into windows of lo..hi consecutive sentences."""
    docs, pos = [], rng.randint(0, 5)
    while pos + lo <= len(rows):
        n = rng.randint(lo, hi)
        d = rows[pos: pos + n]
        if len(d) >= lo:
            docs.append(d)
        pos += n
    return docs


VARIANTS = {
    # name: (description, factory(stats) -> plan_chunks kwargs, segment priors patch)
    "legacy": ("legacy Gale-Church (observed ratio, priors 0.01)", lambda st: {}, None),
}


def _plan(src: str, out: str, kw: dict, w_hi: Optional[float]):
    return plan_chunks(src, out, w_hi=w_hi, **kw)


def omission_flags(plan) -> List[int]:
    return sorted(set(plan.flagged_src))


def evaluate_language(rows: List[dict], variants: Dict[str, Tuple[str, callable, Optional[dict]]], tilings: int = 6, seed: int = 0,
                      max_docs: int = 0) -> Dict[str, Dict[str, float]]:
    rows = sorted(rows, key=lambda r: r["id"])
    st = length_stats([r["english"] for r in rows], [r["reference"] for r in rows])
    w_hi = min(st["p95"], 2.5)
    rng = random.Random(seed)
    docs: List[List[dict]] = []
    for t in range(tilings):
        docs += make_docs(rows, random.Random(seed * 1000 + t))
    if max_docs:
        docs = docs[:max_docs]
    # pool of sentences from other places of the file, for the "extra" test
    out: Dict[str, Dict[str, float]] = {}
    for name, (desc, factory, patch) in variants.items():
        if name == "dpcal":
            c = choose_omission_cost(rows)["cost"]
            kw = {"length_model": model_for(st, omit=c, extra=4.0), "short_frac": 0.7, "short_min": 0.15}
            name = f"dpcal"
        else:
            kw = factory(st)
        saved = dict(segments._BEADS)
        if patch:
            segments._BEADS.update(patch)
        try:
            agg = defaultdict(float)
            rng = random.Random(seed + 17)
            for di, d in enumerate(docs):
                src = " ".join(r["english"] for r in d)
                ref = " ".join(r["reference"] for r in d)
                ss, oo = split_sentences(src), split_sentences(ref)
                p0 = _plan(src, ref, kw, w_hi)
                agg["n_docs"] += 1
                agg["fa_omit"] += bool(p0.n_failed or p0.n_short or p0.flagged_src)
                agg["fa_any"] += any(k is None for k, _ in p0.items)
                agg["fa_nocount"] += bool(len(ss) != len(oo))
                agg["fa_short"] += bool(p0.n_short)
                for tag in set(p0.extra):
                    agg[f"fatag_{tag}"] += 1
                agg["fa_omit_bead"] += bool(p0.n_failed)
                if len(ss) != len(oo) or len(ss) < 3:
                    continue
                n = len(ss)
                agg["n_clean"] += 1
                # ---- drop k sentences
                k = rng.randint(1, max(1, n // 2))
                dropped = sorted(rng.sample(range(n), k))
                kept = [oo[i] for i in range(n) if i not in dropped]
                p1 = _plan(src, " ".join(kept), kw, w_hi)
                fl = set(omission_flags(p1))
                hit = fl & set(dropped)
                agg["drop_docs"] += 1
                agg["drop_sent"] += k
                agg["drop_hit"] += len(hit)
                agg["drop_doc_any"] += bool(hit)
                agg["drop_doc_all"] += len(hit) == k
                agg["flag_total"] += len(fl)
                agg["flag_correct"] += len(hit)
                drop_chars = sum(len(ss[i]) for i in dropped)
                failed_w = sum(w for kk, w in p1.items if kk is None)
                agg["drop_char_cov"] += min(failed_w, drop_chars) / max(drop_chars, 1)
                agg["drop_fail_w_over_chars"] += failed_w / max(drop_chars, 1)
                # k == 1 separately (the hardest case for a length model)
                if k == 1:
                    agg["drop1_docs"] += 1
                    agg["drop1_hit"] += bool(hit)
                # ---- almost everything dropped: only one sentence is translated (the "whole" case: one output sentence for several source sentences)
                if n >= 4:
                    keep = rng.randrange(n)
                    p3 = _plan(src, oo[keep], kw, w_hi)
                    agg["keep1_docs"] += 1
                    agg["keep1_det"] += bool(p3.n_failed or p3.n_short or p3.flagged_src)
                    agg["keep1_failw_over_chars"] += sum(w for kk, w in p3.items if kk is None) / max(sum(len(ss[i]) for i in range(n) if i != keep), 1)
                # ---- extra sentences appended / inserted
                e = rng.randint(1, 2)
                for _ in range(20):
                    other = rng.choice(rows)
                    if abs(other["id"] - d[0]["id"]) > 30:
                        break
                osent = split_sentences(other["reference"])
                extra_s = (osent * 2)[:e]
                if extra_s:
                    for mode in ("append", "insert"):
                        if mode == "append":
                            o2 = oo + extra_s
                        else:
                            pos = rng.randint(1, max(1, n - 1))
                            o2 = oo[:pos] + extra_s + oo[pos:]
                        p2 = _plan(src, " ".join(o2), kw, w_hi)
                        det = any(kk is None for kk, _ in p2.items)
                        agg[f"extra_{mode}_docs"] += 1
                        agg[f"extra_{mode}_det"] += det
                        agg[f"extra_{mode}_falseomit"] += bool(omission_flags(p2))
        finally:
            segments._BEADS.clear()
            segments._BEADS.update(saved)
        nd, nc = max(agg["n_docs"], 1), max(agg["n_clean"], 1)
        out[name] = {
            "docs": agg["n_docs"], "clean_docs": agg["n_clean"], "natural_count_mismatch": agg["fa_nocount"] / nd,
            "false_alarm_omit": agg["fa_omit"] / nd, "false_alarm_any": agg["fa_any"] / nd,
            "false_alarm_shortfall": agg["fa_short"] / nd,
            **{k.replace("fatag_", "false_alarm_tag_"): v / nd for k, v in agg.items() if k.startswith("fatag_")}, "false_alarm_omit_bead": agg["fa_omit_bead"] / nd,
            "keep1_detect": agg["keep1_det"] / max(agg["keep1_docs"], 1), "keep1_failw_over_dropped_chars": agg["keep1_failw_over_chars"] / max(agg["keep1_docs"], 1),
            "drop_sentence_recall": agg["drop_hit"] / max(agg["drop_sent"], 1),
            "drop_sentence_precision": agg["flag_correct"] / max(agg["flag_total"], 1),
            "drop_doc_any": agg["drop_doc_any"] / max(agg["drop_docs"], 1),
            "drop_doc_all": agg["drop_doc_all"] / max(agg["drop_docs"], 1),
            "drop1_recall": agg["drop1_hit"] / max(agg["drop1_docs"], 1),
            "drop_char_coverage": agg["drop_char_cov"] / max(agg["drop_docs"], 1),
            "drop_failw_over_chars": agg["drop_fail_w_over_chars"] / max(agg["drop_docs"], 1),
            "extra_append_det": agg["extra_append_docs"] and agg["extra_append_det"] / agg["extra_append_docs"],
            "extra_insert_det": agg["extra_insert_docs"] and agg["extra_insert_det"] / agg["extra_insert_docs"],
            "extra_append_falseomit": agg["extra_append_docs"] and agg["extra_append_falseomit"] / agg["extra_append_docs"],
            "extra_insert_falseomit": agg["extra_insert_docs"] and agg["extra_insert_falseomit"] / agg["extra_insert_docs"],
        }
    return out


OMISSION_GRID = (4.0, 5.0, 6.0, 8.0, 12.0)


def choose_omission_cost(rows: List[dict], target: float = 0.02, grid: Sequence[float] = OMISSION_GRID, tilings: int = 4,
                         seed: int = 101) -> Dict[str, object]:
    """Per-language operating point of the omission detector (D-30): the SMALLEST omission cost (the extra-sentence cost stays 4) in `grid` for which the UNTOUCHED human
    documents of the language are flagged in at most `target` of the cases (false alarms). Where even the largest cost cannot reach the target
    (Ewe: the FLORES+ references themselves leave sentences out) the largest cost is kept and `met_target` is False. Pure CPU, ~20 s per language."""
    rows = sorted(rows, key=lambda r: r["id"])
    st = length_stats([r["english"] for r in rows], [r["reference"] for r in rows])
    w_hi = min(st["p95"], 2.5)
    docs: List[List[dict]] = []
    for t in range(tilings):
        docs += make_docs(rows, random.Random(seed * 1000 + t))
    texts = []
    for d in docs:
        texts.append((" ".join(r["english"] for r in d), " ".join(r["reference"] for r in d)))
    natural = sum(1 for src, ref in texts if len(split_sentences(src)) != len(split_sentences(ref))) / max(len(texts), 1)
    fa: Dict[str, float] = {}
    for c in grid:
        kw = {"length_model": LengthModel(ratio=st["corpus"], s2=st["s2"], omit_cost=c, extra_cost=4.0), "short_frac": 0.7, "short_min": 0.15}
        n = 0
        for src, ref in texts:
            p = _plan(src, ref, kw, w_hi)
            n += bool(p.n_failed or p.n_short or p.flagged_src)
        fa[str(c)] = n / max(len(texts), 1)
    out = select_cost(fa, target, grid)
    out.update({"false_alarm_by_cost": fa, "natural_count_mismatch": natural, "docs": len(texts)})
    return out


def select_cost(fa: Dict[str, float], target: float = 0.02, grid: Sequence[float] = OMISSION_GRID, hopeless: float = 0.05) -> Dict[str, object]:
    """The smallest cost whose false-alarm rate (untouched human documents) is <= target. If none reaches it: when even the largest cost still flags
    more than `hopeless` of the documents, the references themselves leave sentences out (Ewe) - a higher cost would only blunt the detector, so the
    smallest cost is kept; otherwise (fon: 2.1%) the largest cost is the best available."""
    chosen = next((c for c in grid if fa[str(c)] <= target), None)
    met = chosen is not None
    if not met:
        chosen = grid[0] if fa[str(grid[-1])] > hopeless else grid[-1]
    return {"cost": float(chosen), "met_target": met, "target": target, "false_alarm": fa[str(chosen)],
            "note": "" if met else ("target not reached at any cost; references themselves inconsistent, smallest cost kept" if chosen == grid[0]
                                    else "target not reached; largest cost used")}


def load(path: str, languages: Optional[Iterable[str]] = None) -> Dict[str, List[dict]]:
    by: Dict[str, List[dict]] = defaultdict(list)
    keep = set(languages) if languages else None
    with open(path, encoding="utf-8") as f:
        for l in f:
            r = json.loads(l)
            if keep is None or r["iso"] in keep:
                by[r["iso"]].append(r)
    for iso in list(by):                                     # MAFAND dev rows are inventory-only helpers where FLORES+ exists; ids are not document order
        fl = [r for r in by[iso] if r.get("benchmark") != "mafand_dev"]
        if fl:
            by[iso] = fl
    return by


def model_for(st: Dict[str, float], omit: float = 4.0, extra: float = 4.0, use_s2: bool = True, dp: bool = True,
              directional: bool = True, s2_mult: float = 1.0) -> LengthModel:
    return LengthModel(ratio=st["corpus"], s2=(st["s2"] if use_s2 else 6.8) * s2_mult, omit_cost=omit, extra_cost=extra, dp=dp,
                       directional=directional)


def build_variants(spec: Sequence[str]) -> Dict[str, tuple]:
    """spec items: legacy | dp<omit>[,<extra>] | dpg<omit> (DP, gaussian 6.8) | short<frac>[:<min>] (shortfall check only, legacy alignment) |
    dp<omit>+short<frac>[:<min>] (both) | prior<omit_prior> (legacy cost, raised omission/extra priors)"""
    v: Dict[str, tuple] = {}
    for item in spec:
        if item == "legacy":
            v[item] = ("legacy", lambda st: {}, None)
        elif item == "dpcal":                                   # per-language cost from the calibration step (choose_omission_cost on other tilings)
            def fac_cal(st, _rows=None):
                raise RuntimeError("dpcal is resolved per language in evaluate_language")
            v[item] = (item, fac_cal, None)
        elif item.startswith("prior"):
            p = float(item[5:])
            v[item] = (item, lambda st: {}, {(1, 0): p, (0, 1): p})
        else:
            parts = item.split("+")
            dp_c = None
            sh = None
            opts = {}
            for part in parts:
                if part.startswith("dp"):
                    body = part[2:]
                    nd = body.endswith("n")
                    if nd:
                        body = body[:-1]
                    mult = 1.0
                    if "s" in body:
                        body, m_ = body.split("s")
                        mult = float(m_)
                    dp_c = [float(x) for x in body.split(",")]
                    opts = {"directional": not nd, "s2_mult": mult}
                elif part.startswith("short"):
                    a = part[5:].split(":")
                    sh = (float(a[0]), float(a[1]) if len(a) > 1 else 0.15)

            def fac(st, dp_c=dp_c, sh=sh, opts=opts):
                m = model_for(st, omit=dp_c[0] if dp_c else 4.0, extra=(dp_c[1] if dp_c and len(dp_c) > 1 else (dp_c[0] if dp_c else 4.0)),
                              dp=dp_c is not None, **opts)
                kw: dict = {"length_model": m}
                if sh:
                    kw["short_frac"], kw["short_min"] = sh
                return kw
            v[item] = (item, fac, None)
    return v


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--languages", nargs="*", default=["yor", "hau", "ibo", "swh", "zul", "amh", "kin", "ewe"])
    ap.add_argument("--variants", nargs="+", default=["legacy", "dp4+short0.7"])
    ap.add_argument("--tilings", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-docs", type=int, default=0)
    ap.add_argument("--calibrate-omission", default=None, metavar="SHAPING_JSON",
                    help="instead of the evaluation: choose the omission cost of every language of --languages (smallest cost with <= --target false alarms "
                         "on its untouched dev documents) and write it as the block `omission` into this calibration file (rewritten in place)")
    ap.add_argument("--target", type=float, default=0.02)
    a = ap.parse_args(argv)
    if a.calibrate_omission:
        with open(a.calibrate_omission, encoding="utf-8") as f:
            cal = json.load(f)
        data = load(a.data, a.languages)
        for iso in a.languages:
            if iso not in data or iso not in cal or any(r.get("benchmark") == "mafand_dev" for r in data[iso]):
                print("skip", iso, "(no FLORES+ document order)", file=sys.stderr)
                continue
            cal[iso]["omission"] = choose_omission_cost(data[iso], a.target)
            o = cal[iso]["omission"]
            print(f"{iso:4} cost {o['cost']:4.1f} false_alarm {o['false_alarm']:.3f} met={o['met_target']} natural_count_mismatch {o['natural_count_mismatch']:.2f} "
                  f"by cost {o['false_alarm_by_cost']}", flush=True)
        with open(a.calibrate_omission, "w", encoding="utf-8") as f:
            json.dump(cal, f, indent=1)
        return 0
    data = load(a.data, a.languages)
    variants = build_variants(a.variants)
    res: Dict[str, Dict[str, Dict[str, float]]] = {}
    for iso in a.languages:
        if iso not in data:
            print("no data for", iso, file=sys.stderr)
            continue
        res[iso] = evaluate_language(data[iso], variants, a.tilings, a.seed, a.max_docs)
    keys = ["false_alarm_omit", "false_alarm_shortfall", "false_alarm_any", "keep1_detect", "drop_sentence_recall", "drop_sentence_precision", "drop_doc_any", "drop1_recall",
            "drop_char_coverage", "extra_append_det", "extra_insert_det", "extra_append_falseomit", "extra_insert_falseomit"]
    for name in variants:
        print(f"\n== {name}")
        print("lang  docs  clean " + " ".join(f"{k[:14]:>14}" for k in keys))
        tot = defaultdict(list)
        for iso, r in res.items():
            m = r[name]
            print(f"{iso:5} {int(m['docs']):5} {int(m['clean_docs']):5} " + " ".join(f"{m[k]:14.3f}" for k in keys))
            for k in keys:
                tot[k].append(m[k])
        print("macro            " + " ".join(f"{statistics.mean(tot[k]):14.3f}" for k in keys))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
