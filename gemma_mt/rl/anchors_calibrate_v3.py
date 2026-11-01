# SPDX-License-Identifier: AGPL-3.0-or-later
"""Domain-robust calibration of the v2 detectors for languages whose only human-style text is religious (Bible / JW) or very small (D-68, calibration/anchors_v3.json).

Why: the v2 'pivot_held' calibration of urh sid wal bas dua yao (and wes oke bbj bin ish iso swc) fitted tight thresholds, and a bigram table, on the SAME religious domain; a policy output in another
domain (science, news, web) then fell outside them and the stuffed-output gate failed 26-63 % of the samples of those languages in the live run (steps 1-123). The recipe here
  * fits every threshold per SUB-DOMAIN (set) and takes the loosest one (lowest low-tail threshold, highest high-tail threshold) instead of pooling,
  * moves the tails outwards by a margin `lam` (tail distances) chosen on languages that DO have out-of-domain human text (scripts/v3_domain_study.py),
  * builds the function-word dictionary from words that are frequent in EVERY sub-domain (minimum document frequency, not the geometric mean with one domain),
  * drops the bigram-order feature (the table is religious text) unless it is validated out of domain; padding is then the type / token ratio only,
  * learns a loose loanword / shared-with-English whitelist for the copy detector (>= 2 occurrences, >= 30 % kept, >= 3 letters) and raises the copy threshold floor,
  * never lets a language rebuilt here reach the hard gate (`gate_ok` false; `gate_candidate` records what the leave-one-set-out numbers would allow) and writes `usable` per detector: false = keep the detector off for that language.
Pure functions; numpy only.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from gemma_mt.rl import anchors_calibrate_v2 as C
from gemma_mt.rl import anchors_v2 as V
from gemma_mt.rl import textstats as T

MIN_DOMAIN_ROWS = 25
LOOSE_WL = dict(min_count=2, min_rate=0.3, min_len=3)


def _combine(per_domain: List[List[Optional[float]]], low_tail: bool) -> List[Optional[float]]:
    nb = len(V.V2_LENGTH_EDGES) + 1
    out: List[Optional[float]] = []
    for k in range(nb):
        vals = [d[k] for d in per_domain if d and d[k] is not None]
        out.append((min(vals) if low_tail else max(vals)) if vals else None)
    return out


def domain_rows(rows_by_dom: Dict[str, np.ndarray], bucket_col: int, val_col: int, q: float, low_tail: bool, lam: float) -> List[Optional[float]]:
    per = []
    for name, r in rows_by_dom.items():
        if len(r) >= MIN_DOMAIN_ROWS:
            per.append(C.bucket_thresholds(r[:, bucket_col], r[:, val_col], q, low_tail, min_n=MIN_DOMAIN_ROWS // 2, lam=lam))
    return _combine(per, low_tail)


def robust_fw(train_top: Sequence[Sequence[Any]], n_seg: int, texts_by_dom: Dict[str, Sequence[str]], k: int, min_df: float = 0.004) -> List[str]:
    """The `k` words that are frequent in the training text AND in every sub-domain: ranked by the MINIMUM document frequency over the domains."""
    dfs = []
    for name, texts in texts_by_dom.items():
        if len(texts) < MIN_DOMAIN_ROWS:
            continue
        c: Counter = Counter()
        for t in texts:
            c.update(set(T.out_tokens(t)))
        dfs.append({w: v / len(texts) for w, v in c.items()})
    sc = []
    for w, df, _ in train_top:
        d = [df / max(n_seg, 1)] + [m.get(w, 0.0) for m in dfs]
        if min(d) >= min_df:
            sc.append((min(d), w))
    sc.sort(reverse=True)
    return [w for _, w in sc[:k]]


def fit_func_v3(units_by_dom: Dict[str, Sequence[Dict[str, str]]], fw: Sequence[str], ex: FrozenSet[str], lam: float, alpha: float = C.FUNC_ALPHA) -> Optional[Dict[str, Any]]:
    fws = frozenset(fw)
    rows = {n: C.func_rows(us, fws, ex) for n, us in units_by_dom.items()}
    if not any(len(r) >= MIN_DOMAIN_ROWS for r in rows.values()):
        return None
    return {"t_fw": domain_rows(rows, 5, 1, alpha, True, lam), "t_en": domain_rows(rows, 5, 2, alpha, False, lam), "t_ratio": domain_rows(rows, 5, 3, alpha, True, lam),
            "t_fw_hi": domain_rows(rows, 5, 1, alpha, False, lam), "cap_en": 0.3, "cap_fw": 0.95, "fw": list(fw), "en_excl": sorted(ex), "k": len(fw), "lam": lam, "alpha": alpha,
            "n": int(sum(len(r) for r in rows.values())), "domains": {n: int(len(r)) for n, r in rows.items()}}


def fit_pad_v3(units_by_dom: Dict[str, Sequence[Dict[str, str]]], lam: float, alpha: float = C.FUNC_ALPHA) -> Optional[Dict[str, Any]]:
    rows = {n: C.pad_rows(us, None, ttr_mode="seg100") for n, us in units_by_dom.items()}
    if not any(len(r) >= MIN_DOMAIN_ROWS for r in rows.values()):
        return None
    return {"ttr_mode": "seg100", "has_bigrams": False, "t_gn": [None] * (len(V.V2_LENGTH_EDGES) + 1), "t_ttr": domain_rows(rows, 3, 2, alpha, True, lam), "lam": lam, "alpha": alpha,
            "n": int(sum(len(r) for r in rows.values())), "domains": {n: int(len(r)) for n, r in rows.items()}}


def fit_copy_v3(units: Sequence[Dict[str, str]], k: int = 3, m: int = 20, alpha: float = 0.002, floor: float = 0.15, scale: float = 1.25) -> Optional[Dict[str, Any]]:
    """Loose whitelist, tail quantile `alpha`, threshold = max(floor, quantile) x scale. The whitelist is learned on ALL units given (every sub-domain)."""
    if len(units) < 40:
        return None
    wl_list = C.learn_whitelist(units, **LOOSE_WL)
    wl = frozenset(wl_list)
    tab = C.copy_table(units, wl, k, m)
    ok = ~np.isnan(tab[:, 0])
    if ok.sum() < 30:
        return None
    tw = max(floor, C._thr(tab[:, 0], alpha, 0.0)) * scale
    tc = max(floor, C._thr(tab[:, 1], alpha, 0.0)) * scale if m else 9.0
    tw, tc = min(tw, 1.0), min(tc, 1.0)
    return {"k": k, "m": m, "min_unprot": 2, "t_w": float(tw), "t_c": float(tc), "whitelist": wl_list, "n": int(ok.sum()), "alpha": alpha, "floor": floor, "scale": scale,
            "enabled": bool(tw <= C.COPY_T_MAX)}


# ------------------------------------------------------------------------------------------------ measurement
def pseudo(units: Sequence[Dict[str, str]], ks=(4, 10)) -> List[Dict[str, str]]:
    """Non-overlapping windows of k consecutive units of one set."""
    by: Dict[str, List[Dict[str, str]]] = {}
    for u in units:
        by.setdefault(u.get("set", ""), []).append(u)
    out = []
    for s, us in by.items():
        for k in ks:
            for i in range(0, len(us) - k + 1, k):
                w = us[i: i + k]
                out.append({"src": " ".join(x["src"] for x in w), "ref": " ".join(x["ref"] for x in w), "set": f"{s}/w{k}", "dom": w[0].get("dom", "")})
    return out


def fp_report(blocks: Dict[str, Optional[Dict[str, Any]]], units: Sequence[Dict[str, str]], bigrams: Optional[np.ndarray] = None) -> Dict[str, Dict[str, float]]:
    """Share of `units` with a deficit > 0 and >= 0.9 (the gate) per detector, with the n judged. `blocks` = {"copy","func","pad"} calibration blocks (None = detector absent)."""
    cal = {"pooled": {}, "languages": {"xx": {k: dict(b, enabled=True) for k, b in blocks.items() if b}}}
    A = V.Anchors2(cal)
    A._bigrams["xx"] = bigrams                      # the table of the language, when the pad block uses it
    res: Dict[str, List[float]] = {"copy": [], "func": [], "pad": []}
    for u in units:
        if blocks.get("copy"):
            f = A.copy_fraction("xx", u["src"], u["ref"])
            if f is not None:
                res["copy"].append(1.0 if f > 0 else 0.0)
        sw, ow = T.out_tokens(u["src"]), T.out_tokens(u["ref"])
        if blocks.get("func"):
            d = A.func_deficit("xx", sw, ow)
            if d is not None:
                res["func"].append(d)
        if blocks.get("pad"):
            d = A.pad_deficit("xx", ow)
            if d is not None:
                res["pad"].append(d)
    out = {}
    for k, v in res.items():
        if v:
            a = np.asarray(v)
            out[k] = {"n": len(a), "fp": float((a > 0).mean()), "sat": float((a >= 0.9).mean())}
    return out


def binom_upper(k: int, n: int, conf: float = 0.95) -> float:
    """Clopper-Pearson upper bound of a rate with k events in n trials (what n sentences can resolve: 0 of 100 is only '<= 3.0 %')."""
    from math import comb

    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
    lo, hi = k / n, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        cdf = sum(comb(n, i) * mid ** i * (1 - mid) ** (n - i) for i in range(k + 1))
        if cdf > 1 - conf:
            lo = mid
        else:
            hi = mid
    return hi


# ------------------------------------------------------------------------------------------------ one language, all sets
MIN_SET_ROWS = 40
FP_TARGET = 0.03          # leave-one-set-out false-positive target per detector
SAT_TARGET = 0.01         # share at the gate level (deficit >= 0.9) allowed for gate_ok
LAM_GRID = (0.0, 1.0, 2.0, 3.0, 4.0)
MIN_HELDOUT = 100


def _domains(units_by_set: Dict[str, Sequence[Dict[str, str]]]) -> Dict[str, List[Dict[str, str]]]:
    dom: Dict[str, List[Dict[str, str]]] = {}
    for n, us in units_by_set.items():
        if len(us) < 8:
            continue
        dom[n] = list(us)
        for k, nm in ((4, "w4"), (10, "w10")):
            p = pseudo(us, (k,))
            if len(p) >= 8:
                dom[f"{n}/{nm}"] = p
    return dom


def fit_language_v3(units_by_set: Dict[str, Sequence[Dict[str, str]]], funcs: Dict[str, Any], lam: float, copy_floor: float = 0.15, copy_scale: float = 1.25, kfw: int = 60) -> Dict[str, Any]:
    """The v3 blocks of one language from ALL its sets (each set is a domain). Returns {"copy","func","pad"} (None where nothing can be fitted)."""
    all_units = [u for us in units_by_set.values() for u in us]
    dom = _domains(units_by_set)
    fw = robust_fw(funcs["top"], funcs["n_seg"], {n: [u["ref"] for u in us] for n, us in units_by_set.items()}, kfw)
    ex = frozenset(C.english_excl(funcs["en_tf"], funcs["n_tok"], fw))
    return {"copy": fit_copy_v3(all_units, floor=copy_floor, scale=copy_scale), "func": fit_func_v3(dom, fw, ex, lam) if len(fw) >= 15 else None, "pad": fit_pad_v3(dom, lam)}


def lodo_report(units_by_set: Dict[str, Sequence[Dict[str, str]]], funcs: Dict[str, Any], lam: float, copy_floor: float, copy_scale: float) -> Dict[str, Any]:
    """Leave-one-SET-out: thresholds from the other sets, false-positive shares on the held-out one (sentences + 10-sentence windows). Only sets with >= MIN_SET_ROWS rows are held out."""
    out: Dict[str, Any] = {}
    sets = [n for n, us in units_by_set.items() if len(us) >= MIN_SET_ROWS]
    if len(sets) < 2:
        return out
    for held in sets:
        rest = {n: us for n, us in units_by_set.items() if n != held}
        if sum(len(us) for us in rest.values()) < 2 * MIN_SET_ROWS:
            continue
        b = fit_language_v3(rest, funcs, lam, copy_floor, copy_scale)
        ev = list(units_by_set[held]) + pseudo(units_by_set[held], (10,))
        out[held] = fp_report({k: v for k, v in b.items() if v}, ev)
    return out


def decide(lodo: Dict[str, Any], ood_sets: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """usable / gate_ok per detector from the leave-one-set-out numbers: usable needs max FP <= FP_TARGET over the held-out sets, gate_ok needs max saturated share <= SAT_TARGET AND at least
    one held-out set that is out of domain (`ood_sets`)."""
    res: Dict[str, Dict[str, Any]] = {}
    for det in ("copy", "func", "pad"):
        fps = [(s, r[det]["fp"], r[det]["sat"], r[det]["n"]) for s, r in lodo.items() if det in r and r[det]["n"] >= MIN_HELDOUT]       # a held-out set of < 100 sentences can neither prove nor refute a 3 % rate: it is fitted, not judged
        if not fps:
            res[det] = {"n_heldout_sets": 0, "usable": None, "gate_ok": False}
            continue
        # a small held-out set cannot show a rate below ~3 %: n < 200 is judged by the 95 % Clopper-Pearson UPPER bound (0 of 100 -> 2.95 %, 1 of 100 -> 4.7 %), larger sets by the observed share
        mx = max((binom_upper(int(round(f * n)), n) if n < 200 else f) for _, f, _, n in fps)
        sat = max((binom_upper(int(round(st * n)), n) if n < 200 else st) for _, _, st, n in fps)
        has_ood = any(s in ood_sets for s, _, _, _ in fps)
        res[det] = {"n_heldout_sets": len(fps), "max_fp": mx, "mean_fp": float(np.mean([f for _, f, _, _ in fps])), "max_sat": sat, "has_ood_heldout": has_ood,
                    "per_set": {s: {"fp": f, "sat": st, "n": n} for s, f, st, n in fps}, "usable": bool(mx <= FP_TARGET), "gate_ok": bool(mx <= FP_TARGET and sat <= SAT_TARGET and has_ood)}
    return res


# ------------------------------------------------------------------------------------------------ the v3 entry of one language
DEFAULT_LAM = 4.0
DEFAULT_COPY_FLOOR = 0.2
DEFAULT_COPY_SCALE = 1.5
# What the domain study (scripts/v3_domain_study.py, 13 languages with religious AND out-of-domain human text, thresholds from religious text only) measured for the recipe at lam 4 / copy 0.2 x 1.5:
# out-of-domain false-positive share: copy median 0.2 % (max 6.0 %), pad median 0.1 % (p75 2.9 %, max 76.6 % on web text with English contamination), func median 9 % (p75 15.5 %, max 36 %);
# share at the gate level (deficit >= 0.9): func <= 0.6 %, pad <= 3.2 %. So, for a language WITHOUT out-of-domain text: copy and pad may be used as graded penalties, func may not, nothing may reach the gate.
SINGLE_DOMAIN_RULE = {"copy": True, "func": False, "pad": True}


def build_language_v3(iso: str, units_by_set: Dict[str, Sequence[Dict[str, str]]], funcs: Optional[Dict[str, Any]], lam: float = DEFAULT_LAM, copy_floor: float = DEFAULT_COPY_FLOOR,
                      copy_scale: float = DEFAULT_COPY_SCALE, basis_extra: str = "") -> Dict[str, Any]:
    """{"copy","func","pad": block or None, "diagnostics": ...}. Every block carries `usable` (false = keep the detector off), `gate_ok` (may reach anchors_stuff_gate), `basis` and the leave-one-set-out numbers."""
    sets = {n: us for n, us in units_by_set.items() if us}
    ood_sets = {n for n, us in sets.items() if us[0].get("dom") == "ood"}
    if funcs is None:
        funcs = {"top": [], "n_seg": 1, "n_tok": 1, "en_tf": {}}
    blocks = fit_language_v3(sets, funcs, lam, copy_floor, copy_scale)
    lodo = lodo_report(sets, funcs, lam, copy_floor, copy_scale)
    dec = decide(lodo, ood_sets)
    basis = "+".join(f"{n}({len(us)})" for n, us in sorted(sets.items())) + f"; domain-robust v3 recipe (lam {lam}, copy floor {copy_floor} x {copy_scale})" + (f"; {basis_extra}" if basis_extra else "")
    # detection power: a detector whose thresholds were pushed to <= 0 by the margin sees nothing (usable would then only mean "silent")
    import random

    from gemma_mt.rl import anchors_attacks as AT

    det_rate: Dict[str, Optional[float]] = {"func": None, "pad": None}
    base = [u for us in sets.values() for u in us][:300]
    if base and blocks.get("func") and funcs.get("top"):
        atk = AT.make_attacks(base, funcs["top"], blocks["func"]["fw"], [u["ref"] for u in base], random.Random(7))
        fr = fp_report({"func": dict(blocks["func"], enabled=True)}, [{"src": u["src"], "ref": t} for u, t in zip(base, atk["stuff"])])
        det_rate["func"] = fr.get("func", {}).get("fp")
    if base and blocks.get("pad"):
        loops = []
        for u in base:
            w = T.out_tokens(u["ref"])
            if len(w) >= 8:
                loops.append({"src": u["src"], "ref": " ".join((w[:3] * (len(w) // 3 + 1))[: len(w)])})      # the first three words repeated to the length of the reference
        fr = fp_report({"pad": dict(blocks["pad"], enabled=True)}, loops)
        det_rate["pad"] = fr.get("pad", {}).get("fp")
    out: Dict[str, Any] = {"diagnostics": {"sets": {n: len(us) for n, us in sets.items()}, "ood_sets": sorted(ood_sets), "lodo": lodo}}
    for det in ("copy", "func", "pad"):
        b = blocks.get(det)
        if b is None:
            out[det] = None
            continue
        d = dec[det]
        if d["n_heldout_sets"] and d["has_ood_heldout"]:
            usable, why = d["usable"], f"leave-one-set-out with an out-of-domain held-out set: max FP {d['max_fp']:.3f}"
        else:
            usable, why = SINGLE_DOMAIN_RULE[det], "no out-of-domain held-out set: domain-study rule (religious-text calibration, measured out-of-domain FP of the recipe on 13 other languages)"
            if det != "func" and d["n_heldout_sets"] and d["max_fp"] > FP_TARGET:
                usable, why = False, f"within-domain leave-one-set-out FP {d['max_fp']:.3f} > {FP_TARGET}"
        if det in det_rate and det_rate[det] is not None and det_rate[det] < 0.5:
            usable, why = False, f"blind: detects {det_rate[det]:.2f} of a synthetic {'stuffed bag' if det == 'func' else 'three-word loop'} (< 0.5): the margin pushed its thresholds to <= 0"
        b.update({"detect_synthetic": det_rate.get(det), "enabled": bool(b.get("enabled", True)), "usable": bool(usable and b.get("enabled", True)), "gate_ok": False, "gate_candidate": bool(d.get("gate_ok", False) and usable), "basis": basis, "usable_reason": why,
                  "lodo": {k: v for k, v in d.items() if k != "per_set"}, "lodo_per_set": d.get("per_set", {})})
        out[det] = b
    return out
