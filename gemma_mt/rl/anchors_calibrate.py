# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fitting the thresholds of rl/anchors.py on HUMAN references (D-56): the pure functions behind scripts/build_anchors_calibration.py.

Rule for every detector: the threshold is a low (entity recall) / high (repetition, boilerplate) quantile of the statistic on the language's human texts, `alpha` = 3 % by default, so the
share of human chunks / texts that the detector penalises at all is <= 3 % by construction; the out-of-sample figure is measured by 2-fold cross-validation (`fp_cv`) and a detector
is enabled for a language only when `fp_cv` is small AND the detector is useful (entity: it flags >= 40 % of chunks paired with another source's translation). The human 10th
percentile named in the task is reported too (`p10`), but at sentence level it is 0 for names and 1 sentence in 10 would be penalised, so it cannot be the threshold of a detector that
must stay silent on good output. Languages without references (12 of the 45) use the pooled prior, a conservative quantile over the languages' own thresholds.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from gemma_mt.rl import anchors as A

ALPHA = 0.03
MIN_JUDGED = 60            # judged chunks a language needs for an own entity threshold
MIN_TEXTS = 50             # human texts per length bucket for an own repetition threshold
MIN_BOILER_REFS = 150
FP_CV_MAX = 0.045          # enable a language-detector only if the cross-validated false-positive rate is below this (target 3 %, the 2-fold estimate has ~1 pt of noise)
DETECT_MIN = 0.40
REPEAT_CAP = 0.30
BOILER_CAP = 0.40
BOILER_FLOOR = 0.03
CHANNELS = (("num", A.ENTITY_MIN_NUM), ("name", A.ENTITY_MIN_NAME))


def _q(v: Sequence[float], q: float, method: str) -> float:
    return float(np.quantile(np.asarray(v, dtype=float), q, method=method))


# ------------------------------------------------------------------------------------------------ (a) entities
def entity_table(units: Sequence[Dict[str, str]], shift: int = 1) -> Dict[str, np.ndarray]:
    """Per unit: counts of its own reference and of the reference of the unit `shift` positions on (a different source's translation)."""
    own, oth = [], []
    n = len(units)
    for i, u in enumerate(units):
        c = A.entity_counts(u["src"], u["ref"])
        o = A.entity_counts(u["src"], units[(i + shift) % n]["ref"]) if n > 1 else c
        own.append((c.n_num, c.hit_num, c.n_name, c.hit_name))
        oth.append((o.n_num, o.hit_num, o.n_name, o.hit_name))
    return {"own": np.array(own, dtype=float).reshape(-1, 4), "other": np.array(oth, dtype=float).reshape(-1, 4)}


def _recalls(tab: np.ndarray, ch: str, nmin: int) -> np.ndarray:
    col_n, col_h = (0, 1) if ch == "num" else (2, 3)
    m = tab[:, col_n] >= nmin
    return tab[m, col_h] / tab[m, col_n]


def fit_entity_channel(own: np.ndarray, other: np.ndarray, ch: str, nmin: int, alpha: float = ALPHA, seed: int = 0) -> Optional[Dict[str, Any]]:
    r = _recalls(own, ch, nmin)
    if len(r) < MIN_JUDGED:
        return None
    t = _q(r, alpha, "lower")
    ro = _recalls(other, ch, nmin)
    blk: Dict[str, Any] = {"t": t, "min_n": nmin, "n": int(len(r)), "mean": float(r.mean()), "p10": _q(r, 0.10, "lower"), "p50": _q(r, 0.5, "lower"),
                           "fp": float((r < t).mean()) if t > 0 else 0.0, "detect_other": float((ro < t).mean()) if t > 0 and len(ro) else 0.0}
    # 2-fold cross-validated false-positive rate: threshold from one half, rate on the other
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(r))
    h = len(r) // 2
    fps = []
    for a, b in ((idx[:h], idx[h:]), (idx[h:], idx[:h])):
        ta = _q(r[a], alpha, "lower")
        fps.append(float((r[b] < ta).mean()) if ta > 0 else 0.0)
    blk["fp_cv"] = float(np.mean(fps))
    blk["enabled"] = bool(t > 0 and blk["fp_cv"] <= FP_CV_MAX and blk["detect_other"] >= DETECT_MIN)
    return blk


def fit_entity(units: Sequence[Dict[str, str]], alpha: float = ALPHA) -> Optional[Dict[str, Any]]:
    tab = entity_table(units)
    out = {}
    for ch, nmin in CHANNELS:
        b = fit_entity_channel(tab["own"], tab["other"], ch, nmin, alpha)
        if b:
            out[ch] = b
    return out or None


def pooled_entity(per_lang: Dict[str, Dict[str, Any]], tables: Dict[str, Dict[str, np.ndarray]], q: float = 0.10) -> Dict[str, Any]:
    """Prior for a language without references: the q-quantile over the languages' own thresholds (a language with no usable channel counts as 0), enabled only if it keeps its
    leave-one-out false-positive rate small on the languages that DO have references."""
    out: Dict[str, Any] = {}
    for ch, nmin in CHANNELS:
        ts = [float(b[ch]["t"]) if (b.get(ch) and b[ch].get("enabled")) else 0.0 for b in per_lang.values()]
        if not ts:
            continue
        t = _q(ts, q, "lower")
        fps, dets = [], []
        for iso, tab in tables.items():
            r = _recalls(tab["own"], ch, nmin)
            ro = _recalls(tab["other"], ch, nmin)
            if len(r) >= MIN_JUDGED and t > 0:
                fps.append(float((r < t).mean()))
                dets.append(float((ro < t).mean()))
        out[ch] = {"t": t, "min_n": nmin, "n_languages": len(ts), "fp_median": float(np.median(fps)) if fps else None, "fp_p90": _q(fps, 0.9, "higher") if fps else None,
                   "detect_other": float(np.median(dets)) if dets else None,
                   "enabled": bool(t > 0 and fps and _q(fps, 0.9, "higher") <= FP_CV_MAX and np.median(dets) >= DETECT_MIN)}
    return out


# ------------------------------------------------------------------------------------------------ (b) repetition
def repeat_table(units: Sequence[Dict[str, str]]) -> Dict[str, np.ndarray]:
    """Per text: [bucket, n_words, share3, share4, share5, src_share3, src_share4, src_share5]."""
    rows = []
    for u in units:
        w = A.words(u["ref"])
        if len(w) < A.REPEAT_MIN_WORDS:
            continue
        sw = A.words(u["src"])
        s = A.repeat_shares(w)
        ss = A.repeat_shares(sw) if len(sw) >= A.REPEAT_MIN_WORDS else {n: 0.0 for n in A.REPEAT_NS}
        rows.append([A.length_bucket(len(w)), len(w)] + [s[n] for n in A.REPEAT_NS] + [ss[n] for n in A.REPEAT_NS])
    return {"t": np.array(rows, dtype=float).reshape(-1, 8)}


def doc_floor(tabs: Sequence[np.ndarray], alpha: float = ALPHA, min_docs: int = 100) -> Optional[Dict[str, List[Optional[float]]]]:
    """Pooled high quantile of the repetition of REAL human documents (smol_doc, whole papers), per length bucket: real documents repeat more than the pseudo-documents glued from
    unrelated sentences (4-9 % of them exceeded the pseudo-document threshold), so the thresholds of the long buckets are never below it. None for a bucket with < min_docs documents."""
    allt = np.concatenate([t for t in tabs if len(t)]) if tabs else np.zeros((0, 8))
    if not len(allt):
        return None
    nb = len(A.LENGTH_EDGES) + 1
    out: Dict[str, List[Optional[float]]] = {}
    for j, n in enumerate(A.REPEAT_NS):
        col = 2 + j
        row: List[Optional[float]] = []
        for k in range(nb):
            v = allt[allt[:, 0] == k][:, col]
            row.append(_q(v, 1.0 - alpha, "higher") if len(v) >= min_docs else None)
        out[str(n)] = row
    return out


def _bucket_thr(tab: np.ndarray, alpha: float, prior: Optional[Dict[str, List[float]]] = None, min_texts: int = MIN_TEXTS,
                floor: Optional[Dict[str, List[Optional[float]]]] = None) -> Dict[str, List[float]]:
    nb = len(A.LENGTH_EDGES) + 1
    thr: Dict[str, List[float]] = {}
    for j, n in enumerate(A.REPEAT_NS):
        col = 2 + j
        ts: List[Optional[float]] = []
        for k in range(nb):
            v = tab[tab[:, 0] == k][:, col]
            ts.append(_q(v, 1.0 - alpha, "higher") if len(v) >= min_texts else None)
        fill = []
        last = 0.0
        for k in range(nb):
            t = ts[k] if ts[k] is not None else (prior[str(n)][k] if prior else last)
            if floor and floor[str(n)][k] is not None:
                t = max(t, floor[str(n)][k])
            t = max(t, last)                              # repetition only grows with length
            fill.append(float(t))
            last = t
        thr[str(n)] = fill
    return thr


def _repeat_fp(tab: np.ndarray, thr: Dict[str, List[float]], cap: float = REPEAT_CAP) -> float:
    """Share of texts with a repetition deficit > 0 (the source's own repetition is an allowance, as in Anchors.repeat_deficit)."""
    if not len(tab):
        return 0.0
    bad = np.zeros(len(tab), dtype=bool)
    for j, n in enumerate(A.REPEAT_NS):
        t = np.array([thr[str(n)][int(b)] for b in tab[:, 0]])
        t = np.maximum(t, tab[:, 5 + j])
        bad |= tab[:, 2 + j] > t
    return float(bad.mean())


def fit_repeat(tab: np.ndarray, prior: Optional[Dict[str, List[float]]], alpha: float = ALPHA, seed: int = 0, floor=None) -> Optional[Dict[str, Any]]:
    if len(tab) < 2 * MIN_TEXTS:
        return None
    thr = _bucket_thr(tab, alpha, prior, floor=floor)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(tab))
    h = len(tab) // 2
    fps = []
    for a, b in ((idx[:h], idx[h:]), (idx[h:], idx[:h])):
        fps.append(_repeat_fp(tab[b], _bucket_thr(tab[a], alpha, prior, min_texts=MIN_TEXTS // 2, floor=floor)))
    fp = _repeat_fp(tab, thr)
    fp_cv = float(np.mean(fps))
    return {"thr": thr, "cap": {str(n): REPEAT_CAP for n in A.REPEAT_NS}, "n_texts": int(len(tab)), "fp": fp, "fp_cv": fp_cv, "enabled": bool(fp_cv <= FP_CV_MAX)}


def pooled_repeat(tabs: Dict[str, np.ndarray], alpha: float = ALPHA, floor=None) -> Dict[str, Any]:
    """Prior for the languages without references: the thresholds are the high quantile of the pooled texts; the quantile is made stricter (alpha, alpha/2, alpha/3, alpha/6) until the
    false-positive rate of the 90th-percentile language is <= FP_CV_MAX (a language with a repetitive orthography - sag - would otherwise make the prior unusable)."""
    allt = np.concatenate([t for t in tabs.values() if len(t)]) if tabs else np.zeros((0, 8))
    best = None
    for div in (1.0, 2.0, 3.0, 6.0):
        thr = _bucket_thr(allt, alpha / div, None, min_texts=MIN_TEXTS, floor=floor)
        fps = [_repeat_fp(t, thr) for t in tabs.values() if len(t) >= MIN_TEXTS]
        res = {"thr": thr, "cap": {str(n): REPEAT_CAP for n in A.REPEAT_NS}, "n_texts": int(len(allt)), "alpha_used": alpha / div,
               "fp_median": float(np.median(fps)) if fps else None, "fp_p90": _q(fps, 0.9, "higher") if fps else None,
               "enabled": bool(fps and _q(fps, 0.9, "higher") <= FP_CV_MAX)}
        best = res
        if res["enabled"]:
            break
    return best


# ------------------------------------------------------------------------------------------------ (c) boilerplate
def boiler_shares(units: Sequence[Dict[str, str]], lang: str, seed: int = 0, capacity: int = 600) -> np.ndarray:
    """Boilerplate share of every human reference against a buffer of `capacity` OTHER references of the same language (each its own prompt)."""
    rng = random.Random(seed)
    us = list(units)
    rng.shuffle(us)
    buf = A.BoilerplateBuffer(capacity=capacity, min_outputs=1, min_prompts=1)
    ps = []
    for i, u in enumerate(us):
        ps.append(A.prepare(lang, u["src"], u["ref"], key=i))
    for p in ps[:capacity]:
        buf.add(lang, p.key, p.grams, p.src_hashes)
    out = []
    for p in ps:
        s = buf.score(lang, p.key, p.grams, p.src_hashes)
        if s is not None:
            out.append(s)
    return np.array(out, dtype=float)


def fit_boiler(sh: np.ndarray, alpha: float = ALPHA) -> Optional[Dict[str, Any]]:
    if len(sh) < MIN_BOILER_REFS:
        return None
    t = max(_q(sh, 1.0 - alpha, "higher"), BOILER_FLOOR)
    return {"thr": t, "cap": BOILER_CAP, "n": int(len(sh)), "mean": float(sh.mean()), "p95": _q(sh, 0.95, "higher"), "fp": float((sh > t).mean()), "enabled": bool(t <= 0.5 * BOILER_CAP)}


def pooled_boiler(blks: Dict[str, Dict[str, Any]], q: float = 0.9) -> Dict[str, Any]:
    ts = [b["thr"] for b in blks.values()]
    t = max(_q(ts, q, "higher"), BOILER_FLOOR) if ts else BOILER_FLOOR
    fps = [float((np.asarray(b["_sh"]) > t).mean()) for b in blks.values() if "_sh" in b]
    return {"thr": t, "cap": BOILER_CAP, "n_languages": len(ts), "fp_median": float(np.median(fps)) if fps else None, "fp_p90": _q(fps, 0.9, "higher") if fps else None,
            "enabled": bool(ts and t <= 0.5 * BOILER_CAP)}


# ------------------------------------------------------------------------------------------------ (e) LID probability, (d) lexicon
def fit_lid_prob(tops: Sequence[Sequence[Tuple[str, float]]], accepted: Sequence[str], q: float = 0.05) -> Optional[Dict[str, Any]]:
    """tops: GlotLID top-1 [(label, prob)] per human reference. pmin = the q-quantile of the probability among references whose label is accepted."""
    ok = [p for (lab, p) in (t[0] for t in tops if t) if lab in set(accepted)]
    if len(ok) < 50:
        return None
    return {"pmin": _q(ok, q, "lower"), "label_recall": len(ok) / max(len(tops), 1), "n": len(tops), "p50": _q(ok, 0.5, "lower"),
            "recall_at": {str(x): float(np.mean([p >= x for p in ok])) * len(ok) / max(len(tops), 1) for x in (0.3, 0.5, 0.7, 0.9)}, "enabled": True}


def fit_lexicon(overlap, units: Sequence[Dict[str, str]], alpha: float = ALPHA) -> Optional[Dict[str, Any]]:
    """`overlap(src, ref) -> Optional[float]` (rl/lexicon.py through Anchors.lexicon_overlap). t = the alpha-quantile of the overlap of the HUMAN references that the lexicon judges."""
    cov = []
    for u in units:
        v = overlap(u["src"], u["ref"])
        if v is not None:
            cov.append(float(v))
    if len(cov) < MIN_JUDGED:
        return None
    t = _q(cov, alpha, "lower")
    return {"t": t, "p50": _q(cov, 0.5, "lower"), "n": len(cov), "mean": float(np.mean(cov)), "fp": float((np.asarray(cov) < t).mean()), "enabled": bool(t > 0)}
