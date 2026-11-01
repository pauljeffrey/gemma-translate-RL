# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fitting the v2 anchors (D-61, rl/anchors_v2.py) on HUMAN references: the pure functions behind scripts/build_anchors_v2_calibration.py.

Every block is fitted per language on the human sentences (and pseudo-documents for the long length buckets) of `scripts/anchors_common.py`; the false-positive rate of every threshold is
measured out of sample by 2-fold cross-validation (`fp_cv`: whitelist / dictionary / threshold from one half, rate on the other).

    entity2   d = share of the judged human chunks that lost EVERY entity of the channel (recall 0); w = (1 - d) x scale; enabled when d <= max_drop (0.6) and the channel separates a human
              translation from the translation of another source (mean deficit gap >= 0.25) and >= 25 chunks are judged. Languages without references: pooled 90th-percentile d, weight x 0.5.
    copy      per language: loanword whitelist (source words that the human reference keeps in >= 50 % of >= 4 occurrences), thresholds t_w / t_c = max(floor, 99.5 % quantile) of the
              word-run (k) / character-run (m) copy fraction of the human references, enabled when the 2-fold FP is <= 1.2 % and the word threshold is <= 0.45.
    func      dictionary = the language's most frequent function words (document frequency in the training text AND in the human references, geometric mean), per length bucket the low 1 %
              quantile of the function-word share and of target function words per English function word of the source, the high 99 % quantile of the English-function-word share.
    pad       per length bucket the low 1 % quantile of the attested-bigram share (bigram table of the training text) and of the type / token ratio.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from gemma_mt.rl import anchors as A
from gemma_mt.rl import anchors_calibrate as AC
from gemma_mt.rl import anchors_v2 as V
from gemma_mt.rl import textstats as T

ENTITY_MIN_JUDGED = 25
ENTITY_MIN_SEP = 0.25
POOL_PRIOR_SCALE = 0.5
COPY_ALPHA = 0.005
COPY_FLOOR = 0.08
COPY_FP_MAX = 0.012
COPY_T_MAX = 0.45
FUNC_ALPHA = 0.01
FUNC_FP_MAX = 0.02
MIN_BUCKET = 40
WL_MIN_COUNT = 4
WL_MIN_RATE = 0.5
WL_MIN_LEN = 4


def _q(v: Sequence[float], q: float, method: str = "linear") -> float:
    return float(np.quantile(np.asarray(v, dtype=float), q, method=method))


# ------------------------------------------------------------------------------------------------ entity2
def fit_entity2(units: Sequence[Dict[str, str]], pool: Optional[Dict[str, float]] = None, max_drop: float = V.ENTITY2_MAX_DROP, name_scale: float = V.ENTITY2_NAME_SCALE,
                shrink_m: float = 20.0) -> Optional[Dict[str, Any]]:
    """Per channel: d (human drop rate), w, enabled, plus the figures of the report. `pool` = {channel: pooled d} for the shrinkage of languages with few judged chunks."""
    if not units:
        return None
    tab = AC.entity_table(units)
    out: Dict[str, Any] = {}
    for ch, nmin in AC.CHANNELS:
        r = AC._recalls(tab["own"], ch, nmin)
        ro = AC._recalls(tab["other"], ch, nmin)
        n = len(r)
        if n == 0:
            continue
        k0 = int((r == 0).sum())
        dp = (pool or {}).get(ch)
        d = (k0 + shrink_m * dp) / (n + shrink_m) if dp is not None and n < 60 else k0 / n
        sep = float((1.0 - ro).mean() - (1.0 - r).mean()) if len(ro) else 0.0
        scale = name_scale if ch == "name" else 1.0
        w = (1.0 - d) * scale
        blk = {"min_n": nmin, "n": int(n), "d_raw": k0 / n, "d": float(d), "mean_recall": float(r.mean()), "sep": sep, "w": float(w),
               "human_mean_pen": float(w * (1.0 - r).mean()), "human_pen_ge_0.1": float((w * (1.0 - r) >= 0.1).mean()),
               "detect_other": float(((1.0 - ro) >= 0.5).mean()) if len(ro) else 0.0, "scale": scale,
               "enabled": bool(n >= ENTITY_MIN_JUDGED and d <= max_drop and sep >= ENTITY_MIN_SEP)}
        out[ch] = blk
    return out or None


def pooled_entity2(per_lang: Dict[str, Dict[str, Any]], q: float = 0.9, name_scale: float = V.ENTITY2_NAME_SCALE) -> Dict[str, Any]:
    """Prior for a language without references: the q-quantile of d over the languages with >= 60 judged chunks, weight halved (POOL_PRIOR_SCALE)."""
    out: Dict[str, Any] = {}
    for ch, nmin in AC.CHANNELS:
        ds = [b[ch]["d"] for b in per_lang.values() if b.get(ch) and b[ch]["n"] >= 60]
        if not ds:
            continue
        d = _q(ds, q, "higher")
        scale = (name_scale if ch == "name" else 1.0) * POOL_PRIOR_SCALE
        out[ch] = {"min_n": nmin, "n_languages": len(ds), "d": d, "w": (1.0 - d) * scale, "scale": scale, "enabled": bool(d <= V.ENTITY2_MAX_DROP)}
    return out


def pool_d(per_lang: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """Median d per channel over the languages with >= 60 judged chunks (the shrinkage target)."""
    out = {}
    for ch, _ in AC.CHANNELS:
        ds = [b[ch]["d_raw"] for b in per_lang.values() if b.get(ch) and b[ch]["n"] >= 60]
        if ds:
            out[ch] = float(np.median(ds))
    return out


# ------------------------------------------------------------------------------------------------ copy
def learn_whitelist(units: Sequence[Dict[str, str]], min_count: int = WL_MIN_COUNT, min_rate: float = WL_MIN_RATE, min_len: int = WL_MIN_LEN) -> List[str]:
    """Source words (>= min_len letters, lowercase, not protected as names / numbers) that the human reference keeps verbatim: legitimate loanwords of the language."""
    seen: Counter = Counter()
    kept: Counter = Counter()
    for u in units:
        st = T.source_tokens(u["src"])
        ot = set(T.out_tokens(u["ref"]))
        for w, p in zip(st.words, st.prot):
            if p or len(w) < min_len or w in T.EN_FUNC:
                continue
            seen[w] += 1
            if w in ot:
                kept[w] += 1
    return sorted(w for w, c in seen.items() if c >= min_count and kept[w] / c >= min_rate)


def copy_table(units: Sequence[Dict[str, str]], whitelist: FrozenSet[str], k: int, m: int, key: str = "ref") -> np.ndarray:
    """[n, 2] (frac_w, frac_c) of every unit's `key` text against its source; NaN where the detector does not judge the unit."""
    rows = []
    for u in units:
        st = T.source_tokens(u["src"], whitelist)
        r = T.copy_fractions(st, T.out_tokens(u[key]), k=k, m=m)
        rows.append((np.nan if r.frac_w is None else r.frac_w, np.nan if r.frac_c is None else r.frac_c))
    return np.array(rows, dtype=float).reshape(-1, 2)


def synthetic_partial(units: Sequence[Dict[str, str]], share: float) -> List[Dict[str, str]]:
    """The reference with its LAST `share` of the words replaced by the last `share` of the source's words (a copied tail)."""
    out = []
    for u in units:
        rw, sw = u["ref"].split(), u["src"].split()
        a = max(1, int(round(len(rw) * (1 - share))))
        b = max(1, int(round(len(sw) * share)))
        out.append({"src": u["src"], "mt": " ".join(rw[:a] + sw[-b:])})
    return out


def _thr(v: np.ndarray, alpha: float, floor: float) -> float:
    v = v[~np.isnan(v)]
    return max(floor, _q(v, 1.0 - alpha, "higher")) if len(v) >= 30 else 1.0


def fit_copy(units: Sequence[Dict[str, str]], k: int = 3, m: int = 20, alpha: float = COPY_ALPHA, floor: float = COPY_FLOOR, seed: int = 0) -> Optional[Dict[str, Any]]:
    if len(units) < 80:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(units))
    h = len(units) // 2
    folds = ((idx[:h], idx[h:]), (idx[h:], idx[:h]))
    fps_w, fps_c, fps = [], [], []
    for a, b in folds:
        ua, ub = [units[i] for i in a], [units[i] for i in b]
        wl = frozenset(learn_whitelist(ua))
        ta = copy_table(ua, wl, k, m)
        tw, tc = _thr(ta[:, 0], alpha, floor), (_thr(ta[:, 1], alpha, floor) if m else 9.0)
        tb = copy_table(ub, wl, k, m)
        okb = ~np.isnan(tb[:, 0])
        fw = np.nan_to_num(tb[:, 0]) > tw
        fc = np.nan_to_num(tb[:, 1]) > tc
        fps_w.append(float(fw[okb].mean()) if okb.any() else 0.0)
        fps_c.append(float(fc[okb].mean()) if okb.any() else 0.0)
        fps.append(float((fw | fc)[okb].mean()) if okb.any() else 0.0)
    wl_list = learn_whitelist(units)
    wl = frozenset(wl_list)
    tab = copy_table(units, wl, k, m)
    tw, tc = _thr(tab[:, 0], alpha, floor), (_thr(tab[:, 1], alpha, floor) if m else 9.0)
    ok = ~np.isnan(tab[:, 0])
    judged = int(ok.sum())
    if judged < 30:
        return None
    det = {}
    for share in (0.25, 0.5):
        syn = synthetic_partial(units, share)
        s_units = [{"src": u["src"], "ref": s["mt"]} for u, s in zip(units, syn)]
        st = copy_table(s_units, wl, k, m)
        okk = ~np.isnan(st[:, 0])
        hit = (np.nan_to_num(st[:, 0]) > tw) | (np.nan_to_num(st[:, 1]) > tc)
        det[f"tail{int(share * 100)}"] = float(hit[okk].mean()) if okk.any() else 0.0
    fp = float(((np.nan_to_num(tab[:, 0]) > tw) | (np.nan_to_num(tab[:, 1]) > tc))[ok].mean()) if judged else 0.0
    blk = {"k": k, "m": m, "min_unprot": 2, "t_w": tw, "t_c": tc, "whitelist": wl_list, "n": judged, "fp": fp, "fp_cv": float(np.mean(fps)), "fp_cv_w": float(np.mean(fps_w)),
           "fp_cv_c": float(np.mean(fps_c)), "q99_w": _q(tab[ok, 0], 0.99), "max_w": float(np.nanmax(tab[:, 0])) if judged else 0.0, "detect": det,
           "enabled": bool(judged >= 80 and np.mean(fps) <= COPY_FP_MAX and tw <= COPY_T_MAX)}
    return blk


def pooled_copy(per_lang: Dict[str, Dict[str, Any]], q: float = 0.9) -> Dict[str, Any]:
    """Prior for a language without references: thresholds = the q-quantile of the enabled languages' thresholds, no whitelist."""
    en = [b for b in per_lang.values() if b.get("enabled")]
    if not en:
        return {"enabled": False}
    k, m = en[0]["k"], en[0]["m"]
    tw = _q([b["t_w"] for b in en], q, "higher")
    tc = _q([b["t_c"] for b in en], q, "higher")
    return {"k": k, "m": m, "min_unprot": 2, "t_w": tw, "t_c": tc, "whitelist": [], "n_languages": len(en), "enabled": bool(tw <= COPY_T_MAX)}


# ------------------------------------------------------------------------------------------------ func / pad
def select_fw(train_top: Sequence[Sequence[Any]], n_seg: int, human_df: Counter, n_human: int, k: int, min_df: float = 0.004) -> List[str]:
    """The `k` words with the highest geometric mean of the document frequency in the training text and in the human references (domain-robust function words)."""
    sc = []
    for w, df, _tf in train_top:
        dh = human_df.get(w, 0) / max(n_human, 1)
        dt = df / max(n_seg, 1)
        if dh >= min_df and dt >= min_df:
            sc.append((math.sqrt(dh * dt), w))
    sc.sort(reverse=True)
    return [w for _, w in sc[:k]]


def english_excl(en_tf: Dict[str, int], n_tok: int, fw: Sequence[str], min_share: float = 1e-4) -> List[str]:
    """English function words that the language itself uses (>= min_share of its tokens in the training text, or in its function-word list): not evidence of English."""
    ex = set(fw)
    for w, c in en_tf.items():
        if c / max(n_tok, 1) >= min_share:
            ex.add(w)
    return sorted(w for w in ex if w in T.EN_FUNC)


def func_rows(units: Sequence[Dict[str, str]], fw: FrozenSet[str], ex: FrozenSet[str], key: str = "ref") -> np.ndarray:
    """[n_units, 6]: n output words, fw share, en share, fw per English function word of the source, ttr, bucket. Units that are too short are dropped."""
    rows = []
    for u in units:
        ow = T.out_tokens(u[key])
        sw = T.out_tokens(u["src"])
        if len(ow) < V.FUNC_MIN_WORDS or len(sw) < V.FUNC_MIN_SRC_WORDS:
            continue
        f = T.func_features(ow, fw, ex)
        es = sum(1 for w in sw if w in T.EN_FUNC)
        rows.append((f["n"], f["fw"], f["en"], f["nfw"] / (es + 1.0), f["ttr"], V.v2_bucket(f["n"])))
    return np.array(rows, dtype=float).reshape(-1, 6)


def bucket_thresholds(buckets: np.ndarray, values: np.ndarray, q: float, low_tail: bool, min_n: int = MIN_BUCKET, prior: Optional[Sequence[Optional[float]]] = None,
                      lam: float = 0.0) -> List[Optional[float]]:
    """Per length bucket the q-quantile of `values` (low tail: q is the lower quantile), moved outwards by `lam` x (the distance between the q- and the 5 %-quantile): the extreme quantile of a
    few hundred correlated texts is as noisy as the minimum, the extrapolated tail is not. A bucket with < min_n values takes the prior (pooled) or its nearest neighbour."""
    nb = len(V.V2_LENGTH_EDGES) + 1
    ts: List[Optional[float]] = []
    for k in range(nb):
        v = values[buckets == k]
        v = v[~np.isnan(v)]
        if len(v) < min_n:
            ts.append(prior[k] if prior else None)
            continue
        if low_tail:
            t0, t5 = _q(v, q, "lower"), _q(v, 0.05, "lower")
            t = t0 - lam * max(t5 - t0, 0.0)
            ts.append(max(t, 0.5 * t0) if t0 > 0 else t)             # the margin never moves a positive threshold below half of the quantile (D-68: a threshold <= 0 silences the detector)
        else:
            t0, t5 = _q(v, 1.0 - q, "higher"), _q(v, 0.95, "higher")
            ts.append(min(t0 + lam * max(t0 - t5, 0.0), 2.0 * t0 + 0.05))
    known = [t for t in ts if t is not None]
    if not known:
        return [None] * nb
    for k in range(nb):
        if ts[k] is None:
            near = [(abs(j - k), ts[j]) for j in range(nb) if ts[j] is not None]
            ts[k] = min(near)[1]
    return [float(t) for t in ts]


def _func_deficits(rows: np.ndarray, blk: Dict[str, Any], clip: bool = True) -> np.ndarray:
    """Vectorised Anchors2.func_deficit over rows of func_rows (the same function the runtime uses)."""
    return np.array([V.func_deficit_from(blk, int(r[0]), r[1], r[2], r[3], clip) for r in rows], dtype=float)


LAMBDAS = (0.0, 0.5, 1.0, 1.5, 2.5, 4.0)
GEEZ_FP_MAX = 0.005       # Ge'ez-script languages (amh, tir): a stricter target for the stop-word / padding thresholds (owner 2026-10-22: never false-positive them)


def is_geez(texts: Sequence[str], min_share: float = 0.5) -> bool:
    """True when at least `min_share` of the letters of `texts` are Ethiopic (U+1200..U+139F): Ge'ez-script text."""
    n = e = 0
    for t in texts:
        for ch in t:
            if ch.isalpha():
                n += 1
                e += 0x1200 <= ord(ch) <= 0x139F
    return n > 0 and e / n >= min_share


def fit_func(rows: np.ndarray, lam: Optional[float] = None, alpha: float = FUNC_ALPHA, seed: int = 0, fp_max: float = FUNC_FP_MAX) -> Optional[Dict[str, Any]]:
    """Thresholds at the human `alpha` quantile moved outwards by `lam` tail distances (bucket_thresholds); with lam=None the first of LAMBDAS whose 2-fold false-positive rate (union of the
    features) is <= FUNC_FP_MAX (the smallest margin that meets the target)."""
    if len(rows) < 2 * MIN_BUCKET:
        return None
    if lam is None:
        best = None
        for l in LAMBDAS:
            best = fit_func(rows, l, alpha, seed, fp_max)
            if best["fp_cv"] <= fp_max:
                break
        return best

    def thr(r: np.ndarray) -> Dict[str, Any]:
        return {"t_fw": bucket_thresholds(r[:, 5], r[:, 1], alpha, True, lam=lam), "t_en": bucket_thresholds(r[:, 5], r[:, 2], alpha, False, lam=lam),
                "t_ratio": bucket_thresholds(r[:, 5], r[:, 3], alpha, True, lam=lam), "t_fw_hi": bucket_thresholds(r[:, 5], r[:, 1], alpha, False, lam=lam),
                "cap_en": 0.3, "cap_fw": 0.95}

    blk = thr(rows)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(rows))
    h = len(rows) // 2
    fps = []
    for a, b in ((idx[:h], idx[h:]), (idx[h:], idx[:h])):
        t = thr(rows[a])
        fps.append(float((_func_deficits(rows[b], t) > 0).mean()))
    d = _func_deficits(rows, blk)
    blk.update({"n": int(len(rows)), "alpha": alpha, "lam": lam, "fp": float((d > 0).mean()), "fp_cv": float(np.mean(fps)), "mean_pen": float(d.mean())})
    return blk


def pad_rows(units: Sequence[Dict[str, str]], bigrams: Optional[np.ndarray], key: str = "ref", ttr_mode: str = "plain") -> np.ndarray:
    """[n, 4]: n words, bigram-order gain (NaN without a table), ttr, bucket."""
    rows = []
    for u in units:
        ow = T.out_tokens(u[key])
        if len(ow) < V.FUNC_MIN_WORDS:
            continue
        gn = np.nan
        if bigrams is not None and bigrams.size:
            obs, shuf = V.bigram_gain(bigrams, ow)
            gn = obs - shuf
        rows.append((len(ow), gn, V.pad_ttr(ow, ttr_mode), V.v2_bucket(len(ow))))
    return np.array(rows, dtype=float).reshape(-1, 4)


def _pad_deficits(rows: np.ndarray, blk: Dict[str, Any], clip: bool = True) -> np.ndarray:
    return np.array([V.pad_deficit_from(blk, int(r[0]), None if np.isnan(r[1]) else r[1], r[2], clip) for r in rows], dtype=float)


def fit_pad(rows: np.ndarray, lam: Optional[float] = None, alpha: float = FUNC_ALPHA, seed: int = 0, fp_max: float = FUNC_FP_MAX, ttr_mode: str = "plain") -> Optional[Dict[str, Any]]:
    if len(rows) < 2 * MIN_BUCKET:
        return None
    if lam is None:
        best = None
        for l in LAMBDAS:
            best = fit_pad(rows, l, alpha, seed, fp_max, ttr_mode)
            if best["fp_cv"] <= fp_max:
                break
        return best
    has_bg = bool(np.isfinite(rows[:, 1]).any())

    def thr(r: np.ndarray) -> Dict[str, Any]:
        return {"t_gn": bucket_thresholds(r[:, 3], r[:, 1], alpha, True, lam=lam) if has_bg else [None] * (len(V.V2_LENGTH_EDGES) + 1),
                "t_ttr": bucket_thresholds(r[:, 3], r[:, 2], alpha, True, lam=lam)}

    blk = thr(rows)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(rows))
    h = len(rows) // 2
    fps = []
    for a, b in ((idx[:h], idx[h:]), (idx[h:], idx[:h])):
        fps.append(float((_pad_deficits(rows[b], thr(rows[a])) > 0).mean()))
    d = _pad_deficits(rows, blk)
    blk.update({"n": int(len(rows)), "alpha": alpha, "lam": lam, "ttr_mode": ttr_mode, "has_bigrams": has_bg, "fp": float((d > 0).mean()), "fp_cv": float(np.mean(fps)), "mean_pen": float(d.mean())})
    return blk
