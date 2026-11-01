# SPDX-License-Identifier: AGPL-3.0-or-later
"""QE ensemble of the reward (D-55): AfriCOMET-QE-STL-1.1 + SSA-COMET-MTL (reference-free), OPTIONAL, default off (`reward.qe_ensemble: true`).

Per scored chunk (a sentence pair of the chunk plan; the whole text when chunking is off):

    r1 = shape(c1, language anchors of AfriCOMET-QE)        # exactly the v7 curve, reward_shaping.ContentShaper
    r2 = shape(c2, language anchors of SSA-COMET-MTL)        # the `qe2` block of the calibration file (v8), same curve, same reward scale [-2, +2.25]
    r  = (r1 + r2) / 2   if |r1 - r2| <= delta   else   min(r1, r2)

The mean keeps most of SSA-COMET-MTL's agreement with human adequacy scores (Spearman 0.60 vs 0.47 on SSA-MTE) and cuts its weakness: a joint [mt, src] encoder accepts
a copy / a mixture of the English source as adequate (exact copy >= the human reference in 41% of sentences, AfriCOMET 1.3%; reports/qe_metric_choice_2026-10-21.md).
When the two judges disagree by more than `delta` reward units the PESSIMISTIC one wins, so a chunk only one judge likes (the signature of the copy hack, and of
anything else one model is blind to) earns the lower of the two. The result replaces the shaped QE value `r` of `TranslationReward` at exactly one place; everything after it
(consensus blend, length / lexical / diacritic / science penalties, failed chunks, the aggregation) is unchanged.

Per language (the `qe2` and `ens` blocks of the calibration file shaping_flores_dev_v8.json):
  both models usable        -> the rule above; advantage weight = `ens.reliability[_same_source]` (baked at calibration: max(v7 weight, the ensemble's own measured
                               reliability); an owner-set weight of v7 is kept, never raised)
  only SSA-COMET-MTL usable -> r2 alone (weight from its own measurement)        only AfriCOMET usable (or no `qe2` block) -> r1 alone, as before
  no dev reference          -> `qe2` holds the pooled prior (median of the calibrated languages' anchors); `qe_ensemble_prior_langs: false` leaves such languages on r1
  consensus-mode languages  -> no QE call at all (unchanged)
`content` blend (qe / blend / consensus modes): the ensemble value replaces the QE value of the blend, (1 - beta) * r + beta * consensus is unchanged.

Scorers: `QEPairScorer` is what `train_rl.build_scorer` returns when the `scorer` block has `ensemble_model_id`: ONE worker process (transformers 4.x) holding both
checkpoints; calling it returns the AfriCOMET scores (so every consumer of `scorer` is unchanged), `.secondary` scores with SSA-COMET-MTL.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from gemma_mt.rl.reward_shaping import ContentShaper, FAIL_FLOOR

SSA_MTL_MODEL_ID = "McGill-NLP/ssa-comet-mtl"
DEFAULT_MAX_BATCH_CHARS = 64000   # worker batches of long chunks are cut to this many characters (batch x longest src + mt): bounds the GPU memory of the pair worker (measured, reports/qe_ensemble_2026-10-21.md)
QE2_KEY = "qe2"            # per-language block: {model, anchors, usable, reliability, reliability_same_source, anchors_source, ...}
ENS_KEY = "ens"            # per-language block: {reliability, reliability_same_source, ...} of the ensemble itself
META_KEY = "_meta_v8"      # file-level block: {delta, primary_model, secondary_model, ...}
MEAN, MIN = "mean", "min"


# ------------------------------------------------------------------------------------------------ the rule
def combine_pair(r1: float, r2: float, delta: float) -> Tuple[float, str]:
    """(combined reward, branch): the mean when the two shaped rewards agree within `delta`, else the lower one."""
    if abs(r1 - r2) <= delta:
        return 0.5 * (r1 + r2), MEAN
    return (r1 if r1 < r2 else r2), MIN


# ------------------------------------------------------------------------------------------------ scorers
class QEPairScorer:
    """Two scorers behind one object. `__call__` = the PRIMARY (AfriCOMET) so the rest of the trainer keeps working; `secondary` scores with SSA-COMET-MTL.
    `token_lengths` (prompt filtering, truncation statistics) is the primary's. Times both models (seconds per call, summed since the last `pop_times`)."""

    def __init__(self, primary: Any, secondary: Any, owner: Any = None):
        self.primary, self.secondary, self.owner = primary, secondary, owner
        self.batch_size = getattr(primary, "batch_size", None)
        self.t_primary = 0.0
        self.t_secondary = 0.0

    def __call__(self, srcs, mts, refs=None):
        t0 = time.time()
        out = self.primary(srcs, mts) if refs is None else self.primary(srcs, mts, refs)
        self.t_primary += time.time() - t0
        return out

    def score_secondary(self, srcs, mts):
        t0 = time.time()
        out = self.secondary(srcs, mts)
        self.t_secondary += time.time() - t0
        return out

    def __getattr__(self, name):                 # token_lengths exists only if the primary has it (reward.py tests with hasattr)
        if name == "token_lengths" and hasattr(self.__dict__.get("primary"), "token_lengths"):
            return self.primary.token_lengths
        raise AttributeError(name)

    def pop_times(self) -> Tuple[float, float]:
        t = (self.t_primary, self.t_secondary)
        self.t_primary = self.t_secondary = 0.0
        return t

    def close(self) -> None:
        for o in (self.owner, self.primary, self.secondary):
            if hasattr(o, "close"):
                o.close()
                return


def build_pair_scorer(spec: Dict[str, Any], device: Optional[str] = None) -> QEPairScorer:
    """`scorer` block of the yaml with `ensemble_model_id` (and optionally `ensemble_batch_size`):
        type: worker    -> ONE child process, two checkpoints (the policy environment has transformers 5, unbabel-comet needs 4.x: D-33)
        type: comet     -> two in-process CometScorers (only where transformers < 5 is importable)
        type: fake      -> two FakeScorers (smoke tests)"""
    t = spec.get("type", "comet")
    second_id = spec.get("ensemble_model_id") or SSA_MTL_MODEL_ID
    bs1 = spec.get("batch_size", 32)
    bs2 = spec.get("ensemble_batch_size", bs1)
    if t == "fake":
        from gemma_mt.rl.comet import FakeScorer

        return QEPairScorer(FakeScorer(), FakeScorer())
    if t == "worker":
        from gemma_mt.rl.comet import WorkerScorer

        w = WorkerScorer(python=spec.get("python", sys.executable), model_id=spec.get("model_id"), batch_size=bs1, fake=spec.get("fake", False),
                         pythonpath=spec.get("pythonpath"), extra_model_ids=[second_id],
                         max_batch_chars=spec.get("max_batch_chars", DEFAULT_MAX_BATCH_CHARS))
        return QEPairScorer(w.view(0, bs1), w.view(1, bs2), owner=w)
    from gemma_mt.rl.comet import CometScorer

    return QEPairScorer(CometScorer(spec.get("model_id", "masakhane/africomet-qe-stl-1.1"), batch_size=bs1, device=device),
                        CometScorer(second_id, batch_size=bs2, device=device))


# ------------------------------------------------------------------------------------------------ the ensemble
class QEEnsemble:
    """Shapes the two raw QE scores of a chunk into one reward. Built from the calibration file's `qe2` / `ens` blocks and the reward's own v7 shaper."""

    def __init__(self, shaper1: ContentShaper, qe2: Dict[str, dict], delta: float, *, ens: Optional[Dict[str, dict]] = None, prior_langs: bool = True,
                 reliability: str = "ensemble", reliability_from: str = "auc", preset: str = "repo", neg_scale: float = 2.0, fail_floor: float = FAIL_FLOOR,
                 secondary_model: str = SSA_MTL_MODEL_ID):
        if delta < 0:
            raise ValueError("qe_ensemble_delta must be >= 0")
        if reliability not in ("ensemble", "keep"):
            raise ValueError("qe_ensemble_reliability must be 'ensemble' or 'keep'")
        if shaper1.mode != "calibrated":
            raise ValueError("the QE ensemble needs reward.mode: calibrated (the adaptive shaper moves the anchors of one model only)")
        self.shaper1, self.delta, self.prior_langs, self.reliability_mode = shaper1, float(delta), prior_langs, reliability
        self.reliability_from = reliability_from
        self.secondary_model = secondary_model
        self.qe2: Dict[str, dict] = {k: v for k, v in qe2.items() if v.get("anchors")}
        self.ens: Dict[str, dict] = dict(ens or {})
        self.shaper2 = ContentShaper(mode="calibrated", calibration={k: tuple(v["anchors"]) for k, v in self.qe2.items()}, preset=preset, neg_scale=neg_scale,
                                     fail_floor=fail_floor, default_anchors=shaper1.default)
        self.reset()

    # -- construction
    @classmethod
    def from_file(cls, path: str, shaper1: ContentShaper, *, delta: Optional[float] = None, **kw) -> "QEEnsemble":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        meta = data.get(META_KEY) or {}
        if delta is None:
            if meta.get("delta") is None:
                raise ValueError(f"no qe_ensemble_delta in the config and no {META_KEY}.delta in {path}")
            delta = float(meta["delta"])
        qe2, ens = {}, {}
        for k, v in data.items():
            if k.startswith("_") or not isinstance(v, dict):
                continue
            k = k.split(">")[-1].strip()
            if isinstance(v.get(QE2_KEY), dict):
                qe2[k] = v[QE2_KEY]
            if isinstance(v.get(ENS_KEY), dict):
                ens[k] = v[ENS_KEY]
        if not qe2:
            raise ValueError(f"{path} has no `{QE2_KEY}` blocks: calibrate the second model first (scripts/calibrate_qe_ensemble.py)")
        return cls(shaper1, qe2, delta, ens=ens, secondary_model=meta.get("secondary_model", SSA_MTL_MODEL_ID), **kw)

    @classmethod
    def from_config(cls, cfg: Any, shaper1: ContentShaper) -> "QEEnsemble":
        path = cfg.qe_ensemble_calibration_path or cfg.calibration_path
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"qe_ensemble needs a calibration file with `qe2` blocks (reward.calibration_path / qe_ensemble_calibration_path), got {path!r}")
        ens = cls.from_file(path, shaper1, delta=cfg.qe_ensemble_delta, prior_langs=cfg.qe_ensemble_prior_langs, reliability=cfg.qe_ensemble_reliability,
                            reliability_from=cfg.reliability_from, preset=cfg.preset, neg_scale=cfg.neg_scale, fail_floor=cfg.fail_floor)
        if cfg.qe_ensemble_model_id and ens.secondary_model and cfg.qe_ensemble_model_id != ens.secondary_model:
            raise ValueError(f"qe_ensemble_model_id {cfg.qe_ensemble_model_id!r} differs from the model the anchors were calibrated for ({ens.secondary_model!r})")
        return ens

    # -- which model(s) judge a language
    def has1(self, lang: str) -> bool:
        return self.shaper1.is_usable(lang)

    def has2(self, lang: str) -> bool:
        e = self.qe2.get(lang)
        if not e or not e.get("usable", True) or not e.get("anchors"):
            return False
        if not self.prior_langs and e.get("anchors_source") in ("pooled_prior", "borrowed"):
            return False
        return True

    def plan(self, lang: str) -> str:
        """both | only1 | only2 (only1 also when neither is usable: the language has advantage weight 0 anyway)."""
        h1, h2 = self.has1(lang), self.has2(lang)
        return "both" if h1 and h2 else ("only2" if h2 else "only1")

    def uses_second(self, lang: str) -> bool:
        return self.plan(lang) != "only1"

    def weight(self, lang: str, base: float) -> float:
        """Advantage weight of a language: the ensemble's baked weight where the second model takes part, else `base` (the v7 weight)."""
        if self.reliability_mode == "keep" or not self.uses_second(lang):
            return base
        e = self.ens.get(lang) or {}
        key = "reliability_same_source" if self.reliability_from == "same_source" else "reliability"
        if key in e:
            return float(e[key])
        q = self.qe2.get(lang) or {}
        return float(q.get(key, base))

    # -- scoring
    def reset(self) -> None:
        self._rec: List[tuple] = []               # (lang, plan, c1, c2, r1, r2, branch)
        self._split = [0, 0]                      # SSA pairs cut to fit the joint budget / pairs scored by SSA

    def score_second(self, scorer: Any, seg_src: Sequence[str], seg_mt: Sequence[str], langs: Sequence[str]) -> List[Optional[float]]:
        """SSA-COMET-MTL scores of the segments whose language uses it (None elsewhere). `scorer` is a QEPairScorer (or anything with `score_secondary`)."""
        idx = [k for k, l in enumerate(langs) if self.uses_second(l)]
        out: List[Optional[float]] = [None] * len(seg_src)
        if idx:
            sec = scorer.score_secondary if hasattr(scorer, "score_secondary") else scorer.secondary
            vals = [float(v) for v in sec([seg_src[k] for k in idx], [seg_mt[k] for k in idx])]
            if len(vals) != len(idx):
                raise RuntimeError(f"second scorer returned {len(vals)} scores for {len(idx)} segments")
            for k, v in zip(idx, vals):
                out[k] = v
            st = getattr(getattr(scorer, "secondary", None), "last_stats", None) or {}
            self._split[0] += int(st.get("joint_split", 0))
            self._split[1] += len(idx)
        return out

    def shape(self, c1: Optional[float], c2: Optional[float], lang: str) -> float:
        """The QE part of a chunk's reward. c1 / c2: raw scores of the two models (None = not scored by that model)."""
        p = self.plan(lang)
        r1 = float(self.shaper1(c1, lang)) if (c1 is not None and p != "only2") else None
        r2 = float(self.shaper2(c2, lang)) if (c2 is not None and p != "only1") else None
        if r1 is not None and r2 is not None:
            r, br = combine_pair(r1, r2, self.delta)
        elif r2 is not None:
            r, br = r2, "only2"
        elif r1 is not None:
            r, br = r1, "only1"
        else:
            raise ValueError("no QE score for this chunk")
        self._rec.append((lang, p, c1, c2, r1, r2, br))
        return r

    # -- logging (`reward/qe_ens/*`)
    def pop_stats(self, scorer: Any = None) -> Dict[str, float]:
        rec, self._rec = self._rec, []
        split, self._split = self._split, [0, 0]
        st: Dict[str, float] = {}
        n = len(rec)
        st["qe_ens/chunks"] = float(n)
        if scorer is not None and hasattr(scorer, "pop_times"):
            st["qe_ens/seconds_primary"], st["qe_ens/seconds_secondary"] = scorer.pop_times()
        st["qe_ens/joint_split_rate"] = split[0] / max(split[1], 1)
        if not n:
            return st
        both = [x for x in rec if x[6] in (MEAN, MIN)]
        st["qe_ens/both_share"] = len(both) / n
        st["qe_ens/only1_share"] = sum(1 for x in rec if x[6] == "only1") / n
        st["qe_ens/only2_share"] = sum(1 for x in rec if x[6] == "only2") / n
        c1s = [x[2] for x in rec if x[2] is not None]
        c2s = [x[3] for x in rec if x[3] is not None]
        if c1s:
            st["qe_ens/raw_mean/africomet"] = sum(c1s) / len(c1s)
        if c2s:
            st["qe_ens/raw_mean/ssa_mtl"] = sum(c2s) / len(c2s)
        if both:
            d = [abs(x[4] - x[5]) for x in both]
            st["qe_ens/abs_diff_mean"] = sum(d) / len(d)
            st["qe_ens/signed_diff_mean"] = sum(x[5] - x[4] for x in both) / len(both)       # SSA minus AfriCOMET, shaped
            st["qe_ens/min_share"] = sum(1 for x in both if x[6] == MIN) / len(both)
            st["qe_ens/min_share_ssa_high"] = sum(1 for x in both if x[6] == MIN and x[5] > x[4]) / len(both)      # only SSA likes it: the copy-hack signature
            st["qe_ens/min_share_afri_high"] = sum(1 for x in both if x[6] == MIN and x[4] > x[5]) / len(both)
            st["qe_ens/r1_mean"] = sum(x[4] for x in both) / len(both)
            st["qe_ens/r2_mean"] = sum(x[5] for x in both) / len(both)
            st["qe_ens/combined_mean"] = sum((0.5 * (x[4] + x[5]) if x[6] == MEAN else min(x[4], x[5])) for x in both) / len(both)
            by: Dict[str, List[tuple]] = defaultdict(list)
            for x in both:
                by[x[0]].append(x)
            for l, xs in sorted(by.items()):
                st[f"qe_ens/min_share/{l}"] = sum(1 for x in xs if x[6] == MIN) / len(xs)
                st[f"qe_ens/abs_diff/{l}"] = sum(abs(x[4] - x[5]) for x in xs) / len(xs)
        return st
