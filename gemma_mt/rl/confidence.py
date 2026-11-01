# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reference-free CONFIDENCE for the RL reward (D-47): when is a signal that needs no human reference (consensus, AfriCOMET-QE on a language it barely knows) allowed to move the policy?

Why: GRPO only re-weights what the policy can already sample. In a language the model cannot write, every sample is junk; if the junk is "valid" (right tag, finished, not a copy)
the consensus signal (D-38) rewards the junk that agrees with itself the most: a degenerate attractor, not a translation. The measurements of the study (reports/newlang_viability_*)
show that the three things that separate such a language from a genuinely weak one are cheap and need no reference:
    * the GlotLID pass rate of the T=1 samples (is it the right language at all),
    * the model's own mean token log-prob of its samples (does it "know" this language: about -1.0 .. -1.6 nats for a trained language, below -2.5 for one it cannot write),
    * the number of valid samples in the group (consensus needs >= 3 valid siblings),
while the consensus LEVEL itself does not separate them (junk agrees with itself as much as a weak real language does).

This module is PURE (no torch, no model): it turns per-sample observations that the reward and the trainer already have into
    group_features  -> GroupFeatures        one GRPO group (prompt): n_valid, GlotLID pass, consensus level, distinctness, length CV, mean log-prob
    group_gate      -> (bool, reason)       does the group contribute gradient at all (rule 1)
    language_trust  -> float in [0, 1]      the advantage weight multiplier of a language from its running features (rule 2; 0 = not enough capability yet)
    consensus_cap   -> float                the largest reward a consensus-only completion may receive (rule 3)
    ConfidenceTracker                       running (EMA) language features over steps, the thing the trainer would keep per language

The defaults of `ConfidenceParams` were set from the probe of 2026-10-20 (see `scripts/newlang_confidence_validate.py` and the report); `fit_params` re-derives them from a table of languages.
Wiring (2026-11-01): `TranslationReward` (reward.confidence, default OFF) and `train_rl.py` call these functions; see section [v4] of reports/rl_reward_explained_2026-10-19_v3.md.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# ------------------------------------------------------------------------------------------------------------------ parameters


@dataclass(frozen=True)
class ConfidenceParams:
    """All thresholds in one place. `ramp(x, lo, hi)` is 0 at `lo`, 1 at `hi`, linear between."""
    # rule 1: group gate
    min_valid: int = 3                    # valid samples a group needs (same number as consensus_min_valid, D-38)
    lid_pass_min: float = 0.5             # share of the CHECKED samples of the group whose GlotLID top-1 is accepted
    distinct_min: float = 0.5             # distinct normalised texts / valid samples: below it the "agreement" is a repeated sample, not a consensus
    consensus_junk_chrf: float = 55.0     # mean pairwise chrF++ above this among >= 3 valid samples together with a low log-prob spread: a collapsed mode (reported, not gated alone)
    # rule 2: language trust = weakest link of three ramps on the running features
    lid_lo: float = 0.55
    lid_hi: float = 0.90
    logp_lo: float = -3.1                 # mean token log-prob of the samples (nats): the 15 trained yardsticks lie in -0.37 .. -2.73, the 15 languages the model cannot write in -2.93 .. -4.41
    logp_hi: float = -2.4                 # ABSOLUTE values, valid for the step-925 phase-2c policy only: RL sharpens the distribution (the same 15 languages sit at -2.4 .. -4.3 in the step-2,250 SFT checkpoint)
    logp_rel_lo: float = -2.0             # RELATIVE form (use it in a live run): log-prob minus the median log-prob of the languages that currently work (GlotLID pass >= 0.9); the same two
    logp_rel_hi: float = -1.3             # absolute bounds expressed against the step-925 median of -1.14
    valid_lo: float = 0.10
    valid_hi: float = 0.50
    # rule 3: cap of the consensus-only reward, as a share of the way from the rubric-only value `skip` to the top of the range `hi`
    cap_floor: float = 0.0
    min_trust_to_pay: float = 0.15        # below it a language is not paid for consensus at all (weight 0)


def ramp(x: Optional[float], lo: float, hi: float) -> float:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 0.0
    if hi == lo:
        return 1.0 if x >= hi else 0.0
    return min(1.0, max(0.0, (x - lo) / (hi - lo)))


# ------------------------------------------------------------------------------------------------------------------ one group


@dataclass
class GroupFeatures:
    n: int                                # samples in the group
    n_valid: int                          # passed every hard check (tag, finished, non-empty, not a copy, not a loop, length ratio, GlotLID)
    valid_frac: float
    lid_pass: Optional[float]             # share of the LID-checked samples in the accepted label set (None: no sample was long enough to check)
    cons_mean: Optional[float]            # mean consensus chrF++ (0..100) of the valid samples (None: fewer than 3 valid)
    distinct: Optional[float]             # distinct normalised texts / valid samples (None: no valid sample)
    len_cv: Optional[float]               # coefficient of variation of the valid samples' character lengths
    logp_valid: Optional[float]           # mean token log-prob of the valid samples (sampler's distribution)
    logp_all: Optional[float]             # ... of all samples


def _norm(t: str) -> str:
    return " ".join(t.split()).lower()


def group_features(valid: Sequence[bool], lid_ok: Sequence[Optional[bool]], texts: Sequence[str], logp: Sequence[Optional[float]],
                   consensus: Optional[Sequence[Optional[float]]] = None) -> GroupFeatures:
    """One GRPO group. `valid[i]` = sample i passed every hard check of the reward (including GlotLID where it is active); `lid_ok[i]` = GlotLID accepted it (None: not checked);
    `texts[i]` = the translation (tag stripped, NFC); `logp[i]` = mean token log-prob of the sampled completion (None when unknown); `consensus[i]` = its consensus chrF++ (rl/consensus.py)."""
    n = len(valid)
    if not (len(lid_ok) == len(texts) == len(logp) == n):
        raise ValueError("valid, lid_ok, texts and logp must have the same length")
    vi = [i for i in range(n) if valid[i]]
    chk = [lid_ok[i] for i in range(n) if lid_ok[i] is not None]
    lens = [len(texts[i]) for i in vi]
    cv = None
    if len(lens) > 1 and sum(lens) > 0:
        m = sum(lens) / len(lens)
        cv = math.sqrt(sum((x - m) ** 2 for x in lens) / len(lens)) / m
    lp_all = [x for x in logp if x is not None]
    lp_v = [logp[i] for i in vi if logp[i] is not None]
    cvals = [consensus[i] for i in vi if consensus is not None and consensus[i] is not None]
    return GroupFeatures(
        n=n, n_valid=len(vi), valid_frac=len(vi) / n if n else 0.0, lid_pass=(sum(1 for c in chk if c) / len(chk)) if chk else None,
        cons_mean=(sum(cvals) / len(cvals)) if cvals else None,
        distinct=(len({_norm(texts[i]) for i in vi}) / len(vi)) if vi else None, len_cv=cv,
        logp_valid=(sum(lp_v) / len(lp_v)) if lp_v else None, logp_all=(sum(lp_all) / len(lp_all)) if lp_all else None)


def group_gate(f: GroupFeatures, p: ConfidenceParams = ConfidenceParams()) -> Tuple[bool, str]:
    """Rule 1. Does this group contribute gradient? (the reward sets the advantages of a gated-out group to 0).
    Reasons: "ok" | "few_valid" (< min_valid valid samples: no consensus possible) | "wrong_language" (GlotLID pass below the minimum) | "collapsed" (the valid samples are copies of each other)."""
    if f.n_valid < p.min_valid:
        return False, "few_valid"
    if f.lid_pass is not None and f.lid_pass < p.lid_pass_min:
        return False, "wrong_language"
    if f.distinct is not None and f.distinct < p.distinct_min:
        return False, "collapsed"
    return True, "ok"


# ------------------------------------------------------------------------------------------------------------------ one language


@dataclass
class LanguageState:
    """Running means of a language's group features (EMA over the groups seen). `alpha` = weight of a new group."""
    alpha: float = 0.05
    lid_pass: Optional[float] = None
    logp_valid: Optional[float] = None
    valid_frac: Optional[float] = None
    seen: int = 0

    def update(self, f: GroupFeatures) -> None:
        a = self.alpha if self.seen >= 20 else 1.0 / (self.seen + 1)            # plain mean for the first 20 groups, then an EMA
        for name, val in (("lid_pass", f.lid_pass), ("logp_valid", f.logp_valid if f.logp_valid is not None else f.logp_all), ("valid_frac", f.valid_frac)):
            if val is None:
                continue
            cur = getattr(self, name)
            setattr(self, name, val if cur is None else (1 - a) * cur + a * val)
        self.seen += 1


def language_trust(lid_pass: Optional[float], logp_valid: Optional[float], valid_frac: Optional[float], p: ConfidenceParams = ConfidenceParams(),
                   logp_ref: Optional[float] = None, missing_ok: bool = False) -> float:
    """Rule 2. Advantage-weight multiplier in [0, 1] of a language whose signal needs no reference: the WEAKEST of three ramps (GlotLID pass rate of the T=1 samples, mean log-prob of the
    valid samples, valid fraction). A language with no valid samples, or whose samples are mostly in another language, or which the model assigns low likelihood, has 0.
    `missing_ok` (default False = the original behaviour: a missing observation counts as 0): a ramp whose input is None (GlotLID guard off or no text long enough to check, no log-probs
    from the sampler) counts as 1 instead, i.e. no information is no penalty. The RL wiring (reward.confidence) uses True."""
    if logp_ref is not None and logp_valid is not None:
        lp = ramp(logp_valid - logp_ref, p.logp_rel_lo, p.logp_rel_hi)
    else:
        lp = ramp(logp_valid, p.logp_lo, p.logp_hi)

    def r(x, lo, hi):
        return 1.0 if (missing_ok and (x is None or (isinstance(x, float) and math.isnan(x)))) else ramp(x, lo, hi)
    if missing_ok and logp_valid is None:
        lp = 1.0
    t = min(r(lid_pass, p.lid_lo, p.lid_hi), lp, r(valid_frac, p.valid_lo, p.valid_hi))
    return 0.0 if t < p.min_trust_to_pay else t


def consensus_cap(trust: float, skip_value: float = 0.0, hi: float = 2.25, p: ConfidenceParams = ConfidenceParams()) -> float:
    """Rule 3. The largest reward a completion may get from consensus alone: `skip_value` (the rubric-only value, the shaping curve's "acceptable" point) at trust 0, the top of the
    range at trust 1. In GRPO the group standardisation removes any CONSTANT scale, so the cap acts where it matters: it makes all completions above it tie, so a self-agreeing mode
    cannot out-score a more varied but correct sibling by more than the cap allows."""
    t = min(1.0, max(0.0, trust))
    return skip_value + (p.cap_floor + (1.0 - p.cap_floor) * t) * (hi - skip_value)


def group_weight(f: GroupFeatures, group_size: int, p: ConfidenceParams = ConfidenceParams()) -> float:
    """Optional smooth version of rule 1: 0 when gated out, otherwise (n_valid - 2) / (group_size - 2) (the pairwise-agreement estimate of n valid samples has n - 1 partners; with 3 valid of 16
    it rests on 2 comparisons per sample, with 16 on 15)."""
    ok, _ = group_gate(f, p)
    if not ok or group_size <= 2:
        return 0.0
    return max(0.0, min(1.0, (f.n_valid - 2) / (group_size - 2)))


class ConfidenceTracker:
    """Per-language running state + the three rules. The trainer would call `observe(lang, features)` for every group and read `trust(lang)` when it builds the advantages."""

    def __init__(self, params: ConfidenceParams = ConfidenceParams(), alpha: float = 0.05):
        self.p, self.alpha = params, alpha
        self.state: Dict[str, LanguageState] = {}

    def observe(self, lang: str, f: GroupFeatures) -> Tuple[bool, str]:
        st = self.state.setdefault(lang, LanguageState(alpha=self.alpha))
        st.update(f)
        return group_gate(f, self.p)

    def reference_logp(self, min_seen: int = 20) -> Optional[float]:
        """Median running log-prob of the languages that currently work (GlotLID pass >= 0.9, >= `min_seen` groups seen); None when fewer than 5 such languages exist."""
        xs = sorted(s.logp_valid for s in self.state.values() if s.seen >= min_seen and s.lid_pass is not None and s.lid_pass >= 0.9 and s.logp_valid is not None)
        if len(xs) < 5:
            return None
        m = len(xs) // 2
        return xs[m] if len(xs) % 2 else 0.5 * (xs[m - 1] + xs[m])

    def trust(self, lang: str, relative: bool = True, missing_ok: bool = False, absolute_fallback: bool = True) -> float:
        """`relative`: log-prob against the median of the languages that work (`reference_logp`). When that reference does not exist yet: `absolute_fallback` True = the absolute bounds of
        ConfidenceParams (the original behaviour; valid for the step-925 policy only), False = the log-prob ramp is skipped (counts as 1) until the reference exists."""
        st = self.state.get(lang)
        if st is None or st.seen == 0:
            return 0.0
        ref = self.reference_logp() if relative else None
        lp = st.logp_valid
        if relative and ref is None and not absolute_fallback:
            lp = None
        return language_trust(st.lid_pass, lp, st.valid_frac, self.p, ref, missing_ok=missing_ok or (relative and ref is None and not absolute_fallback))

    def state_dict(self) -> Dict[str, Dict[str, Optional[float]]]:
        return {l: {"alpha": s.alpha, "lid_pass": s.lid_pass, "logp_valid": s.logp_valid, "valid_frac": s.valid_frac, "seen": s.seen} for l, s in self.state.items()}

    def load_state_dict(self, d: Optional[Dict[str, Dict[str, Optional[float]]]]) -> int:
        """Restore the running per-language state (a checkpoint written before this existed has none: returns 0)."""
        if not d:
            return 0
        for l, v in d.items():
            self.state[l] = LanguageState(alpha=float(v.get("alpha", self.alpha)), lid_pass=v.get("lid_pass"), logp_valid=v.get("logp_valid"), valid_frac=v.get("valid_frac"), seen=int(v.get("seen", 0)))
        return len(d)

    def summary(self) -> Dict[str, Dict[str, Optional[float]]]:
        return {l: {"seen": s.seen, "lid_pass": s.lid_pass, "logp_valid": s.logp_valid, "valid_frac": s.valid_frac, "trust": self.trust(l)} for l, s in self.state.items()}


# ------------------------------------------------------------------------------------------------------------------ rollout helpers


def params_from_dict(d: Optional[Dict[str, float]]) -> ConfidenceParams:
    """ConfidenceParams with the given overrides (unknown keys raise: a typo must not silently keep a default)."""
    if not d:
        return ConfidenceParams()
    names = {f.name for f in dataclasses.fields(ConfidenceParams)}
    bad = sorted(set(d) - names)
    if bad:
        raise ValueError(f"unknown ConfidenceParams key(s) {bad}; valid: {sorted(names)}")
    return ConfidenceParams(**d)


def tag_prefix_len(decode, token_ids: Sequence[int], candidates: Sequence[str], max_len: int = 4) -> int:
    """Number of leading tokens of a completion that spell the language tag: the smallest k <= max_len for which `decode(token_ids[:k]).strip()` is one of `candidates` (the iso,
    `<iso>` and the `<unusedK>` slot form); 1 when none matches (the single-token tag). `decode` is any callable ids -> text (keeps special/added tokens)."""
    cands = set(candidates)
    for k in range(1, min(max_len, len(token_ids)) + 1):
        if decode(list(token_ids[:k])).strip() in cands:
            return k
    return 1


def completion_mean_logp(logps: Optional[Sequence[float]], token_ids: Sequence[int], tag_len: int = 1, stop_ids: Sequence[int] = ()) -> Optional[float]:
    """Mean token log-prob (nats) of a sampled completion over the CONTENT tokens: the leading language tag (`tag_len` tokens: 1 with the single-token tags, 1-3 text pieces before) and a
    trailing stop id are left out (the tag is copied from the prompt and the end token is near-certain; neither says whether the model knows the language). None when the sampler gave
    no log-probs or nothing is left."""
    if logps is None or len(logps) == 0:
        return None
    n = min(len(logps), len(token_ids))
    lo, hi = max(0, int(tag_len)), n
    if hi > lo and int(token_ids[hi - 1]) in set(int(x) for x in stop_ids):
        hi -= 1
    vals = [float(x) for x in logps[lo:hi] if x is not None and math.isfinite(float(x))]
    return (sum(vals) / len(vals)) if vals else None


# ------------------------------------------------------------------------------------------------------------------ fitting


def expected_informative_groups(p_valid: float, group_size: int, min_valid: int = 3) -> float:
    """P(a group of `group_size` iid samples has >= min_valid valid ones) for a per-sample valid rate p_valid: the share of the prompts of a language that can carry any gradient."""
    p_valid = min(1.0, max(0.0, p_valid))
    tot = 0.0
    for k in range(min_valid, group_size + 1):
        tot += math.comb(group_size, k) * p_valid ** k * (1 - p_valid) ** (group_size - k)
    return tot
