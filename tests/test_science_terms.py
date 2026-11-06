# SPDX-License-Identifier: AGPL-3.0-or-later
"""Science reference terms (owner 2026-11-03, rl/science_terms.py + RewardConfig.sci_*): pure functions and the reward pass. Off by default; penalties only (no lift out of a failure class)."""
import math

import pytest

from gemma_mt.fake_tokenizer import FakeGemmaTokenizer, SPECIALS
from gemma_mt.rl import science_terms as st
from gemma_mt.rl.reward import RewardConfig, Sample, TranslationReward

TOK = FakeGemmaTokenizer()
EOT = SPECIALS["<end_of_turn>"]
STOP = [EOT, SPECIALS["<eos>"]]
SRC = "A dose of 5 mg of paracetamol reduced HIV viral load by 40% in Nairobi."
REF = "Doze ya 5 mg ya paracetamol yaha HIV kukata 40% mu Nairobi."


def test_recall_precision_identity_and_extremes():
    r, p = st.chrf_recall_precision(REF, REF)
    assert r == pytest.approx(1.0) and p == pytest.approx(1.0)
    r2, p2 = st.chrf_recall_precision("Doze ya 5 mg", REF)                         # short hypothesis: precision high, recall low
    assert p2 > r2
    r3, p3 = st.chrf_recall_precision(REF + " " + REF + " kuongeza maneno mengi sana", REF)   # padded hypothesis: recall high, precision lower
    assert r3 > p3


def test_length_log_ratio_sign_and_symmetry():
    assert st.length_log_ratio("abcd", "abcd") == 0.0
    assert st.length_log_ratio("a" * 40, "a" * 10) > 0 > st.length_log_ratio("a" * 10, "a" * 40)


def test_source_items_numbers_units_acronyms_names():
    it = st.source_items(SRC)
    assert "5mg" in it["units"] and "40%" in it["units"]
    assert "HIV" in it["acronyms"]
    assert "5" in it["numbers"] and "40" in it["numbers"]
    assert any("nairobi" == n for n in it["names"])


def test_preservation_share_missing_and_threshold():
    miss, n, per = st.preservation(SRC, REF)
    assert n >= 5 and miss == pytest.approx(0.0)
    miss2, _, per2 = st.preservation(SRC, "Doze ya paracetamol yaha kukata mu mji.")   # numbers, unit, acronym and name dropped
    assert miss2 > 0.7 and per2["acronyms"] == 1
    assert st.preservation("The sky is blue.", "Anga ni bluu.")[0] is None            # fewer than min_items items: not judged


def test_adjustment_penalties_only_without_ssa_and_ssa_slope():
    cfg = st.ScienceTermsConfig()
    adj, d = st.science_adjustment(cfg, SRC, REF, REF, None)
    assert adj == pytest.approx(0.0) and "ssa" not in d                             # perfect copy of the reference: no penalty, no ssa term
    bad, d2 = st.science_adjustment(cfg, SRC, "Doze ya", REF, None)
    assert bad < -0.3 and d2["omit"] > 0 and d2["preserve_missing"] > 0
    a1, _ = st.science_adjustment(cfg, SRC, REF, REF, 0.9)
    a2, _ = st.science_adjustment(cfg, SRC, REF, REF, 0.5)
    assert a1 - a2 == pytest.approx(cfg.ssa_weight * cfg.ssa_slope * 0.4)


def test_batch_stats_rates():
    cfg = st.ScienceTermsConfig()
    dets = [st.science_adjustment(cfg, SRC, REF, REF, None)[1], st.science_adjustment(cfg, SRC, "Doze ya", REF, None)[1]]
    s = st.batch_stats(dets)
    assert s["sci/n"] == 2 and s["sci/omit_rate"] == 0.5 and s["sci/preserve_judged"] == 1.0 and s["sci/preserve_missing_mean"] > 0.3


class ConstScorer:
    def __init__(self, value=0.7, ref=None):
        self.value, self.ref_value = value, ref
        self.ref_scorer = None if ref is None else object()
        self.calls = []

    def __call__(self, s, m, r=None):
        return [self.value] * len(s)

    def score_reference(self, s, m, r):
        self.calls.append(len(s))
        return [self.ref_value] * len(s)


def ids(text, end=True):
    out = TOK.encode(text, add_special_tokens=False)
    return out + [EOT] if end else out


def reward(scorer, **kw):
    return TranslationReward(RewardConfig(**kw), scorer, TOK, STOP)


def sample(text, ref="", end=True, iso="yor"):
    return Sample(ids(f"{iso} {text}", end), iso, SRC, reference=ref)


def test_off_by_default_changes_nothing():
    a = reward(ConstScorer())([sample(REF, REF), sample("Doze ya", REF)]).rewards
    b = reward(ConstScorer(), sci_terms=False)([sample(REF, REF), sample("Doze ya", REF)]).rewards
    assert a == b


def test_reward_pass_penalises_dropped_content_and_leaves_failures_alone():
    sc = ConstScorer()
    r = reward(sc, sci_terms=True)
    base = reward(ConstScorer())
    good, short, noend = sample(REF, REF), sample("Doze ya 5 mg ya paracetamol yaha HIV", REF), sample(REF, REF, end=False)
    out, ref_out = r([good, short, noend]), base([good, short, noend])
    assert out.rewards[0] == pytest.approx(ref_out.rewards[0])                      # perfect: no change
    assert out.status[1] == "ok" and out.rewards[1] < ref_out.rewards[1] - 0.05      # dropped 40% / Nairobi: preservation (and omission) penalty
    assert out.rewards[2] == ref_out.rewards[2] and out.status[2] != "ok"          # hard failure untouched
    assert out.stats["sci/n"] == 2.0 and out.stats["sci/preserve_missing_mean"] > 0 and out.stats["sci/ssa_available"] == 0.0


def test_ssa_reference_scores_are_batched_once_and_shape_the_reward():
    sc = ConstScorer(value=0.4, ref=0.9)
    out = reward(sc, sci_terms=True)([sample(REF, REF), sample(REF, REF)])
    assert sc.calls == [2]                                                          # one batched call for all reference rows
    assert out.stats["sci/ssa_available"] == 1.0 and out.stats["sci/ssa_mean"] == pytest.approx(0.9)
    plain = reward(ConstScorer(value=0.4), sci_terms=True)([sample(REF, REF)]).rewards[0]
    assert out.rewards[0] - plain == pytest.approx(0.5 * 4.0 * (0.9 - 0.5))


def test_rows_without_reference_are_not_touched():
    out = reward(ConstScorer(ref=0.9), sci_terms=True)([sample(REF, ""), sample("Doze ya", "")])
    base = reward(ConstScorer())([sample(REF, ""), sample("Doze ya", "")])
    assert out.rewards == base.rewards and out.stats["sci/n"] == 0.0


def test_reward_never_exceeds_the_configured_range():
    sc = ConstScorer(value=0.95, ref=1.0)                                            # an extreme SSA score and a perfect copy of the reference: the additive terms must not leave the range
    r = reward(sc, sci_terms=True)
    out = r([sample(REF, REF) for _ in range(4)])
    lo, hi = r.shaper.reward_range
    assert max(out.rewards) <= hi + 1e-9 and min(out.rewards) >= lo - 1e-9
