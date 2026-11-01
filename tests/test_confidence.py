# SPDX-License-Identifier: AGPL-3.0-or-later
"""D-47: reference-free confidence (rl/confidence.py): group features, the group gate, the language trust, the consensus cap, the tracker, and the junk-consensus failure mode."""
import math

import pytest

from gemma_mt.rl import confidence as C
from gemma_mt.rl import consensus as cons

P = C.ConfidenceParams()


def _group(n=8, n_valid=8, lid=True, logp=-1.2, texts=None):
    texts = texts or [f"ala ni ogun kan wa si ile {i} oja agbe" for i in range(n)]
    valid = [i < n_valid for i in range(n)]
    lid_ok = [lid if valid[i] else False for i in range(n)]
    return C.group_features(valid, lid_ok, texts, [logp] * n)


def test_ramp():
    assert C.ramp(0.0, 0.5, 1.0) == 0.0 and C.ramp(1.0, 0.5, 1.0) == 1.0 and C.ramp(0.75, 0.5, 1.0) == pytest.approx(0.5)
    assert C.ramp(None, 0, 1) == 0.0 and C.ramp(float("nan"), 0, 1) == 0.0


def test_group_features_basic():
    f = _group()
    assert f.n == 8 and f.n_valid == 8 and f.valid_frac == 1.0 and f.lid_pass == 1.0 and f.distinct == 1.0 and f.logp_valid == pytest.approx(-1.2)
    f2 = _group(n_valid=2)
    assert f2.n_valid == 2 and f2.lid_pass == pytest.approx(2 / 8)         # the invalid ones are LID-rejected in this fixture


def test_group_features_length_mismatch_raises():
    with pytest.raises(ValueError):
        C.group_features([True], [True, True], ["a"], [None])


def test_group_features_no_valid():
    f = C.group_features([False] * 4, [None] * 4, ["x"] * 4, [None] * 4)
    assert f.n_valid == 0 and f.lid_pass is None and f.distinct is None and f.logp_valid is None and f.cons_mean is None


def test_gate_reasons():
    assert C.group_gate(_group(), P) == (True, "ok")
    assert C.group_gate(_group(n_valid=2), P) == (False, "few_valid")
    f = _group()
    f.lid_pass = 0.3
    assert C.group_gate(f, P) == (False, "wrong_language")
    same = _group(texts=["identical text here"] * 8)
    assert C.group_gate(same, P) == (False, "collapsed")


def test_language_trust_weakest_link():
    good = C.language_trust(0.99, -1.2, 0.95, P)
    assert good == 1.0
    assert C.language_trust(0.0, -3.3, 0.0, P) == 0.0                       # a language the model cannot write (fuv-like numbers)
    assert C.language_trust(0.99, -3.3, 0.95, P) == 0.0                     # right language per LID but the model assigns it very low likelihood
    assert C.language_trust(0.2, -1.2, 0.95, P) == 0.0                      # likely well-formed but in another language
    mid = C.language_trust(0.75, -2.2, 0.3, P)
    assert 0.0 < mid < 1.0
    assert C.language_trust(None, None, None, P) == 0.0


def test_trust_monotone():
    xs = [C.language_trust(l, -1.5, 0.9, P) for l in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)]
    assert xs == sorted(xs)


def test_consensus_cap_range():
    assert C.consensus_cap(0.0, 0.0, 2.25, P) == 0.0
    assert C.consensus_cap(1.0, 0.0, 2.25, P) == 2.25
    assert C.consensus_cap(0.5, 0.0, 2.25, P) == pytest.approx(1.125)
    assert C.consensus_cap(7.0, 0.0, 2.25, P) == 2.25                       # trust is clipped


def test_group_weight():
    assert C.group_weight(_group(n=16, n_valid=16), 16, P) == 1.0
    def g(n_valid):                                   # invalid samples that were not language-checked (e.g. unfinished): they do not lower the LID pass rate
        valid = [i < n_valid for i in range(16)]
        return C.group_features(valid, [True if v else None for v in valid], [f"t {i}" for i in range(16)], [-1.2] * 16)
    assert C.group_weight(g(3), 16, P) == pytest.approx(1 / 14)
    assert C.group_weight(g(2), 16, P) == 0.0


def test_tracker_junk_language_gets_no_trust():
    tr = C.ConfidenceTracker(P)
    junk = C.group_features([False] * 8, [False] * 8, ["x"] * 8, [-3.3] * 8)
    for _ in range(30):
        tr.observe("fuv", junk)
    assert tr.trust("fuv") == 0.0
    good = _group()
    for _ in range(30):
        ok, why = tr.observe("yor", good)
        assert ok and why == "ok"
    assert tr.trust("yor") == 1.0
    assert tr.trust("never_seen") == 0.0


def test_tracker_recovers_when_the_language_improves():
    tr = C.ConfidenceTracker(P, alpha=0.2)
    junk = C.group_features([False] * 8, [False] * 8, ["x"] * 8, [-3.3] * 8)
    for _ in range(25):
        tr.observe("gaa", junk)
    assert tr.trust("gaa") == 0.0
    good = _group(logp=-1.5)
    for _ in range(60):
        tr.observe("gaa", good)
    assert tr.trust("gaa") > 0.9


def test_informative_group_probability():
    assert C.expected_informative_groups(0.0, 16) == 0.0
    assert C.expected_informative_groups(1.0, 16) == 1.0
    assert 0.0 < C.expected_informative_groups(0.1, 16) < 0.25
    assert C.expected_informative_groups(0.5, 16) > 0.99
    assert C.expected_informative_groups(0.1, 16, 3) < C.expected_informative_groups(0.1, 16, 2)


def test_consensus_among_junk_is_not_separated_by_the_consensus_level():
    """The failure mode itself: 8 'valid' samples that are all variations of one repeated junk phrase agree with each other strongly; their consensus level is as high as that of
    real, varied translations, so the LEVEL cannot be the gate - the log-prob / LID features must be."""
    junk = ["ye ka ni ye ka mansa ye bolo", "ye ka ni ye ka mansa ye bolo ma", "ye ka ni ye ka mansa bolo ye", "ni ye ka ni ye ka mansa ye bolo", "ye ka ni ye ka mansa ye bolo ye"] * 2
    real = ["The farmers sell vegetables at the market.", "Farmers sell their vegetables in the market.", "At the market the farmers sell vegetables.",
            "The farmers are selling vegetables at the market.", "In the market, farmers sell vegetables."] * 2
    def level(texts):
        profs = [cons.Profile(t) for t in texts]
        vals = cons.group_consensus(profs, 3)
        return sum(vals) / len(vals)
    assert level(junk) > 60.0 and level(real) > 30.0
    fj = C.group_features([True] * 10, [None] * 10, junk, [-3.3] * 10, [level(junk)] * 10)
    fr = C.group_features([True] * 10, [None] * 10, real, [-1.2] * 10, [level(real)] * 10)
    tj = C.language_trust(1.0, fj.logp_valid, fj.valid_frac, P)
    tr = C.language_trust(1.0, fr.logp_valid, fr.valid_frac, P)
    assert tj == 0.0 and tr == 1.0


def test_relative_logp_form_follows_the_run():
    """After RL sharpened the policy every language has a higher log-prob; the relative form (against the median of the working languages) keeps the verdicts."""
    assert C.language_trust(0.99, -2.7, 0.95, P, logp_ref=-1.14) == pytest.approx(C.ramp(-2.7 + 1.14, P.logp_rel_lo, P.logp_rel_hi))
    # an SFT-stage policy (everything ~1.5 nats lower) judged against its own median is not distrusted wholesale
    assert C.language_trust(0.99, -3.0, 0.95, P, logp_ref=-2.9) == 1.0
    assert C.language_trust(0.99, -3.0, 0.95, P) < 0.2                      # the absolute form WOULD distrust it


def test_tracker_reference_logp():
    tr = C.ConfidenceTracker(P)
    good = lambda lp: C.group_features([True] * 8, [True] * 8, [f"t{i}" for i in range(8)], [lp] * 8)      # noqa: E731
    assert tr.reference_logp() is None
    for i, lp in enumerate((-0.5, -0.9, -1.1, -1.4, -2.7)):
        for _ in range(25):
            tr.observe(f"l{i}", good(lp))
    assert tr.reference_logp() == pytest.approx(-1.1)
    junk = C.group_features([False] * 8, [False] * 8, ["x"] * 8, [-3.3] * 8)
    for _ in range(25):
        tr.observe("fuv", junk)
    assert tr.reference_logp() == pytest.approx(-1.1)                       # a language that does not work does not move the reference
    assert tr.trust("fuv") == 0.0 and tr.trust("l0") == 1.0
