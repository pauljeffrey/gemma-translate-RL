# SPDX-License-Identifier: AGPL-3.0-or-later
"""Post-editor RL reward: op alignment, anchors, dead zone / asymmetry, copy rules (CPU, no models)."""
from gemma_mt.rl import post_edit_reward as per

W = per.Weights()
SRC = "In 2019 Dr. Adeyemi visited Lagos and met 45 farmers."


def test_align_counts():
    ops, ch = per.align_ops("a b c d", "a x c d e")
    assert (ops.inserted, ops.deleted, ops.substituted) == (1, 0, 1)
    assert not ops.identical and ch
    ops, _ = per.align_ops("kaaro o", "Kaaro, o")
    assert ops.identical


def test_diacritic_only_substitution():
    ops, _ = per.align_ops("ola nla", "ọla nla")
    assert ops.substituted == 1 and ops.diacritic_only == 0     # the dot below is a letter, not a tone mark
    ops, _ = per.align_ops("olá nla", "ola nla")
    assert ops.diacritic_only == 1


def test_identity_is_zero_unless_draft_is_defective():
    assert per.score(SRC, "a b", "a b", 0.1, 0.1, severity=0.0).total == 0.0
    assert per.score(SRC, "a b", "a b", 0.1, 0.1, severity=1.0).total == -W.copy


def test_dead_zone_and_asymmetry():
    up = per.score(SRC, "aa bb cc", "aa bb dd", 0.0, 0.10)
    down = per.score(SRC, "aa bb cc", "aa bb dd", 0.0, -0.10)
    noise = per.score(SRC, "aa bb cc", "aa bb dd", 0.0, 0.01)
    assert up.gain > 0 and down.gain < 0
    assert abs(down.gain) > up.gain                  # regressions weigh kappa times
    assert noise.gain == 0.0 and noise.churn < 0     # a cosmetic edit with no measured gain costs


def test_gain_on_a_defective_draft_counts_more():
    bad = per.score(SRC, "aa bb cc", "aa bb dd", 0.0, 0.2, severity=1.0)
    ok = per.score(SRC, "aa bb cc", "aa bb dd", 0.0, 0.2, severity=0.0)
    assert bad.gain > ok.gain and abs(bad.gain - 2 * ok.gain) < 1e-9


def test_anchor_recovered_and_lost():
    rec = per.score(SRC, "ni oko ni", "ni oko 2019 Lagos", 0.0, 0.0)
    lost = per.score(SRC, "ni 2019 Lagos oko", "ni oko", 0.0, 0.0)
    assert rec.ledger > 0 and lost.ledger < 0
    assert per.score(SRC, "ni oko", "ni oko 777 99", 0.0, 0.0).ledger < 0      # invented numbers


def test_reference_ledger_fix_and_break():
    ref = "mo ri ile nla"
    fix = per.score(SRC, "mo ri nla", "mo ri ile nla", 0.0, 0.0, reference=ref, chrf_d=40.0, chrf_e=80.0)
    brk = per.score(SRC, "mo ri ile nla", "mo ri nla", 0.0, 0.0, reference=ref, chrf_d=80.0, chrf_e=40.0)
    assert fix.total > 0 > brk.total and fix.ref > 0 > brk.ref


def test_repair_of_a_loop():
    loop = "ni oko ni oko ni oko ni oko ni oko ni oko ni oko"
    r = per.score(SRC, loop, "ni oko", 0.0, 0.0)
    assert r.repair > 0
    assert per.score(SRC, "ni oko", loop, 0.0, 0.0).repair < 0


def test_ungrounded_insertion_penalty():
    r = per.score(SRC, "ni oko", "ni oko zxqwvu plmokn bhgtyr", 0.0, 0.0)
    assert r.ungrounded < 0


def test_failure_value_untouched():
    assert per.score(SRC, "a", "b", 0.0, 0.5, failed=True).total == -2.0


def test_defect_severity():
    assert per.defect_severity(SRC, "", 0.0, W) == 1.0
    assert per.defect_severity(SRC, "ni oko 2019 Lagos Adeyemi 45", 0.2, W) < 0.5
    assert per.defect_severity(SRC, "ni oko ni oko ni oko ni oko ni oko ni oko ni oko", 0.2, W) == 1.0
