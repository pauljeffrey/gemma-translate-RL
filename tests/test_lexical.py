# SPDX-License-Identifier: AGPL-3.0-or-later
"""Lexical fidelity (rl/lexical.py, D-25): numbers / names / content words of the output must come from the source."""
import json

import pytest

from gemma_mt.rl import lexical
from gemma_mt.rl.reward import RewardConfig, Sample
from test_reward import ConstScorer, ids, make

SRC = "Governor Okonkwo said 250 farmers in Kaduna will receive fertiliser subsidies before the planting season."
GOOD = "Governor Okonkwo talk say 250 farmers for Kaduna go collect fertiliser subsidies before the planting season."
WRONG = "President Tinubu don announce new electricity tariffs for Lagos and Abuja yesterday evening."


def test_anchors_and_content_words():
    a = lexical.anchors(SRC)
    assert {"okonkwo", "250", "kaduna"} <= a and "governor" not in a      # sentence-initial capital is not a name
    c = lexical.content_words(SRC)
    assert "farmers" in c and "the" not in c and "250" not in c


def test_features_separate_a_translation_from_unrelated_text():
    g, w = lexical.features(SRC, GOOD), lexical.features(SRC, WRONG)
    assert g["anchor_precision"] == 1.0 and g["anchor_recall"] == 1.0 and g["cognate_precision"] >= 0.75
    assert w["anchor_precision"] == 0.0 and w["anchor_recall"] == 0.0 and w["cognate_precision"] < 0.2


def test_deficit_is_zero_above_threshold_and_grows_below():
    th = {"anchor_precision": 0.8, "cognate_precision": 0.4}
    assert lexical.deficit(lexical.features(SRC, GOOD), th) == 0.0
    assert lexical.deficit(lexical.features(SRC, WRONG), th) == pytest.approx(1.0)
    assert lexical.deficit({"anchor_precision": None}, th) == 0.0           # nothing to judge -> no penalty
    assert lexical.deficit(lexical.features(SRC, WRONG), {}) == 0.0


def test_short_outputs_do_not_trigger_cognate_features():
    f = lexical.features("The dog barked.", "Dog bark.")
    assert f["cognate_precision"] is None and f["cognate_recall"] is None


def test_summarise_keeps_only_features_the_references_carry():
    rows = [(SRC, GOOD)] * 40
    s = lexical.summarise(rows)
    assert set(s) <= set(lexical.PENALTY_FEATURES) and "anchor_precision" in s and "cognate_recall" not in s
    # a language whose references share no words with the English (Yoruba-like): only "output names come from the source" stays enabled
    assert set(lexical.summarise([(SRC, "Gomina so pe agbe ogorun meji yoo gba iranlowo ni ose to n bo.")] * 40)) == {"anchor_precision"}


def _cal(tmp_path, th):
    p = tmp_path / "cal.json"
    p.write_text(json.dumps({"pcm": {"anchors": [0.46, 0.55, 0.65], "usable": True, "reliability": 0.9,
                                     "lexical": {k: {"p10": v, "p50": 0.9} for k, v in th.items()}}}))
    return str(p)


def test_reward_penalises_unrelated_text_only_when_enabled(tmp_path):
    cal = _cal(tmp_path, {"anchor_precision": 0.8, "cognate_precision": 0.4})
    off = make(RewardConfig(calibration_path=cal, lexical_penalty=0.0), scorer=ConstScorer(0.9))
    on = make(RewardConfig(calibration_path=cal, lexical_penalty=1.5), scorer=ConstScorer(0.9))
    g0, w0 = (off([Sample(ids("pcm " + t), "pcm", SRC)]).rewards[0] for t in (GOOD, WRONG))
    g1, w1 = (on([Sample(ids("pcm " + t), "pcm", SRC)]).rewards[0] for t in (GOOD, WRONG))
    assert g1 == pytest.approx(g0), "a faithful translation is untouched"
    assert w0 == pytest.approx(g0), "without the penalty QE (constant here) cannot tell them apart"
    assert w1 < w0 - 1.0 and w1 >= on.penalty - 1e-9


def test_penalty_only_never_a_bonus_and_copy_stays_failed(tmp_path):
    cal = _cal(tmp_path, {"anchor_precision": 0.8, "cognate_precision": 0.4})
    on = make(RewardConfig(calibration_path=cal, lexical_penalty=1.5), scorer=ConstScorer(0.9))
    off = make(RewardConfig(calibration_path=cal, lexical_penalty=0.0), scorer=ConstScorer(0.9))
    # perfect overlap earns nothing extra
    assert on([Sample(ids("pcm " + GOOD), "pcm", SRC)]).rewards[0] == pytest.approx(off([Sample(ids("pcm " + GOOD), "pcm", SRC)]).rewards[0])
    # copying the English source is still a failure class at the floor, whatever its lexical overlap
    o = on([Sample(ids("pcm " + SRC), "pcm", SRC)])
    assert o.status[0] == "copy_source" and o.rewards[0] == pytest.approx(on.penalty)


def test_stuffing_source_words_cannot_beat_a_faithful_translation(tmp_path):
    cal = _cal(tmp_path, {"anchor_precision": 0.8, "cognate_precision": 0.4})
    on = make(RewardConfig(calibration_path=cal, lexical_penalty=1.5), scorer=ConstScorer(0.9))
    stuffed = "Governor Okonkwo 250 Kaduna farmers fertiliser subsidies planting season Okonkwo 250 Kaduna"
    a = on([Sample(ids("pcm " + GOOD), "pcm", SRC)]).rewards[0]
    b = on([Sample(ids("pcm " + stuffed), "pcm", SRC)]).rewards[0]
    assert b <= a + 1e-9
