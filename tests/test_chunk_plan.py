# SPDX-License-Identifier: AGPL-3.0-or-later
"""The pure chunk planner / aggregator (chunk_plan.py): the semantics the chunked reward is built on, tested without a scorer."""
import pytest

from gemma_mt.rl.chunk_plan import aggregate, plan_chunks

SRC = ("The market opens at eight o'clock in the morning. The farmers bring fresh vegetables from the village. "
       "Children sell oranges near the gate. Everyone is happy to see the new bridge.")
GOOD = ("Oja n si ni agogo mejo owuro. Awon agbe mu ewebe tutu wa lati abule. Awon omode n ta osan leba geeti. "
        "Inu gbogbo eniyan dun lati ri afara tuntun.")


def test_off_is_one_segment_weighted_by_the_source():
    p = plan_chunks(SRC, GOOD, chunking="off")
    assert p.segments == [(SRC, GOOD)] and p.items == [(0, float(len(SRC)))] and p.how == "off"
    assert plan_chunks(SRC, GOOD, chunking="off", chunk_agg="mean").items == [(0, 1.0)]


def test_equal_counts_pair_the_i_th_sentences_and_weights_are_source_characters():
    p = plan_chunks(SRC, GOOD)
    assert p.how == "aligned" and len(p.segments) == 4 and [k for k, _ in p.items] == [0, 1, 2, 3]
    assert [w for _, w in p.items] == [float(len(a)) for a, _ in p.segments]
    assert [w for _, w in plan_chunks(SRC, GOOD, chunk_agg="mean").items] == [1.0] * 4


def test_omission_bead_becomes_a_failed_chunk_with_its_source_weight():
    """Forced with a stub alignment: with the Gale-Church priors an omission bead (1,0) is almost never chosen (a 2:1 merge is cheaper;
    measured in the report reports/rl_reward_explained_2026-10-17.md), so the branch is exercised directly."""
    import gemma_mt.rl.segments as seg
    out = "Oja n si ni agogo mejo owuro. Awon agbe mu ewebe tutu wa lati abule. Inu gbogbo eniyan dun lati ri afara tuntun."
    real = seg.gale_church
    seg.gale_church = lambda s, o: [((0,), (0,)), ((1,), (1,)), ((2,), ()), ((3,), (2,))]
    try:
        p = plan_chunks(SRC, out)
    finally:
        seg.gale_church = real
    assert p.how == "gale_church" and p.n_failed == 1 and len(p.segments) == 3
    assert [w for k, w in p.items if k is None] == [float(len("Children sell oranges near the gate."))]


def test_a_dropped_sentence_is_usually_absorbed_by_a_merge_not_flagged():
    out = "Oja n si ni agogo mejo owuro. Awon agbe mu ewebe tutu wa lati abule. Inu gbogbo eniyan dun lati ri afara tuntun."
    p = plan_chunks(SRC, out)
    assert p.how == "gale_church" and p.n_failed == 0 and len(p.segments) == 3        # the 2:1 bead is scored by QE, with the source side longer


def test_copied_sentence_is_a_failed_chunk_only_when_there_are_several_pairs():
    out = GOOD.replace("Awon omode n ta osan leba geeti.", "Children sell oranges near the gate.")
    p = plan_chunks(SRC, out)
    assert [k for k, _ in p.items].count(None) == 1 and len(p.segments) == 3
    one = plan_chunks("The market opens.", "The market opens.")                  # single sentence: the hard copy guard handles it
    assert one.how == "single" and one.items == [(0, float(len("The market opens.")))]


def test_extra_output_text_is_one_failed_chunk_weighted_by_its_characters():
    src3 = "The market opens at eight o'clock. The farmers bring fresh vegetables. Everyone is happy today."
    ok = "Oja n si ni agogo mejo. Awon agbe mu ewebe tutu wa. Inu gbogbo eniyan dun loni."
    extra = ok + " Gbogbo eniyan ni won pe lati wa si ipade nla ni ojo Satide to n bo ni ilu naa ati si ibi ayeye ti a ti n reti fun igba pipe, ki won si wa pelu ayo nla ninu okan won."
    p = plan_chunks(src3, extra)
    assert [k for k, _ in p.items].count(None) >= 1 and "appended_text" in p.extra
    assert aggregate(p.items, {k: 1.0 for k in range(len(p.segments))}, -2.0) < 0.0
    off = plan_chunks(src3, extra, extra_output_penalty=False)
    assert all(k is not None for k, _ in off.items)


def test_appended_text_inside_a_chunk_uses_the_calibrated_window_or_the_default_limit():
    src = "The farmers bring fresh vegetables to the market every single morning."
    long = "Awon agbe mu ewebe tutu wa si oja ni gbogbo owuro. " + "Ati pe awon omode ti won n ta osan leba geeti n gba owo. " * 2
    default = plan_chunks(src, long)                                              # limit 1.8
    tight = plan_chunks(src, long, w_hi=1.0)                                      # window p95 = 1.0 -> limit 1.15
    assert [k for k, _ in default.items].count(None) >= 1
    over_default = [w for k, w in default.items if k is None][0]
    over_tight = [w for k, w in tight.items if k is None][0]
    assert over_tight > over_default
    normal = plan_chunks(src, "Awon agbe mu ewebe tutu wa si oja ni gbogbo owuro.")
    assert all(k is not None for k, _ in normal.items)


def test_extra_on_single_false_leaves_a_one_sentence_pair_alone():
    src = "The farmers bring fresh vegetables to the market."
    long = "Awon agbe mu ewebe tutu wa si oja ni gbogbo owuro ati ni gbogbo ale ni gbogbo ose ati ni gbogbo odun."
    assert any(k is None for k, _ in plan_chunks(src, long).items)
    assert all(k is not None for k, _ in plan_chunks(src, long, extra_on_single=False).items)


def test_aggregate_is_the_weighted_mean_with_the_penalty_for_failed_chunks():
    items = [(0, 10.0), (None, 30.0), (1, 60.0)]
    assert aggregate(items, [2.0, 1.0], -2.0) == pytest.approx((20 - 60 + 60) / 100)
    assert aggregate([(0, 5.0)], [1.2345678901234567], -2.0, exact_single=True) == 1.2345678901234567
    assert aggregate([(None, 5.0)], [], -2.0) == -2.0
    assert aggregate([(0, 0.0), (1, 0.0)], [1.0, 3.0], -2.0) == 2.0               # zero total weight: plain mean, no ZeroDivisionError


# ------------------------------------------------------------------------------------------------ omission detector (D-30)
from gemma_mt.rl.segments import LengthModel, align_beads, gale_church, split_sentences

_SRC5 = ["The market opens at eight o'clock in the morning.",
         "The farmers from the village bring fresh vegetables and ripe fruit to sell.",
         "Children sell oranges near the old gate.",
         "Everyone is happy to see the new bridge that the government built last year.",
         "In the evening the traders count their money and go home."]
_WORDS = "awon agbe mu ewebe tutu wa si oja ni owuro gbogbo eniyan dun lati ri afara tuntun".split()


def _fit(n, k=0):
    """A sentence of exactly n characters (a stand-in translation whose length is proportional to the source)."""
    s, i = "", k
    while len(s) < n:
        s += _WORDS[i % len(_WORDS)] + " "
        i += 1
    s = (s[: n - 1].rstrip() + "a" * n)[: n - 1] + "."
    return s[0].upper() + s[1:]


_OUT5 = [_fit(len(s), i) for i, s in enumerate(_SRC5)]
_SRC5_TEXT, _OUT5_TEXT = " ".join(_SRC5), " ".join(_OUT5)
_MODEL = LengthModel(ratio=1.0)


def test_legacy_gale_church_absorbs_a_dropped_sentence_the_language_aware_one_finds_it():
    out = " ".join(o for i, o in enumerate(_OUT5) if i != 2)
    old = plan_chunks(_SRC5_TEXT, out)
    assert old.n_failed == 0 and [k for k, _ in old.items].count(None) == 0               # F2: absorbed into a 2:1 bead
    new = plan_chunks(_SRC5_TEXT, out, length_model=_MODEL, short_frac=0.7)
    assert new.how == "gale_church" and new.n_failed == 1 and new.flagged_src == [2]
    assert [w for k, w in new.items if k is None] == [float(len(_SRC5[2]))]               # weight = the SOURCE characters of the dropped sentence
    assert len(new.segments) == 4 and all(a != _SRC5[2] for a, _ in new.segments)


def test_every_single_dropped_sentence_is_found_on_length_matched_text():
    for d in range(5):
        out = " ".join(o for i, o in enumerate(_OUT5) if i != d)
        p = plan_chunks(_SRC5_TEXT, out, length_model=_MODEL)
        assert p.flagged_src == [d], d


def test_equal_sentence_counts_and_single_sentences_are_planned_exactly_as_before():
    for src, out in ((_SRC5_TEXT, _OUT5_TEXT), (_SRC5[0], _OUT5[0]), (_SRC5_TEXT, " ".join(reversed(_OUT5)))):
        a = plan_chunks(src, out)
        b = plan_chunks(src, out, length_model=_MODEL, short_frac=0.7)
        assert (a.segments, a.items, a.how, a.n_failed, a.extra) == (b.segments, b.items, b.how, b.n_failed, b.extra)
        assert b.flagged_src == [] and b.n_short == 0


def test_faithful_merge_of_two_sentences_is_not_flagged():
    out = " ".join([_OUT5[0] + " " + _OUT5[1]] + _OUT5[2:]).replace(". ", ", ", 1)    # the translator joined two sentences into one
    assert len(split_sentences(out)) == 4
    p = plan_chunks(_SRC5_TEXT, out, length_model=_MODEL, short_frac=0.7)
    assert p.n_failed == 0 and p.n_short == 0 and p.flagged_src == []


def test_extra_sentence_appended_or_inserted_is_a_failed_chunk_and_does_not_shift_the_pairing():
    extra = _fit(90, 7)
    for out_sents in (_OUT5 + [extra], _OUT5[:2] + [extra] + _OUT5[2:]):
        p = plan_chunks(_SRC5_TEXT, " ".join(out_sents), length_model=_MODEL)
        assert "unmatched_output" in p.extra and p.flagged_src == []
        pairs = {a: b for a, b in p.segments}
        assert all(pairs[s] == o for s, o in zip(_SRC5, _OUT5))                         # every source sentence still meets its own translation


def test_directional_costs():
    beads = gale_church(_SRC5, _OUT5[:4], _MODEL)
    assert not any(si == () for si, oi in beads)                                         # fewer output sentences: no "extra output" bead
    beads = gale_church(_SRC5, _OUT5 + [_fit(60, 3)], _MODEL)
    assert not any(oi == () for si, oi in beads)                                         # more output sentences: no omission bead


def test_shortfall_check_flags_a_whole_output_that_is_far_too_short():
    out = _OUT5[0] + " " + _OUT5[1][: len(_OUT5[1]) // 2].rstrip() + "."
    out_one = _OUT5[0]                                                                   # only the first of five sentences is translated
    p = plan_chunks(_SRC5_TEXT, out_one, length_model=_MODEL, short_frac=0.7)
    assert p.how == "whole" and p.n_short == 1 and "shortfall" in p.extra
    tot = float(len(" ".join(_SRC5)))
    failed = [w for k, w in p.items if k is None][0]
    assert failed == pytest.approx(tot - len(_OUT5[0]), rel=0.02) and sum(w for _, w in p.items) == pytest.approx(tot)   # total weight unchanged
    assert plan_chunks(_SRC5_TEXT, out_one).n_short == 0                                  # off without a model
    assert plan_chunks(_SRC5_TEXT, out_one, length_model=_MODEL).n_short == 0             # short_frac None: off
    assert plan_chunks(_SRC5_TEXT, out, length_model=_MODEL, short_frac=0.7).how in ("gale_church", "aligned", "whole")


def test_per_language_omission_cost_is_chosen_on_untouched_documents_and_read_by_the_loader(tmp_path):
    import json

    from gemma_mt.rl.omission_eval import choose_omission_cost, length_stats
    from gemma_mt.rl.reward_shaping import ContentShaper

    rows = [{"id": i, "iso": "yor", "english": _SRC5[i % 5], "reference": _OUT5[i % 5]} for i in range(120)]
    o = choose_omission_cost(rows, target=0.02, tilings=2)
    assert o["cost"] == 4.0 and o["met_target"] and o["false_alarm"] <= 0.02 and set(o["false_alarm_by_cost"]) == {"4.0", "5.0", "6.0", "8.0", "12.0"}
    st = length_stats([r["english"] for r in rows], [r["reference"] for r in rows])
    cal = {"yor": {"anchors": [0.5, 0.6, 0.7], "usable": True, "len_ratio": {"p05": 0.7, "p95": 1.3, **{k: st[k] for k in ("corpus", "s2")}},
                   "omission": {"cost": 6.0}}}
    p = tmp_path / "cal.json"
    p.write_text(json.dumps(cal))
    sh = ContentShaper.from_file(str(p))
    assert sh.length_model("yor").omit_cost == 6.0 and sh.length_model("yor").extra_cost == 4.0              # only the omission side is calibrated
    assert sh.length_model("yor", 3.0, 3.0).omit_cost == 3.0                                                # config value overrides the calibrated one
