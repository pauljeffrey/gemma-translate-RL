# SPDX-License-Identifier: AGPL-3.0-or-later
"""D-47 wiring (2026-11-01): the reference-free confidence rules inside TranslationReward (gate / trust / cap), default OFF, checkpoint state, the log-prob helpers.
CPU only, no torch (the fake tokenizer and scorer of the other reward tests)."""
import pytest

from gemma_mt.fake_tokenizer import FakeGemmaTokenizer, SPECIALS
from gemma_mt.rl import confidence as C
from gemma_mt.rl.comet import FakeScorer
from gemma_mt.rl.reward import RewardConfig, Sample, TranslationReward
from gemma_mt.rl.reward_shaping import ContentShaper

TOK = FakeGemmaTokenizer()
EOT = SPECIALS["<end_of_turn>"]
STOP = [EOT]
SRC = "The market opens at eight o'clock in the morning and the farmers from the village bring fresh vegetables to sell."
GOOD = ["Oja n si ni agogo mejo owuro, awon agbe lati abule si mu ewebe tutu wa lati ta.",
        "Oja n si ni agogo mejo owuro, awon agbe lati abule mu ewebe tutu wa lati ta.",
        "Oja n si ni agogo mejo ni owuro, awon agbe lati abule si mu ewebe tutu wa lati ta.",
        "Oja ni won n si ni agogo mejo owuro, awon agbe lati abule mu ewebe tutu wa lati ta.",
        "Oja n si ni agogo mejo owuro, awon agbe lati abule si mu ewebe tuntun wa lati ta.",
        "Oja n si ni agogo mejo owuro, awon agbe lati abule si mu ewebe tutu wa fun tita.",
        "Oja n si ni agogo mejo owuro, awon agbe lati abule si mu ewebe tutu wa lati ta.",
        "Ni agogo mejo owuro oja n si, awon agbe lati abule si mu ewebe tutu wa lati ta."]


def ids(text):
    return TOK.encode(text, add_special_tokens=False) + [EOT]


def shaper_with(content):
    sh = ContentShaper.from_file(None)
    sh.content = dict(content)
    sh.lid, sh.rel = {}, {}
    return sh


def reward(content=None, lid_fn=None, **kw):
    cfg = RewardConfig(chunking="sentences", consensus=True, **kw)
    return TranslationReward(cfg, FakeScorer(), TOK, STOP, lid=lid_fn, shaper=shaper_with(content or {"ach": {"mode": "consensus"}}))


def group(lang, texts, logp=-1.0, source=SRC):
    return [Sample(ids(f"{lang} {t}"), lang, source, logp=logp) for t in texts]


ON = dict(confidence=True, confidence_min_groups=1, confidence_relative=False)


# ------------------------------------------------------------------------------------------------ pure helpers
def test_completion_mean_logp_leaves_out_the_tag_and_the_end_token():
    assert abs(C.completion_mean_logp([-9.0, -1.0, -2.0, -0.1], [5, 6, 7, EOT], 1, [EOT]) - (-1.5)) < 1e-12
    assert abs(C.completion_mean_logp([-9.0, -8.0, -1.0, -3.0], [5, 6, 7, 8], 2, [EOT]) - (-2.0)) < 1e-12      # two-piece text tag, no end token kept
    assert C.completion_mean_logp(None, [1, 2], 1) is None and C.completion_mean_logp([-1.0], [5], 1, []) is None
    assert C.completion_mean_logp([-1.0, float("nan"), -3.0], [5, 6, 7], 0) == -2.0                              # non-finite values are ignored


def test_tag_prefix_len_for_text_and_single_token_tags():
    dec = lambda x: TOK.decode(x, skip_special_tokens=False)
    cands = {"yor", "<yor>", "<unused0>"}
    assert C.tag_prefix_len(dec, TOK.encode("<unused0> E kaaro", add_special_tokens=False), cands) == 1       # dedicated token (training packs since growth 1)
    assert C.tag_prefix_len(dec, TOK.encode("yor E kaaro", add_special_tokens=False), cands) == 3              # text tag: its pieces
    assert C.tag_prefix_len(dec, TOK.encode("Ekaaro o", add_special_tokens=False), cands) == 1                 # no tag at all: falls back to 1


def test_language_trust_missing_observations_and_tracker_state_roundtrip():
    assert C.language_trust(None, None, 0.6) == 0.0 and C.language_trust(None, None, 0.6, missing_ok=True) == 1.0
    assert C.language_trust(0.2, None, 0.6, missing_ok=True) == 0.0                      # an observed bad value still counts
    tr = C.ConfidenceTracker()
    f = C.group_features([True] * 8, [True] * 8, ["a b c"] * 8, [-1.0] * 8)
    for _ in range(25):
        tr.observe("ach", f)
    tr2 = C.ConfidenceTracker()
    assert tr2.load_state_dict(tr.state_dict()) == 1 and tr2.state["ach"].seen == 25 and tr2.trust("ach", relative=False) == tr.trust("ach", relative=False)
    assert tr2.load_state_dict(None) == 0
    with pytest.raises(ValueError):
        C.params_from_dict({"lid_lo_typo": 0.5})
    assert C.params_from_dict({"lid_lo": 0.3}).lid_lo == 0.3


# ------------------------------------------------------------------------------------------------ the reward with the rules
def test_off_by_default_is_bit_identical():
    base = reward()(group("ach", GOOD), group_size=8)
    off = reward(confidence=False)(group("ach", GOOD, logp=-4.0), group_size=8)         # logp present but the rules are off
    assert base.rewards == off.rewards and base.weights == off.weights and off.confidence == {} and off.conf_gate == []
    on = reward(**ON)(group("ach", GOOD), group_size=8)
    assert on.rewards == base.rewards and on.weights == base.weights                    # a healthy group: nothing changes
    assert on.confidence["confidence/groups"] == 1.0 and on.confidence["confidence/gate_rate"] == 0.0
    assert all(k.startswith("confidence/") for k in on.confidence) and on.confidence["confidence/trust/ach"] == 1.0 and abs(on.confidence["confidence/logp_mean/ach"] + 1.0) < 1e-9


def test_a_group_with_fewer_than_three_valid_samples_gets_zero_weight():
    bad = group("ach", GOOD)
    for i in range(2, 8):
        bad[i] = Sample(ids(f"ach {GOOD[i]}")[:-1], "ach", SRC, logp=-1.0)             # truncated: no end of turn
    out = reward(**ON)(bad, group_size=8)
    assert out.weights == [0.0] * 8 and set(out.conf_gate) == {"few_valid"} and out.confidence["confidence/gate_rate"] == 1.0 and out.confidence["confidence/gate_reason/few_valid"] == 1.0
    assert out.confidence["confidence/weight_zeroed_rate"] == 1.0
    kept = reward(**ON, confidence_gate=False, confidence_trust=False)(bad, group_size=8)     # gate and trust switched off: weights untouched
    assert kept.weights == [1.0] * 8 and kept.confidence["confidence/gate_rate"] == 0.0
    trust_only = reward(**ON, confidence_gate=False)(bad, group_size=8)                  # trust alone: 2 of 8 valid -> the valid-fraction ramp scales the weight, never to a gate
    assert all(0.0 < w < 1.0 for w in trust_only.weights)


def test_a_group_in_the_wrong_language_is_gated_by_the_glotlid_pass_rate():
    def lid(texts):                                                                      # accepts 3 of the 8 completions only: 3 valid, GlotLID pass rate 3/8 < 0.5
        return ["ach_Latn" if j < 3 else "eng_Latn" for j in range(len(texts))]
    r = reward(lid_fn=lid, lid_enabled=True, **ON)
    r.shaper.lid = {"ach": {"enabled": True, "labels": ["ach_Latn"], "min_chars": 15}}
    out = r(group("ach", GOOD), group_size=8)
    assert out.status.count("wrong_language") == 5 and set(out.conf_gate) == {"wrong_language"}
    assert out.weights == [0.0] * 8 and out.confidence["confidence/gate_reason/wrong_language"] == 1.0


def test_low_log_prob_makes_trust_zero_and_caps_the_consensus_reward():
    healthy = reward(**ON)(group("ach", GOOD, logp=-1.0), group_size=8)
    assert healthy.confidence["confidence/cap/ach"] == pytest.approx(2.25)                   # trust 1: the cap is the top of the range
    r = reward(**ON)
    out = r(group("ach", GOOD, logp=-4.5), group_size=8)                                    # the model gives its own samples a log-prob a language it cannot write would get
    assert out.confidence["confidence/trust/ach"] == 0.0 and out.weights == [0.0] * 8
    assert out.confidence["confidence/cap/ach"] == pytest.approx(0.0)                       # trust 0: capped at the rubric-only value
    assert max(out.rewards) <= 1e-9 and max(healthy.rewards) > 0.5                         # the same texts earn > 0 when trusted
    nocap = reward(**ON, confidence_cap=False)(group("ach", GOOD, logp=-4.5), group_size=8)
    assert max(nocap.rewards) > 0.5 and nocap.weights == [0.0] * 8


def test_neutral_until_enough_groups_and_without_log_probs():
    r = reward(confidence=True, confidence_min_groups=3, confidence_relative=False)
    outs = [r(group("ach", GOOD, logp=-4.5), group_size=8) for _ in range(3)]
    assert [o.confidence["confidence/trust/ach"] for o in outs] == [1.0, 1.0, 0.0]          # the first two groups only fill the tracker
    none = reward(**ON)(group("ach", GOOD, logp=None), group_size=8)                       # a sampler without log-probs: no information is no penalty
    assert none.confidence["confidence/trust/ach"] == 1.0 and "confidence/logp_mean/ach" not in none.confidence and none.weights == [1.0] * 8


def test_relative_form_waits_for_the_reference_languages():
    r = reward(confidence=True, confidence_min_groups=1, confidence_relative=True)
    assert r(group("ach", GOOD, logp=-4.5), group_size=8).confidence["confidence/trust/ach"] == 1.0       # < 5 reference languages: log-prob ramp skipped
    ra = reward(confidence=True, confidence_min_groups=1, confidence_relative=True, confidence_absolute_fallback=True)
    assert ra(group("ach", GOOD, logp=-4.5), group_size=8).confidence["confidence/trust/ach"] == 0.0


def test_greedy_evaluation_and_qe_languages_do_not_touch_the_tracker():
    r = reward(content={"ach": {"mode": "consensus"}, "yor": {"mode": "qe"}}, **ON)
    r(group("ach", GOOD, logp=-4.5), group_size=None)
    r(group("ach", GOOD[:1], logp=-4.5), group_size=1)
    assert r.conf.state == {}
    out = r(group("yor", GOOD, logp=-4.5), group_size=8)                                    # qe language: not in confidence_modes
    assert r.conf.state == {} and out.weights == [1.0] * 8 and out.confidence == {}


def test_checkpoint_state_roundtrip_through_the_reward():
    r = reward(**ON)
    for _ in range(3):
        r(group("ach", GOOD, logp=-1.2), group_size=8)
    st = r.conf_state()
    assert st["ach"]["seen"] == 3
    r2 = reward(**ON)
    assert r2.load_conf_state(st) == 1 and r2.conf.state["ach"].seen == 3 and r2.conf.state["ach"].logp_valid == pytest.approx(-1.2)
    assert reward().conf_state() is None and reward().load_conf_state(st) == 0              # off: nothing stored, nothing restored


def test_bad_modes_are_refused():
    with pytest.raises(ValueError):
        reward(confidence=True, confidence_modes=["consensus", "nope"])
