# SPDX-License-Identifier: AGPL-3.0-or-later
"""D-61: anchors v2 (rl/textstats.py, rl/anchors_v2.py, rl/anchors_calibrate_v2.py, the reward hooks): substring copy detection, graded entities, stop-word profile, padding."""
import json
import random

import numpy as np
import pytest

from gemma_mt.fake_tokenizer import FakeGemmaTokenizer, SPECIALS
from gemma_mt.rl import anchors_attacks as AT
from gemma_mt.rl import anchors_calibrate_v2 as K
from gemma_mt.rl import anchors_v2 as V
from gemma_mt.rl import textstats as T
from gemma_mt.rl.comet import FakeScorer
from gemma_mt.rl.reward import RewardConfig, Sample, TranslationReward
from gemma_mt.rl.reward_shaping import ContentShaper

TOK = FakeGemmaTokenizer()
EOT = SPECIALS["<end_of_turn>"]
SRC = "Ada Obi bought three goats and four hens at the village market for five hundred naira on Monday morning."
GOOD = "Ada Obi ra ewure mẹta ati adiẹ mẹrin ni ọja abule fun ọgọrun marun naira ni owurọ ọjọ Aje."


# ------------------------------------------------------------------------------------------------ tokenisation / protection
def test_source_tokens_protect_numbers_urls_names_acronyms_and_connectors():
    st = T.source_tokens("Visit www.who.int on 5 May: Dr. Tedros of the WHO met the University of Oxford team in 2020.")
    prot = dict(zip(st.words, st.prot))
    assert prot["5"] and prot["2020"] and prot["who"] and prot["tedros"] and prot["oxford"] and prot["university"]
    assert not prot["visit"] and not prot["team"] and not prot["met"] and not prot["in"]
    assert prot["www"] and prot["int"]                                             # URL tokens
    st = T.source_tokens("Stanford University School of Medicine announced a chip.")
    d = dict(zip(st.words, st.prot))
    assert d["stanford"] and d["university"] and d["school"] and d["of"] and d["medicine"] and not d["announced"]


def test_whitelist_words_are_protected():
    st = T.source_tokens("the radio and the computer", frozenset({"radio"}))
    assert dict(zip(st.words, st.prot))["radio"] and not dict(zip(st.words, st.prot))["computer"]


def test_geez_tokenisation_uses_whitespace_and_the_ethiopic_word_separator():
    a = "እንኳን ደህና መጡ ወደ አዲስ አበባ።"
    b = "እንኳን፡ደህና፡መጡ፡ወደ፡አዲስ፡አበባ።"                                    # U+1361 word separator, no spaces
    assert T.out_tokens(a) == T.out_tokens(b) == ["እንኳን", "ደህና", "መጡ", "ወደ", "አዲስ", "አበባ"]
    src = "Welcome to Addis Ababa, the capital city of Ethiopia and the seat of the African Union."
    r = T.copy_fractions(src, a, k=3, m=20)
    assert r.frac_w == 0.0 and r.frac_c == 0.0                                       # a Ge'ez translation shares no character run with the English source
    r = T.copy_fractions(src, b + " Welcome to Addis Ababa, the capital city of Ethiopia", k=3, m=20)
    assert r.frac_w > 0.2                                                           # an English tail IS found after the Ethiopic part


# ------------------------------------------------------------------------------------------------ copy fractions
def test_copy_fraction_exact_partial_none_and_protected_runs_ignored():
    src = "The committee approved the new budget for rural schools after a long debate on Tuesday."
    exact = T.copy_fractions(src, src, k=3, m=14)
    assert exact.frac_w == pytest.approx(1.0) and exact.frac_c == pytest.approx(1.0)
    half = T.copy_fractions(src, "Komiti gba isuna tuntun fun awon ile iwe igberiko lẹhin ariyanjiyan gigun. " + src[: len(src) // 2], k=3, m=14)
    assert 0.3 < half.frac_w < 0.7 and half.frac_c > 0.3
    none = T.copy_fractions(src, "Komiti gba isuna tuntun fun awon ile iwe igberiko lẹhin ariyanjiyan gigun ni ọjọ Isegun.", k=3, m=14)
    assert none.frac_w == 0.0 and none.frac_c == 0.0
    names = "Dr Ngozi Okonjo Iweala and Prof Wole Soyinka met the committee at Lagos State University on 12 March 2020 after lunch."
    r = T.copy_fractions(names, "Dr Ngozi Okonjo Iweala ati Prof Wole Soyinka pade komiti ni Lagos State University ni 12 March 2020 leyin ounje osan.", k=3, m=14)
    assert r.frac_w == 0.0                                                           # names / numbers / dates alone are not a copy
    assert T.copy_fractions("Hello there", "Hello there", k=3).frac_w is None        # fewer than 3 unprotected words: not judged


def test_copy_fraction_isolated_swapped_words_are_not_a_run():
    src = "Farmers in the northern region planted maize and cassava before the first rains arrived."
    r = T.copy_fractions(src, "Awon agbe ni region ariwa gbin maize ati cassava saaju ojo akoko de.", k=3, m=14)
    assert r.frac_w == 0.0                                                           # documented limit: scattered loanwords are below k=3


# ------------------------------------------------------------------------------------------------ Anchors2 blocks
def cal_v2(**over):
    c = {"pooled": {}, "languages": {
        "yor": {"entity2": {"num": {"min_n": 1, "enabled": True, "w": 0.8, "d": 0.2}, "name": {"min_n": 3, "enabled": True, "w": 0.25, "d": 0.5}},
                "copy": {"k": 3, "m": 14, "min_unprot": 2, "t_w": 0.15, "t_c": 0.2, "whitelist": ["market"], "enabled": True},
                "func": {"fw": ["ni", "ati", "ti", "fun", "ra", "si", "ọja"], "en_excl": [], "t_fw": [0.1] * 5, "t_en": [0.05] * 5, "t_ratio": [0.3] * 5, "t_fw_hi": [0.7] * 5,
                         "cap_en": 0.3, "cap_fw": 0.95, "enabled": True},
                "pad": {"has_bigrams": False, "t_gn": [None] * 5, "t_ttr": [0.6] * 5, "enabled": True}},
        "amh": {"copy": {"k": 3, "m": 14, "min_unprot": 2, "t_w": 0.1, "t_c": 0.1, "whitelist": [], "enabled": True}}}}
    c.update(over)
    return c


def test_copy_fraction_thresholds_and_apply_modes():
    A = V.Anchors2(cal_v2(), v2_detectors=("copy",), copy_scale=2.0)
    assert A.copy_fraction("yor", SRC, GOOD) == 0.0
    assert A.copy_fraction("yor", SRC, GOOD + " " + SRC) == pytest.approx(1.0)       # the whole source appended: 100 % of it is copied
    f = A.copy_fraction("yor", SRC, GOOD + " " + " ".join(SRC.split()[:9]))           # the first half of the source appended
    assert f is not None and 0.4 < f < 0.7
    assert A.copy_apply(1.5, f, -2.0) == pytest.approx(-2.0 * f)                     # cap: reward set to -2 x fraction
    assert A.copy_apply(-1.9, 0.01, -2.0) == pytest.approx(-1.9) or A.copy_apply(-1.9, 0.01, -2.0) <= -1.9
    assert A.copy_apply(1.5, 1.0, -2.0) == -2.0                                      # fraction 1 -> -2
    B = V.Anchors2(cal_v2(), v2_detectors=("copy",), copy_scale=2.0, copy_mode="subtract")
    assert B.copy_apply(1.5, 0.5, -2.0) == pytest.approx(0.5)
    assert B.copy_apply(-1.5, 0.5, -2.0) == -2.0                                     # never below the failure penalty
    assert A.copy_fraction("amh", SRC, "እንኳን ደህና መጡ ወደ አዲስ አበባ ኢትዮጵያ ዋና ከተማ ናት።") == 0.0
    assert A.copy_fraction("zzz", SRC, SRC) is None                                  # no block, no pooled prior -> off


def test_entity2_graded_penalty_scales_with_weight_and_channel_enable():
    A = V.Anchors2(cal_v2(), v2_detectors=("entity2",))
    src = "Ada Obi paid 500 naira and 20 goats in Lagos."
    pu, n = A.entity2_penalty("yor", src, "Ada Obi san 500 naira ati 20 ewure ni Lagos.")
    assert pu == 0.0 and n > 0
    pu, n = A.entity2_penalty("yor", src, "Ada Obi san owo ati ewure ni Lagos.")        # both numbers dropped (only 2 names: the name channel needs 3)
    assert pu == pytest.approx(0.8) and n == 2                                         # num: w 0.8 x deficit 1
    src3 = "Ada Obi paid 500 naira to Chinedu Okafor and Ngozi Eze in Lagos."
    pu, n = A.entity2_penalty("yor", src3, "Ada Obi san owo fun Chinedu Okafor ati Ngozi Eze ni Lagos.")
    assert n == 1 + 6 and pu == pytest.approx(0.8 * 1 / 7)                             # numbers dropped, every name kept: 1 of 7 entities carries the number weight
    cal = cal_v2()
    cal["languages"]["yor"]["entity2"]["num"]["enabled"] = False
    A = V.Anchors2(cal, v2_detectors=("entity2",))
    assert A.entity2_penalty("yor", src3, "owo ati ewure")[0] == pytest.approx(0.25 * 6 / 6)     # numbers off; names all dropped: w 0.25
    assert not V.Anchors2(cal_v2(), v2_detectors=()).enabled2("yor", "entity2")


def test_func_and_pad_deficits_human_like_vs_stuffed_vs_english():
    A = V.Anchors2(cal_v2(), v2_detectors=("func", "pad"))
    sw = T.out_tokens(SRC)
    good = T.out_tokens("Ada Obi ra ewure ati adie ni ọja abule fun owo ni owuro ọjọ ti Aje.")
    assert A.func_deficit("yor", sw, good) == 0.0 and A.pad_deficit("yor", good) == 0.0
    stuffed = T.out_tokens("omi igi ewe ina oru ojo osu odun ata isu eja aso bata ile iwe omo")
    assert A.func_deficit("yor", sw, stuffed) == 1.0                                 # no function word at all
    english = T.out_tokens("the committee of the town has been in the house for the winter and the spring")
    assert A.func_deficit("yor", sw, english) > 0.9                                   # English function words (also below the target share)
    fw_only = T.out_tokens("ni ati ni ti fun ra si ni ati ni")
    assert A.func_deficit("yor", sw, fw_only) > 0.5                                   # a bag of function words
    loop = T.out_tokens("ni ọja ni ọja ni ọja ni ọja ni ọja ni ọja")
    assert A.pad_deficit("yor", loop) > 0.5                                          # type / token ratio
    assert A.func_deficit("yor", sw, good[:5]) is None                                # too short to judge


def test_bigram_gain_is_deterministic_and_separates_order_from_bag():
    words = "a b c d e f g h i j k l".split()
    pairs = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
    tab = np.sort(np.array([V.bigram_hashes(list(p))[0] for p in pairs], dtype=np.uint32))
    o1, s1 = V.bigram_gain(tab, words)
    o2, s2 = V.bigram_gain(tab, words)
    assert (o1, s1) == (o2, s2) and o1 == 1.0 and s1 < 0.5                           # ordered text is attested, its shuffles mostly are not
    bag = list(reversed(words))[::2] + words[::2]
    ob, sb = V.bigram_gain(tab, bag)
    assert ob - sb < o1 - s1


def test_bucket_thresholds_tail_extrapolation_and_nearest_fill():
    rng = np.random.default_rng(0)
    b = np.array([1] * 300 + [2] * 10)
    v = np.concatenate([rng.normal(0.4, 0.05, 300), rng.normal(0.4, 0.05, 10)])
    t0 = K.bucket_thresholds(b, v, 0.01, True, lam=0.0)
    t1 = K.bucket_thresholds(b, v, 0.01, True, lam=1.5)
    assert t1[1] < t0[1] and t0[2] == t0[1] and t0[0] == t0[1]                        # extrapolated tail is lower; sparse buckets copy the nearest one
    th = K.bucket_thresholds(b, v, 0.01, False, lam=0.0)
    assert th[1] > 0.45


def test_fit_entity2_weights_shrink_with_the_human_drop_rate():
    rng = random.Random(0)

    def units(drop_p, n=200):
        out = []
        for i in range(n):
            src = f"Ada paid {rng.randint(10, 99)} naira to Chinedu Okafor and Ngozi Eze in Lagos."
            ref = "Ada san owo fun won ni ilu" if rng.random() < drop_p else src
            out.append({"src": src, "ref": ref})
        return out

    lo = K.fit_entity2(units(0.05))
    hi = K.fit_entity2(units(0.4))
    vhi = K.fit_entity2(units(0.8))
    assert lo["num"]["w"] > hi["num"]["w"] > 0 and lo["num"]["enabled"] and hi["num"]["enabled"]
    assert lo["num"]["w"] == pytest.approx(1 - lo["num"]["d"]) and hi["num"]["d"] > 0.3
    assert not vhi["num"]["enabled"]                                                 # humans drop > 60 %: the channel is off
    assert lo["name"]["w"] == pytest.approx((1 - lo["name"]["d"]) * 0.5)


def test_fit_copy_threshold_fp_and_detection_of_tails():
    rng = random.Random(1)
    def word(i, pre):
        return pre + "".join(chr(97 + (i // 26 ** k) % 26) for k in range(3))

    vocab = [word(i, "x") for i in range(300)]
    ens = [word(i, "e") for i in range(300)]
    us = []
    for i in range(300):
        s = " ".join(rng.choice(ens) for _ in range(14))
        r = " ".join(rng.choice(vocab) for _ in range(14))
        if i % 100 == 0:                                                              # a few human references that keep a source phrase
            r += " " + " ".join(s.split()[:4])
        us.append({"src": s, "ref": r})
    b = K.fit_copy(us, 3, 14)
    assert b["fp_cv"] <= 0.012 and b["enabled"] and b["detect"]["tail50"] > 0.9


def test_attacks_are_deterministic_and_length_matched():
    top = [[f"w{chr(97 + i // 26)}{chr(97 + i % 26)}x", 100 - i, 5] for i in range(200)]
    fw = [t[0] for t in top[:20]]
    us = [{"src": "a b c d e f g h", "ref": " ".join(t[0] for t in top[30:40])} for _ in range(5)]
    a1 = AT.make_attacks(us, top, fw, ["x y z " * 4], random.Random(3))
    a2 = AT.make_attacks(us, top, fw, ["x y z " * 4], random.Random(3))
    assert a1 == a2
    assert all(len(t.split()) == 10 for t in a1["stuff"]) and all(len(t.split()) == 30 for t in a1["stuff30"])
    assert all(set(t.split()) <= set(fw) for t in a1["fw_only"]) and a1["en_copy"][0] == us[0]["src"]


# ------------------------------------------------------------------------------------------------ the reward hooks
def ids(text):
    return TOK.encode(text, add_special_tokens=False) + [EOT]


def mk(tmp_path, calibration=None, **kw):
    path = tmp_path / "anchors_v2.json"
    path.write_text(json.dumps(calibration or cal_v2()))
    base = dict(anchors_enabled=True, anchors_calibration_path=str(path), consensus=True, chunking="sentences", anchors_entity_weight=0.5, anchors_repeat_weight=0.0, anchors_boiler_weight=0.0)
    base.update(kw)
    sh = ContentShaper.from_file(None)
    sh.content = {"yor": {"mode": "consensus"}, "hau": {"mode": "qe"}, "amh": {"mode": "qe"}}
    return TranslationReward(RewardConfig(**base), FakeScorer(), TOK, [EOT], shaper=sh)


def grp(lang, text, src=SRC, g=4):
    return [Sample(ids(f"{lang} {text}"), lang, src) for _ in range(g)]


def test_reward_v2_default_off_and_validation(tmp_path):
    base = mk(tmp_path)
    assert base.anchors is not None and not hasattr(base.anchors, "applies2")        # v1 object when the v2 flag is off
    with pytest.raises(ValueError):
        mk(tmp_path, anchors_copy_scale=2.0)                                          # v2 key without anchors_v2_enabled
    with pytest.raises(ValueError):
        mk(tmp_path, anchors_v2_enabled=True, anchors_copy_mode="nope")
    with pytest.raises(ValueError):
        mk(tmp_path, anchors_v2_enabled=True, anchors_func_weight=-1.0)
    with pytest.raises(ValueError):
        mk(tmp_path, anchors_v2_enabled=True, anchors_v2_modes=["bogus"])
    r0 = base(grp("yor", GOOD), group_size=4)
    rv = mk(tmp_path, anchors_v2_enabled=True)(grp("yor", GOOD), group_size=4)       # v2 on, every v2 weight 0: bit-identical
    assert r0.rewards == rv.rewards and r0.anchors_penalty == rv.anchors_penalty


def test_reward_copy_cap_partial_copy_and_control_language(tmp_path):
    kw = dict(anchors_v2_enabled=True, anchors_copy_scale=2.0)
    rw = mk(tmp_path, **kw)
    good = rw(grp("yor", GOOD), group_size=4)
    part = rw(grp("yor", GOOD + " " + " ".join(SRC.split()[:9])), group_size=4)
    whole = rw(grp("yor", GOOD + " " + SRC), group_size=4)
    assert good.anchors_penalty == [0.0] * 4 and good.stats["anchors/copy/penalised_rate"] == 0.0
    assert all(r < -0.5 for r in part.rewards) and all(p > 0 for p in part.anchors_penalty)      # half of the source copied -> reward <= -1.0
    assert part.stats["anchors/copy/penalised_rate"] == 1.0 and 0.3 < part.stats["anchors/copy/fraction_mean"] < 0.8
    assert whole.rewards == [rw.penalty] * 4                                                     # the whole source appended -> -2 x 1.0
    assert min(part.rewards) >= rw.penalty - 1e-9
    ctl = mk(tmp_path, anchors_v2_off_languages=["yor"], **kw)
    assert ctl(grp("yor", GOOD + " " + SRC), group_size=4).anchors_penalty == [0.0] * 4
    ctl2 = mk(tmp_path, anchors_off_languages=["yor"], **kw)
    assert ctl2(grp("yor", GOOD + " " + SRC), group_size=4).anchors_penalty == [0.0] * 4
    qe_only = mk(tmp_path, anchors_v2_modes=["blend"], **kw)
    assert qe_only(grp("hau", GOOD + " " + SRC), group_size=4).anchors_penalty == [0.0] * 4
    full = rw(grp("yor", SRC), group_size=4)                                          # the exact copy stays the hard gate (-2), not double penalised
    assert full.status == ["copy_source"] * 4 and full.rewards == [rw.penalty] * 4


def test_reward_func_pad_and_soft_entity_weights(tmp_path):
    rw = mk(tmp_path, anchors_v2_enabled=True, anchors_func_weight=0.3, anchors_pad_weight=1.0)
    stuffed = "omi igi ewe ina oru ojo osu odun ata isu eja aso bata ile iwe omo"
    base = mk(tmp_path)
    r0, r1 = base(grp("yor", stuffed), group_size=4), rw(grp("yor", stuffed), group_size=4)
    assert all(a > b for a, b in zip(r0.rewards, r1.rewards)) and r1.anchors_penalty[0] > 0.2
    assert r1.stats["anchors/func/penalised_rate"] == 1.0
    g = rw(grp("yor", GOOD), group_size=4)
    assert g.anchors_penalty == [0.0] * 4
    drop = "Ada Obi ra ewure ati adiẹ ni ọja abule fun owo ni owurọ ọjọ Aje."
    soft = mk(tmp_path, anchors_v2_enabled=True, anchors_entity_mode="soft", anchors_modes=["consensus"])
    strict = mk(tmp_path, anchors_modes=["consensus"])
    src = "Ada Obi bought 3 goats and 4 hens at the village market for 500 naira on Monday morning."
    a, b = soft(grp("yor", drop, src), group_size=4), strict(grp("yor", drop, src), group_size=4)
    assert all(p > 0 for p in a.anchors_penalty) and a.stats["anchors/entity2/penalised_rate"] == 1.0
    assert a.anchors_penalty[0] != b.anchors_penalty[0] or b.anchors_penalty[0] == 0.0
    assert min(a.rewards) >= soft.penalty - 1e-9


def test_reward_v2_is_deterministic_across_instances(tmp_path):
    kw = dict(anchors_v2_enabled=True, anchors_copy_scale=2.0, anchors_func_weight=0.3, anchors_pad_weight=1.0, anchors_entity_mode="soft")
    batch = grp("yor", GOOD + " " + SRC) + grp("yor", "omi igi ewe ina oru ojo osu odun ata isu eja aso bata ile iwe omo") + grp("yor", GOOD)
    o1 = mk(tmp_path, **kw)(batch, group_size=4)
    o2 = mk(tmp_path, **kw)(batch, group_size=4)                                       # a fresh instance (what a restart from a checkpoint builds)
    assert o1.rewards == o2.rewards and o1.anchors_penalty == o2.anchors_penalty


def test_reward_stuff_gate_is_a_failure_class_and_validated(tmp_path):
    stuffed = "omi igi ewe ina oru ojo osu odun ata isu eja aso bata ile iwe omo"
    with pytest.raises(ValueError):
        mk(tmp_path, anchors_v2_enabled=True, anchors_stuff_gate=0.9)                 # needs a func / pad weight
    rw = mk(tmp_path, anchors_v2_enabled=True, anchors_func_weight=0.3, anchors_pad_weight=1.0, anchors_stuff_gate=0.9)
    out = rw(grp("yor", stuffed) + grp("yor", GOOD), group_size=4)
    assert out.status[:4] == ["stuffed_output"] * 4 and out.rewards[:4] == [rw.penalty] * 4
    assert out.status[4:] == ["ok"] * 4 and out.stats["fail/stuffed_output"] == pytest.approx(0.5)
    off = mk(tmp_path, anchors_v2_enabled=True, anchors_func_weight=0.3, anchors_pad_weight=1.0)
    assert off(grp("yor", stuffed), group_size=4).status == ["ok"] * 4                # gate 0 = a graded penalty only
    ctl = mk(tmp_path, anchors_v2_enabled=True, anchors_func_weight=0.3, anchors_stuff_gate=0.9, anchors_v2_off_languages=["yor"])
    assert ctl(grp("yor", stuffed), group_size=4).status == ["ok"] * 4                # the control list is honoured by the gate


def test_v3_usable_false_keeps_a_detector_off_and_gate_ok_false_keeps_it_out_of_the_hard_gate(tmp_path):
    stuffed = "omi igi ewe ina oru ojo osu odun ata isu eja aso bata ile iwe omo"
    sw = T.out_tokens(SRC)
    c = cal_v2()
    c["languages"]["yor"]["func"]["usable"] = False
    A = V.Anchors2(c, v2_detectors=("func", "pad"))
    assert not A.enabled2("yor", "func") and A.func_deficit("yor", sw, T.out_tokens(stuffed)) is None and A.enabled2("yor", "pad")
    c = cal_v2()
    c["languages"]["yor"]["func"]["gate_ok"] = False
    c["languages"]["yor"]["pad"]["gate_ok"] = False
    A = V.Anchors2(c, v2_detectors=("func", "pad"))
    assert A.enabled2("yor", "func") and not A.gate_ok("yor", "func") and A.func_deficit("yor", sw, T.out_tokens(stuffed)) == 1.0     # still a graded penalty
    rw = mk(tmp_path, calibration=c, anchors_v2_enabled=True, anchors_func_weight=0.3, anchors_pad_weight=1.0, anchors_stuff_gate=0.9)
    out = rw(grp("yor", stuffed), group_size=4)
    assert out.status == ["ok"] * 4 and all(p > 0 for p in out.anchors_penalty)          # penalised, not failed
    assert V.Anchors2(cal_v2()).gate_ok("yor", "func")                                    # default: allowed


def test_segmental_type_token_ratio_is_length_independent_while_the_plain_one_is_not():
    rng = random.Random(0)
    vocab = [f"w{chr(97 + i // 26)}{chr(97 + i % 26)}" for i in range(400)]
    weights = [1.0 / (i + 1) for i in range(len(vocab))]                              # Zipf-like text
    short = rng.choices(vocab, weights, k=200)
    long_ = rng.choices(vocab, weights, k=1500)
    assert V.pad_ttr(long_, "plain") < 0.6 * V.pad_ttr(short, "plain")
    assert abs(V.pad_ttr(long_, "seg100") - V.pad_ttr(short, "seg100")) < 0.08
    assert V.pad_ttr(short[:60], "seg100") == V.pad_ttr(short[:60], "plain") and V.pad_ttr([], "seg100") == 1.0
    loop = ["a", "b", "c", "d"] * 400
    assert V.pad_ttr(loop, "seg100") < 0.1                                             # a loop stays a loop at any length
