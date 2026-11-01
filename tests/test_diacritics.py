# SPDX-License-Identifier: AGPL-3.0-or-later
"""Diacritics score (rl/diacritics.py, D-31): a penalty for stripped / fabricated diacritics, never a bonus."""
import random
import unicodedata

import pytest

from gemma_mt.fake_tokenizer import FakeGemmaTokenizer, SPECIALS
from gemma_mt.rl import diacritics as D
from gemma_mt.rl.comet import FakeScorer
from gemma_mt.rl.reward import RewardConfig, Sample, TranslationReward
from gemma_mt.rl.reward_shaping import ContentShaper

TOK = FakeGemmaTokenizer()
EOT = SPECIALS["<end_of_turn>"]

# hand-written Yoruba (tone marks + under-dots) and Hausa-like / plain text: the calibration is built from these, not from FLORES+
YOR = [
    "Àwọn àgbẹ̀ láti Ibadan mú ẹ̀fọ́ tútù wá sí ọjà ní òwúrọ̀ ọjọ́ Mọ́ńdè.",
    "Ọmọ náà jókòó sí abẹ́ igi ńlá, ó sì ń kọ́ ẹ̀kọ́ rẹ̀ pẹ̀lú ìfarabalẹ̀.",
    "Ìjọba ti kéde pé àwọn ilé ẹ̀kọ́ gíga yóò ṣí padà ní oṣù tó ń bọ̀.",
    "Ó ṣòro fún mi láti gbàgbọ́ pé o ti padà dé láti ìrìn àjò rẹ̀ ní àná.",
    "Àwọn ọmọdé ń ṣeré lórí pápá, wọ́n sì ń kọrin ayọ̀ pẹ̀lú ara wọn.",
    "Ẹ̀gbọ́n mi ṣiṣẹ́ ní ilé ìwòsàn ńlá kan ní ìlú Èkó fún ọdún mẹ́wàá.",
    "Ọjà ńlá náà kún fún àwọn ènìyàn tí wọ́n ń ra ọbẹ̀ àti ẹran lọ́jọ́ Àìkú.",
    "Olùkọ́ wa sọ pé ìdánwò yóò wáyé ní ọ̀sẹ̀ tó ń bọ̀, nítorí náà a gbọ́dọ̀ múra.",
    "Àwọn onímọ̀ sáyẹ́ǹsì ti ṣàwárí ọ̀nà tuntun láti tọ́jú àìsàn ibà àti ikọ́ ẹ̀gbẹ.",
    "Ìyá mi ṣe oúnjẹ aládùn fún gbogbo ẹbí wa ní ọjọ́ ìbí àbúrò mi obìnrin.",
    "Ọ̀rọ̀ yìí ṣe pàtàkì gan-an, nítorí pé ó kan ọjọ́ iwájú àwọn ọmọ wa.",
    "Ẹ jọ̀wọ́, ẹ má ṣe jẹ́ kí a pàdánù àǹfààní tó ṣọ̀wọ́n yìí ní ọdún yìí.",
]
YOR_REFS = [s for s in YOR] * 3 + [s.replace("Àwọn", "Àwọn ọ̀rẹ́ mi àti") for s in YOR]      # 48 sentences
SRC = "The farmers from Ibadan bring fresh vegetables to the big market every single morning."


def strip_marks(t):
    out = []
    for c in unicodedata.normalize("NFD", t):
        if D.is_mark(c):
            continue
        out.append(D.SPECIAL.get(c, c))
    return unicodedata.normalize("NFC", "".join(out))


YOR_ENTRY = D.summarise(YOR_REFS)
assert YOR_ENTRY is not None


def test_density_is_computed_on_nfd_and_copies_the_restoration_logic():
    nfc = "Ọmọ náà jókòó"
    nfd = unicodedata.normalize("NFD", nfc)
    assert D.diacritic_units(nfc) == D.diacritic_units(nfd) > 0
    assert D.diacritic_density(nfc) == D.diacritic_density(nfd)
    assert D.diacritic_units("ɛ ɔ Ɛ") == 3 and D.diacritic_units("plain ascii text") == 0
    assert D.is_mark("́") and not D.is_mark("a")


def test_yoruba_is_a_diacritic_language_and_english_like_text_is_not():
    yor_entry = YOR_ENTRY
    assert yor_entry["applies"] and yor_entry["density"]["p50"] >= D.MIN_DENSITY
    assert yor_entry["threshold"] > 0.05
    plain = ["This is plain text without any special letters at all number %d." % i for i in range(40)]
    e = D.summarise(plain)
    assert e is not None and not e["applies"]
    assert D.deficit("This is plain text.", e) == 0.0                 # not a diacritic language: nothing is judged
    assert D.summarise(YOR_REFS[:5]) is None                          # too few references to calibrate


def test_correct_yoruba_is_not_penalised_stripped_yoruba_is():
    yor_entry = YOR_ENTRY
    good = YOR[0]
    assert D.deficit(good, yor_entry) == 0.0
    assert not any(D.flags(good, yor_entry).values())
    assert D.deficit(strip_marks(good), yor_entry) == pytest.approx(1.0)            # all marks gone: deficit 1
    # partially stripped: a proportional deficit between 0 and 1
    nfd = unicodedata.normalize("NFD", good)
    half = unicodedata.normalize("NFC", "".join(c for i, c in enumerate(nfd) if not (D.is_mark(c) and i % 2)))
    d = D.deficit(half, yor_entry)
    assert 0.0 < d < 1.0


def test_nfd_input_is_handled_like_nfc():
    yor_entry = YOR_ENTRY
    for t in (YOR[1], strip_marks(YOR[1])):
        assert D.deficit(unicodedata.normalize("NFD", t), yor_entry) == D.deficit(unicodedata.normalize("NFC", t), yor_entry)
        assert D.diacritic_report(unicodedata.normalize("NFD", t), "yor", {"yor": yor_entry})["density"] == pytest.approx(D.diacritic_density(t))


def test_random_sprinkled_marks_are_penalised():
    yor_entry = YOR_ENTRY
    rng = random.Random(0)
    allowed = [chr(int(h, 16)) for h in yor_entry["allowed"]["marks"]]
    foreign = ["̈", "̇", "̊", "̧"]
    n_hit = n_hit_f = 0
    for t in YOR[:10]:
        base = strip_marks(t)
        for pool, name in ((allowed, "allowed"), (foreign, "foreign")):
            out = []
            for c in unicodedata.normalize("NFD", base):
                out.append(c)
                if c.isalpha() and rng.random() < 0.3:
                    out.append(rng.choice(pool))
            sprinkled = unicodedata.normalize("NFC", "".join(out))
            d = D.deficit(sprinkled, yor_entry)
            n_hit += (d == 1.0) if name == "allowed" else 0
            n_hit_f += (d == 1.0) if name == "foreign" else 0
    assert n_hit >= 9 and n_hit_f == 10        # random placement of the language's own marks is caught by the unseen (letter, marks) clusters / over-decoration


def test_guards_foreign_orphan_over_decorated():
    yor_entry = YOR_ENTRY
    assert D.flags("Ọmọ̈ náà jókòó sí abẹ́ igi ńlá gan-an", yor_entry)["foreign_mark"]               # a diaeresis is not a Yoruba mark
    assert D.flags("Ọmọ náà ́ jókòó sí abẹ́ igi ńlá", yor_entry)["orphan_mark"]                       # a mark attached to a space
    heavy = "".join(c + "́" if c.isalpha() else c for c in "Ọmọ náà jókòó sí abẹ́ igi ńlá gan an")   # an accent on EVERY letter
    assert D.deficit(heavy, yor_entry) == 1.0
    assert D.deficit(YOR[3], yor_entry) == 0.0


def test_diacritic_report_is_a_plain_reusable_function():
    yor_entry = YOR_ENTRY
    rep = D.diacritic_report(strip_marks(YOR[2]), "yor", {"yor": yor_entry})
    assert rep["applies"] and rep["deficit"] == pytest.approx(1.0) and rep["density"] == 0.0 and rep["lang"] == "yor"
    assert set(rep["flags"]) == {"foreign_mark", "orphan_mark", "unseen_cluster", "over_decorated"}
    ok = D.diacritic_report(YOR[2], "yor", {"yor": yor_entry})
    assert ok["deficit"] == 0.0 and ok["density"] > 0.1
    assert D.diacritic_report("anything", "xxx", {})["deficit"] == 0.0                     # unknown language: nothing to judge
    # the source-aware rule: an English chunk with < 3 content words is not judged
    assert D.diacritic_report(strip_marks(YOR[2]), "yor", {"yor": yor_entry}, src="Yes, sir.")["deficit"] == 0.0


def test_short_chunks_get_no_density_deficit_but_still_the_guards():
    yor_entry = YOR_ENTRY
    assert D.deficit("Ẹ ṣé o", yor_entry) == 0.0 and D.deficit("E se o", yor_entry) == 0.0           # < 25 characters
    assert D.deficit("Ë se o", yor_entry) == 1.0


def test_yoruba_references_that_are_partly_unmarked_use_the_marked_sentences_for_the_threshold():
    mixed = YOR_REFS + [strip_marks(t) for t in YOR] * 3              # 36 of 84 references carry no marks at all (FLORES+ Yoruba: 15%)
    e = D.summarise(mixed)
    assert e["density"]["p10"] == 0.0 and e["threshold_source"] == "p10_of_marked_sentences" and e["threshold"] > 0.1
    assert D.deficit(strip_marks(YOR[0]), e) == pytest.approx(1.0)
    # sparse special letters (Hausa-like): whole sentences without any are normal -> no density penalty, guards only
    sparse = ["Kowane mutum yana da ƙwarin gwiwa a cikin zuciyarsa kuma ya yi aiki tuƙuru number %d." % i if i % 3 != 0 else
              "Yara suna wasa a cikin gida kuma mahaifiyarsu tana dafa abinci number %d." % i for i in range(60)]
    e2 = D.summarise(sparse)
    assert e2["applies"] and e2["threshold"] == 0.0 and e2["threshold_source"] == "disabled"
    assert D.deficit("Yara suna wasa a cikin gida kuma mahaifiyarsu tana dafa abinci.", e2) == 0.0


def test_extra_dev_sets_only_widen_the_inventory():
    e = D.summarise(YOR_REFS, extra_refs=["Ọ̩mọ náà"])           # U+0329 (MAFAND-style under-line) is not in the main references
    assert "0329" in e["allowed"]["marks"]
    e0 = D.summarise(YOR_REFS)
    assert "0329" not in e0["allowed"]["marks"] and e["density"] == e0["density"]


# ------------------------------------------------------------------------------------------ inside the reward
def _shaper(entry):
    sh = ContentShaper.from_file(None)
    sh.diacritics = {"yor": entry}
    return sh


def _reward(entry, **kw):
    cfg = RewardConfig(chunking="sentences", diacritic_penalty=kw.pop("penalty", 1.0), **kw)
    return TranslationReward(cfg, FakeScorer(), TOK, [EOT], shaper=_shaper(entry))


def _sample(text, iso="yor", src=SRC):
    return Sample(TOK.encode(text, add_special_tokens=False) + [EOT], iso, src)


def test_reward_penalises_stripped_output_not_correct_output():
    yor_entry = YOR_ENTRY
    r = _reward(yor_entry)
    good = "yor Àwọn àgbẹ̀ láti Ibadan mú ẹ̀fọ́ tútù wá sí ọjà ńlá ní gbogbo òwúrọ̀."
    out = r([_sample(good), _sample(strip_marks(good))])
    assert out.status == ["ok", "ok"]
    assert out.diacritic_deficit[0] == 0.0 and out.diacritic_deficit[1] == pytest.approx(1.0)
    assert out.rewards[1] < out.rewards[0] - 0.99                      # one unit of penalty (same QE: the fake scorer sees the same length)
    off = _reward(yor_entry, penalty=0.0)([_sample(strip_marks(good))])
    assert off.rewards[0] == pytest.approx(out.rewards[1] + 1.0)       # penalty 0 = off; the deficit is still measured
    assert off.diacritic_deficit[0] == pytest.approx(1.0)


def test_penalty_never_below_the_failure_floor_and_never_a_bonus():
    yor_entry = YOR_ENTRY
    r = _reward(yor_entry, penalty=50.0)
    out = r([_sample("yor " + strip_marks(YOR[0]), src=SRC)])
    assert out.rewards[0] >= r.penalty - 1e-9
    base = _reward(yor_entry, penalty=0.0)([_sample("yor " + YOR[0])])
    pen = _reward(yor_entry, penalty=1.0)([_sample("yor " + YOR[0])])
    assert pen.rewards == base.rewards                                  # correct output: identical, no bonus either way


def test_languages_without_diacritics_are_unaffected():
    yor_entry = YOR_ENTRY
    r = _reward(yor_entry)
    plain = "hau Manoma daga Ibadan suna kawo sabbin kayan lambu zuwa babbar kasuwa kowace safiya."
    base = TranslationReward(RewardConfig(chunking="sentences"), FakeScorer(), TOK, [EOT])
    a = r([_sample(plain, "hau")])
    b = base([_sample(plain, "hau")])
    assert a.rewards == b.rewards and a.diacritic_deficit == [None]


def test_source_with_too_few_content_words_is_not_judged():
    yor_entry = YOR_ENTRY
    r = _reward(yor_entry)
    out = r([_sample("yor " + strip_marks(YOR[4]), src="Thank you, sir.")])
    assert out.diacritic_deficit == [0.0]


def test_stats_and_per_language_logs():
    yor_entry = YOR_ENTRY
    r = _reward(yor_entry)
    out = r([_sample("yor " + YOR[0]), _sample("yor " + strip_marks(YOR[0]))])
    assert out.stats["diacritic/measured_rate"] == 1.0
    assert out.stats["diacritic/penalised_rate"] == 0.5
    assert out.stats["diacritic/deficit/yor"] == pytest.approx(0.5)
