# SPDX-License-Identifier: AGPL-3.0-or-later
"""D-37: the GlotLID guard of the reward (fake LID; the real model is exercised on the rental), the numpy-2-safe classifier wrapper, sibling groups."""
import pytest

from gemma_mt.fake_tokenizer import FakeGemmaTokenizer, SPECIALS
from gemma_mt.rl import lid as L
from gemma_mt.rl.comet import FakeScorer
from gemma_mt.rl.reward import FAILURES, RewardConfig, Sample, TranslationReward
from gemma_mt.rl.reward_shaping import ContentShaper

TOK = FakeGemmaTokenizer()
EOT = SPECIALS["<end_of_turn>"]
SRC = "The market opens at eight o'clock in the morning and the farmers from the village bring fresh vegetables to sell."
YOR = "Oja n si ni agogo mejo owuro, awon agbe lati abule si mu ewebe tutu wa lati ta."
SHORT = "Oja n si loni"


def ids(text):
    return TOK.encode(text, add_special_tokens=False) + [EOT]


def mk(lid_block, fake, **kw):
    sh = ContentShaper.from_file(None)
    sh.lid = lid_block
    return TranslationReward(RewardConfig(lid_enabled=True, **kw), FakeScorer(), TOK, [EOT], lid=fake, shaper=sh)


LID = {"yor": {"enabled": True, "labels": ["yor_Latn"], "min_chars": 0},
       "kin": {"enabled": True, "labels": ["kin_Latn", "run_Latn"], "min_chars": 0},
       "pcm": {"enabled": False, "labels": ["pcm_Latn"], "min_chars": 0},
       "swh": {"enabled": True, "labels": ["swh_Latn", "swc_Latn"], "min_chars": 60}}


def test_wrong_language_is_a_failure_and_right_or_sibling_labels_pass():
    fake = L.FakeLID(fn=lambda t: {"A": "yor_Latn", "B": "eng_Latn", "C": "run_Latn"}.get(t[0], "und"))
    r = mk(LID, fake)
    smp = [Sample(ids("yor A" + YOR), "yor", SRC), Sample(ids("yor B" + YOR), "yor", SRC), Sample(ids("kin C" + YOR), "kin", SRC)]
    out = r(smp)
    assert out.status == ["ok", "wrong_language", "ok"]                 # kin accepts the sibling label run_Latn
    assert out.rewards[1] == -2.0 and out.qe[1] is None and out.lid_label == ["yor_Latn", "eng_Latn", "run_Latn"]
    assert out.stats["fail/wrong_language"] == 1 / 3 and out.stats["fail/wrong_language/yor"] == 0.5 and "fail/wrong_language/kin" not in out.stats
    assert out.stats["lid/checked_rate"] == 1.0 and abs(out.stats["lid/fail_rate_of_checked"] - 1 / 3) < 1e-9 and out.stats["lid/fail_rate/yor"] == 0.5


def test_one_batched_call_for_all_samples_and_failed_rows_are_not_checked():
    fake = L.FakeLID(default="yor_Latn")
    r = mk(LID, fake)
    smp = [Sample(ids("yor " + YOR), "yor", SRC) for _ in range(6)] + [Sample(ids("yor " + YOR)[:-1], "yor", SRC)]       # the last one is truncated
    out = r(smp)
    assert len(fake.calls) == 1 and len(fake.calls[0]) == 6 and out.status[-1] == "no_end_of_turn" and out.lid_label[-1] is None


def test_languages_with_the_guard_off_or_unmeasured_and_the_minimum_length():
    fake = L.FakeLID(default="eng_Latn")
    r = mk(LID, fake, lid_min_chars=15)
    smp = [Sample(ids("pcm " + YOR), "pcm", SRC),                          # measured, guard OFF (recall < 0.90)
           Sample(ids("hau " + YOR), "hau", SRC),                          # not in the calibration at all: unmeasured -> no guard
           Sample(ids("yor " + SHORT), "yor", "The market is open."),                        # 13 characters: below lid_min_chars
           Sample(ids("swh " + YOR[:50]), "swh", SRC),                     # above the global minimum but below the language's own (60)
           Sample(ids("swh " + YOR), "swh", SRC)]
    out = r(smp)
    assert out.status == ["ok", "ok", "ok", "ok", "wrong_language"]
    assert sum(len(c) for c in fake.calls) == 1 and out.lid_label[:4] == [None] * 4


def test_without_lid_blocks_the_labels_of_languages_py_are_used_and_the_switch_off_disables_everything():
    fake = L.FakeLID(default="eng_Latn")
    sh = ContentShaper.from_file(None)
    r = TranslationReward(RewardConfig(lid_enabled=True), FakeScorer(), TOK, [EOT], lid=fake, shaper=sh)
    assert r([Sample(ids("yor " + YOR), "yor", SRC)]).status == ["wrong_language"]
    off = TranslationReward(RewardConfig(lid_enabled=False), FakeScorer(), TOK, [EOT], lid=fake, shaper=sh)
    assert off([Sample(ids("yor " + YOR), "yor", SRC)]).status == ["ok"]
    dis = TranslationReward(RewardConfig(lid_enabled=True, lid_disable=["yor"]), FakeScorer(), TOK, [EOT], lid=fake, shaper=ContentShaper.from_file(None))
    assert dis([Sample(ids("yor " + YOR), "yor", SRC)]).status == ["ok"]
    assert "wrong_language" in FAILURES


# ------------------------------------------------------------------------------------------------ the classifier wrapper
class FakeFastText:
    """Mimics the C++ binding: f.predict(line + newline, k, threshold, on_unicode_error) -> [(prob, '__label__x'), ...]. `predict` (the numpy path) is broken on purpose."""

    def __init__(self):
        self.seen = []
        self.f = self

    def predict(self, *a, **k):               # attribute used by the real wrapper: model.f.predict
        line, k_, thr, err = a
        assert line.endswith("\n") and "\n" not in line[:-1] and thr == 0.0 and err == "strict"
        self.seen.append(line)
        lab = "__label__eng_Latn" if line.startswith("Hello") else "__label__yor_Latn"
        return [(0.9, lab), (0.05, "__label__kin_Latn")][:k_]

    def get_labels(self):
        return ["__label__eng_Latn", "__label__yor_Latn"]


def _classifier():
    c = L.GlotLIDClassifier()
    c._model = FakeFastText()
    return c


def test_classifier_uses_the_binding_not_numpy_and_caches():
    c = _classifier()
    out = c(["Hello there\nmy friend", "Mo fe lo si oja", "Hello there\nmy friend", ""])
    assert out == ["eng_Latn", "yor_Latn", "eng_Latn", "und"]
    assert len(c._model.seen) == 2 and c.calls == 2                       # newline -> space, the duplicate and the empty text never reach the model
    assert c._model.seen[0] == "Hello there my friend\n"
    assert c.top(["Mo fe lo"], 2)[0] == [("yor_Latn", 0.9), ("kin_Latn", 0.05)]
    assert c.labels() == ["eng_Latn", "yor_Latn"]
    assert L.clean_for_fasttext("a " * 800, 100).endswith("a") and len(L.clean_for_fasttext("a " * 800, 100)) <= 100


def test_selftest_catches_a_broken_model():
    good = L.FakeLID(fn=lambda t: "eng_Latn" if t.startswith("Hello") else ("yor_Latn" if t.startswith("Mo") else "hau_Latn"))
    L.selftest(good)
    with pytest.raises(RuntimeError):
        L.selftest(L.FakeLID(default="und"))


def test_sibling_groups_only_contain_close_languages():
    assert "xho_Latn" in L.siblings_of(["zul_Latn"]) and "nbl_Latn" in L.siblings_of(["ssw_Latn"]) and "tsn_Latn" in L.siblings_of(["sot_Latn"])
    assert L.siblings_of(["kin_Latn"]) == ("run_Latn",) and "swc_Latn" in L.siblings_of(["swh_Latn"]) and "dyu_Latn" in L.siblings_of(["bam_Latn"])
    for lab in ("eng_Latn", "fra_Latn", "por_Latn", "ara_Arab"):
        assert not any(lab in g for g in L.SIBLING_GROUPS)               # English and the other big languages never become acceptable
    assert L.siblings_of(["yor_Latn"]) == ()


def test_the_shipped_calibration_lid_blocks_are_consistent():
    import json
    import os

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "calibration", "shaping_flores_dev_v6.json")                       # v6 = v5 + the 8 consensus languages of 2026-11-01
    if not os.path.exists(path):
        pytest.skip("calibration v6 not built yet")
    cal = json.load(open(path, encoding="utf-8"))
    from gemma_mt.languages import SUPPORTED_RL

    for k in SUPPORTED_RL:
        b = cal[k]["lid"]
        assert isinstance(b["enabled"], bool) and b["labels"] and b["min_chars"] >= 15
        if b["enabled"]:
            assert b["recall"] >= 0.90 and b["n"] >= 300                    # the owner's bar: guard only where the human recall is >= 0.90
        for lab in b["labels"]:
            assert not lab.startswith("eng_")                                # never accept English
    assert cal["_meta_v5"]["lid"]["enabled"] and "pcm" in cal["_meta_v5"]["lid"]["disabled"]
