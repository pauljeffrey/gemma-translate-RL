# SPDX-License-Identifier: AGPL-3.0-or-later
"""D-29: the reward sees the completion and the source in Unicode NFC, so a policy cannot move character-based rules by sampling decomposed text."""
import unicodedata

from gemma_mt.fake_tokenizer import FakeGemmaTokenizer, SPECIALS
from gemma_mt.rl import lexical
from gemma_mt.rl.comet import FakeScorer
from gemma_mt.rl.reward import RewardConfig, Sample, TranslationReward
from gemma_mt.rl.reward_shaping import ContentShaper
from gemma_mt.rl.text import decode_completion, nfc

TOK = FakeGemmaTokenizer()
EOT = SPECIALS["<end_of_turn>"]
STOP = [EOT]
YOR = "Àwọn àgbẹ̀ láti Ibadan mú ẹ̀fọ́ tútù wá sí ọjà ńlá ní gbogbo òwúrọ̀ ọjọ́ Mọ́ńdè."
FON = "Mɛ̌ɖé ɖò gbɛ̀ ɔ̌ mɛ, é nɔ̌ ɖò xwé tɔ́n mɛ, bɔ é na ɖɔ ɖɔ xɔ́ntɔn tɔ́n lɛ́ ɔ̀ nɔ̌ gbɔ̌n."
SRC = "The farmers from Ibadan bring fresh vegetables to the big market every single morning."


def ids(text):
    return TOK.encode(text, add_special_tokens=False) + [EOT]


class Const:
    def __init__(self, v=0.70):
        self.v = v
        self.seen = []

    def __call__(self, s, m, r=None):
        self.seen.append((list(s), list(m)))
        return [self.v] * len(s)


def _shaper():
    sh = ContentShaper.from_file(None)
    sh.len_window = {"yor": (0.77, 1.63), "fon": (0.7, 1.7)}
    sh.lexical = {"yor": {"anchor_precision": 0.33, "cognate_precision": 0.2}}
    return sh


def _cfg(**kw):
    return RewardConfig(chunking="sentences", length_penalty=1.0, lexical_penalty=1.0, **kw)


def test_decode_completion_is_nfc_and_idempotent():
    nfd = unicodedata.normalize("NFD", "yor " + YOR)
    assert nfd != unicodedata.normalize("NFC", nfd) and len(nfd) > len(unicodedata.normalize("NFC", nfd))
    d = decode_completion(TOK, ids(nfd), STOP)
    assert d.text == unicodedata.normalize("NFC", "yor " + YOR) and nfc(d.text) == d.text
    assert decode_completion(TOK, ids(nfd), STOP, normalise=False).text == nfd.strip()


def test_nfd_and_nfc_completions_get_exactly_the_same_reward_and_features():
    for lang, text in (("yor", YOR), ("fon", FON)):
        rewards, texts = [], []
        for form in ("NFC", "NFD"):
            sc = Const(0.70)
            r = TranslationReward(_cfg(), sc, TOK, STOP, shaper=_shaper())
            out = r([Sample(ids(unicodedata.normalize(form, f"{lang} {text}")), lang, SRC)])
            assert out.status == ["ok"]
            rewards.append(out.rewards[0])
            texts.append(out.texts[0])
            # what QE saw
            assert sc.seen[0][1][0] == unicodedata.normalize("NFC", sc.seen[0][1][0])
        assert rewards[0] == rewards[1] and texts[0] == texts[1]
        a, b = (lexical.features(SRC, t) for t in texts)
        assert a == b
        sh = _shaper()
        assert sh.length_excess(lang, len(SRC), len(texts[0])) == sh.length_excess(lang, len(SRC), len(texts[1]))


def test_nfd_source_is_normalised_too():
    src_nfd = unicodedata.normalize("NFD", "The café owner, José, brings crème brûlée to the market every single morning.")
    out_a = TranslationReward(_cfg(), Const(), TOK, STOP, shaper=_shaper())([Sample(ids("yor " + YOR), "yor", src_nfd)])
    out_b = TranslationReward(_cfg(), Const(), TOK, STOP, shaper=_shaper())(
        [Sample(ids("yor " + YOR), "yor", unicodedata.normalize("NFC", src_nfd))])
    assert out_a.rewards == out_b.rewards


def test_without_nfc_a_decomposed_completion_can_trip_the_character_length_rule():
    """The byte / character length effect: Yoruba text in NFD is ~15% longer in characters (and in UTF-8 bytes), so a length-ratio limit
    that the composed text respects can be crossed by emitting the same text decomposed."""
    nfd = unicodedata.normalize("NFD", YOR)
    nfc_ = unicodedata.normalize("NFC", YOR)
    assert len(nfd) > len(nfc_) and len(nfd.encode()) != len(nfc_.encode())
    s = len(nfc_) // 5 + 1                         # source length: nfc ratio just under max_len_ratio 5.0, nfd ratio above it
    assert len(nfc_) / s <= 5.0 < len(nfd) / s
    src = ("abcde fghij klmno pqrst uvwxy zabcd efghi jklmn " * 4)[:s]
    for use_nfc, expect_nfd in ((True, "ok"), (False, "length_ratio")):
        r = TranslationReward(RewardConfig(nfc=use_nfc, max_len_ratio=5.0), Const(), TOK, STOP, shaper=_shaper())
        a = r([Sample(ids("yor " + nfc_), "yor", src)]).status[0]
        b = r([Sample(ids("yor " + nfd), "yor", src)]).status[0]
        assert a == "ok" and b == expect_nfd
