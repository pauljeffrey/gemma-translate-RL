# SPDX-License-Identifier: AGPL-3.0-or-later
"""End-to-end check of the SINGLE-TOKEN language-tag mode with the REAL (renamed) Gemma tokenizer: prompt -> example -> completion -> decode -> strip -> reward -> confidence log-prob.

Skips without `transformers` or without the renamed tokenizer directory (artifacts/tokenizer_gemma3_tags, or $GEMMA_MT_TAG_TOKENIZER).
Tag ids (see reports/data/tag_token_map_2026-11-01.csv): yor 6, hau 7, kon 58, adh 262118.
"""
import os

import pytest

import gemma_mt.format as fmt
from gemma_mt import tags
from gemma_mt.languages import ALL_ISOS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKDIR = os.environ.get("GEMMA_MT_TAG_TOKENIZER", os.path.join(ROOT, "artifacts", "tokenizer_gemma3_tags"))
EXPECT = {"yor": 6, "hau": 7, "kon": 58, "adh": 262118}
SRC = "The market opens at eight o'clock and the farmers bring fresh vegetables."
GOOD = "Oja n si ni agogo mejo, awon agbe si mu ewebe tutu wa."

_TOK = {}


def _tok():
    pytest.importorskip("transformers")
    if not os.path.isdir(TOKDIR):
        pytest.skip(f"renamed tokenizer directory not found: {TOKDIR}")
    if "t" not in _TOK:
        from transformers import AutoTokenizer
        _TOK["t"] = AutoTokenizer.from_pretrained(TOKDIR)
    return _TOK["t"]


def _stop(tok):
    return [tok.convert_tokens_to_ids("<end_of_turn>"), tok.eos_token_id]


def _id_of(iso):
    k = tags.unused_index(tags.tag_slot(iso))
    return k + 6 if k < 99 else 256001 + (k - 99)


def test_all_125_tags_are_single_renamed_tokens_with_the_documented_ids():
    tok = _tok()
    assert len(tok) == 262145
    for iso in ALL_ISOS:
        i = tok.convert_tokens_to_ids(f"<{iso}>")
        assert i == _id_of(iso), (iso, i)
        assert tok(f"<{iso}>", add_special_tokens=False)["input_ids"] == [i]
    for iso, i in EXPECT.items():
        assert _id_of(iso) == i


@pytest.mark.parametrize("iso", ["yor", "kon", "adh"])
def test_prompt_ends_with_the_generation_prompt_and_the_example_target_starts_with_the_tag_token(iso):
    tok = _tok()
    with fmt.tag_mode("single_token"):
        p = fmt.build_prompt("Good morning", iso, tok)
        assert tok.decode(p).endswith("<start_of_turn>model\n")
        ex = fmt.build_example("Good morning", iso, "E kaaro", tok)
        assert ex.labels[ex.prompt_len] == EXPECT[iso] and ex.input_ids[ex.prompt_len] == EXPECT[iso]
        assert all(x == fmt.IGNORE_INDEX for x in ex.labels[:ex.prompt_len])
        tgt = tok.decode([x for x in ex.labels if x != fmt.IGNORE_INDEX])
        assert tgt.startswith(f"<{iso}>") and "<end_of_turn>" in tgt
        fmt.assert_template_matches_reference(tok)


def test_completion_decode_and_strip_accept_token_and_text_forms_and_reject_a_wrong_tag():
    from gemma_mt.rl.text import decode_completion
    tok = _tok(); stop = _stop(tok); eot = stop[0]
    with fmt.tag_mode("single_token"):
        ok = tok.encode("<yor> E kaaro", add_special_tokens=False) + [eot]
        assert ok[0] == 6
        d = decode_completion(tok, ok, stop)
        assert d.text.startswith("<yor>") and not d.special_ids_stripped and d.finished
        r = fmt.strip_iso_prefix(d.text, "yor")
        assert r.status == "ok" and r.text == "E kaaro"
        wrong = decode_completion(tok, tok.encode("<hau> E kaaro", add_special_tokens=False) + [eot], stop)
        assert fmt.strip_iso_prefix(wrong.text, "yor").status == "wrong"
        assert fmt.strip_iso_prefix("<unused0> E kaaro", "yor").status == "ok"            # the original-tokenizer form of the same tag
        assert fmt.strip_iso_prefix("yor E kaaro", "yor").status == "ok"                 # text form: accepted in either mode (policy of format.strip_iso_prefix)
        assert fmt.strip_iso_prefix("E kaaro", "yor").status == "missing"


def test_reward_treats_the_single_token_tag_like_the_text_tag():
    from gemma_mt.rl.comet import FakeScorer
    from gemma_mt.rl.reward import RewardConfig, Sample, TranslationReward
    tok = _tok(); stop = _stop(tok); eot = stop[0]
    with fmt.tag_mode("single_token"):
        rw = TranslationReward(RewardConfig(prefix_policy="require"), FakeScorer(), tok, stop)
        enc = lambda s: tok.encode(s, add_special_tokens=False) + [eot]
        out = rw([Sample(enc(f"<yor> {GOOD}"), "yor", SRC), Sample(enc(f"<hau> {GOOD}"), "yor", SRC), Sample(enc(f"yor {GOOD}"), "yor", SRC), Sample(enc(GOOD), "yor", SRC),
                  Sample(enc(f"<yor> {GOOD}")[:-1], "yor", SRC)])
        assert out.status == ["ok", "bad_prefix", "ok", "bad_prefix", "no_end_of_turn"], out.status
        assert out.texts[0] == GOOD and out.texts[2] == GOOD                              # the tag (either form) is stripped before scoring
        assert out.rewards[0] > -2.0 and out.rewards[1] == -2.0 and out.rewards[3] == -2.0


def test_confidence_log_prob_mean_excludes_the_tag_and_the_end_token():
    pytest.importorskip("torch")
    from gemma_mt.rl.confidence import completion_mean_logp
    from gemma_mt.rl.train_rl import _tag_len
    tok = _tok(); stop = _stop(tok)
    with fmt.tag_mode("single_token"):
        for iso in ("yor", "kon", "adh"):
            ids = tok.encode(f"<{iso}> E kaaro", add_special_tokens=False) + [stop[0]]
            assert ids[0] == EXPECT[iso]
            n = _tag_len(tok, ids, iso)
            assert n == 1, (iso, n)
            lp = [-9.0] + [-1.0] * (len(ids) - 2) + [-0.01]
            assert completion_mean_logp(lp, ids, n, stop) == pytest.approx(-1.0)
