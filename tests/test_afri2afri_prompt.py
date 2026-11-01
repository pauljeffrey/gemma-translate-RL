# SPDX-License-Identifier: AGPL-3.0-or-later
import pytest

from gemma_mt import format as fmt
from gemma_mt.afri2afri import prompt as P
from gemma_mt.fake_tokenizer import FakeGemmaTokenizer

TOK = FakeGemmaTokenizer()


def test_user_content_is_the_en2x_template_with_an_explicit_source():
    c = P.build_user_content_x2x("Ẹ kaaro", "yor", "hau")
    assert c == "Translate the text from Yoruba to Hausa:  Ẹ kaaro"
    assert c.replace("from Yoruba ", "") == fmt.build_user_content("Ẹ kaaro", "hau")      # literally the old template + "from Yoruba "
    assert P.build_user_content_x2x("Good morning", "eng", "yor") == "Translate the text from English to Yoruba:  Good morning"
    assert P.build_user_content_x2x("Ẹ kaaro", None, "hau") == fmt.build_user_content("Ẹ kaaro", "hau")      # implicit (en->X) form kept


def test_aliases_and_same_language_and_empty():
    assert P.build_user_content_x2x("x", "swa", "ti") == "Translate the text from Swahili to Tigrinya:  x"
    with pytest.raises(ValueError):
        P.build_user_content_x2x("x", "yor", "yor")
    with pytest.raises(ValueError):
        P.build_user_content_x2x("   ", "yor", "hau")
    with pytest.raises(KeyError):
        P.build_user_content_x2x("x", "xxx", "hau")


def test_answer_starts_with_the_target_tag_and_strip_prefix_works():
    m = P.build_messages_x2x("Ẹ kaaro", "yor", "hau", "Ina kwana")
    assert m[1] == {"role": "assistant", "content": "hau Ina kwana"}
    assert fmt.strip_iso_prefix("hau Ina kwana", "hau").status == "ok"


def test_example_masks_prompt_and_ends_with_end_of_turn():
    ex = P.build_example_x2x("Ẹ kaaro", "yor", "hau", "Ina kwana", TOK)
    assert ex.prompt_len > 0 and ex.target_len > 0
    assert all(l == fmt.IGNORE_INDEX for l in ex.labels[: ex.prompt_len])
    assert ex.input_ids[ex.prompt_len + ex.target_len - 1] == TOK.convert_tokens_to_ids(fmt.END_OF_TURN)
    assert "from Yoruba to Hausa" in ex.text and ex.text.startswith("<bos><start_of_turn>user\n")
    assert ex.text.count("<bos>") == 1


def test_implicit_form_is_identical_to_the_en2x_example():
    a = P.build_example_x2x("Good morning", None, "yor", "E kaaro", TOK)
    b = fmt.build_example("Good morning", "yor", "E kaaro", TOK)
    assert a.input_ids == b.input_ids and a.labels == b.labels


def test_prompt_ids_end_with_the_generation_prompt():
    ids = P.build_prompt_x2x("Ẹ kaaro", "yor", "hau", TOK)
    ex = P.build_example_x2x("Ẹ kaaro", "yor", "hau", "Ina kwana", TOK)
    assert ids == ex.input_ids[: ex.prompt_len]


def test_prompt_form_is_deterministic_and_balanced():
    ids = [P.example_id_of(f"{i:016x}", "yor", "hau") for i in range(20000)]
    forms = [P.prompt_form_of(x) for x in ids]
    assert forms == [P.prompt_form_of(x) for x in ids]                           # a pure function of the id
    assert forms == [P.prompt_form_of(x) for x in reversed(ids[::-1])]
    share = forms.count("explicit") / len(forms)
    assert 0.49 < share < 0.51
    assert set(forms) == set(P.PROMPT_FORMS)
    assert P.prompt_form_of(ids[0], explicit_share=1.0) == "explicit" and P.prompt_form_of(ids[0], explicit_share=0.0) == "implicit"


def test_mixture_renders_both_forms_and_the_implicit_one_is_the_en2x_prompt():
    from gemma_mt.afri2afri import packing as PK

    items = [("Ẹ kaaro", "yor", "hau", "Ina kwana"), ("Ẹ kaaro", None, "hau", "Ina kwana")]
    (a, pa_, ta), (b, pb, tb) = PK.encode_batch(TOK, items)
    assert "from Yoruba to Hausa" in TOK.decode(a[:pa_]) and "from" not in TOK.decode(b[:pb])
    ex = fmt.build_example("Ẹ kaaro", "hau", "Ina kwana", TOK)                       # the old en->X builder, same text
    assert b == ex.input_ids and pb == ex.prompt_len and tb == ex.target_len
    assert ta == tb                                                                # the answer is identical in both forms
