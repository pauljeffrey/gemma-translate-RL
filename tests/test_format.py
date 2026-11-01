# SPDX-License-Identifier: AGPL-3.0-or-later
import os

import pytest

from gemma_mt import format as fmt
from gemma_mt.fake_tokenizer import FakeGemmaTokenizer, SPECIALS

TOK = FakeGemmaTokenizer()


def test_instruction_template_is_literal_with_two_spaces():
    assert fmt.INSTRUCTION_TEMPLATE == "Translate the text to {african_language}:  {english_text}"
    assert ":  {english_text}" in fmt.INSTRUCTION_TEMPLATE and ":   " not in fmt.INSTRUCTION_TEMPLATE
    assert fmt.build_user_content("Good morning", "yor") == "Translate the text to Yoruba:  Good morning"
    assert fmt.build_user_content("  hi  ", "pcm") == "Translate the text to Pidgin:  hi"


def test_model_content_and_alias_normalisation():
    assert fmt.build_model_content("yor", " E kaaro ") == "yor E kaaro"
    assert fmt.build_model_content("SWA", "x") == "swh x"
    with pytest.raises(KeyError):
        fmt.build_model_content("zzz", "x")
    with pytest.raises(ValueError):
        fmt.build_model_content("yor", "   ")


def test_prompt_text_is_gemma3_template():
    got = fmt.prompt_text(TOK, "Good morning", "yor")
    assert got == ("<bos><start_of_turn>user\nTranslate the text to Yoruba:  Good morning<end_of_turn>\n"
                   "<start_of_turn>model\n")


def test_build_prompt_ids_end_with_generation_prompt_and_single_bos():
    ids = fmt.build_prompt("Good morning", "yor", TOK)
    assert ids[0] == SPECIALS["<bos>"] and ids[1] != SPECIALS["<bos>"]
    assert TOK.decode(ids).endswith("<start_of_turn>model\n")
    assert ids.count(SPECIALS["<bos>"]) == 1


def test_example_loss_only_on_iso_translation_and_end_of_turn():
    ex = fmt.build_example("Good morning", "yor", "E kaaro", TOK)
    assert len(ex.input_ids) == len(ex.labels)
    target_ids = [t for t, l in zip(ex.input_ids, ex.labels) if l != fmt.IGNORE_INDEX]
    assert TOK.decode(target_ids) == "yor E kaaro<end_of_turn>"
    assert target_ids[-1] == SPECIALS["<end_of_turn>"]
    # everything before the target is masked and is exactly the prompt
    assert ex.input_ids[: ex.prompt_len] == fmt.build_prompt("Good morning", "yor", TOK)
    assert all(l == fmt.IGNORE_INDEX for l in ex.labels[: ex.prompt_len])
    # the template's trailing newline is masked
    assert ex.tail_len == 1 and ex.labels[-1] == fmt.IGNORE_INDEX
    assert TOK.decode(ex.input_ids) == ("<bos><start_of_turn>user\nTranslate the text to Yoruba:  Good morning"
                                        "<end_of_turn>\n<start_of_turn>model\nyor E kaaro<end_of_turn>\n")
    assert ex.target_len == len(target_ids)


def test_unicode_translation_roundtrip():
    tr = "Ẹ káàárọ̀, báwo ni o ṣe wà?"
    ex = fmt.build_example("Good morning, how are you?", "yor", tr, TOK)
    target_ids = [t for t, l in zip(ex.input_ids, ex.labels) if l != fmt.IGNORE_INDEX]
    assert TOK.decode(target_ids[:-1], skip_special_tokens=True) == "yor " + tr


def test_template_check_accepts_gemma_and_rejects_other_templates():
    fmt.assert_template_matches_reference(TOK)

    class Other(FakeGemmaTokenizer):
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kw):
            s = "".join(f"<|{m['role']}|>{m['content']}</s>" for m in messages)
            return s + ("<|assistant|>" if add_generation_prompt else "")

    with pytest.raises(ValueError):
        fmt.assert_template_matches_reference(Other())


def test_reference_render_matches_real_gemma_template_when_jinja_available():
    jinja2 = pytest.importorskip("jinja2")
    path = os.path.join(os.path.dirname(__file__), "gemma3_chat_template.jinja")
    env = jinja2.Environment(trim_blocks=False, lstrip_blocks=False)

    def raise_exception(msg):
        raise ValueError(msg)

    env.globals["raise_exception"] = raise_exception
    tpl = env.from_string(open(path, encoding="utf-8").read().rstrip("\n"))
    cases = [
        [{"role": "user", "content": fmt.build_user_content("Good morning, friend.", "hau")}],
        [{"role": "user", "content": fmt.build_user_content("Hi", "ibo")}, {"role": "assistant", "content": "ibo Ndewo"}],
        [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "x"}],
    ]
    for msgs in cases:
        for gen in (True, False):
            want = tpl.render(messages=msgs, bos_token="<bos>", add_generation_prompt=gen)
            assert fmt.reference_render(msgs, gen) == want


@pytest.mark.parametrize("completion,iso,status,text", [
    ("yor E kaaro", "yor", "ok", "E kaaro"),
    ("  yor   E kaaro  ", "yor", "ok", "E kaaro"),
    ("hau Ina kwana", "yor", "wrong", "Ina kwana"),
    ("E kaaro o", "yor", "missing", "E kaaro o"),
    ("", "yor", "empty", ""),
    ("yor", "yor", "ok", ""),
    ("eng Good morning", "yor", "wrong", "Good morning"),
    ("yoruba E kaaro", "yor", "missing", "yoruba E kaaro"),
])
def test_strip_iso_prefix(completion, iso, status, text):
    r = fmt.strip_iso_prefix(completion, iso)
    assert (r.status, r.text) == (status, text)


def test_prefix_policies():
    assert fmt.prefix_violation("ok", "require") is False
    assert fmt.prefix_violation("missing", "require") is True
    assert fmt.prefix_violation("wrong", "require") is True
    assert fmt.prefix_violation("ok", "forbid") is True
    assert fmt.prefix_violation("missing", "forbid") is False
    assert fmt.prefix_violation("missing", "optional") is False
    with pytest.raises(ValueError):
        fmt.prefix_violation("ok", "bogus")


def test_stop_token_resolution_and_generation_config_repair():
    class GC:
        eos_token_id = 1

    stop = fmt.resolve_stop_tokens(TOK, GC())
    assert stop.end_of_turn_id == 106 and stop.ids == [106, 1]
    assert stop.warnings, "a generation_config without <end_of_turn> must be reported"
    gc = fmt.ensure_generation_config(GC(), stop, pad_id=0)
    assert gc.eos_token_id == [106, 1] and gc.pad_token_id == 0

    class NoEot(FakeGemmaTokenizer):
        def convert_tokens_to_ids(self, t):
            return self.unk_token_id

    with pytest.raises(ValueError):
        fmt.resolve_stop_tokens(NoEot())


def test_reverse_direction_same_template_with_english_pivot():
    assert fmt.build_user_content("E kaaro", "eng") == "Translate the text to English:  E kaaro"
    assert fmt.build_model_content("en", "Good morning") == "eng Good morning"
    assert fmt.directed_pair("Good morning", "E kaaro", "yor", "en2x") == ("Good morning", "yor", "E kaaro")
    assert fmt.directed_pair("Good morning", "E kaaro", "yor", "x2en") == ("E kaaro", "eng", "Good morning")
    with pytest.raises(ValueError):
        fmt.directed_pair("a", "b", "yor", "sideways")
    r = fmt.strip_iso_prefix("eng Good morning", "eng")
    assert (r.status, r.text) == ("ok", "Good morning")
    assert fmt.strip_iso_prefix("yor Good morning", "eng").status == "wrong"
    assert fmt.strip_iso_prefix("Good morning", "eng").status == "missing"


def test_reverse_example_loss_only_on_english_answer():
    from gemma_mt.fake_tokenizer import FakeGemmaTokenizer

    tok = FakeGemmaTokenizer()
    src, tag, tgt = fmt.directed_pair("Good morning", "E kaaro", "yor", "x2en")
    ex = fmt.build_example(src, tag, tgt, tok)
    assert ex.text.startswith("<bos><start_of_turn>user\nTranslate the text to English:  E kaaro<end_of_turn>")
    assert "<start_of_turn>model\neng Good morning<end_of_turn>" in ex.text
    assert all(l == fmt.IGNORE_INDEX for l in ex.labels[: ex.prompt_len])
    assert tok.decode([i for i in ex.labels if i != fmt.IGNORE_INDEX]) == "eng Good morning<end_of_turn>"


def test_pivot_is_not_a_table_language():
    from gemma_mt import languages as L

    assert "eng" not in L.ALL_ISOS and L.canonical_iso("eng") is None and not L.is_rl_supported("eng")
    assert L.target_tag_and_name("en") == ("eng", "English") and L.target_tag_and_name("yo") == ("yor", "Yoruba")
