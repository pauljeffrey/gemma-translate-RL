# SPDX-License-Identifier: AGPL-3.0-or-later
"""The prompt of the inter-African model: the en->X template with an EXPLICIT source language (D-46).

    user turn   :  Translate the text from {source_language} to {target_language}:  {text}
    model turn  :  {target_iso} {translation}<end_of_turn>

It is the literal en->X template (`format.INSTRUCTION_TEMPLATE`, two spaces after the colon) with "from {source_language} " inserted, so
  * the model turn, the stop token, the tag handling (`format.strip_iso_prefix`), the loss mask and the RL reward sanitisation are UNCHANGED
    (the answer still starts with the TARGET tag; the reward only ever looks at the answer);
  * the tokenizer needs no new special token: language NAMES are ordinary text (the same fixed names as the en->X prompt, `languages.py`);
  * English can be the source too (`Translate the text from English to Yoruba:`), so one model can replay the en->X and X->en data in the
    explicit form, or keep them in the implicit en->X form (`format.build_user_content`) when `source=None`.
Why name the source at all (the en->X model infers it): Nguni / Sotho-Tswana / Manding / Akan sources are near-identical to a bag of n-grams, and
for a thin direction an explicit source is the only thing that tells the model which of two languages it is reading; the RL prompt file also
needs a `src_iso` column anyway, and the reward's pivot-consistency check needs to know the source language.
`FORMAT_VERSION_X2X` is stored in the packed dataset's meta.json next to the en->X format version.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from gemma_mt import format as fmt
from gemma_mt.languages import PIVOT_ISO, PIVOT_NAME, get_language, is_pivot, target_tag_and_name

FORMAT_VERSION_X2X = 2
INSTRUCTION_TEMPLATE_FROM = "Translate the text from {source_language} to {target_language}:  {text}"
DIRECTION_X2X = 2              # value in direction.npy (0 = en2x, 1 = x2en as in the en->X packed format)


PROMPT_FORMS = ("explicit", "implicit")          # `Translate the text from X to Y:` | `Translate the text to Y:` (the en->X-compatible form)
PROMPT_FORM_SEED = "x2x-prompt-form-v1"


def prompt_form_of(example_id: str, explicit_share: float = 0.5, seed: str = PROMPT_FORM_SEED) -> str:
    """Prompt form of one training example, a pure function of its id (deterministic, independent of row order, of the run and of the machine):
    `explicit` for a fraction `explicit_share` of the ids, `implicit` for the rest (D-46, owner 2026-10-20: 50 / 50)."""
    from gemma_mt.afri2afri.text import split_key_fraction

    return PROMPT_FORMS[0] if split_key_fraction(example_id, 0, seed) < explicit_share else PROMPT_FORMS[1]


def example_id_of(group_key: str, src_iso: str, tgt_iso: str) -> str:
    """Id of a pair for the prompt-form hash: the English-group (or text-pair) key plus the ordered language pair. Both directions of the pair share the form."""
    return f"{group_key}|{src_iso}|{tgt_iso}"


def language_name(tag: str) -> str:
    """The fixed instruction name of a source or target language (African table language, or 'English')."""
    return target_tag_and_name(tag)[1]


def source_tag(tag: str) -> str:
    """Canonical tag of a source (African iso, or 'eng')."""
    return PIVOT_ISO if is_pivot(tag) else get_language(tag).iso


def build_user_content_x2x(text: str, src: Optional[str], tgt: str) -> str:
    """`Translate the text from {src} to {tgt}:  {text}`; `src=None` gives the implicit en->X form (`Translate the text to {tgt}:  {text}`)."""
    if src is None:
        return fmt.build_user_content(text, tgt)
    text = (text or "").strip()
    if not text:
        raise ValueError("empty source text")
    if source_tag(src) == target_tag_and_name(tgt)[0]:
        raise ValueError(f"source and target language are the same ({src})")
    return INSTRUCTION_TEMPLATE_FROM.format(source_language=language_name(src), target_language=language_name(tgt), text=text)


def build_messages_x2x(text: str, src: Optional[str], tgt: str, translation: Optional[str] = None) -> List[Dict[str, str]]:
    msgs = [{"role": "user", "content": build_user_content_x2x(text, src, tgt)}]
    if translation is not None:
        msgs.append({"role": "assistant", "content": fmt.build_model_content(tgt, translation)})
    return msgs


def prompt_text_x2x(tokenizer: Any, text: str, src: Optional[str], tgt: str) -> str:
    return fmt.render_chat(tokenizer, build_messages_x2x(text, src, tgt), add_generation_prompt=True)


def build_prompt_x2x(text: str, src: Optional[str], tgt: str, tokenizer: Any) -> List[int]:
    """Token ids for `generate()`: ends with the generation prompt, so the model's first token is the target tag."""
    ids = fmt._encode(tokenizer, prompt_text_x2x(tokenizer, text, src, tgt))
    fmt._check_single_bos(tokenizer, ids)
    return ids


def build_example_x2x(text: str, src: Optional[str], tgt: str, translation: str, tokenizer: Any) -> fmt.Example:
    """Training example: loss ONLY on `{tgt_iso} {translation}<end_of_turn>` (same construction as `format.build_example`)."""
    msgs = build_messages_x2x(text, src, tgt, translation)
    prompt = fmt.render_chat(tokenizer, msgs[:1], add_generation_prompt=True)
    full = fmt.render_chat(tokenizer, msgs, add_generation_prompt=False)
    if not full.startswith(prompt):
        raise ValueError("chat template: the full conversation does not start with the generation prompt")
    suffix = full[len(prompt):]
    cut = suffix.rfind(fmt.END_OF_TURN)
    if cut < 0:
        raise ValueError(f"no {fmt.END_OF_TURN} in the rendered model turn: {suffix!r}")
    target_str = suffix[: cut + len(fmt.END_OF_TURN)]
    tail_str = suffix[cut + len(fmt.END_OF_TURN):]
    p_ids = fmt._encode(tokenizer, prompt)
    t_ids = fmt._encode(tokenizer, target_str)
    tail_ids = fmt._encode(tokenizer, tail_str) if tail_str else []
    fmt._check_single_bos(tokenizer, p_ids)
    eot = tokenizer.convert_tokens_to_ids(fmt.END_OF_TURN)
    if not t_ids or t_ids[-1] != eot:
        raise ValueError(f"the target does not end with the {fmt.END_OF_TURN} token id ({eot})")
    ids = p_ids + t_ids + tail_ids
    labels = [fmt.IGNORE_INDEX] * len(p_ids) + list(t_ids) + [fmt.IGNORE_INDEX] * len(tail_ids)
    return fmt.Example(ids, labels, len(p_ids), len(t_ids), len(tail_ids), text=full)
