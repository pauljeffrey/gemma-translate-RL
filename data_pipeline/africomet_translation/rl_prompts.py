"""Render SabiYarn RL translation prompts for English SOURCE WINDOWS (RL bin v2: the Gemma length-binned windows + the b2048 bin).

The path is the one `assemble.render_shard` took for the v1 bin, minus the data plumbing: balanced instruction languages (shuffled), a template of
the instruction pool (position prepend / append, 50 / 50 by default), `compose_prompt`, and the chat-template prompt text
`<s><|user|>PROMPT</s><|assistant|>` that rl/bin_data.build tokenizes.  Source language is always English (`eng`); the target is the language
of the window file.  Everything is seeded: `rng_for(seed, "rl_v2", stream, target)` with `stream` naming the batch of rows (e.g. the length bin),
so a rebuild gives the same prompts.

    from africomet_translation.rl_prompts import render_prompts
    rows = render_prompts(english_windows, "hau", seed=0, stream="b64")      # list of dicts: text, input, instruction, instruction_language, ...

The instruction pool is whatever `sft_new.pools_multi` registered (20 instruction languages, owner 2026-10-11).  `instruction_languages=` restricts it
(the v1 bin used the first seven: eng pcm yor hau ibo ewe twi).
"""
from __future__ import annotations

import random
from typing import Optional, Sequence

from .build import rng_for
from .instructions import compose_prompt, input_lang_tags, instruction_languages, make_instruction

SOURCE_LANG = "eng"
V1_INSTRUCTION_LANGUAGES = ("eng", "pcm", "yor", "hau", "ibo", "ewe", "twi")


def available_instruction_languages(restrict: Optional[Sequence[str]] = None) -> list[str]:
    """Registered instruction languages (import sft_new.pools_multi first to get the 20-language pools), optionally restricted, in registry order."""
    have = instruction_languages()
    if restrict is None:
        return have
    missing = [l for l in restrict if l not in have]
    if missing:
        raise ValueError(f"instruction language(s) without a template bank: {missing}; available: {have}")
    return [l for l in have if l in set(restrict)]


def chat_text(prompt: str) -> str:
    return f"<s><|user|>{prompt}</s><|assistant|>"


def render_prompts(english: Sequence[str], target: str, seed: int = 0, stream: str = "",
                   ins_langs: Optional[Sequence[str]] = None, position: Optional[str] = None) -> list[dict]:
    """One prompt per English window. Instruction languages are dealt out evenly (round-robin over `ins_langs`, then shuffled), the template, its
    position and the separator are drawn from the seeded generator. Returns dicts with: text (the tokenized string), input (the English source),
    prompt, instruction, instruction_language, instruction_position, output_structure (the scaffold the policy must emit before the translation)."""
    il = list(ins_langs) if ins_langs else instruction_languages()
    n = len(english)
    rng: random.Random = rng_for(seed, "rl_v2", stream, target)
    assign = [il[j % len(il)] for j in range(n)]
    rng.shuffle(assign)
    out = []
    for j in range(n):
        ins = make_instruction(assign[j], SOURCE_LANG, target, rng, position=position)
        prompt = compose_prompt(ins, english[j], rng)
        out.append({"text": chat_text(prompt), "input": english[j], "prompt": prompt, "instruction": ins.text,
                    "instruction_language": ins.language, "instruction_position": ins.position,
                    "output_structure": f"<|input_lang|>{input_lang_tags(SOURCE_LANG, ins.language, ins.position)}"
                                        f"<task_plan><translate></task_plan><|target_lang|><{target}><response>"})
    return out
