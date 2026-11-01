# SPDX-License-Identifier: AGPL-3.0-or-later
"""README.md (model card) generator.

Every example in the card is produced from gemma_mt.format, so the card cannot drift from what the model was
trained on. The licence identifier is `gemma` by default: the model is a derivative of Gemma 3 and therefore
under the Gemma Terms of Use (https://ai.google.dev/gemma/terms); the licence of this repository's CODE
(AGPL-3.0-or-later) says nothing about the model weights. Data generated with the model is covered by the
additional contract terms in OUTPUT_DATA_TERMS (a grant to the publisher; no copyleft on anyone's models).
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional, Sequence

from gemma_mt import format as fmt
from gemma_mt.languages import ALL_ISOS, LANGUAGES, SUPPORTED_RL, UNSUPPORTED

DEFAULT_BASE = "google/gemma-3-270m-it"
GEMMA_TERMS_URL = "https://ai.google.dev/gemma/terms"
DEFAULT_LICENSE_NOTE = (
    "This model is a derivative of Gemma 3 and is provided under, and subject to, the Gemma Terms of Use "
    f"({GEMMA_TERMS_URL}), including the Gemma Prohibited Use Policy. You must comply with those terms when you use, "
    "modify or redistribute it. The licence of the training code in the source repository does not change the "
    "terms of the weights."
)
OUTPUT_DATA_TERMS = (
    "Additional terms for data generated with this model (offered by the publisher of this fine-tune as terms for its "
    "modifications; they apply in addition to the Gemma Terms of Use and never override them). If you publish, share or "
    "hand over to others a dataset whose texts were produced with this model, you grant the publisher of this model a "
    "worldwide, non-exclusive, perpetual, irrevocable, royalty-free licence to use, copy, modify and train models on "
    "that dataset for any purpose, including closed or commercial models, without any obligation on the publisher to "
    "release the resulting models or to pay for the data. You keep all your own rights and may use, publish or sell the "
    "dataset yourself. Nothing in these terms obliges you to release your own models or data that you do not share. "
    "These terms are a contract offer, not a statement of law: they bind those who accept them (for example by "
    "accepting the access conditions of a gated repository), and the Gemma terms continue to apply to models trained "
    "on Gemma outputs. This is not legal advice."
)
EXAMPLE_EN = "Good morning, how are you today?"
EXAMPLE_ISO = "yor"


def _length_section() -> str:
    """Source-length statistics of the SFT training data (hub/source_length_stats.json, written by the stats script from the packed
    training split); empty when the file is absent."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "source_length_stats.json")
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as f:
        st = json.load(f)
    tok, ch, wd, cn = st["tokens"], st["characters"], st["words"], st["counts"]
    rows = [("min", "min"), ("25%", "p25"), ("50% (median)", "median"), ("mean", "mean"), ("75%", "p75"), ("99%", "p99"),
            ("99.5%", "p99.5"), ("max", "max")]
    fmt_n = lambda v: f"{v:,.0f}" if v >= 100 else (f"{v:.1f}" if v != int(v) else f"{int(v)}")
    table = "\n".join(f"| {name} | {fmt_n(tok[k])} | {fmt_n(ch[k])} | {fmt_n(wd[k])} |" for name, k in rows)
    hist = st.get("hist_tokens", {})
    hist_line = ", ".join(f"{k}: {100 * v:.1f}%" for k, v in hist.items() if v > 0)
    n = cn["total"]
    pct = lambda c: f"{100 * c / n:.2f}%"
    return f"""## Length of the source text in the training data

Statistics of the English source text of **all {n:,} training samples** ({st['packs']:,} packed sequences of 4,096 tokens; one
sample = one input text, some contain several consecutive sentences). Use them to judge whether your input is typical for
the model: inputs far above the 99.5% row were rarely or never seen in training, and quality is less certain there.

| statistic | Gemma tokens | characters | words |
|---|---|---|---|
{table}

| samples | count | share |
|---|---|---|
| total | {n:,} | 100% |
| shorter than the 25% value (< {fmt_n(cn['p25'])} tokens) | {cn['below_p25']:,} | {pct(cn['below_p25'])} |
| longer than the 75% value (> {fmt_n(cn['p75'])} tokens) | {cn['above_p75']:,} | {pct(cn['above_p75'])} |
| longer than the 99% value (> {fmt_n(cn['p99'])} tokens) | {cn['above_p99']:,} | {pct(cn['above_p99'])} |
| longer than the 99.5% value (> {fmt_n(cn['p99.5'])} tokens) | {cn['above_p99.5']:,} | {pct(cn['above_p99.5'])} |

- Tokens, characters and words are counted on the source text alone (without the instruction), over every sample.
- Share of samples by source length in Gemma tokens: {hist_line}.

"""


def _yaml_list(items: Sequence[str]) -> str:
    return "\n".join(f"- {i}" for i in items)


def build_model_card(repo_id: str, *, stage: str = "sft", base_model: str = DEFAULT_BASE, license_id: str = "gemma",
                     license_note: Optional[str] = None, languages: Optional[Sequence[str]] = None,
                     training_data: Optional[str] = None, results_markdown: Optional[str] = None,
                     max_new_tokens: int = 256, extra_tags: Sequence[str] = (),
                     directions: Sequence[str] = ("en2x",)) -> str:
    both = "x2en" in directions
    langs = list(languages) if languages else [i for i in ALL_ISOS if LANGUAGES[i].africomet_status != UNSUPPORTED]
    prompt_text = fmt.reference_render(fmt.build_messages(EXAMPLE_EN, EXAMPLE_ISO), add_generation_prompt=True)
    full_text = fmt.reference_render(fmt.build_messages(EXAMPLE_EN, EXAMPLE_ISO, "E kaaro, bawo ni o se wa loni?"))
    user_example = fmt.build_user_content(EXAMPLE_EN, EXAMPLE_ISO)
    tags = ["translation", "african-languages", "gemma3", "en2african", f"stage-{stage}"] + list(extra_tags)
    if stage == "rl":
        tags.append("grpo")
    train_desc = training_data or (
        "English to African-language sentence pairs prepared with the format described above (data description to be "
        "completed by the model author)."
    )
    stage_desc = {"sft": "Supervised fine-tuning (full fine-tune) on packed sentence pairs, loss on the target tokens only.",
                  "rl": "Supervised fine-tuning followed by GRPO reinforcement learning with a reference-free "
                        "AfriCOMET-QE reward (shaped per language); policy-gradient updates with a KL penalty to the "
                        "SFT checkpoint."}[stage]
    name_rows = "\n".join(f"| `{i}` | {LANGUAGES[i].name} | {'yes' if i in SUPPORTED_RL else 'no'} |" for i in langs)
    note = license_note or (DEFAULT_LICENSE_NOTE + "\n\n" + OUTPUT_DATA_TERMS)
    header = f"""---
license: {license_id}
license_name: gemma
license_link: {GEMMA_TERMS_URL}
base_model: {base_model}
library_name: transformers
pipeline_tag: text-generation
language:
- en
{_yaml_list(langs)}
tags:
{_yaml_list(tags)}
---
"""
    reverse_prompt = fmt.reference_render(fmt.build_messages("E kaaro, bawo ni o se wa loni?", fmt.PIVOT_ISO), add_generation_prompt=True)
    reverse_full = fmt.reference_render(fmt.build_messages("E kaaro, bawo ni o se wa loni?", fmt.PIVOT_ISO, EXAMPLE_EN))
    length_section = _length_section()
    reverse_section = f"""## Reverse direction (African language to English)

The same template with `English` as the target language; the source text is in the African language and the model
answers with the tag `eng`:

```
Translate the text to English:  {{text in the African language}}
eng <English translation>{fmt.END_OF_TURN}
```

Prompt for a Yoruba sentence:

```
{reverse_prompt}```

and the complete training example:

```
{reverse_full}```

""" if both else ""
    body = f"""# {repo_id.split('/')[-1]}

{"English to African-language and African-language to English translation" if both else "English to African-language translation"}, fine-tuned from [`{base_model}`](https://huggingface.co/{base_model}).
{"It translates **from English into** the languages below and **from those languages into English**." if both else "This model only translates **from English**."} It is not a general chat model.

## Input format (exact)

The model must be prompted with the chat template of the base model and ONE user turn of exactly this form
(note the **two spaces** after the colon):

```
{fmt.INSTRUCTION_TEMPLATE}
```

`{{african_language}}` is one of the names in the language table below (use the name exactly as written).
The model answers with the language tag, one space, then the translation, then `<end_of_turn>`:

```
<tag> <translation>{fmt.END_OF_TURN}
```

Rendered with the chat template, the prompt for `{EXAMPLE_EN!r}` into Yoruba is (this is what the model sees):

```
{prompt_text}```

and a complete training example is:

```
{full_text}```

{reverse_section}## Usage

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

repo = "{repo_id}"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.bfloat16, device_map="auto")

def translate(text, language_name, language_tag):
    messages = [{{"role": "user", "content": "Translate the text to " + language_name + ":  " + text}}]   # two spaces
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens={max_new_tokens}, do_sample=False)   # greedy
    completion = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    # the model writes "<tag> <translation>": drop the tag
    return completion[len(language_tag) + 1:] if completion.startswith(language_tag + " ") else completion

print(translate({EXAMPLE_EN!r}, "{LANGUAGES[EXAMPLE_ISO].name}", "{EXAMPLE_ISO}"))
```

`pipeline` version:

```python
from transformers import pipeline

pipe = pipeline("text-generation", model="{repo_id}", torch_dtype="bfloat16", device_map="auto")
messages = [{{"role": "user", "content": {user_example!r}}}]
out = pipe(messages, max_new_tokens={max_new_tokens}, do_sample=False)
print(out[0]["generated_text"][-1]["content"])      # "yor E kaaro, ..."  -> remove the leading tag
```

Recommended generation settings: **greedy decoding**, `max_new_tokens={max_new_tokens}`. The checkpoint's
`generation_config.json` lists both `<end_of_turn>` and `<eos>` as stop tokens; if you build your own generation
config make sure `<end_of_turn>` (id {106}) is among the `eos_token_id` values, otherwise generation will not stop.

To remove the tag programmatically, drop the first word when it equals the requested tag (see
`gemma_mt.format.strip_iso_prefix` in the training code).

## Languages

Tag = what the model writes before the translation. Name = what you write in the instruction.
"AfriCOMET" marks languages covered by the AfriCOMET encoder (used for the RL reward and for evaluation).

| tag | name in the instruction | AfriCOMET |
|---|---|---|
{name_rows}

## Training

- Stage: **{stage}**. {stage_desc}
- Data: {train_desc}
- Packing: several pairs per 4096-token sequence with block-diagonal attention (no attention across pairs).

{length_section}{('## Results' + chr(10) + chr(10) + results_markdown + chr(10)) if results_markdown else ''}
## Limitations

- A 270M-parameter model: expect omissions, hallucinated names/numbers and errors on long inputs, on rare
  vocabulary and on low-resource languages. Do not use it for legal, medical or safety-critical translation.
- {"Sources other than English and the listed languages are untested; the source language is not named in the prompt." if both else "English source only; other source languages are untested."}
- Inputs longer than the 99.5% row of the source-length table (about 500 tokens; the longest training input was under 1,000 tokens) were rarely or never seen in training and may be truncated or poorly translated.
- Automatic metrics (chrF++, spBLEU, AfriCOMET) are imperfect, especially AfriCOMET for the lowest-resource languages.
- Output may reproduce biases of the training data and of the base model.

## License

{note}
"""
    return header + body
