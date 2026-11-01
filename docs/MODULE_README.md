<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# gemma_mt: English to African-language translation with Gemma 3 270M IT

Fine-tunes `google/gemma-3-270m-it` for ONE job, English to an African language, in two stages:

1. **SFT**: full fine-tune on packed sentence pairs, loss on the translation only, block-diagonal attention so
   packed pairs never see each other.
2. **RL**: GRPO on top of the SFT model with a reference-free **AfriCOMET-QE** reward, shaped per language.

The module is self-contained (nothing is imported from the research repo it was split from). The data is produced
elsewhere; this module consumes it. The code is meant to run on a rented GPU box (Vast.ai, one 11-24 GB GPU,
Linux). It was written and unit-tested offline on a CPU laptop; see "What was and was not verified".

Read `DECISIONS.md` for the reasoning behind every choice and for the open questions.

## The format (one format everywhere: SFT, RL, evaluation, model card)

User turn (literal, **two spaces after the colon**):

```
Translate the text to {african_language}:  {english_text}
```

Model turn:

```
{iso} {translation}<end_of_turn>
```

wrapped by the model's own Gemma 3 chat template:

```
<bos><start_of_turn>user
Translate the text to Yoruba:  Good morning<end_of_turn>
<start_of_turn>model
yor E kaaro<end_of_turn>
```

* `african_language` is the fixed display name from `gemma_mt/languages.py` (Yoruba, Hausa, Igbo, Pidgin, Efik, Fon,
  Swahili, Zulu, Xhosa, Afrikaans, Twi, Ewe, Wolof, Kinyarwanda, Chichewa, Shona, Sesotho, Setswana, Somali, Oromo,
  ...). `iso` is the tag in the same table (`yor`, `hau`, `ibo`, `pcm`, `efi`, ...).
* Only `{iso} {translation}<end_of_turn>` carries loss. `<end_of_turn>` (id 106) is the token that stops
  `generate()`; the saved `generation_config.json` lists it (and `<eos>`) as stop tokens.
* The single source of truth is `gemma_mt/format.py`.

## Layout

```
gemma_mt/                    project root
  README.md  DECISIONS.md  LICENSE  NOTICE  requirements.txt  pyproject.toml
  configs/sft.yaml  configs/rl.yaml
  scripts/                   vast_setup.sh  vast_prepare.sh  vast_train_sft.sh  vast_calibrate.sh
                             vast_train_rl.sh  vast_eval.sh        (for the GPU box; never run on the laptop)
  gemma_mt/                  the package
    languages.py             ONE language table: tag, display name, AfriCOMET code, benchmark codes, SUPPORTED_RL
    format.py                prompt / target / label mask / prefix stripping / stop-token checks
    attention.py             block-diagonal masks for packed sequences (explicit, position_ids, flash varlen)
    losses.py                chunked token log-probs (262k vocab), SFT NLL sum, RL completion log-probs
    modeling.py              model loading/saving, tiny random Gemma3 for tests
    train_sft.py             stage 1
    data/                    io.py  packing.py  prepare_sft.py  packed_dataset.py  collate.py
    rl/                      train_rl.py  grpo.py  reward.py  reward_shaping.py  calibrate_shaping.py
                             comet.py  comet_worker.py  guards.py  text.py  sampling.py  data.py
    eval/                    benchmarks.py  metrics.py  eval_translation.py  make_report.py
    hub/                     model_card.py  push_model.py  dataset_card.py  push_dataset.py
    fake_tokenizer.py  smoke.py  utils.py
  tests/                     pytest tests + run_tests.py (runs them without pytest)
```

## Data formats

**SFT pairs** (parquet or jsonl, a file or a directory of files):

| column | type | |
|---|---|---|
| `english` | str | English source sentence |
| `target` | str | translation |
| `iso` | str | tag of `target` (canonical tags; aliases such as `swa`, `ti` are normalised) |
| `id` | str, optional | stable id (used only when `--split-key id`) |

**RL prompts** (parquet or jsonl; no reference translation needed):

| column | type | |
|---|---|---|
| `english` | str | English source text |
| `iso` | str | target tag; must be in `SUPPORTED_RL` (AfriCOMET-covered) |
| `language` | str | display name, checked against `languages.py` |
| `src_afr_tokens` | int | length of `english` in `Davlan/afro-xlmr-large-76L` tokens, `<= 256` |
| `source_dataset` | str | provenance (kept for diagnostics) |

**Benchmarks** (`gemma_mt.eval.benchmarks`, normalised to `english, reference, iso, benchmark, id`): FLORES+
(`openlanguagedata/flores_plus`, gated), NTREX-128 (local checkout), MAFAND-MT (`masakhane/mafand`). Local files or
the Hub; the network is touched only at run time on the GPU box. Opt-in extras for the languages that have no FLORES+ / NTREX / MAFAND
test set (D-42, `--benchmarks smol_sent,smol_doc,gatitos,ibom_mt,flores_moz,glot_story,salt`, default unchanged): Google SMOL (SmolSent,
SmolDoc) and GATITOS (`google/smol`, CC-BY-4.0), IbomMT (CC-BY-NC-4.0), FLORES-MOZ (CC-BY-SA-4.0), GlotStoryBook (per-row CC licences) and SALT
(CC-BY-SA-4.0, gated); `python -m gemma_mt.eval.benchmarks --list --extra` prints the code table.

**Packed dataset** (written by `prepare_sft`, directory with `train/`, `val/`, `meta.json`): per split a flat
`tokens.u32.bin` (uint32: Gemma's 262k vocabulary does not fit uint16) and `.npy` indexes (sample offsets/lengths,
prompt lengths, target lengths, language ids, pack boundaries). `meta.json` holds the tokenizer name + fingerprint,
counts, packing efficiency and per-language token counts.

## Commands (on the GPU box)

```bash
export HF_TOKEN=...                                  # read from the environment, never written to files
bash scripts/vast_setup.sh                           # venv, requirements, unit tests

# 1. data
RAW=/workspace/raw_sft OUT=/workspace/packed bash scripts/vast_prepare.sh
#    = python -m gemma_mt.data.prepare_sft --input $RAW --out $OUT --seq-len 2048 --val-fraction 0.01

# 2. SFT (single GPU; NGPU=2 for torchrun)
DATA=/workspace/packed OUT=/workspace/runs/sft bash scripts/vast_train_sft.sh
#    = python -m gemma_mt.train_sft --config configs/sft.yaml data_dir=... out_dir=...

# 3. baseline + SFT evaluation (before RL)
MODEL=google/gemma-3-270m-it NAME=base OUT=results/base bash scripts/vast_eval.sh
MODEL=/workspace/runs/sft/best NAME=sft OUT=results/sft bash scripts/vast_eval.sh

# 4. reward calibration (per language, on the FLORES+ DEV split), then RL
MODEL=/workspace/runs/sft/best bash scripts/vast_calibrate.sh          # writes calibration/shaping.json
SFT=/workspace/runs/sft/best PROMPTS=/workspace/rl_prompts OUT=/workspace/runs/rl bash scripts/vast_train_rl.sh

# 5. evaluation + report
MODEL=/workspace/runs/rl/best NAME=rl OUT=results/rl bash scripts/vast_eval.sh
python -m gemma_mt.eval.make_report --run base=results/base/results.json --run sft=results/sft/results.json \
    --run rl=results/rl/results.json --baseline-csv published.csv --reference base --out report.md

# 6. publish (dry run unless --yes; repositories are private unless --public)
python -m gemma_mt.hub.push_model --model-dir runs/rl/best --stage rl --results-md report.md          # preview
python -m gemma_mt.hub.push_model --model-dir runs/rl/best --stage rl --results-md report.md --yes
python -m gemma_mt.hub.push_dataset --data-dir data/rl_parquet --kind rl --yes
```

Offline checks that work anywhere (CPU, seconds): `python -m gemma_mt.train_sft --smoke`,
`python -m gemma_mt.rl.train_rl --smoke`, `python tests/run_tests.py` (or `pytest tests`).

Default Hub names for the manual push tools (account `<HF_ORG_LEGACY>`, private; the rental pipeline uses the org `<HF_ORG>` since 2026-10-19: SFT `<HF_ORG>/gemma3-270m-translate`, RL `<HF_ORG>/gemma-3-270m-translate-rl`): models `gemma3-270m-translate` (SFT, owner-confirmed 2026-10-14),
`gemma3-270m-en2african-rl`; datasets `gemma3-270m-en2african-sft-data`, `gemma3-270m-en2african-rl-prompts`.

## Hardware notes (estimates, not measured on a GPU)

* **bf16 needs an Ampere-or-newer GPU** (RTX 30xx/40xx, A-series, L4...). Gemma 3 overflows in fp16, and pre-Ampere
  cards have no usable bf16. Weights are kept in fp32 with bf16 autocast (270M parameters: 1.1 GB weights +
  1.1 GB gradients + 2.2 GB Adam state).
* **SFT**: most memory is activations and the 262k-row LM head. The loss is computed only at target positions in
  chunks of 512 tokens (`loss_chunk_tokens`), so logits never exceed ~0.5 GB. `micro_batch_size: 4` (4 x 2048
  tokens) is the default for ~12 GB; use 8 on 24 GB. If you hit OOM halve `micro_batch_size` and double
  `grad_accum_steps` (the update is unchanged: the loss is normalised by the target tokens of the whole step), or set
  `gradient_checkpointing: true`.
* **RL** keeps policy (fp32) + optimizer + a frozen fp32 reference (1.1 GB) + the AfriCOMET-QE scorer (XLM-R large,
  fp16, ~1.1 GB) on one GPU. Defaults: 16 prompts x 8 samples = 128 translations per step, scoring batch 32,
  scoring forward 16 sequences. On 11 GB lower `micro_batch_seqs` to 8 and `scorer.batch_size` to 16 if needed.
* **flash-attn** is optional (`attn_implementation=flash_attention_2`, padding-free varlen path). SDPA with explicit
  block-diagonal masks is the default and needs nothing extra.
* Multi-GPU: SFT supports `torchrun` (DDP). RL is single-GPU by design.

## What was and was not verified (laptop, CPU, offline)

Run with the research repo's `.venv-comet` (torch 2.14, transformers 4.57.6, sacrebleu, unbabel-comet 2.2.7); the
numpy-only parts also run on a bare Python. See the summary at the end of `DECISIONS.md` for the exact counts.

Verified: format and label masks (fake byte tokenizer, plus the Gemma template re-stated in jinja), packing,
splitting, loaders, collators; packed == separate for logits, loss and gradients on a tiny random Gemma3 with a
sliding window smaller than the samples (eager + sdpa, explicit masks and position-id masks, with negative controls);
chunked loss == Hugging Face loss; token-weighted accumulation; exact (bitwise) resume; GRPO formulas, KL
estimators and their gradient meaning; reward plumbing and failure classes; shaping presets (bounded, monotone,
per-language); benchmark adapters on synthetic files; metrics, report, cards, push guards; SFT and RL end-to-end
smoke runs; the in-process AfriCOMET-QE scorer against the locally cached real checkpoint.

NOT verified here (no GPU, no downloads): the real `google/gemma-3-270m-it` tokenizer and template, the
FlashAttention-2 varlen path, bf16 numerics and memory figures, multi-GPU DDP, FLORES+/NTREX/MAFAND loading from the
Hub, parquet I/O (no pyarrow on the laptop venv), GlotLID, real training results. Env-gated integration tests for
the tokenizer, the model and AfriCOMET are in `tests/test_integration_gated.py`.

## Licence

Code: AGPL-3.0-or-later (`LICENSE`, `NOTICE`; the verbatim licence text must still be added, see the TODO in
`LICENSE`). Model weights: Gemma Terms of Use (`license: gemma`) plus the additional data-generation terms on the
model card (a grant to the publisher for datasets generated with the model; nobody has to open-source their models).
Datasets: cc-by-4.0 cards (attribution only), per-row source licence. The code licence does not govern model
weights or data generated with the models. See DECISIONS.md Q-13.

## Reverse direction (African -> English)

Added 2026-10-12 at the owner's request. The format is the same, the target is the pivot language:

```
Translate the text to English:  {african_text}      ->      eng {english}<end_of_turn>
```

* Data: `python -m gemma_mt.data.prepare_sft --directions en2x x2en [--x2en-fraction 0.5] ...` makes the reverse sample of every
  pair (same pair, same split; `direction.npy` marks them, `meta.json` counts them per direction and language).
* Training: `train_sft` reports `val_loss_en2x`, `val_loss_x2en` and per-language losses of both directions.
* Evaluation: `python -m gemma_mt.eval.eval_translation --direction x2en ...` swaps the benchmark rows (African text is the
  source, English the reference); chrF++ / spBLEU / BLEU are on English text, AfriCOMET gets the African source.
* RL stays English -> African (reward preset `repo`); the `eng` tag is only a training/eval tag here.

## Long inputs and chunked reward (D-23)

* RL prompt files carry `length_bin` (b64 / b256 / b512 / b1024, balanced per language). `configs/rl.yaml` trains the bins in
  stages (`curriculum_steps`), then in a random mixture, with per-bin prompts-per-step and `max_new_tokens`; `micro_batch_tokens`
  bounds the scoring forward.
* `reward.chunking: sentences` splits source and translation into sentences (no newline needed), pairs them (equal counts) or
  aligns them by length (Gale-Church), shapes each chunk and takes the source-length-weighted mean.
* SFT on regrouped windows (optional) needs a longer pack length: `prepare_sft --seq-len 4096 --max-sample-tokens 4096`, then
  `configs/sft_long.yaml`.

