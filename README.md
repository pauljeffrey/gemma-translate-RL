# gemma-translate-RL

English to African-language translation with a small model: **Gemma-3-270M-it**, fine-tuned (SFT), improved with reinforcement
learning (GRPO) against a reference-free quality-estimation reward, and then grown in depth to a 385M-parameter model.
This repository holds all the code of the project (data sourcing, cleaning, scoring, packing, SFT, RL, evaluation, language-ID and
fluency tooling, model growth), the experimental results, and the papers.

> Private identifiers (organisations, model and dataset repositories we built, buckets, hosts, tokens) are **not** in this code base.
> Everything of that kind is read from the environment: copy `.env.example` to `.env` and fill it in.

## What the project did

| Stage | What | Where |
|---|---|---|
| 1. Sourcing | crawl and audit the Hugging Face catalogue for English-African parallel text, instruction and chat data; licence screening | `data_pipeline/catalog_crawl`, `data_pipeline/sft_new` |
| 2. Clean build | decontaminate against every benchmark (dev, devtest, test), score every pair with AfriCOMET-QE, gate by a per-language threshold taken from human references, window, pack | `data_pipeline/sft_clean` |
| 3. SFT | full fine-tune of Gemma-3-270M-it on packed pairs, loss on the translation only, block-diagonal attention, UniMax language sampling | `gemma_mt/train_sft.py`, `configs/` |
| 4. RL | GRPO with a shaped AfriCOMET-QE reward, a QE-free consensus signal for languages the scorer does not cover, rule-based guards (language ID, length, copy, loops, diacritics), a confidence gate | `gemma_mt/rl/`, `calibration/` |
| 5. Growth | grow the SFT model from 24 to 39 layers (385M parameters) with identity-initialised layers, dedicated single-token language tags, staged unfreezing | `gemma_mt/growth.py`, `configs/growth1/`, `scripts/upcycle_layers.py` |
| 6. Evaluation | chrF++ and AfriCOMET on FLORES+, NTREX, MAFAND, SMOL, SALT and a long-document benchmark | `gemma_mt/eval/` |
| 7. Language ID and fluency | GlotLID precision/recall per language, a 1 MB fastText classifier for languages GlotLID lacks, a character n-gram fluency model | `data_pipeline/sft_clean/fresh/lid_measure.py`, `tools/language_id/`, `scripts/fluency_lm.py` |

The model reads and writes one fixed format everywhere (`gemma_mt/format.py`):

```
<start_of_turn>user
Translate the text to Yoruba:  Good morning<end_of_turn>
<start_of_turn>model
<yor> E kaaro<end_of_turn>
```

The language tag in front of the answer is a **single dedicated token** (one free vocabulary slot per language, `gemma_mt/tags.py`). The first
experiments used the tag as plain text (`yor E kaaro`); both forms are accepted by the evaluation and RL code.

## Repository layout

```
gemma_mt/              the Python package: languages, format, tags, attention, losses, SFT trainer, growth, RL (reward, consensus, guards, GRPO), evaluation
configs/               YAML configs of every SFT, RL, growth and ablation run (${VAR} is expanded from the environment)
calibration/           per-language reward calibration (QE anchors, length windows, language-ID label sets, consensus entries)
scripts/               one-off and batch tools (calibration builders, ablations, benchmark fetchers, report builders); see scripts/README.md
tests/                 unit tests (run with `python tests/run_tests.py` or pytest)
tools/language_id/     fastText language-ID classifier for languages without a GlotLID label
data_pipeline/
  catalog_crawl/       crawl and statistics of the African Hugging Face catalogue
  sft_new/             first-generation data builders: converters, deduplication, licence scan, combine, tokenise
  africomet_translation/  RL prompt pools and the registry of text sources
  sft_clean/           the clean (decontaminated) rebuild: protected index, QE chain, gate, windows, packs, schedules, shards
experiments/           results of the clean-rerun generation of experiments (reports, tables, training logs of the growth run)
paper/                 working draft of the second paper (clean rerun, 125 languages, growth)
version_1/             the first paper and the experiments behind it (the model called rl2c)
docs/                  pipeline, data policy, RL design, growth design
DECISIONS.md           numbered design decisions (D-xx) with their reasons
```

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # fill in the tokens and the identifiers you use
python tests/run_tests.py       # CPU-only unit tests
```

Training and scoring run on rented GPU boxes; see `docs/PIPELINE.md` for the order of the steps and the commands, and
`data_pipeline/sft_clean/fresh/README_RUNBOOK.md` for the operations runbook of the clean rebuild.

## Data and licences

The code and the results are in this repository; the data and the model checkpoints are not. `docs/DATA.md` states the data policy:
which sources are used, how licences are screened, which non-commercial and ambiguous sources are excluded or disclosed. Third-party
datasets are referenced by their public names and are fetched at run time.

## Status

* `version_1/`: complete. A 270M model, SFT then RL (the released-candidate model `rl2c`), evaluated on eight benchmarks; the paper notes
  that the first pipeline had seen part of the English side of the test sets (Section 6.9a of that paper).
* This generation (`experiments/`, `paper/`): the pipeline was rebuilt without any benchmark text in training, the data was scaled to
  35M new pairs plus an instruction stream, the model was grown to 385M parameters, and RL is being prepared for 77 to about 100 languages.
  The SFT/growth run is in progress; RL and the final evaluation have not been run. `paper/README.md` lists what is measured and what is pending.

## Licence

AGPL-3.0-or-later for the code (see `LICENSE`, `NOTICE`). The Gemma weights remain under the Gemma terms of use.
