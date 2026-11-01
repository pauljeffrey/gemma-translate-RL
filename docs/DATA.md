# Data policy

The repository contains no training data and no model weights. Third-party datasets are referred to by their public names in the code (for example the
FLORES+, NTREX-128, MAFAND-MT, SMOL, SALT, IbomMT and Aya releases) and fetched at run time with your own Hugging Face token. Datasets and checkpoints that we built are
referred to through environment variables (`.env.example`).

## What the training data is

1. **Parallel English-African text** from public Hugging Face datasets, converted by `data_pipeline/sft_new` and `data_pipeline/sft_clean/chain`.
2. **NLLB mined bitext** (OPUS NLLB v1, 43 languages) used with a per-language cap and the same quality gate as everything else; attribution is carried in the data card.
3. **Instruction, chat and QA data** in 29 languages (about 1.2B tokens, 86% synthetic or machine translated), including the aligned part of the Aya collection and Tumbuka instruction sets.
4. **Replay** of the older translation packs of the 40 smallest languages.

## Rules that every source passes

* **Licence screen** (`license_scan.py`, `nc_filter.py`): sources with a non-commercial licence are excluded from training (one exception for a small benchmark-derived set, disclosed);
  `nc_hidden.py` also drops sources whose card says one thing but whose origin is non-commercial.
* **Decontamination** (`fetch_protected.py`, `decon.py`): no sentence of any dev, devtest or test split of the benchmarks we report on may appear on either side of a training pair.
  The first generation of experiments (see `version_1/`) lacked this and the audit that found it is `experiments/reports/benchmark_contamination_audit_2026-10-26.md`.
* **Language verification**: GlotLID with accepted label sets per language (`calibration/lid_labels_v6.json`), and a small fastText classifier for languages GlotLID has no label for.
* **Quality gate**: AfriCOMET-QE-STL-1.1 on every pair; a pair is kept when its score reaches the per-language threshold (10th percentile of the QE of human references of that language,
  pooled fallback). The scoring cap per language of the first build was removed in the clean build (all 35M new pairs were scored).
* **Provenance flags**: every row carries `synthetic`, `kind` (human, mined, machine translated, LLM generated) and the licence text of its source.

## Disclosure of ambiguous sources

| Source | Size | Why it is flagged | Decision |
|---|---|---|---|
| NLLB mined bitext | 13.5M pairs, 43 languages (about 0.65B tokens, estimate) | web-mined, dataset released under an open data licence; the underlying sentences stay with their publishers | used, attribution kept |
| Aya collection, aligned part | 169M tokens, 12 languages | some subsets were produced with NLLB models; the collection is published under Apache-2.0 | used |
| Tumbuka instruction sets | 164M tokens | machine translated, system not stated | used, disclosed |
| muri-it | | instruction side translated with NLLB | excluded |
| TranslatePsy | | non-commercial data | excluded |

Per-language token tables: `experiments/data/token_table_provisional_2026-11-01.csv`, `experiments/data/nllb_in_stageB_2026-11-01.csv`.
