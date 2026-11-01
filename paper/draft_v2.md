# Small but Sabi, second generation: a decontaminated rebuild, 125 languages and a model grown to 385M parameters

**Jeffrey Otoibhi**
Aletheia AI

*Working draft, 2026-11-01. Sections 1 to 6 report measurements that exist today; Sections 7 and 8 are plans or placeholders and say so. Nothing in this draft has been
reviewed. Numbers are taken from the files under `experiments/`.*

## Abstract (working)

The first paper (`version_1/`) showed that reinforcement learning with a quality-estimation reward lifts a 270M English-to-African translator on 77 languages, and also found that
its pipeline had seen part of the English side of the evaluation sets. This generation rebuilds the pipeline without any dev, devtest or test text in training, scores 35M new
parallel pairs and a 1.2B-token instruction stream with AfriCOMET-QE, extends the language table from 85 to 125 rows, and grows the fine-tuned model in depth from 24 to 39 layers
(385M parameters) with identity-initialised layers and single-token language tags. We report the data, the growth recipe and the first training curves, the language-ID and
fluency tooling that the RL stage depends on, and a full disclosure of ambiguous data sources. RL on the grown model and the final evaluation have not been run.

## 1 Why a second generation

The audit of 2026-10-26 (`experiments/reports/benchmark_contamination_audit_2026-10-26.md`) found that 95% of the FLORES+ devtest English sentences were in the old fine-tuning text,
and that some reward calibrations were fitted on test splits. The decision of 2026-10-28 (`experiments/reports/preregistration_clean_rerun_2026-10-28.md`) was to rerun the whole pipeline
with a hard filter on both sides of every pair against every benchmark split, no commissioned human test set, and new calibration sets that are disjoint from every evaluation set.

## 2 Data

* **Decontaminated parallel data.** `data_pipeline/sft_clean`: protected index of all benchmark splits, strict match on both sides (`sft_clean_rebuild_2026-10-28.md`).
* **Scoring without a cap.** The first build scored at most 150k pairs per language and that cap, not the quality gate, limited the data. Here all pairs are scored
  (AfriCOMET-QE-STL-1.1, about 1,100 to 1,500 pairs/s per GPU in fp16) and gated by a per-language threshold, the 10th percentile of the QE of human references.
* **New parallel data.** 35.19M pairs passed the gate that were not in the first run's training set (2.17B tokens), including 13.5M NLLB mined pairs in 43 languages.
* **Instruction stream.** 4.19M samples, 1.23B tokens in 29 languages, 86% synthetic or machine translated (`intake2_instr_packs_2026-11-01.md`).
* **Sampling.** UniMax over languages: every pack is seen at least once and small languages up to four times in total.

## 3 Language table and tags

125 languages (85 + 40 added on 2026-11-01), each with an ISO tag that is a single dedicated token (`docs/GROWTH.md`). Token ids and names: `experiments/data/tag_token_map_2026-11-01.csv`.

## 4 SFT

Gemma-3-270M-it with six inserted local layers, Muon on the decoder matrices, AdamW on embeddings and norms, fused chunked loss, block-diagonal attention. After 5,625 steps
(0.74B tokens) the macro validation loss over 78 languages is 1.772 (`experiments/growth1/metrics_a0u_up6.jsonl`).

## 5 Growth to 385M parameters

Recipe in `docs/GROWTH.md`: 15 identity-initialised local layers, tags in free vocabulary slots, stage A (new layers and tag rows only), stage B (all parameters, six shards).

| Point | Macro validation loss | Micro |
|---|---|---|
| end of the 270M run | 1.772 | |
| end of stage A | 1.785 | 1.750 |
| end of shard 1 (no replay, learning rate 5e-5) | 1.980 | 1.816 |
| shard 2, step 9,000 (replay of 40 small languages, 3e-5) | 1.676 | 1.592 |
| shard 2, step 10,000 | 1.640 | 1.561 |

Finding: with every pack drawn once, languages with little new data were forgotten within a few hundred steps at the peak learning rate (macro 1.785 to 2.03). A replay of 125M tokens
of the 40 smallest languages (about 9% of each shard) and a lower learning rate reversed it: all 78 languages improved over the end of shard 1 at the first evaluation of shard 2.
Not tested: whether the replay or the learning rate carries the effect (they were changed together).

## 6 Language identification and fluency for the RL guards

* GlotLID v3 on 77k training targets (`experiments/data/lid_precision_recall_fixed_2026-11-01.csv`): mean recall 0.955 for the new languages after adding sibling labels
  (Dinka 0.004 to 0.993, Tamasheq 0.36 to 1.0, Konzo 0.87 to 0.99). No label exists for Nupe, Afar and Berber.
* A 1 MB fastText classifier (accuracy 0.864) covers languages without a label (Nupe 0.945 recall, Kongo 0.909, Tamazight 0.999) but confuses close siblings (Zulu, Xhosa, Ndebele).
* A character 5-gram fluency model per language separates human text from wrong-language text (AUC 0.996), garbled text (0.992) and shuffled words (0.868, 78 of 103 languages at 0.8 or
  better) but not from text stripped of diacritics (0.72), because the training text itself often lacks them (`experiments/data/fluency_validation_2026-11-01/`).

## 7 RL (not run)

Plan: GRPO from the grown model on 77 languages plus 8 new consensus-only languages (`docs/RL.md`), with the confidence gate enabled and per-language thresholds measured on human
references where they exist. Evaluation: chrF++ and AfriCOMET on FLORES+, NTREX-128, MAFAND-MT, SMOL and SALT, with the evaluation sets absent from every training and calibration file.

## 8 Data disclosure

`docs/DATA.md`. About 25% of the unique training tokens come from sources with an open question about their origin: the NLLB mined bitext (accepted, attribution kept), the aligned Aya
subsets and the Tumbuka instruction sets (machine translated, system not stated), and one small benchmark-derived set kept by an explicit exception.

## 9 Limitations so far

* No test result exists for the grown model; all numbers are validation losses on a held-out split of the training distribution.
* The validation set is dominated by translation pairs from the same sources as the training data; it measures fit, not generalisation to the benchmarks.
* One run per configuration; no seeds repeated.
* The replay and the learning-rate change were applied together.
* The fluency model and the fastText classifier were trained on web-mined targets and inherit their noise.

## 10 Still to do

- [ ] finish stage B (shards 3 to 6) and evaluate the final SFT/growth model on the five benchmarks
- [ ] RL on the grown model; compare with RL on the 270M model under the same recipe
- [ ] measure the 32 new languages that are not yet in RL
- [ ] decide about the Tumbuka and Aya subsets
- [ ] human or pairwise-judge evaluation for a subset of languages
