# Reinforcement learning

GRPO on top of the SFT model: 8 samples per prompt, group-normalised advantages, a KL penalty to the frozen SFT model, one on-policy update per batch.
The reward is what makes it work for languages without references. Full step-by-step description with measurements:
`version_1/experiments/reports/rl_reward_explained_2026-10-19_v3.md` (sections 0 to 33). Design decisions: `DECISIONS.md` (D-18 onwards).

## The reward

1. **Hard checks** (failure classes, reward -2, nothing is scored): missing or wrong language tag, no end-of-turn, copy of the source, degenerate loops, self-repetition,
   length outside the language's window, wrong language by GlotLID (`gemma_mt/rl/lid.py`, label sets in the calibration).
2. **Content score** per sentence-aligned chunk (`chunk_plan.py`, Gale-Church alignment when the sentence counts differ):
   * `qe` languages (42 validated): AfriCOMET-QE-STL-1.1 shaped by per-language anchors (`reward_shaping.py`, `calibration/shaping_flores_dev_v*.json`);
   * `blend` languages (named on the AfriCOMET card but not validated): half shaped QE, half consensus;
   * `consensus` languages (outside AfriCOMET, plus the newest additions): the mean chrF++ of a completion against the other valid completions of its group
     (`consensus.py`). It is a self-consistency signal; it cannot detect a consistently wrong answer, which is why it is capped, guarded and weighted lower.
3. **Penalties** with per-language calibration: length, lexical overlap, diacritics (`lexical.py`, `diacritics.py`, `anchors*.py`).
4. **Advantage weight** per language (`lang_weights.py`): reliability of the signal (validated 1.0, listed 0.3 to 0.4, consensus-only 0.5).

## The confidence gate (`rl/confidence.py`, off by default: `reward.confidence`)

Three cheap, reference-free signals separate a language the model can write from one it cannot: the GlotLID pass rate of the samples, the model's own mean token log-probability,
and the number of valid samples per group. They decide whether a group contributes gradient (group gate), scale the language's advantage weight (language trust, an
exponential moving average that is stored in the trainer checkpoint) and cap the consensus-only reward. The log-prob bounds are relative to the other languages
because absolute bounds measured on the 270M policy do not transfer.

## Language sets

* 42 validated QE languages, 12 listed, 23 consensus-only (the 77 of the first generation).
* 8 further consensus-only languages from the new table rows (those with at least 2,000 training packs and a usable GlotLID plan): `guw lue tll zne lub crs mfe arz`
  (`calibration/shaping_flores_dev_v6.json`, prompts built by `scripts/build_rl_prompts_new.py`). The other 32 new rows are SFT-only for now.
* Language-ID decisions per language: `calibration/lid_labels_v6.json`; measurements: `experiments/data/lid_precision_recall_fixed_2026-11-01.csv`.

## Reference-free fluency (prepared, not wired into the reward)

`scripts/fluency_lm.py`: a character 5-gram model per language, validated against damaged text (wrong language AUC 0.996, garbled characters 0.992, shuffled words
0.868, stripped diacritics 0.72: not usable for diacritics). Intended use: a penalty below a per-language threshold, never a bonus.

## Running

```bash
python -m gemma_mt.rl.train_rl --config configs/rl.yaml
```

Rollouts use vLLM (`gemma_mt/rl/vllm_sampler.py`) or Hugging Face generate; the QE scorer runs in a separate process (`comet_worker.py`).
