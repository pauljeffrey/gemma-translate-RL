## 5. The SFT-final model (sentence level)

{{sft_block}}

## 6. Cost summary

| configuration | generated tokens vs greedy | wall clock vs greedy (same engine) | where |
|---|---|---|---|
| greedy | 1x | 1x (vLLM ~200 sentences/s, R5) | |
| one sample, any T / cut-off | 1x | 1.0 - 1.3x | |
| beam 4 (+ fallback) | ~4x (HF) | 4.6x HF greedy (R5) | HF |
| MBR n = 8 / 16 | 8x / 16x | ~10x / ~20x (R5: MBR8 10x including the CPU chrF++ matrix; n = 16: 120 pairs per sentence, 58 s on 10 cores for 3,970 sentences) | vLLM + CPU |
| sentence-by-sentence documents | 2.0 per source token (as whole-document greedy) | R6: faster than whole-document greedy in wall time | |
| repair best-of-8 of the 0.6% unfinished units | +8 x 0.6% of the units, up to 1,024 tokens each | negligible | |

## 7. Caveats

* **Metric.** chrF++ / spBLEU / AfriCOMET against one reference per sentence; the owner's concern ("very repetitive") is about readability, and the only evidence on it here is the repeated-3-gram statistic; no human judgement was collected. Selection by consensus (MBR) optimises chrF++ and gains little on spBLEU; selection by QE optimises AfriCOMET and loses chrF++; the oracle selection shows several chrF++ points of headroom that none of the selectors reach.
* **Sample sizes.** 200 sentences per language, equal language weight: differences below ~0.15 chrF++ (macro) are noise; per-language differences below ~0.5 are not resolved. `st1024` human documents: 20 per language for 4 languages (hau yor swh amh), 1-2 for the other two; `st2048`: 3-5 per language for 4 languages (15 in total): indicative only. Document-level sampling rows use a subset of 1,278 documents (see section 1), R6 rows are re-aggregated on it.
* **Not run (and why).** Beam search on the sentence units of long documents (HF-only, 100k units); MBR of sentence units at document level (the sentence-level result applies unit by unit; 8 samples x 37k units did not fit the 4 GB / nice 19 budget); the SFT model on documents; repetition-aware selection (MBR with a repetition penalty on the candidates) which would be the natural next experiment if the repetition matters to readers; RL rollouts for `b1024` and longer bins (generation of 16 x 3,584 tokens per prompt at 2k tokens/s) and with the live reward (QE ensemble, anchors, lexicon of D-55 - D-57); the rollout statistics are for the final model's *own* samples, not for the policy while it trains.
* **Engines.** vLLM and HF sampling agree (T 1.0: 30.50 vs 30.52 chrF++). vLLM applies min-p, top-k and top-p after the temperature; the bias numbers (section 4) rebuild this order in PyTorch with bf16 logits (the vLLM sample sometimes sits on the exact nucleus boundary, which is why I report the exact per-position KL instead of the log-ratio of the sampled tokens).
* **Housekeeping slip (R8's job).** One of my cleanup commands matched `comet_worker` by name and terminated the AfriCOMET worker of R8's `reward_run.py` (nice 19, `rw_abl_norep.jsonl`) at about 05:3x UTC on 2026-10-22. R8's driver relaunched the script within a minute (new process ids), so that run restarted from its beginning; I did not touch it again. Every kill after that was restricted to processes whose working directory is under `/workspace/jobs/r12`. R8 should know that one run of theirs was restarted.

## 8. Reproduction

Box scripts are in `reports/data/generation_config_study/scripts/` (sentence level: `gen_sent.py` -> `gen_hf.py` -> `eval_grid.py` -> `pool_eval.py` -> `qe_pools.py` / `qe_eval.py` -> `export_csv.py`; document level `doc/`: `mkreq.py`, `mkreq2.py`, `gen_doc.py`, `asm.py`, `score_items.py` (R6's, unchanged), `docagg.py`, `unit_repair.py`; RL `rl/`: `rl_prompts.py` -> `rl_gen.py` -> `rl_reward.py` -> `rl_bias.py` -> `rl_analyze.py`; the queue scripts `chain*.sh` document the order and the GPU limits). Results: `json/*.json`, `export/*.csv`. The tables of this report are rendered by `gemma_mt/scripts/build_generation_config_report.py` from the JSON.
