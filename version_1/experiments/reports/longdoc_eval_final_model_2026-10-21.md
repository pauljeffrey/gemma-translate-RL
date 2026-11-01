# Document-level / long-context evaluation of the final model (rl2c, step 1433) on AfriLongDoc, and the best generation configuration for long inputs - 2026-10-21 (D-50)

Model: `<HF_ORG_LEGACY>/gemma-3-270m-translate-rl2c@latest` (step 1433; `runs/final_1433`). Yardstick model: the SFT final (step 27,638). Data: `<HF_ORG>/AfriLongDoc` (private): 450 English core documents in six length bins (b64 ... b4096), SmolDoc human references (CC-BY-4.0), NLLB-200 3.3B and MADLAD-400 3B sentence-by-sentence pseudo-references. Harness: `gemma_mt/eval/longdoc.py` (document statistics, alignment plan, loop detector), `longdoc_reward.py` (total RL reward), prompts from `gemma_mt/format.py::build_prompt`, vLLM generation. All work was done on the rented RTX PRO 6000 box under `/workspace/jobs/a2/` (outputs `hyp/`, per-document scores `sc/`, tables `w5/`). Nothing was pushed.

## 0. Summary (read this first)

1. **Whole-document generation does not work for inputs beyond about 512 source tokens, whatever the decoding setting.** Greedy: finished 98 / 97 / 92 / 65 / 42 / 31 % of the documents for b64 / b256 / b512 / b1024 / b2048 / b4096, loop rate 0 / 1 / 13 / 53 / 73 / 73 %, chrF++ vs the NLLB reference 47.6 / 47.7 / 47.7 / 29.7 / 18.2 / 9.8. This reproduces the reward you measured on b2048 (greedy total reward -1.95, "ok" 7 %). Repetition / frequency penalty, no-repeat-4gram-style blocking, T = 0.3 ... 1.0 and best-of-N all trade the loops for **truncation**: T = 0.7 finishes 99 / 95 / 78 % of the documents (T = 1.0: 98-100 %) but the output is only 77 % (b1024), 56 % (b2048), 33 % (b4096) as long as the reference, so chrF++ stays at 26-37 (b1024-b2048) and 14.5 (b4096). The best whole-document setting at b1024 / b2048 / b4096 (best-of-4 at T = 0.7) is 36.9 / 26.7 / 14.5 chrF++.
2. **The fix is chunking with the same model.** Translating sentence by sentence and joining (`greedy_sent`), with the few unfinished sentences (1.0 % of units) re-decoded by best-of-4 at T = 0.7 (`sentfix`), gives chrF++ vs NLLB of **47.6 / 49.0 / 53.5 / 55.6 / 58.4 / 60.1** (b64 ... b4096; +0.0 / +1.4 / +5.8 / +25.9 / +40.2 / +50.3 vs greedy whole-document), 100 % sentence coverage, loop rate 0-4 %, and it is the same or better against the **human** SmolDoc references (clean, 6 languages: 47.2 / 49.1 / 50.6 chrF++ on the short / ~512 / ~1,024-token documents; 52.4 on ~2,048-token documents, 4 languages). Sentence-by-sentence needs 2.0 generated tokens per source token at every length (whole-document greedy: 2.0-2.7, its loops run to the token limit) and was faster in wall time (1,196 s vs 1,912 s for all 9,506 items on the shared GPU).
3. **b4096 works with chunking** (sentfix 60.1 chrF++, AfriCOMET 0.659 against 0.691 at b64, coverage 100 %, loop 4 %); whole-document b4096 does not (9.8-17.7 chrF++ over all settings, 26-98 % finished, output 30-58 % of the reference length when it finishes).
4. **Best configuration per bin** (details in section 6): b64: any (tie; beam 4 +1.6 on a 4-language test); b256 ... b4096: sentence-by-sentence + best-of-4 repair of unfinished sentences. Overall: always chunk by sentence, greedy, repair the unfinished units; no whole-document decoding above ~256 tokens.
5. **Pseudo-reference bias is real** (section 8): the model agrees with NLLB more than MADLAD does (65.9 vs 62.7 chrF++ on 5 languages at b1024), the NLLB-reference scores are 1-16 chrF++ above the human-reference scores for 12 of 12 human languages, and the sentence-over-window margin that the NLLB reference shows (+1.0 ... +1.3 chrF++, sentfix vs winfix) is only +0.0 ... +0.6 against the human references.
6. The final model is +2.1 ... +3.3 chrF++ above the SFT model under the same chunking; the SFT model does not stop reliably (with sentence chunking it finishes only 48 / 39 / 24 / 13 % of the documents at b512 / b1024 / b2048 / b4096, loops in 42-59 %), and SFT whole-document generation is dead from b256 (74 % finished) on.

## 1. What was run

**Languages.** 16 = 12 with a SmolDoc human reference **and** a low-flagged NLLB reference (hau yor ibo swh zul xho amh kin som wol twi mlg) + 4 with NLLB only (kea kam sag nus). The language list was fixed before I took over (agent A2); `sna lug bem tir` from the brief are not in it. Only the greedy, sentence, window and repair runs cover all 16; every other configuration was run on an **8-language subset B8 = hau yor swh amh zul wol kea sag** (6 with human references, 2 NLLB-only; fixed by A2; Latin and Ge'ez scripts, high to low scores). **All tables below use B8 so that every row is comparable**; the 16-language table is section 7.

**Documents.** Core set: 450 documents (b64 73, b256 90, b512 100, b1024 89, b2048 63, b4096 35 documents; the bins are different texts, so absolute scores are not comparable between bins, only configurations within a bin). Human set: SmolDoc documents (`sm`, <= ~680 tokens) and SmolDoc documents stitched to ~512 / ~1,024 / ~2,048 Gemma tokens (`st512`, `st1024`, `st2048`; stitched = unrelated consecutive documents, so they test length but not discourse). Only the **clean** human documents are used (the English side not in our SFT windows): up to 60 / 40 / 20 / 10 per language and bin; `st2048` exists for 4 of the 6 B8 languages (hau swh amh yor). The contaminated documents were generated too (`sc/`) but are not reported.

**Generation** (vLLM, bf16, `max_model_len` 10,240, prompt = `build_prompt(text, iso)`, `max_new_tokens = min(3.5 x source tokens, 10,240 - prompt)`, stop ids 106 / 1):

| name | what |
|---|---|
| greedy | whole document, greedy |
| rep105 / rep110 | + repetition_penalty 1.05 / 1.10 |
| freq005 | + frequency_penalty 0.05 (frequency 0.10 was **not run**, cut for time) |
| t03 / t05 / t07 / t10 | whole document, sampling T = 0.3 / 0.5 / 0.7 / 1.0 (top_p 1, top_k 0), one sample |
| bon4 / bon8 | best-of-N from T = 0.7 samples, pick = highest mean chrF++ to the other candidates (consensus); N = 4 for all bins, N = 8 only up to b512 (cost) |
| beam2 / beam4 / nr6 | HF `generate`: beam 2 / beam 4 / greedy with `no_repeat_ngram_size=6` (vLLM has neither): **4 languages (hau yor swh amh), b64-b512 only**; nr6 also on hau + yor at b1024 / b2048 (`nr6hl`) |
| greedy_sent | (a) every source sentence (over-long ones split at 700 characters) translated alone, greedy, joined with the source paragraph structure |
| greedy_win | (b) windows of whole sentences, paragraph-aware, <= 256 source tokens, no overlap |
| greedy_winov | (b') the same plus the preceding sentence as context, removed from the output again by alignment (1,242 of 16,489 windows could not be cleanly stripped and keep the context sentence) |
| sentfix / winfix | greedy_sent / greedy_win where each **unfinished** unit (did not emit the end token: 1.0 % of sentence units, 2.0 % of windows) is re-decoded by best-of-4 (T = 0.7, consensus among the finished candidates; if none finishes the greedy output is kept) |
| sft:... | the SFT final under greedy / greedy_sent / greedy_win (B8, same requests) |

**Metrics.** Document-level corpus chrF++ and BLEU (sacrebleu statistics, summed per language over documents) against the primary reference (NLLB for the core set, SmolDoc for the human set), **macro over languages**, with 95 % bootstrap CIs (400 resamples, documents resampled with replacement, the same resample for all configurations, so the differences "d vs greedy" are paired). Failure metrics from `longdoc.py`: finished (the end token was generated), loop (a repeated word cycle in the hypothesis that is not in the reference), sentence coverage (1 - (source sentences with no output + copied as-is) / sentences, from the alignment plan), median length ratio (hypothesis / reference characters). AfriCOMET (reference-based `africomet-stl` and reference-free QE `africomet-qe-stl-1.1`) on aligned chunks of a subset: 17 core documents per bin x 8 languages, up to 15 human documents per language x bin (the `st2048` COMET subset uses the *contaminated* stitched documents, see the code rule in `comet_eval.py`). Total RL reward: `scripts/longdoc_score_reward.py` with `rl_rtxpro_phase2c.yaml` and `calibration/shaping_flores_dev_v7.json`, AfriCOMET-QE worker, 12 documents per bin per language (72 per language), per-output reward without group consensus.

**Not done / cut** (to fit the 2-3 h GPU window): frequency 0.10; best-of-8 above b512; beam search on sentence units (needs a beam decoder in the vLLM path); beam / no-repeat-ngram on long whole documents beyond the 2-language `nr6hl` check; AfriCOMET and reward on all 18 configurations (done for the 8-11 that matter, see the tables); NLLB / MADLAD on the *human* documents (no NLLB model on the box), hence no direct "pseudo-reference vs human reference" yardstick; the SFT model under repetition / sampling settings; bootstrap over languages.

## 2. Why greedy fails, and what the other whole-document settings do instead (core set, vs NLLB, B8)

Rows with their numbers per bin are in section 3; the pattern:

* **Greedy** loops: b512 13 %, b1024 53 %, b2048 73 %, b4096 73 % of the documents contain a loop that the reference does not; the loop runs to the token limit (generated tokens per source token: 2.0 at b64, 2.7 at b2048), so the "finished" rate falls to 42 % / 31 %.
* **Repetition penalty** hurts short documents (rep105 -1.5 / -1.9 chrF++ at b64 / b256, rep110 -5.5 / -6.9) and helps long ones only a little (b2048 +3.6 / +5.3, b4096 +2.5 / +2.8) while the loops remain (62 % / 55 % at b2048). **Frequency penalty 0.05** is the better of the three (b64 -0.1, b256 0.0, b512 +1.0, b1024 +3.7, b2048 +7.1, b4096 +7.9) but still ends at 17.7 chrF++ on b4096.
* **Temperature**: T = 0.3 is greedy with a few fewer loops; T = 0.5 / 0.7 progressively removes the loops (b2048 loop 40 % / 6 %) but the model then stops early: length ratio 0.63 / 0.56 at b2048 and 0.38 / 0.33 at b4096; T = 1.0 never loops (0 %) but loses 3.7 chrF++ on b64 and covers only 65 % of the sentences at b4096. The failure is the same one that makes the RL run's T = 1 rollouts look healthy ("finished 99 %"): the model ends the translation instead of continuing.
* **Best-of-N** by consensus: +0.7 / +1.1 chrF++ (N = 4 / 8) at b512 vs greedy, nothing at b64-b256 (it picks a mean sample, -0.6 / -0.5 at N = 4), +7.2 at b1024, +8.5 at b2048, +4.6 at b4096, always from truncated candidates. N = 4 costs 4x the tokens (3.5-8.0 generated tokens per source token).
* **Beam search** (4 languages, short bins, section 4) is the only whole-document setting that beats greedy cleanly (+1.2 ... +1.6 chrF++ for beam 4), and it removes loops at b512 (13 % -> 4 %), but it is HF-only and slow (380 tok/s on a shared GPU), and it is still 2.8 chrF++ below sentence-by-sentence at b512.
* **no_repeat_ngram 6** removes the loops (b512 0 %) but costs -0.6 ... -0.7 chrF++ on short documents; on b1024 / b2048 (hau, yor) it reaches 38.5 / 27.3 chrF++ against 61.8 / 64.8 for sentfix, i.e. it also truncates.

## 3. Results per bin x configuration (core set, vs NLLB primary reference, B8)

Columns: chrF++ [95 % CI] macro over the 8 languages; "d vs greedy" is the paired difference to whole-document greedy; finished / loop / sentence coverage / median length ratio as defined above. Bins: documents per bin 73 / 90 / 100 / 89 / 63 / 35. `bon8` exists up to b512 only.

#### core bin b64 (vs NLLB primary, macro over 8 languages)

| config | chrF++ [95% CI] | d chrF++ vs greedy [CI] | BLEU | finished | loop | sent. cov. | len ratio |
|---|---|---|---|---|---|---|---|
| greedy | 47.6 [46.2, 49.0] | 0.0 [0.0, 0.0] | 24.9 | 98% | 0% | 100% | 1.01 |
| rep105 | 46.1 [44.7, 47.5] | -1.5 [-1.9, -1.0] | 21.9 | 98% | 0% | 100% | 1.03 |
| rep110 | 42.1 [40.8, 43.4] | -5.5 [-6.2, -4.7] | 16.2 | 96% | 0% | 100% | 1.09 |
| freq005 | 47.5 [46.1, 48.8] | -0.1 [-0.4, 0.1] | 24.9 | 99% | 0% | 100% | 1.01 |
| t03 | 47.7 [46.4, 49.1] | 0.1 [-0.2, 0.4] | 24.6 | 98% | 0% | 100% | 1.01 |
| t05 | 46.9 [45.5, 48.4] | -0.7 [-1.0, -0.3] | 24.1 | 98% | 0% | 100% | 1.01 |
| t07 | 46.4 [45.1, 47.8] | -1.2 [-1.6, -0.8] | 23.2 | 97% | 0% | 100% | 1.02 |
| t10 | 43.9 [42.5, 45.4] | -3.7 [-4.2, -3.1] | 21.0 | 98% | 0% | 100% | 1.02 |
| bon4 | 47.1 [45.7, 48.5] | -0.6 [-1.0, -0.1] | 23.8 | 98% | 0% | 100% | 1.01 |
| bon8 | 47.6 [46.3, 48.9] | -0.0 [-0.4, 0.4] | 24.5 | 98% | 0% | 100% | 1.01 |
| greedy_sent | 47.6 [46.0, 49.2] | -0.0 [-0.8, 0.6] | 24.8 | 99% | 0% | 100% | 1.01 |
| sentfix | 47.6 [46.0, 49.2] | -0.0 [-0.8, 0.5] | 24.8 | 100% | 0% | 100% | 1.01 |
| greedy_win | 47.4 [45.8, 49.0] | -0.2 [-1.1, 0.3] | 24.6 | 99% | 0% | 100% | 1.01 |
| winfix | 47.5 [45.7, 49.1] | -0.2 [-1.0, 0.4] | 24.6 | 100% | 0% | 100% | 1.01 |
| greedy_winov | 47.4 [45.8, 49.0] | -0.2 [-1.0, 0.3] | 24.7 | 99% | 0% | 100% | 1.01 |
| sft:greedy | 44.3 [42.8, 45.9] | -3.3 [-4.3, -2.2] | 22.8 | 93% | 5% | 100% | 0.99 |
| sft:greedy_sent | 44.3 [42.6, 45.9] | -3.3 [-4.5, -2.1] | 22.1 | 93% | 6% | 100% | 1.00 |
| sft:greedy_win | 43.4 [41.7, 45.0] | -4.2 [-5.4, -3.0] | 22.2 | 94% | 6% | 100% | 0.99 |

#### core bin b256 (vs NLLB primary, macro over 8 languages)

| config | chrF++ [95% CI] | d chrF++ vs greedy [CI] | BLEU | finished | loop | sent. cov. | len ratio |
|---|---|---|---|---|---|---|---|
| greedy | 47.7 [46.7, 48.5] | 0.0 [0.0, 0.0] | 22.8 | 97% | 1% | 100% | 1.00 |
| rep105 | 45.7 [44.7, 46.7] | -1.9 [-2.2, -1.7] | 19.5 | 97% | 1% | 100% | 1.01 |
| rep110 | 40.7 [39.9, 41.5] | -6.9 [-7.3, -6.5] | 13.0 | 99% | 0% | 100% | 1.06 |
| freq005 | 47.6 [46.7, 48.5] | -0.0 [-0.2, 0.2] | 22.7 | 97% | 2% | 99% | 1.00 |
| t03 | 47.5 [46.6, 48.4] | -0.2 [-0.4, 0.0] | 22.4 | 98% | 0% | 100% | 1.01 |
| t05 | 47.1 [46.3, 48.0] | -0.5 [-0.7, -0.3] | 21.8 | 97% | 0% | 99% | 1.01 |
| t07 | 46.7 [45.8, 47.5] | -1.0 [-1.2, -0.7] | 21.1 | 99% | 0% | 100% | 1.01 |
| t10 | 44.5 [43.7, 45.2] | -3.2 [-3.5, -2.9] | 18.5 | 99% | 0% | 100% | 1.02 |
| bon4 | 47.2 [46.3, 48.0] | -0.5 [-0.7, -0.3] | 21.6 | 99% | 0% | 100% | 1.01 |
| bon8 | 47.6 [46.8, 48.4] | -0.0 [-0.2, 0.2] | 22.0 | 99% | 0% | 99% | 1.01 |
| greedy_sent | 49.0 [48.2, 49.9] | 1.4 [1.1, 1.6] | 23.7 | 98% | 0% | 100% | 1.01 |
| sentfix | 49.0 [48.2, 49.9] | 1.4 [1.1, 1.6] | 23.7 | 100% | 0% | 100% | 1.01 |
| greedy_win | 47.6 [46.6, 48.4] | -0.1 [-0.3, 0.2] | 22.8 | 96% | 2% | 100% | 1.00 |
| winfix | 47.9 [47.0, 48.7] | 0.2 [-0.0, 0.4] | 23.0 | 100% | 1% | 100% | 1.00 |
| greedy_winov | 47.7 [46.8, 48.5] | 0.0 [-0.2, 0.2] | 22.9 | 97% | 2% | 100% | 1.00 |
| sft:greedy | 39.9 [38.7, 41.2] | -7.7 [-8.8, -6.8] | 18.3 | 74% | 24% | 98% | 1.02 |
| sft:greedy_sent | 45.7 [44.7, 46.5] | -2.0 [-2.5, -1.5] | 21.4 | 72% | 23% | 100% | 1.01 |
| sft:greedy_win | 39.1 [37.7, 40.6] | -8.5 [-9.7, -7.4] | 18.0 | 72% | 27% | 98% | 1.02 |

#### core bin b512 (vs NLLB primary, macro over 8 languages)

| config | chrF++ [95% CI] | d chrF++ vs greedy [CI] | BLEU | finished | loop | sent. cov. | len ratio |
|---|---|---|---|---|---|---|---|
| greedy | 47.7 [47.1, 48.3] | 0.0 [0.0, 0.0] | 20.7 | 92% | 13% | 98% | 1.00 |
| rep105 | 46.9 [46.4, 47.5] | -0.8 [-1.0, -0.5] | 18.2 | 95% | 9% | 98% | 1.00 |
| rep110 | 41.7 [41.2, 42.2] | -6.0 [-6.3, -5.6] | 11.5 | 96% | 6% | 99% | 1.03 |
| freq005 | 48.7 [48.1, 49.2] | 1.0 [0.8, 1.2] | 20.8 | 99% | 6% | 98% | 0.99 |
| t03 | 47.7 [47.1, 48.3] | 0.1 [-0.2, 0.2] | 20.5 | 96% | 8% | 98% | 0.99 |
| t05 | 48.1 [47.5, 48.6] | 0.4 [0.2, 0.6] | 20.2 | 97% | 2% | 98% | 0.99 |
| t07 | 47.8 [47.2, 48.3] | 0.1 [-0.1, 0.3] | 19.0 | 100% | 0% | 98% | 1.00 |
| t10 | 45.4 [44.8, 46.0] | -2.2 [-2.5, -2.0] | 16.2 | 100% | 0% | 98% | 1.01 |
| bon4 | 48.4 [47.9, 48.9] | 0.7 [0.5, 0.9] | 19.7 | 100% | 0% | 98% | 1.00 |
| bon8 | 48.8 [48.2, 49.3] | 1.1 [0.9, 1.3] | 20.0 | 100% | 0% | 98% | 1.00 |
| greedy_sent | 53.5 [52.9, 54.0] | 5.8 [5.4, 6.1] | 25.0 | 94% | 1% | 100% | 1.01 |
| sentfix | 53.5 [52.9, 54.0] | 5.8 [5.4, 6.1] | 25.0 | 99% | 1% | 100% | 1.01 |
| greedy_win | 51.8 [51.3, 52.4] | 4.1 [3.9, 4.4] | 23.7 | 93% | 6% | 100% | 1.01 |
| winfix | 52.2 [51.7, 52.7] | 4.5 [4.2, 4.8] | 24.0 | 100% | 2% | 100% | 1.01 |
| greedy_winov | 51.6 [51.1, 52.1] | 3.9 [3.6, 4.2] | 23.6 | 93% | 7% | 100% | 1.01 |
| sft:greedy | 32.5 [31.1, 33.7] | -15.2 [-16.2, -14.1] | 11.8 | 41% | 50% | 85% | 1.33 |
| sft:greedy_sent | 51.1 [50.6, 51.7] | 3.4 [3.0, 3.9] | 23.4 | 48% | 42% | 98% | 1.01 |
| sft:greedy_win | 43.4 [42.6, 44.0] | -4.3 [-5.0, -3.8] | 18.6 | 47% | 44% | 92% | 1.08 |

#### core bin b1024 (vs NLLB primary, macro over 8 languages)

| config | chrF++ [95% CI] | d chrF++ vs greedy [CI] | BLEU | finished | loop | sent. cov. | len ratio |
|---|---|---|---|---|---|---|---|
| greedy | 29.7 [28.8, 30.6] | 0.0 [0.0, 0.0] | 8.6 | 65% | 53% | 95% | 1.03 |
| rep105 | 33.7 [32.9, 34.5] | 4.0 [3.5, 4.4] | 9.0 | 81% | 36% | 94% | 0.86 |
| rep110 | 32.4 [31.8, 32.9] | 2.6 [2.1, 3.2] | 6.1 | 90% | 22% | 94% | 0.81 |
| freq005 | 33.4 [32.7, 34.3] | 3.7 [3.3, 4.1] | 9.7 | 96% | 39% | 94% | 0.79 |
| t03 | 31.4 [30.5, 32.3] | 1.7 [1.2, 2.2] | 9.1 | 81% | 42% | 95% | 0.88 |
| t05 | 34.2 [33.5, 35.0] | 4.5 [4.1, 5.0] | 10.2 | 96% | 15% | 95% | 0.82 |
| t07 | 36.0 [35.3, 36.7] | 6.3 [5.7, 6.9] | 9.8 | 99% | 0% | 94% | 0.77 |
| t10 | 34.3 [33.7, 35.0] | 4.6 [4.0, 5.2] | 7.1 | 100% | 0% | 94% | 0.78 |
| bon4 | 36.9 [36.2, 37.5] | 7.2 [6.6, 7.7] | 10.3 | 100% | 1% | 94% | 0.78 |
| greedy_sent | 55.6 [55.2, 56.0] | 25.9 [25.0, 26.9] | 25.5 | 92% | 2% | 100% | 1.01 |
| sentfix | 55.6 [55.2, 56.0] | 25.9 [25.0, 26.9] | 25.5 | 98% | 2% | 100% | 1.01 |
| greedy_win | 54.0 [53.6, 54.5] | 24.3 [23.4, 25.2] | 24.1 | 91% | 9% | 100% | 1.01 |
| winfix | 54.4 [54.0, 54.8] | 24.7 [23.8, 25.7] | 24.3 | 100% | 4% | 100% | 1.01 |
| greedy_winov | 53.7 [53.2, 54.2] | 24.0 [23.1, 24.9] | 23.8 | 90% | 11% | 100% | 1.01 |
| sft:greedy | 17.0 [16.3, 17.7] | -12.7 [-13.5, -11.9] | 2.6 | 8% | 76% | 76% | 1.76 |
| sft:greedy_sent | 53.3 [52.8, 53.8] | 23.6 [22.6, 24.6] | 23.6 | 39% | 48% | 98% | 1.01 |
| sft:greedy_win | 46.2 [45.4, 46.8] | 16.4 [15.4, 17.5] | 19.2 | 42% | 48% | 93% | 1.12 |

#### core bin b2048 (vs NLLB primary, macro over 8 languages)

| config | chrF++ [95% CI] | d chrF++ vs greedy [CI] | BLEU | finished | loop | sent. cov. | len ratio |
|---|---|---|---|---|---|---|---|
| greedy | 18.2 [17.5, 18.9] | 0.0 [0.0, 0.0] | 1.8 | 42% | 73% | 95% | 1.19 |
| rep105 | 21.8 [21.2, 22.4] | 3.6 [3.1, 4.1] | 2.7 | 57% | 62% | 92% | 0.96 |
| rep110 | 23.5 [23.0, 24.1] | 5.3 [4.7, 5.9] | 2.3 | 62% | 55% | 92% | 0.85 |
| freq005 | 25.3 [24.7, 25.8] | 7.1 [6.4, 7.7] | 3.7 | 91% | 60% | 89% | 0.65 |
| t03 | 19.9 [19.2, 20.6] | 1.7 [1.3, 2.1] | 2.6 | 62% | 65% | 93% | 0.83 |
| t05 | 22.9 [22.3, 23.5] | 4.7 [4.2, 5.2] | 3.4 | 84% | 40% | 91% | 0.63 |
| t07 | 26.2 [25.7, 26.9] | 8.1 [7.4, 8.7] | 3.6 | 95% | 6% | 91% | 0.56 |
| t10 | 25.4 [24.8, 26.0] | 7.3 [6.6, 7.9] | 2.0 | 99% | 0% | 90% | 0.57 |
| bon4 | 26.7 [26.1, 27.2] | 8.5 [7.8, 9.0] | 3.7 | 99% | 5% | 91% | 0.56 |
| greedy_sent | 58.4 [57.7, 58.9] | 40.2 [39.3, 41.0] | 26.9 | 86% | 4% | 100% | 1.01 |
| sentfix | 58.4 [57.7, 58.9] | 40.2 [39.3, 41.0] | 26.9 | 97% | 3% | 100% | 1.01 |
| greedy_win | 57.0 [56.3, 57.5] | 38.8 [37.9, 39.6] | 25.8 | 87% | 11% | 100% | 1.00 |
| winfix | 57.4 [56.7, 57.9] | 39.2 [38.3, 40.0] | 26.0 | 100% | 6% | 100% | 1.00 |
| greedy_winov | 56.7 [56.1, 57.3] | 38.5 [37.7, 39.4] | 25.5 | 87% | 13% | 100% | 1.00 |
| sft:greedy | 11.1 [10.6, 11.6] | -7.1 [-7.8, -6.5] | 0.7 | 2% | 78% | 70% | 1.78 |
| sft:greedy_sent | 56.3 [55.7, 57.0] | 38.1 [37.3, 38.9] | 25.2 | 24% | 55% | 98% | 1.01 |
| sft:greedy_win | 49.3 [48.6, 50.1] | 31.2 [30.1, 32.2] | 20.6 | 30% | 53% | 93% | 1.14 |

#### core bin b4096 (vs NLLB primary, macro over 8 languages)

| config | chrF++ [95% CI] | d chrF++ vs greedy [CI] | BLEU | finished | loop | sent. cov. | len ratio |
|---|---|---|---|---|---|---|---|
| greedy | 9.8 [9.2, 10.7] | 0.0 [0.0, 0.0] | 0.7 | 31% | 73% | 83% | 0.53 |
| rep105 | 12.4 [11.7, 13.2] | 2.5 [2.2, 2.9] | 1.1 | 30% | 71% | 86% | 0.56 |
| rep110 | 12.6 [12.0, 13.4] | 2.8 [2.2, 3.2] | 0.7 | 26% | 70% | 85% | 0.58 |
| freq005 | 17.7 [17.0, 18.4] | 7.9 [7.1, 8.6] | 1.0 | 62% | 66% | 83% | 0.56 |
| t03 | 10.8 [10.1, 11.7] | 1.0 [0.6, 1.3] | 0.9 | 40% | 70% | 84% | 0.46 |
| t05 | 12.2 [11.5, 13.1] | 2.4 [2.0, 2.8] | 1.0 | 60% | 56% | 85% | 0.38 |
| t07 | 14.5 [13.8, 15.5] | 4.7 [4.2, 5.2] | 0.9 | 78% | 15% | 81% | 0.33 |
| t10 | 13.0 [12.5, 13.6] | 3.1 [2.5, 3.7] | 0.2 | 98% | 0% | 65% | 0.30 |
| bon4 | 14.5 [13.8, 15.2] | 4.6 [4.0, 5.2] | 0.9 | 87% | 14% | 78% | 0.31 |
| greedy_sent | 60.1 [59.4, 60.9] | 50.3 [49.3, 51.2] | 27.3 | 82% | 5% | 100% | 1.00 |
| sentfix | 60.1 [59.4, 60.9] | 50.3 [49.3, 51.2] | 27.3 | 94% | 4% | 100% | 1.00 |
| greedy_win | 58.6 [57.8, 59.4] | 48.8 [47.7, 49.7] | 26.0 | 86% | 13% | 100% | 1.00 |
| winfix | 59.1 [58.4, 59.9] | 49.3 [48.3, 50.2] | 26.3 | 100% | 8% | 100% | 1.00 |
| greedy_winov | 58.4 [57.6, 59.2] | 48.6 [47.5, 49.5] | 25.8 | 85% | 14% | 100% | 1.00 |
| sft:greedy | 6.9 [6.3, 7.6] | -2.9 [-3.6, -2.3] | 0.2 | 1% | 72% | 70% | 0.73 |
| sft:greedy_sent | 57.8 [57.1, 58.6] | 48.0 [47.0, 48.9] | 25.3 | 13% | 59% | 98% | 1.01 |
| sft:greedy_win | 50.6 [49.7, 51.7] | 40.8 [39.6, 42.0] | 20.3 | 19% | 54% | 94% | 1.15 |

#### all bins pooled (vs NLLB primary, macro over 8 languages)

| config | chrF++ [95% CI] | d chrF++ vs greedy [CI] | BLEU | finished | loop | sent. cov. | len ratio |
|---|---|---|---|---|---|---|---|
| greedy | 22.7 [21.1, 24.3] | 0.0 [0.0, 0.0] | 5.6 | 77% | 29% | 97% | 1.00 |
| rep105 | 25.3 [23.7, 27.1] | 2.6 [2.4, 2.9] | 6.2 | 83% | 23% | 96% | 0.94 |
| rep110 | 24.8 [23.4, 26.4] | 2.1 [1.7, 2.5] | 4.4 | 85% | 19% | 96% | 0.94 |
| freq005 | 28.3 [27.0, 29.7] | 5.7 [5.3, 6.0] | 7.1 | 94% | 23% | 95% | 0.87 |
| t03 | 24.0 [22.4, 25.7] | 1.3 [1.1, 1.6] | 6.5 | 85% | 25% | 96% | 0.91 |
| t05 | 26.0 [24.4, 27.8] | 3.4 [3.1, 3.7] | 7.0 | 93% | 13% | 96% | 0.86 |
| t07 | 28.2 [26.6, 29.9] | 5.5 [5.1, 5.9] | 6.7 | 97% | 2% | 96% | 0.85 |
| t10 | 26.8 [25.2, 28.5] | 4.1 [3.7, 4.6] | 4.6 | 99% | 0% | 94% | 0.85 |
| bon4 | 28.6 [26.9, 30.4] | 5.9 [5.5, 6.3] | 6.8 | 98% | 2% | 95% | 0.85 |
| greedy_sent | 57.5 [56.9, 57.9] | 34.8 [32.6, 36.6] | 26.4 | 93% | 2% | 100% | 1.01 |
| sentfix | 57.5 [56.9, 57.9] | 34.8 [32.6, 36.6] | 26.4 | 98% | 1% | 100% | 1.01 |
| greedy_win | 55.9 [55.4, 56.4] | 33.3 [31.0, 35.0] | 25.2 | 93% | 6% | 100% | 1.00 |
| winfix | 56.4 [55.8, 56.8] | 33.7 [31.4, 35.5] | 25.4 | 100% | 3% | 100% | 1.00 |
| greedy_winov | 55.7 [55.2, 56.1] | 33.1 [30.8, 34.8] | 24.9 | 93% | 7% | 100% | 1.00 |
| sft:greedy | 14.7 [13.7, 15.8] | -8.0 [-8.7, -7.4] | 2.8 | 41% | 48% | 85% | 1.32 |
| sft:greedy_sent | 55.2 [54.7, 55.6] | 32.5 [30.4, 34.3] | 24.6 | 52% | 37% | 99% | 1.01 |
| sft:greedy_win | 48.0 [47.4, 48.6] | 25.4 [23.2, 27.3] | 19.9 | 54% | 37% | 95% | 1.08 |

## 4. Beam search and no-repeat n-grams (4 languages hau yor swh amh, short bins; vs NLLB)

Beam 2 / beam 4 / no_repeat_ngram 6 with HF `generate` (greedy, sampling and sentence rows are the same 4 languages for comparison). Not run above b512 for beams.

#### core bin b64 (vs NLLB primary, macro over 4 languages)

| config | chrF++ [95% CI] | d chrF++ vs greedy [CI] | BLEU | finished | loop | sent. cov. | len ratio |
|---|---|---|---|---|---|---|---|
| greedy | 50.4 [48.8, 51.9] | 0.0 [0.0, 0.0] | 30.4 | 98% | 0% | 100% | 1.01 |
| beam2 | 51.1 [49.4, 52.7] | 0.7 [0.0, 1.4] | 30.7 | 97% | 0% | 100% | 1.00 |
| beam4 | 52.0 [50.5, 53.6] | 1.6 [0.9, 2.4] | 31.5 | 97% | 0% | 100% | 1.00 |
| nr6 | 49.8 [48.2, 51.3] | -0.6 [-1.3, 0.1] | 28.7 | 98% | 0% | 100% | 1.00 |
| greedy_sent | 50.2 [48.5, 51.9] | -0.1 [-0.8, 0.5] | 29.8 | 99% | 0% | 100% | 1.01 |
| sentfix | 50.2 [48.5, 51.9] | -0.2 [-0.7, 0.5] | 29.8 | 100% | 0% | 100% | 1.01 |
| t03 | 50.2 [48.6, 51.8] | -0.2 [-0.7, 0.4] | 29.9 | 98% | 0% | 100% | 1.01 |
| bon4 | 49.5 [47.8, 51.2] | -0.9 [-1.5, -0.1] | 28.9 | 98% | 0% | 100% | 1.01 |

#### core bin b256 (vs NLLB primary, macro over 4 languages)

| config | chrF++ [95% CI] | d chrF++ vs greedy [CI] | BLEU | finished | loop | sent. cov. | len ratio |
|---|---|---|---|---|---|---|---|
| greedy | 49.7 [48.7, 50.7] | 0.0 [0.0, 0.0] | 27.8 | 97% | 1% | 100% | 1.00 |
| beam2 | 50.6 [49.5, 51.6] | 0.9 [0.6, 1.2] | 28.3 | 98% | 0% | 100% | 1.00 |
| beam4 | 51.0 [49.8, 52.0] | 1.2 [0.9, 1.6] | 28.9 | 98% | 0% | 100% | 1.00 |
| nr6 | 49.0 [48.0, 49.9] | -0.7 [-1.0, -0.4] | 26.0 | 99% | 0% | 100% | 1.00 |
| greedy_sent | 50.4 [49.4, 51.4] | 0.7 [0.3, 1.0] | 28.4 | 98% | 0% | 100% | 1.01 |
| sentfix | 50.4 [49.4, 51.4] | 0.7 [0.3, 1.0] | 28.4 | 100% | 0% | 100% | 1.01 |
| t03 | 49.3 [48.2, 50.2] | -0.5 [-0.7, -0.2] | 26.9 | 98% | 0% | 100% | 1.01 |
| bon4 | 48.7 [47.7, 49.7] | -1.0 [-1.3, -0.7] | 26.3 | 99% | 0% | 100% | 1.01 |

#### core bin b512 (vs NLLB primary, macro over 4 languages)

| config | chrF++ [95% CI] | d chrF++ vs greedy [CI] | BLEU | finished | loop | sent. cov. | len ratio |
|---|---|---|---|---|---|---|---|
| greedy | 50.0 [49.2, 50.7] | 0.0 [0.0, 0.0] | 25.2 | 92% | 13% | 98% | 1.00 |
| beam2 | 50.8 [50.1, 51.4] | 0.8 [0.6, 1.1] | 26.0 | 98% | 2% | 97% | 0.99 |
| beam4 | 51.2 [50.5, 51.9] | 1.3 [1.0, 1.6] | 26.4 | 98% | 4% | 97% | 0.99 |
| nr6 | 49.4 [48.7, 50.0] | -0.6 [-0.8, -0.4] | 23.5 | 100% | 0% | 98% | 0.98 |
| greedy_sent | 54.0 [53.3, 54.6] | 4.1 [3.7, 4.4] | 28.9 | 94% | 1% | 100% | 1.01 |
| sentfix | 54.0 [53.3, 54.6] | 4.0 [3.7, 4.4] | 29.0 | 99% | 1% | 100% | 1.01 |
| t03 | 49.7 [49.0, 50.4] | -0.2 [-0.4, -0.0] | 24.9 | 96% | 8% | 98% | 0.99 |
| bon4 | 49.1 [48.4, 49.7] | -0.9 [-1.1, -0.6] | 23.6 | 100% | 0% | 98% | 1.00 |

`no_repeat_ngram_size=6` on long whole documents (hau + yor only, 2 languages, so the numbers differ from the 8-language tables):

#### core bin b1024 (vs NLLB primary, macro over 2 languages)

| config | chrF++ [95% CI] | d chrF++ vs greedy [CI] | BLEU | finished | loop | sent. cov. | len ratio |
|---|---|---|---|---|---|---|---|
| greedy | 34.2 [32.7, 35.7] | 0.0 [0.0, 0.0] | 13.3 | 65% | 53% | 95% | 1.03 |
| nr6hl | 38.5 [37.6, 39.5] | 4.3 [3.1, 5.5] | 13.7 | 99% | 0% | 93% | 0.75 |
| rep105 | 37.3 [36.3, 38.2] | 3.0 [1.9, 4.2] | 11.7 | 81% | 36% | 94% | 0.86 |
| t07 | 39.2 [38.2, 40.4] | 5.0 [3.8, 6.2] | 13.5 | 99% | 0% | 94% | 0.77 |
| sentfix | 61.8 [61.0, 62.5] | 27.6 [26.1, 29.0] | 34.6 | 98% | 2% | 100% | 1.01 |

#### core bin b2048 (vs NLLB primary, macro over 2 languages)

| config | chrF++ [95% CI] | d chrF++ vs greedy [CI] | BLEU | finished | loop | sent. cov. | len ratio |
|---|---|---|---|---|---|---|---|
| greedy | 20.9 [20.1, 21.7] | 0.0 [0.0, 0.0] | 2.4 | 42% | 73% | 95% | 1.19 |
| nr6hl | 27.3 [26.4, 28.2] | 6.4 [5.3, 7.3] | 5.3 | 95% | 0% | 90% | 0.56 |
| rep105 | 25.5 [24.7, 26.3] | 4.6 [3.8, 5.5] | 4.1 | 57% | 62% | 92% | 0.96 |
| t07 | 28.3 [27.4, 29.1] | 7.4 [6.5, 8.3] | 5.3 | 95% | 6% | 91% | 0.56 |
| sentfix | 64.8 [63.8, 65.6] | 43.9 [42.7, 44.9] | 36.4 | 97% | 3% | 100% | 1.01 |

## 5. Human references (SmolDoc, clean documents, B8 languages with human references: hau yor swh amh zul wol; st2048 only hau yor swh amh)

Macro over languages of the per-language corpus chrF++; CIs from resampling documents within each language. `finished` / `loop` are computed on the same documents.

#### human bin sm (macro over 6 languages)

| config | chrF++ [95% CI] | d vs greedy [CI] | BLEU | finished | loop |
|---|---|---|---|---|---|
| greedy | 46.0 [45.4, 46.6] | 0.0 [0.0, 0.0] | 17.8 | 100% | 0% |
| rep105 | 45.1 [44.5, 45.6] | -0.9 [-1.1, -0.7] | 15.7 | 100% | 0% |
| rep110 | 40.6 [40.2, 41.0] | -5.4 [-5.8, -5.0] | 10.7 | 100% | 0% |
| freq005 | 46.1 [45.5, 46.7] | 0.1 [-0.1, 0.3] | 17.8 | 100% | 0% |
| t03 | 45.8 [45.2, 46.4] | -0.2 [-0.4, -0.1] | 17.6 | 100% | 0% |
| t05 | 45.6 [45.0, 46.2] | -0.4 [-0.6, -0.2] | 17.2 | 100% | 0% |
| t07 | 45.2 [44.6, 45.8] | -0.8 [-1.0, -0.5] | 16.8 | 100% | 0% |
| t10 | 43.7 [43.2, 44.3] | -2.3 [-2.5, -2.0] | 15.2 | 100% | 0% |
| bon4 | 45.5 [44.9, 46.1] | -0.5 [-0.7, -0.3] | 16.9 | 100% | 0% |
| bon8 | 45.7 [45.1, 46.3] | -0.3 [-0.5, -0.1] | 17.2 | 100% | 0% |
| greedy_sent | 47.2 [46.6, 47.8] | 1.2 [0.9, 1.5] | 18.7 | 99% | 0% |
| sentfix | 47.2 [46.6, 47.8] | 1.2 [0.9, 1.5] | 18.7 | 100% | 0% |
| greedy_win | 46.6 [45.9, 47.1] | 0.6 [0.3, 0.8] | 18.1 | 100% | 0% |
| winfix | 46.6 [45.9, 47.1] | 0.6 [0.3, 0.8] | 18.1 | 100% | 0% |
| greedy_winov | 46.6 [46.0, 47.1] | 0.6 [0.4, 0.8] | 17.9 | 100% | 0% |
| sft:greedy | 40.0 [38.9, 41.3] | -6.0 [-7.1, -5.0] | 15.1 | 88% | 12% |
| sft:greedy_sent | 44.4 [43.7, 45.1] | -1.6 [-2.1, -1.1] | 17.3 | 84% | 13% |
| sft:greedy_win | 41.8 [40.9, 42.8] | -4.2 [-5.0, -3.4] | 16.0 | 87% | 12% |

#### human bin st512 (macro over 6 languages)

| config | chrF++ [95% CI] | d vs greedy [CI] | BLEU | finished | loop |
|---|---|---|---|---|---|
| greedy | 46.8 [46.2, 47.3] | 0.0 [0.0, 0.0] | 17.8 | 100% | 1% |
| rep105 | 45.9 [45.4, 46.4] | -0.9 [-1.2, -0.6] | 15.5 | 100% | 1% |
| rep110 | 41.6 [41.2, 42.1] | -5.2 [-5.7, -4.7] | 10.4 | 100% | 0% |
| freq005 | 47.0 [46.4, 47.5] | 0.2 [-0.1, 0.4] | 17.6 | 100% | 1% |
| t03 | 46.6 [46.0, 47.1] | -0.2 [-0.4, -0.0] | 17.4 | 100% | 1% |
| t05 | 46.3 [45.8, 46.8] | -0.5 [-0.8, -0.3] | 16.9 | 100% | 0% |
| t07 | 45.9 [45.3, 46.4] | -1.0 [-1.2, -0.7] | 16.2 | 100% | 0% |
| t10 | 44.7 [44.1, 45.1] | -2.2 [-2.5, -1.9] | 14.9 | 100% | 0% |
| bon4 | 46.3 [45.8, 46.8] | -0.5 [-0.7, -0.3] | 16.7 | 100% | 0% |
| bon8 | 46.5 [46.0, 47.0] | -0.3 [-0.6, -0.0] | 16.8 | 100% | 0% |
| greedy_sent | 49.1 [48.5, 49.6] | 2.3 [2.0, 2.6] | 19.3 | 96% | 1% |
| sentfix | 49.1 [48.5, 49.6] | 2.3 [2.0, 2.6] | 19.3 | 100% | 1% |
| greedy_win | 48.7 [48.1, 49.2] | 1.8 [1.6, 2.1] | 18.7 | 99% | 1% |
| winfix | 48.7 [48.1, 49.3] | 1.9 [1.6, 2.1] | 18.7 | 100% | 1% |
| greedy_winov | 48.5 [48.0, 49.0] | 1.7 [1.4, 1.9] | 18.2 | 99% | 1% |
| sft:greedy | 39.3 [37.7, 40.9] | -7.5 [-9.0, -6.0] | 14.2 | 68% | 31% |
| sft:greedy_sent | 47.0 [46.4, 47.6] | 0.2 [-0.3, 0.7] | 18.2 | 75% | 22% |
| sft:greedy_win | 44.0 [42.9, 44.9] | -2.9 [-3.7, -1.9] | 16.8 | 77% | 22% |

#### human bin st1024 (macro over 6 languages)

| config | chrF++ [95% CI] | d vs greedy [CI] | BLEU | finished | loop |
|---|---|---|---|---|---|
| greedy | 29.0 [27.9, 30.1] | 0.0 [0.0, 0.0] | 6.7 | 72% | 60% |
| rep105 | 32.3 [31.5, 33.1] | 3.3 [2.2, 4.4] | 6.8 | 87% | 34% |
| rep110 | 31.8 [31.3, 32.4] | 2.8 [1.6, 4.0] | 4.7 | 97% | 11% |
| freq005 | 30.8 [28.9, 32.5] | 1.8 [0.0, 3.7] | 7.5 | 94% | 43% |
| t03 | 28.9 [26.6, 31.4] | -0.1 [-2.1, 2.4] | 6.8 | 82% | 44% |
| t05 | 31.6 [30.7, 32.5] | 2.6 [1.8, 3.6] | 7.9 | 93% | 18% |
| t07 | 33.7 [32.9, 34.6] | 4.7 [3.6, 5.9] | 7.9 | 99% | 1% |
| t10 | 33.9 [33.1, 34.7] | 4.9 [3.9, 6.2] | 6.2 | 100% | 0% |
| bon4 | 33.3 [32.6, 34.0] | 4.3 [3.3, 5.5] | 7.6 | 100% | 0% |
| greedy_sent | 50.6 [50.0, 51.3] | 21.7 [20.4, 23.1] | 17.7 | 93% | 1% |
| sentfix | 50.6 [50.0, 51.3] | 21.7 [20.4, 23.1] | 17.7 | 99% | 0% |
| greedy_win | 50.2 [49.5, 50.9] | 21.3 [19.9, 22.7] | 17.5 | 99% | 1% |
| winfix | 50.2 [49.5, 50.9] | 21.3 [19.9, 22.7] | 17.5 | 100% | 1% |
| greedy_winov | 50.1 [49.5, 50.8] | 21.2 [19.9, 22.6] | 17.0 | 98% | 0% |
| sft:greedy | 15.9 [14.4, 17.3] | -13.1 [-14.8, -11.5] | 1.6 | 8% | 89% |
| sft:greedy_sent | 47.9 [47.2, 48.5] | 18.9 [17.7, 20.3] | 16.3 | 57% | 39% |
| sft:greedy_win | 44.9 [44.0, 45.7] | 15.9 [14.5, 17.3] | 14.6 | 65% | 34% |

#### human bin st2048 (macro over 4 languages)

| config | chrF++ [95% CI] | d vs greedy [CI] | BLEU | finished | loop |
|---|---|---|---|---|---|
| greedy | 18.2 [15.4, 20.5] | 0.0 [0.0, 0.0] | 2.1 | 53% | 91% |
| rep105 | 22.1 [20.8, 23.4] | 3.9 [1.6, 6.6] | 2.7 | 57% | 77% |
| rep110 | 23.1 [21.6, 24.5] | 5.0 [2.3, 8.3] | 2.1 | 64% | 61% |
| freq005 | 25.4 [23.3, 27.1] | 7.2 [5.5, 8.8] | 4.1 | 95% | 81% |
| t03 | 19.8 [17.7, 21.8] | 1.6 [-1.6, 5.3] | 2.3 | 66% | 81% |
| t05 | 22.5 [20.3, 24.5] | 4.3 [1.0, 7.6] | 3.4 | 82% | 45% |
| t07 | 25.4 [22.5, 27.5] | 7.2 [4.0, 10.4] | 3.2 | 98% | 7% |
| t10 | 26.3 [23.8, 28.7] | 8.1 [4.9, 12.5] | 2.3 | 100% | 0% |
| bon4 | 25.0 [23.5, 26.3] | 6.8 [4.4, 9.7] | 3.1 | 98% | 6% |
| greedy_sent | 52.4 [51.8, 53.0] | 34.2 [32.0, 37.3] | 20.2 | 88% | 2% |
| sentfix | 52.4 [51.8, 53.0] | 34.2 [32.0, 37.3] | 20.2 | 98% | 2% |
| greedy_win | 52.4 [51.6, 53.1] | 34.2 [32.0, 37.2] | 20.3 | 99% | 2% |
| winfix | 52.4 [51.6, 53.1] | 34.2 [32.0, 37.2] | 20.3 | 100% | 2% |
| greedy_winov | 52.3 [51.6, 53.0] | 34.1 [32.0, 37.1] | 19.8 | 97% | 2% |
| sft:greedy | 10.1 [8.7, 11.2] | -8.1 [-10.6, -5.3] | 0.5 | 2% | 96% |
| sft:greedy_sent | 50.1 [49.1, 51.0] | 31.9 [29.4, 35.0] | 18.8 | 44% | 52% |
| sft:greedy_win | 48.6 [47.3, 50.0] | 30.4 [27.8, 33.5] | 18.7 | 52% | 47% |

### 5.1 AfriCOMET on a subset (reference-based / reference-free QE; macro over languages)

Core: 17 documents per bin x 8 languages (reference = NLLB); human: up to 15 documents per language and bin (reference = SmolDoc; the `st2048` column is the contaminated stitched subset). Computed on aligned chunks, so a document that is half missing is scored on what the alignment can pair - read together with the coverage / length-ratio columns above.

**Core set**: AfriCOMET (ref-based / QE), macro over languages

| config | b64 ref / QE | b256 ref / QE | b512 ref / QE | b1024 ref / QE | b2048 ref / QE | b4096 ref / QE |
|---|---|---|---|---|---|---|
| greedy_sent | 0.692 / 0.617 | 0.664 / 0.595 | 0.679 / 0.617 | 0.671 / 0.604 | 0.665 / 0.601 | 0.658 / 0.598 |
| greedy | 0.686 / 0.610 | 0.656 / 0.584 | 0.611 / 0.558 | 0.321 / 0.297 | 0.201 / 0.180 | 0.128 / 0.113 |
| t03 | 0.678 / 0.607 | 0.644 / 0.578 | 0.612 / 0.561 | 0.329 / 0.311 | 0.220 / 0.210 | 0.133 / 0.124 |
| greedy_win | 0.687 / 0.612 | 0.653 / 0.581 | 0.665 / 0.603 | 0.653 / 0.588 | 0.649 / 0.587 | 0.644 / 0.586 |
| sentfix | 0.691 / 0.617 | 0.664 / 0.595 | 0.679 / 0.617 | 0.670 / 0.603 | 0.665 / 0.602 | 0.659 / 0.600 |
| bon4 | 0.677 / 0.607 | 0.640 / 0.573 | 0.602 / 0.555 | 0.394 / 0.391 | 0.245 / 0.249 | 0.129 / 0.140 |
| greedy_winov | 0.685 / 0.610 | 0.653 / 0.582 | 0.662 / 0.602 | 0.645 / 0.582 | 0.635 / 0.577 | 0.639 / 0.582 |
| winfix | 0.687 / 0.614 | 0.653 / 0.581 | 0.665 / 0.603 | 0.653 / 0.588 | 0.649 / 0.587 | 0.644 / 0.586 |
| sft:greedy_sent | 0.634 / 0.558 | 0.574 / 0.508 | 0.601 / 0.535 | 0.589 / 0.517 | 0.573 / 0.504 | 0.569 / 0.505 |
| sft:greedy | 0.640 / 0.565 | 0.535 / 0.471 | 0.345 / 0.305 | 0.154 / 0.137 | 0.084 / 0.064 | 0.101 / 0.072 |
| sft:greedy_win | 0.638 / 0.567 | 0.535 / 0.466 | 0.489 / 0.440 | 0.481 / 0.426 | 0.438 / 0.394 | 0.430 / 0.392 |

**Human set**: AfriCOMET (ref-based / QE), macro over languages

| config | bsm ref / QE | bst512 ref / QE | bst1024 ref / QE | bst2048 ref / QE |
|---|---|---|---|---|
| greedy_sent | 0.691 / 0.638 | 0.683 / 0.642 | 0.658 / 0.622 | 0.680 / 0.647 |
| greedy | 0.651 / 0.602 | 0.621 / 0.585 | 0.334 / 0.333 | 0.220 / 0.204 |
| t03 | 0.648 / 0.601 | 0.620 / 0.585 | 0.337 / 0.345 | 0.216 / 0.209 |
| greedy_win | 0.661 / 0.613 | 0.655 / 0.615 | 0.638 / 0.604 | 0.669 / 0.636 |
| sentfix | 0.691 / 0.638 | 0.683 / 0.642 | 0.658 / 0.622 | 0.680 / 0.648 |
| bon4 | 0.646 / 0.599 | 0.617 / 0.581 | 0.363 / 0.375 | 0.244 / 0.251 |
| greedy_winov | 0.655 / 0.605 | 0.642 / 0.605 | 0.624 / 0.588 | 0.645 / 0.616 |
| winfix | 0.661 / 0.613 | 0.655 / 0.615 | 0.638 / 0.604 | 0.669 / 0.636 |
| sft:greedy_sent | 0.611 / 0.565 | 0.613 / 0.577 | 0.566 / 0.539 | 0.604 / 0.579 |
| sft:greedy | 0.541 / 0.503 | 0.482 / 0.459 | 0.167 / 0.152 | 0.115 / 0.097 |
| sft:greedy_win | 0.556 / 0.515 | 0.572 / 0.540 | 0.483 / 0.461 | 0.533 / 0.518 |

### 5.2 Total RL reward (the reward the RL run optimises; per-output, no group consensus)

Mean reward, macro over the B8 languages, 12 documents per bin per language. A document with a repetition loop anywhere ("degenerate"), no end token, or a broken length ratio gets the failure reward (-2.0). The observed minimum is -2.0.

| system | b64 | b256 | b512 | b1024 | b2048 | b4096 | all bins |
|---|---|---|---|---|---|---|---|
| rl2c:bon4 | 0.62 | 0.72 | -0.09 | -1.07 | -1.64 | -1.84 | -0.55 |
| rl2c:greedy | 0.73 | 0.60 | -0.25 | -1.53 | -1.95 | -2.00 | -0.73 |
| rl2c:greedy_sent | 0.75 | 0.83 | -0.02 | -0.33 | -0.59 | -1.08 | -0.07 |
| rl2c:greedy_win | 0.71 | 0.64 | 0.01 | -0.30 | -0.48 | -1.15 | -0.09 |
| rl2c:rep105 | 0.75 | 0.61 | 0.03 | -1.28 | -1.86 | -2.00 | -0.63 |
| rl2c:sentfix | 0.77 | 0.83 | 0.12 | -0.19 | -0.46 | -1.03 | 0.01 |
| rl2c:t03 | 0.69 | 0.69 | -0.24 | -1.47 | -1.95 | -2.00 | -0.71 |
| rl2c:t07 | 0.54 | 0.67 | -0.19 | -1.10 | -1.59 | -1.84 | -0.59 |
| rl2c:t10 | 0.56 | 0.53 | -0.05 | -0.70 | -1.20 | -1.78 | -0.44 |
| rl2c:winfix | 0.75 | 0.78 | 0.01 | -0.27 | -0.48 | -1.15 | -0.06 |
| sft:greedy | 0.42 | -0.15 | -1.26 | -1.91 | -1.99 | -2.00 | -1.15 |
| sft:greedy_sent | 0.30 | -0.11 | -0.88 | -1.12 | -1.39 | -1.65 | -0.81 |
| sft:greedy_win | 0.45 | -0.31 | -0.98 | -1.21 | -1.27 | -1.66 | -0.83 |

ok-rate (status == ok):

| system | b64 | b256 | b512 | b1024 | b2048 | b4096 |
|---|---|---|---|---|---|---|
| rl2c:bon4 | 0.96 | 0.99 | 0.82 | 0.67 | 0.47 | 0.45 |
| rl2c:greedy | 0.99 | 0.92 | 0.75 | 0.30 | 0.07 | 0.00 |
| rl2c:greedy_sent | 0.99 | 1.00 | 0.74 | 0.67 | 0.58 | 0.39 |
| rl2c:greedy_win | 0.99 | 0.93 | 0.77 | 0.69 | 0.64 | 0.36 |
| rl2c:rep105 | 0.99 | 0.95 | 0.86 | 0.52 | 0.20 | 0.00 |
| rl2c:sentfix | 1.00 | 1.00 | 0.79 | 0.71 | 0.62 | 0.41 |
| rl2c:t03 | 0.97 | 0.96 | 0.76 | 0.35 | 0.08 | 0.00 |
| rl2c:t07 | 0.95 | 0.99 | 0.81 | 0.66 | 0.50 | 0.40 |
| rl2c:t10 | 0.96 | 0.98 | 1.00 | 1.00 | 0.96 | 0.65 |
| rl2c:winfix | 1.00 | 0.99 | 0.77 | 0.70 | 0.64 | 0.36 |
| sft:greedy | 0.96 | 0.73 | 0.36 | 0.05 | 0.01 | 0.00 |
| sft:greedy_sent | 0.93 | 0.72 | 0.45 | 0.35 | 0.24 | 0.14 |
| sft:greedy_win | 0.99 | 0.68 | 0.44 | 0.34 | 0.32 | 0.15 |

Yardsticks under the same reward and documents (B8; `yard_nllb` = the NLLB pseudo-reference itself used as the hypothesis; `yard_madlad` has only swh and zul in B8):

| system | b64 | b256 | b512 | b1024 | b2048 | b4096 | all bins |
|---|---|---|---|---|---|---|---|
| yard_madlad | 0.59 | 0.62 | 0.50 | 0.16 | -0.08 | -0.73 | 0.18 |
| yard_nllb | 1.03 | 0.89 | 0.31 | 0.10 | -0.20 | -0.96 | 0.19 |

ok-rate (status == ok):

| system | b64 | b256 | b512 | b1024 | b2048 | b4096 |
|---|---|---|---|---|---|---|
| yard_madlad | 0.96 | 0.96 | 0.96 | 0.88 | 0.79 | 0.54 |
| yard_nllb | 0.99 | 0.98 | 0.85 | 0.80 | 0.69 | 0.40 |

Two things to read from this: (i) the greedy whole-document row reproduces the -1.9 you saw on b2048 prompts, (ii) **the reward's own detector fails even the NLLB translations at long bins** (b1024 0.10, b2048 -0.20, b4096 -0.96; the "degenerate" status fires on e.g. every sag and 25 of 36 wol long documents of the NLLB translations), so a total reward of about -0.2 / -0.5 / -1.0 at b1024 / b2048 / b4096 is the **ceiling in practice**, and sentfix is at that level (-0.19 / -0.46 / -1.03). The reward is not a useful model-selection signal at b4096.

### 5.3 Cost: generated tokens per source token (B8) and measured vLLM throughput

| config | b64 | b256 | b512 | b1024 | b2048 | b4096 |
|---|---|---|---|---|---|---|
| greedy | 1.97 | 1.97 | 2.01 | 2.34 | 2.66 | 1.29 |
| greedy_sent | 2.01 | 2.00 | 1.99 | 1.97 | 1.98 | 1.99 |
| sentfix | 2.01 | 2.00 | 1.99 | 1.97 | 1.98 | 1.99 |
| greedy_win | 1.97 | 1.98 | 1.97 | 1.95 | 1.95 | 1.96 |
| winfix | 1.96 | 1.93 | 1.92 | 1.90 | 1.90 | 1.89 |
| greedy_winov | 1.97 | 1.98 | 2.19 | 2.18 | 2.19 | 2.18 |
| t03 | 1.97 | 1.96 | 1.96 | 2.06 | 2.16 | 1.23 |
| t05 | 1.97 | 1.96 | 1.90 | 1.75 | 1.62 | 1.04 |
| t07 | 2.00 | 1.94 | 1.87 | 1.55 | 1.30 | 0.89 |
| t10 | 2.02 | 1.96 | 1.88 | 1.48 | 1.12 | 0.61 |
| bon4 | 7.95 | 7.74 | 7.47 | 6.20 | 5.22 | 3.48 |
| bon8 | 15.85 | 15.47 | 14.94 | - | - | - |
| rep105 | 1.98 | 1.93 | 1.93 | 1.96 | 2.22 | 1.33 |
| rep110 | 2.07 | 1.93 | 1.86 | 1.71 | 2.10 | 1.35 |
| freq005 | 1.95 | 1.94 | 1.86 | 1.60 | 1.52 | 1.15 |
| sft:greedy | 1.98 | 2.30 | 2.78 | 3.32 | 3.45 | 1.57 |
| sft:greedy_sent | 2.09 | 2.12 | 2.11 | 2.07 | 2.07 | 2.09 |

| job | requests | generated tokens | wall s | tok/s |
|---|---|---|---|---|
| rl2c/beam2:full_h | - | 0.5M | 845 | 596 |
| rl2c/beam4:full_h | - | 0.5M | 1331 | 381 |
| rl2c/freq005:full_b | 4792 | 6.3M | 737 | 8514 |
| rl2c/greedy:full | 9506 | 17.6M | 1912 | 9224 |
| rl2c/greedy:sent | 331642 | 16.3M | 1196 | 13593 |
| rl2c/greedy:win | 42160 | 16.8M | 1173 | 14336 |
| rl2c/greedy:winov_b | 21281 | 9.0M | 702 | 12833 |
| rl2c/nr6:full_h | - | 0.5M | 332 | 1487 |
| rl2c/nr6:full_hl | - | 0.7M | 1518 | 443 |
| rl2c/rep105:full_b | 4792 | 7.7M | 769 | 10061 |
| rl2c/rep110:full_b | 4792 | 7.3M | 1139 | 6406 |
| rl2c/t03:full_b | 4792 | 7.7M | 801 | 9561 |
| rl2c/t05:full_b | 4792 | 6.5M | 880 | 7340 |
| rl2c/t07x4:full_bl | 1864 | 16.4M | 1847 | 8859 |
| rl2c/t07x4:sentfail | 3250 | 1.2M | 90 | 13771 |
| rl2c/t07x4:winfail | 861 | 1.7M | 130 | 12764 |
| rl2c/t07x8:full_bs | 2928 | 12.2M | 976 | 12514 |
| rl2c/t10:full_b | 4792 | 5.1M | 501 | 10125 |
| sft/greedy:full_b | 4792 | 11.5M | 1475 | 7807 |
| sft/greedy:sent_b | 168829 | 8.0M | 777 | 10321 |
| sft/greedy:win_b | 21281 | 9.7M | 733 | 13185 |

Throughput is whatever the shared GPU gave at the time (two vLLM engines + other users' jobs for part of the run); it is a relative guide only. The token counts of `sentfix` / `winfix` exclude the re-decoding of the failed units (+1.2 M and +1.7 M tokens, about +7 % and +10 % of the base run).

## 6. Best configuration per bin

Decision rule: highest macro chrF++ against the primary reference, confirmed (not contradicted) by the human-reference tables, AfriCOMET and the reward; ties within the CIs go to the cheaper / simpler setting.

| bin | winner | chrF++ vs NLLB (B8) | vs human (clean) | notes |
|---|---|---|---|---|
| b64 | **tie**: greedy whole-doc = greedy_sent = bon8 (47.6); sentfix for uniformity; beam 4 is +1.6 [0.9, 2.4] on its 4-language test | 47.6 | `sm`: sentfix 47.2 vs greedy 46.0 (+1.2) | everything within +-0.7 except penalties / T >= 0.5 (worse) |
| b256 | **sentfix** | 49.0 (greedy 47.7; +1.4 [1.1, 1.6]) | (no human bin of this length) | reward 0.83 vs 0.60 |
| b512 | **sentfix** | 53.5 (greedy 47.7; best whole-doc bon8 48.8, beam 4 +1.3) | `st512` 49.1 vs 46.8 (+2.3) | sentence beats 256-token windows (winfix) by 1.3 on NLLB, 0.4 on human refs |
| b1024 | **sentfix** | 55.6 (greedy 29.7; best whole-doc bon4 36.9) | `st1024` 50.6 vs 29.0; windows 50.2 (tie) | |
| b2048 | **sentfix** | 58.4 (greedy 18.2; best whole-doc bon4 26.7) | `st2048` 52.4 vs 18.2; windows 52.4 (tie) | |
| b4096 | **sentfix** | 60.1 (greedy 9.8; best whole-doc t07 / bon4 14.5) | (no human b4096) | |

**Overall recommendation.** Translate **sentence by sentence with the same model, greedy**, join with the source's paragraph structure, and re-decode the units that do not emit the end token with best-of-4 at T = 0.7 (consensus pick); do not feed whole documents above ~256-512 tokens and do not use repetition penalties (they cost 1.5-7 chrF++ on short text and do not fix long text). 256-token windows (`winfix`) are an acceptable alternative when the sentence boundaries are unreliable: against the human references they are tied with sentences at >= 1,024 tokens (st1024 50.2 vs 50.6, st2048 52.4 vs 52.4), 1.0-1.3 chrF++ lower against NLLB, and AfriCOMET QE is about 0.015 lower; they have 2x the loop rate (b4096 loop 8 % vs 4 %). Window overlap by one sentence does not help (greedy_winov is within -0.4 ... +0.1 chrF++ of greedy_win everywhere and has 1-2 points more loops).

## 7. Per-language results for the best config (sentfix, all 16 languages; chrF++ vs the NLLB reference per bin, vs the human reference per human bin)

"finished" and "loop" are averaged over the six bins. For the language-level contrast, the same table for whole-document greedy is below.

| lang | refs | b64 | b256 | b512 | b1024 | b2048 | b4096 | macro core | h:sm | h:st512 | h:st1024 | h:st2048 | finished | loop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| amh | human+NLLB | 21.1 | 22.9 | 26.3 | 27.9 | 30.8 | 32.4 | 26.9 | 24.8 | 26.1 | 27.8 | 30.3 | 100% | 1% |
| hau | human+NLLB | 64.7 | 64.6 | 68.1 | 69.5 | 72.5 | 74.2 | 68.9 | 53.0 | 56.6 | 59.2 | 61.9 | 100% | 0% |
| ibo | human+NLLB | 63.4 | 62.3 | 66.4 | 68.2 | 71.0 | 72.4 | 67.3 | 56.0 | 59.5 | 61.2 | - | 100% | 0% |
| kam | NLLB only | 30.2 | 37.2 | 44.3 | 47.6 | 51.6 | 53.2 | 44.0 | - | - | - | - | 92% | 0% |
| kea | NLLB only | 32.3 | 35.6 | 40.7 | 43.0 | 45.7 | 46.8 | 40.7 | - | - | - | - | 100% | 9% |
| kin | human+NLLB | 48.8 | 51.3 | 56.1 | 58.3 | 61.8 | 63.6 | 56.7 | 50.9 | 53.6 | 57.5 | - | 100% | 0% |
| mlg | human+NLLB | 56.5 | 58.3 | 63.1 | 65.7 | 68.5 | 69.7 | 63.6 | 56.2 | 58.4 | 61.6 | - | 100% | 1% |
| nus | NLLB only | 30.7 | 32.8 | 37.2 | 39.0 | 40.7 | 41.9 | 37.0 | - | - | - | - | 70% | 6% |
| sag | NLLB only | 50.8 | 54.9 | 62.6 | 66.1 | 69.4 | 71.3 | 62.5 | - | - | - | - | 100% | 1% |
| som | human+NLLB | 55.4 | 58.2 | 63.1 | 64.2 | 67.0 | 69.0 | 62.8 | 55.5 | 58.0 | 59.3 | - | 100% | 0% |
| swh | human+NLLB | 68.6 | 66.9 | 70.7 | 71.5 | 73.8 | 75.1 | 71.1 | 66.9 | 67.6 | 69.3 | 71.1 | 100% | 1% |
| twi | human+NLLB | 49.2 | 51.4 | 57.0 | 59.1 | 62.7 | 64.5 | 57.3 | 43.8 | 45.6 | 50.4 | - | 100% | 0% |
| wol | human+NLLB | 37.3 | 39.8 | 44.9 | 47.7 | 50.3 | 52.8 | 45.5 | 38.6 | 41.6 | 44.7 | - | 100% | 0% |
| xho | human+NLLB | 53.5 | 56.6 | 59.6 | 62.6 | 64.4 | 66.7 | 60.6 | 57.5 | 59.2 | 62.0 | - | 100% | 0% |
| yor | human+NLLB | 46.5 | 47.3 | 50.9 | 54.1 | 57.0 | 59.3 | 52.5 | 41.7 | 42.4 | 46.6 | 46.4 | 84% | 0% |
| zul | human+NLLB | 59.5 | 60.3 | 63.7 | 65.3 | 67.7 | 69.2 | 64.3 | 58.1 | 60.3 | 56.4 | - | 100% | 0% |

Whole-document greedy for the same languages (the failure is language-independent; nothing finishes at b4096 for any language):

| lang | refs | b64 | b256 | b512 | b1024 | b2048 | b4096 | macro core | h:sm | h:st512 | h:st1024 | h:st2048 | finished | loop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| amh | human+NLLB | 20.9 | 22.1 | 23.0 | 14.0 | 8.5 | 6.0 | 15.7 | 24.0 | 23.8 | 13.0 | 8.4 | 73% | 35% |
| hau | human+NLLB | 64.2 | 63.8 | 65.3 | 49.3 | 25.4 | 14.4 | 47.1 | 52.2 | 55.3 | 35.7 | 25.4 | 72% | 29% |
| ibo | human+NLLB | 63.4 | 61.7 | 63.5 | 29.0 | 19.1 | 14.7 | 41.9 | 54.7 | 57.2 | 23.3 | - | 65% | 43% |
| kam | NLLB only | 29.8 | 35.2 | 35.1 | 21.6 | 16.6 | 8.2 | 24.4 | - | - | - | - | 50% | 49% |
| kea | NLLB only | 31.9 | 31.1 | 23.9 | 18.5 | 13.4 | 7.5 | 21.1 | - | - | - | - | 61% | 60% |
| kin | human+NLLB | 48.9 | 50.2 | 52.7 | 30.0 | 21.2 | 11.3 | 35.7 | 48.5 | 50.9 | 23.0 | - | 71% | 38% |
| mlg | human+NLLB | 56.5 | 57.6 | 60.2 | 41.2 | 25.4 | 13.3 | 42.4 | 54.6 | 56.1 | 36.0 | - | 83% | 32% |
| nus | NLLB only | 29.0 | 31.8 | 30.2 | 18.8 | 15.6 | 7.6 | 22.2 | - | - | - | - | 63% | 27% |
| sag | NLLB only | 50.5 | 53.2 | 56.4 | 29.6 | 21.1 | 6.8 | 36.3 | - | - | - | - | 66% | 38% |
| som | human+NLLB | 54.5 | 56.5 | 59.8 | 34.7 | 21.8 | 9.3 | 39.4 | 53.3 | 55.0 | 21.5 | - | 71% | 34% |
| swh | human+NLLB | 69.4 | 66.1 | 68.6 | 55.1 | 25.5 | 14.1 | 49.8 | 66.2 | 66.0 | 48.2 | 25.7 | 79% | 29% |
| twi | human+NLLB | 49.0 | 48.9 | 51.6 | 28.3 | 20.9 | 11.2 | 35.0 | 41.2 | 42.7 | 25.8 | - | 74% | 34% |
| wol | human+NLLB | 38.3 | 38.8 | 41.0 | 21.2 | 15.9 | 7.2 | 27.1 | 35.9 | 39.1 | 18.6 | - | 81% | 15% |
| xho | human+NLLB | 53.6 | 55.8 | 56.5 | 31.2 | 20.3 | 12.0 | 38.2 | 56.2 | 57.1 | 29.0 | - | 73% | 40% |
| yor | human+NLLB | 47.0 | 46.8 | 43.0 | 19.2 | 16.4 | 10.4 | 30.5 | 42.0 | 39.6 | 17.1 | 13.2 | 64% | 35% |
| zul | human+NLLB | 58.9 | 59.3 | 60.3 | 30.9 | 19.2 | 12.3 | 40.2 | 55.8 | 57.2 | 41.3 | - | 73% | 44% |

**Failing languages.** `amh` is low everywhere (26.9 macro vs NLLB, 24.8-30.3 vs human): Ge'ez-script outputs that agree little with NLLB or SmolDoc at the chrF++ level (AfriCOMET and the reward are more informative here); `nus` (37.0) and `kea` / `kam` (40.7 / 44.0) are the NLLB-only languages with the lowest scores, and `nus` leaves 30 % of the documents unfinished and `kam` 8 % after the repair. Of the 3,250 sentence units that did not finish in the greedy run, 1,708 were `nus`, 881 `kam`, 555 `yor`, 57 `kea`; 372 units stayed unfinished after the best-of-4 repair. `yor` has 16 % unfinished documents after repair, and `kea` has a 9 % loop rate. Everything else finishes 100 %.

**Failing bins.** Whole-document greedy and every whole-document variant from b1024; for the chunked runs the weakest bin by *reward* is b4096 (-1.03), but the NLLB reference scores the same there (-0.96), so this is the detector, not the model. No bin fails by coverage (100 % with chunking).

## 8. Yardsticks and honest pseudo-reference bias notes

**NLLB vs MADLAD vs the model** (the 5 languages where MADLAD is clean: mlg som swh xho zul; chrF++ macro over those languages, corpus level, per bin):

reference = NLLB-200 3.3B (5 languages: mlg som swh xho zul)

| system | b64 | b256 | b512 | b1024 | b2048 | b4096 |
|---|---|---|---|---|---|---|
| greedy | 58.6 | 59.1 | 61.1 | 38.6 | 22.4 | 12.2 |
| greedy_sent | 58.7 | 60.1 | 64.0 | 65.9 | 68.3 | 69.9 |
| sentfix | 58.7 | 60.1 | 64.0 | 65.9 | 68.3 | 69.9 |
| yard_nllb | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| yard_madlad | 53.9 | 56.4 | 60.4 | 62.7 | 65.3 | 67.1 |

reference = MADLAD-400 3B (same 5 languages)

| system | b64 | b256 | b512 | b1024 | b2048 | b4096 |
|---|---|---|---|---|---|---|
| greedy | 51.6 | 53.5 | 56.1 | 37.1 | 22.1 | 12.1 |
| greedy_sent | 51.7 | 54.2 | 58.1 | 60.6 | 63.6 | 65.3 |
| sentfix | 51.7 | 54.2 | 58.1 | 60.6 | 63.6 | 65.3 |
| yard_nllb | 53.4 | 56.3 | 60.0 | 62.1 | 64.8 | 66.7 |
| yard_madlad | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |

(`yard_*` rows = the pseudo-reference of that system used as hypothesis, so 100.0 against itself; `sentfix` equals `greedy_sent` here because none of the 5 languages has an unfinished unit.)

* The model (sentfix) agrees **more with NLLB (65.9 at b1024) than MADLAD does (62.7)**, while against MADLAD it is 1.5 below NLLB (60.6 vs 62.1). NLLB and MADLAD agree with each other about as much in both directions (62.1 / 62.7), so a reference that was independent of the model would give the model a similar score; here the NLLB reference scores the model about 3-5 chrF++ higher than the MADLAD reference does. Whether this comes from shared data lineage (SFT/RL data translated with NLLB-like systems) we did not test. All NLLB-referenced chrF++ numbers in this report are therefore **optimistic in absolute terms**; the *ordering of the strategies* (chunked >> whole-document above ~512 tokens; sentence >= window) is the same against MADLAD (5 languages), against the human references and under reference-free AfriCOMET-QE.
* **NLLB-referenced scores are higher than human-referenced scores for all 12 human languages** (sentfix, core macro vs the `sm` / `st512` human bins; documents differ, so this is indicative): hau 68.9 vs 53.0 / 56.6, ibo 67.3 vs 56.0 / 59.5, twi 57.3 vs 43.8 / 45.6, yor 52.5 vs 41.7 / 42.4, kin 56.7 vs 50.9 / 53.6, mlg 63.6 vs 56.2 / 58.4, som 62.8 vs 55.5 / 58.0, swh 71.1 vs 66.9 / 67.6, zul 64.3 vs 58.1 / 60.3, xho 60.6 vs 57.5 / 59.2, wol 45.5 vs 38.6 / 41.6, amh 26.9 vs 24.8 / 26.1. Quote the human-reference column for absolute quality (about 47 -> 52 chrF++ from short to ~2,048-token documents) and the NLLB column only for comparisons between configurations.
* **The sentence-vs-window gap is partly an artefact of the pseudo-reference.** NLLB references are made sentence by sentence, so a sentence-by-sentence hypothesis is aligned with it by construction. Against NLLB, sentences (sentfix) beat windows (winfix) by 1.0-1.3 chrF++ at b512-b4096; against human references the gap is +0.6 (sm), +0.4 (st512, st1024: 49.1 vs 48.7, 50.6 vs 50.2) and 0.0 at st2048. The reference-free AfriCOMET-QE still prefers sentences slightly (+0.014 ... +0.015 at b1024-b4096). So "sentence >= window" is safe; "sentence >> window" is not established.
* **Absolute chrF++ rises with the bin** (47.6 at b64 -> 60.1 at b4096 for the same pipeline): the bins are different texts (short opening passages vs long ones) and chrF++ is a corpus statistic; it is not evidence that the model translates long documents better. AfriCOMET (sentfix 0.691 -> 0.659) is the comparable-across-bins view: a mild decline, no collapse.
* The yardsticks are *systems* scored by the same detectors: NLLB and MADLAD sentence-by-sentence translations also get the "degenerate" status (all of the sag long documents, 25 of 36 wol, 19 of 36 nus ...) and low total rewards at b4096; treat -1 at b4096 as "indistinguishable from a good reference".
* Loops/degeneracy are measured with a word-cycle detector that ignores loops in the reference; short-word languages (sag, wol, kea) trip the reward's degeneration detector more often than the loop metric.
* SmolDoc human references are professional translations of short documents (<= ~680 tokens); the longer human documents are stitched from unrelated ones, so none of these results measure discourse quality (pronoun / terminology consistency across paragraphs); only length robustness and sentence-level accuracy.
* Bootstrap CIs resample documents only (35-100 per bin); the language set is fixed (8 / 12 / 16 languages), so the CIs do not cover language sampling. The bin `st2048` has 4 languages and 10 documents per language.

**SFT final under the same configurations** (B8, vs NLLB / human, from the tables above): with the sentence chunking the SFT model gets 44.3 / 45.7 / 51.1 / 53.3 / 56.3 / 57.8 chrF++ (b64 ... b4096), i.e. 3.3 / 3.3 / 2.4 / 2.3 / 2.1 / 2.3 below the final model with the same pipeline; against human references `sm` 44.4 vs 47.2, `st512` 47.0 vs 49.1, `st1024` 47.9 vs 50.6, `st2048` 50.1 vs 52.4. The SFT model does not stop reliably (finished 93 -> 13 % of the documents from b64 to b4096 even with sentence chunking, loop 6 -> 59 %), so its reward with sentence chunking is 0.30 / -0.11 / -0.88 / -1.12 / -1.39 / -1.65; whole-document SFT generation is dead from b256 (74 % finished) and at b512 / b1024 / b2048 / b4096 produces 32.5 / 17.0 / 11.1 / 6.9 chrF++.

## 9. Files on the box (`/workspace/jobs/a2/`)

`out/rl2c/*.jsonl.gz` (token ids per request), `hyp/<model>/<name>.jsonl` (assembled translations per document, e.g. `sentfix.jsonl`), `sc/<model>/<name>.pkl` (per-document metric rows), `w5/t_main.json` / `t_all.json` / `t_beam.json` / `t_nr.json` / `t_yard.json` (all table numbers including per-language), `w5/reward_{a,b,c}.jsonl`, `w5/comet_*.json`, scripts `gen_vllm.py`, `assemble.py`, `score_items.py`, `agg.py` (BLEU statistics bug fixed 2026-10-21: the BLEU columns of the earlier `res/test.json` were wrong), `w5/build.py`, `w5/sentfix.py`, `w5/winfix.py`, `hf_gen.py`. The large request files (`work/req_sent*`, `req_win*`) and the vLLM cache were removed after the run (regenerate with `prep.py`); the two model copies (`models/rl2c`, `models/sft`, 4.3 GB) can be deleted. No A2 process is running any more (GPU released).
