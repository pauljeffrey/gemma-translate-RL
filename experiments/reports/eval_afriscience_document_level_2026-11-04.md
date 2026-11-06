# AfriScience-MT, FULL document-level evaluation (2026-11-04)

168 test documents = 28 papers x 6 languages (amh hau lug nso yor zul); a document = the paper's plain-language summary rebuilt as sentences joined by single spaces in `sentence_id` order (`scripts/eval_afriscience.load_corpus`, ~1,000 AfriXLM-R tokens, 600-1,500). Models: the 385M start (alpha 0, the mixed-stage best), the 385M after the science SFT (alpha 0.5), and **the first research's 270M RL-final (rl2c, step 1,433, `<HF_ORG_LEGACY>/gemma-3-270m-translate-rl2c@latest`) under identical settings on the same 168 documents**. The science RL models (best and final) are NOT in this report yet: the RL is still training (150-step cap, about 4 h at 2 minutes a step); they are evaluated with the same two commands right after it ends (below). Code: `gemma_mt/scripts/eval_afriscience_documents.py` (+ `_neural.py`), data and builders `reports/data/eval_afriscience_document_level_2026-11-04/`.

## Method
* **Mode `whole`** (the paper's document-level setting): the whole document in one prompt, greedy, 4,096 new tokens (the requested 1,024-1,536 would truncate Yoruba / Amharic documents that need about 2-3k Gemma tokens). **Mode `chunked`** (the first research's fix, D-50): every sentence alone, greedy, unfinished sentences re-decoded best-of-4 (T 0.7, consensus pick), sentences joined by single spaces.
* **Metrics.** The paper's statement (from the owner): *SSA-COMET (Li et al. 2025) is the primary metric, chrF (Popovic 2015) a surface-level secondary metric*; the paper evaluates sentence and document level. Headline here: **SSA-COMET**, then **plain chrF (word order 0)**; extra columns chrF++, BLEU, spBLEU, AfriCOMET-STL (`masakhane/africomet-stl-1.1`, reference-based, the RL held-out monitor) and SSA-COMET-MTL (reference-based, our RL reward model's checkpoint).
* **Which SSA-COMET variant?** The paper text and the released `metrics` config only say `ssa_comet`. The released *predictions* cover sentence-level runs only (no document-level predictions), so I calibrated on the sentence-level runs: 9 released English -> amh/hau/zul runs of GPT-5.4, Gemini-3.1-Flash-Lite and Gemma2-9B were re-scored by three variants and compared with the released value:

| released run | released SSA-COMET | ssa-comet-mtl, reference-based (src+mt+ref, mean of 3 views) | ssa-comet-mtl, reference-free (src+mt) | ssa-comet-qe, reference-free |
|---|---:|---:|---:|---:|
| gpt54 semantic_10shot eng-amh | 0.666 | 0.608 | 0.604 | 0.667 |
| gpt54 semantic_10shot eng-hau | 0.646 | 0.585 | 0.560 | 0.647 |
| gpt54 semantic_10shot eng-zul | 0.663 | 0.598 | 0.576 | 0.646 |
| gemini_31_flash_lite semantic_10shot eng-amh | 0.676 | 0.621 | 0.615 | 0.674 |
| gemini_31_flash_lite semantic_10shot eng-hau | 0.653 | 0.593 | 0.567 | 0.650 |
| gemini_31_flash_lite semantic_10shot eng-zul | 0.661 | 0.596 | 0.575 | 0.647 |
| gemma2_9b_it template1_0shot eng-amh | 0.323 | 0.152 | 0.171 | 0.404 |
| gemma2_9b_it template1_0shot eng-hau | 0.485 | 0.375 | 0.387 | 0.566 |
| gemma2_9b_it template1_0shot eng-zul | 0.366 | 0.194 | 0.209 | 0.446 |

Mean absolute difference to the released value (all 9 runs / the 6 strongest runs): ssa-comet-mtl, reference-based (src+mt+ref, mean of 3 views): 0.091 / 0.061; ssa-comet-mtl, reference-free (src+mt): 0.097 / 0.078; ssa-comet-qe, reference-free: 0.031 / 0.006

**The reference-free `McGill-NLP/ssa-comet-qe` reproduces the released values on the strongest systems (mean absolute difference 0.006 over the six GPT-5.4 / Gemini runs), the reference-based SSA-COMET-MTL does not (0.06; 0.09 over all nine).** So SSA-COMET-QE (source + translation, no reference) is the paper-matching variant for strong systems; it deviates by up to 0.08 on the weak Gemma2-9B runs (unexplained: possibly different treatment of degenerate outputs). I therefore use **SSA-COMET-QE as the headline** and report SSA-COMET-MTL (reference-based) as an extra column; this replaces the earlier plan (MTL reference-based as primary). Variant of the released checkpoint, batch size, and handling of empty outputs are the only things not verified.
* **Documents longer than the encoder window (read this before the numbers).** The COMET encoders read at most 512 tokens, so **a concatenated-document SSA-COMET (or AfriCOMET) only sees the first ~512 tokens of a ~1,000-token document: the `concat` numbers are NOT full-document scores** (the paper's document-level handling is not stated in the text available to me and its document-level predictions are not released, so it cannot be calibrated; its released document-level values, Table 4, are 0.38-0.52 and lie in the range of our `concat` values, consistent with truncation but not proof). **The honest document scores are the 4-sentence-window numbers**: for the sentence-aligned `chunked` outputs, windows of 4 consecutive sentences with their own source (and reference) are scored separately and averaged per document, so the whole document is judged. For the `whole` mode (unaligned output) no honest window score exists; its evidence is chrF, finished / loop rate, length ratio and tail coverage.

## Table 1. Document level, 168 test documents (28 papers x 6 languages), macro over the 6 languages

**Headline = SSA-COMET-QE on 4-sentence WINDOWS (the honest document score), then plain chrF.** The COMET encoders only read the first 512 tokens, so the *concatenated* document score (`concat`) sees only the first part of a ~1,000-token document and is NOT a full-document score; it is shown only because it is the literal 'metric on the concatenated document'. Windows need a sentence-aligned hypothesis, so they exist for the `chunked` mode only; for `whole` mode (unaligned output) only the truncated `concat` value exists and chrF / failure statistics are the real evidence.

| model | mode | **SSA-COMET-QE window** | chrF | AfriCOMET window | SSA-MTL(ref) window | SSA-QE concat (first 512 tokens) | AfriCOMET concat | SSA-MTL(ref) concat | chrF++ | BLEU | spBLEU | finished | loop | length ratio | tail coverage | concat COMET x100 - chrF |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 385M start (alpha 0, no science SFT) | whole | - | 22.8 | - | - | 0.378 | 0.470 | 0.214 | 20.0 | 4.0 | 5.4 | 0.71 | 0.86 | 1.03 | 0.27 | 15.0 |
| 385M start (alpha 0, no science SFT) | chunked | **0.574** | 57.3 | 0.687 | 0.448 | 0.436 | 0.661 | 0.219 | 53.0 | 20.0 | 25.8 | 1.00 | 0.14 | 0.99 | 0.61 | -13.7 |
| 385M science SFT (alpha 0.5) | whole | - | 24.9 | - | - | 0.381 | 0.495 | 0.208 | 22.1 | 5.2 | 6.5 | 0.51 | 0.86 | 1.30 | 0.30 | 13.2 |
| 385M science SFT (alpha 0.5) | chunked | **0.574** | 58.8 | 0.692 | 0.448 | 0.434 | 0.661 | 0.213 | 54.7 | 22.0 | 28.5 | 1.00 | 0.23 | 1.01 | 0.63 | -15.4 |
| old 270M RL-final (rl2c, first research) | whole | - | 34.9 | - | - | 0.420 | 0.599 | 0.268 | 30.9 | 6.9 | 10.2 | 0.70 | 0.61 | 1.10 | 0.41 | 7.2 |
| old 270M RL-final (rl2c, first research) | chunked | **0.593** | 59.3 | 0.703 | 0.468 | 0.457 | 0.682 | 0.244 | 55.0 | 22.5 | 29.3 | 1.00 | 0.11 | 1.00 | 0.64 | -13.6 |

`whole` = the document in one generation (greedy, 4,096 new tokens); `chunked` = sentence by sentence (greedy) + best-of-4 repair of unfinished sentences, sentences joined by single spaces. finished = share of documents (whole) or of documents with every sentence finished (chunked); loop = repetition detector (rl/guards); tail coverage = chrF recall of the last third of the reference against the hypothesis (does the output reach the end of the paper); the last column is the paper's degeneration flag computed with the truncated concat value (a COMET far above chrF means fluent but off-target output).

## Table 2. Chunked mode: sentence coverage and repairs

| model | sentences | unfinished before repair (repaired) | unfinished after | sentence coverage |
|---|---:|---:|---:|---:|
| 385M start (alpha 0, no science SFT) | 5,820 | 7 repaired | 0 | 1.000 |
| 385M science SFT (alpha 0.5) | 5,820 | 29 repaired | 0 | 1.000 |
| old 270M RL-final (rl2c, first research) | 5,820 | 0 repaired | 0 | 1.000 |

## Table 3. Per language (chunked mode): chrF / SSA-COMET-QE window

| model | amh | hau | lug | nso | yor | zul |
|---|---:|---:|---:|---:|---:|---:|
| 385M start (alpha 0, no science SFT) | 38.0 / 0.531 | 65.7 / 0.536 | 54.0 / 0.563 | 64.8 / 0.621 | 49.8 / 0.619 | 71.4 / 0.575 |
| 385M science SFT (alpha 0.5) | 39.3 / 0.538 | 67.4 / 0.538 | 55.6 / 0.565 | 67.3 / 0.618 | 50.6 / 0.607 | 72.7 / 0.577 |
| old 270M RL-final (rl2c, first research) | 34.9 / 0.497 | 69.1 / 0.571 | 60.6 / 0.625 | 67.8 / 0.633 | 49.5 / 0.638 | 73.7 / 0.592 |

## Table 4. The paper's released document-level results (English -> the same 6 languages, test, 28 documents per language, unweighted mean; dataset config `metrics`)

| system (model / experiment / prompt) | SSA-COMET | chrF | BLEU |
|---|---:|---:|---:|
| gpt54|doc_0shot_template2 | 0.519 | 64.5 | 26.5 |
| gemini_31_flash_lite|doc_0shot_template2 | 0.461 | 50.8 | 21.6 |
| gemini_31_flash_lite|doc_0shot_template1 | 0.457 | 50.9 | 22.1 |
| gpt54|doc_0shot | 0.454 | 50.0 | 19.7 |
| gemini_31_flash_lite|doc_0shot_template3 | 0.453 | 51.7 | 22.8 |
| gemini_31_flash_lite|doc_0shot | 0.452 | 47.1 | 19.4 |
| translategemma_12b|doc_1shot | 0.382 | 21.8 | 4.3 |

## Findings
1. **Whole-document generation does not work for the 385M models either, and is worse after the science SFT.** Start: finished 71%, loop 86%, chrF 22.8, tail coverage 0.27 (the last third of the paper is mostly not translated); science SFT: finished 51%, loop 86%, length ratio 1.30, chrF 24.9. The old 270M rl2c is less bad (finished 70%, loop 61%, chrF 34.9) but equally unusable. The degeneration flag (concat COMET x100 minus chrF) is positive (7-15) for every `whole` row and negative (-14 to -15) for every `chunked` row.
2. **Chunked decoding fixes it for all three models** (finished 100%, sentence coverage 1.000, loop rate 0.11-0.23, length ratio about 1.0): **window SSA-COMET-QE 0.574 / 0.574 / 0.593 (start / science SFT / old rl2c)**, chrF 57.3 / 58.8 / 59.3, BLEU 20.0 / 22.0 / 22.5, AfriCOMET window 0.687 / 0.692 / 0.703. The repair step is rarely needed (7 and 29 of 5,820 sentences for the 385M models, none for the old model).
3. **The first research's 270M RL-final is as good as or better than the 385M at document level** (chunked): +0.019 window SSA-COMET-QE, +0.011 AfriCOMET window, +0.5 to +2.0 chrF over the 385M models, better on hau, lug, nso, zul, worse on amh (34.9 vs 38-39 chrF). The science SFT adds +1.5 chrF and +2.0 BLEU over its start in chunked mode but **nothing in the headline metric (0.574 both)** and makes `whole` worse. Goodhart note: the old rl2c was also trained against a QE-based reward (AfriCOMET-QE; whether the SSA ensemble was already in its reward I did not check), so part of its QE advantage may be optimised; chrF and the reference-based AfriCOMET (which it leads too) are the independent checks.
4. **Against the paper (Table 4, English -> the same 6 languages, released metrics)**: the best released document-level system is GPT-5.4 with SSA-COMET 0.519 (chrF 64.5, BLEU 26.5); Gemini-3.1-Flash-Lite 0.461; TranslateGemma-12B 1-shot 0.382 (the 44.0 / 48.3 numbers quoted in the paper's abstract average more directions or configurations than these six English -> X rows). Our `concat` values (0.43-0.46, first 512 tokens) and the paper's values are not comparable in any strict sense (unverified document handling); our chrF 57-59 (chunked) is above the Gemini rows (47-56) and below GPT-5.4's best row (64.5).
5. **Windowed vs concatenated**: the window values (SSA-COMET-QE 0.57-0.59, AfriCOMET 0.69-0.70) are much higher than the concat values (0.43-0.46 and 0.66-0.68) because concat sees only the start of the document.

### Goodhart caution
The RL stages train against a QE reward that includes a model of the family of the headline metric. **In the RL reward:** `masakhane/africomet-qe-stl-1.1` (reference-free, primary) and `McGill-NLP/ssa-comet-mtl` (reference-free ensemble member; in the science RL also reference-based on the reference rows, plus a chrF++ reference term and reference-derived recall / precision / length / preservation terms). **In the evaluation, not in any reward:** `McGill-NLP/ssa-comet-qe` (headline; a different checkpoint of the same family as the reward's ssa-comet-mtl), `masakhane/africomet-stl-1.1` (reference-based, the RL's held-out monitor) and plain chrF / chrF++ / BLEU. SSA-COMET-QE gains obtained by RL models therefore overstate quality; the independent checks are plain chrF on held-out sentences/documents and the reference-based AfriCOMET-STL (SSA-COMET-MTL reference-based is in the science RL reward, so it is not independent for those models).

## Training context of the models (for reading the results)
The science RL trains on prompts of at most 256 source tokens (bins b64 and b256) only; the only long-context training the 385M models have had is the science SFT's whole-paper windows, which were packed in 4,096-token packs (79 whole-paper windows per language were cut; samples above 4,096 tokens were dropped by the packer, 18 samples in total). The SFT before it used packs of 4,096 tokens too. Whole-document generation of 1,000-token inputs is therefore outside the training distribution of the RL stages; production use should be sentence by sentence (the owner's standing rule), which is what the `chunked` mode measures.

## Commands for the remaining models (run when the science RL has ended, or at low share earlier)
```
/workspace/science_sft/run_doc_eval.sh /dev/shm/rl_sci_ckpt/best sci_rl_best single_token    # then .../final sci_rl_final
python scripts/eval_afriscience_documents_neural.py --pred ... --names ... --out /workspace/science_sft/doc_neural.json     # AfriCOMET + SSA-COMET-MTL
python /workspace/science_sft/ssa_qe_pass.py sci_rl_best sci_rl_final                                                      # SSA-COMET-QE
```
(then `build_doc_tables.py --models a0 a05 rl2c sci_rl_best sci_rl_final`; `ssa_qe_pass.py` must be run after the documents are generated and the neural pass before the tables).

## Caveats
One seed, greedy; 28 papers per language (the intervals of such a small set are wide: no bootstrap run); the SSA-COMET-QE variant is calibrated on sentence-level released runs only; the `whole` mode with a 4,096-token cap can truncate (finished rate reports it); the old model was run in text tag mode (its own format) with its own tokenizer.
