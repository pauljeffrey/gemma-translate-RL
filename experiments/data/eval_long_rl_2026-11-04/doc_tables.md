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
| 385M science RL (best) | whole | - | 26.7 | - | - | 0.388 | 0.519 | 0.216 | 23.8 | 5.7 | 7.9 | 0.62 | 0.85 | 1.19 | 0.31 | 12.1 |
| 385M science RL (best) | chunked | **0.574** | 58.7 | 0.692 | 0.450 | 0.437 | 0.661 | 0.216 | 54.5 | 21.9 | 28.3 | 1.00 | 0.18 | 1.00 | 0.63 | -14.9 |
| 385M science RL (final) | whole | - | 27.4 | - | - | 0.398 | 0.531 | 0.228 | 24.4 | 5.8 | 8.1 | 0.70 | 0.85 | 1.12 | 0.32 | 12.4 |
| 385M science RL (final) | chunked | **0.577** | 58.7 | 0.696 | 0.455 | 0.440 | 0.669 | 0.217 | 54.6 | 21.9 | 28.3 | 1.00 | 0.20 | 1.00 | 0.63 | -14.8 |
| 385M long-document RL (best, step 50) | whole | - | 30.9 | - | - | 0.403 | 0.564 | 0.231 | 27.8 | 7.3 | 9.9 | 0.82 | 0.73 | 0.96 | 0.36 | 9.5 |
| 385M long-document RL (best, step 50) | chunked | **0.579** | 58.9 | 0.698 | 0.458 | 0.442 | 0.675 | 0.220 | 54.7 | 22.0 | 28.3 | 1.00 | 0.17 | 0.99 | 0.63 | -14.7 |
| 385M long-document RL (final, step 80) | whole | - | 29.7 | - | - | 0.403 | 0.552 | 0.225 | 26.7 | 6.9 | 9.2 | 0.85 | 0.72 | 0.81 | 0.34 | 10.6 |
| 385M long-document RL (final, step 80) | chunked | **0.579** | 58.8 | 0.698 | 0.458 | 0.441 | 0.676 | 0.220 | 54.7 | 22.0 | 28.3 | 1.00 | 0.13 | 0.99 | 0.63 | -14.7 |

`whole` = the document in one generation (greedy, 4,096 new tokens); `chunked` = sentence by sentence (greedy) + best-of-4 repair of unfinished sentences, sentences joined by single spaces. finished = share of documents (whole) or of documents with every sentence finished (chunked); loop = repetition detector (rl/guards); tail coverage = chrF recall of the last third of the reference against the hypothesis (does the output reach the end of the paper); the last column is the paper's degeneration flag computed with the truncated concat value (a COMET far above chrF means fluent but off-target output).

## Table 2. Chunked mode: sentence coverage and repairs

| model | sentences | unfinished before repair (repaired) | unfinished after | sentence coverage |
|---|---:|---:|---:|---:|
| 385M start (alpha 0, no science SFT) | 5,820 | 7 repaired | 0 | 1.000 |
| 385M science SFT (alpha 0.5) | 5,820 | 29 repaired | 0 | 1.000 |
| old 270M RL-final (rl2c, first research) | 5,820 | 0 repaired | 0 | 1.000 |
| 385M science RL (best) | 5,820 | 12 repaired | 0 | 1.000 |
| 385M science RL (final) | 5,820 | 6 repaired | 0 | 1.000 |
| 385M long-document RL (best, step 50) | 5,820 | 1 repaired | 0 | 1.000 |
| 385M long-document RL (final, step 80) | 5,820 | 0 repaired | 0 | 1.000 |

## Table 3. Per language (chunked mode): chrF / SSA-COMET-QE window

| model | amh | hau | lug | nso | yor | zul |
|---|---:|---:|---:|---:|---:|---:|
| 385M start (alpha 0, no science SFT) | 38.0 / 0.531 | 65.7 / 0.536 | 54.0 / 0.563 | 64.8 / 0.621 | 49.8 / 0.619 | 71.4 / 0.575 |
| 385M science SFT (alpha 0.5) | 39.3 / 0.538 | 67.4 / 0.538 | 55.6 / 0.565 | 67.3 / 0.618 | 50.6 / 0.607 | 72.7 / 0.577 |
| old 270M RL-final (rl2c, first research) | 34.9 / 0.497 | 69.1 / 0.571 | 60.6 / 0.625 | 67.8 / 0.633 | 49.5 / 0.638 | 73.7 / 0.592 |
| 385M science RL (best) | 39.1 / 0.536 | 67.1 / 0.537 | 55.4 / 0.564 | 67.2 / 0.620 | 50.6 / 0.612 | 72.7 / 0.577 |
| 385M science RL (final) | 39.3 / 0.539 | 67.3 / 0.539 | 55.5 / 0.571 | 67.0 / 0.621 | 50.6 / 0.615 | 72.7 / 0.578 |
| 385M long-document RL (best, step 50) | 39.4 / 0.541 | 67.6 / 0.541 | 55.7 / 0.572 | 67.2 / 0.626 | 50.6 / 0.617 | 72.7 / 0.577 |
| 385M long-document RL (final, step 80) | 39.4 / 0.542 | 67.5 / 0.539 | 55.7 / 0.575 | 67.2 / 0.624 | 50.6 / 0.616 | 72.7 / 0.578 |

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
