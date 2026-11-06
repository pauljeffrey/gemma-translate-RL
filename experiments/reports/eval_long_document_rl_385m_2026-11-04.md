# Long-document RL (385M): evaluation of best and final (2026-11-04)

Stage: `runs_385m_long/final`, started from the science RL `final/` (step 160; chosen over `best/` on the whole AfriScience-MT dev set, see `eval_science_rl_385m_2026-11-04.md`), 80 steps, 32 prompts per step of which ~51 % long (b512 / b1024 / b2048), group 8, no guide, gradient checkpointing, hold-out-reward checkpoint selection, pool `long_rl_filtered` (decontaminated against the protected index including AfriScience-MT dev/test and AfriLongDoc; LibreTexts rows removed), reward = QE ensemble + chunked long-document path (`longdoc_v2`). Plan, data and the crash-loop incident: `rl_long_document_plan_2026-11-04.md` (sections 3, 7, 8). Evaluation: `post_long_rl_eval.sh` (best/ = step 50, final/ = step 80): AfriScience-MT dev/test sentences, NTREX, MAFAND, the 168 test documents whole and chunked, neural passes; plus the paired comparisons `sci_pairs_long.py` (sentences) and `doc_pairs.py` (documents). Data and scripts: `reports/data/eval_long_rl_2026-11-04/`.

## 0. Short answer

* **The long-document RL improves whole-document generation modestly and leaves everything else where it was.** Whole mode (the paper's document is generated in one pass, greedy, 4,096 new tokens), 168 test documents, macro over 6 languages, long RL best (step 50) against its start (science RL final): finished **0.70 -> 0.82 (+0.125 [+0.046, +0.204])**, loop rate **0.85 -> 0.73 (-0.113 [-0.179, -0.056])**, chrF **27.4 -> 30.9 (+3.4 [+1.5, +5.3])**, tail coverage 0.32 -> 0.36. The final checkpoint (step 80) is similar (finished 0.85, loop 0.72, chrF 29.7, +2.2 [+0.4, +4.2]); best and final do not differ significantly (chrF -1.2 [-3.0, +0.7]).
* **Whole-document generation is still not usable.** Even after the stage 15-18 % of the documents do not finish, 72-73 % show a repetition loop, the translated text covers only 34-36 % of the last third of the paper (chunked: 63 %), and the final checkpoint under-generates (length ratio 0.81). Chunked decoding (sentence by sentence + repair) remains the way to translate a paper with this model.
* **Chunked mode and sentence level: no change, no regression.** Chunked window SSA-COMET-QE 0.577 (science RL final) -> **0.579** (best) / 0.579 (final); chrF 58.7 -> 58.9 / 58.8 (best minus start +0.13 [+0.04, +0.22], final +0.09 [-0.01, +0.19]). AfriScience-MT TEST sentences: chrF 47.10 -> 47.16 / 47.16 (+0.06 [-0.09, +0.20] / +0.06 [-0.10, +0.22]), held-out AfriCOMET +0.0001 / +0.0004 (intervals around zero), SSA-COMET-QE -0.0000 / +0.0007. NTREX / MAFAND chrF++ 37.33 / 41.14 (start) -> 37.43 / 41.15 (best), 37.38 / 41.16 (final).
* **Against the old 270M** (first research, rl2c, chunked): window SSA-COMET-QE 0.593, still ahead of every 385M model (0.579 at best); whole mode chrF 34.9 against our 30.9.
* **Costs and caveats:** the stage took 7.7 h of wall-clock time on the GPU (10:03-17:44 UTC), of which about 4.5 h were a crash loop (disk, `rl_long_document_plan_2026-11-04.md` section 8), so about 3.2 h were productive; one seed; the hold-out reward that selected `best/` is the optimised quantity, and the SSA-COMET-QE of the windows shares its encoder family with the RL reward (the document finished / loop / chrF numbers are the independent ones).

## 1. Training record

Hold-out evaluation (greedy, all five bins, held-out prompts): reward -0.353 (step 25) / **-0.289 (step 50 = `best/`)** / -0.301 (75) / -0.358 (80 = `final/`); finished 0.963 / 0.963 / 1.000 / 0.993; greedy loop rate 0.134 / 0.119 / 0.104 / 0.112. Sentence-level monitor (AfriScience dev subset): chrF++ 46.16-46.28 against 46.30 at step 0; held-out AfriCOMET 0.7209 (step 50), 0.7189 (step 80). KL guard: 0 cuts; hacking alarm: never. The hold-out loop rate fell by only 2 points over 80 steps: the sentence-level reward gives little signal for whole-paper loops, since 49 % of the prompts are short.

## 2. Documents, 168 test documents (28 papers x 6 languages)

| model | mode | **SSA-COMET-QE window** | chrF | AfriCOMET window | SSA-MTL(ref) window | SSA-QE concat (first 512 tokens) | AfriCOMET concat | SSA-MTL(ref) concat | chrF++ | BLEU | spBLEU | finished | loop | length ratio | tail coverage | concat COMET x100 - chrF |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 385M science SFT (alpha 0.5) | whole | - | 24.9 | - | - | 0.381 | 0.495 | 0.208 | 22.1 | 5.2 | 6.5 | 0.51 | 0.86 | 1.30 | 0.30 | 13.2 |
| 385M science SFT (alpha 0.5) | chunked | **0.574** | 58.8 | 0.692 | 0.448 | 0.434 | 0.661 | 0.213 | 54.7 | 22.0 | 28.5 | 1.00 | 0.23 | 1.01 | 0.63 | -15.4 |
| old 270M RL-final (rl2c, first research) | whole | - | 34.9 | - | - | 0.420 | 0.599 | 0.268 | 30.9 | 6.9 | 10.2 | 0.70 | 0.61 | 1.10 | 0.41 | 7.2 |
| old 270M RL-final (rl2c, first research) | chunked | **0.593** | 59.3 | 0.703 | 0.468 | 0.457 | 0.682 | 0.244 | 55.0 | 22.5 | 29.3 | 1.00 | 0.11 | 1.00 | 0.64 | -13.6 |
| 385M science RL (final) | whole | - | 27.4 | - | - | 0.398 | 0.531 | 0.228 | 24.4 | 5.8 | 8.1 | 0.70 | 0.85 | 1.12 | 0.32 | 12.4 |
| 385M science RL (final) | chunked | **0.577** | 58.7 | 0.696 | 0.455 | 0.440 | 0.669 | 0.217 | 54.6 | 21.9 | 28.3 | 1.00 | 0.20 | 1.00 | 0.63 | -14.8 |
| 385M long-document RL (best, step 50) | whole | - | 30.9 | - | - | 0.403 | 0.564 | 0.231 | 27.8 | 7.3 | 9.9 | 0.82 | 0.73 | 0.96 | 0.36 | 9.5 |
| 385M long-document RL (best, step 50) | chunked | **0.579** | 58.9 | 0.698 | 0.458 | 0.442 | 0.675 | 0.220 | 54.7 | 22.0 | 28.3 | 1.00 | 0.17 | 0.99 | 0.63 | -14.7 |
| 385M long-document RL (final, step 80) | whole | - | 29.7 | - | - | 0.403 | 0.552 | 0.225 | 26.7 | 6.9 | 9.2 | 0.85 | 0.72 | 0.81 | 0.34 | 10.6 |
| 385M long-document RL (final, step 80) | chunked | **0.579** | 58.8 | 0.698 | 0.458 | 0.441 | 0.676 | 0.220 | 54.7 | 22.0 | 28.3 | 1.00 | 0.13 | 0.99 | 0.63 | -14.7 |

Paired differences (resampling documents, Poisson weights, 1000 resamples, macro over the 6 languages; 95 % CI): whole-mode chrF, finished share, loop share; chunked chrF.

| comparison | whole chrF | whole finished | whole loop | chunked chrF |
|---|---|---|---|---|
| long RL best (step 50) minus science RL final (start) | 27.4 -> 30.9 (+3.4 [+1.5, +5.3]) | 0.696 -> 0.821 (+0.125 [+0.046, +0.204]) | 0.845 -> 0.732 (-0.113 [-0.179, -0.056]) | 58.74 -> 58.87 (+0.13 [+0.04, +0.22]) |
| long RL final (step 80) minus science RL final (start) | 27.4 -> 29.7 (+2.2 [+0.4, +4.2]) | 0.696 -> 0.851 (+0.155 [+0.069, +0.242]) | 0.845 -> 0.720 (-0.125 [-0.192, -0.064]) | 58.74 -> 58.83 (+0.09 [-0.01, +0.19]) |
| long RL final minus long RL best | 30.9 -> 29.7 (-1.2 [-3.0, +0.7]) | 0.821 -> 0.851 (+0.030 [-0.037, +0.103]) | 0.732 -> 0.720 (-0.012 [-0.078, +0.063]) | 58.87 -> 58.83 (-0.04 [-0.13, +0.06]) |

The window SSA-COMET-QE is a corpus-level number per model (no paired interval computed); its 0.002 difference should not be read as an effect. SSA-COMET-QE and the RL reward share an encoder family: the independent document numbers are finished, loop, chrF and tail coverage.

## 3. Sentence level and non-regression

| AfriScience-MT TEST (macro, 6 languages) | science RL final (start) | long RL best | long RL final |
|---|---:|---:|---:|
| chrF | 47.10 | 47.16 (+0.06 [-0.09, +0.20]) | 47.16 (+0.06 [-0.10, +0.22]) |
| AfriCOMET-STL (held out) | 0.6932 | 0.6933 (+0.0001 [-0.0013, +0.0016]) | 0.6936 (+0.0004 [-0.0011, +0.0019]) |
| SSA-COMET-QE (reward family) | 0.6158 | 0.6158 (-0.0000 [-0.0011, +0.0011]) | 0.6165 (+0.0007 [-0.0004, +0.0017]) |

On DEV the differences against the start are also within noise (chrF -0.04 [-0.18, +0.10] best, +0.09 [-0.06, +0.24] final). NTREX / MAFAND chrF++ (point estimates): 37.33 / 41.14 start; 37.43 / 41.15 best; 37.38 / 41.16 final.

## 4. What this shows

* The stage does what its design aimed at, in whole mode, by a clear but moderate margin (loop -11 points, finished +12 to +15 points, chrF +2 to +3.4), and costs nothing at sentence level.
* It is not enough: with only ~1,300 long prompts in 80 steps, group 8 and a reward that averages chunks, the loops of whole-paper generation (72 %) are barely touched. The effect is of the size of the other late-stage steps (B1, mixed, science RL), i.e. real but small.
* One seed; the document evaluation has 168 documents (28 papers), so the intervals reflect paper sampling only; best/ vs final/ are indistinguishable.
