# E-14 delta for the LaTeX versions (2026-11-05)

Source of truth: `paper_385m/paper.md` after the E-14 pass. Each item gives the section of `paper.md`, the old text (or `NEW`), the new text, and the numbers with their source. Apply the same change to both LaTeX versions; table and figure numbering shifts by the added table (`t_e14`). Numbers come from `reports/e14_science_rl_without_science_sft_2026-11-05.md` (data `reports/data/e14_science_rl_no_sft_2026-11-05/`) unless another source is named. No new figure was made for E-14.

## A. E-14 changes

### A1. Abstract, science sentence
**Old:** `A science RL adds +0.46 chrF [+0.27, +0.65] on AfriScience test sentences over its start and a long-document RL improves ...`

**New:** `A science RL adds +0.46 chrF [+0.27, +0.65] on AfriScience test sentences over its start, and the science SFT, not the science RL, carries the science gain (the same RL started without the science SFT ends 1.44 chrF [1.25, 1.61] lower, and the registered rule says the science SFT is needed) and a long-document RL improves ...` (rest of the sentence unchanged).

Numbers: -1.44 [-1.61, -1.25] chrF (E-14 final minus science RL final, AfriScience test, 6 languages), written in the abstract as "1.44 chrF [1.25, 1.61] lower".

### A2. Section 7.8 (Science: SFT with weight interpolation, then RL), end of the science-RL paragraph
**Old:** `An RL-only science arm (mixed best to science RL without the science SFT) is in progress (Section 13), so the science SFT's own contribution to the science RL result is not separated.`

**New:** `The RL-only science arm (mixed best to science RL without the science SFT) is reported next (E-14).` followed by the new paragraph and tables below (inserted before the figure `fig5a_science_sentence`; the paragraph is placed after the sentence above).

**New paragraph and tables, verbatim from `paper.md`:**

~~~markdown
**Science RL without the science SFT (E-14).** To ask whether the science SFT is needed, the executed science RL was repeated with the start model changed to the mixed best (= science-SFT alpha 0, the model the science SFT was built from): same seed 1234, same pool, same reward, Muon at 1e-6, KL 0.1 against the start model, 160 steps (150 plus a 10-step wind-down), pre-declared checkpoint = final. The rule was registered before the run: the science SFT is *not needed* if the lower 95% bounds of E-14 minus the science RL final on AfriScience-MT test are above -0.5 chrF and above -0.004 held-out AfriCOMET; it is *needed* if the upper bound of either is below its margin. {{table:t_e14|AfriScience-MT TEST, sentence level, macro over 6 languages; E-14 minus X with 95% bootstrap (independent weights per language).}}

| model | chrF | chrF++ | AfriCOMET-STL (held out) | SSA-COMET-QE (reward family) |
|---|---:|---:|---:|---:|
| mixed best (= alpha 0; E-14's start) | 45.40 | 42.44 | 0.6828 | 0.6130 |
| B1 final (stage-B1 RL) | 45.45 | 42.50 | 0.6846 | 0.6141 |
| **E-14 final (science RL, no science SFT)** | **45.66** | 42.69 | **0.6850** | 0.6153 |
| science SFT alpha 0.5 | 46.63 | 43.80 | 0.6879 | 0.6118 |
| science RL final (from the science SFT) | 47.10 | 44.22 | 0.6932 | 0.6158 |

| E-14 final minus | chrF | AfriCOMET-STL (held out) | SSA-COMET-QE (reward family) |
|---|---|---|---|
| science RL final (from the science SFT) | **-1.44 [-1.61, -1.25]** | **-0.0082 [-0.0101, -0.0065]** | -0.0006 [-0.0018, +0.0007] |
| science SFT alpha 0.5 | -0.97 [-1.20, -0.75] | -0.0030 [-0.0050, -0.0009] | +0.0035 [+0.0022, +0.0047] |
| mixed best (its own start) | +0.26 [+0.12, +0.39] | +0.0022 [+0.0007, +0.0036] | +0.0023 [+0.0013, +0.0033] |
| B1 final | +0.21 [+0.07, +0.36] | +0.0004 [-0.0012, +0.0020] | +0.0012 [+0.0000, +0.0023] |

**By the registered rule the science SFT is needed:** the upper bounds are -1.25 chrF and -0.0065 AfriCOMET, both below their margins, chrF is lower in all six languages (-0.72 to -2.70) and the dev split agrees (-1.63 [-1.81, -1.44]). The science gain is carried by the science SFT (+1.23 chrF at alpha 0.5), not by the science RL: the RL alone on top of the mixed best adds +0.26 chrF [+0.12, +0.39] (+0.0022 AfriCOMET), a gain of the size at which seed noise matters (E-04), against +0.46 for the RL on top of the science SFT. The trainer's dev chrF++ stayed flat (43.50 -> 43.43) while the executed science RL rose from 44.62 to 45.13, and no `best/` checkpoint was written (the dev macro never rose by the trainer's 0.2 margin), so only the final exists. **Reward-family caveat:** SSA-COMET-QE rates E-14 level with the science RL final (-0.0006 [-0.0018, +0.0007]) and above alpha 0.5 (+0.0035), while chrF and held-out AfriCOMET rate it below both: the metric the RL optimises moved without a matching move in the independent metrics, so it is not evidence of quality here. On the general benchmarks (93 cells) E-14 is +0.13 chrF++ [+0.10, +0.15] over the mixed best (FLORES+ +0.17, NTREX +0.12, MAFAND 0.00) and +0.34 [+0.31, +0.36] over B1 final, with no regression on NTREX and MAFAND (+0.12 / 0.00 against the mixed best; the executed science RL final is higher on MAFAND by 0.36); at document level it does not change chunked chrF (57.30 against 57.29 for its start) and whole-document decoding still fails (finished 0.69, loop 0.82, chrF 25.6). Caveats: one seed; one crash (a CUDA out-of-memory in the COMET reward worker at about step 38; the launcher's ladder halved the micro-batch from 12,288 to 6,144 tokens and resumed from the step-20 checkpoint, replaying steps 21 to 38, which can differ from the first pass because vLLM sampling is not bit-reproducible); the comparison with the science RL final mixes the effect of the science SFT with its alpha 0.5 interpolation (a science SFT at alpha 1.0 followed by RL was not run); 160 steps only. <!-- src: reports/e14_science_rl_without_science_sft_2026-11-05.md sections 0-3; reports/data/e14_science_rl_no_sft_2026-11-05/ (ledger E14 rows SC7, SC8) -->
~~~

Key numbers (test, macro over 6 languages unless stated): E-14 final chrF 45.66 / chrF++ 42.69 / AfriCOMET 0.6850 / SSA-COMET-QE 0.6153; science RL final 47.10 / 44.22 / 0.6932 / 0.6158; science SFT alpha 0.5 46.63 / 43.80 / 0.6879 / 0.6118; mixed best 45.40 / 42.44 / 0.6828 / 0.6130; B1 final 45.45 / 42.50 / 0.6846 / 0.6141. Differences (E-14 final minus X, 95% bootstrap): vs science RL final chrF -1.44 [-1.61, -1.25], AfriCOMET -0.0082 [-0.0101, -0.0065], SSA-COMET-QE -0.0006 [-0.0018, +0.0007]; vs alpha 0.5 -0.97 [-1.20, -0.75], -0.0030 [-0.0050, -0.0009], +0.0035 [+0.0022, +0.0047]; vs mixed best +0.26 [+0.12, +0.39], +0.0022 [+0.0007, +0.0036], +0.0023 [+0.0013, +0.0033]; vs B1 final +0.21 [+0.07, +0.36], +0.0004 [-0.0012, +0.0020], +0.0012 [+0.0000, +0.0023]. Dev: -1.63 [-1.81, -1.44] chrF vs science RL final. Per language chrF vs science RL final: amh -1.21, hau -1.34, lug -1.11, nso -2.70, yor -0.72, zul -1.55 (all significant). Trainer dev chrF++ 43.50 -> 43.43 (executed science RL 44.62 -> 45.13). Benchmarks (93 cells) chrF++ +0.13 [+0.10, +0.15] over mixed best, +0.34 [+0.31, +0.36] over B1 final. Documents: chunked chrF 57.30 vs 57.29, whole chrF 25.6, finished 0.69, loop 0.82.

### A3. Section 9 (Stage chain summary), table
**NEW row**, inserted between the science RL row and the long-document RL row:

`| science RL without the science SFT (E-14) | mixed best | 160 (150 + 10 wind-down; one OOM restart from step 20; final only) | 42 x 16, guide 0.25, LR 1e-6, KL 0.1 | b64, b256 | about 5 h wall-clock (15:18 to 20:17 UTC; 14,395 s in the last process) | AfriScience test chrF 45.66 (start 45.40; science RL from the science SFT 47.10); 93-cell chrF++ +0.13 over the mixed best (not a FLORES+ stage) |`

### A4. Section 12.1 (Discussion), claim 5
**Old (end of claim 5):** `... nothing at document level (SSA-COMET-QE +0.004, chunked windows 0.574 to 0.577).`

**New:** the same sentence followed by: `E-14 shows the science SFT is needed: the same science RL started from the mixed best (no science SFT) ends -1.44 chrF [-1.61, -1.25] and -0.0082 AfriCOMET [-0.0101, -0.0065] below the executed science RL, and adds only +0.26 chrF [+0.12, +0.39] over its start, while the reward-family SSA-COMET-QE rates it level (-0.0006 [-0.0018, +0.0007]); one seed, one OOM restart, no best/ checkpoint.`

### A5. Section 13 (Limitations), item 15
**Old:** `15. **In progress, used by no claim of this paper:** an RL-only science arm (E-14). **Not run:** human evaluation ...; beam widths other than 4, length penalties and the 54-language decoding run.`

**New:** `15. **Not run:** human evaluation (needs native-speaker annotators); the packed-shard contamination scans, the content check of A0 against g0, the embedding and MT round-trip paraphrase scans and the source of the other 34 dev-train-half languages (need the data box); the NLLB-data ablation (owner skipped); beam widths other than 4, length penalties and the 54-language decoding run; a science SFT at alpha 1.0 followed by RL, a second seed of E-14 and a long-document RL from the E-14 model.`

### A6. Section 14.2 (RL prompts and reward-side data), science-stages bullet
**Insert** before `Science SFT: AfriScience-MT train (Apache-2.0; 28,750 windows) plus replay ...`: `The E-14 run (science RL started from the mixed best) used the same pool and reward as the executed science RL.`

### A7. Section 14.5 (Process disclosures)
(a) **Incident bullet, append:** `The E-14 run (science RL from the mixed best) had one CUDA out-of-memory in the COMET reward worker at about step 38; the launcher lowered the micro-batch from 12,288 to 6,144 tokens and resumed from the step-20 checkpoint, so steps 21 to 38 were replayed (same prompt schedule, vLLM sampling not bit-reproducible); metrics.jsonl holds two records for those steps and the later one is used.`

(b) **Hub release status bullet, experiment weights:** in the list of branches of `<HF_ORG>/gemma-0.4B-translate-rl-experiments` add `e14-science-rl-only-final`; add the sentence `The E-14 branch (science RL started from the mixed best) was trained with the reward-side data of the executed science RL (Section 14.2).` before `The E-03 control checkpoint is not in that repository or anywhere else public.` (the E-14 report is added to the source comment).

### A8. Section 15 (Reproducibility)
(a) **Weights and branches, experiment weights:** after `e04-d_noguide-step100` (E-11 arms, forks of seed 2's step-40 state), add `, e14-science-rl-only-final (E-14, step 160, no best/ exists)`.

(b) **Scripts/data list:** `reports/data/e11_*` becomes `reports/data/e11_*`, `reports/data/e14_*`.

## B. Open-item fixes made in the same pass (apply to the LaTeX too)

### B1. Section 5 (Stages paragraph), B1 final
**Old:** `B1's step-135 weights exist on the Hub but were never evaluated.`

**New:** `B1's step-135 weights (Hub branch stage-b1-final) were evaluated after the fact (E-20, Section 7.3) and equal B1 best within noise.`

**Section 14.5 (Start checkpoints bullet), old:** `B1 final exists on the Hub and is unevaluated.` **New:** `B1 final exists on the Hub and was evaluated after the fact (E-20): it equals B1 best within noise (+0.01 chrF++ [-0.02, +0.04] over 93 cells).`

**Section 15 (branch list), old:** `stage-b1-final (step 135, unevaluated)` **New:** `stage-b1-final (step 135; evaluated in E-20, equal to B1 best within noise)`.

Numbers: E-20, B1 final minus B1 best +0.01 chrF++ [-0.02, +0.04] over 93 cells (FLORES+ +0.04 [+0.00, +0.08], NTREX -0.00 [-0.05, +0.05], MAFAND -0.09 [-0.15, -0.01]); `reports/e20_b1_final_2026-11-04.md`.

### B2. fp32 / engine statements (Section 7.5, Section 13 item 9, Section 15)
**Section 7.5, old:** `... the earlier +0.2 to +1.8 range for the SFT is superseded.` **New:** `... the earlier +0.24 [-0.07, +0.55] (4,000 sentences) to +1.81 range for the SFT is superseded.`

**Section 13 item 9, old:** `... the earlier +0.2 to +1.8 range is superseded.` **New:** `... the earlier +0.24 to +1.81 range is superseded.`

**Section 15 (Hardware and software), old:** `a reader reproducing with another engine should expect differences of the order of 0.2 to 2 chrF++ (vLLM fp32 against bf16: +0.24 [-0.07, +0.55]).` **New:** `a reader reproducing with another engine should expect small, unsystematic differences (240 sentences, HF bf16 minus vLLM bf16: +0.54 [-0.67, +1.79] chrF++ for the SFT and -0.23 [-0.73, +0.23] for the RL model; the SFT in fp32 against bf16 on the full set: -0.04 [-0.13, +0.07] over 93 cells, E-09b, which supersedes the earlier +0.24 [-0.07, +0.55] on 4,000 sentences and +1.81 on 240 sentences).`

Numbers: E-09b -0.04 [-0.13, +0.07] (93 cells); E-09 parity bootstrap (`reports/data/e09_engine_parity_2026-11-04/parity_bootstrap.json`): HF minus vLLM bf16 SFT +0.54 [-0.67, +1.79], RL -0.23 [-0.73, +0.23]; ledger D6.

### B3. Eight versus nine added languages (Section 7.2 'Where RL does not help beyond finishing' and Section 7.4)
**Section 7.2, old:** `the eight languages added after A0 show -0.13 [-0.54, +0.27] on clean rows`

**New:** `the eight languages on FLORES+ that were added to the registry after A0 (aeb ary arz cjk dik mfe taq zgh; the RL evidence's group, which does not contain seh) show -0.13 [-0.54, +0.27] on clean rows`

**Section 7.4, old:** `Of the nine languages added in 2026-11, mfe (28.5) and seh (21.7) work,`

**New:** `Of the nine languages that the SFT report lists as added (the eight on FLORES+ that are among the 40 registry rows appended after A0, plus seh, which the registry counts among the first 85 rows), mfe (28.5) and seh (21.7) work,`

Source: `reports/data/rl_vs_sft_evidence_2026-11-04/groups_a0new.json` (seh = "A0 language (first 85)"; the group "new language (40)" has 8 FLORES+ languages); EVAL section 6 lists 9; ledger W2 count note.

## C. Not changed in this pass
No figure was added for E-14; `make_figures.py` is unchanged. The abstract's other numbers, the E-03, E-02, E-04 and E-11 text and all earlier tables are unchanged.
