# Proposal: is the Ge'ez-script / Fon deficit a capacity problem? (draft 2026-10-25, not run)

Owner idea (2026-10-25): train a specialist model for Ge'ez-script languages (and Fon) and a larger model to test whether sharing 270M parameters across 77 languages ("capacity dilution") explains the gap to NLLB; do something similar for long documents.

## What the stored results say (all from files)
- FLORES+ devtest `no_pair`, chrF++: amh RL-final 19.0 vs NLLB-3.3B 38.0 / NLLB-600M 37.3; tir 13.8 vs 24.7 / 22.4. Excluding amh and tir the 42-language macro gap to NLLB-3.3B is 3.32 chrF++ (35.10 vs 38.42), with both included 3.84 (computed from `reports/data/model_comparison_master.csv`). So Ge'ez accounts for about 0.5 chrF++ of the macro gap but it is the largest single-language gap.
- Judge (Gemini-3.8-Flash, long documents, wrapper mode): amh 37.4 vs NLLB-3.3B 87.7 (-50.3); yor -19.5 and kin -25.8 are also large; hau -5.7, swh -6.5.
- Whole-document failure is not a Ge'ez effect (final model whole-document judge scores amh 7.7, yor 9.0, ibo 25.9, kin 14.6, hau 43.0, swh 43.6).
- Causes of the Ge'ez gap are NOT established. Capacity is only one of at least four candidates: (a) tokenisation (how many tokens a Ge'ez word costs in the Gemma tokenizer; byte fallback), (b) the amount and quality of amh/tir training data in our SFT mixture, (c) pretraining coverage of Gemma-3-270M, (d) capacity sharing. NLLB is also larger, trained on far more data and many more language pairs, so a comparison with NLLB cannot isolate (d).

## Step 0 (no GPU, hours of CPU/laptop work; do before any training)
1. Tokenizer fertility: tokens per word and per character for amh, tir, fon and a Latin-script control set (hau, yor, swh, kin) in the Gemma-3 tokenizer, and the share of byte-fallback tokens. If amh costs several times more tokens per word than hau, sequence length and effective data per step explain part of the gap.
2. Data: pairs and tokens per language in each SFT stage for amh, tir, fon vs controls (from `reports/weak_language_data_2026-10-22.md` and the SFT manifests); quality filters' pass rate.
3. Learning curves: per-language chrF++/loss across the stored SFT checkpoints and RL steps; if the amh/tir curves are still rising, the cause is more likely data or steps than capacity.

## Step 1: training arms (SFT, same recipe and commercially-clean data as the production run)
Target languages: amh and tir (Ge'ez), fon (and one Latin-script control pair, for example hau and yor, to see whether a specialist helps everywhere).
- A0: multilingual SFT-final (existing).
- A1: Ge'ez specialist: Gemma-3-270M-it trained only on en->amh and en->tir data, same number of target-language tokens as A0 saw, same optimiser and schedule. Same for a Fon specialist.
- A2: multilingual 270M with the target languages upweighted so that it sees as many amh/tir tokens as A1 (separates specialisation from sampling frequency; without this arm a specialist win says nothing about capacity).
- A3: Gemma-3-1B-it (about 4 times the parameters), same recipe and mixture as A0 (tests size directly; the effect for amh/tir should be larger than for the Latin-script controls if capacity matters most there).
- Optional A4: 1B Ge'ez specialist.
Held-out evaluation: FLORES+ devtest `no_pair` and `clean` subsets (our SFT data contains most FLORES+ sentences, so never use `all`), NTREX/MAFAND for amh where available; chrF++, spBLEU, AfriCOMET and SSA-COMET (amh only; tir and fon have no learned-metric support), and the Flash judge on the 200-sentence sample for amh (validated language). Paired bootstrap over sentences. Pre-register the decision rule and hash it before training, as in AB2.

## Decision rule (sketch, to be finalised before training)
- Capacity/interference is supported if A1 beats A0 AND A2 by more than the pre-set margin (for example 2 chrF++ with the CI excluding 0) on amh/tir, and A3 improves amh/tir by clearly more than the Latin-script controls.
- Data/steps explanation is supported if A2 closes most of the gap to A1. Tokenisation is supported if Step 0 shows a large fertility gap and A3 (same tokenizer) does not close the gap.
- Nothing is claimed about NLLB beyond "gap to NLLB-3.3B/600M under matched conditions of our own family".

## Long documents
A second question is whether whole-document quality is capacity-limited. Reduced test after Step 1: apply the same short long-document SFT stage (A2W-like, fixed number of tokens) to the 270M and the 1B base from the same multilingual SFT stage, and score on the Run 3 sample (judge, finish, omission, loops) plus the LC1 benchmark. This is more expensive and depends on Step 1 results; estimate cost after Step 0.

## Cost (estimates from AB2 on the RTX PRO 6000 at USD 1.19/h: a 367-step SFT arm took about 20 minutes)
270M specialist/upweighted arms: about 1-3 GPU-hours each (USD 1-4); 1B arms about 3-4 times that; evaluation about 10 minutes per arm. Order of magnitude USD 30-80 for Step 1 including evaluation. These are estimates, not a quote; no GPU instance is available to the lead agent today.

## Risks
Tokenizer limitation can cap any model that uses the same vocabulary; specialists risk memorising FLORES-like text; only two Ge'ez languages and one Fon variety, so conclusions do not generalise to all scripts; Fon has no validated learned metric or judge language, so Fon results rest on chrF++ and spBLEU only; a 1B result does not isolate capacity from better pretraining of the 1B model.

## Addendum 2026-10-25 (owner remarks): keep everything fixed; precision; model size
1. **One factor at a time.** A vocabulary swap changes the tokenizer, so it is itself an experimental factor and must be isolated before (or crossed with) the capacity question. Fixed across all arms: the base checkpoint, the SFT examples and their order (same examples and same number of steps; token counts differ because the swapped tokenizer makes Ge'ez sequences shorter, report both), optimiser, learning-rate schedule, batch size, seed, precision, evaluation sets and decoding settings. Factors: tokenizer {original, swapped} x training set {multilingual, Ge'ez specialist} x model {270M, 1B}. Add a control for the embedding re-training itself: original tokenizer, but the same number of embedding rows re-initialised and re-trained, so that 'new tokens' is not confused with 'embeddings were trained longer'. Latin-script text tokenises identically under the swap (only Ethiopic strings get new merges), which the Latin controls (hau, yor, swh, kin, ibo) must confirm. Pre-register the rule and hash it before training (as in AB2).
2. **Numeric precision is a factor too.** If FP8 is used for the 1B, use the same precision in every arm that is compared with each other; do not compare a bf16 270M arm with an FP8 1B arm as if only size changed. Validate FP8 against bf16 once on the 270M (same data, same steps: loss curve and FLORES+ no_pair chrF++) before adopting it.
3. **Model size.** The owner rules out 4B (loading and inference cost); the candidate is 1B (internal use). Whether the 1B's pretraining covers African languages is being checked by S0 (reports/ge_ez_step0_2026-10-25.md).
