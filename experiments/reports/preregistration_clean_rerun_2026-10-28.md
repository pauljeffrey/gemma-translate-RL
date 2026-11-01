# Pre-registration: clean rerun and the Ge'ez / weak-language specialist (2026-10-28)

Frozen rule file: `reports/data/prereg_clean_2026-10-28/DECISION_RULE.txt`, **sha256 `7b86fc11de28cf952bbc8f7a5f634b5fceda6adbeb76996548360c8ed923c0f5`**, written before any training of the rerun. Owner approved the three arms and the 2 chrF++ threshold on 2026-10-28.

## Why
The 2026-10-26 audit (`reports/benchmark_contamination_audit_2026-10-26.md`) found test text in the old SFT text, RL prompts and reward calibration. The rerun keeps the old pipeline (SFT, multi-phase RL on length bins up to 1,024 tokens, AfriScience stage, all reward components) and changes only what enters each role (T1 training, T2 calibration, S selection, E reported).

**Amendment 1 (2026-10-28, before any training):** Yoruba added to the specialist set (W now 18 languages, specialist set 20), control set = hau swh kin ibo. File `reports/data/prereg_clean_2026-10-28/AMENDMENT_1.txt`, sha256 `a5885cf2b6218411af3807b763b5b8aeb7003e8c9c759cf3425ae05cf533c4c2`. The original rule file is unchanged. Wherever the tables below list 19 languages or `yor` as a control, read the amendment.

**Amendment 2 (2026-10-28, before any training):** the 1,024-token cap of the RL bins is on the SOURCE (English) side, as in the previous run, not the target side. File `reports/data/prereg_clean_2026-10-28/AMENDMENT_2.txt`, sha256 `955941c5f6699511cc76a73cf77e2b3ca715c7cd4521de0635a4556cd3a92caa`.

**Amendment 3 (2026-10-28, before any training or fit):** the H_cal / H_sel split is made at article level (FLORES+ `url`) instead of by sha256 bit: H_cal 261 ids, H_sel 296 ids, all 100 weak-selection ids in H_sel. File `reports/data/prereg_clean_2026-10-28/AMENDMENT_3.txt`, sha256 `d6d5ecb28dc72eac85ddb5cba0d2d6b0e178c8efa3f7b2de81602fc1e665b853`. Numbers 282/275 in earlier text are superseded.

**Amendment 4 (2026-10-29, before any training):** commercial-use-only data rule (non_commercial rows removed everywhere), new arms G0 (Ge'ez-only SFT), A2-base/A2-up (+6-layer upcycling ablation on a 20% stratified subset), A0-R/A0-C (RL versus compute-matched SFT), exploratory P1, X1 last, extra S1 seeds dropped, Muon optimizer, patience-based RL early stopping on every bin, vLLM for all evaluation generation, private <HF_ORG> repositories. File `reports/data/prereg_clean_2026-10-28/AMENDMENT_4.txt`, sha256 `85e20a35c613627e021d70504fda7375f92f6f640d7c0b6d0b43e61c9ff7a1c2`.

**Clarification of Amendment 4 section 3 (2026-10-29, lead):** the lr of 5e-5 applies to the SFT arms only. RL stages keep lr 2e-6 and the final run 1e-6 (Muon with match_rms_adamw, so the AdamW learning rate carries over). The frozen amendment file is not edited.

**Amendment 5 (2026-10-29, before any training):** IbomMT kept as a disclosed non-commercial exception; hidden-NC filter; statistics-only tables unrestricted; AfriScience-MT and AfriCOMET-QE accepted; NLLB pseudo-references in RL only; A2 ablation redesigned (A2-base, A2-up6, A2-up12, A2-1b on a shared 25% stratified subset, new layers local attention); longer RL patience on cheap bins; 0.58 USD/h. File `reports/data/prereg_clean_2026-10-28/AMENDMENT_5.txt`, sha256 `b8d5a012123c4b3d2575daed172025b841f694571a4c50cc4c2c79894a4d1100`.

**Amendment 6 (2026-10-30, before any training):** four new SFT languages (nde, ful, knc, tzm) with sources from the data hunt; Fante and Ga skipped; AfroLingu-MT kept as a disclosed exception; NLLB-mined Fulah and KanuriMT allowed after the strict filter. File `reports/data/prereg_clean_2026-10-28/AMENDMENT_6.txt`, sha256 `2b1ba75163c73aab3455c4439019cccf51f12134bca8c4a8d06ddb481d1eb1b3`.

## The three arms
| arm | languages | tokenizer | purpose |
|---|---|---|---|
| A0 | 77 | original | clean multilingual baseline (needed for the paper anyway) |
| S1 | amh tir + fon dyu kbp nus mos kam bam umb kea kmb vmw kab seh wol ewe luo tum | original | specialisation at matched passes per language (R1, R2) |
| S2 | same as S1 | original + 3,000 Ge'ez tokens | tokenizer effect and the production candidate (R3) |

Decision rule: a difference of at least 2.0 chrF++ with a 95% paired-bootstrap interval excluding 0 on FLORES+ devtest (group means over G = amh, tir and W = the 17 weak Latin languages). Everything is reported whatever it shows.

## What the design can and cannot say
* S1 vs A0 measures the gain from specialising at the same number of passes over each language. A gain is consistent with parameter interference or limited capacity, but the arm cannot separate those from the effect of fewer total tokens and steps. The rule file fixes the wording.
* S2 vs S1 isolates the tokenizer (everything else fixed; Latin tokenisation identical, checked by script).
* Capacity itself needs a size change. The optional exploratory arm X1 (S1 recipe on Gemma-3-1B-it, SFT only, specialist data only) gives a scaling data point; because the 1B has a different pretraining it is not proof of capacity dilution. It is cheap because the specialist data is small.
* One seed per arm (budget): the bootstrap covers test sentences, not training-seed variance. Two extra S1 seeds are run if budget allows.

## Roles and hold-outs
Protected index `protected-2026-10-28.2` (agent D2). FLORES+ dev is split once for all languages by id parity: the train half is usable only for the specialist-set languages; the held-out half H (containing the 100 ids that defined the weak list) is split again into H_cal (reward calibration) and H_sel (selection). Devtest and all other benchmark dev/test splits are never used in T1 or T2. The NLLB-signal language rule is evaluated on H_sel, not devtest. Gemma-3 pretraining exposure cannot be audited and is a stated limitation.

## Licences (disclose in the paper)
IbomMT train (CC-BY-NC, efi/ibb, about 1,992 pairs) is used by owner decision, tagged non_commercial per row. SSA-MTE train is CC-BY-SA, FLORES+ dev train half CC-BY-SA, the mt560 sources carry an unverified licence mix (provenance check by D2 pending). NLLB-200 3.3B is used only as a reward signal in the final run. Gemma-4 synthetic data is allowed and tagged per row.

## Status
Data (D2 steps A to C) pending. Nothing trained. The launch manifest (section 9 of the rule file) is written when the data is ready.
