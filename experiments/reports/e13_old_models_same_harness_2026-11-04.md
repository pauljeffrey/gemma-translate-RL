# E-13: the old 270M models re-run in the 385M harness (2026-11-04)

> Parallel write-up of the same analysis: `reports/e13_old270m_same_harness_2026-11-04.md`, which is the source cited by `paper_385m/paper.md` (same json; all shared numbers agree).

The paper compared the 385M models with the first research's 270M SFT and RL models using numbers from earlier harness runs (different code state, no intervals). Here both old models were **re-generated and re-scored in the same harness, flags and rows as the 385M models**: old SFT = `<HF_ORG>/gemma3-270m-translate` (main), old RL = rl2c (`<HF_ORG_LEGACY>/gemma-3-270m-translate-rl2c`, GRPO step 1,433, the old RL-final), text tag mode (`GEMMA_MT_TAG_MODE=text`), greedy, 768 new tokens, vLLM bf16, FLORES+ devtest / NTREX / MAFAND, AfriCOMET (`run_old_eval.sh`; both ran after a first attempt failed on the noexec cache, see the overnight log). Rows: the 49 languages that the old models cover (FLORES+ 45, NTREX 24, MAFAND 15), restricted to rows that are **clean for both generations**: not flagged by the 385M audit and not flagged by the first research's audit (union of the flags; the old SFT text held the English of 95 % of FLORES+ devtest, so only 2,160 FLORES+ rows remain, about 48 per language, and the FLORES+ intervals are wide). 113,477 rows scored. Paired bootstrap as in `rl_vs_sft_evidence_2026-11-04.md`. Data: `reports/data/e13_old_models_2026-11-04/evidence_e13.json`.

## Result (chrF++, language macro on the clean rows; 95 % CI)

| system | FLORES+ (45 languages) | NTREX (24) | MAFAND (15) |
|---|---:|---:|---:|
| old 270M SFT | 24.69 | 31.46 | 38.26 |
| old 270M RL (rl2c) | 31.68 | 35.52 | 41.80 |
| 385M SFT | 22.34 | 27.33 | 34.71 |
| 385M RL (mixed best) | 32.03 | 35.55 | 40.25 |

| paired difference | FLORES+ | NTREX | MAFAND | all 84 cells |
|---|---|---|---|---|
| old RL minus old SFT | +6.99 [+6.08, +7.69] | +4.06 [+3.83, +4.30] | +3.54 [+2.91, +4.09] | +5.54 [+5.03, +5.91] |
| 385M RL minus 385M SFT | +9.69 [+8.51, +10.69] | +8.22 [+7.95, +8.50] | +5.54 [+4.86, +6.10] | |
| **385M SFT minus old SFT** | -2.35 [-3.25, -1.38] | -4.13 [-4.40, -3.85] | -3.54 [-4.22, -2.75] | |
| **385M RL minus old RL** | +0.35 [+0.08, +0.62] | +0.02 [-0.07, +0.11] | **-1.55 [-1.72, -1.39]** | about -0.1 (weighted by cells) |

Held-out AfriCOMET-STL (clean rows, 76 cells): old SFT 0.5257, old RL 0.5844 (+0.0587 [+0.0557, +0.0618]); 385M RL minus old RL: FLORES+ -0.0051 [-0.0104, -0.0001], NTREX -0.0129 [-0.0144, -0.0115], MAFAND -0.0203 [-0.0222, -0.0182]; 385M SFT minus old SFT -0.0113 / -0.0250 / -0.0158. Finished / loop rate (FLORES+): old SFT 0.850 / 0.171, old RL 1.000 / 0.002, 385M SFT 0.852 / 0.196, 385M RL 1.000 / 0.017 (on these rows).

## Reading

* **In the same harness the 385M RL model is level with the old 270M RL model on lexical chrF++ (FLORES+ +0.35, NTREX +0.02, MAFAND -1.55) and below it on the held-out AfriCOMET in all three benchmarks.** The 385M model is not an improvement over the first research's 270M RL model on these rows; it is ahead only on FLORES+ by a third of a point and behind on MAFAND by 1.5.
* **The 385M SFT is 2.4 to 4.1 chrF++ below the old SFT**, so the RL gain (+5.5 to +9.7 across the three benchmarks, against +3.5 to +7.0 for the old model) is partly a larger recovery from a weaker start: the RL brings both generations to about the same place. This matches the earlier comparison (R16 in the ledger: NTREX 36.7-37.0 vs 35.6, MAFAND 40.8 vs 41.8) now with intervals and a common harness.
* The old SFT also loops (17 % on FLORES+) and the old RL fixes it (0.2 %): the old RL gain contains the same finishing component as the 385M one (E-01).
* Caveats: rows clean for both audits are few on FLORES+ (about 48 per language); the old models are 270M, trained on the earlier text (95 % FLORES+ English contamination removed by the filter, but other exposure of the old SFT text to NTREX / MAFAND is unmeasured here); one run each; vLLM bf16.
