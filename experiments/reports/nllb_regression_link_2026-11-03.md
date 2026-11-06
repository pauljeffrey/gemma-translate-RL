# NLLB regression link (2026-11-03)

Written from the final report of the analysis run (CPU only: laptop copy plus the stored eval tables; no GPU, no box access). Tables, scripts `s01`-`s12` (run in order) and one figure are in `reports/data/nllb_regression_link_2026-11-03/`. The numbers have not been independently re-run.

Question: do the per-language regressions of the 385M versus the first research version (old 270M SFT / RL) come from the NLLB-mined data in the 385M's stage-B SFT data?

## Verdict: inconclusive, leaning not supported for the Sotho-Tswana-Tsonga / hau / run / lug / kin regressions

### 1. The gap opens before any NLLB-mined text was trained on
- Regression = 385M SFT minus old 270M SFT, on the sentences the old models had not seen (clean).
- 17 languages are more than 5 chrF++ below old SFT: som hau tsn kin tso sot nso twi run orm lin lug bem kab sag ven ssw.
- All 17 were already more than 5 below old SFT with A0, the 270M clean-rerun SFT that the 385M was grown from. A0 has no NLLB-mined, Aya or Tumbuka rows.
- Mean gap old SFT to A0 is -10.7. A0 to 385M SFT is +2.1 (14 of 17 are >= 0). Stage B, which brought the NLLB rows, left the gap at -8.6 on average. NLLB can at most explain why stage B did not close the gap, not why it opened.
- Venda (`ven`) has zero NLLB rows and regressed -9.7, like nso.
- Same-share contrast: sna and nya (NLLB share 0.11 and 0.14, 116-117 M stage-B tokens) regress only -4.0 and -2.2. tsn and nso (0.11 and 0.14, 73-84 M tokens) regress -8.2 and -12.5.

### 2. Cross-language link at the SFT stage (Spearman; share = NLLB-mined tokens / the language's stage-B tokens)

| set | n | rho [95% CI] |
|---|---|---|
| NTREX+MAFAND clean (large sets) | 28 | -0.45 [-0.73, -0.09] |
| same, partial after resource and old-level control | 28 | -0.38 to -0.56 |
| 4-benchmark pooled index | 57 | -0.07 [-0.35, +0.20] |
| FLORES, 48 old-unseen sentences | 45 | +0.04 |
| SmolSent clean | 45 | -0.36 [-0.62, -0.07] |
| A0 minus old SFT (before NLLB) | 57 | +0.19 |

- The 17 regressed languages have mean share 22.8%; the 40 others 25.0%.
- Regression vs old SFT on NTREX+MAFAND with controls (resource level, old level, script, instruction share): the slope is not distinguishable from zero. Simplest model: OLS slope per +10 points of share, 95% bootstrap CI about -3.3 to +0.4 chrF++. At the share of the Sotho-Tswana-Tsonga block (0.11-0.17) that implies about -0.5 chrF++ at most, against observed regressions of -8 to -12.
- After RL there is no negative link: B1 best vs old RL on NTREX+MAFAND gives rho -0.17 [-0.54, +0.23]; with FLORES +0.20.

### 3. Marginal value of NLLB tokens (385M SFT minus A0, FLORES devtest, both models clean)
- Each factor of 10 in new non-NLLB tokens (relative to A0's data) adds +9.6 chrF++ (se 2.0).
- Each factor of 10 in NLLB-mined tokens adds -0.4 (bootstrap 95% [-11.0, +4.8]).
- Difference -9.9 [-19.7, -4.2]; pooled-benchmark version -11.1 [-20.1, -5.7]. NLLB tokens added far less than other new tokens; the data cannot say whether they were neutral or harmful.
- Matched contrasts (treated: share >= 0.30; controls: share <= 0.10, similar non-NLLB tokens and A0 level; exploratory): treated languages gain 3.1 chrF++ less from A0 to SFT (67% negative) but are +1.6 vs old SFT and +2.4 after RL. No consistent sign.

### 4. Within-language evidence
- NLLB-mined rows are measurably noisier than the same language's other rows:
  - GlotLID acceptance of the target: median 0.77 vs 0.99 (39 languages; ful, kon, tzm label mismatch and bam excluded); 26 of 39 are more than 0.1 lower.
  - AfriCOMET-QE is below the language median in 38 of 43 languages (median -0.07).
  - The QE gate kept 54-94% of NLLB rows (median 71%).
  - English text in the target: nso 17%, luo 10%, twi 8%, kab 8%.
  - Fewer than half of NLLB targets are distinct in 15 of 43 languages (kam 12%, nus 10%, kik 15%, kea 21%, kbp 22%). Several of those still improved a lot (kam +8.5, kbp +8.1, nus +4.6 vs A0), so duplication alone is not harmful here.
- Mechanism check: the NLLB LID-failure excess correlates with the drop in the 385M's output language-ID rate vs A0 (rho -0.43 [-0.68, -0.09], n=39); the drop is concentrated in tum, umb, bem, dyu, lua (uncontrolled). The Sotho-Tswana block's NLLB LID failures (27-31%) did not change its output LID.
- kea is the one credible data-quality casualty at the SFT stage: 82% of its stage-B tokens are NLLB-mined (11.3 M of 13.8 M; A0 had only 1.7 M tokens). FLORES chrF++ fell from 36.5 (A0) to 18.9 with hallucination and loops (unfinished 0.17 vs 0.02, repetition 0.20 vs 0.02). RL repaired it to 38.2. It cannot be separated from a variety or domain shift without the ablation.
- The contamination audit already scanned all stage-B packs including the NLLB rows: no >= 3-word benchmark sentence. Paraphrase / MT-of-benchmark contamination is not covered.
- Decoding is not the cross-language explanation: NLLB share and duplicate rate are unrelated to the rise in unfinished or repeating outputs vs A0 (rho -0.03 to -0.14), and the SFT regression is unrelated to the unfinished rate across languages (R2 about 0).

### Regression size by stage
- SFT stage (385M SFT minus old SFT, clean sentences): NTREX -4.1 (12 of 24 languages below -3), MAFAND -3.4, SmolSent -2.9, FLORES-48 -2.7.
- After RL (B1 best minus old RL), estimated, because the stored RL runs have all-row scores only; shifted to the clean sentences by the offset the SFT model shows between clean and all rows: NTREX -0.5 (0 languages below -3), MAFAND -2.0 (3 of 15), FLORES-48 -0.4 (10 of 45 below -3).
- What remains after RL: sot -7.3, pcm -7.2, ibo -5.3, kin -4.8, lug -4.5.

### What the data can and cannot show
- 28 languages with large clean sets give power only for |rho| >= 0.51. The 57-language pooled index mixes in 48-76-sentence sets (chrF++ noise about +-3).
- One seed per model, so no training-noise estimate.
- NLLB share is confounded with language size: the high-share languages (nus, kbp, dyu, kea, ewe, kam, kik, fon) are tail languages.
- The old models differ in data, steps and tokenizer, so "regression vs old" is a composite. The clean rebuild removed the private bundles that carried FLORES English, and the NC filter removed MAFAND-MT.
- AfriScience is not usable as a clean test here: old RL-final and science-stage models trained on that domain; the 385M has had no science stage. (rho -0.77 with share on n=6, p=0.07; old RL-final beats B1 best by 9.0 on lug, 4.0 hau, 3.7 nso, 2.9 zul; B1 is ahead on amh +3.6, yor +0.3.) Not a valid comparison until the science stage.
- GlotLID is unreliable for sibling languages (nso/tsn/sot, bem/lub) and for bam. The LID and quality samples are 500 pairs per language and group.

## Licence-disclosure figure (stage B)

Basis: token composition of the stage-B packs. The packs carry no source field, so this is a reconstruction from the pair shards and the schedule. Check: translation tokens plus instruction tokens reproduce every language's pack tokens within 0.95-1.03.

| quantity | tokens | share of stage-B pack tokens (3.990 B) | share of whole 385M SFT (5.043 B seen) |
|---|---|---|---|
| NLLB-mined bitext | 576 M | **14.4%** | 10.8% |
| + NLLB-Seed (tzm) and Aya collection aligned part (NLLB-3.3B output): NLLB-derived strict | 746 M | **18.7%** | 14.4% |
| + Tumbuka machine-translated instruction sets (system not stated, "possibly NLLB") | 909 M | **22.8%** | 17.9% |

- By rows: 13.91 M NLLB-mined pairs of 42.31 M translation pairs (32.9%). OPUS-NLLB covers 43 languages (13.46 M rows); the Fulah NLLB-mined set adds 0.45 M rows; Aya aligned is 169 M tokens in 12 languages, in the instruction packs.
- A0 contains none of the three. Stage A saw only Aya and Tumbuka via the instruction packs (about 36 M tokens, estimated).
- Executed stage B was 31,353 steps (4.109 B tokens): NLLB-mined 13.3%, strict 17.2%.
- Not counted but worth a legal look: the `*_mt560` OPUS sets (an earlier audit concluded "NLLB-derived = no"), `African-Languages-Lab/multi-open` (3.4 M rows, licence "other"), AfroLingu-MT (owner exception), and the NLLB-3.3B pseudo-reference reward used in RL (reward only).
- Row counts agree with `reports/data/nllb_in_stageB_2026-11-01.csv` (ratio 1.000 in 38 of 43 languages; luo 1.34, ful 1.03; total 13.91 M here vs 13.54 M there).

## Per-language NLLB share and regression, named languages

Columns: NLLB-mined share and NLLB-derived strict share (adds Aya) of stage-B tokens (%), stage-B tokens (M), A0 - old SFT, 385M SFT - A0, 385M SFT - old SFT (4 benchmarks), B1 - old RL (estimated), GlotLID acceptance of NLLB targets / other sources.

| iso | share% | strict% | B Mtok | A0-old | SFT-A0 | SFT-old | B1-oldRL | LID nllb / other |
|---|---|---|---|---|---|---|---|---|
| nso | 14.0 | 34.3 | 72.6 | -11.7 | -0.9 | -12.5 | -1.4 | 0.69 / 0.99 |
| tsn | 10.6 | 10.6 | 83.5 | -17.6 | 9.4 | -8.2 | -2.5 | 0.69 / 0.98 |
| sot | 16.1 | 36.7 | 74.1 | -11.1 | 1.1 | -10.1 | -7.3 | 0.73 / 0.94 |
| tso | 17.2 | 17.2 | 81.0 | -14.3 | 2.3 | -12.1 | -3.1 | 0.73 / 1.00 |
| ssw | 39.9 | 39.9 | 12.4 | -11.8 | 2.0 | -9.7 | -0.3 | 0.76 / 0.85 |
| hau | 13.3 | 27.2 | 100.1 | -10.2 | 4.1 | -6.1 | -2.5 | 0.97 / 1.00 |
| run | 22.2 | 22.2 | 60.5 | -7.8 | -0.9 | -8.8 | -2.1 | 0.77 / 0.91 |
| lug | 44.5 | 44.5 | 34.4 | -15.6 | 0.6 | -15.0 | -4.5 | 0.69 / 0.97 |
| kin | 18.2 | 18.2 | 81.9 | -9.6 | 2.6 | -7.1 | -4.8 | 0.93 / 0.99 |
| kea | 81.9 | 81.9 | 13.8 | +21.9 | -18.4 | +3.4 | +7.6 | 0.78 / 0.70 |
| orm | 22.5 | 22.5 | 53.9 | -7.6 | 0.0 | -7.6 | -1.5 | 0.78 / 0.97 |
| kab | 39.4 | 39.4 | 30.8 | -7.6 | -0.6 | -8.2 | -0.3 | 0.80 / 0.97 |
| bem | 31.4 | 31.4 | 33.7 | -6.9 | 1.2 | -5.6 | 0.0 | 0.37 / 1.00 |
| dyu | 89.4 | 89.4 | 12.2 | 0.0 | -1.8 | -1.8 | +2.3 | 0.58 / 0.98 |
| ven | 0 | 0 | 14.1 | -10.8 | 1.1 | -9.7 | -2.2 | no NLLB |
| twi | 16.0 | 16.0 | 68.6 | -15.8 | 8.6 | -7.2 | -3.1 | 0.65 / 0.99 |
| sna | 10.7 | 22.9 | 117.1 | -12.5 | 8.5 | -4.0 | -2.4 | 0.90 / 0.99 |
| nya | 14.2 | 14.2 | 116.1 | -12.9 | 10.8 | -2.2 | -1.8 | 0.95 / 1.00 |
| zul | 0 | 0 | 140.7 | -13.6 | 12.3 | -1.4 | -0.6 | no NLLB |
| xho | 0 | 9.1 | 170.1 | -16.5 | 16.4 | -0.2 | +0.6 | no NLLB |

- Highest shares: nus 95%, kbp 93%, dyu 89%, kea 82%, ewe 80%, kam 73%, kik 69%, fon 66%.
- Full tables: `nllb_share_eval_languages.csv` (evaluated languages), `nllb_share_by_language.csv` (all 116 schedule languages), `link_language_table_full.csv` (pooled regression columns).

## Proposed decisive test (not run)

Re-run stage B from the stage-A checkpoint with the NLLB-mined rows removed.
- Measured A100 speed: 5.14 s per step (131,072 tokens). Stage B as executed: 31,353 steps, 44.8 h. Without NLLB (547 M tokens fewer): about 27,160 steps.
- Full version: no-NLLB arm 38.8 h plus a second-seed control of the existing recipe 44.8 h, about 84 A100-hours.
- Recommended reduced version: stratified 25% of the stage-B packs, three arms: control seed 1 (7,840 steps, 11.2 h), control seed 2 (7,840 steps, 11.2 h), no-NLLB (6,790 steps, 9.7 h), plus about 1.5 h of evaluation. Total about 34 A100-hours.
- Cheap supplement (about 3 GPU-h): evaluate the existing intermediate shard checkpoints on the 28 big languages. Timing only, not causality.
- Pre-registered readout: mean chrF++ difference (no-NLLB minus control) over nso tsn sot tso ssw hau run lug kin kea orm kab, on NTREX+MAFAND where available and SmolSent/FLORES-48 otherwise. Supported if the difference is >= +2 chrF++, the paired-bootstrap CI excludes 0, and it exceeds the difference between the two control seeds. Rejected if the CI lies within +-1. The NLLB-heavy tail (kea, dyu, kbp, nus, ewe) is reported separately; it is expected to lose without NLLB.
- A post-test RL pass is optional (stage A: 4.4 h). With the current evidence the post-RL gap is expected to be small either way.

## Files and method notes
- `s01_stageB_composition.py`, `s03_nllb_share.py`: composition and share. Universe = keys of `gate_old/keep_final` (A0's data) plus `gate_all/keep_delta` (new pairs); all 42,314,940 keys found. `data*` and `specialist` shards carry no `key`, so it was recomputed with the chain's xxh64 definition. `s01` as saved includes the `extra2_hold` data via a second pass (`--hold-only --finalize-only`).
- `s02`, `s04`, `s11`: eval tables. A0 and the 385M SFT are re-scored on the old-unseen sentences from the archived predictions; the 385M re-score matches `vs_old270m_same_sentences_per_language.csv` exactly.
- `s05`, `s08`, `s09`, `s10`, `s12`: link tests, dose-response, matched contrasts, family and partial-rank analyses. `fig_nllb_share_vs_change.png` is the scatter figure.
- `s06`, `s07`: GlotLID, QE and duplication quality (GlotLID model from the local HF cache).
