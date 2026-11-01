# Did the reward hardening protect against anything? (H1 report, stored outputs, CPU only, saved 2026-10-25)

Files: `reports/data/hardening_evidence_2026-10-25/` (README.txt lists everything; summary_*.csv, paired_*.csv, judges_*.csv, training_*.csv, longdoc_*.csv, fig1-6), scripts `gemma_mt/scripts/hardening_evidence_*.py` (run order 01, 02, 03, 03b, 04, 04b, 05, 06, 07). Detectors = the project's own code (Anchors2 with anchors_v3.json, guards, textstats, segments/chunk_plan/longdoc with longdoc_v2.json, GlotLID v3), settings of final2/config.json (pad weight 0.5, stuff gate OFF). Scope: no_pair rows, 77 RL languages; paired bootstrap over sentences within language (1,000 resamples), macro over languages.

## Verdict
**The stored outputs do not show that the hardening protected against anything.**
- RL-final never showed the failures the hardening targets: copy-of-source hard failures (Jaccard >= 0.9) about 0 in every 100-step block of its 1,433 training steps (max 0.0003); Jaccard copies are 0.08% of benchmark rows for both RL-final and final2.
- final2 is level with RL-final on most indicators. Where it is better, the improvement arrives with the science SFT stage and the hardened RL run then undoes part of that stage's damage. There is no control run (same start, plain reward), so attribution to the hardening is impossible.
- On two targeted indicators final2 is not better: graded entity loss equal; names missing +1.6 pp higher; stop-word profile equal.
- Whole-document loops are the biggest difference and the least attributable: part is the long-document reward v2, but the science stage did the first half and step 250 to 401 got worse.

## 1. Sentence level, RL-final vs final2 (8 benchmark sets, 62 languages, 158,050 paired rows; macro %, difference in pp with 95% CI)
| indicator | RL-final | final2 | diff | human refs |
|---|---|---|---|---|
| unfinished | 0.27 | 0.02 | -0.25 [-0.35,-0.16] | n/a |
| degenerate (v1 hard class) | 0.69 | 0.20 | -0.49 [-0.64,-0.35] | 0 |
| wrong language (GlotLID) | 0.47 | 0.52 | +0.05 [+0.01,+0.10] | 3.74 |
| copy of source, Jaccard >= 0.9 | 0.08 | 0.08 | 0.00 | 0.05 |
| substring copy flag | 1.49 | 1.32 | -0.17 [-0.22,-0.13] | 0.42 |
| numbers lost | 6.44 | 5.66 | -0.78 [-1.27,-0.35] | 12.0 |
| names lost | 31.78 | 33.35 | +1.57 [+1.28,+1.91] | 45.95 |
| graded entity loss | 9.20 | 9.08 | -0.12 [-0.52,+0.26] (equal) | 12.9 |
| stop-word flag | 4.01 | 4.02 | +0.01 [-0.12,+0.14] (equal) | 0.91 |
| padding flag | 11.40 | 10.08 | -1.32 [-1.57,-1.10] | 0.85 |
| repetition flag | 14.41 | 12.76 | -1.65 [-1.94,-1.36] | 1.74 |
| penalty the final reward would charge (0.01 units) | 8.27 | 6.18 | -2.09 [-2.34,-1.88] | 1.18 |
Almost all of the difference is in GlotStoryBook (8 languages: degenerate -3.59, unfinished -1.88, repetition -7.23, padding -6.18 pp); the other benchmarks are within +/-1 pp. By group: clearly better in weak and consensus-only languages (weak padding -2.34, repetition -3.05; consensus-only padding -3.52, repetition -4.06, degenerate -2.43, unfinished -1.24 pp); equal in strong languages (marginally worse by 0.2-0.3 pp). Boilerplate: consensus-only 2.29% (RL-final) to 2.45% (final2), human 0.38%; weak 3.16% to 2.66%.

## 2. Decomposition RL-final -> final2 = science stage + hardened RL run (FLORES+ and AfriScience, pp)
Padding +2.11 (RL-final to A2W, no RL) then -2.76 (A2W to final2); repetition +2.31 then -3.07; names lost +2.02 then -0.48; substring copy -0.28 then +0.04; penalty +1.39 then -2.24. The same indicators fall during RL-final's own plain-reward training (SFT-final to RL-final on FLORES+: padding -14.4, repetition -13.8, degenerate -9.1 pp), so any RL with a QE reward and hard failure classes removes loops. The substring-copy flag rose with RL (SFT-final 0.87%, RL-final 1.51%, human 0.42%) and did not fall in the hardened run.

## 3. Training logs of the final run (rates per sample; steps 1-150 used the earlier detector)
Copy penalised rate 22.9% to 20.9% (slope -1.2 pp per 100 steps [-2.0,-0.2]); padding 3.7% to 4.6% (+0.5 pp per 100 steps); repetition severity 0.014 to 0.024; failed share of the source 14.8% to 17.2%: no avoidance learning visible. The stuffed-output hard failure was 3-4.7% in steps 1-150 and exactly 0 afterwards (the v3 length-bug fix, not policy learning). Ensemble min-branch share 15.7% to 14.3% (flat); the SSA-minus-AfriCOMET gap in r falls -0.024 per 100 steps: no widening toward exploiting one judge.

## 4. Judge disagreement (AfriCOMET-QE was the reward for both models: not independent)
Per row with reference (22 languages): share of rows with |z-gap| > 1: RL-final 10.8%, final2 10.7% (-0.10 pp [-0.35,+0.15]); FLORES+ 5 systems 13.1-13.4%, SFT-final 15.8%. QE-vs-with-reference gaps (cell means; RL-final per-row QE scores are not stored): final2 minus RL-final AfriCOMET -0.0035 [-0.0044,-0.0025], SSA-COMET -0.0018 [-0.0025,-0.0012]: no sign of metric-specific gaming, differences tiny. On Jaccard-copy rows (n = 108) the ensemble mechanism is real (r1 -0.16 vs r2 +1.21; min branch fires on 56%), but rows flagged for copy, padding or repetition get the min branch at the base rate (12.7-14.4% vs 12.6%).

## 5. Long documents, whole-document mode (macro over languages)
| scope | metric | SFT | RL-final | A2W | step 250 | final2 |
|---|---|---|---|---|---|---|
| >= ~2,048 tokens | unfinished % | 97.4 | 56.4 | 27.1 | 14.9 | 15.8 |
| | repetition severe % | 99.1 | 97.4 | 53.5 | 21.9 | 28.8 |
| ~1,024 tokens | unfinished % | 91.8 | 35.5 | 5.2 | 0.4 | 1.3 |
| | repetition severe % | 95.5 | 66.9 | 17.5 | 6.1 | 9.4 |
RL-final to final2 at >= 2,048 tokens: unfinished -40.6 pp [-45.8,-35.4]; RL-final to A2W (no RL) contributes -29.3, A2W to final2 adds -11.3 [-16.0,-6.4] (severe -24.7 [-30.2,-19.6]). Step 250 to 401: severe at >= 2,048 rises +6.9 pp [+2.8,+11.3]. Wrapper mode (sentfix): severe 0 for every system, unfinished about 1%. 77 languages (final2, sentfix): documents with a looping sentence 39.4% (consensus-only), 24.1% (blend), 5.5% (QE languages): weak and consensus-only languages still degenerate heavily.

## 6. Paper claims against this evidence
- 'Reward-hardened' as a design description: accurate (the terms exist and are active); any suggestion that the hardening is demonstrated to protect: UNSUPPORTED; 'level' for final2 vs RL-final: supported.
- 4.3 mean-or-min rule: mechanism supported on real outputs, effect on the policy unmeasured. 4.3 penalties and 7.3 detectors: no demonstrated protective effect for copy (flat), entity (equal; names lost +1.6 pp), stop-word profile (equal): unsupported; padding and repetition in weak and consensus-only languages: partly supported (2-4 pp lower than RL-final, no control); v3 length-bug fix: supported. 7.3 'a policy can satisfy it with outputs that are not translations': no such outputs appear (Jaccard copy 0.08%, stuffed output 0). 7.2 ensemble: rests on synthetic corruptions; the predicted residual search has no support in either run. 4.4/7.4 long-document reward v2: partly supported (large improvement over RL-final, the first and larger step comes with the science stage; step 250 to 401 worsens; no reward-v1 control).

## Caveats
No matched control (start model, data and reward terms all differ); steps 1-150 used earlier anchors; greedy benchmark outputs vs T=1 training samples (exploits that appear only under sampling are missed; the training logs are the only view); padding is a lower bound for 30 languages (bigram table box-only); lexicon detector not computed; only 40-51 languages in the sentence benchmarks (consensus-only 11-13, 2 in FLORES+).
