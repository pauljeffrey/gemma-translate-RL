H1, 2026-10-25: what did the reward hardening protect against? (stored outputs of SFT-final, RL-final, final2 step 401, final-run step 250, A2W alpha 0.7)
Scripts: gemma_mt/scripts/hardening_evidence_*.py (run in order 01, 02, 03, 03b, 04, 04b, 05, 06, 07; common helpers in _common.py, detectors in _detectors.py).
Environment: HE_WORK (scratch dir, default /tmp/hardening_evidence_work; needs the extracted tarball in $HE_WORK/tar and the Hub dataset in $HE_WORK/hubds, both fetched by the scripts when missing, Hub with HF_TOKEN), HE_OUT.
No text of FLORES+ / benchmark sentences is stored here; the per-row table lives in HE_WORK.

language_groups.csv                         content mode of the final run's reward, tertile of SFT-final sentence chrF++ (strong / mid / weak), consensus-only
sentence_indicators_by_system.csv           indicator means (macro over languages and pooled) per system, benchmark, group; human references included
paired_<A>_vs_<B>.csv                       B minus A, paired bootstrap (rows within language, macro over languages), language-resampling interval, verdict; pairs rl:final2 (main), a2w:final2, sft:rl, live250:final2, a2w:live250, rl:a2w
summary_final2_vs_rl_all.csv                the main comparison in one table (all groups, all benchmarks)
summary_decomposition_rl_to_final2.csv      RL-final -> final2 = (RL-final -> A2W, science stage) + (A2W -> final2, hardened RL run); AfriScience rows use the mean of the dev and test intervals (approximate)
summary_per_language_rl_vs_final2.csv       per language difference and interval for 7 indicators
boilerplate_*.csv                           cross-prompt 5-gram boilerplate (consensus attractor symptom)
judges_A..D_*.csv                           judge disagreement analyses (A reference-based rows, B reference-free cell level, C final2 shaped disagreement, D ensemble on flagged outputs)
training_blocks_*.csv, training_trend_tests.csv   training logs of the final run and phase 2c
longdoc_lc1_*.csv, longdoc_afs_papers_*.csv, longdoc_hub77_*.csv   long documents
fig1..fig6                                  figures
