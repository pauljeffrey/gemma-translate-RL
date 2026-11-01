## 2. Sentence level (R5's 20 languages x 200 `no_pair` sentences, n = 3,970)

### 2.1 One sample from a sampler: nothing beats greedy

The grid has 8 temperatures x 10 cut-off rules (none, top-p 0.8 / 0.9 / 0.95, top-k 10 / 20 / 50, min-p 0.02 / 0.05 / 0.1) = 80 configurations, one sample per sentence, seed per sentence (the same seeds as R5; my T 0.3 / 0.5 / 0.7 reproduce R5's numbers to 0.01). Delta of macro chrF++ against greedy (32.52), `*` = the 95% paired-bootstrap interval excludes 0:

{{heat_chrf}}

* **No sampler configuration is better than greedy.** The best point estimate is T 0.2 + min-p 0.05 (+0.02, interval [-0.07, +0.12]); 20 of the 30 configurations at T <= 0.4 are statistically tied with greedy (all ten at T 0.2, nine of ten at T 0.3, T 0.4 + top-p 0.8). Without a cut-off every temperature from 0.3 upward is significantly worse (T 0.5 + top-p 0.8, -0.11, is the only tie above T 0.4), and the loss grows monotonically with T: plain T 1.0 loses 2.02 chrF++ [-2.19, -1.84] and 1.89 spBLEU.
* **A cut-off rule recovers part of the loss, never all of it.** At T 1.0 the order is top-p 0.8 (-0.71) > min-p 0.1 (-0.81) > min-p 0.05 (-1.15) > top-p 0.9 (-1.25) > min-p 0.02 (-1.45) > top-p 0.95 (-1.50) > top-k 10 (-1.69) > top-k 20 (-1.77) > top-k 50 (-1.93) > nothing (-2.02): the more probability mass the rule removes, the closer to greedy; a fixed top-k is the worst because it does not adapt to the sharpness of the distribution (and does nothing at all at T <= 0.2, where `T0.2_k20` = `T0.2_k50` = `T0.2` exactly). The **Gemma default** that ships in the model's `generation_config.json` (`do_sample: true`, `top_k: 64`, `top_p: 0.95`, temperature unset = 1.0) is `T1.0_k64_p0.95`: **-1.40 chrF++ [-1.55, -1.24], -1.38 spBLEU, numbers kept 92.7% (greedy 95.7%)**. Anyone who calls `model.generate(...)` without arguments gets this, not greedy.
* **Typical, epsilon and eta sampling** (HF `generate`, vLLM 0.31 has none of them) behave like the other cut-offs: at T 1.0 typical-p 0.8 -0.87, 0.9 -1.20, 0.95 -1.49; epsilon 0.02 -1.21, 0.01 -1.38, 0.003 -1.67, 0.0003 -1.91; eta 0.01 -1.37 (epsilon and eta give the same numbers at 0.001 and 0.0003 to 0.01). Plain HF T 1.0 (-2.00) equals vLLM T 1.0 (-2.02), so the two engines agree. None is better than min-p or top-p at the same removed mass.

{{hf}}

* **Seed variance is small** (5 independent seeds per configuration, sd of the macro chrF++ 0.02-0.09), so the differences above are not seed noise; a single draw is as reliable as the average of five to +-0.1:

{{seeds}}

By quality group (delta chrF++, T x cut-off, strong = swh hau zul kin som; medium = ibo sna lug bem nbl; weak = the other 10):

Strong group:

{{heat_strong}}

Weak group:

{{heat_weak}}

The strong group is the most sensitive to randomness (T 0.5 already -0.70*, T 1.0 -2.75*), the weak group the least (T <= 0.3 within noise of greedy, T 1.0 -1.59*, T 1.0 + min-p 0.1 -0.42*). So **mid- and low-resource languages tolerate sampling better than the strong ones, but even there nothing wins on a single draw**. Medium group and the best-of-group listing:

{{group_best}}

Canonical rows with the quality-adjacent metrics (loop = unfinished / repetitive / empty by `rl.guards`; repeated 3-gram = share of outputs with a repeated word 3-gram that the human reference does not have):

{{canon}}

### 2.2 Is greedy "very repetitive" at sentence level?

Yes, in the weak languages; and the repetition is not what costs chrF++. The human references have a repeated word 3-gram in 2.7% of the sentences; the greedy output in **14.7%** (13.2% not present in the reference). By language (greedy, repeated word 3-gram not in the reference): kbp 52%, nus 52%, ach 36%, fon 33%, yor 17.5%, efi 17%, wol 9%, kab 8.5%, bem 7.5%, amh 5.5%, tir 5%, som 5%; the strong languages 0-3% (zul 0%, swh 2.5%, hau 2.4%). Sampling removes it (T 0.7: weak group 23.5% -> 12.6%, T 1.0: 6.9%) but costs 0.44 / 1.59 chrF++ in that group; cut-offs keep part of the repetition (T 1.0 + min-p 0.05: 11.0%). The hard failure mode (unfinished / looping, the `loop` column) is 0.10% under greedy, so the repetition is the mild kind (repeated function words and phrases), not runaway loops. Consensus selection does not remove it either (beam search was not re-measured: R5's beam hypotheses are no longer on the box; R5 found beam 4 loops more than greedy, 0.6% against 0.1%, mostly in Nuer and Kabiye):

{{rep}}

(`no_repeat_ngram` and `repetition_penalty` remove the repetition by construction, but R5 showed that they hurt chrF++ and delete copied numbers.) I cannot say from these data whether the remaining repetitions are an error a reader would object to: the reference-based metrics do not penalise them (chrF++ is unchanged between greedy and T 0.2 / T 0.3 + cut-off while the repetition falls a little), and no human judgement was collected.

### 2.3 Several samples and a selection rule

Pools of 16 samples per sentence (n = 4, 8 are the first 4 / 8 of the 16), selection by **MBR with chrF++ utility** (the candidate with the highest mean sentence chrF++ against the other candidates; R5's rule), delta chrF++ against greedy:

{{pools}}

* **MBR needs enough samples and a sampler that does not add noise.** With 16 samples the gain is **+0.7 ... +0.8 chrF++** for every sampler except plain T 1.0 (+0.21) and T 1.0 + top-p 0.95 (+0.59): T 0.7 + min-p 0.05 +0.83 [+0.72, +0.94], T 0.7 +0.81, T 0.7 + top-p 0.9 +0.80, T 1.0 + min-p 0.1 +0.79, T 0.5 +0.73. That equals R5's beam search 4 / 8 (+0.66 / +0.74 chrF++) but at 16x the tokens (beam 4 is about 4.6x the HF greedy time), and it **does not improve spBLEU** the way beam does (-0.2 ... +0.2 against beam 4's +0.69), because the utility is chrF++. With 8 samples +0.5 ... +0.6 (as in R5); with 4 samples about +0.2 ... +0.3 or nothing. The oracle (best candidate against the human reference) shows the headroom: +1.7 / +3.1 / +4.4 chrF++ for n = 4 / 8 / 16 at T 1.0 (+2.4 / +3.8 / +5.0 with min-p 0.05), so a better selector than chrF++ consensus could gain several points; none of the ones I tried does.
* **Which sampler for a pool?** For n = 16 the ranking by chrF++ is T 0.7 + min-p 0.05 (+0.83) ~ T 0.7 (+0.81) ~ T 0.7 + top-p 0.9 (+0.80) ~ T 1.0 + min-p 0.1 (+0.79) > T 0.5 (+0.73) ~ T 1.0 + top-p 0.9 (+0.74) > T 0.5 + min-p 0.05 (+0.69) ~ T 1.0 + min-p 0.05 (+0.69) > T 0.3 (+0.57); the intervals overlap. For n = 8: T 0.5 (+0.60) ~ T 0.7 + min-p 0.05 (+0.57) ~ T 0.7 + top-p 0.9 (+0.57) > T 0.7 (+0.49) > T 1.0 + min-p 0.1 (+0.43). **Diversity from a higher temperature pays only when n is large; for n <= 8 use T 0.5 - 0.7 (with min-p 0.05 as a guard).**
* **By group** (MBR n = 16 / 8):

{{pool_groups}}

MBR helps every group, and **helps the weak and medium languages more than the strong ones at the same n** (n = 16 T 0.7: strong +0.47, medium +1.01, weak +0.88; Yoruba +0.7, Bemba +1.6, Kabiye (`kbp`) +1.6, Swahili +0.7, Hausa +0.3). Consensus is a worse selector for the languages where the model is already near the reference (hau, som, zul +0.3 ... +0.4).
* **Adding the greedy output to the pool** (R5's "MBR + greedy"): +0.55 (n = 8) / +0.73 (n = 16) at T 0.7 against +0.49 / +0.81 without; the greedy output is picked for 44-49% of the sentences. It helps when n is small and not when n is large; its spBLEU is a little better (+0.06 ... +0.25 against -0.11 ... +0.22), which is the only reason to include it.
* **Selection by AfriCOMET-QE is metric gaming, not a gain.** Picking the best of 16 by reference-free AfriCOMET-QE raises AfriCOMET (with reference) from 0.623 to 0.668 (+0.045), more than any other config in this study, but **chrF++ is below greedy (-0.10 [T 0.7 pool] and -0.41 [T 1.0 + min-p 0.05 pool])** and spBLEU -0.37 / -0.59: the QE metric and the reference-based metric share an encoder and a training set, so the QE selector and the AfriCOMET judge agree with each other and not with the n-gram metrics. A 50/50 z-score mixture of MBR and QE (`mbrqe`) is +0.67 chrF++ / +0.036 AfriCOMET at n = 16 (T 0.7): a little less chrF++ than plain MBR (+0.81) for a large AfriCOMET gain; I would not use it without a human check because I cannot tell which of the two metrics is right. The oracle selection (best chrF++ against the human reference among 8) gains +3.8 chrF++ but only +0.012 AfriCOMET, which shows how far the two metrics are from each other in what they reward.

{{selection}}

{{oracle}}

### 2.4 AfriCOMET (18 validated languages)

{{africomet}}

### 2.5 Per language (delta chrF++ against greedy; S / M / W = strong / medium / weak group)

{{perlang}}

For Yoruba (26.9 chrF++ greedy; weak group): T 0.3 +0.1, T 0.2 + min-p 0.05 +0.1 (ties), T 0.5 -0.4, T 0.7 -0.3, T 1.0 -1.4 (+top-p 0.9 -0.6, +min-p 0.05 -0.8), MBR n = 16 +0.5 ... +0.7 (every sampler), MBR + greedy +0.5, R5's beam 4 / 8 +0.8 / +0.8. Igbo (medium): T 1.0 -1.3 but T 0.5 only -0.2; MBR n = 16 +0.2 ... +0.7, beam 4 / 8 +0.6 / +0.8.
