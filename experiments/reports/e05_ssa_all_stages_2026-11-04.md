# E-05: SSA-COMET-QE (the AfriScience-MT paper's headline) for every stage (2026-11-04)

The AfriScience-MT paper ranks systems by SSA-COMET (primary) and chrF (secondary). Until now only the science-SFT alpha sweep and three document-level rows carried it (`eval_science_sft_385m_2026-11-04.md`, `eval_afriscience_document_level_2026-11-04.md`). This table scores the stored AfriScience-MT dev and test sentence predictions of every stage with `McGill-NLP/ssa-comet-qe` (reference-free; the variant that reproduces the paper's released SSA-COMET values on its strongest runs, mean absolute difference 0.006), the held-out reference-based `masakhane/africomet-stl-1.1`, `McGill-NLP/ssa-comet-mtl` (reference-based use) and plain chrF / chrF++ (sacrebleu, word order 0 / 2, NFC). Scoring only, no generation (`overnight/sent_neural_all.py`, GPU cap 8 GB, 2,482 s, run next to the long-RL evaluation; data and script `reports/data/e05_ssa_all_stages_2026-11-04/`). The row `long-document RL final` comes from the post-run neural pass of the long-RL evaluation (same scorers; it was not yet evaluated when E-05 started).

**Independence rule.** The RL stages were trained against `africomet-qe-stl-1.1` and `ssa-comet-mtl` (the science RL also against its reference-based use and a chrF++ term on AfriScience-MT TRAIN pairs). **SSA-COMET-QE and SSA-COMET-MTL are therefore reward family for every row from "RL stage A" on** and are shown only because the paper's headline is SSA-COMET; they are not evidence of quality for the RL rows. Read them next to chrF (not a reward input on this test split, partly trained on the same domain through the train pairs) and the held-out AfriCOMET-STL. Differences in the third decimal are noise (test-sentence bootstrap intervals are +/-0.001 to +/-0.002, `eval_science_rl_385m_2026-11-04.md` table S2).

## AfriScience-MT TEST, sentence level, macro over the 6 languages (amh hau lug nso yor zul)

| stage | training with | chrF | chrF++ | **SSA-COMET-QE** | AfriCOMET-STL (held out) | SSA-COMET-MTL ref |
|---|---|---:|---:|---:|---:|---:|
| SFT (385M, start of RL) | none | 36.00 | 33.50 | **0.5686** | 0.6064 | 0.4293 |
| RL stage A (final) | RL (sentence reward) | 45.06 | 42.18 | **0.6118** | 0.6822 | 0.5080 |
| RL B1 best | RL (sentence reward) | 45.30 | 42.40 | **0.6113** | 0.6813 | 0.5095 |
| RL mixed best (= science SFT alpha 0) | RL (sentence reward) | 45.40 | 42.44 | **0.6130** | 0.6828 | 0.5123 |
| weak-group run final (not in the main chain) | RL (sentence reward) | 45.19 | 42.22 | **0.6136** | 0.6817 | 0.5126 |
| science-SFT alpha 0 | RL (sentence reward) | 45.40 | 42.44 | **0.6130** | 0.6828 | 0.5123 |
| science-SFT alpha 0.3 | RL + science SFT | 46.28 | 43.40 | **0.6128** | 0.6865 | 0.5159 |
| science-SFT alpha 0.5 (science RL start) | RL + science SFT | 46.63 | 43.80 | **0.6118** | 0.6879 | 0.5150 |
| science-SFT alpha 0.7 | RL + science SFT | 46.56 | 43.81 | **0.6092** | 0.6859 | 0.5114 |
| science-SFT alpha 1.0 | RL + science SFT | 44.92 | 42.30 | **0.5986** | 0.6708 | 0.4949 |
| science RL best (step 75) | RL (science reward incl. ssa-comet-mtl ref) | 46.79 | 43.93 | **0.6135** | 0.6893 | 0.5185 |
| science RL final (step 160) | RL (science reward incl. ssa-comet-mtl ref) | 47.10 | 44.22 | **0.6158** | 0.6932 | 0.5224 |
| long-document RL best (step 50) | RL (+ long-document reward) | 47.16 | 44.26 | **0.6158** | 0.6933 | 0.5220 |
| long-document RL final (step 80) | RL (+ long-document reward) | 47.16 | 44.27 | **0.6165** | 0.6936 | 0.5234 |

## AfriScience-MT DEV, sentence level, macro over the 6 languages (amh hau lug nso yor zul)

| stage | training with | chrF | chrF++ | **SSA-COMET-QE** | AfriCOMET-STL (held out) | SSA-COMET-MTL ref |
|---|---|---:|---:|---:|---:|---:|
| SFT (385M, start of RL) | none | 36.38 | 33.88 | **0.5804** | 0.6226 | 0.4413 |
| RL stage A (final) | RL (sentence reward) | 45.31 | 42.46 | **0.6226** | 0.6962 | 0.5204 |
| RL B1 best | RL (sentence reward) | 45.46 | 42.56 | **0.6229** | 0.6962 | 0.5214 |
| RL mixed best (= science SFT alpha 0) | RL (sentence reward) | 45.74 | 42.81 | **0.6242** | 0.6988 | 0.5249 |
| weak-group run final (not in the main chain) | RL (sentence reward) | 45.52 | 42.58 | **0.6243** | 0.6971 | 0.5248 |
| science-SFT alpha 0 | RL (sentence reward) | 45.74 | 42.81 | **0.6242** | 0.6988 | 0.5249 |
| science-SFT alpha 0.3 | RL + science SFT | 46.53 | 43.64 | **0.6250** | 0.7031 | 0.5296 |
| science-SFT alpha 0.5 (science RL start) | RL + science SFT | 47.17 | 44.35 | **0.6235** | 0.7043 | 0.5292 |
| science-SFT alpha 0.7 | RL + science SFT | 46.96 | 44.22 | **0.6200** | 0.7011 | 0.5264 |
| science-SFT alpha 1.0 | RL + science SFT | 46.02 | 43.36 | **0.6117** | 0.6902 | 0.5144 |
| science RL best (step 75) | RL (science reward incl. ssa-comet-mtl ref) | 47.38 | 44.51 | **0.6255** | 0.7067 | 0.5326 |
| science RL final (step 160) | RL (science reward incl. ssa-comet-mtl ref) | 47.62 | 44.74 | **0.6277** | 0.7098 | 0.5360 |
| long-document RL best (step 50) | RL (+ long-document reward) | 47.58 | 44.69 | **0.6275** | 0.7094 | 0.5358 |
| long-document RL final (step 80) | RL (+ long-document reward) | 47.71 | 44.80 | **0.6274** | 0.7095 | 0.5357 |


## Reading

* **SSA-COMET-QE separates the SFT from every RL stage clearly (0.5686 -> 0.611-0.616 on test) and almost nothing after that.** From RL stage A to the long-document RL final it moves within 0.598-0.617: 0.6118 (stage A), 0.6113 (B1), 0.6130 (mixed), 0.6128 / 0.6118 (science SFT alpha 0.3 / 0.5), 0.6135 / 0.6158 (science RL best / final), 0.6158 / 0.6165 (long-document RL best / final). The paper's headline therefore says: **the RL (sentence-level reward) gives +0.043 over the SFT on test (0.5686 -> 0.6118, in the reward family), and the later stages add at most +0.005 more.** chrF and AfriCOMET keep rising by small steps (chrF 45.06 -> 45.40 -> 46.63 -> 47.10 -> 47.16) that SSA-COMET-QE does not see.
* **Alpha 1.0 (the full science SFT) is the only row clearly worse on every metric** (0.5986): the interpolation at 0.5 was needed.
* The 385M numbers are far below the best released systems of the paper (document level GPT-5.4 0.519 on the concatenated document; different scale and level, see `eval_afriscience_document_level_2026-11-04.md`); a sentence-level SSA-COMET-QE of 0.6165 is not comparable to the paper's document-level table.
* Limitations: one seed per stage; reward-family metrics; test = 5,820 sentences in 6 languages.
