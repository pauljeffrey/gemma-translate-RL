# Plain-language summary (2026-11-06)

**What we did.** We built a small English-to-African translation model (385 million parameters, 125 languages). We trained it in two steps: ordinary supervised fine-tuning (the "SFT baseline"), then reinforcement learning (RL), where the model produces several translations of each sentence, automatic quality scorers rate them, and the model is pushed towards the better ones (the "RL model"). We then asked what RL really adds, by comparing it with cheaper alternatives: a decoding setting that stops repetition, beam search and minimum Bayes risk (MBR) decoding, and spending the same GPU time on more supervised fine-tuning (the "SFT-only control").

**What we found.**
1. **RL improves every language we test** on three standard benchmarks: about nine points of chrF++ on FLORES+ (22.9 to 31.7), about eight on NTREX, about five on MAFAND-MT.
2. **About half of that gain is finishing sentences and not repeating.** The baseline often stops midway or loops. A one-line decoding setting recovers the same half for the baseline without any training. RL still adds about four more points on top of that, and it stays ahead of the best decoded baseline.
3. **Beam search and MBR decoding help the baseline a lot and the RL model very little.**
4. **More fine-tuning for the same GPU time is inconclusive.** By the rule we wrote down before the run, the control matched RL on the 19 languages for which we gave it extra data and was far behind on all the others. We did not change the rule afterwards.
5. **Almost all of RL's gain comes in the first RL stage.** The last RL step is not distinguishable from random-seed noise.
6. **The model reaches about 92% of NLLB-3.3B's chrF++** on the 43 languages both cover, at about one ninth of the size (nominal 3.3B against 385M). It matches or beats NLLB on seven of those languages, all among NLLB's weakest. We do not claim to beat NLLB.
7. **For science text, the extra fine-tuning step carries the gain, not the science RL.** The same RL without that step ends about 1.4 chrF below.
8. **Whole-document translation still does not work**; translating sentence by sentence does.

**What we could not show.**
- That RL is a better use of compute than more fine-tuning in general (the control is one run, inconclusive by our rule).
- That the improvement beyond finishing is quality a human would confirm: no person judged any output; all scoring is automatic and several scores belong to the reward's own family.
- Seed-to-seed variation for later RL stages and for the supervised baseline: we measured it only for the first RL stage on 20 languages.
- Anything for Arabic dialects, Tamazight, Dinka, Tamasheq and Kanuri: quality stays very low and RL makes the model answer in English more often there.
- That our model beats the earlier 270M model: in the same test harness it is behind before RL, level on two benchmarks after RL, and behind at document level.

**What the owner still has to decide:** the release licence (the Hub repositories declare none), which data parts are released, the author list and contributions, and the title (five options in `TITLE_OPTIONS.md`).
