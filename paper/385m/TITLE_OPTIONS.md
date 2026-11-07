# Title options (2026-11-06)

All are at most 14 words, have no colon, and do not claim more than the evidence (no "beats NLLB"; the 385M model reaches about 92% of NLLB-3.3B's chrF++, which has about 8.6 times the parameters).

1. **What Does Reinforcement Learning Add to a Small Translator for African Languages?** (13 words; **default**). A plain question that matches the paper's design: it tests what RL adds beyond decoding and more fine-tuning. It promises no result and uses no jargon.
2. **Reinforcement Learning Teaches a Small African-Language Translator to Finish Its Sentences** (13 words). The catchiest and closest to the main finding (about half of the gain is finishing and not repeating). Risk: it can read as "RL is the only way to teach this"; the paper shows a decoding setting recovers the same half for the fine-tuned model.
3. **Half of Reinforcement Learning's Gain in a Small African Translator Is Finishing Sentences** (13 words). States the measured fact in the headline (49% of the chrF++ gain). Risk: the number "half" is specific to our model and benchmarks; the hedge sits in the abstract, not the title.
4. **Is Reinforcement Learning Better Than More Fine-Tuning for Small African-Language Translators?** (13 words). Frames the matched-compute control. Risk: the registered verdict is "inconclusive", so the title promises an answer the paper does not give.
5. **What the Gain from Reinforcement Learning Consists of in a Small African Translator** (14 words). Descriptive and accurate: the paper decomposes the gain into finishing sentences and translation quality. Less direct than option 1 and less memorable than option 2.

Recommendation: option 1 for the review build; option 2 or 3 if the owner wants a headline for the preprint.
