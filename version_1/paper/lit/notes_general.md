# General-ML literature notes (refs_general.bib)

Prepared 2026-10-22. Companion file: `refs_general.bib` (207 entries, keys identical to the keys below).

**Verification method.** Nothing here is from memory alone. arXiv entries: the arXiv abs page was fetched and the exact title, full author list and first-submission date read from its citation meta tags (arXiv comments / journal-ref copied into the bib `note`). ACL Anthology entries: the Anthology's own BibTeX was fetched. Web/blog/model-card entries: page fetched and title/author/date read. Facts in the notes come from abstracts or pages actually read in this session; where a statement rests only on a search-result summary I say so ("search summary"). Numbers appear only when seen in a source. I did not read full papers except where stated (TranslateGemma full text, WALAR and EA-RLVR HTML, Schulman's blog, the Gemma pages).

**Caveats.** (1) arXiv years are first-submission years; several works have later published versions (venues noted only when the arXiv comment or an Anthology record shows them). (2) The entry count exceeds the 50-80 requested; use the **Core shortlist** (90 keys, marked [core] in the sections) if you need a tighter bibliography. (3) Items I considered but could not verify are in the 'unverified' list. (4) The search tool is a general web search; absence of a paper from it is not proof it does not exist (see Novelty check).


## Core shortlist (keys)

shao2024deepseekmath, guo2025deepseekr1, ouyang2022instructgpt, ahmadian2024backtobasics, liu2025understanding, yu2025dapo, he2025rewarding, schulman2020approximating, zuo2025ttrl, xu2024alma, xu2024cpo, alves2024tower, rei2025towerplus, feng2025mtr1zero, finkelstein2026translategemma, liu2026walar, attia2026roundtrip, garciaestrada2026reffree, zheng2025hymt15, gritta2026translatepsy, ramos2025finegrained, ramos2023aligning, choshen2020weaknesses, song2025formatrl, zhou2026entity, wu2025longwriterzero, freitag2022mbr, fernandes2022qualityaware, farinhas2025translate, rei2020comet, rei2022comet22, rei2022cometkiwi, guerreiro2024xcomet, juraska2023metricx, juraska2024metricx24, popovic2015chrf, popovic2017chrfpp, post2018call, koehn2004significance, amrhein2022identifying, yan2023bleurt, zouhar2024pitfalls, wang2024africomet, li2025ssacomet, kargaran2023glotlid, kargaran2024glotcc, caswell2020langid, kreutzer2022quality, gao2023scaling, amodei2016concrete, skalse2022defining, krakovna2020specification, singhal2024long, eisenstein2024helping, coste2024ensembles, wang2023document, wu2024adapting, wang2024benchmarking, varis2021sequence, wang2025segale, alabi2025afridocmt, wang2026doc2frc, post2023escaping, holtzman2020curious, welleck2020unlikelihood, xu2022loop, guerreiro2023needle, raunak2021curious, gemmateam2025gemma3, lacombe2025introducing, gemma3270mit_card, chang2024multilinguality, petrov2023tokenizer, nllbteam2022nllb, haque2026robustrl, song2025fastcurl, biderman2024lora, wortsman2022robust, lin2024mitigating, chu2025sft, shenfeld2025razor, kim2016sequence, shumailov2023curse, longpre2023data, gebru2021datasheets, mitchell2019modelcards, sainz2023nlp, jacovi2023stop, gemma_terms, nllb200_card


## Entries by topic


### A. RL algorithms, KL regularisation, advantage normalisation

- **ahmadian2024backtobasics** [core] - Ahmadian et al. (2024). *Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs*. arXiv:2402.14740 (2024-02-22). <https://arxiv.org/abs/2402.14740> [arXiv comment: 27 pages, 7 figures, 2 tables]
  RLOO: argues most PPO machinery is unnecessary for RLHF and that simple REINFORCE-style variants beat PPO; leave-one-out baseline over k samples. Group-relative methods (GRPO) are close cousins; cite when justifying a critic-free, sample-group baseline.

- **cui2025entropy** - Cui et al. (2025). *The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models*. arXiv:2505.22617 (2025-05-28). <https://arxiv.org/abs/2505.22617>
  Entropy mechanism of RL for reasoning LMs: policy entropy collapses early in RL and performance saturates (empirical law R = -a*e^H + b in abstract). Relevant to exploration collapse in a 270M policy with low-entropy translations.

- **guo2025deepseekr1** [core] - DeepSeek-AI et al. (2025). *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*. arXiv:2501.12948 (2025-01-22). <https://arxiv.org/abs/2501.12948> [journal-ref: Nature volume 645, pages 633-638 (2025)]
  DeepSeek-R1 / R1-Zero: shows reasoning can be incentivised by pure RL with rule-based rewards (published Nature 645, 633-638, 2025 per arXiv journal-ref). Origin of the 'R1-Zero-like' recipe that MT-R1-Zero, SSR-Zero etc. port to translation.

- **hammoud2025trainlong** - Hammoud et al. (2025). *Train Long, Think Short: Curriculum Learning for Efficient Reasoning*. arXiv:2508.08940 (2025-08-12). <https://arxiv.org/abs/2508.08940> [arXiv comment: Under Review]
  Train Long, Think Short: GRPO curriculum that starts with generous token budgets and tightens them (abstract). Reasoning, not MT; length-budget curriculum analogue.

- **he2025rewarding** [core] - He et al. (2025). *Rewarding the Unlikely: Lifting GRPO Beyond Distribution Sharpening*. arXiv:2506.02355 (2025-06-03). <https://arxiv.org/abs/2506.02355>
  Rewarding the Unlikely (EMNLP 2025 per search result): identifies a degenerate rank bias in GRPO where highly probable trajectories are reinforced and rare ones neglected, leading to distribution sharpening; proposes an unlikeliness reward and more PPO updates per batch (abstract). Theorem proving, not MT; relevant to GRPO sharpening in a small policy.

- **hu2025reinforcepp** - Hu et al. (2025). *REINFORCE++: Stabilizing Critic-Free Policy Optimization with Global Advantage Normalization*. arXiv:2501.03262 (2025-01-04). <https://arxiv.org/abs/2501.03262> [arXiv comment: refactor]
  REINFORCE++: critic-free; argues prompt-level (local) advantage normalisation in GRPO/RLOO gives inaccurate advantage estimates and proposes global advantage normalisation. Cite for the std-normalisation critique.

- **liu2025understanding** [core] - Liu et al. (2025). *Understanding R1-Zero-Like Training: A Critical Perspective*. arXiv:2503.20783 (2025-03-26). <https://arxiv.org/abs/2503.20783>
  Dr. GRPO. Abstract: identifies an optimisation bias in GRPO that artificially increases response length (especially incorrect outputs) and proposes Dr. GRPO to remove it. Directly relevant to our length/std-normalisation design choices and to long-document outputs where length bias matters.

- **ouyang2022instructgpt** [core] - Ouyang et al. (2022). *Training language models to follow instructions with human feedback*. arXiv:2203.02155 (2022-03-04). <https://arxiv.org/abs/2203.02155>
  InstructGPT: RLHF (SFT -> reward model -> PPO) aligning LMs with human feedback. Canonical reference for 'optimise a learned proxy reward with a KL leash to the SFT policy'.

- **rafailov2023dpo** - Rafailov et al. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. arXiv:2305.18290 (2023-05-29). <https://arxiv.org/abs/2305.18290>
  DPO: preference optimisation without an explicit RL loop. Background for offline preference baselines (CPO is a DPO-family relative) against which online GRPO is compared.

- **schulman2017ppo** - Schulman et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347 (2017-07-20). <https://arxiv.org/abs/1707.06347>
  PPO: clipped surrogate objective allowing multiple minibatch epochs per batch of samples. GRPO's clipped ratio comes from here.

- **schulman2020approximating** [core] - John Schulman (2020). *Approximating KL Divergence*. blog, 2020-03-07. <http://joschu.net/blog/kl-approx.html>
  Blog post (2020-03-07). Defines Monte-Carlo KL estimators; k3 = (r-1) - log r with r = p/q is unbiased, always non-negative and low variance (page text read). This is the 'k3' KL penalty used in GRPO; cite for it.

- **shao2024deepseekmath** [core] - Shao et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*. arXiv:2402.03300 (2024-02-05). <https://arxiv.org/abs/2402.03300>
  Introduces GRPO inside DeepSeekMath (7B, continued pre-training on 120B math tokens; 51.7% MATH is stated in the abstract). GRPO = PPO-style objective whose baseline comes from a group of sampled outputs, so no value network is needed. Our algorithmic base; cite for the group-relative advantage and the k3 KL penalty.

- **song2025fastcurl** [core] - Song et al. (2025). *FastCuRL: Curriculum Reinforcement Learning with Stage-wise Context Scaling for Efficient Training R1-like Reasoning Models*. arXiv:2503.17287 (2025-03-21). <https://arxiv.org/abs/2503.17287> [arXiv comment: Accepted by EMNLP 2025]
  FastCuRL (EMNLP 2025): curriculum RL with stage-wise context-length scaling for R1-distilled 1.5B models; controlling context length also mitigates entropy collapse (abstract). Closest published analogue to an output-length curriculum under RL, but for math reasoning, not MT.

- **yu2025dapo** [core] - Yu et al. (2025). *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*. arXiv:2503.14476 (2025-03-18). <https://arxiv.org/abs/2503.14476> [arXiv comment: Project Page: this https URL]
  DAPO: open large-scale RL system; four techniques incl. decoupled clip and dynamic sampling (abstract: 50 points on AIME 2024 with Qwen2.5-32B). Source for clip-higher / dynamic-sampling (drop zero-variance groups) style fixes to GRPO; cite when we filter all-equal-reward groups.

- **zheng2025gspo** - Zheng et al. (2025). *Group Sequence Policy Optimization*. arXiv:2507.18071 (2025-07-24). <https://arxiv.org/abs/2507.18071>
  GSPO: sequence-level importance ratio and clipping instead of token-level; claims better stability than GRPO incl. for MoE. Relevant to long (2,048-token) outputs where token-level ratios are noisy.


### B. Consensus / self-consistency / label-free reward analogues

- **agarwal2025entropymin** - Agarwal et al. (2025). *The Unreasonable Effectiveness of Entropy Minimization in LLM Reasoning*. arXiv:2505.15134 (2025-05-21). <https://arxiv.org/abs/2505.15134>
  Entropy minimisation alone (EM-FT / EM-RL, no labels) improves LLM performance on math, physics, coding. Warns that unsupervised self-confidence rewards can work yet may just sharpen the policy; relevant caveat for consensus rewards.

- **gulcehre2023rest** - Gulcehre et al. (2023). *Reinforced Self-Training (ReST) for Language Modeling*. arXiv:2308.08998 (2023-08-17). <https://arxiv.org/abs/2308.08998> [arXiv comment: 23 pages, 16 figures]
  ReST: generate samples from the policy, filter by reward, fine-tune offline; the abstract states the focus is machine translation, with gains in automatic metrics and human evaluation. Early RL-ish MT-with-reward-model precedent, offline.

- **shao2025spurious** - Shao et al. (2025). *Spurious Rewards: Rethinking Training Signals in RLVR*. arXiv:2506.10947 (2025-06-12). <https://arxiv.org/abs/2506.10947>
  Spurious Rewards: RLVR with random or even wrong rewards improves Qwen2.5-Math-7B (abstract: +21.4 points on MATH-500 with random rewards vs 29.1 with ground truth) and attributes it to GRPO clipping behaviour. Caution: gains from a reward ensemble may partly be a base-model/algorithm artefact; use controls.

- **wang2023selfconsistency** - Wang et al. (2022). *Self-Consistency Improves Chain of Thought Reasoning in Language Models*. arXiv:2203.11171 (2022-03-21). <https://arxiv.org/abs/2203.11171> [arXiv comment: Published at ICLR 2023. V2: added PaLM results; V3: added UL2 results; V4: camera ready version at I]
  Self-consistency: sample many reasoning paths and take the most consistent answer. Conceptual ancestor of majority/consensus signals; our 'consensus chrF++' reward is the text-generation analogue.

- **zhao2025intuitor** - Zhao et al. (2025). *Learning to Reason without External Rewards*. arXiv:2505.19590 (2025-05-26). <https://arxiv.org/abs/2505.19590> [arXiv comment: ICLR 2026]
  Intuitor (RLIF): uses the model's own self-certainty as the sole reward (ICLR 2026 per arXiv comment). Another label-free reward analogue.

- **zuo2025ttrl** [core] - Zuo et al. (2025). *TTRL: Test-Time Reinforcement Learning*. arXiv:2504.16084 (2025-04-22). <https://arxiv.org/abs/2504.16084>
  TTRL: RL on unlabeled data using majority-vote as the reward estimate (abstract). Closest published analogue of a consensus-derived pseudo-reference reward; cite for 'majority voting yields effective rewards without labels' and its limits.


### C. RL and preference optimisation for MT (incl. 2025-2026)

- **alves2024tower** [core] - Alves et al. (2024). *Tower: An Open Multilingual Large Language Model for Translation-Related Tasks*. arXiv:2402.17733 (2024-02-27). <https://arxiv.org/abs/2402.17733>
  Tower: open LLM for translation workflows (continued pretraining on multilingual mono+parallel, then instruction tuning). Standard open baseline family; also source of the translation-specialisation vs generality trade-off.

- **attia2026roundtrip** [core] - Attia et al. (2026). *Improving Low-Resource Machine Translation via Round-Trip Reinforcement Learning*. arXiv:2601.12535 (2026-01-18). <https://arxiv.org/abs/2601.12535>
  Round-trip RL for low-resource MT (Jan 2026): GRPO-style self-supervised RL on NLLB 600M and 1.3B with chrF++/BLEU reward on back-translated English; gains for Central Aymara, Friulian, Wolof, Dyula, Bhojpuri, Russian. Counter-evidence for novelty: sub-1.3B + low-resource RL (incl. Wolof, Dyula) exists, but encoder-decoder NLLB and no documents.

- **bahdanau2017actorcritic** - Bahdanau et al. (2016). *An Actor-Critic Algorithm for Sequence Prediction*. arXiv:1607.07086 (2016-07-24). <https://arxiv.org/abs/1607.07086>
  Actor-critic for sequence prediction; critic predicts token value to reduce train/test mismatch. Classical RL-for-seq2seq reference.

- **cheng2025seedx** - Cheng et al. (2025). *Seed-X: Building Strong Multilingual Translation LLM with 7B Parameters*. arXiv:2507.13618 (2025-07-18). <https://arxiv.org/abs/2507.13618>
  Seed-X (Jul 2025): 7B open translation LLMs, 28 languages; instruct model fine-tuned with CoT reasoning and then enhanced with RL for generalisation across language pairs (abstract). Strong 7B RL-trained MT baseline; far larger than 270M.

- **choshen2020weaknesses** [core] - Choshen et al. (2019). *On the Weaknesses of Reinforcement Learning for Neural Machine Translation*. arXiv:1907.01752 (2019-07-03). <https://arxiv.org/abs/1907.01752> [arXiv comment: Accepted to ICLR 2020 (matching content, different style)]
  ICLR 2020. Proves one common RL method for MT does not optimise expected reward and finds others converge very slowly; argues observed gains may be from peakiness of the output distribution. Important sceptical citation for RL-for-MT claims.

- **feng2025mtr1zero** [core] - Feng et al. (2025). *MT-R1-Zero: Advancing LLM-based Machine Translation via R1-Zero-like Reinforcement Learning*. arXiv:2504.10160 (2025-04-14). <https://arxiv.org/abs/2504.10160> [arXiv comment: Work in progress. Our code is available at this https URL]
  MT-R1-Zero (arXiv Apr 2025, 'work in progress'): first open adaptation of R1-Zero RL to MT with a rule-metric mixed reward, no SFT/cold start; variants 3B and 7B; main evaluation en->zh on WMT24 with claimed OOD multilingual/low-resource generalisation (abstract + page). Closest MT-R1 precedent but >=3B and not African/document-level.

- **feng2025mtrewardtree** - Feng et al. (2025). *MT-RewardTree: A Comprehensive Framework for Advancing LLM-Based Machine Translation via Reward Modeling*. arXiv:2503.12123 (2025-03-15). <https://arxiv.org/abs/2503.12123> [arXiv comment: EMNLP 2025 Findings. Project page:this https URL]
  MT-RewardTree: process reward models for MT with automatically generated preference pairs (EMNLP 2025 Findings per arXiv comment). Reward-modelling alternative to metric ensembles.

- **finkelstein2024mbrqe** - Finkelstein et al. (2023). *MBR and QE Finetuning: Training-time Distillation of the Best and Most Expensive Decoding Methods*. arXiv:2309.10966 (2023-09-19). <https://arxiv.org/abs/2309.10966>
  MBR and QE finetuning: distil expensive MBR/QE-reranking decoding into the model at training time. Evidence that metric-guided data selection transfers quality into a single greedy model.

- **finkelstein2024newspalm** - Finkelstein et al. (2024). *Introducing the NewsPaLM MBR and QE Dataset: LLM-Generated High-Quality Parallel Data Outperforms Traditional Web-Crawled Data*. arXiv:2408.06537 (2024-08-13). <https://arxiv.org/abs/2408.06537>
  NewsPaLM MBR and QE dataset: LLM-generated, MBR-decoded and QE-reranked parallel data with both sentence-level and multi-sentence examples; training on it can beat web-crawled data (abstract). Template for synthetic teacher data incl. multi-sentence.

- **finkelstein2026translategemma** [core] - Finkelstein et al. (2026). *TranslateGemma Technical Report*. arXiv:2601.09012 (2026-01-13). <https://arxiv.org/abs/2601.09012>
  TranslateGemma (Jan 2026): Gemma 3 4B/12B/27B, SFT then RL with an ensemble of reward models. Full text read: rewards = MetricX-24-XXL-QE (as 5.0 - score, QE via empty reference), Gemma-AutoMQM-QE, chrF (scaled x2; only reward using synthetic references), a naturalness autorater and a generalist RM; sequence- and token-level advantages are added and batch-normalised. Synthetic data from Gemini 2.5 Flash filtered by MetricX-24-QE, up to 512-token text blobs. Closest large-scale precedent for our reward ensemble; no 270M model, no African-focused evaluation.

- **garciaestrada2026reffree** [core] - Garcia-Estrada et al. (2026). *Reference-Free Reinforcement Learning Fine-Tuning for MT: A Seq2Seq Perspective*. arXiv:2605.15976 (2026-05-15). <https://arxiv.org/abs/2605.15976>
  Reference-free GRPO on NLLB-200 (600M, 1.3B) with LaBSE + COMET-Kiwi reward over 13 languages; up to +5.03 chrF++ and competitive with 3-epoch SFT on morphologically complex languages (abstract). Counter-evidence: small-model GRPO for MT with QE reward exists, sentence-level, seq2seq.

- **gritta2026translatepsy** [core] - Gritta et al. (2026). *TranslatePsy-AfriSLM: High-Quality Data Scaling For Low-Resource Machine Translation*. arXiv:2608.18655 (2026-08-19). <https://arxiv.org/abs/2608.18655> [arXiv comment: EMNLP 2026 (main conference)]
  TranslatePsy-AfriSLM (arXiv 19 Aug 2026; EMNLP 2026 main per arXiv comment): open parallel+synthetic data and fine-tuned small LMs for 19 Sub-Saharan African languages; QE filtering removes up to 96% of training tokens without hurting quality; models with as few as 0.8B parameters beat TranslateGemma-27B and Qwen3.5-122B-A10B (abstract). Closest small-model African MT work found; abstract mentions no RL and no documents. Supports the data-quality-filtering side of our pipeline and sets the small-model bar.

- **he2025r1t1** - He et al. (2025). *R1-T1: Fully Incentivizing Translation Capability in LLMs via Reasoning Learning*. arXiv:2502.19735 (2025-02-27). <https://arxiv.org/abs/2502.19735>
  R1-T1: RL to incentivise human-like reasoning CoT templates for MT; Flores-101, 10+ languages, 40+ directions, stronger on unseen languages (page read). Reasoning-style, not small-model.

- **hu2026rlunseen** - Hu et al. (2026). *Reinforcement Learning Elicits Contextual Learning of Unseen Language Translation*. arXiv:2606.06428 (2026-06-04). <https://arxiv.org/abs/2606.06428> [arXiv comment: 15 pages, 2 figures]
  RL with chrF reward lets LLMs learn to translate completely unseen languages from in-context linguistic knowledge (June 2026); search summary: GRPO on Qwen3-4B-Base. Shows chrF alone is a usable RL reward for low-resource MT.

- **jayarao2026translater1** - Jayarao et al. (2026). *Translate-R1: Cost-Aware Translation Tool Use via Reinforcement Learning*. arXiv:2606.06835 (2026-06-05). <https://arxiv.org/abs/2606.06835> [arXiv comment: 14 pages main text plus appendix, 7 figures, 11 tables]
  Translate-R1 (Jun 2026): RL (confidence-gated GSPO) on Qwen3-4B across 22 languages in 3 resource tiers and 5 domains to learn when to call a translation tool (abstract). RL with low-resource languages (page/search lists Hausa, Swahili, Yoruba, Zulu) but about tool-use policy, not translation quality; not a counter-example.

- **liu2026walar** [core] - Liu et al. (2026). *Mending the Holes: Mitigating Reward Hacking in Reinforcement Learning for Multilingual Translation*. arXiv:2603.13045 (2026-03-13). <https://arxiv.org/abs/2603.13045> [arXiv comment: Our code is available at this https URL]
  WALAR / 'Mending the Holes' (Findings of ACL 2026): GRPO with MetricX-24-Hybrid-XXL QE reward on monolingual text, 101 languages, 8-sample groups, KL beta 0.01 (page read). Documents QE-reward holes: self-generated reference (100% in preliminary Qwen2.5-0.5B run), non-translation, over/under-translation, wrong-language (7.67% -> 92.43% with QE-only on LLaMAX3-8B); fixes via word-alignment and language-alignment rewards. Sentence-level; must cite for reward hacking in MT-RL.

- **lu2026merit** - Lu et al. (2026). *MERIT: Multilingual Expert-Reward Informed Tuning for Chinese-Centric Low-Resource Machine Translation*. arXiv:2604.04839 (2026-04-06). <https://arxiv.org/abs/2604.04839>
  MERIT: reward-informed tuning with GRPO for Chinese to low-resource Southeast Asian languages (3B model per search summary). Another RL-for-low-resource-MT data point; not African.

- **onyame2026curemed** - Onyame et al. (2026). *CURE-Med: Curriculum-Informed Reinforcement Learning for Multilingual Medical Reasoning*. arXiv:2601.13262 (2026-01-19). <https://arxiv.org/abs/2601.13262> [arXiv comment: Accepted at ACL 2026, main conference, oral presentation]
  CURE-Med (ACL 2026 oral per arXiv comment): curriculum RL with GRPO for multilingual medical reasoning across 13 languages incl. Amharic, Yoruba, Swahili; reports language-consistency as a metric (abstract). Evidence that GRPO + curriculum over resource tiers is used for African languages, and that language drift is a failure mode.

- **ramos2023aligning** [core] - Ramos et al. (2023). *Aligning Neural Machine Translation Models: Human Feedback in Training and Inference*. arXiv:2311.09132 (2023-11-15). <https://arxiv.org/abs/2311.09132> [arXiv comment: EAMT 2024]
  EAMT 2024 (per arXiv comment). Compares using a quality metric as reward model for data filtering, RL during training, and reranking at inference, and combinations (abstract). Direct precedent for metric-as-reward in MT and for combining training- and inference-time use.

- **ramos2025finegrained** [core] - Ramos et al. (2024). *Fine-Grained Reward Optimization for Machine Translation using Error Severity Mappings*. arXiv:2411.05986 (2024-11-08). <https://arxiv.org/abs/2411.05986>
  Token-level RL for MT using xCOMET error-span severity mappings to address reward sparsity. Directly relevant to dense/fine-grained reward design; TranslateGemma cites this line of work.

- **ranzato2016mixer** - Ranzato et al. (2015). *Sequence Level Training with Recurrent Neural Networks*. arXiv:1511.06732 (2015-11-20). <https://arxiv.org/abs/1511.06732>
  MIXER: sequence-level training to directly optimise the test metric and reduce exposure bias. First deep-RL-for-seq2seq paper; historic reference.

- **rei2025towerplus** [core] - Rei et al. (2025). *Tower+: Bridging Generality and Translation Specialization in Multilingual LLMs*. arXiv:2506.17080 (2025-06-20). <https://arxiv.org/abs/2506.17080>
  Tower+: adds preference optimisation and RL with verifiable rewards to balance translation specialisation and general capabilities (Pareto frontier). Relevant to catastrophic forgetting when specialising a small model for MT.

- **shang2026scaling** - Shang et al. (2026). *Scaling Model and Data for Multilingual Machine Translation with Open Large Language Models*. arXiv:2602.11961 (2026-02-12). <https://arxiv.org/abs/2602.11961>
  MiLMMT-46 (Feb 2026): scaling model and data for multilingual MT on Gemma 3 via continual pretraining and instruction fine-tuning, 46 languages. Search snippet reported FLORES+ scores for Gemma3-270M (not African); useful for 270M baseline context.

- **shen2016mrt** - Shen et al. (2015). *Minimum Risk Training for Neural Machine Translation*. arXiv:1512.02433 (2015-12-08). <https://arxiv.org/abs/1512.02433> [arXiv comment: Accepted for publication in Proceedings of ACL 2016]
  Minimum risk training for NMT (ACL 2016): optimise parameters directly w.r.t. arbitrary non-differentiable metrics. Classical precedent for optimising MT metrics.

- **song2025formatrl** [core] - Song et al. (2025). *Structured Document Translation via Format Reinforcement Learning*. arXiv:2512.05100 (2025-12-04). <https://arxiv.org/abs/2512.05100> [arXiv comment: IJCNLP-AACL 2025 Main (Oral)]
  FormatRL (IJCNLP-AACL 2025 oral): GRPO on top of SFT for document-level structured (XML/HTML) translation with TreeSim and Node-chrF rewards. Closest 'document-level + GRPO' item found; structure-focused, SAP software documentation, not prose or small models.

- **wang2025deeptrans** - Wang et al. (2025). *DeepTrans: Deep Reasoning Translation via Reinforcement Learning*. arXiv:2504.10187 (2025-04-14). <https://arxiv.org/abs/2504.10187> [arXiv comment: Accepted by Transactions of the Association for Computational Linguistics (TACL)]
  DeepTrans: reasoning translation (free translation) via RL with a reward model; TACL (per arXiv comment). Sentence-level, 7B-class; shows RL for MT beyond metric rewards.

- **wang2025extrans** - Wang et al. (2025). *ExTrans: Multilingual Deep Reasoning Translation via Exemplar-Enhanced Reinforcement Learning*. arXiv:2505.12996 (2025-05-19). <https://arxiv.org/abs/2505.12996> [arXiv comment: 12 pages, 2 figures]
  ExTrans: exemplar-enhanced RL for multilingual deep-reasoning MT; abstract notes prior RL-MT work focused on a few high-resource languages (English, Chinese). Supports the 'low-resource languages are under-served' gap.

- **wang2026loong** - Wang et al. (2026). *Loong: A Human-Like Long Document Translation Agent with Observe-and-Act Adaptive Context Selection*. arXiv:2605.30274 (2026-05-28). <https://arxiv.org/abs/2605.30274>
  Loong (May 2026): long-document translation agent with a 3E memory; context policy optimised by RL from sampled trajectories (preference data). Document-level + RL, but an agentic context-selection policy on large LLMs, not end-to-end document generation.

- **wu2025longwriterzero** [core] - Wu et al. (2025). *LongWriter-Zero: Mastering Ultra-Long Text Generation via Reinforcement Learning*. arXiv:2506.18841 (2025-06-23). <https://arxiv.org/abs/2506.18841> [arXiv comment: ICLR 2026 Oral]
  LongWriter-Zero (ICLR 2026 oral): RL from a base model for ultra-long generation (Qwen2.5-32B) with reward models for length control, quality, structure. Limitations section (search summary) lists reward hacking: templated openings, near-duplicate repetition to hit length, keyword inflation. Precedent for long-output RL and its hacks.

- **xu2024alma** [core] - Xu et al. (2023). *A Paradigm Shift in Machine Translation: Boosting Translation Performance of Large Language Models*. arXiv:2309.11674 (2023-09-20). <https://arxiv.org/abs/2309.11674> [arXiv comment: Accepted at ICLR 2024]
  ALMA: two-stage recipe (monolingual fine-tuning then small high-quality parallel set) for 7B/13B LLM translation (ICLR 2024 per arXiv comment). Baseline family for LLM-MT.

- **xu2024cpo** [core] - Xu et al. (2024). *Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation*. arXiv:2401.08417 (2024-01-16). <https://arxiv.org/abs/2401.08417> [arXiv comment: Accepted at ICML 2024]
  CPO (ICML 2024 per arXiv comment): contrastive preference optimisation to teach models to avoid 'adequate but not perfect' translations; ALMA-R trained on 22K parallel sentences. Preference-optimisation rival to RL with metric rewards; abstract also notes reference quality problems in SFT data.

- **yang2025ssrzero** - Yang et al. (2025). *SSR-Zero: Simple Self-Rewarding Reinforcement Learning for Machine Translation*. arXiv:2505.16637 (2025-05-22). <https://arxiv.org/abs/2505.16637>
  SSR-Zero: reference-free, fully online self-rewarding RL for MT; Qwen2.5-7B, 13K monolingual examples, en<->zh, WMT23/24, Flores200 (page read); SSR-X adds COMET. Self-judging reward is another label-free analogue; no African languages.

- **yang2026grrm** - Yang et al. (2026). *GRRM: Group Relative Reward Modeling for Machine Translation*. arXiv:2602.14028 (2026-02-15). <https://arxiv.org/abs/2602.14028> [arXiv comment: Accepted by EMNLP2026; 23 pages, 7 figures]
  GRRM (EMNLP 2026 per arXiv comment): group-level reward model that ranks the whole GRPO group jointly; argues pointwise metrics miss comparative context. Relevant to group-relative rewards for MT.

- **ye2026twt** - Ye et al. (2026). *Translation with Thought: Difficulty-Adaptive Reasoning via Reinforcement Learning for Multi-Domain Machine Translation*. arXiv:2607.29287 (2026-07-31). <https://arxiv.org/abs/2607.29287> [arXiv comment: 34 pages, 17 figures, and 21 tables. Accepted to ACL 2026]
  Translation with Thought (ACL 2026 per arXiv comment): SFT on R1-distilled CoT then RL with a hybrid quality/efficiency reward; TwT-7B and 14B; evaluated on 3 seen and 59 unseen languages (abstract). Sentence-level multi-domain MT RL at 7-14B.

- **zheng2025hunyuanmt** - Zheng et al. (2025). *Hunyuan-MT Technical Report*. arXiv:2509.05209 (2025-09-05). <https://arxiv.org/abs/2509.05209>
  Hunyuan-MT-7B: 33 languages, emphasis on Mandarin and minority languages; pipeline pretrain -> SFT -> RL -> weak-to-strong RL; WMT25 results (search summary).

- **zheng2025hymt15** [core] - Zheng et al. (2025). *HY-MT1.5 Technical Report*. arXiv:2512.24092 (2025-12-30). <https://arxiv.org/abs/2512.24092>
  HY-MT1.5-1.8B and 7B (Dec 2025): multi-stage pipeline incl. on-device-sized 1.8B model with RL (GRPO per search summary); abstract highlights context-aware translation and format preservation. Smallest RL-trained MT LLM found (1.8B ~ 6.7x our 270M).

- **zhou2026entity** [core] - Zhou et al. (2026). *Incentivizing Parametric Knowledge via Reinforcement Learning with Verifiable Rewards for Cross-Cultural Entity Translation*. arXiv:2604.16881 (2026-04-18). <https://arxiv.org/abs/2604.16881> [arXiv comment: 23 pages, 11 figures, 11 tables]
  EA-RLVR (Apr 2026): entity-anchored RLVR for cross-cultural entity translation. Full text read: removing the format/length gates caused 'length explosion' and reward hacking by enumerating synonymous entities (Appendix A; length gate zeroes reward when output > 2x average reference length). Useful precedent for length gating.


### D. MBR, quality-aware decoding, QE reranking

- **bertsch2023mbr** - Bertsch et al. (2023). *It's MBR All the Way Down: Modern Generation Techniques Through the Lens of Minimum Bayes Risk*. arXiv:2310.01387 (2023-10-02). <https://arxiv.org/abs/2310.01387> [arXiv comment: Under submission]
  Unifies many generation techniques (self-consistency, reranking, etc.) as MBR variants. Conceptual link between consensus rewards, self-consistency and MBR.

- **eikema2020map** - Eikema et al. (2020). *Is MAP Decoding All You Need? The Inadequacy of the Mode in Neural Machine Translation*. arXiv:2005.10283 (2020-05-20). <https://arxiv.org/abs/2005.10283> [arXiv comment: COLING 2020 camera-ready]
  Argues the mode (MAP) is inadequate for NMT and proposes MBR over samples (COLING 2020). Motivation for consensus-based selection instead of likelihood.

- **farinhas2025translate** [core] - Farinhas et al. (2025). *Translate Smart, not Hard: Cascaded Translation Systems with Quality-Aware Deferral*. arXiv:2502.12701 (2025-02-18). <https://arxiv.org/abs/2502.12701> [arXiv comment: Preprint]
  Translate Smart, not Hard (EMNLP 2025 per ACL Anthology listing): cascade where a small model defaults and QE metrics decide deferral to a larger model; QE-based deferral matches the large model while calling it for 30-50% of examples (abstract). Evidence that QE metrics are usable as per-example quality signals, and a deployment angle for a 270M model.

- **fernandes2022qualityaware** [core] - Fernandes et al. (2022). *Quality-Aware Decoding for Neural Machine Translation*. arXiv:2205.00978 (2022-05-02). <https://arxiv.org/abs/2205.00978> [arXiv comment: NAACL2022]
  Quality-aware decoding (NAACL 2022): reranking and MBR using QE/metrics at decoding time. Background to QE-as-reward.

- **freitag2022mbr** [core] - Freitag et al. (2021). *High Quality Rather than High Model Probability: Minimum Bayes Risk Decoding with Neural Metrics*. arXiv:2111.09388 (2021-11-17). <https://arxiv.org/abs/2111.09388> [arXiv comment: Accepted at TACL, presented at NAACL22]
  MBR with neural metrics (TACL): model probability only vaguely correlates with quality; MBR on unbiased samples optimises neural metrics and wins human evaluation. Foundation for consensus / metric-utility rewards.

- **kumar2004mbr** - Kumar and Byrne (2004). *Minimum Bayes-Risk Decoding for Statistical Machine Translation*. Proceedings of the Human Language Technology Conference of the North American Chapter of the Association for Computational Linguistics: HLT-NAACL 2004. <https://aclanthology.org/N04-1022/>
  Original MBR decoding for statistical MT. Root of the MBR / consensus idea behind our consensus reward.

- **muller2021understanding** - Muller and Sennrich (2021). *Understanding the Properties of Minimum Bayes Risk Decoding in Neural Machine Translation*. Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers). <https://aclanthology.org/2021.acl-long.22/>
  ACL 2021. MBR still shows length and token-frequency bias owing to the MT metric used as utility, but increases robustness against copy noise and domain shift (abstract). Utility-metric biases carry into whatever optimises it.


### E. Metrics, benchmarks, significance testing

- **adjovi2026hausafongbe** - Adjovi et al. (2026). *Evaluating Large Language Models for Hausa and Fongbe Machine Translation: Benchmarks, Failures, and Metric Reliability*. arXiv:2606.22269 (2026-06-20). <https://arxiv.org/abs/2606.22269> [arXiv comment: 19 pages, 10 tables]
  Hausa/Fongbe study (Jun 2026): four LLMs, 500-10,000 sentences, BLEU/chrF++/TER/COMET/BERTScore validated against native-speaker judgment (abstract). Finds quality varies strongly by language and performance on one low-resource African language does not predict another.

- **amrhein2022identifying** [core] - Amrhein et al. (2022). *Identifying Weaknesses in Machine Translation Metrics Through Minimum Bayes Risk Decoding: A Case Study for COMET*. arXiv:2202.05148 (2022-02-10). <https://arxiv.org/abs/2202.05148> [arXiv comment: accepted to AACL-IJCNLP 2022]
  AACL-IJCNLP 2022. Sample-based MBR exposes that COMET is not sensitive enough to discrepancies in numbers and named entities; biases hard to remove by synthetic data (abstract). Warns before optimising neural metrics.

- **deutsch2023paragraph** - Deutsch et al. (2023). *Training and Meta-Evaluating Machine Translation Evaluation Metrics at the Paragraph Level*. arXiv:2308.13506 (2023-08-25). <https://arxiv.org/abs/2308.13506> [arXiv comment: Removing extra "and" from author list]
  Paragraph-level meta-evaluation of MT metrics; finds sentence-level metrics scoring whole paragraphs work as well as paragraph-trained metrics on their data. Relevant to scoring long outputs.

- **deutsch2025wmt24pp** - Deutsch et al. (2025). *WMT24++: Expanding the Language Coverage of WMT24 to 55 Languages & Dialects*. arXiv:2502.12404 (2025-02-18). <https://arxiv.org/abs/2502.12404>
  WMT24++: extends WMT24 to 55 languages and dialects with new human references/post-edits for 46 new languages. Evaluation set for large-LLM MT incl. several African languages.

- **freitag2023results** - Freitag et al. (2023). *Results of WMT23 Metrics Shared Task: Metrics Might Be Guilty but References Are Not Innocent*. Proceedings of the Eighth Conference on Machine Translation. <https://aclanthology.org/2023.wmt-1.51/>
  Results of the WMT23 Metrics Shared Task ('Metrics Might Be Guilty but References Are Not Innocent'). Meta-evaluation reference.

- **goyal2022flores101** - Goyal et al. (2021). *The FLORES-101 Evaluation Benchmark for Low-Resource and Multilingual Machine Translation*. arXiv:2106.03193 (2021-06-06). <https://arxiv.org/abs/2106.03193>
  FLORES-101 benchmark (3001 sentences, 101 languages); basis of FLORES-200 used by NLLB. Standard low-resource eval set; also an evaluation-contamination risk.

- **guerreiro2024xcomet** [core] - Guerreiro et al. (2023). *xCOMET: Transparent Machine Translation Evaluation through Fine-grained Error Detection*. arXiv:2310.10482 (2023-10-16). <https://arxiv.org/abs/2310.10482> [arXiv comment: Work in progress]
  xCOMET: learned metric that jointly gives a score and error spans with severity. Used in WALAR as an evaluation metric and by fine-grained RL work.

- **juraska2023metricx** [core] - Juraska et al. (2023). *MetricX-23: The Google Submission to the WMT 2023 Metrics Shared Task*. Proceedings of the Eighth Conference on Machine Translation. <https://aclanthology.org/2023.wmt-1.63/>
  MetricX-23 (WMT23 metrics shared task submission): mT5-based regression metric trained on DA/MQM.

- **juraska2024metricx24** [core] - Juraska et al. (2024). *MetricX-24: The Google Submission to the WMT 2024 Metrics Shared Task*. arXiv:2410.03983 (2024-10-04). <https://arxiv.org/abs/2410.03983> [arXiv comment: Accepted to WMT24]
  MetricX-24 (WMT24): hybrid reference-based/-free metric scoring with or without source/reference; two-stage training. The MetricX-24 Hybrid-XXL variant is the QE reward in WALAR and the MetricX-24-XXL-QE in TranslateGemma.

- **juraska2025metricx25** - Juraska et al. (2025). *MetricX-25 and GemSpanEval: Google Translate Submissions to the WMT25 Evaluation Shared Task*. arXiv:2510.24707 (2025-10-28). <https://arxiv.org/abs/2510.24707> [arXiv comment: Accepted to WMT25]
  MetricX-25 and GemSpanEval (WMT25): newer MetricX generation and an error-span model. Search result: MetricX-25-Ref best average correlation on SSA-MTE.

- **kocmi2024findings** - Kocmi et al. (2024). *Findings of the WMT24 General Machine Translation Shared Task: The LLM Era Is Here but MT Is Not Solved Yet*. Proceedings of the Ninth Conference on Machine Translation. <https://aclanthology.org/2024.wmt-1.1/>
  Findings of WMT24 General MT task ('The LLM Era Is Here but MT Is Not Solved Yet'). Establishes LLM-era evaluation; motivates harder/longer test sets.

- **kocmi2024metricsmaze** - Kocmi et al. (2024). *Navigating the Metrics Maze: Reconciling Score Magnitudes and Accuracies*. arXiv:2401.06760 (2024-01-12). <https://arxiv.org/abs/2401.06760>
  Navigating the Metrics Maze: studies the 'dynamic range' of modern metrics so that metric deltas can be interpreted; cite when reporting small COMET/chrF gains.

- **kocmi2025findings** - Kocmi et al. (2025). *Findings of the WMT25 General Machine Translation Shared Task: Time to Stop Evaluating on Easy Test Sets*. Proceedings of the Tenth Conference on Machine Translation. <https://aclanthology.org/2025.wmt-1.22/>
  Findings of WMT25 General MT task ('Time to Stop Evaluating on Easy Test Sets'): 30 language pairs, harder test sets, source texts supplied as whole multi-paragraph documents (search summary). Supports the move to document-level evaluation.

- **koehn2004significance** [core] - Koehn (2004). *Statistical Significance Tests for Machine Translation Evaluation*. Proceedings of the 2004 Conference on Empirical Methods in Natural Language Processing. <https://aclanthology.org/W04-3250/>
  Paired bootstrap resampling significance tests / confidence intervals for MT metrics. Cite for bootstrap CIs.

- **li2025ssacomet** [core] - Li et al. (2025). *SSA-COMET: Do LLMs Outperform Learned Metrics in Evaluating MT for Under-Resourced African Languages?*. arXiv:2506.04557 (2025-06-05). <https://arxiv.org/abs/2506.04557>
  SSA-MTE: human-annotated MT evaluation data for 14 African language pairs (news, >73,000 sentence annotations); SSA-COMET and SSA-COMET-QE metrics (abstract). Notes AfriCOMET inconsistency in extremely low-resource scenarios.

- **popovic2015chrf** [core] - Popovic (2015). *chrF: character n-gram F-score for automatic MT evaluation*. Proceedings of the Tenth Workshop on Statistical Machine Translation. <https://aclanthology.org/W15-3049/>
  chrF: character n-gram F-score for MT evaluation. Core string metric; robust for morphologically rich/low-resource languages and used as a reward in TranslateGemma and RL-for-unseen-language work.

- **popovic2017chrfpp** [core] - Popovic (2017). *chrF++: words helping character n-grams*. Proceedings of the Second Conference on Machine Translation. <https://aclanthology.org/W17-4770/>
  chrF++: adds word n-grams to chrF. The 'chrF++' we report/optimise.

- **post2018call** [core] - Post (2018). *A Call for Clarity in Reporting BLEU Scores*. arXiv:1804.08771 (2018-04-23). <https://arxiv.org/abs/1804.08771> [arXiv comment: 6 pages, 1 figure; journal-ref: Proceedings of the Third Conference on Machine Translation (WMT18). 2018]
  sacreBLEU paper ('A Call for Clarity in Reporting BLEU Scores'): BLEU is parameterised and scores are not comparable without signatures. Cite for sacreBLEU/chrF++ signatures.

- **rei2020comet** [core] - Rei et al. (2020). *COMET: A Neural Framework for MT Evaluation*. Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP). <https://aclanthology.org/2020.emnlp-main.213/>
  COMET (EMNLP 2020): learned MT metric from cross-lingual encoder; base of COMET-22 and CometKiwi.

- **rei2022comet22** [core] - Rei et al. (2022). *COMET-22: Unbabel-IST 2022 Submission for the Metrics Shared Task*. Proceedings of the Seventh Conference on Machine Translation (WMT). <https://aclanthology.org/2022.wmt-1.52/>
  COMET-22 (wmt22-comet-da): WMT22 submission, the default reference-based COMET used in practice.

- **rei2022cometkiwi** [core] - Rei et al. (2022). *CometKiwi: IST-Unbabel 2022 Submission for the Quality Estimation Shared Task*. Proceedings of the Seventh Conference on Machine Translation (WMT). <https://aclanthology.org/2022.wmt-1.60/>
  CometKiwi: reference-free QE model (WMT22 QE task). Candidate QE reward; limited by encoder language coverage.

- **sellam2020bleurt** - Sellam et al. (2020). *BLEURT: Learning Robust Metrics for Text Generation*. Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. <https://aclanthology.org/2020.acl-main.704/>
  BLEURT (ACL 2020): pretrained learned metric; also the target of the 'BLEURT has universal translations' MRT exploitation study.

- **wang2024africomet** [core] - Wang et al. (2023). *AfriMTE and AfriCOMET: Enhancing COMET to Embrace Under-resourced African Languages*. arXiv:2311.09828 (2023-11-16). <https://arxiv.org/abs/2311.09828> [arXiv comment: Accepted by NAACL 2024]
  AfriMTE/AfriCOMET (NAACL 2024): human-rated MT evaluation data and COMET-style metrics for 13 African languages built on AfroXLM-R. Reliability depends on encoder language coverage.

- **yan2023bleurt** [core] - Yan et al. (2023). *BLEURT Has Universal Translations: An Analysis of Automatic Metrics by Minimum Risk Training*. arXiv:2307.03131 (2023-07-06). <https://arxiv.org/abs/2307.03131> [arXiv comment: Accepted to ACL 2023 main conference]
  ACL 2023. Minimum risk training reveals universal adversarial translations in BLEURT and BARTScore; causes: distribution biases in training data and metric model weaknesses (abstract). Core citation that neural metrics can be gamed when optimised against.

- **zouhar2024pitfalls** [core] - Zouhar et al. (2024). *Pitfalls and Outlooks in Using COMET*. arXiv:2408.15366 (2024-08-27). <https://arxiv.org/abs/2408.15366>
  COMET pitfalls (abstract): technical (software versions, precision), data (empty content, language mismatch, translationese, domain bias) and reporting issues; scores not comparable across setups. Cite for language-mismatch blind spot and for reporting COMET configuration.


### F. Language identification

- **burchell2023openlid** - Burchell et al. (2023). *An Open Dataset and Model for Language Identification*. arXiv:2305.13820 (2023-05-23). <https://arxiv.org/abs/2305.13820> [arXiv comment: To be published in ACL 2023]
  OpenLID (ACL 2023): LID for 201 languages, macro-F1 0.93 and FPR 0.033 (abstract). Alternative LID baseline.

- **caswell2020langid** [core] - Caswell et al. (2020). *Language ID in the Wild: Unexpected Challenges on the Path to a Thousand-Language Web Text Corpus*. arXiv:2010.14571 (2020-10-27). <https://arxiv.org/abs/2010.14571> [arXiv comment: Accepted to COLING 2020. 9 pages with 8 page abstract]
  COLING 2020. LangID models trained on up to 1,629 languages look comparable on held-out sets, but human-judged accuracy on web-crawled corpora is only around 5% for many lower-resource languages (abstract). Core citation for LID unreliability on low-resource languages.

- **glotlid_card** - cis-lmu (Hugging Face organisation) (accessed 2026). *GlotLID model card (cis-lmu/glotlid)*. HF model card. <https://huggingface.co/cis-lmu/glotlid>
  Hugging Face card for cis-lmu/glotlid: 'V3 supports 2102 labels' (ISO code + script). Version-specific citation for the model we run.

- **kargaran2023glotlid** [core] - Kargaran et al. (2023). *GlotLID: Language Identification for Low-Resource Languages*. arXiv:2310.16248 (2023-10-24). <https://arxiv.org/abs/2310.16248> [arXiv comment: EMNLP 2023]
  GlotLID-M (Findings of EMNLP 2023): language ID for 1665 languages, evaluated for reliability on low-resource languages. LID used to detect off-target/wrong-language outputs.

- **kargaran2024glotcc** [core] - Kargaran et al. (2024). *GlotCC: An Open Broad-Coverage CommonCrawl Corpus and Pipeline for Minority Languages*. arXiv:2410.23825 (2024-10-31). <https://arxiv.org/abs/2410.23825> [arXiv comment: NeurIPS 2024]
  GlotCC (NeurIPS 2024): corpus + pipeline using GlotLID v3.0, which per the paper covers over 2,000 linguistic labels with a better rejection model; the HF model card lists 2102 labels. Cite for GlotLID v3.

- **kreutzer2022quality** [core] - Kreutzer et al. (2021). *Quality at a Glance: An Audit of Web-Crawled Multilingual Datasets*. arXiv:2103.12028 (2021-03-22). <https://arxiv.org/abs/2103.12028> [arXiv comment: Accepted at TACL; pre-MIT Press publication version; journal-ref: Transactions of the Association for Computational Linguistics (2022) 10: 50-72]
  TACL. Manual audit of 205 language-specific corpora (CCAligned, ParaCrawl, WikiMatrix, OSCAR, mC4); lower-resource corpora show systematic issues, with many corpora largely unusable or mislabelled (abstract). Cite for noisy low-resource data and LID errors.

- **marchisio2024confusion** - Marchisio et al. (2024). *Understanding and Mitigating Language Confusion in LLMs*. arXiv:2406.20052 (2024-06-28). <https://arxiv.org/abs/2406.20052> [arXiv comment: EMNLP 2024 Main Conference Camera-ready. v3: hi, ru not run for monolingual Okapi]
  Language Confusion Benchmark (EMNLP 2024): LLMs fail to consistently generate in the requested language. Motivates wrong-language/off-target penalties.


### G. Reward hacking, Goodhart, length exploitation

- **amodei2016concrete** [core] - Amodei et al. (2016). *Concrete Problems in AI Safety*. arXiv:1606.06565 (2016-06-21). <https://arxiv.org/abs/1606.06565> [arXiv comment: 29 pages]
  Concrete Problems in AI Safety: includes 'reward hacking' as a practical research problem. Historic citation.

- **casper2023open** - Casper et al. (2023). *Open Problems and Fundamental Limitations of Reinforcement Learning from Human Feedback*. arXiv:2307.15217 (2023-07-27). <https://arxiv.org/abs/2307.15217>
  Open Problems and Fundamental Limitations of RLHF: surveys flaws incl. reward misspecification/hacking. Survey citation.

- **coste2024ensembles** [core] - Coste et al. (2023). *Reward Model Ensembles Help Mitigate Overoptimization*. arXiv:2310.02743 (2023-10-04). <https://arxiv.org/abs/2310.02743> [arXiv comment: Accepted at ICLR 2024]
  ICLR 2024. Ensembles of reward models mitigate overoptimisation in best-of-n and PPO. Supports our reward-ensemble defence.

- **eisenstein2024helping** [core] - Eisenstein et al. (2023). *Helping or Herding? Reward Model Ensembles Mitigate but do not Eliminate Reward Hacking*. arXiv:2312.09244 (2023-12-14). <https://arxiv.org/abs/2312.09244> [arXiv comment: Published at the 2024 Conference on Language Modeling (CoLM)]
  Helping or Herding? (COLM 2024): reward-model ensembles mitigate but do not eliminate reward hacking; pretrain-seed ensembles generalise better. Cite for ensemble-as-defence caveat.

- **fu2025rewardshaping** - Fu et al. (2025). *Reward Shaping to Mitigate Reward Hacking in RLHF*. arXiv:2502.18770 (2025-02-26). <https://arxiv.org/abs/2502.18770>
  Reward shaping study: the RL reward should be bounded and grow fast initially then saturate (abstract). Relevant to bounding/clipping each reward component.

- **gao2023scaling** [core] - Gao et al. (2022). *Scaling Laws for Reward Model Overoptimization*. arXiv:2210.10760 (2022-10-19). <https://arxiv.org/abs/2210.10760>
  Gao, Schulman, Hilton: proxy-reward overoptimisation follows predictable functional forms under RL and best-of-n, scaling smoothly with reward-model size. Canonical citation for reward overoptimisation / Goodhart in RLHF.

- **karwowski2023goodhart** - Karwowski et al. (2023). *Goodhart's Law in Reinforcement Learning*. arXiv:2310.09144 (2023-10-13). <https://arxiv.org/abs/2310.09144>
  Goodhart's law in RL: optimising an imperfect proxy beyond a critical point lowers true performance; gives early-stopping method (abstract). Supports monitoring held-out true metrics.

- **krakovna2020specification** [core] - Victoria Krakovna (2020). *Specification gaming: the flip side of AI ingenuity*. DeepMind blog, 2020-04-21. <https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/>
  DeepMind blog (2020-04-21): defines specification gaming as satisfying the literal specification of an objective without achieving the intended outcome. Accessible definition + examples list.

- **manheim2019categorizing** - Manheim et al. (2018). *Categorizing Variants of Goodhart's Law*. arXiv:1803.04585 (2018-03-13). <https://arxiv.org/abs/1803.04585> [arXiv comment: 10 pages]
  Four variants of Goodhart's law. Taxonomy for metric overoptimisation.

- **pan2022effects** - Pan et al. (2022). *The Effects of Reward Misspecification: Mapping and Mitigating Misaligned Models*. arXiv:2201.03544 (2022-01-10). <https://arxiv.org/abs/2201.03544> [arXiv comment: ICLR 2022; 19 pages]
  ICLR 2022. Constructs RL environments with misspecified rewards and shows more capable agents exploit them more; reports phase transitions. Cite for 'hacking grows with capability/training time'.

- **park2024disentangling** - Park et al. (2024). *Disentangling Length from Quality in Direct Preference Optimization*. arXiv:2403.19159 (2024-03-28). <https://arxiv.org/abs/2403.19159>
  Disentangling Length from Quality in DPO: verbosity exploitation and a length-regularised fix. Length-bias mitigation citation.

- **singhal2024long** [core] - Singhal et al. (2023). *A Long Way to Go: Investigating Length Correlations in RLHF*. arXiv:2310.03716 (2023-10-05). <https://arxiv.org/abs/2310.03716> [arXiv comment: 21 pages, 13 figures, Accepted to COLM 2024]
  COLM 2024. Reward gains in RLHF are largely driven by increasing response length; even a purely length-based reward reproduces most downstream RLHF improvements (abstract). Core citation for length exploitation.

- **skalse2022defining** [core] - Skalse et al. (2022). *Defining and Characterizing Reward Hacking*. arXiv:2209.13085 (2022-09-27). <https://arxiv.org/abs/2209.13085> [arXiv comment: 23 pages; modified (fix typo in Figure 1, update link to code in Appendix, remove unrendered charact]
  First formal definition of reward hacking; defines an unhackable proxy and shows it is rare (abstract). Theory citation.


### H. Document-level and long-context MT; alignment

- **alabi2025afridocmt** [core] - Alabi et al. (2025). *AFRIDOC-MT: Document-level MT Corpus for African Languages*. arXiv:2501.06374 (2025-01-10). <https://arxiv.org/abs/2501.06374> [arXiv comment: EMNLP 2025]
  AFRIDOC-MT (EMNLP 2025): document-level multi-parallel corpus for English with Amharic, Hausa, Swahili, Yoruba, Zulu (605 documents); sentence-trained models struggled to generalise to longer documents and LLMs show under-generation, repetition and off-target output especially for African languages (abstract). Key African document-level benchmark.

- **an2024effective** - An et al. (2024). *Why Does the Effective Context Length of LLMs Fall Short?*. arXiv:2410.18745 (2024-10-24). <https://arxiv.org/abs/2410.18745>
  Effective context length of open LLMs falls short, often not exceeding half of training length; attributes to left-skewed position frequency (abstract). Motivates training on long documents.

- **bai2025longwriter** - Bai et al. (2024). *LongWriter: Unleashing 10,000+ Word Generation from Long Context LLMs*. arXiv:2408.07055 (2024-08-13). <https://arxiv.org/abs/2408.07055>
  LongWriter: effective generation length is bounded by the longest samples seen in SFT; builds long-output data (AgentWrite). Supports training on long targets.

- **gale1993program** - Gale and Church (1993). *A Program for Aligning Sentences in Bilingual Corpora*. Computational Linguistics. <https://aclanthology.org/J93-1004/>
  Gale and Church: length-based sentence alignment of bilingual corpora. Classical alignment algorithm relevant to building/scoring document-level pairs.

- **junczysdowmunt2019microsoft** - Junczys-Dowmunt (2019). *Microsoft Translator at WMT 2019: Towards Large-Scale Document-Level Neural Machine Translation*. arXiv:1907.06170 (2019-07-14). <https://arxiv.org/abs/1907.06170> [arXiv comment: WMT 2019 Shared Task submission]
  WMT19 English-German document-level NMT: builds training sequences of up to 1000 subword segments using document boundaries (abstract). Early evidence that long-sequence training is feasible for NMT.

- **kang2020dynamic** - Kang et al. (2020). *Dynamic Context Selection for Document-level Neural Machine Translation via Reinforcement Learning*. arXiv:2010.04314 (2020-10-09). <https://arxiv.org/abs/2010.04314> [arXiv comment: Accepted to EMNLP 2020 long paper]
  EMNLP 2020. RL (policy gradient) to select dynamic context for document-level NMT. Earlier RL-for-DocMT precedent (pre-LLM, context selection, not generation).

- **karpinska2023literary** - Karpinska et al. (2023). *Large language models effectively leverage document-level context for literary translation, but critical errors persist*. arXiv:2304.03245 (2023-04-06). <https://arxiv.org/abs/2304.03245> [arXiv comment: preprint (31 pages)]
  LLMs translating whole literary paragraphs beat sentence-by-sentence translation in human evaluation over 18 language pairs, but critical errors including content omissions persist (abstract).

- **koehn2017six** - Koehn et al. (2017). *Six Challenges for Neural Machine Translation*. arXiv:1706.03872 (2017-06-12). <https://arxiv.org/abs/1706.03872> [arXiv comment: 12 pages; First Workshop on Neural Machine Translation, 2017]
  Six Challenges for NMT: includes long sentences, domain mismatch, beam search. Historic reference for NMT degrading on long inputs.

- **liu2020mbart** - Liu et al. (2020). *Multilingual Denoising Pre-training for Neural Machine Translation*. arXiv:2001.08210 (2020-01-22). <https://arxiv.org/abs/2001.08210> [arXiv comment: Work in progress]
  mBART: multilingual denoising pre-training for seq2seq; one of the first pre-trained complete seq2seq models; includes document-level fine-tuning. Background for multilingual seq2seq baselines.

- **liu2024lost** - Liu et al. (2023). *Lost in the Middle: How Language Models Use Long Contexts*. arXiv:2307.03172 (2023-07-06). <https://arxiv.org/abs/2307.03172> [arXiv comment: 18 pages, 16 figures. Accepted for publication in Transactions of the Association for Computational]
  Lost in the Middle (TACL): LLMs use information in long contexts unevenly, degrading when relevant content is in the middle. Caution for long source documents.

- **mrozinski2025qerank** - Mrozinski et al. (2025). *Quality Estimation Reranking for Document-Level Translation*. arXiv:2510.08870 (2025-10-10). <https://arxiv.org/abs/2510.08870> [arXiv comment: 9 pages, 4 figures]
  QE reranking for document-level MT: gains of +2.00 BLEURT with 2 candidates and +5.09 with 32 (abstract), shrinking on the longest documents (512-1024 source tokens). Evidence on QE at long lengths.

- **murray2018correcting** - Murray et al. (2018). *Correcting Length Bias in Neural Machine Translation*. arXiv:1808.10006 (2018-08-29). <https://arxiv.org/abs/1808.10006> [arXiv comment: WMT 2018]
  Correcting length bias: NMT's tendency toward too-short translations is tied to the beam problem via label bias; a per-word reward helps (abstract). Relevant to under-translation and length rewards.

- **post2023escaping** [core] - Post et al. (2023). *Escaping the sentence-level paradigm in machine translation*. arXiv:2304.12959 (2023-04-25). <https://arxiv.org/abs/2304.12959>
  Post and Junczys-Dowmunt: argue MT is stuck in a sentence-level paradigm and address architectures, document-level training data, and evaluation together (abstract). Motivation for a document-level extension.

- **ramos2025docblocks** - Ramos et al. (2025). *Multilingual Contextualization of Large Language Models for Document-Level Machine Translation*. arXiv:2504.12140 (2025-04-16). <https://arxiv.org/abs/2504.12140> [arXiv comment: COLM 2025]
  DocBlocks (COLM 2025): fine-tuning on document-level data to improve long-document LLM translation. Search summary: concatenated documents up to 32,768 tokens; some instruction-tuned models degrade as chunk size grows.

- **sennrich2011iterative** - Sennrich and Volk (2011). *Iterative, MT-based Sentence Alignment of Parallel Texts*. Proceedings of the 18th Nordic Conference of Computational Linguistics (NODALIDA 2011). <https://aclanthology.org/W11-4624/>
  Bleualign-style iterative MT-based sentence alignment (NODALIDA 2011). Cite for MT-assisted alignment.

- **stahlberg2019cat** - Stahlberg et al. (2019). *On NMT Search Errors and Model Errors: Cat Got Your Tongue?*. arXiv:1908.10090 (2019-08-27). <https://arxiv.org/abs/1908.10090> [arXiv comment: EMNLP-2019]
  For >50% of WMT15 en-de sentences the model's global best score is the empty translation; root cause is an inherent bias toward shorter translations (abstract). Relevant to under-generation/early EOS on long outputs.

- **thompson2019vecalign** - Thompson and Koehn (2019). *Vecalign: Improved Sentence Alignment in Linear Time and Space*. Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP). <https://aclanthology.org/D19-1136/>
  Vecalign: sentence alignment from multilingual embeddings in linear time/space. Practical aligner for noisy low-resource document pairs.

- **unanue2020discourse** - Unanue et al. (2020). *Leveraging Discourse Rewards for Document-Level Neural Machine Translation*. arXiv:2010.03732 (2020-10-08). <https://arxiv.org/abs/2010.03732> [arXiv comment: Accepted at COLING 2020]
  COLING 2020. RL objective optimising lexical cohesion and coherence rewards for document-level NMT. Pre-LLM precedent for document-level reward design.

- **varis2021sequence** [core] - Variš et al. (2021). *Sequence Length is a Domain: Length-based Overfitting in Transformer Models*. arXiv:2109.07276 (2021-09-15). <https://arxiv.org/abs/2109.07276>
  Sequence Length is a Domain: length-based overfitting in Transformers; models trained on short sequences generalise poorly to longer ones. Core citation for length generalisation failure in MT.

- **voita2019good** - Voita et al. (2019). *When a Good Translation is Wrong in Context: Context-Aware Machine Translation Improves on Deixis, Ellipsis, and Lexical Cohesion*. arXiv:1905.05979 (2019-05-15). <https://arxiv.org/abs/1905.05979> [arXiv comment: ACL 2019 (camera-ready)]
  ACL 2019. Context-aware NMT improves deixis, ellipsis and lexical cohesion; shows standard metrics are insensitive to these. Argues document-level improvements need targeted evaluation.

- **wang2023document** [core] - Wang et al. (2023). *Document-Level Machine Translation with Large Language Models*. arXiv:2304.02210 (2023-04-05). <https://arxiv.org/abs/2304.02210> [arXiv comment: Longyue Wang, Chenyang Lyu, Tianbo Ji, Zhirui Zhang are equal contributors]
  Document-level MT with LLMs (ChatGPT as testbed): effects of context-aware prompts, comparison with document-level MT, discourse modelling probes. Foundational LLM DocMT study.

- **wang2024benchmarking** [core] - Wang et al. (2024). *Benchmarking and Improving Long-Text Translation with Large Language Models*. Findings of the Association for Computational Linguistics: ACL 2024. <https://aclanthology.org/2024.findings-acl.428/>
  Findings of ACL 2024. Instruction-based benchmark and comparison of MT systems vs LLMs on document-level translation; LLM quality degrades with document length (search-result summary); explores extrapolation strategies.

- **wang2025segale** [core] - Wang et al. (2025). *Extending Automatic Machine Translation Evaluation to Book-Length Documents*. arXiv:2509.17249 (2025-09-21). <https://arxiv.org/abs/2509.17249> [arXiv comment: Accepted for EMNLP 2025 main conference]
  SEGALE (EMNLP 2025): extends metrics to long documents via sentence segmentation/alignment, handling under-/over-translation; shows many open-weight LLMs fail to translate documents at their reported maximum context lengths (abstract). Evaluation method for our long outputs and evidence of long-output failure.

- **wang2026doc2frc** [core] - Wang et al. (2026). *Doc2FRC: Length-Consistent Document-Level Machine Translation via Fixed-Range Chunking*. arXiv:2609.12674 (2026-09-11). <https://arxiv.org/abs/2609.12674> [arXiv comment: To appear in Proceedings of the Eleventh Conference on Machine Translation (WMT2026)]
  Doc2FRC (to appear WMT 2026): full-document translation in one pass suffers n-gram repetition and progressive quality degradation; fixed-range chunking aligns train/inference length distributions; 7B LLMs, IWSLT2017 and a 10-language GlobVDoc test set (abstract + page). Supporting evidence for length-related failure; no RL.

- **wu2024adapting** [core] - Wu et al. (2024). *Adapting Large Language Models for Document-Level Machine Translation*. arXiv:2401.06468 (2024-01-12). <https://arxiv.org/abs/2401.06468> [arXiv comment: 25 pages, 18 tables, 7 figures; ARR Feb 2024, 4/3/2, meta 2, rejected by ACL2024; ARR June 2024, 4.5]
  Adapting LLMs for DocMT: two fine-tuning methods, three backbones, 18 tasks over nine language pairs; specialised models can beat GPT-4 but suffer off-target translation due to error propagation in decoding (abstract). Relevant failure modes for long outputs.


### I. Degeneration, repetition, hallucination, exposure bias, off-target

- **arora2022exposure** - Arora et al. (2022). *Why Exposure Bias Matters: An Imitation Learning Perspective of Error Accumulation in Language Generation*. arXiv:2204.01171 (2022-04-03). <https://arxiv.org/abs/2204.01171> [arXiv comment: Accepted in Findings of ACL 2022. v2: Equation 7 updated, typo fixes]
  Findings of ACL 2022. Exposure bias leads to error accumulation that perplexity fails to capture, and results in poor generation quality (abstract). Motivates on-policy RL for long outputs.

- **bengio2015scheduled** - Bengio et al. (2015). *Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks*. arXiv:1506.03099 (2015-06-09). <https://arxiv.org/abs/1506.03099>
  Scheduled sampling: train/test discrepancy of conditioning on gold vs own tokens. Original exposure-bias mitigation.

- **dale2023detecting** - Dale et al. (2022). *Detecting and Mitigating Hallucinations in Machine Translation: Model Internal Workings Alone Do Well, Sentence Similarity Even Better*. arXiv:2212.08597 (2022-12-16). <https://arxiv.org/abs/2212.08597>
  Hallucination detection from the translation model's own internals (percentage of source contribution) and, per the title, sentence similarity does even better. Supports using an embedding-similarity guard against detached outputs.

- **guerreiro2023hallucinations** - Guerreiro et al. (2023). *Hallucinations in Large Multilingual Translation Models*. arXiv:2303.16104 (2023-03-28). <https://arxiv.org/abs/2303.16104>
  Comprehensive hallucination analysis in the M2M family and ChatGPT across diverse translation scenarios (abstract); fills the gap beyond small bilingual high-resource models. Cite for hallucination in massively multilingual MT.

- **guerreiro2023needle** [core] - Guerreiro et al. (2022). *Looking for a Needle in a Haystack: A Comprehensive Study of Hallucinations in Neural Machine Translation*. arXiv:2208.05309 (2022-08-10). <https://arxiv.org/abs/2208.05309> [arXiv comment: Accepted at EACL23 (main)]
  EACL 2023. Annotates >3.4k sentences with hallucinations/critical errors in natural settings, revisits detectors and proposes glass-box uncertainty-based detectors (abstract). Hallucination taxonomy and detection.

- **holtzman2020curious** [core] - Holtzman et al. (2019). *The Curious Case of Neural Text Degeneration*. arXiv:1904.09751 (2019-04-22). <https://arxiv.org/abs/1904.09751> [arXiv comment: Published in ICLR 2020]
  Neural text degeneration (ICLR 2020): likelihood-trained models decode to dull/repetitive text under maximisation; introduces nucleus sampling.

- **li2023repetition** - Li et al. (2023). *Repetition In Repetition Out: Towards Understanding Neural Text Degeneration from the Data Perspective*. arXiv:2310.10226 (2023-10-16). <https://arxiv.org/abs/2310.10226> [arXiv comment: Accepted to NeurIPS 2023]
  Repetition In Repetition Out (NeurIPS 2023): degeneration correlated with repetition present in training data; sentence-level data fix. Relevant to repetitive references/synthetic data.

- **mahaut2025repetitions** - Mahaut et al. (2025). *Repetitions are not all alike: distinct mechanisms sustain repetition in language models*. arXiv:2504.01100 (2025-04-01). <https://arxiv.org/abs/2504.01100>
  Repetitions are not all alike: natural and in-context-induced repetition rely on different mechanisms. Cite for repetition analysis.

- **muller2020domain** - Müller et al. (2019). *Domain Robustness in Neural Machine Translation*. arXiv:1911.03109 (2019-11-08). <https://arxiv.org/abs/1911.03109> [arXiv comment: V2: AMTA camera-ready]
  Domain robustness (AMTA): NMT is fluent but inadequate in unknown domains; hallucination is a key reason for low domain robustness (abstract).

- **raunak2021curious** [core] - Raunak et al. (2021). *The Curious Case of Hallucinations in Neural Machine Translation*. arXiv:2104.06683 (2021-04-14). <https://arxiv.org/abs/2104.06683> [arXiv comment: Accepted to NAACL 2021]
  NAACL 2021. Links hallucinations to corpus-level noise patterns and shows hallucination amplification in back-translation and sequence-level knowledge distillation (abstract). Directly relevant to synthetic/teacher data pipelines.

- **welleck2020unlikelihood** [core] - Welleck et al. (2019). *Neural Text Generation with Unlikelihood Training*. arXiv:1908.04319 (2019-08-12). <https://arxiv.org/abs/1908.04319> [arXiv comment: Sean Welleck and Ilia Kulikov contributed equally]
  Unlikelihood training: penalise repeated tokens/sentences during training to reduce repetition.

- **xu2022loop** [core] - Xu et al. (2022). *Learning to Break the Loop: Analyzing and Mitigating Repetitions for Neural Text Generation*. arXiv:2206.02369 (2022-06-06). <https://arxiv.org/abs/2206.02369> [arXiv comment: Accepted by NeurIPS 2022. Code is released at this https URL]
  Learning to Break the Loop (NeurIPS 2022): sentence-level repetition loops under greedy decoding are self-reinforcing; proposes DITTO. Repetition mechanism citation.

- **zhang2020improving** - Zhang et al. (2020). *Improving Massively Multilingual Neural Machine Translation and Zero-Shot Translation*. arXiv:2004.11867 (2020-04-24). <https://arxiv.org/abs/2004.11867> [arXiv comment: ACL2020]
  ACL 2020. Identifies off-target translation (translating into a wrong target language) as the major source of poor zero-shot performance in massively multilingual NMT; proposes random online back-translation (abstract). Cite for wrong-language outputs.


### J. Small models, multilinguality, tokenisation, base models

- **ahia2023languages** - Ahia et al. (2023). *Do All Languages Cost the Same? Tokenization in the Era of Commercial Language Models*. arXiv:2305.13707 (2023-05-23). <https://arxiv.org/abs/2305.13707>
  Analyses the tokenisation-driven cost and utility disparity across languages in commercial LLM APIs (abstract). Companion to Petrov et al.

- **alabi2022adapting** - Alabi et al. (2022). *Adapting Pre-trained Language Models to African Languages via Multilingual Adaptive Fine-Tuning*. arXiv:2204.06487 (2022-04-13). <https://arxiv.org/abs/2204.06487> [arXiv comment: Accepted to COLING 2022]
  AfroXLM-R: MAFT adapts XLM-R to 17 African languages (COLING 2022). Backbone of AfriCOMET; coverage defines metric reliability.

- **allal2025smollm2** - Allal et al. (2025). *SmolLM2: When Smol Goes Big -- Data-Centric Training of a Small Language Model*. arXiv:2502.02737 (2025-02-04). <https://arxiv.org/abs/2502.02737>
  SmolLM2: data-centric training of small LMs (up to 1.7B). Small-model comparator.

- **arivazhagan2019massively** - Arivazhagan et al. (2019). *Massively Multilingual Neural Machine Translation in the Wild: Findings and Challenges*. arXiv:1907.05019 (2019-07-11). <https://arxiv.org/abs/1907.05019>
  Massively multilingual NMT in the wild (103 languages): temperature-based sampling and capacity trade-offs. Source of temperature sampling for language balance.

- **chang2024multilinguality** [core] - Chang et al. (2023). *When Is Multilinguality a Curse? Language Modeling for 250 High- and Low-Resource Languages*. arXiv:2311.09205 (2023-11-15). <https://arxiv.org/abs/2311.09205>
  When Is Multilinguality a Curse?: over 10,000 models for >250 languages, up to 45M parameters; in moderation, adding multilingual data helps low-resource language modelling (equivalent to up to 33% more low-resource data) with benefit depending on syntactic similarity (abstract). Evidence on capacity/transfer limits at small scale.

- **chung2023unimax** - Chung et al. (2023). *UniMax: Fairer and more Effective Language Sampling for Large-Scale Multilingual Pretraining*. arXiv:2304.09151 (2023-04-18). <https://arxiv.org/abs/2304.09151>
  UniMax: language sampling with explicit epoch caps to avoid overfitting tail languages. Used in our sampling design.

- **comanici2025gemini25** - Comanici et al. (2025). *Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities*. arXiv:2507.06261 (2025-07-07). <https://arxiv.org/abs/2507.06261> [arXiv comment: 72 pages, 17 figures]
  Gemini 2.5 report. Cite when Gemini is used as translation teacher.

- **conneau2020unsupervised** - Conneau et al. (2019). *Unsupervised Cross-lingual Representation Learning at Scale*. arXiv:1911.02116 (2019-11-05). <https://arxiv.org/abs/1911.02116> [arXiv comment: ACL 2020 (+ updated results)]
  XLM-R: 100-language MLM; origin of the 'curse of multilinguality' observation and the encoder behind COMET/CometKiwi.

- **dang2024ayaexpanse** - Dang et al. (2024). *Aya Expanse: Combining Research Breakthroughs for a New Multilingual Frontier*. arXiv:2412.04261 (2024-12-05). <https://arxiv.org/abs/2412.04261>
  Aya Expanse 8B/32B: combines data arbitrage, multilingual preference training and model merging (abstract). Related multilingual post-training recipe.

- **fernandes2023scaling** - Fernandes et al. (2023). *Scaling Laws for Multilingual Neural Machine Translation*. arXiv:2302.09650 (2023-02-19). <https://arxiv.org/abs/2302.09650> [arXiv comment: 19 pages, 20 figures]
  Mixing rates only change the multiplicative factor of the scaling law; same exponent across mixtures; effective parameters per language pair (abstract). Capacity allocation across 77 targets.

- **gemma3270mit_card** [core] - Google (accessed 2026). *google/gemma-3-270m-it model card*. HF model card. <https://huggingface.co/google/gemma-3-270m-it>
  Hugging Face card for google/gemma-3-270m-it (page read): 32K context, 6T training tokens, cutoff Aug 2024, IFEval 51.2; family-wide 'over 140 languages'; no FLORES/WMT24++ results at 270M; evaluations used only English prompts.

- **gemmateam2024gemma2** - Gemma Team et al. (2024). *Gemma 2: Improving Open Language Models at a Practical Size*. arXiv:2408.00118 (2024-07-31). <https://arxiv.org/abs/2408.00118>
  Gemma 2: 2B-27B with local/global attention interleaving. Predecessor reference.

- **gemmateam2025gemma3** [core] - Gemma Team et al. (2025). *Gemma 3 Technical Report*. arXiv:2503.19786 (2025-03-25). <https://arxiv.org/abs/2503.19786>
  Gemma 3 technical report: 1B-27B, multilingual coverage widened, >=128K context for larger sizes; 270M variant released later. Base-model reference.

- **haque2026robustrl** [core] - Haque et al. (2026). *Towards Robust Reinforcement Learning for Small-Scale Language Model Agents*. arXiv:2607.25091 (2026-07-27). <https://arxiv.org/abs/2607.25091> [arXiv comment: Proceedings of the 2026 IEEE International Conference on Systems, Man, and Cybernetics (SMC), Bellev]
  Robust RL for 70-500M LMs (PPO on Pythia-70M/160M/410M, SmolLM2-135M/360M): three failure modes (silent LoRA freezing, bf16 importance-ratio overflow, reward-model-driven collapse) and a fluency threshold PPL<20 (abstract). Closest evidence for RL stability at 270M scale (not MT).

- **he2024scalinglaws** - He et al. (2024). *Scaling Laws for Multilingual Language Models*. arXiv:2410.12883 (2024-10-15). <https://arxiv.org/abs/2410.12883>
  Hypothesis that loss for each language family depends solely on its own sampling ratio; derives power-law relationships for multilingual mixtures (abstract). Guidance for language sampling.

- **kudugunta2023madlad** - Kudugunta et al. (2023). *MADLAD-400: A Multilingual And Document-Level Large Audited Dataset*. arXiv:2309.04662 (2023-09-09). <https://arxiv.org/abs/2309.04662> [arXiv comment: Preprint]
  MADLAD-400 (3T tokens, 419 languages) and a 10.7B MT model; data used as monolingual source in TranslateGemma. Data source.

- **lacombe2025introducing** [core] - Olivier Lacombe (2025). *Introducing Gemma 3 270M: The compact model for hyper-efficient AI*. Google Developers Blog, 2025-08-14. <https://developers.googleblog.com/introducing-gemma-3-270m/>
  Official blog (2025-08-14; Lacombe, Kenealy et al.): Gemma 3 270M, 270M params (about 170M embeddings, 100M transformer blocks), 256k-token vocabulary, intended for task-specific fine-tuning, not complex conversation. Primary source for the model size breakdown.

- **li2025smallstruggle** - Li et al. (2025). *Small Models Struggle to Learn from Strong Reasoners*. arXiv:2502.12143 (2025-02-17). <https://arxiv.org/abs/2502.12143>
  Small Model Learnability Gap: <=3B models do not consistently benefit from long CoT / distillation from strong reasoners (abstract). Evidence for capacity limits in small models.

- **liu2025muon** - Liu et al. (2025). *Muon is Scalable for LLM Training*. arXiv:2502.16982 (2025-02-24). <https://arxiv.org/abs/2502.16982>
  Muon optimizer scales to larger LMs with weight decay and update-scale tuning. Cite for optimiser choice.

- **nllbteam2022nllb** [core] - NLLB Team et al. (2022). *No Language Left Behind: Scaling Human-Centered Machine Translation*. arXiv:2207.04672 (2022-07-11). <https://arxiv.org/abs/2207.04672> [arXiv comment: 190 pages]
  NLLB-200: 200-language MT; origin of FLORES-200 and mined data. Baseline and data source; licence is CC-BY-NC (see nllb200_card).

- **petrov2023tokenizer** [core] - Petrov et al. (2023). *Language Model Tokenizers Introduce Unfairness Between Languages*. arXiv:2305.15425 (2023-05-17). <https://arxiv.org/abs/2305.15425> [arXiv comment: Published at NeurIPS 2023, Project webpage: this https URL, Code: this https URL]
  NeurIPS 2023. The same text can have up to 15 times different tokenisation lengths across languages, persisting even for multilingual tokenizers (abstract). Matters for token budgets: a 2,048-token window holds far less text in some African languages.

- **robinson2023chatgpt** - Robinson et al. (2023). *ChatGPT MT: Competitive for High- (but not Low-) Resource Languages*. arXiv:2309.07423 (2023-09-14). <https://arxiv.org/abs/2309.07423> [arXiv comment: 27 pages, 9 figures, 14 tables]
  FLORES-200, 204 languages: GPT models lag traditional MT for low-resource languages, under-performing for 84.1% of languages covered (abstract). Evidence of LLM weakness on low-resource MT.

- **salamanca2026tinyaya** - Salamanca et al. (2026). *Tiny Aya: Bridging Scale and Multilingual Depth*. arXiv:2603.11510 (2026-03-12). <https://arxiv.org/abs/2603.11510>
  Tiny Aya (Mar 2026): 3.35B multilingual small model trained on 70 languages; abstract claims state-of-the-art translation quality among its class. Small multilingual model comparator.

- **tonja2024inkubalm** - Tonja et al. (2024). *InkubaLM: A small language model for low-resource African languages*. arXiv:2408.17024 (2024-08-30). <https://arxiv.org/abs/2408.17024>
  InkubaLM: 0.4B small LM for African languages (Aug 2024); not RL-trained. Closest small African LM; cite for small-model African NLP.

- **ustun2024aya** - Üstün et al. (2024). *Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model*. arXiv:2402.07827 (2024-02-12). <https://arxiv.org/abs/2402.07827>
  Aya model: instruction-tuned multilingual model for 101 languages. Multilingual instruction baseline.

- **xu2025phi4mini** - Xu et al. (2025). *Phi-4-Mini-Reasoning: Exploring the Limits of Small Reasoning Language Models in Math*. arXiv:2504.21233 (2025-04-30). <https://arxiv.org/abs/2504.21233>
  Phi-4-Mini-Reasoning: four-step recipe for small LMs (mid-training on distilled long-CoT, SFT, rollout DPO, RL with verifiable reward) because small models have limited capacity (abstract). Evidence that RL on small models needs careful scaffolding.

- **yang2025qwen3** - Yang et al. (2025). *Qwen3 Technical Report*. arXiv:2505.09388 (2025-05-14). <https://arxiv.org/abs/2505.09388>
  Qwen3 technical report: dense and MoE series, multilingual coverage. Comparator family.

- **yue2025doesrl** - Yue et al. (2025). *Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?*. arXiv:2504.13837 (2025-04-18). <https://arxiv.org/abs/2504.13837> [arXiv comment: 31 pages, 27 figures; journal-ref: NeurIPS 2025 Oral; ICML 2025 AI4MATH workshop best paper]
  NeurIPS 2025 oral. RLVR-trained models beat base models at small k, but base models achieve higher pass@k at large k; RLVR does not elicit fundamentally new reasoning patterns (abstract). Caution about what RL adds beyond the base/SFT model.

- **zhu2024multilingual** - Zhu et al. (2023). *Multilingual Machine Translation with Large Language Models: Empirical Results and Analysis*. arXiv:2304.04675 (2023-04-10). <https://arxiv.org/abs/2304.04675> [arXiv comment: Accepted to Findings of NAACL 2024]
  Findings of NAACL 2024. Evaluates eight LLMs; GPT-4 beat NLLB in 40.91% of directions but remains well behind Google Translate, especially on low-resource languages (abstract).


### K. Transfer, forgetting, merging, RL-vs-SFT

- **biderman2024lora** [core] - Biderman et al. (2024). *LoRA Learns Less and Forgets Less*. arXiv:2405.09673 (2024-05-15). <https://arxiv.org/abs/2405.09673> [arXiv comment: Final version with new experiments and analyses, as accepted to Transactions on Machine Learning Res]
  LoRA Learns Less and Forgets Less: on code and math, LoRA underperforms full fine-tuning but forgets less. Cite for the forgetting-vs-learning trade-off.

- **chen2025retaining** - Chen et al. (2025). *Retaining by Doing: The Role of On-Policy Data in Mitigating Forgetting*. arXiv:2510.18874 (2025-10-21). <https://arxiv.org/abs/2510.18874> [journal-ref: Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026]
  RL leads to less forgetting than SFT with comparable or higher target performance across Llama and Qwen families (abstract); attributes it to on-policy data. Supports RL-after-SFT forgetting arguments.

- **chu2025sft** [core] - Chu et al. (2025). *SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training*. arXiv:2501.17161 (2025-01-28). <https://arxiv.org/abs/2501.17161> [arXiv comment: Website at this https URL]
  RL, especially with outcome-based reward, generalises to unseen rule-based and visual variants while SFT tends to memorise (abstract).

- **hu2022lora** - Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685 (2021-06-17). <https://arxiv.org/abs/2106.09685> [arXiv comment: Draft V2 includes better baselines, experiments on GLUE, and more on adapter latency]
  LoRA low-rank adaptation. Base for PEFT discussion.

- **ilharco2023editing** - Ilharco et al. (2022). *Editing Models with Task Arithmetic*. arXiv:2212.04089 (2022-12-08). <https://arxiv.org/abs/2212.04089> [arXiv comment: In Proceedings of the 11th International Conference on Learning Representations (ICLR 2023)]
  Task arithmetic: task vectors added/negated in weight space to edit behaviour. Basis for merging.

- **kalajdzievski2024scaling** - Kalajdzievski (2024). *Scaling Laws for Forgetting When Fine-Tuning Large Language Models*. arXiv:2401.05605 (2024-01-11). <https://arxiv.org/abs/2401.05605>
  Forgetting scales as a shifted power law in parameters tuned and update steps; strong inverse linear relation between fine-tuning performance and forgetting with LoRA; not avoided by early stopping (abstract).

- **kirkpatrick2017overcoming** - Kirkpatrick et al. (2016). *Overcoming catastrophic forgetting in neural networks*. arXiv:1612.00796 (2016-12-02). <https://arxiv.org/abs/1612.00796>
  EWC: elastic weight consolidation to overcome catastrophic forgetting. Classical reference.

- **lin2024mitigating** [core] - Lin et al. (2023). *Mitigating the Alignment Tax of RLHF*. arXiv:2309.06256 (2023-09-12). <https://arxiv.org/abs/2309.06256> [arXiv comment: EMNLP 2024 Main]
  EMNLP 2024. Model averaging (interpolating pre- and post-RLHF weights) achieves the strongest alignment-forgetting Pareto front among competing methods (abstract). Supports WiSE-style interpolation after RL.

- **luo2025empirical** - Luo et al. (2023). *An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning*. arXiv:2308.08747 (2023-08-17). <https://arxiv.org/abs/2308.08747>
  Catastrophic forgetting observed in LLMs from 1B to 7B during continual instruction tuning, and severity intensifies with scale in that range (abstract).

- **mukherjee2025subnetworks** - Mukherjee et al. (2025). *Reinforcement Learning Finetunes Small Subnetworks in Large Language Models*. arXiv:2505.11711 (2025-05-16). <https://arxiv.org/abs/2505.11711> [arXiv comment: NeuRIPS 2025]
  RL updates only 5-30% of parameters (a subnetwork) across 7 RL algorithms (incl. PPO, GRPO, DPO) and 10 LLMs; fine-tuning the subnetwork alone recovers performance (abstract). Relevant to update sparsity and merging.

- **rolnick2019experience** - Rolnick et al. (2018). *Experience Replay for Continual Learning*. arXiv:1811.11682 (2018-11-28). <https://arxiv.org/abs/1811.11682> [arXiv comment: NeurIPS 2019]
  Experience replay for continual learning (NeurIPS 2019). Replay baseline reference.

- **shenfeld2025razor** [core] - Shenfeld et al. (2025). *RL's Razor: Why Online Reinforcement Learning Forgets Less*. arXiv:2509.04259 (2025-09-04). <https://arxiv.org/abs/2509.04259>
  RL's Razor: on-policy RL is implicitly biased toward KL-minimal solutions; forgetting is governed by KL shift from the base policy (abstract).

- **wortsman2022robust** [core] - Wortsman et al. (2021). *Robust fine-tuning of zero-shot models*. arXiv:2109.01903 (2021-09-04). <https://arxiv.org/abs/2109.01903> [arXiv comment: CVPR 2022]
  WiSE-FT: weight-space interpolation between pretrained and fine-tuned models retains robustness. Basis for interpolating back toward the instruct model.

- **wortsman2022soups** - Wortsman et al. (2022). *Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time*. arXiv:2203.05482 (2022-03-10). <https://arxiv.org/abs/2203.05482> [arXiv comment: ICML 2022. The last three authors contributed equally]
  Model soups: averaging fine-tuned weights improves accuracy without extra inference cost.

- **yadav2023ties** - Yadav et al. (2023). *TIES-Merging: Resolving Interference When Merging Models*. arXiv:2306.01708 (2023-06-02). <https://arxiv.org/abs/2306.01708> [arXiv comment: Published at NeurIPS 2023, 23 Pages, 13 Figures, 14 Tables]
  TIES-Merging: resolves interference (sign conflicts, redundant parameters) when merging models.


### L. Teachers, distillation, synthetic data

- **gerstgrasser2024collapse** - Gerstgrasser et al. (2024). *Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data*. arXiv:2404.01413 (2024-04-01). <https://arxiv.org/abs/2404.01413>
  Replacing real data by synthetic data each generation leads to collapse, but accumulating data avoids it (abstract). Mitigation when mixing teacher data.

- **hinton2015distilling** - Hinton et al. (2015). *Distilling the Knowledge in a Neural Network*. arXiv:1503.02531 (2015-03-09). <https://arxiv.org/abs/1503.02531> [arXiv comment: NIPS 2014 Deep Learning Workshop]
  Knowledge distillation (NIPS 2014 workshop). Background.

- **kim2016sequence** [core] - Kim et al. (2016). *Sequence-Level Knowledge Distillation*. arXiv:1606.07947 (2016-06-25). <https://arxiv.org/abs/1606.07947> [arXiv comment: EMNLP 2016]
  Sequence-level KD for NMT (EMNLP 2016): train student on teacher outputs. Foundation for synthetic teacher data.

- **sennrich2016improving** - Sennrich et al. (2015). *Improving Neural Machine Translation Models with Monolingual Data*. arXiv:1511.06709 (2015-11-20). <https://arxiv.org/abs/1511.06709> [arXiv comment: accepted to ACL 2016; new section on effect of back-translation quality]
  Back-translation (ACL 2016): synthetic parallel data from target-side monolingual data. Cite for synthetic bitext.

- **shumailov2023curse** [core] - Shumailov et al. (2023). *The Curse of Recursion: Training on Generated Data Makes Models Forget*. arXiv:2305.17493 (2023-05-27). <https://arxiv.org/abs/2305.17493> [arXiv comment: Fixed typos in eqn 4,5]
  Use of model-generated content in training causes irreversible defects in which the tails of the original distribution disappear (abstract); arXiv version, later journal version not verified here. Risk for synthetic low-resource data.


### M. Licensing, ethics, documentation, contamination

- **agic2019jw300** - Agic and Vulic (2019). *JW300: A Wide-Coverage Parallel Corpus for Low-Resource Languages*. Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics. <https://aclanthology.org/P19-1310/>
  JW300: wide-coverage parallel corpus for low-resource languages (ACL 2019). Licence concerns arise for downstream redistribution.

- **bender2018data** - Bender and Friedman (2018). *Data Statements for Natural Language Processing: Toward Mitigating System Bias and Enabling Better Science*. Transactions of the Association for Computational Linguistics. <https://aclanthology.org/Q18-1041/>
  Data Statements (TACL 2018). Documentation practice for NLP data.

- **gebru2021datasheets** [core] - Gebru et al. (2018). *Datasheets for Datasets*. arXiv:1803.09010 (2018-03-23). <https://arxiv.org/abs/1803.09010> [arXiv comment: Published in CACM in December, 2021]
  Datasheets for Datasets (CACM 2021). Documentation practice.

- **gemma_terms** [core] - Google (2026). *Gemma Terms of Use*. web page (last modified 2026-04-01). <https://ai.google.dev/gemma/terms>
  Gemma Terms of Use (page shows 'Last modified: April 1, 2026'). Governs use/distribution of Gemma weights and derivatives.

- **jacovi2023stop** [core] - Jacovi et al. (2023). *Stop Uploading Test Data in Plain Text: Practical Strategies for Mitigating Data Contamination by Evaluation Benchmarks*. arXiv:2305.10160 (2023-05-17). <https://arxiv.org/abs/2305.10160> [arXiv comment: Accepted to EMNLP 2023]
  EMNLP 2023. Three strategies: encrypt public test data and license against derivatives, demand training-exclusion controls from closed API holders, and avoid data that appears with its solution online (abstract).

- **longpre2023data** [core] - Longpre et al. (2023). *The Data Provenance Initiative: A Large Scale Audit of Dataset Licensing & Attribution in AI*. arXiv:2310.16787 (2023-10-25). <https://arxiv.org/abs/2310.16787> [arXiv comment: 30 pages (18 main), 6 figures, 5 tables]
  Audit of 1800+ text datasets tracing source, licence conditions and use; shows closed datasets monopolise lower-resource languages and creative tasks (abstract). Core licence-provenance citation.

- **mitchell2019modelcards** [core] - Mitchell et al. (2018). *Model Cards for Model Reporting*. arXiv:1810.03993 (2018-10-05). <https://arxiv.org/abs/1810.03993> [journal-ref: FAT* '19: Conference on Fairness, Accountability, and Transparency, January 29--31, 2019, Atlanta, G]
  Model Cards (FAT* 2019). Documentation practice.

- **nekoto2020participatory** - Nekoto et al. (2020). *Participatory Research for Low-resourced Machine Translation: A Case Study in African Languages*. arXiv:2010.02353 (2020-10-05). <https://arxiv.org/abs/2010.02353> [arXiv comment: Findings of EMNLP 2020; updated benchmarks]
  Participatory research for low-resourced MT in African languages (Findings of EMNLP 2020). Community-driven methodology.

- **nllb200_card** [core] - facebook (HF organisation) (accessed 2026). *facebook/nllb-200-distilled-600M model card*. HF model card. <https://huggingface.co/facebook/nllb-200-distilled-600M>
  Hugging Face card for facebook/nllb-200-distilled-600M shows licence cc-by-nc-4.0. Non-commercial; constrains use of NLLB-generated data/models.

- **sainz2023nlp** [core] - Sainz et al. (2023). *NLP Evaluation in trouble: On the Need to Measure LLM Data Contamination for each Benchmark*. arXiv:2310.18018 (2023-10-27). <https://arxiv.org/abs/2310.18018> [arXiv comment: Accepted at EMNLP2024-Findings]
  Position paper: defines levels of data contamination and argues for a community effort to document it per benchmark (abstract). Accepted at Findings of EMNLP 2024.


## Unverified / not found (left out of the bib)

- Williams (1992) REINFORCE paper: not searched on a verifiable source here; cite only after checking.
- Kool, van Hoof, Welling (2019) 'Buy 4 REINFORCE Samples, Get a Baseline for Free!': search returned no page for it; unverified.
- Original Bleualign paper (Sennrich and Volk 2010, AMTA): not located; the 2011 NODALIDA paper (sennrich2011iterative) is verified instead.
- A standalone 'GlotLID v3' paper: none found. v3 is documented only in the GlotCC paper (kargaran2024glotcc) and the HF model card (glotlid_card).
- 'TransR1' as a separate paper: not found; R1-T1 (he2025r1t1) is the verified match.
- 'BLEU meets COMET' and 'pathological translations in COMET QE' as titled in the task brief: no paper with those exact titles was found. Closest verified: Amrhein and Sennrich (amrhein2022identifying), Yan et al. (yan2023bleurt), Zouhar et al. (zouhar2024pitfalls).
- 'Lost in translation: long inputs' as an exact title: not found; closest verified: wang2024benchmarking, wang2025segale, wang2026doc2frc, ramos2025docblocks.
- 'Understanding in-context repetition' exact title: not found; mahaut2025repetitions is the nearest verified; a related repetition-neurons paper (arXiv 2507.07810, Doan et al.) was verified but left out as peripheral.
- Gemma 3n and SmolLM3 reports: search returned nothing for them; omitted. (SmolLM2 and Qwen3 are verified.)
- Masakhane principles / participatory-research principles document: only the participatory MT case study (nekoto2020participatory) is verified; no separate 'principles' document was verified.
- JW300 licence concerns: the JW300 paper is verified (agic2019jw300) but I did not find a verified source for the licence/ethics controversy itself; do not assert it without one.
- Nature/journal version of 'curse of recursion': only the arXiv version is verified.
- Official Gemma 3 270M technical report: none exists in what I found; the official blog post and HF model card are used instead (lacombe2025introducing, gemma3270mit_card).
- Full-text details of Seed-X, Hunyuan-MT and HY-MT1.5 RL recipes (GRPO etc.) were not read; the notes only restate abstracts / search summaries.
- Items verified (arXiv title/authors/date) but intentionally excluded as peripheral: 1805.10627 (Kreutzer et al., bandit feedback for seq2seq RL); 2609.06048 (adversarial text rewriting for MT via GRPO), 2606.08011 (RL for source rewriting in MT), 2608.03077 (PAMT), 2602.16469, 2609.39321 (GRPO dynamics of 1.5B-7B models), 2510.03515 (RAPID), 2605.30789.

## NOVELTY CHECK

**Claim under test.** Nobody has trained a ~270M-parameter model with RL to translate into low-resource African languages and extended it to whole-document translation up to 2,048 tokens.

**Method.** About 20 web searches (dates of search: 2026-10-22) with combinations of: GRPO/RL + machine translation + low-resource/African/Nigerian languages + small/tiny/270M/sub-1B model + document-level/long-document + length curriculum + reward hacking; plus targeted fetches of every promising hit (arXiv abstract pages, and full text for WALAR, EA-RLVR, TranslateGemma). No paper matching the full claim was found. Several partially overlapping works exist and the claim should be phrased narrowly.

### Evidence AGAINST (prior art overlapping parts of the claim)

1. **Small-model RL for MT exists, but encoder-decoder and above 270M.**
   - Attia and Aji, *Improving Low-Resource Machine Translation via Round-Trip Reinforcement Learning*, arXiv 2601.12535, 18 Jan 2026 (v3 17 Mar 2026): RL fine-tuning of NLLB 600M and 1.3B with a chrF++/BLEU round-trip reward; improvements for Central Aymara, Friulian, Wolof, Dyula, Bhojpuri, Russian. <https://arxiv.org/abs/2601.12535> (Wolof and Dyula are African; search summary says GRPO is the optimiser; the abstract says RL fine-tuning.)
   - Garcia-Estrada, Escolano, Fonallosa, *Reference-Free Reinforcement Learning Fine-Tuning for MT: A Seq2Seq Perspective*, arXiv 2605.15976, 15 May 2026: GRPO on NLLB-200 600M/1.3B, hybrid reference-free reward (LaBSE + COMET-Kiwi), 13 languages; competitive with 3-epoch SFT on morphologically complex languages. <https://arxiv.org/abs/2605.15976> Notably the abstract states RL for MT has 'largely targeted decoder-only LLMs at >=7B', which is consistent with our novelty claim for decoder-only small models but not for 600M seq2seq.
   - Tencent HY-MT1.5-1.8B, arXiv 2512.24092, 30 Dec 2025: RL is part of the pipeline for a 1.8B translation model (about 6.7x our 270M). <https://arxiv.org/abs/2512.24092>
   - WALAR (arXiv 2603.13045, 13 Mar 2026; Findings of ACL 2026): reports a preliminary QE-only-reward run on Qwen2.5-0.5B-Instruct in which the policy produced a 'self-generated reference' in 100% of cases (full text). So a sub-1B decoder-only RL-for-MT attempt has been reported, as a failed preliminary run, not as a result. <https://arxiv.org/abs/2603.13045>
   - Haque et al., arXiv 2607.25091, 27 Jul 2026: PPO on 70-500M LMs (Pythia, SmolLM2) with reward-model-driven collapse failure modes - not MT. <https://arxiv.org/abs/2607.25091>
2. **R1-style / GRPO RL for MT at 3B-14B, high-resource focus.** MT-R1-Zero (arXiv 2504.10160, 14 Apr 2025; 3B/7B), SSR-Zero (arXiv 2505.16637, 22 May 2025; 7B), DeepTrans (2504.10187), ExTrans (2505.12996), R1-T1 (2502.19735), TwT (2607.29287; 7B/14B; 59 unseen languages), Seed-X (2507.13618; 7B), Hunyuan-MT (2509.05209; 7B). TranslateGemma (arXiv 2601.09012, 13 Jan 2026) is the closest large-lab analogue: Gemma 3 4B/12B/27B, SFT then RL with a reward ensemble (MetricX-24-XXL-QE, Gemma-AutoMQM-QE, chrF, naturalness autorater, generalist RM) - so 'RL with a quality-estimation reward ensemble on Gemma 3' is NOT novel; only the 270M size, the African 77-language coverage, and the document extension could be.
3. **Reward-hacking defences for MT-RL are already studied.** WALAR documents self-generated reference, non-translation, over/under-translation and wrong-language hacking of a QE reward and adds word-alignment and language-alignment rewards (multilingual, incl. Swahili, 101 languages, 8B and 4B models). EA-RLVR (arXiv 2604.16881, 18 Apr 2026) reports length explosion and enumeration hacking without a length gate (reward zeroed above 2x average reference length). Reward-ensemble and length-bias literature (Eisenstein, Coste, Singhal, Park) predates. Our 'reward-hacking defences' contribution should therefore be framed as extension/validation to African languages, small models and documents, not as first discovery.
4. **African-language + RL (not translation quality) and African small-model MT.**
   - Translate-R1 (arXiv 2606.06835, 5 Jun 2026): RL (confidence-gated GSPO) on Qwen3-4B over 22 languages incl. Hausa, Swahili, Yoruba, Zulu (as listed in search summary) - tool-use policy, not translation training.
   - CURE-Med (arXiv 2601.13262, Jan 2026; ACL 2026 oral): GRPO with curriculum over resource tiers for Amharic/Yoruba/Swahili etc. - medical reasoning, language-consistency reward.
   - TranslatePsy-AfriSLM (arXiv 2608.18655, 19 Aug 2026; EMNLP 2026 main): 19 Sub-Saharan African languages, SLMs from 0.8B, QE filtering + synthetic data; abstract does not mention RL or documents. <https://arxiv.org/abs/2608.18655> This is the nearest competitor on 'small model + African languages' and appeared two months ago; it must be cited and compared.
   - InkubaLM (arXiv 2408.17024, Aug 2024): 0.4B African LM, not RL.
   - RL with chrF reward for unseen-language translation from context (arXiv 2606.06428, 4 Jun 2026; Qwen3-4B-Base per search summary) shows a chrF-only reward works for low-resource MT.
5. **Document-level translation with RL or long outputs.**
   - FormatRL (arXiv 2512.05100, 4 Dec 2025; IJCNLP-AACL 2025 oral): GRPO on document-level structured XML/HTML translation with TreeSim and Node-chrF rewards (SAP documentation). Closest 'GRPO + document-level MT' paper; structure-focused, not small-model, not African. <https://arxiv.org/abs/2512.05100>
   - Loong (arXiv 2605.30274, 28 May 2026): long-document translation agent whose context policy is optimised with RL from sampled trajectories (agentic, not end-to-end).
   - Pre-LLM RL for document MT: Kang et al. 2020 (2010.04314), Unanue et al. 2020 (2010.03732).
   - TranslateGemma (full text): its synthetic SFT data include 'text blobs of up to 512 tokens', and the RL stage reuses the SFT translation data except GATITOS and SMOL - so multi-sentence translation up to roughly 512 tokens is already inside an RL mixture at 4B+ (it is not an RL study of documents per se).
   - Long-output RL outside MT: LongWriter-Zero (arXiv 2506.18841; ICLR 2026 oral) - RL for ultra-long writing; reports repetition/length hacks.
   - Document-level African benchmark exists: AFRIDOC-MT (arXiv 2501.06374, Jan 2025; EMNLP 2025): 5 African languages, finds sentence-trained models generalise poorly to longer documents and LLMs under-generate/repeat/go off-target. Useful as evidence for the problem and as an evaluation set; not RL.
   - Long-document MT failure evidence: SEGALE (2509.17249), Doc2FRC (2609.12674; Sept 2026; SFT, 7B), DocBlocks (2504.12140).
6. **Length curricula under RL** exist for reasoning (FastCuRL 2503.17287, EMNLP 2025; Train Long, Think Short 2508.08940) but none found for MT.

### Evidence FOR novelty (what was searched and not found)

- No hit for: GRPO/RL + Gemma-3-270M + translation; the only Gemma-3-270M African-language items found were non-RL (a diacritisation fine-tune on the Hugging Face org <HF_ORG> - which appears to be the project owner's own org - and a 2026 multilingual-MT paper listing Gemma3-270M FLORES+ scores for non-African pairs, 2602.11961).
- No hit for: RL for MT on any decoder-only model <=1B in African languages; no hit for RL-for-MT covering ~77 African target languages; no hit for end-to-end (single-pass) RL translation of documents up to 2,048 tokens with a small model; no hit for an output-length curriculum (sentence -> paragraph -> document) in RL for MT.
- The Garcia-Estrada et al. abstract itself says RL for MT 'has largely targeted decoder-only LLMs at >=7B' (2605.15976), supporting a gap for small decoder-only models; HY-MT1.5-1.8B is the smallest RL-trained MT LLM found.
- Search recall limits: the search tool is US-only web search; recent (2026) arXiv papers appear in results, but a negative result is not conclusive. Before submission, repeat on Google Scholar, arXiv listing search (cs.CL, 'GRPO' AND 'translation'), ACL Anthology (WMT 2026, EMNLP 2026 findings), and Hugging Face papers; EMNLP 2026 and WMT 2026 papers were appearing during the search window.

### Recommended phrasing

Safe: 'To our knowledge, this is the first study applying GRPO with a quality-estimation reward ensemble to a ~270M-parameter decoder-only model for English-to-African-language translation across 77 targets, and the first to extend RL-trained translation to documents up to 2,048 tokens in a single pass.' Avoid 'first RL for low-resource MT' (Attia and Aji, Garcia-Estrada et al., WALAR), 'first small-model RL for MT' (NLLB-600M work, HY-MT1.5-1.8B), and 'first reward-hacking analysis for MT-RL' (WALAR, EA-RLVR). Cite TranslatePsy-AfriSLM, TranslateGemma, WALAR, Attia and Aji, Garcia-Estrada et al., FormatRL and AFRIDOC-MT as closest work.
