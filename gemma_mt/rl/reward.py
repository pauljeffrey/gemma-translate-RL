# SPDX-License-Identifier: AGPL-3.0-or-later
"""The RL reward: from sampled token ids to one scalar per completion.

    token ids --(cut at first <end_of_turn>/<eos>, drop specials)--> text
              --(strip leading language tag)----------------------> translation
              --(failure checks)---------------------------------->  penalty (floor of the range)
              --(AfriCOMET-QE on (english source, translation))---->  c
              --(per-language shaping curve)----------------------> reward in [-2, +2.25] ("repo") / [-2, +1]

Failure classes, all scored with the SAME penalty = the floor of the shaped range (`-neg_scale`, -2.0 by
default; they never receive a QE score, so markup cannot buy reward):

    no_end_of_turn   no stop id within max_new_tokens (the model does not know how to stop): TRUNCATED, always on
    post_eos_rubbish tokens that are not padding follow the first stop id (eos discipline, D-26; `eos_discipline`)
    dangling_marker  a turn / role marker (`<start_of_turn>`, `<|assistant|>`, a bare `model` / `user` line) inside the answer,
                     before its end token (D-26; `eos_discipline`)
    bad_prefix       leading tag missing / wrong language tag (prefix_policy="require"), or present when "forbid"
    empty            nothing left after stripping
    copy_source      output is (nearly) the English input
    degenerate       word loops / stuttering / absurd compressibility
    length_ratio     output length vs source outside [min_len_ratio, max_len_ratio]
    wrong_language   GlotLID disagrees with the requested language (only when enabled and trusted for that language)
    special_tokens   special-token ids/strings inside the text (only when penalize_special_tokens=True; otherwise
                     they are removed and counted)

Each failure class has its own penalty value (default: the same floor); the three eos classes can be made MORE negative
with `eos_truncated_penalty` / `eos_post_end_penalty` / `eos_dangling_penalty`, never less (see __init__).

Content signal per language (D-38, `content_mode` of the calibration `content` block / `reward.content_modes`):

    qe         the shaped AfriCOMET-QE of each chunk (default; the 42 validated languages)
    blend      (1 - beta) * shaped QE + beta * consensus value, per chunk
    consensus  the consensus value only, no QE call (languages AfriCOMET does not name)

    consensus value = mean sentence chrF++ of the completion against the OTHER valid completions of its GRPO group (rl/consensus.py), mapped linearly
    from [0, 100] to the reward scale; needs the group size (`__call__(samples, group_size=G)`) and >= `consensus_min_valid` valid samples in the group,
    otherwise a blend language falls back to QE only and a consensus language gets the rubric-only value `consensus_skip_value`. The per-chunk penalties
    that have a calibration (length, lexical, diacritics), the omission/extra-text failed chunks and every failure class apply to all modes.

GlotLID (D-37): a completion whose top-1 label is not in the language's accepted label set is `wrong_language` (-2). The label sets, the languages for
which the guard is on (measured human recall >= 0.90) and the minimum text length per language come from the calibration `lid` blocks.

Advantage weights (repo behaviour): a language flagged `usable: false` by the calibration gets weight 0 (QE cannot
tell real from mismatched translations there), otherwise the language's measured reliability in [0, 1].
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from gemma_mt.format import prefix_violation
from gemma_mt.languages import LANGUAGES, canonical_iso
from gemma_mt.rl import confidence as conf
from gemma_mt.rl import consensus as cons
from gemma_mt.rl import guards
from gemma_mt.rl import longdoc
from gemma_mt.rl.chunk_plan import aggregate, plan_chunks
from gemma_mt.rl.reward_shaping import DEFAULT_ANCHORS, FAIL_FLOOR, ContentShaper
from gemma_mt.rl.text import decode_completion, nfc, translation_from

FAILURES = ("no_end_of_turn", "bad_prefix", "empty", "copy_source", "degenerate", "length_ratio", "wrong_language",
            "special_tokens", "post_eos_rubbish", "dangling_marker", "stuffed_output")
EOS_FAILURES = ("no_end_of_turn", "post_eos_rubbish", "dangling_marker")      # D-26: the end-of-turn discipline classes
CONTENT_MODES = ("qe", "blend", "consensus")                                       # D-38


@dataclass
class RewardConfig:
    preset: str = "repo"                       # repo | owner_note   (see reward_shaping.py; Q-01)
    mode: str = "calibrated"                   # calibrated | adaptive
    calibration_path: Optional[str] = None
    neg_scale: float = 2.0
    default_anchors: Tuple[float, float, float] = DEFAULT_ANCHORS
    adaptive_window: int = 2000
    adaptive_warmup: int = 200
    adaptive_ema: float = 0.9
    penalty: Optional[float] = None            # None -> floor of the range = -neg_scale
    prefix_policy: str = "require"             # require | forbid | optional
    copy_jaccard: float = 0.9
    min_len_ratio: float = 0.2
    max_len_ratio: float = 5.0
    repetition_guard: bool = True
    lid_enabled: bool = False                  # D-37: GlotLID guard (labels / on-off / min length per language from the calibration `lid` blocks; languages.py labels without them)
    lid_min_chars: int = 15                    # no LID below this many characters (a language may need more: calibration `lid.min_chars`)
    lid_model_path: Optional[str] = None       # local model_v3.bin (null = cis-lmu/glotlid from the HF cache / Hub); an unusable LID model STOPS the run at construction
    lid_disable: List[str] = field(default_factory=list)   # languages whose guard is switched off whatever the calibration says
    penalize_special_tokens: bool = False
    train_unusable: bool = False               # True: unusable languages keep weight 1 (not recommended)
    max_comet_tokens: int = 510
    # Chunked scoring (owner 2026-10-12): split source and translation into sentences (rl/segments.py, no newline involved),
    # score the i-th source sentence against the i-th output sentence, shape each chunk score, average. When the number
    # of sentences differs the chunks come from a length-based alignment (Gale-Church, segments.py): merged / split sentences
    # become many-to-one beads and an omitted sentence a failed chunk with the penalty; never a shifted pairing.
    chunking: str = "off"                      # off | sentences
    chunk_agg: str = "source_length"           # source_length (weighted by source characters; owner 2026-10-12) | mean
    chunk_align: str = "gale_church"           # gale_church | whole  (what to do when the sentence counts differ)
    # Two blind spots of AfriCOMET-QE measured on Pidgin (owner 2026-10-13; D-24): extra, hallucinated text does not lower the score (same-source
    # accuracy 0.47 = chance), and wrong content of the right length is already ranked correctly (0.94).
    extra_output_penalty: bool = True          # chunked mode: output sentences matched to no source sentence count as a failed chunk (weighted by their size)
    extra_ratio_limit: float = 1.8             # a chunk whose output is longer than this many times its source: the excess characters count as a failed chunk
    length_penalty: float = 0.0                # reward subtracted per log-unit outside the language's human length-ratio window (0 = off)
    length_slack: float = 0.15                 # the human [p05, p95] window is widened by this fraction before the penalty starts
    word_ratio_penalty: float = 0.0            # reward subtracted per log-unit of words(output)/words(source) ABOVE the language's human p99 (calibration `word_ratio`, ContentShaper.word_excess; 0 = off)
    word_ratio_slack: float = 0.15             # the human p99 word ratio is widened by this fraction before the penalty starts
    lexical_penalty: float = 0.0               # reward subtracted per unit of lexical deficit (numbers / names / content words of the output not in the source; rl/lexical.py; 0 = off)
    # D-52 (science run): numbers / units / formulas / protected scientific tokens must survive each chunk (rl/sci_anchor.py, the metrics of the hard long-document split).
    # Penalty only, one-sided, per chunk; applies to samples whose prompt domain is in `science_anchor_domains` (the general replay prompts are untouched). 0 = off (the general runs).
    science_anchor_penalty: float = 0.0        # reward subtracted per unit of science-anchor deficit (0..1 per chunk)
    science_anchor_path: Optional[str] = None  # calibration json (human P10 of the features; scripts/build_science_anchor_calibration.py); null = the built-in defaults
    science_anchor_domains: List[str] = field(default_factory=lambda: ["science"])
    # D-29: Unicode NFC of the decoded completion AND of the source at the start of the reward path (before failure checks, lengths, chunking, QE)
    nfc: bool = True
    # D-31: diacritics score (rl/diacritics.py): reward subtracted per unit of deficit (stripped diacritics against the language's human P10 density, or
    # fabricated / foreign / over-decorated marks = 1.0), only for languages whose human references use diacritics; 0 = off. Penalty only, never a bonus.
    diacritic_penalty: float = 0.0
    # D-30: omission detector for chunked scoring when the sentence counts of source and output differ (language-aware Gale-Church + shortfall check,
    # see chunk_plan.plan_chunks). Needs the calibration's `len_ratio.corpus`; a language without it is planned as before.
    omission_detector: bool = False
    omission_cost: Optional[float] = None      # nats for a source sentence with no output (and for an output sentence with no source); None = the language's calibrated cost (calibration `omission.cost`, default 4.0)
    omission_short_frac: Optional[float] = 0.7  # a many-to-one bead whose output is shorter than this share of the expected length has missing source text; None = off
    omission_short_min: float = 0.15           # ... if the missing share of the bead's source is at least this
    # D-32: t_fail never below this. 0.46 (the first 12 languages' AfriCOMET scale) is wrong for languages whose human references score below it (kam umb lua ...)
    fail_floor: float = FAIL_FLOOR
    reliability_from: str = "auc"              # auc | same_source (calibration file field `reliability_same_source`)
    # End-of-turn discipline (owner 2026-10-17, D-26): the answer must END with the end token and nothing but padding may follow it.
    # `no_end_of_turn` (truncated) was always a failure class; `eos_discipline` adds post_eos_rubbish and dangling_marker. A penalty of
    # None = the common failure penalty (floor of the range); a value must be <= that floor (a failure may never outscore a translation).
    eos_discipline: bool = True
    eos_truncated_penalty: Optional[float] = None
    eos_post_end_penalty: Optional[float] = None
    eos_dangling_penalty: Optional[float] = None
    # D-38: QE-free consensus signal (minimum-Bayes-risk style). Needs the GRPO group size at call time.
    consensus: bool = False                    # master switch; off = every language is scored exactly as before (blend -> QE only, consensus -> rubric-only value)
    consensus_min_valid: int = 3               # valid samples a group needs for the signal; fewer -> skipped for the whole group
    consensus_skip_value: float = 0.0          # value of a consensus-mode sample whose group has no consensus (rubric-only): = the shaping curve's "acceptable" point (t_fail -> 0)
    content_mode_default: str = "qe"           # languages without a calibration `content` block
    content_modes: Dict[str, str] = field(default_factory=dict)       # per-language override: qe | blend | consensus
    consensus_beta: Dict[str, float] = field(default_factory=dict)    # per-language override of the blend weight
    consensus_beta_default: float = 0.5        # blend languages without a beta of their own
    consensus_scale: Optional[Tuple[float, float]] = None   # (lo, hi) of the chrF -> reward map; None = the reward range of the shaping preset ([-2, +2.25])
    selfrep_guard: str = "consensus"           # near-loop guard (guards.is_self_repetitive): off | consensus (languages in blend / consensus mode) | all
    min_len_ratio_uncalibrated: float = 0.4    # hard lower length bound for blend / consensus languages WITHOUT a human length window (pooled prior: no calibrated language has a p01 below 0.44)
    # D-47 (wired 2026-11-01): REFERENCE-FREE CONFIDENCE (rl/confidence.py). OFF by default = the reward, the weights and the advantages are bit-identical. On, for TRAINING calls
    # (group_size > 1; greedy evaluation never reads or writes the tracker) and the languages whose content mode is in `confidence_modes` (qe languages have the real signal):
    #   rule 1 (confidence_gate)  a group with < 3 valid samples, a GlotLID pass rate below the minimum or a collapsed (copied) valid set gets advantage weight 0 for all its samples;
    #   rule 2 (confidence_trust) the language's advantage weight is multiplied by its running trust in [0, 1] (weakest of GlotLID pass rate, mean log-prob of the valid samples, valid fraction);
    #   rule 3 (confidence_cap)   the reward of a consensus-MODE sample is capped at consensus_cap(trust): a self-agreeing mode cannot out-score a varied correct sibling by more than the cap.
    # The mean log-prob needs the sampler's per-token log-probs (train_rl asks for them when this is on; vLLM sampler: cheap; HF generate: output_scores, tiny vocab / short runs only).
    confidence: bool = False                   # master switch
    confidence_gate: bool = True               # rule 1
    confidence_trust: bool = True              # rule 2
    confidence_cap: bool = True                # rule 3
    confidence_modes: List[str] = field(default_factory=lambda: ["consensus", "blend"])      # content modes the three rules apply to
    confidence_min_groups: int = 10            # a language's trust and cap stay neutral (1.0 / no cap) until its running state holds this many groups (the first groups only fill the tracker)
    confidence_alpha: float = 0.05             # EMA weight of a new group once 20 groups were seen (before: plain mean)
    confidence_relative: bool = True           # log-prob against the median of the languages that work (rl/confidence.py reference_logp) instead of the absolute bounds (valid for another policy)
    confidence_absolute_fallback: bool = False  # relative requested but fewer than 5 reference languages yet: True = use the absolute bounds, False = skip the log-prob ramp (neutral) until the reference exists
    confidence_params: Dict[str, float] = field(default_factory=dict)     # overrides of ConfidenceParams (lid_lo, lid_hi, logp_rel_lo, logp_rel_hi, valid_lo, valid_hi, min_valid, lid_pass_min, distinct_min, cap_floor, min_trust_to_pay ...)
    # D-55: QE ENSEMBLE (rl/qe_ensemble.py). OFF by default. On: the QE part of a chunk's reward is mean(r1, r2) of the per-language shaped AfriCOMET-QE (r1) and SSA-COMET-MTL
    # reference-free (r2) when |r1 - r2| <= delta, else min(r1, r2). Needs a calibration file with `qe2` blocks (shaping_flores_dev_v8.json) and a `scorer` block with `ensemble_model_id`.
    qe_ensemble: bool = False
    qe_ensemble_delta: Optional[float] = None          # reward units; None = `_meta_v8.delta` of the calibration file
    qe_ensemble_model_id: str = "McGill-NLP/ssa-comet-mtl"   # must be the model the `qe2` anchors were measured for (checked); the scorer itself is built from the `scorer` block
    qe_ensemble_calibration_path: Optional[str] = None  # None = calibration_path (v8 is a superset of v7)
    qe_ensemble_prior_langs: bool = True               # languages with a pooled-prior `qe2` (no dev reference): True = ensemble too, False = AfriCOMET alone as before
    qe_ensemble_reliability: str = "ensemble"          # advantage weights: ensemble = the baked `ens.reliability*` of the calibration (max of v7 and the ensemble's own measurement) | keep = the v7 weights

    # D-56: CONSTRAINT PENALTIES for the languages whose content signal is consensus (no AfriCOMET) or a blend (weak AfriCOMET): rl/anchors.py, thresholds fitted on HUMAN references
    # (calibration/anchors_v1.json, scripts/build_anchors_calibration.py). PENALTIES only (never a bonus), each subtracted from the chunk / sample reward (floor: the failure penalty), each
    # with its own weight, each active for a language only where its false-positive rate on human text is small. Off by default.
    anchors_enabled: bool = False
    anchors_calibration_path: Optional[str] = None     # calibration/anchors_v1.json; required when anchors_enabled
    anchors_modes: List[str] = field(default_factory=lambda: ["consensus", "blend"])     # content modes the penalties apply to ("qe" languages have the real QE signal)
    anchors_off_languages: List[str] = field(default_factory=list)     # CONTROL list: these languages never get an anchor penalty (A/B inside one run), whatever the mode
    anchors_entity_weight: float = 0.5                 # (a) chunk level: reward per unit of number / name preservation deficit (0..1: share of the entities missing below the human low quantile)
    anchors_repeat_weight: float = 1.0                 # (b) sample level: reward per unit of word 3/4/5-gram repetition deficit (0..1 from the human p97 of that length to a 30 % repeated share)
    anchors_boiler_weight: float = 1.0                 # (c) sample level: reward per unit of cross-prompt boilerplate deficit (0..1 from the human 3 % to 40 % of the 5-grams found in >= 3 other prompts' outputs)
    anchors_lexicon_weight: float = 0.0                # (d) chunk level, hook of rl/lexicon.py: reward per unit of lexicon-overlap deficit below the human low quantile (0 = off)
    anchors_lid_prob_weight: float = 0.0               # (e) sample level: reward per unit of GlotLID-probability deficit below the human p05 of that language (0 = off); needs reward.lid_enabled
    anchors_lexicon_dir: Optional[str] = None          # default: $GEMMA_MT_LEXICON_DIR or ./lexicon
    anchors_lexicon_guard: bool = True                 # D-59: the overlap is scaled by R3's stuffing guard (density of lexicon candidates in the output above the human p95 -> 0 at 1)
    anchors_lexicon_per_language: bool = False         # D-59: penalty weight of a language = anchors_lexicon_weight x min(anchors_lexicon_weight_cap, the lexicon's validated weight) (0 = no validated lexicon)
    anchors_lexicon_weight_cap: float = 0.25           # D-59: cap of the per-language lexicon weight
    anchors_buffer_capacity: int = 600                 # outputs kept per language for (c)
    anchors_buffer_per_prompt: int = 4                 # ... at most this many of one prompt's group (so 600 outputs cover >= 150 prompts)
    anchors_buffer_min_outputs: int = 150              # no boilerplate penalty before the language's buffer holds this many outputs ...
    anchors_buffer_min_prompts: int = 30               # ... from this many distinct prompts
    anchors_buffer_min_other: int = 3                  # an output 5-gram is boilerplate when it occurs in the outputs of >= this many OTHER prompts
    anchors_related_min: int = 1                       # prompts whose English sources share this many word 6-grams with the current one do not count as "other"
    # D-61 (owner 2026-10-21/22): ANCHORS V2 (rl/anchors_v2.py, calibration/anchors_v2.json). OFF by default; every key below is inert unless anchors_enabled AND anchors_v2_enabled.
    anchors_v2_enabled: bool = False                   # master flag of the v2 detectors (anchors_calibration_path must be calibration/anchors_v2.json, the v1 blocks are inside it)
    anchors_v2_modes: List[str] = field(default_factory=lambda: ["qe", "blend", "consensus"])    # content modes of the copy / func / pad detectors (they are language independent attack guards)
    anchors_v2_off_languages: List[str] = field(default_factory=list)    # CONTROL list of the v2 detectors only (anchors_off_languages is honoured too)
    anchors_entity_mode: str = "strict"                # strict = v1 (penalty below the human 3 % quantile, 4 languages) | soft = v2 (deficit 1 - recall x anchors_entity_weight x per-language w = (1 - human drop rate), channel off above 60 % drop)
    anchors_copy_scale: float = 0.0                    # copy detection (substring): chunk reward := min(reward, -scale x copied fraction) above the language's human threshold (0 = off; the owner's value is 2.0)
    anchors_copy_mode: str = "cap"                     # cap (owner: reward set to -scale x fraction) | subtract (reward - scale x fraction)
    anchors_func_weight: float = 0.0                   # stop-word profile: reward per unit of deficit (0..1) of a sample (0 = off)
    anchors_pad_weight: float = 0.0                    # padded / stuffed output (bigram-order gain, type / token ratio): reward per unit of deficit (0 = off)
    anchors_stuff_gate: float = 0.0                    # 0 = off. In (0, 1]: a sample whose stop-word-profile or padding deficit is >= this level is a HARD failure (status stuffed_output, reward = the failure penalty, no QE, out of the consensus pool). Needs anchors_func_weight or anchors_pad_weight > 0 (they decide which detectors are on)
    anchors_bigram_path: Optional[str] = None          # npz of the training-text bigram tables (calibration/anchors_v2_bigrams.npz); without it `pad` judges the type / token ratio only
    # R14 (2026-10-22, D-66): NLLB as a REWARD-ONLY signal (rl/nllb_signal.py; NLLB output is CC-BY-NC, never training data). OFF by default = the reward is bit-identical.
    nllb_signal: bool = False                          # master switch of the pseudo-reference signal (precomputed NLLB translations of the RL prompts)
    nllb_pseudoref_dir: str = ""                       # directory with <iso>.parquet (scripts/precompute_nllb_pseudorefs.py)
    nllb_calibration_path: str = ""                    # JSON with a `nllb` block per language (human-validated weights; scripts/r14/s3_analyze.py writes it)
    nllb_modes: List[str] = field(default_factory=lambda: ["consensus", "blend"])    # content modes the signal applies to
    nllb_languages: List[str] = field(default_factory=list)                          # extra languages (e.g. qe ones as a check)
    nllb_weight_default: float = 0.35                  # weight = nllb_weight_default * calibrated reliability (capped by nllb_weight_max)
    nllb_weight_max: float = 0.5
    nllb_weights: Dict[str, float] = field(default_factory=dict)                     # per-language override (0 disables)
    nllb_min_teacher_delta: float = 2.0                # teacher chrF++ minus student chrF++ on human references must be >= this, else weight 0
    nllb_min_auc: float = 0.80
    nllb_forced: bool = False                          # forced-decoding score, combined with the pseudo-reference as min(); needs nllb_ct2_dir / nllb_hf_dir (one NLLB forward pass per reward call)
    nllb_forced_combine: str = "min"                   # min | mean
    nllb_forced_min_auc: float = 0.80
    nllb_ct2_dir: str = ""
    nllb_hf_dir: str = ""
    nllb_ct2_compute_type: str = "int8_float16"
    nllb_forced_batch_tokens: int = 4096

    # D-60: REWARD V2 FOR LONG DOCUMENTS (rl/longdoc.py; reports/longdoc_reward_v2_2026-10-22.md). OFF by default (every run is scored exactly as before). On, for samples whose English
    # source has >= `longdoc_min_src_words` words ("long"): (a) the repetition checks are replaced by a CONTINUOUS multi-level penalty with limits taken from human documents of the same
    # length bucket (word 4-gram / repeated 8-gram / duplicate sentence / repeated block / word cycle), (b) sentences are aligned with numbers / names (/ lexicon words) as anchors after the
    # redundant copies were removed, (c) the chunk rewards are aggregated as mean minus the worst-quartile deficit beyond what human documents show, (d) the share of the source with no usable output scales an extra drop
    # towards the failure penalty. Reward of a long sample = r - s * (r - penalty), s in [0, 1] the combined severity: never below the failure penalty.
    longdoc_v2: bool = False
    longdoc_calibration_path: Optional[str] = None     # calibration/longdoc_v2.json (repetition limits per length bucket); None = the built-in defaults
    longdoc_min_src_words: int = 150                   # "long" source: at least this many English words (b256 median 110, b512 405, b1024 786, b2048 1579, b4096 3141)
    longdoc_repetition: bool = True                    # (a)
    longdoc_align: str = "anchored"                    # (b) anchored | gale_church (the v1 aligner)
    longdoc_lexicon: bool = False                      # (b) lexicon words of validated lexicons are anchors too
    longdoc_agg: str = "cvar_excess"                   # (c) mean | cvar | cvar_excess | p25 | min   (cvar_excess: mean - alpha x the tail deficit beyond the human gap of the length bucket)
    longdoc_agg_alpha: float = 1.0
    longdoc_agg_q: float = 0.25
    longdoc_coverage_kappa: float = 0.5                # (d) 0 = off
    longdoc_pool_severity: float = 0.5                 # a sample with a repetition severity at or above this stays out of the consensus pool (near-loops would agree with each other)
    keep_chunk_rewards: bool = False                   # diagnostics: RewardOutput.chunk_rewards = per sample [(reward, weight)] of its plan items



@dataclass
class Sample:
    token_ids: List[int]       # completion ids exactly as sampled (stop id included when produced)
    iso: str
    source: str                # the English source text
    post_stop_tokens: int = 0  # non-padding tokens the SAMPLER produced after the first stop id and that were cut before this call
    domain: str = ""           # D-52: domain label of the prompt ("science" / "general" / ""), read by the science anchor only
    logp: Optional[float] = None   # D-47: mean token log-prob (nats) of the sampled completion under the sampling distribution, language tag and end token left out (None = unknown)


@dataclass
class RewardOutput:
    rewards: List[float]
    weights: List[float]
    qe: List[Optional[float]]
    status: List[str]
    texts: List[str]           # the translation text the scorer saw ("" for failures before scoring)
    decoded: List[str] = field(default_factory=list)   # the cleaned completion before tag stripping (diagnostics)
    stats: Dict[str, float] = field(default_factory=dict)
    diacritic_deficit: List[Optional[float]] = field(default_factory=list)   # per sample: source-weighted mean diacritic deficit of its scored chunks (None = not applicable)
    lid_label: List[Optional[str]] = field(default_factory=list)            # per sample: GlotLID top-1 label when the guard looked at it (None = not checked)
    consensus_chrf: List[Optional[float]] = field(default_factory=list)     # per sample: mean chrF++ against the valid siblings of its group (None = no consensus)
    content_mode: List[str] = field(default_factory=list)                   # per sample: qe | blend | consensus (the mode that was applied; blend without consensus is reported as qe)
    n_chunks: List[int] = field(default_factory=list)                        # per sample: chunks of its plan (scored + failed; 0 for a hard failure that never got a plan) (D-41, archive)
    failed_chunks: List[int] = field(default_factory=list)                  # per sample: chunks of the plan that are FAILED chunks (omitted / extra / copied text; D-30)
    science_deficit: List[Optional[float]] = field(default_factory=list)    # per sample: mean science-anchor deficit of its scored chunks (D-52; None = not judged / anchor off)
    qe2: List[Optional[float]] = field(default_factory=list)                # per sample: mean RAW SSA-COMET-MTL score of its scored chunks (D-55; None = not used / ensemble off)
    anchors_penalty: List[float] = field(default_factory=list)              # per sample: reward subtracted by the constraint penalties (D-56; 0.0 = none / anchors off)
    chunk_rewards: List[Optional[List[Tuple[float, float]]]] = field(default_factory=list)   # D-60 (keep_chunk_rewards): per sample [(chunk reward, weight)] of its plan items (failed chunk = the penalty); None = no plan
    longdoc: List[Optional[Dict[str, float]]] = field(default_factory=list)  # D-60: per sample, for "long" samples: {rep_severity, rep_level, cover_failed, agg_mean, agg_tail, severity, ...}; None = v1 path
    confidence: Dict[str, float] = field(default_factory=dict)              # D-47: metrics of the confidence rules of this call, keys `confidence/...` (empty when reward.confidence is off); logged as they are
    conf_gate: List[Optional[str]] = field(default_factory=list)            # D-47: per sample, the reason its group was gated out ("few_valid" | "wrong_language" | "collapsed"), "ok" if not, None = not judged


class TranslationReward:
    def __init__(self, cfg: RewardConfig, scorer: Any, tokenizer: Any, stop_ids: Sequence[int], lid: Any = None,
                 shaper: Optional[ContentShaper] = None, pad_id: Optional[int] = None):
        self.cfg, self.scorer, self.tok, self.stop_ids, self.lid = cfg, scorer, tokenizer, list(stop_ids), lid
        pad = pad_id if pad_id is not None else getattr(tokenizer, "pad_token_id", None)
        self.pad_ids = [] if pad is None else [int(pad)]
        self.shaper = shaper or ContentShaper.from_file(
            cfg.calibration_path, reliability_from=cfg.reliability_from, mode=cfg.mode, preset=cfg.preset, neg_scale=cfg.neg_scale,
            window=cfg.adaptive_window, warmup=cfg.adaptive_warmup, ema=cfg.adaptive_ema,
            default_anchors=cfg.default_anchors, fail_floor=cfg.fail_floor)
        lo, hi = self.shaper.reward_range
        self.penalty = float(lo if cfg.penalty is None else cfg.penalty)
        if self.penalty > lo + 1e-9:
            raise ValueError(f"penalty {self.penalty} must be <= the floor of the reward range ({lo}): a failure may "
                             "never outscore a scored translation")
        # per-class penalty (D-26): the three end-of-turn classes may be MORE negative than the common floor, never less
        self.class_penalty: Dict[str, float] = {}
        for name, v in (("no_end_of_turn", cfg.eos_truncated_penalty), ("post_eos_rubbish", cfg.eos_post_end_penalty),
                        ("dangling_marker", cfg.eos_dangling_penalty)):
            p = self.penalty if v is None else float(v)
            if p > lo + 1e-9:
                raise ValueError(f"{name} penalty {p} must be <= the floor of the reward range ({lo})")
            self.class_penalty[name] = p
        self.reward_range = (min(lo, self.penalty, *self.class_penalty.values()), hi)
        self._profiles = cons.ProfileCache()
        self.conf = None                                # D-47: running per-language state of the confidence rules (None = off)
        self.conf_params = conf.params_from_dict(cfg.confidence_params)
        if cfg.confidence:
            bad = [m for m in cfg.confidence_modes if m not in CONTENT_MODES]
            if bad:
                raise ValueError(f"reward.confidence_modes {bad} not in {CONTENT_MODES}")
            self.conf = conf.ConfidenceTracker(self.conf_params, alpha=cfg.confidence_alpha)
        self.sci_anchor = None
        if cfg.science_anchor_penalty > 0:
            from gemma_mt.rl.sci_anchor import ScienceAnchor

            self.sci_anchor = ScienceAnchor(cfg.science_anchor_path)
        self.anchors = None
        if cfg.anchors_enabled:                         # D-56: built here, so a missing / unreadable calibration stops the run before step 1
            from gemma_mt.rl.anchors import Anchors

            if not cfg.anchors_calibration_path:
                raise ValueError("reward.anchors_enabled needs reward.anchors_calibration_path (calibration/anchors_v1.json)")
            if cfg.anchors_entity_mode not in ("strict", "soft"):
                raise ValueError("anchors_entity_mode must be strict | soft")
            if cfg.anchors_copy_mode not in ("cap", "subtract"):
                raise ValueError("anchors_copy_mode must be cap | subtract")
            if not 0.0 <= cfg.anchors_stuff_gate <= 1.0 or (cfg.anchors_stuff_gate > 0 and not (cfg.anchors_v2_enabled and (cfg.anchors_func_weight > 0 or cfg.anchors_pad_weight > 0))):
                raise ValueError("anchors_stuff_gate must be in [0, 1] and needs anchors_v2_enabled with anchors_func_weight or anchors_pad_weight > 0")
            for name in ("copy_scale", "func_weight", "pad_weight"):
                if getattr(cfg, f"anchors_{name}") < 0:
                    raise ValueError(f"anchors_{name} must be >= 0 (penalties only)")
            if not cfg.anchors_v2_enabled and (cfg.anchors_entity_mode == "soft" or cfg.anchors_copy_scale > 0 or cfg.anchors_func_weight > 0 or cfg.anchors_pad_weight > 0):
                raise ValueError("anchors_entity_mode soft / anchors_copy_scale / anchors_func_weight / anchors_pad_weight need anchors_v2_enabled")
            bad_v2 = [m for m in cfg.anchors_v2_modes if m not in CONTENT_MODES]
            if bad_v2:
                raise ValueError(f"anchors_v2_modes {bad_v2} are not content modes {CONTENT_MODES}")
            det = [d for d, w in (("entity", cfg.anchors_entity_weight if cfg.anchors_entity_mode == "strict" else 0.0), ("repeat", cfg.anchors_repeat_weight), ("boiler", cfg.anchors_boiler_weight),
                                  ("lexicon", cfg.anchors_lexicon_weight), ("lid_prob", cfg.anchors_lid_prob_weight)) if w > 0]
            for name in ("entity", "repeat", "boiler", "lexicon", "lid_prob"):
                if getattr(cfg, f"anchors_{name}_weight") < 0:
                    raise ValueError(f"anchors_{name}_weight must be >= 0 (penalties only)")
            if cfg.anchors_lid_prob_weight > 0 and not cfg.lid_enabled:
                raise ValueError("anchors_lid_prob_weight > 0 needs reward.lid_enabled (the probability comes from the GlotLID guard's classifier)")
            if cfg.anchors_lexicon_weight > 0:
                if cfg.anchors_lexicon_weight_cap <= 0:
                    raise ValueError("anchors_lexicon_weight_cap must be > 0")
                lexd = cfg.anchors_lexicon_dir or os.environ.get("GEMMA_MT_LEXICON_DIR")
                if not lexd or not os.path.isdir(lexd):
                    raise ValueError(f"anchors_lexicon_weight > 0 needs a lexicon directory (anchors_lexicon_dir / $GEMMA_MT_LEXICON_DIR), got {lexd!r}")
            bad = [m for m in cfg.anchors_modes if m not in CONTENT_MODES]
            if bad:
                raise ValueError(f"anchors_modes {bad} are not content modes {CONTENT_MODES}")
            ctor, v2kw = Anchors.from_file, {}
            if cfg.anchors_v2_enabled:                  # D-61
                from gemma_mt.rl.anchors_v2 import Anchors2

                ctor = Anchors2.from_file
                v2kw = dict(v2_detectors=[d for d, on in (("entity2", cfg.anchors_entity_mode == "soft" and cfg.anchors_entity_weight > 0), ("copy", cfg.anchors_copy_scale > 0),
                                                         ("func", cfg.anchors_func_weight > 0), ("pad", cfg.anchors_pad_weight > 0)) if on],
                            v2_modes=cfg.anchors_v2_modes, v2_off_languages=cfg.anchors_v2_off_languages, copy_scale=cfg.anchors_copy_scale, copy_mode=cfg.anchors_copy_mode,
                            bigram_path=cfg.anchors_bigram_path)
            self.anchors = ctor(cfg.anchors_calibration_path, modes=cfg.anchors_modes, off_languages=cfg.anchors_off_languages, detectors=det, **v2kw,
                                             lexicon_dir=cfg.anchors_lexicon_dir, buffer_capacity=cfg.anchors_buffer_capacity, buffer_min_outputs=cfg.anchors_buffer_min_outputs,
                                             buffer_min_prompts=cfg.anchors_buffer_min_prompts, buffer_min_other=cfg.anchors_buffer_min_other,
                                             buffer_per_prompt=cfg.anchors_buffer_per_prompt, related_min=cfg.anchors_related_min,
                                             lexicon_guard=cfg.anchors_lexicon_guard, lexicon_per_language=cfg.anchors_lexicon_per_language,
                                             lexicon_weight_cap=cfg.anchors_lexicon_weight_cap)
        self.longdoc_cal = None
        self._src_prof_cache: Dict[str, Any] = {}
        if cfg.longdoc_v2:                              # D-60: built here, so a bad flag / unreadable calibration stops the run before step 1
            from gemma_mt.rl import longdoc as LD

            if cfg.longdoc_agg not in ("mean", "cvar", "cvar_excess", "p25", "min"):
                raise ValueError(f"reward.longdoc_agg must be mean | cvar | cvar_excess | p25 | min, got {cfg.longdoc_agg!r}")
            if cfg.longdoc_align not in ("anchored", "gale_church"):
                raise ValueError(f"reward.longdoc_align must be anchored | gale_church, got {cfg.longdoc_align!r}")
            if not (0.0 <= cfg.longdoc_agg_alpha <= 1.0 and 0.0 < cfg.longdoc_agg_q <= 1.0 and 0.0 <= cfg.longdoc_coverage_kappa <= 1.0):
                raise ValueError("reward.longdoc_agg_alpha in [0, 1], longdoc_agg_q in (0, 1], longdoc_coverage_kappa in [0, 1]")
            self.longdoc_cal = LD.LongdocCalibration.from_file(cfg.longdoc_calibration_path)
        self.qe_ens = None
        if cfg.qe_ensemble:                             # D-55: needs a QEPairScorer (train_rl.build_scorer with scorer.ensemble_model_id) or any scorer with `score_secondary`
            from gemma_mt.rl.qe_ensemble import QEEnsemble

            if not (hasattr(scorer, "score_secondary") or hasattr(scorer, "secondary")):
                raise ValueError("reward.qe_ensemble needs a two-model scorer: set scorer.ensemble_model_id (rl/qe_ensemble.py build_pair_scorer)")
            self.qe_ens = QEEnsemble.from_config(cfg, self.shaper)
        self.nllb = None                                # R14 hook (flag-gated, reward-only NLLB signal): None = the reward is unchanged
        if cfg.nllb_signal:
            from gemma_mt.rl.nllb_signal import ForcedDecodingScorer, NllbSignal, NllbSignalConfig

            ncfg = NllbSignalConfig.from_reward_cfg(cfg)
            self.nllb = NllbSignal(ncfg, forced=ForcedDecodingScorer(cfg.nllb_ct2_dir, cfg.nllb_hf_dir, compute_type=cfg.nllb_ct2_compute_type,
                                                                     batch_tokens=cfg.nllb_forced_batch_tokens) if cfg.nllb_forced else None)
        if cfg.lid_enabled and self.lid is None:       # D-37: built and self-tested here, so a missing fastText / model file stops the run before step 1
            from gemma_mt.rl.lid import make_lid

            self.lid = make_lid(cfg.lid_model_path)

    # ------------------------------------------------------------------------------------------ content mode / lid spec of a language
    def content_spec(self, lang: str) -> Tuple[str, float]:
        """(mode, beta) applied to `lang`: config override > calibration `content` block > `content_mode_default`. mode in qe | blend | consensus; beta is the weight of the
        consensus value (0 for qe, 1 for consensus)."""
        cfg = self.cfg
        blk = self.shaper.content.get(lang) or {}
        mode = cfg.content_modes.get(lang) or blk.get("mode") or cfg.content_mode_default
        if mode not in CONTENT_MODES:
            raise ValueError(f"content mode of {lang!r} must be one of {CONTENT_MODES}, got {mode!r}")
        if mode == "qe":
            return "qe", 0.0
        if mode == "consensus":
            return "consensus", 1.0
        beta = cfg.consensus_beta.get(lang, blk.get("beta", cfg.consensus_beta_default))
        return "blend", min(max(float(beta), 0.0), 1.0)

    def lid_spec(self, lang: str) -> Tuple[Tuple[str, ...], int, bool]:
        """(accepted labels, min characters, active) of the language's GlotLID guard (calibration `lid` block; languages.py labels when the file has none at all)."""
        if lang in self.cfg.lid_disable:
            return (), 0, False
        if self.shaper.lid:                               # a calibration with lid blocks: only the measured, enabled languages are checked
            e = self.shaper.lid.get(lang)
            if e is None:
                return (), 0, False
            return tuple(e.get("labels") or ()), int(e.get("min_chars") or 0), bool(e.get("enabled")) and bool(e.get("labels"))
        lab = tuple(LANGUAGES[lang].glotlid)
        return lab, 0, bool(lab)

    # ------------------------------------------------------------------------------------------ main
    def __call__(self, samples: Sequence[Sample], group_size: Optional[int] = None) -> RewardOutput:
        """`group_size`: the GRPO group size G (rows [kG, (k+1)G) are G samples of one prompt). Needed for the consensus signal; None / 1 = every sample alone (greedy
        evaluation: blend languages are scored by QE only, consensus languages get the rubric-only value). A ragged batch (last group shorter) and a block that
        changes prompt in the middle are split into the groups they really are."""
        cfg, n = self.cfg, len(samples)
        status = ["ok"] * n
        texts = [""] * n
        decoded = [""] * n
        stripped_special = 0
        gated: List[int] = []
        langs = [canonical_iso(s.iso) for s in samples]
        for i, s in enumerate(samples):
            if langs[i] is None:
                raise KeyError(f"unknown language tag {s.iso!r}")
        modes = [self.content_spec(l) for l in langs]
        srcs = [nfc(s.source) if cfg.nfc else s.source for s in samples]       # D-29: the source is NFC too (every length / lexical rule counts characters)
        long_i = [self.longdoc_cal is not None and len(x.split()) >= cfg.longdoc_min_src_words for x in srcs]      # D-60
        rep_sev = [0.0] * n                              # D-60: repetition severity of each long sample (0 = none)
        rep_lvl = [""] * n
        rep_prof: List[Optional[Dict[str, float]]] = [None] * n
        for i, s in enumerate(samples):
            iso = langs[i]
            dec = decode_completion(self.tok, s.token_ids, self.stop_ids, pad_ids=self.pad_ids, normalise=cfg.nfc)
            stripped_special += int(dec.special_ids_stripped or dec.special_text_stripped)
            decoded[i] = dec.text
            if not dec.finished:
                status[i] = "no_end_of_turn"                   # truncated: no end token inside max_new_tokens
                continue
            if cfg.eos_discipline:                             # D-26
                if dec.post_stop_tokens + int(s.post_stop_tokens) > 0:
                    status[i] = "post_eos_rubbish"             # something that is not padding after the first end token
                    continue
                if dec.dangling or guards.has_role_tail(dec.text, srcs[i]):
                    status[i] = "dangling_marker"              # turn / role marker leaked into the answer
                    continue
            if cfg.penalize_special_tokens and (dec.special_ids_stripped or dec.special_text_stripped):
                status[i] = "special_tokens"
                continue
            strip = translation_from(dec, iso)
            if prefix_violation(strip.status, cfg.prefix_policy):
                status[i] = "bad_prefix"
                continue
            text = strip.text if strip.has_prefix else dec.text
            if not text.strip():
                status[i] = "empty"
                continue
            if guards.is_copy_of_source(srcs[i], text, cfg.copy_jaccard):
                status[i] = "copy_source"
                continue
            if long_i[i] and cfg.longdoc_repetition:           # D-60: continuous, length-calibrated repetition severity instead of the whole-text v1 rules (those fire on every natural long text)
                prof = longdoc.repetition_profile(text)
                rep_sev[i], rep_lvl[i] = self.longdoc_cal.severity(prof, len(srcs[i].split()), self._src_profile(srcs[i]))
                rep_prof[i] = prof.levels
                if rep_sev[i] >= 1.0:
                    status[i] = "degenerate"                   # severity 1 IS the failure penalty; a stuttering sentence is a failed CHUNK (below), not a failed document
                    continue
            else:
                if cfg.repetition_guard and guards.is_repetitive(text):
                    status[i] = "degenerate"
                    continue
                if cfg.selfrep_guard != "off" and (cfg.selfrep_guard == "all" or modes[i][0] != "qe") and guards.is_self_repetitive(text):
                    status[i] = "degenerate"                   # D-38: near-loops would agree with each other in the consensus pool
                    continue
            lo_ratio = cfg.min_len_ratio
            if modes[i][0] != "qe" and iso not in self.shaper.len_window:
                lo_ratio = max(lo_ratio, cfg.min_len_ratio_uncalibrated)       # D-38: no human length window for this language -> the pooled-prior hard bound
            if guards.length_ratio_bad(srcs[i], text, lo_ratio, cfg.max_len_ratio):
                status[i] = "length_ratio"
                continue
            lo_ratio = cfg.min_len_ratio
            if modes[i][0] != "qe" and iso not in self.shaper.len_window:
                lo_ratio = max(lo_ratio, cfg.min_len_ratio_uncalibrated)       # D-38: no human length window for this language -> the pooled-prior hard bound
            if guards.length_ratio_bad(srcs[i], text, lo_ratio, cfg.max_len_ratio):
                status[i] = "length_ratio"
                continue
            if self.anchors is not None and cfg.anchors_stuff_gate > 0 and self._stuff_gate(iso, modes[i][0], srcs[i], text):
                status[i] = "stuffed_output"                    # D-61: a saturated stop-word / padding deficit is a failure class (reward = the penalty, out of the consensus pool)
                continue
            texts[i] = text
            gated.append(i)

        # ---- GlotLID (D-37): one batched call for every gated sample the guard applies to
        lid_label: List[Optional[str]] = [None] * n
        t_lid = 0.0
        n_lid = 0
        if self.lid is not None and cfg.lid_enabled and gated:
            check, accept = [], {}
            for i in gated:
                labels, min_chars, active = self.lid_spec(langs[i])
                if active and len(texts[i]) >= max(cfg.lid_min_chars, min_chars):
                    check.append(i)
                    accept[i] = labels
            if check:
                t0 = time.time()
                got = self.lid([texts[i] for i in check])
                t_lid = time.time() - t0
                if len(got) != len(check):
                    raise RuntimeError(f"language identification returned {len(got)} labels for {len(check)} texts")
                n_lid = len(check)
                for i, lab in zip(check, got):
                    lid_label[i] = lab
                    if lab not in accept[i]:
                        status[i] = "wrong_language"
                gated = [i for i in gated if status[i] == "ok"]

        # ---- consensus (D-38): mean chrF++ against the other valid samples of the group, for blend / consensus languages
        chrf_mean: List[Optional[float]] = [None] * n
        t_cons = 0.0
        groups_seen = groups_skipped = 0
        if cfg.consensus:
            t0 = time.time()
            okset = {i for i in gated if rep_sev[i] < cfg.longdoc_pool_severity}       # D-60: near-loops stay out of the pool (rep_sev is 0 unless longdoc_v2)
            for grp in cons.split_groups([(langs[i], samples[i].source) for i in range(n)], group_size):
                mode = modes[grp[0]][0]
                if mode == "qe":
                    continue
                groups_seen += 1
                profs = [self._profile(texts[i]) if i in okset else None for i in grp]
                vals = cons.group_consensus(profs, cfg.consensus_min_valid)
                if all(v is None for v in vals):
                    groups_skipped += 1
                for i, v in zip(grp, vals):
                    chrf_mean[i] = v
            t_cons = time.time() - t0
        lo_c, hi_c = (self.shaper.reward_range if cfg.consensus_scale is None else cfg.consensus_scale)
        cons_value: List[Optional[float]] = [None if v is None else cons.consensus_value(v, lo_c, hi_c) for v in chrf_mean]
        applied: List[Tuple[str, float]] = []        # (mode, beta) actually used per sample
        for i in range(n):
            mode, beta = modes[i]
            if mode == "blend" and cons_value[i] is None:
                mode, beta = "qe", 0.0                # no consensus available: QE only
            applied.append((mode, beta))

        # ---- D-47 reference-free confidence (off by default): gate / trust / cap, training calls only
        conf_mult = [1.0] * n                           # factor on the advantage weight of each sample
        conf_cap: Dict[str, float] = {}                 # per language: ceiling of a consensus-mode sample's content reward
        conf_gate: List[Optional[str]] = [None] * n
        conf_metrics: Dict[str, float] = {}
        if self.conf is not None and group_size is not None and int(group_size) > 1:
            self._confidence_rules(samples, langs, modes, status, texts, lid_label, chrf_mean, gated, group_size, conf_mult, conf_cap, conf_gate, conf_metrics)

        # ---- build the segments that go to the scorer: one per sample, or one per aligned sentence pair (chunk_plan.py)
        seg_src: List[str] = []
        seg_mt: List[str] = []
        seg_owner: List[int] = []                       # sample index of each segment
        chunk_plan: Dict[int, List[Tuple[Optional[int], float]]] = {}   # sample -> [(segment index or None=failed chunk, weight)]
        aligned_n = mismatch_n = omitted_n = short_n = extra_n = 0
        for i in gated:
            lang_i = langs[i]
            w_hi = (self.shaper.len_window.get(lang_i) or (None, None))[1] if cfg.chunking == "sentences" else None
            om: Dict[str, Any] = {}
            if cfg.chunking == "sentences" and cfg.omission_detector:       # D-30: language-aware alignment + shortfall check (count mismatches only)
                lm = self.shaper.length_model(lang_i, cfg.omission_cost, cfg.omission_cost)       # None = the language's calibrated omission cost, extra cost 4
                if lm is not None:
                    om = {"length_model": lm, "short_frac": cfg.omission_short_frac, "short_min": cfg.omission_short_min}
            if long_i[i] and cfg.chunking == "sentences" and cfg.longdoc_align == "anchored":
                om = dict(om, beads_fn=longdoc.beads_fn(om.get("length_model"), lang_i if cfg.longdoc_lexicon else None))     # D-60
            cp = plan_chunks(srcs[i], texts[i], chunking=cfg.chunking, chunk_align=cfg.chunk_align,
                             chunk_agg=cfg.chunk_agg, extra_output_penalty=cfg.extra_output_penalty,
                             extra_ratio_limit=cfg.extra_ratio_limit, length_slack=cfg.length_slack,
                             copy_jaccard=cfg.copy_jaccard, w_hi=w_hi, **om)
            if cfg.chunking == "sentences":
                aligned_n += cp.how == "aligned"
                mismatch_n += cp.how in ("gale_church", "whole")
                omitted_n += cp.n_failed
                short_n += cp.n_short
                extra_n += "unmatched_output" in cp.extra
            base = len(seg_src)
            for a, b in cp.segments:
                seg_src.append(a); seg_mt.append(b); seg_owner.append(i)
            chunk_plan[i] = [(None if k is None else base + k, w) for k, w in cp.items]

        if self.nllb is not None:                       # R14 hook: batch the forced-decoding passes of this call (no-op unless nllb_forced)
            self.nllb.prepare([(langs[seg_owner[j]], seg_src[j], seg_mt[j], applied[seg_owner[j]][0]) for j in range(len(seg_src))])

        seg_qe: List[Optional[float]] = [None] * len(seg_src)
        seg_qe2: List[Optional[float]] = [None] * len(seg_src)       # D-55: raw SSA-COMET-MTL scores (ensemble only)
        need_qe = [j for j in range(len(seg_src)) if applied[seg_owner[j]][0] != "consensus"]       # consensus-mode samples are never sent to the scorer
        if need_qe:
            vals = [float(c) for c in self.scorer([seg_src[j] for j in need_qe], [seg_mt[j] for j in need_qe])]
            if len(vals) != len(need_qe):
                raise RuntimeError(f"scorer returned {len(vals)} scores for {len(need_qe)} segments")
            by_lang: Dict[str, List[float]] = {}
            for j, c in zip(need_qe, vals):
                seg_qe[j] = c
                by_lang.setdefault(langs[seg_owner[j]], []).append(c)
            if self.qe_ens is not None:             # D-55: the second judge scores the same segments (those whose language uses it)
                for j, c2 in zip(need_qe, self.qe_ens.score_second(self.scorer, [seg_src[j] for j in need_qe], [seg_mt[j] for j in need_qe],
                                                                  [langs[seg_owner[j]] for j in need_qe])):
                    seg_qe2[j] = c2
            for lang, v in by_lang.items():
                self.shaper.observe(lang, v)          # adaptive mode only

        qe: List[Optional[float]] = [None] * n
        dia_def: List[Optional[float]] = [None] * n
        dia_by_lang: Dict[str, List[float]] = {}
        sci_def: List[Optional[float]] = [None] * n       # D-52: per sample, source-weighted mean science-anchor deficit of its scored chunks (None = not judged)
        rewards, weights = [], []
        ld_info: List[Optional[Dict[str, float]]] = [None] * n
        chunk_rw: List[Optional[List[Tuple[float, float]]]] = [None] * n
        n_chunks = [len(chunk_plan[i]) if i in chunk_plan else 0 for i in range(n)]
        failed_chunks = [sum(1 for seg, _ in chunk_plan[i] if seg is None) if i in chunk_plan else 0 for i in range(n)]
        anc = self._anchors_prepass(samples, langs, modes, srcs, texts, chunk_plan, lid_label, group_size) if self.anchors is not None else None   # D-56
        for i, s in enumerate(samples):
            lang = langs[i]
            mode_i, beta_i = applied[i]
            if i not in chunk_plan:
                rewards.append(self.class_penalty.get(status[i], self.penalty))
            else:
                raw: List[float] = []
                seg_reward: Dict[int, float] = {}
                dia_w = [0.0, 0.0]                  # [weight of the chunks judged, weighted deficit]
                sci_w = [0.0, 0.0]                  # D-52: [weight of the chunks judged by the science anchor, weighted deficit]
                content_c = cons_value[i] if cons_value[i] is not None else cfg.consensus_skip_value      # consensus value of this sample (rubric-only value without consensus)
                if mode_i == "consensus" and lang in conf_cap:                      # D-47 rule 3
                    content_c = min(float(content_c), conf_cap[lang])
                for seg, w in chunk_plan[i]:
                    if seg is not None:
                        if mode_i == "consensus":
                            r = float(content_c)
                        else:
                            raw.append(seg_qe[seg])
                            r = self._shape_qe(seg_qe[seg], seg_qe2[seg], lang)
                            if beta_i > 0:
                                r = (1.0 - beta_i) * r + beta_i * float(content_c)
                        if self.nllb is not None:                 # R14 hook: blend with the NLLB pseudo-reference value (identity unless the language has a calibrated weight)
                            r = self.nllb.blend_chunk(lang, seg_src[seg], seg_mt[seg], r, mode_i)
                        if long_i[i] and cfg.longdoc_repetition and longdoc.has_stutter(seg_mt[seg]):       # D-60: a looping sentence is a failed chunk
                            r = self.penalty
                        if cfg.length_penalty > 0:
                            ex = self.shaper.length_excess(lang, len(seg_src[seg]), len(seg_mt[seg]), cfg.length_slack)
                            if ex > 0:
                                r = max(self.penalty, r - cfg.length_penalty * ex)
                        if cfg.word_ratio_penalty > 0:
                            wx = self.shaper.word_excess(lang, seg_src[seg], seg_mt[seg], cfg.word_ratio_slack)
                            if wx > 0:
                                r = max(self.penalty, r - cfg.word_ratio_penalty * wx)
                        if cfg.lexical_penalty > 0:
                            d = self.shaper.lexical_deficit(lang, seg_src[seg], seg_mt[seg])
                            if d > 0:
                                r = max(self.penalty, r - cfg.lexical_penalty * d)
                        if anc is not None and anc["on"][i]:                      # D-56: chunk-level constraint penalties (entities, lexicon); penalties only
                            r = self._anchors_chunk(anc, i, lang, seg_src[seg], seg_mt[seg], r, w)
                        if self.sci_anchor is not None and s.domain in cfg.science_anchor_domains:       # D-52
                            sd = self.sci_anchor.deficit(lang, seg_src[seg], seg_mt[seg])
                            sci_w[0] += w; sci_w[1] += w * sd
                            if sd > 0:
                                r = max(self.penalty, r - cfg.science_anchor_penalty * sd)
                        if (self.shaper.diacritics.get(lang) or {}).get("applies"):   # D-31: only measured for languages whose human references use diacritics
                            dd = self.shaper.diacritic_deficit(lang, seg_src[seg], seg_mt[seg])
                            dia_w[0] += w; dia_w[1] += w * dd
                            if cfg.diacritic_penalty > 0 and dd > 0:
                                r = max(self.penalty, r - cfg.diacritic_penalty * dd)
                        seg_reward[seg] = r
                if long_i[i]:                                                           # D-60: tail-sensitive aggregation, repetition and coverage severity
                    r_i, info = self._longdoc_reward(chunk_plan[i], seg_reward, rep_sev[i], rep_lvl[i], len(srcs[i].split()))
                    rewards.append(r_i)
                    ld_info[i] = dict(info, **(rep_prof[i] or {}))
                else:
                    rewards.append(aggregate(chunk_plan[i], seg_reward, self.penalty))
                if cfg.keep_chunk_rewards:
                    chunk_rw[i] = [(self.penalty if seg is None else float(seg_reward[seg]), float(w)) for seg, w in chunk_plan[i]]
                if anc is not None and anc["on"][i]:                                   # D-56: sample-level constraint penalties (repetition, boilerplate, LID probability)
                    rewards[-1] = self._anchors_sample(anc, i, rewards[-1])
                if anc is not None and anc["on"][i]:                                   # D-56: sample-level constraint penalties (repetition, boilerplate, LID probability)
                    rewards[-1] = self._anchors_sample(anc, i, rewards[-1])
                qe[i] = sum(raw) / len(raw) if raw else None
                if dia_w[0] > 0:
                    dia_def[i] = dia_w[1] / dia_w[0]
                    dia_by_lang.setdefault(lang, []).append(dia_def[i])
                if sci_w[0] > 0:
                    sci_def[i] = sci_w[1] / sci_w[0]
            weights.append((1.0 if cfg.train_unusable else self._weight(lang)) * conf_mult[i])
        out = RewardOutput(rewards, weights, qe, status, texts, decoded,
                           self._stats(status, qe, rewards, stripped_special, n, gated, texts), diacritic_deficit=dia_def,
                           lid_label=lid_label, consensus_chrf=chrf_mean, content_mode=[m for m, _ in applied],
                           n_chunks=n_chunks, failed_chunks=failed_chunks)
        out.science_deficit = sci_def
        if self.conf is not None:
            out.confidence = conf_metrics
            out.conf_gate = conf_gate
        if cfg.keep_chunk_rewards:
            out.chunk_rewards = chunk_rw
        if self.longdoc_cal is not None:
            for i in range(n):                         # a long sample that failed before planning still carries its repetition severity
                if ld_info[i] is None and long_i[i]:
                    ld_info[i] = {"rep_severity": rep_sev[i], "severity": 1.0 if status[i] != "ok" else 0.0}
            out.longdoc = ld_info
            out.stats.update(self._longdoc_stats(ld_info, long_i, rep_lvl, n))
        if anc is not None:                           # D-56
            out.anchors_penalty = list(anc["pen"])
            out.stats.update(self._anchors_finish(anc, langs, n))
        if self.qe_ens is not None:                   # D-55
            out.qe2 = [self._mean_raw(chunk_plan.get(i), seg_qe2) for i in range(n)]
            out.stats.update(self.qe_ens.pop_stats(self.scorer))
        if self.nllb is not None:                       # D-67: coverage of the NLLB pseudo-references (logged as reward/nllb/coverage_<iso>)
            out.stats.update(self.nllb.pop_stats())
        sci_m = [d for d in sci_def if d is not None]
        if self.sci_anchor is not None:
            out.stats["sci_anchor/judged_rate"] = len(sci_m) / max(n, 1)
            out.stats["sci_anchor/deficit_mean"] = sum(sci_m) / len(sci_m) if sci_m else 0.0
            out.stats["sci_anchor/penalised_rate"] = sum(1 for d in sci_m if d > 0) / max(len(sci_m), 1)
        measured = [d for d in dia_def if d is not None]
        if self.shaper.diacritics:
            out.stats["diacritic/measured_rate"] = len(measured) / max(len(gated), 1)
            out.stats["diacritic/deficit_mean"] = sum(measured) / len(measured) if measured else 0.0
            out.stats["diacritic/penalised_rate"] = sum(1 for d in measured if d > 0) / max(len(measured), 1)
            for lang, v in sorted(dia_by_lang.items()):
                out.stats[f"diacritic/deficit/{lang}"] = sum(v) / len(v)
        self._diagnostics(out, langs, status, lid_label, chrf_mean, modes, applied, n_lid, t_lid, t_cons, groups_seen, groups_skipped)
        if cfg.chunking == "sentences":
            g = max(len(gated), 1)
            out.stats["chunk/aligned_rate"] = aligned_n / g
            out.stats["chunk/count_mismatch_rate"] = mismatch_n / g
            out.stats["chunk/segments_per_sample"] = len(seg_src) / g
            out.stats["chunk/omitted_sentences_per_sample"] = omitted_n / g
            out.stats["chunk/shortfall_beads_per_sample"] = short_n / g
            out.stats["chunk/unmatched_output_rate"] = extra_n / g
            if seg_src and hasattr(self.scorer, "token_lengths"):
                lens = self.scorer.token_lengths(seg_mt)
                out.stats["chunk/mt_truncated_rate"] = sum(1 for l in lens if l - 2 > cfg.max_comet_tokens) / len(lens)
        return out

    # ------------------------------------------------------------------------------------------ long documents (D-60; rl/longdoc.py)
    def _longdoc_reward(self, items, seg_reward, rep_sev: float, rep_lvl: str, src_words: int):
        """(reward, info) of one long sample: tail-sensitive aggregation of its chunk rewards, then the combined severity (repetition + coverage) pulls it towards the failure penalty."""
        cfg = self.cfg
        mean_r = aggregate(items, seg_reward, self.penalty)
        r = longdoc.aggregate_tail(items, seg_reward, self.penalty, mode=cfg.longdoc_agg, alpha=cfg.longdoc_agg_alpha, q=cfg.longdoc_agg_q,
                                  gap0=self.longdoc_cal.gap0(src_words))
        cov = longdoc.coverage_severity(items, cfg.longdoc_coverage_kappa) if cfg.longdoc_coverage_kappa > 0 else 0.0
        sev = longdoc.combine_severity(rep_sev, cov)
        out = longdoc.apply_severity(r, sev, self.penalty)
        return out, {"agg_mean": mean_r, "agg_tail": r, "rep_severity": rep_sev, "cover_failed": longdoc.coverage_severity(items, 1.0), "severity": sev, "n_items": float(len(items))}

    def _src_profile(self, src: str):
        """Repetition profile of an English source (cached: the 16 samples of a GRPO group share it)."""
        c = self._src_prof_cache
        p = c.get(src)
        if p is None:
            if len(c) > 512:
                c.clear()
            p = c[src] = longdoc.repetition_profile(src)
        return p

    def _longdoc_stats(self, info, long_i, rep_lvl, n) -> Dict[str, float]:
        got = [x for x in info if x is not None]
        st: Dict[str, float] = {"longdoc/long_rate": sum(1 for x in long_i if x) / max(n, 1)}
        if got:
            for k in ("agg_mean", "agg_tail", "rep_severity", "cover_failed", "severity"):
                v = [x[k] for x in got if k in x]
                if v:
                    st[f"longdoc/{k}"] = sum(v) / len(v)
            st["longdoc/rep_penalised_rate"] = sum(1 for x in got if x.get("rep_severity", 0.0) > 0) / len(got)
            st["longdoc/rep_full_rate"] = sum(1 for x in got if x.get("rep_severity", 0.0) >= 1.0) / len(got)
        return st

    # ------------------------------------------------------------------------------------------ constraint penalties (D-56; rl/anchors.py)
    def _stuff_gate(self, lang: str, mode: str, src: str, text: str) -> bool:
        """D-61: True when the sample's stop-word-profile or padding deficit reaches `anchors_stuff_gate` (languages / modes of the v2 detectors only; pure, no state)."""
        A, cfg = self.anchors, self.cfg
        if not cfg.anchors_v2_enabled or not A.applies2(lang, mode):
            return False
        from gemma_mt.rl import anchors as AN

        ow = AN.words(text)
        if cfg.anchors_func_weight > 0 and A.enabled2(lang, "func") and A.gate_ok(lang, "func"):
            d = A.func_deficit(lang, AN.words(src), ow)
            if d is not None and d >= cfg.anchors_stuff_gate:
                return True
        if cfg.anchors_pad_weight > 0 and A.enabled2(lang, "pad") and A.gate_ok(lang, "pad"):
            d = A.pad_deficit(lang, ow)
            if d is not None and d >= cfg.anchors_stuff_gate:
                return True
        return False

    def _anchors_prepass(self, samples, langs, modes, srcs, texts, chunk_plan, lid_label, group_size) -> Dict[str, Any]:
        """Sample-level work that needs all samples of the call: which samples the penalties apply to, repetition / boilerplate / LID-probability deficits. The boilerplate buffer is
        READ here (every sample is scored against the outputs of EARLIER calls) and filled afterwards (`_anchors_observe`), so the order of the groups in a call never matters.
        Only a TRAINING call (group_size > 1) scores boilerplate and feeds the buffer; the greedy evaluation (group_size 1 / None) neither reads nor writes it."""
        from gemma_mt.rl import anchors as AN

        A, cfg, n = self.anchors, self.cfg, len(samples)
        train_call = group_size is not None and int(group_size) > 1
        on1 = [i in chunk_plan and A.applies(langs[i], modes[i][0]) for i in range(n)]
        v2 = cfg.anchors_v2_enabled
        on2 = [v2 and i in chunk_plan and A.applies2(langs[i], modes[i][0]) for i in range(n)]       # D-61: copy / func / pad (own mode list); entity2 follows on1
        on = [a or b for a, b in zip(on1, on2)]
        d: Dict[str, Any] = {"on": on, "on1": on1, "on2": on2, "func": [None] * n, "pad": [None] * n, "cp": [[0.0, 0.0, 0.0] for _ in range(n)], "e2": [[0.0, 0.0] for _ in range(n)],
                             "pen": [0.0] * n, "train": train_call, "rep": [None] * n, "boi": [None] * n, "lidp": [None] * n, "prep": {},
                             "ent": [[0.0, 0.0] for _ in range(n)], "lex": [[0.0, 0.0] for _ in range(n)], "chunk_dec": [0.0] * n, "sample_dec": [0.0] * n,
                             "boiler_ready": [None] * n,
                             "wtot": [sum(w for _, w in chunk_plan[i]) or 1.0 if i in chunk_plan else 1.0 for i in range(n)]}
        for i in range(n):
            if on2[i] and (cfg.anchors_func_weight > 0 or cfg.anchors_pad_weight > 0):            # D-61: stop-word profile / padding, sample level, no state
                lang = langs[i]
                sw, ow = AN.words(srcs[i]), AN.words(texts[i])
                if cfg.anchors_func_weight > 0 and A.enabled2(lang, "func"):
                    d["func"][i] = A.func_deficit(lang, sw, ow)
                if cfg.anchors_pad_weight > 0 and A.enabled2(lang, "pad"):
                    d["pad"][i] = A.pad_deficit(lang, ow)
            if not on1[i]:
                continue
            lang = langs[i]
            want_rep = cfg.anchors_repeat_weight > 0 and A.enabled(lang, "repeat")
            want_boi = train_call and cfg.anchors_boiler_weight > 0 and A.enabled(lang, "boiler")
            if want_rep or want_boi:
                p = AN.prepare(lang, srcs[i], texts[i], with_grams=want_boi)
                d["prep"][i] = p
                if want_rep:
                    d["rep"][i] = A.repeat_deficit(lang, p)
                if want_boi:
                    d["boiler_ready"][i] = A.buffer.ready(lang)
                    d["boi"][i] = A.boiler_deficit(lang, p)
            if cfg.anchors_lid_prob_weight > 0 and lid_label[i] is not None and A.enabled(lang, "lid_prob") and hasattr(self.lid, "top"):
                try:
                    top = self.lid.top([texts[i]], 1)[0]
                    d["lidp"][i] = A.lid_prob_deficit(lang, True, float(top[0][1]) if top else None)
                except Exception:
                    d["lidp"][i] = None
        if train_call and cfg.anchors_boiler_weight > 0:     # fill the buffer AFTER every sample was scored
            for i, p in d["prep"].items():
                if p.grams.size:
                    A.observe(langs[i], p)
        return d

    def _anchors_chunk(self, anc: Dict[str, Any], i: int, lang: str, src: str, mt: str, r: float, w: float) -> float:
        cfg, A = self.cfg, self.anchors
        on1, on2 = anc["on1"][i], anc["on2"][i]
        if on1 and cfg.anchors_entity_mode == "soft" and cfg.anchors_entity_weight > 0 and A.enabled2(lang, "entity2"):        # D-61: graded, weight shrinks with the human drop rate
            pu, nj = A.entity2_penalty(lang, src, mt)
            if nj:
                anc["e2"][i][0] += w; anc["e2"][i][1] += w * pu
                if pu > 0:
                    r2 = max(self.penalty, r - cfg.anchors_entity_weight * pu)
                    anc["chunk_dec"][i] += w / anc["wtot"][i] * (r - r2)
                    r = r2
        if on1 and cfg.anchors_entity_mode == "strict" and cfg.anchors_entity_weight > 0 and A.enabled(lang, "entity"):
            de, nj = A.entity_deficit(lang, src, mt)
            if nj:
                anc["ent"][i][0] += w; anc["ent"][i][1] += w * de
                if de > 0:
                    r2 = max(self.penalty, r - cfg.anchors_entity_weight * de)
                    anc["chunk_dec"][i] += w / anc["wtot"][i] * (r - r2)
                    r = r2
        if on1 and cfg.anchors_lexicon_weight > 0 and A.enabled(lang, "lexicon"):
            dl = A.lexicon_deficit(lang, src, mt)
            if dl is not None:
                anc["lex"][i][0] += w; anc["lex"][i][1] += w * dl
                if dl > 0:
                    r2 = max(self.penalty, r - cfg.anchors_lexicon_weight * A.lexicon_weight_factor(lang) * dl)
                    anc["chunk_dec"][i] += w / anc["wtot"][i] * (r - r2)
                    r = r2
        if on2 and cfg.anchors_copy_scale > 0 and A.enabled2(lang, "copy"):                  # D-61: substring copy detection (Jaccard-copied chunks never get here: they are failed chunks)
            fr = A.copy_fraction(lang, src, mt)
            if fr is not None:
                anc["cp"][i][0] += w; anc["cp"][i][1] += w * fr; anc["cp"][i][2] += w * (fr > 0)
                if fr > 0:
                    r2 = A.copy_apply(r, fr, self.penalty)
                    anc["chunk_dec"][i] += w / anc["wtot"][i] * (r - r2)
                    r = r2
        return r

    def _anchors_sample(self, anc: Dict[str, Any], i: int, r: float) -> float:
        cfg = self.cfg
        dec = 0.0
        for key, wt in (("rep", cfg.anchors_repeat_weight), ("boi", cfg.anchors_boiler_weight), ("lidp", cfg.anchors_lid_prob_weight), ("func", cfg.anchors_func_weight),
                        ("pad", cfg.anchors_pad_weight)):
            v = anc[key][i]
            if v is not None and v > 0:
                dec += wt * v
        r2 = max(self.penalty, r - dec) if dec > 0 else r
        anc["sample_dec"][i] = r - r2
        anc["pen"][i] = anc["chunk_dec"][i] + anc["sample_dec"][i]
        return r2

    def _anchors_finish(self, anc: Dict[str, Any], langs, n: int) -> Dict[str, float]:
        """reward/anchors/*: coverage, per-detector judged / penalised rates and mean deficits, penalty sizes, boilerplate buffer state."""
        on = [i for i in range(n) if anc["on"][i]]
        st: Dict[str, float] = {"anchors/applied_rate": len(on) / max(n, 1)}
        pen = [anc["pen"][i] for i in on]
        st["anchors/penalty_mean"] = sum(pen) / len(on) if on else 0.0
        st["anchors/penalty_max"] = max(pen) if on else 0.0
        st["anchors/penalised_rate"] = sum(1 for x in pen if x > 1e-9) / len(on) if on else 0.0

        def det(name: str, vals: List[Optional[float]], weight: float) -> None:
            j = [v for v in vals if v is not None]
            st[f"anchors/{name}/judged_rate"] = len(j) / len(on) if on else 0.0
            if j:
                st[f"anchors/{name}/penalised_rate"] = sum(1 for v in j if v > 0) / len(j)
                st[f"anchors/{name}/deficit_mean"] = sum(j) / len(j)
                st[f"anchors/{name}/penalty_mean"] = weight * sum(j) / len(on)

        cfg = self.cfg
        ent = [(anc["ent"][i][1] / anc["ent"][i][0]) if anc["ent"][i][0] > 0 else None for i in on]
        lex = [(anc["lex"][i][1] / anc["lex"][i][0]) if anc["lex"][i][0] > 0 else None for i in on]
        lex_wt = cfg.anchors_lexicon_weight
        if cfg.anchors_lexicon_per_language and on:
            lex_wt *= sum(self.anchors.lexicon_weight_factor(langs[i]) for i in on) / len(on)       # mean per-language factor (the logged penalty_mean is per judged-or-not sample)
        for name, vals, wt in (("entity", ent, cfg.anchors_entity_weight), ("lexicon", lex, lex_wt), ("repeat", [anc["rep"][i] for i in on], cfg.anchors_repeat_weight),
                               ("boiler", [anc["boi"][i] for i in on], cfg.anchors_boiler_weight), ("lid_prob", [anc["lidp"][i] for i in on], cfg.anchors_lid_prob_weight)):
            if wt > 0:
                det(name, vals, wt)
        if anc["train"] and cfg.anchors_boiler_weight > 0:
            rd = [anc["boiler_ready"][i] for i in on if anc["boiler_ready"][i] is not None]
            st["anchors/boiler/ready_rate"] = sum(1 for x in rd if x) / len(rd) if rd else 0.0
            sizes = [self.anchors.buffer.size(l) for l in sorted({langs[i] for i in on})]
            st["anchors/boiler/buffer_outputs_mean"] = sum(s[0] for s in sizes) / max(len(sizes), 1)
            st["anchors/boiler/buffer_prompts_mean"] = sum(s[1] for s in sizes) / max(len(sizes), 1)
        if cfg.anchors_v2_enabled:                          # D-61
            for name, vals, wt in (("func", [anc["func"][i] for i in on], cfg.anchors_func_weight), ("pad", [anc["pad"][i] for i in on], cfg.anchors_pad_weight),
                                   ("entity2", [(anc["e2"][i][1] / anc["e2"][i][0]) if anc["e2"][i][0] > 0 else None for i in on], cfg.anchors_entity_weight if cfg.anchors_entity_mode == "soft" else 0.0)):
                if wt > 0:
                    det(name, vals, wt)
            if cfg.anchors_copy_scale > 0:
                cp = [anc["cp"][i] for i in on]
                j = [c for c in cp if c[0] > 0]
                st["anchors/copy/judged_rate"] = len(j) / len(on) if on else 0.0
                if j:
                    st["anchors/copy/penalised_rate"] = sum(1 for c in j if c[2] > 0) / len(j)
                    st["anchors/copy/fraction_mean"] = sum(c[1] / c[0] for c in j) / len(j)
        by: Dict[str, List[float]] = {}
        for i in on:
            by.setdefault(langs[i], []).append(anc["pen"][i])
        for l, v in sorted(by.items()):
            st[f"anchors/penalty/{l}"] = sum(v) / len(v)
        return st

    # ------------------------------------------------------------------------------------------ QE ensemble hooks (D-55; rl/qe_ensemble.py)
    # ------------------------------------------------------------------------------------------------------------------ D-47: reference-free confidence

    def _conf_trust(self, lang: str) -> float:
        """Running trust of a language in [0, 1]; neutral (1.0) while its state holds fewer than `confidence_min_groups` groups."""
        st = self.conf.state.get(lang)
        if st is None or st.seen < self.cfg.confidence_min_groups:
            return 1.0
        return self.conf.trust(lang, relative=self.cfg.confidence_relative, missing_ok=True, absolute_fallback=self.cfg.confidence_absolute_fallback)

    def _confidence_rules(self, samples, langs, modes, status, texts, lid_label, chrf_mean, gated, group_size, conf_mult, conf_cap, conf_gate, metrics) -> None:
        """Observe every GRPO group of a training call and apply the three rules (fills conf_mult / conf_cap / conf_gate / metrics in place)."""
        cfg, n = self.cfg, len(samples)
        valid_set = set(gated)                          # passed every hard check, GlotLID included
        hi = self.shaper.reward_range[1]
        reasons: Dict[str, int] = {}
        per_lang: Dict[str, Dict[str, float]] = {}
        n_groups = n_gated = n_zero = 0
        for grp in cons.split_groups([(langs[i], samples[i].source) for i in range(n)], group_size):
            lang, mode = langs[grp[0]], modes[grp[0]][0]
            if mode not in cfg.confidence_modes:
                continue
            f = conf.group_features([i in valid_set for i in grp], [None if lid_label[i] is None else status[i] != "wrong_language" for i in grp], [texts[i] for i in grp],
                                    [samples[i].logp for i in grp], [chrf_mean[i] for i in grp])
            ok, reason = self.conf.observe(lang, f)
            trust = self._conf_trust(lang)
            n_groups += 1
            d = per_lang.setdefault(lang, {"groups": 0.0, "gated": 0.0, "logp_sum": 0.0, "logp_n": 0.0})
            d["groups"] += 1
            if f.logp_valid is not None or f.logp_all is not None:
                d["logp_sum"] += f.logp_valid if f.logp_valid is not None else f.logp_all
                d["logp_n"] += 1
            for i in grp:
                conf_gate[i] = reason
            if cfg.confidence_gate and not ok:
                n_gated += 1
                d["gated"] += 1
                reasons[reason] = reasons.get(reason, 0) + 1
                for i in grp:
                    conf_mult[i] = 0.0
                n_zero += len(grp)
                continue
            if cfg.confidence_trust:
                for i in grp:
                    conf_mult[i] *= trust
                if trust <= 0.0:
                    n_zero += len(grp)
            if cfg.confidence_cap and mode == "consensus":
                conf_cap[lang] = conf.consensus_cap(trust, cfg.consensus_skip_value, hi, self.conf_params)
        if not n_groups:
            return
        metrics["confidence/groups"] = float(n_groups)
        metrics["confidence/gate_rate"] = n_gated / n_groups
        metrics["confidence/weight_zeroed_rate"] = n_zero / max(n, 1)
        for r in ("few_valid", "wrong_language", "collapsed"):
            metrics[f"confidence/gate_reason/{r}"] = float(reasons.get(r, 0))
        ref = self.conf.reference_logp()
        if ref is not None:
            metrics["confidence/logp_ref"] = ref
        for lang, d in sorted(per_lang.items()):
            st = self.conf.state[lang]
            metrics[f"confidence/gate_rate/{lang}"] = d["gated"] / d["groups"]
            metrics[f"confidence/trust/{lang}"] = self._conf_trust(lang)
            if d["logp_n"]:
                metrics[f"confidence/logp_mean/{lang}"] = d["logp_sum"] / d["logp_n"]
            if st.logp_valid is not None:
                metrics[f"confidence/logp_running/{lang}"] = st.logp_valid
            if st.lid_pass is not None:
                metrics[f"confidence/lid_pass_running/{lang}"] = st.lid_pass
            if lang in conf_cap:
                metrics[f"confidence/cap/{lang}"] = conf_cap[lang]

    def conf_state(self) -> Optional[Dict[str, Any]]:
        """The tracker's running state for the trainer checkpoint (None when the rules are off)."""
        return self.conf.state_dict() if self.conf is not None else None

    def load_conf_state(self, d: Optional[Dict[str, Any]]) -> int:
        return self.conf.load_state_dict(d) if self.conf is not None else 0

    def _shape_qe(self, c1: float, c2: Optional[float], lang: str) -> float:
        """The shaped QE value of one chunk: the language curve of AfriCOMET alone (default), or the ensemble rule."""
        if self.qe_ens is None:
            return float(self.shaper(c1, lang))
        return self.qe_ens.shape(c1, c2, lang)

    def _weight(self, lang: str) -> float:
        base = self.shaper.reliability(lang)
        return base if self.qe_ens is None else self.qe_ens.weight(lang, base)

    @staticmethod
    def _mean_raw(plan, seg_vals) -> Optional[float]:
        v = [seg_vals[seg] for seg, _ in (plan or []) if seg is not None and seg_vals[seg] is not None]
        return sum(v) / len(v) if v else None

    def _profile(self, text: str) -> "cons.Profile":
        return self._profiles.get(text)

    def _diagnostics(self, out: RewardOutput, langs, status, lid_label, chrf_mean, modes, applied, n_lid, t_lid, t_cons, groups_seen, groups_skipped) -> None:
        """Per-class / per-language statistics (they reach the logger as `reward/<key>`): failure rates per language, GlotLID, consensus."""
        n = len(status)
        st = out.stats
        per_lang: Dict[str, List[int]] = {}
        for i, l in enumerate(langs):
            per_lang.setdefault(l, []).append(i)
        for l, ii in sorted(per_lang.items()):
            for f in FAILURES:
                k = sum(1 for i in ii if status[i] == f)
                if k:
                    st[f"fail/{f}/{l}"] = k / len(ii)                 # e.g. fail/wrong_language/yor: share of this language's samples in the batch
        if self.lid is not None and self.cfg.lid_enabled:
            checked = [i for i in range(n) if lid_label[i] is not None]
            st["lid/checked_rate"] = len(checked) / max(n, 1)
            st["lid/fail_rate_of_checked"] = sum(1 for i in checked if status[i] == "wrong_language") / max(len(checked), 1)
            st["lid/seconds"] = t_lid
            for l, ii in sorted(per_lang.items()):
                c = [i for i in ii if lid_label[i] is not None]
                if c:
                    st[f"lid/fail_rate/{l}"] = sum(1 for i in c if status[i] == "wrong_language") / len(c)
        if self.cfg.consensus:
            pooled = [i for i in range(n) if modes[i][0] != "qe"]
            got = [i for i in pooled if chrf_mean[i] is not None]
            st["consensus/seconds"] = t_cons
            st["consensus/groups"] = float(groups_seen)
            st["consensus/groups_skipped"] = float(groups_skipped)
            st["consensus/applied_rate"] = len(got) / max(len(pooled), 1) if pooled else 0.0
            if got:
                st["consensus/chrf_mean"] = sum(chrf_mean[i] for i in got) / len(got)
            for l, ii in sorted(per_lang.items()):
                g = [i for i in ii if chrf_mean[i] is not None]
                if g:
                    st[f"consensus/chrf/{l}"] = sum(chrf_mean[i] for i in g) / len(g)
        for m in CONTENT_MODES:
            st[f"mode/{m}_rate"] = sum(1 for a in applied if a[0] == m) / max(n, 1)

    # ------------------------------------------------------------------------------------------ stats
    def _stats(self, status, qe, rewards, stripped_special, n, gated, texts) -> Dict[str, float]:
        st: Dict[str, float] = {"reward_mean": sum(rewards) / max(n, 1),
                                "gate_rate": len(gated) / max(n, 1),
                                "special_stripped_rate": stripped_special / max(n, 1)}
        for f in FAILURES:
            st[f"fail/{f}"] = sum(1 for s in status if s == f) / max(n, 1)
        scored = [q for q in qe if q is not None]
        st["qe_mean_scored"] = sum(scored) / len(scored) if scored else float("nan")
        if gated and hasattr(self.scorer, "token_lengths"):
            lens = self.scorer.token_lengths([texts[i] for i in gated])
            st["comet_mt_truncated_rate"] = sum(1 for l in lens if l - 2 > self.cfg.max_comet_tokens) / len(lens)
        return st
