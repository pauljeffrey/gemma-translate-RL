# Dictionary-overlap signal for the weak-AfriCOMET languages (D-57, agent B3), 2026-10-21

Question (owner): for the 45 target languages with no / weak AfriCOMET signal (23 consensus-only + 22 blend in `calibration/shaping_flores_dev_v7.json`), can an
English -> X lexicon mined from our own pivot data give a usable reward term ("share of the source's content words whose X translation occurs in the output")?

**Answer.** Yes for 10 languages, no for the rest. The signal separates a human translation from a translation of another sentence (AUC 0.79-0.91), from the English
copy and from the empty-of-content outputs a policy falls into, but it is a COARSE, recall-only signal: within one prompt it correlates only weakly with sentence chrF++
(mean Spearman 0.07-0.27), and it can be inflated by padding with common words. Use it with a small weight and next to the precision guards (LID, length window,
consensus), never alone.

Only languages with a FLORES+ dev human reference could be validated. FLORES+ has 13 of our 45; the other 32 have a lexicon but stay `validated=false`.

## Recommended list (hook: `lexicon_available(iso)` is True only for these)

Criteria (pre-set, in `scripts/build_lexicons.py finalize`): FLORES+ dev AUC(human reference vs reference of another sentence) >= 0.75 (affix match), mean
coverage >= 0.30, model-output attractor AUC >= 0.70. Weight = `0.25 * clip((AUC-0.70)/0.20, 0, 1) * min(1, coverage/0.5)` (an upper bound on the share of the
language's reward, rounded to 0.01; never above the consensus / AfriCOMET term).

| iso | mode now | AUC vs other-ref | coverage | proposed weight | lexicon entries |
|---|---|---|---|---|---|
| sag | blend | 0.898 | 0.73 | 0.25 | 5193 |
| kam | blend | 0.875 | 0.60 | 0.22 | 2733 |
| fon | blend | 0.857 | 0.68 | 0.20 | 4026 |
| kab | blend | 0.846 | 0.54 | 0.18 | 2125 |
| kmb | blend | 0.846 | 0.66 | 0.18 | 3509 |
| kea | consensus | 0.913 | 0.33 | 0.17 | 781 |
| umb | blend | 0.828 | 0.71 | 0.16 | 4424 |
| vmw | consensus | 0.803 | 0.64 | 0.13 | 3267 |
| mos | blend | 0.797 | 0.67 | 0.12 | 4324 |
| kbp | blend | 0.785 | 0.37 | 0.08 | 1003 |

Failed: **dyu** (AUC 0.659, coverage 0.48: only 146 non-religious pairs, 11k pairs in total, the human reference scores only 0.118 mean overlap), **nus** (coverage 0.10,
AUC 0.749: 555 pairs, 146 entries), **bam** (no lexicon: 175 usable pairs, below the 200 minimum).
Not validated (no FLORES+ dev file, 31 languages; lexicons shipped with `validated=false`): ach bas bbj bin dag dua efi her ibb ish iso kon kqn kri kua loz lun nbl ndc nyn
oke seh shi sid swc tiv tsc urh ven wal wes yao. Their held-out AUC (0.86-0.99) is NOT evidence: on the 12 FLORES languages held-out AUC is on average ~0.10 higher than
FLORES AUC (sag 0.97 vs 0.90, vmw 0.91 vs 0.80, dyu 0.93 vs 0.66, kbp 0.93 vs 0.79), because the held-out pairs are the training genre (Bible / JW / news-like), not
open-domain text. To certify them a human reference is needed (a few hundred sentences each).

## Data and method

* Source: `<HF_ORG_LEGACY>/interlaanguage-translation`, `full_pairs/pivot_en/<iso>/` (two translations of one English sentence, joined through `pivot_en_sentences.parquet`,
  X as src or tgt) plus the `pivot_en` rows of `sft_pairs_train/val.parquet`. 44 languages, 4k-300k pairs each (kept after cleaning: LID flag, length ratio 0.4-2.5,
  English == X dropped, FLORES+ dev English sentences removed, an English sentence used at most twice, an X sentence at most twice). Downloads (2.0 GB) deleted afterwards.
* Cap 300k pairs per language, non-religious pairs first. 600 whole English sentences per language (300 religious + 300 non-religious where available) are held out of
  the training pairs.
* Religion flag per pair: provenance `religious_jw`, source name contains "bible", or a religious cue word in the English sentence. Median share of religious pairs over the
  44 languages: **89 %** (range 1 % nbl to 100 % ach / bbj). Religious pairs have EM weight 0.3.
* English side: content words only (stop list, >= 3 letters, mid-sentence capitalised words are names and skipped), light stemmer as key; entry needs >= 5 sentences,
  IDF >= 2.3 over a GENERAL English corpus (English of the non-religious pairs + FLORES+ dev English), and >= 3 sentences containing both words. `bib` flag: >= 6x more
  frequent in the religious English than in the general English (3.4 % of entries, median).
* X side: NFC, lowercase, apostrophes folded, tone/accent marks folded (the dot below, ogonek, cedilla kept: ọ is not o); surface variants kept in `v`.
* Alignment: IBM Model 1 both directions with NULL word, 6 EM iterations on unique types per sentence (numpy, sparse expansion), candidate score = sqrt(t(f|e) * t(e|f)),
  top-3 per English stem, floor 0.02. 2-300 s per language on 2 threads.
* Runtime (`gemma_mt/rl/lexicon.py`, stdlib only): `lexicon_overlap(iso, src, mt)` -> `LexiconOverlap(overlap, coverage, n_content, n_covered, n_hit, density)`.
  `overlap` = IDF-weighted (cap 8) share of the covered content words that have a candidate in the output; **None** when fewer than 2 words are covered (no reward, no
  punishment). `match="affix"` (used by every validated lexicon): a candidate also matches a token that contains it (or vice versa) when the shorter has >= 4 letters and
  >= 60 % of the longer, which handles Bantu noun-class prefixes and suffixed verbs. Affix beat exact on all 12 languages (+0.01 to +0.105 AUC; umb 0.723 -> 0.828), so the
  mode is the same for all, not chosen per language.

## Validation (FLORES+ dev, 997 sentences per language, HF_TOKEN only in the environment)

AUC = Mann-Whitney, pooled over sources the lexicon covers (>= 2 covered words, 75-98 % of sources), "other-ref" = mean over 5 shifts (sd 0.02-0.03).

| iso | status | cov | AUC other-ref (exact / **affix**) | vs EN copy | vs half dropped | w/o biblical entries | rel-only lexicon | non-rel-only lexicon (entries) | within-prompt rho | pooled rho | own vs other-source output |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dyu | failed | 0.48 | 0.615 / **0.659** | 0.668 | 0.581 | 0.658 | 0.657 | - | 0.12 | 0.08 | 0.75 |
| fon | ok | 0.68 | 0.849 / **0.857** | 0.911 | 0.685 | 0.856 | 0.852 | 0.840 (2415) | 0.07 | 0.19 | 0.90 |
| kab | ok | 0.54 | 0.809 / **0.846** | 0.844 | 0.668 | 0.846 | 0.719 | 0.838 (1420) | 0.19 | 0.40 | 0.91 |
| kam | ok | 0.60 | 0.850 / **0.875** | 0.870 | 0.687 | 0.871 | 0.868 | 0.833 (797) | 0.27 | 0.27 | 0.83 |
| kbp | ok | 0.37 | 0.766 / **0.785** | 0.834 | 0.631 | 0.784 | 0.789 | 0.757 (188) | 0.16 | 0.13 | 0.77 |
| kea | ok | 0.33 | 0.890 / **0.913** | 0.933 | 0.720 | 0.912 | 0.906 | - | 0.26 | 0.26 | 0.81 |
| kmb | ok | 0.66 | 0.781 / **0.846** | 0.881 | 0.678 | 0.845 | 0.839 | 0.790 (1095) | 0.20 | 0.19 | 0.88 |
| mos | ok | 0.67 | 0.760 / **0.797** | 0.835 | 0.646 | 0.796 | 0.790 | 0.775 (1457) | 0.15 | 0.17 | 0.90 |
| nus | failed | 0.10 | 0.724 / **0.749** | 0.813 | 0.625 | 0.751 | 0.773 | 0.705 (83) | 0.21 | 0.16 | 0.93 |
| sag | ok | 0.73 | 0.886 / **0.898** | 0.920 | 0.727 | 0.898 | 0.900 | 0.861 (2028) | 0.21 | 0.32 | 0.93 |
| umb | ok | 0.71 | 0.723 / **0.828** | 0.851 | 0.644 | 0.827 | 0.826 | 0.791 (1473) | 0.27 | 0.26 | 0.91 |
| vmw | ok | 0.64 | 0.777 / **0.803** | 0.794 | 0.641 | 0.802 | 0.801 | 0.767 (413) | 0.12 | 0.20 | 0.82 |

Model samples: 8 samples at T=1 for 80 FLORES+ dev prompts per language, policy `gemma-3-270m-translate-rl2c@latest` (reused from agent A5, `/workspace/jobs/a5/policy_samples.jsonl`,
640 rows per language). "within-prompt rho" = mean over prompts of Spearman(overlap, sentence chrF++) (positive in 54-77 % of the prompts); "own vs other-source output" =
AUC of the overlap of the model's output for the source against the overlap of the output the model gave for ANOTHER source (the attractor / gaming check).
Per-language details (exact and affix, top-1, uniform weights, held-out, guard numbers): `reports/lexicon_overlap_2026-10-21/validation.json`.

Reading: (1) the signal is strong at "right content vs wrong content" (AUC vs other-ref, vs EN copy 0.79-0.93); (2) half a translation is only moderately separated from a full one
(0.58-0.73), because content words are spread over the sentence and the lexicon covers 33-73 % of them; (3) within a group of 8 samples of the same prompt it is a weak ranker
(rho ~0.2): on its own it would add noise to the GRPO advantage, which is why the weights are small; (4) model outputs have mean overlap 0.08-0.43 against 0.12-0.50 for the human
references (kbp 0.08 vs 0.31: the policy is poor there), while an output for a DIFFERENT source scores 0.01-0.09 and the English copy 0.00-0.05: the mismatch / attractor gaming
check passes (AUC 0.75-0.93).

## Biblical bias: how large is it?

* 89 % of the pairs are religious (median), so most lexicons are trained mostly on Bible / JW text. 3.4 % of the entries (median) are flagged `bib` (>= 6x more frequent in religious
  English); stripping them changes the FLORES AUC by -0.0008 on average (worst -0.004). Down-weighting religious pairs to 0.3 and IDF over general English works as intended: the
  common function-like vocabulary is not lexicalised and the biblical words are rare in FLORES sources.
* A lexicon trained ONLY on religious pairs scores within 0.01 of the full lexicon on FLORES for 9 of 10 languages; kab (8k religious of 28k pairs) drops 0.846 -> 0.719. A lexicon
  trained ONLY on the non-religious pairs (3-10x fewer pairs) is 0.00-0.06 lower (fon 0.857 -> 0.840, sag 0.898 -> 0.861, mos 0.797 -> 0.775, kbp 0.785 -> 0.757). So the loss from
  the domain is moderate and mostly a data-quantity effect; there is no sign that the signal is "just" biblical vocabulary.
* The domain shift shows in the level, not in the ranking: covered share of the content words is 0.33-0.73 on FLORES (news / wiki-like English) against 0.84-0.97 on held-out
  pairs, and the AUC is ~0.10 lower. Coverage is the limiting factor for kea (0.33), kbp (0.37), nus (0.10).
* Religious sentences are held-out separately (300 whole sentences): AUC vs other-ref 0.93-0.97 (religious) vs 0.91-0.97 (non-religious) for the 10 validated languages (affix).

## Gaming (recall-only signal)

* Pure stuffing (the top candidates of the 30 / 80 commonest English content words, one word each) scores 0.15-0.33 / 0.28-0.62 against 0.18-0.50 for human references (nus 0.69 / 1.00):
  **the raw overlap alone is a gameable reward.**
* `density` (share of the output's tokens that are a candidate of any lexicon entry) and `meta.density_max` (95th percentile of human FLORES references, 0.40-0.95) implement a guard:
  `lexicon_reward(iso, src, mt)` scales the overlap linearly to 0 between `density_max` and 1. Pure stuffing has density 1.0 -> reward 0 for all 12 languages; human references lose
  <= 0.01 (mean overlap 0.337 -> 0.332 fon, 0.435 -> 0.425 sag); model outputs lose 20-30 % for umb, kmb, dyu (umb 0.247 -> 0.177, kmb 0.309 -> 0.247) and < 8 % for the rest.
* A padded attack (30 stuffed candidates + up to 30 non-candidate words taken from another human reference) is only partly caught: raw overlap 0.14-0.27 (human 0.18-0.49),
  after the guard 0.08-0.21 (`padding_attack_check.py`). Languages where the candidate set contains the language's function words have a weak guard (sag density_max 0.95, fon 0.82,
  mos 0.74). So the term needs the existing guards (LID, length window, consensus agreement) and the 0.25 cap; do not raise weights without a new red-team round.
* English-copy and mismatch attractors are not rewarded (0.00-0.05 and 0.01-0.09 mean overlap).

## Files

* `gemma_mt/rl/lexicon.py` (runtime), `lexicon/<iso>.json` (44 files, 11 MB; `meta` carries `validated`, `weight`, `match`, `density_max`, the validation numbers and build parameters),
  `scripts/build_lexicons.py` (`prep`, `train`, `finalize`), `scripts/validate_lexicons.py`, `tests/test_lexicon.py` (16 tests incl. a planted-dictionary corpus; also run on the box),
  `reports/lexicon_overlap_2026-10-21/{validation.json, prep_stats.json, padding_attack_check.py}`.
* Reproduce: `prep --raw corpus_raw --pivot full_pairs/pivot_en_sentences.parquet --flores-en flores/eng_Latn.jsonl`, `train --corpus corpus --out lexicon`, `validate_lexicons.py ... --out validation.json`,
  `build_lexicons.py finalize --lexicon lexicon --validation validation.json`. Box working dir `/workspace/jobs/b3` (corpus, flores, lexicon*, validation*.json; downloads deleted, 359 MB left).
* Hook for rl/anchors.py (agent R2): `from gemma_mt.rl.lexicon import lexicon_overlap, lexicon_available` (+ `lexicon_reward` with the stuffing guard and `lexicon_weight(iso)`):
  `if lexicon_available(iso): r = lexicon_reward(iso, english_source, translation)`; `None` means "no information" (skip the term for that sample). Env `GEMMA_MT_LEXICON_DIR` overrides the directory.

## Limits

FLORES+ dev only (devtest untouched, D-24); 12 of 45 languages measurable; AUC sd across shifts 0.02-0.03, one sample set of 80 prompts for the model tests (rho SE ~0.04 per language);
the policy samples are from the rl2c checkpoint, a different policy may shift the model-based numbers. Mode `affix` and the weight formula were set after seeing the 12 FLORES results (affix wins everywhere, so the choice
is not language-specific), the thresholds were fixed in the brief.
