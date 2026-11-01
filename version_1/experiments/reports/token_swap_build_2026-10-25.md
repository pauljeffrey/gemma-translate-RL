# Vocabulary swap for Ge'ez and Fon: build and CPU tests (agent T1, 2026-10-25/26)

Scope: build the tooling, the swapped tokenizers and the swapped 270M checkpoints, and test them on CPU. **No training, no GPU, nothing pushed to the Hub.** Inputs read with the repo's read token (loaded inside the scripts, never printed). Tested = measured by a script here; planned = not run.

**Framing.** Step 0 (`reports/ge_ez_step0_2026-10-25.md`) showed that tokenisation does not explain the amh gap to NLLB (amh needs 1.31x NLLB's tokens, below the 1.39x mean over eight African languages). The swap is therefore justified by sequence length, decoding speed and context, not as a proven quality fix. The tokenizer is a separate experimental factor (see `reports/data/tokswap_2026-10-25/DECISION_RULE_DRAFT.txt`). The swapped checkpoints are **worse** than the original before any retraining (section 5).

## 0. What was built (main result)

Plan (coordinator, after S0): re-use the 6,143 `<unused99>..<unused6241>` ids (256001-262143) of the Gemma-3 vocabulary; **nothing is removed from the vocabulary or the merge list**; vocabulary size stays 262,144 and the embedding matrix shape is unchanged; all special/template ids (`<bos>` 2, `<start_of_turn>` 105, `<end_of_turn>` 106, `<unused0..98>` 6-104 ...) stay.

| variant | new tokens N | ids | merge rules appended | status |
|---|---|---|---|---|
| `geez_N1000` | 61 single Ethiopic letters + 939 merges | 256001..257000 | 939 | built, tested |
| `geez_N3000` | 61 letters + 2,939 merges | 256001..259000 | 2,939 | built, tested |
| `fon_N1000` | 1,000 merges | 262143..261144 (descending) | 1,000 | built, tested, **flagged** (changes other languages) |
| `fon_N3000` | 3,000 merges | 262143..259144 | 3,000 | built, tested, **flagged** |
| Ge'ez N=8,000 | - | needs ids beyond the unused tail, i.e. removing tokens | - | **not built** (optional; S0's simulation: amh 34.4, tir 36.8 tokens/sentence) |

Ge'ez (amh+tir joint) and Fon use disjoint id ranges (ascending vs descending from the tail), so a combined tokenizer is possible with Ge'ez N + Fon M <= 6,143 (not built). Prefix property: N=1000 is a prefix of N=3000 (same ids, same strings).

Tool: `gemma_mt/scripts/tokswap_swap_vocab.py` (copy on the box: `/workspace/tokswap/swap_vocab.py`): `classify`, `train`, `build`, `spm`, `surgery`, `simcheck`. Other scripts: `tokswap_eval.py` (identity, fertility), `tokswap_sanity.py`, `tokswap_spm_check.py`, `tokswap_hf_check.py`, `tokswap_gguf_check.py`, `tokswap_report_tables.py`, `tokswap_fetch_laptop.py`, `tokswap_run_all.sh` (all commands), `tokswap_sanity_lane.sh`.

**Construction.** (a) The training text is tokenised with the original tokenizer; new merges are learned on top of that segmentation (continued BPE) and **appended at the end** of the merge list, so on any text the original merges apply first. (b) Ge'ez: every new token consists only of Ethiopic letters (optional leading `▁`); letters without a single-character token (61 with at least 3 occurrences; today spelled with 3 byte tokens) get one first. Fon: a new token is made of Latin letters and combining marks and must contain a Fon-specific character (ɖ ɛ ɔ ŋ, capitals, haček vowels ǎ ě ǐ ǒ ǔ, combining haček). Because one of the two parts of such a merge already contains the character, a new merge can only fire on text that contains it. (c) The `<unusedN>` entries are deleted from `model.vocab` and `added_tokens` and replaced by the new strings at the same ids. (d) `tokenizer.model` (SentencePiece): same ids, piece = new string, type NORMAL, score = (lowest existing score) - 1 - rank.

## 1. Free list (tested counts, nothing removed)

`reports/data/tokswap_2026-10-25/free_list/` (`free_pool.json`, `class_counts.csv`, `token_classes.tsv.gz`). Assumptions: English source; 77 target languages = Latin (75) and Ethiopic (2); French and Arabic source items; digits, punctuation, symbols and emoji (letter-less tokens), byte tokens, all added/special/unused tokens and Greek (science/LaTeX letters) are kept. A token is freeable if it has a letter outside the kept scripts (mixed tokens count). Counts (letters only / plus the 6,143 tail, S0 convention):

| scenario | freeable | + unused tail | share |
|---|---|---|---|
| S1 keep Latin + Ethiopic | 93,360 | 99,503 | 37.96% |
| S2 + Arabic | 85,033 | 91,176 | 34.78% |
| S3 + Greek | 83,538 | 89,681 | 34.21% |
| S4 + Cyrillic + Hebrew | 69,340 | 75,483 | 28.79% |

By script (tokens): Latin 152,467 (kept); CJK ideographs 19,261 (+ 1,317 CJK/kana mixed, 1,543 hiragana, 2,035 katakana); Devanagari 13,638; Cyrillic 13,397; Bengali 9,786; Arabic 8,327 (kept); Hangul 4,678; Tamil 3,834; Thai 2,142; Malayalam 1,926; Greek 1,495 (kept); Ethiopic 847 (kept); added/special/unused 6,414. These differ from S0's S1 figure (98,275) by about 1,200 tokens because my classifier counts letter-like tokens of "other" scripts (cuneiform 706, mathematical alphanumerics 409, modifier letters) as freeable and S0 did not. **Not used**: only the unused tail is re-used.

## 2. Training text and merges (tested)

Training-side text only; FLORES+ devtest never used; FLORES+ dev not used for training (only for the fertility table "dev").

| language | sources (all loaded read-only) | size |
|---|---|---|
| amh | Wikipedia am 20231101 (CC-BY-SA) + FineTranslations amh original-language chunks (ODC-By) | 25.2 M characters, 15.4 M original tokens |
| tir | Wikipedia ti + FineTranslations tir original chunks | 25.2 M characters, 17.8 M tokens |
| fon | Wikipedia fon, `michsethowusu/english-fon_sentence-pairs_mt560` target side (cc-by-4.0; JW-style text, spacing before accents and punctuation repaired, no NFC because FLORES+ fon mixes precomposed and combining tone marks), FineWeb-2 fon_Latn (ODC-By) | 8.9 M characters (0.4 + 2.9 + 5.6) |

Merge learning (min pair count 5): Ge'ez joint 976,396 word types, last merge count 1,155 at N=3,000 (3,089 at N=1,000); Fon 33,505 word types, count 26 at N=3,000 (152 at N=1,000). No Fon merge file reaches 8,000 usefully (the data supports about 5,000 merges at count >= 3). The text is web/religious-heavy for amh/tir/fon and the Fon corpus is small; the merge lists are not tuned on any evaluation text. Simulation check (`simcheck`): the learning simulation equals the real `tokenizers` BPE on 2,459 + 2,212 Ge'ez and 1,705 Fon text lines, 0 mismatches (lines with one of the 61 new single letters are excluded because the original tokenizer spells them with bytes).

## 3. Tokenizer tests (tested)

**Identity.** 464 text sets: FLORES+ dev (233 languages) + devtest (228), English and French inside them, 18 code/LaTeX/URL/digit snippets, 30,000 Python code lines, 5,000 random digit strings = 498,055 sentences, plus the decoded string of 255,474 kept vocabulary tokens. "Plain" = sentences with no Ethiopic (Ge'ez variants) / Fon-specific (Fon variants) character and no character of a freeable script.

| variant | plain sentences | **differences** | special-char sentences changed | other-script (CJK, Hindi, Cyrillic ...) changed | vocab strings changed |
|---|---|---|---|---|---|
| geez_N1000 | 376,277 | **0** | 3,983 / 4,018 | 0 / 117,760 | 0 |
| geez_N3000 | 376,277 | **0** | 4,017 / 4,018 | 0 / 117,760 | 0 |
| fon_N1000 | 362,332 | **0** | 15,082 / 18,634 | 0 / 117,089 | 0 |
| fon_N3000 | 362,332 | **0** | 15,412 / 18,634 | 0 / 117,089 | 0 |

Project languages only (the 85 codes of `cpu_translate/language_names.json` present in FLORES+, plus eng, fra, arb; dev + devtest; sentences without the special characters): geez 99,369 sentences, fon 89,050 sentences, **0 differences**. The same test for `geez_N1000` was also run on the box (plain 376,282 sentences, 0 differences; the code lines came from the box's Python library, hence the tiny count difference); the other three were run on the laptop with the box-built tokenizers (the container's CPU quota is shared with the owner's generation job, see section 8). Full tables: `tests/tables.md`.

**Fertility** (FLORES+ devtest = measurement only; mean tokens per sentence, tokens per word in brackets; change vs original). NLLB-600M tokens per sentence from S0: amh 39.6, tir 43.4, fon 48.2.

| language | original | geez N=1000 | geez N=3000 | fon N=1000 | fon N=3000 |
|---|---|---|---|---|---|
| amh | 51.9 (3.03) | 46.4 (2.71) -10.7% | **40.3 (2.36) -22.3%** | = | = |
| tir | 66.3 (3.46) | 50.0 (2.61) -24.6% | **43.0 (2.24) -35.2%** | = | = |
| fon | 70.9 (2.34) | = | = | 53.6 (1.77) -24.4% | **51.6 (1.71) -27.2%** |
| hau, yor, swh, kin, ibo, lin, wol, eng, fra | unchanged (identical in every variant) | | | | |
| bam | 55.7 | = | = | 50.2 -10.0% | 49.2 -11.6% |
| ewe | 58.6 | = | = | 54.4 -7.1% | 53.7 -8.4% |
| twi (akua / asan) | 52.1 / 59.0 | = | = | 49.3 / 56.7 | 49.0 / 56.5 |

Same numbers on dev: amh 39.6, tir 41.7, fon 49.7 at N=3,000. Reproduction of S0's simulation at N=3,000: amh 40.3 (S0 40.6), tir 43.0 (S0 43.0) agree; **fon 51.6 vs S0 48.0 (S0 `fon`) / 49.9 (S0 `fon_strict`) does not**: S0's Fon training text included FLORES+ fon dev (same domain as devtest) and used a looser letter set; my build does not use any FLORES text for training. At N=3,000 amh reaches 1.02x, tir 0.99x and fon 1.07x NLLB's token count. Byte fallback in tir falls from 0.18% to 0.01%.

**Fon variant is not safe for other languages (tested).** The Fon-specific letters are shared. Changed sentences (devtest, N=3,000): Kabiye 1012/1012, Nuer 1010/1012, Ewe 937/991, Bambara 929/940, Twi 924/999 and 795/959, Dinka 873/982, Mossi 165/234, Kabyle 56/319, Dyula 41/44, Tamasheq 23/67; Wolof, Lingala, Luganda, Hausa, Yoruba, Igbo, French, Czech (has ě): 0 changed. Token counts fall for these (Bambara -11.6%, Ewe -8.4%, Twi -4 to -6%), so the effect is probably beneficial for fertility but it is an uncontrolled change of 8 project languages; any Fon arm must carry them in training and evaluation.

**SentencePiece / HF / GGUF.** `tokenizer.model` edit: SentencePiece ids equal the `tokenizer.json` ids on 24,108 sentences (12 languages x dev + devtest) for the original and all four variants (100%), in line with S0's 4 sets. `transformers.AutoTokenizer` loads each variant (`GemmaTokenizer`, length 262,145 incl. `<image_soft_token>`), chat template text and ids equal the original, all special ids equal except the re-used ones (`<unused99>` / `<unused6241>`), decode(encode(x)) == x on 997/997 dev sentences for amh, tir, fon, hau, yor, eng, ewe. llama.cpp's `convert_hf_to_gguf.py` (installed on the box; Python only, nothing built, the running llama-server untouched) converted `geez_N3000` and `fon_N3000` (q8_0, 285 MB each, deleted afterwards): the GGUF vocabulary has 262,144 entries, all strings equal the `tokenizer.json` strings by id, the 3,000 new pieces have type NORMAL and scores -258,504..-255,505 below every old piece (lowest old -255,504). **Not tested:** tokenisation by llama.cpp itself (no tokenizer binary present), vLLM, CTranslate2.

## 4. Model surgery (tested)

Base: SFT-final `<HF_ORG>/gemma3-270m-translate` @ `step-27638` (fp32 file, md5 4838a759...). Rule: the row of each new token (tied input/output embedding, no `lm_head` tensor) = mean, in float32, of the original embedding rows of the sub-tokens that the **original** tokenizer produces for the token's string (`▁` -> space; Ethiopic letters without a token: their 3 byte tokens), then cast to bf16. All other tensors: fp32 -> bf16 (a `--dtype fp32` option exists). Checkpoints: `gemma_mt/artifacts/tokswap_2026-10-25/model_{geez,fon}_N{1000,3000}/` (536,223,056 bytes each) and `control_*` (the same weights, original tokenizer files; "same number of rows re-initialised, original tokenizer"). The checkpoint files were built on the box and rebuilt on the laptop; **sha256 are identical** (`tests/box_model_sha256.txt`, `tests/local_model_sha256.txt`, `SHA256SUMS_artefacts.txt`). Only the laptop copies were kept; the box copies were deleted.

## 5. Model sanity (tested, CPU, bf16 weights computed in fp32, SFT-final cast to bf16 as reference)

Teacher-forced loss of `{iso} {reference}<end_of_turn>` under the exact training prompt, FLORES+ dev (300 sentences per language for N=3,000, 100 for N=1,000, 150 for the controls; the SFT data may contain FLORES+ dev text, so absolute values are not held-out generalisation). Full per-language tables: `tests/tables.md`.

| checkpoint | language | tokens/sentence | nll per token | **nll per character** |
|---|---|---|---|---|
| geez_N3000 | amh | 53.8 -> 41.7 | 2.18 -> 5.37 | **1.327 -> 2.540** |
| geez_N3000 | tir | 65.9 -> 43.8 | 2.71 -> 6.94 | **1.974 -> 3.353** |
| geez_N1000 | amh / tir | 58.0 -> 51.6 / 72.0 -> 55.6 | 2.27 -> 3.82 / 2.79 -> 5.65 | 1.386 -> 2.076 / 2.047 -> 3.200 |
| fon_N3000 | fon | 75.0 -> 56.1 | 3.95 -> 6.68 | **2.074 -> 2.628** |
| fon_N1000 | fon | 80.5 -> 61.9 | 3.99 -> 6.33 | 2.088 -> 2.550 |
| control_geez_N3000 (original tokenizer) | amh / tir | unchanged | 2.21 -> 2.84 / 2.86 -> 3.75 | 1.348 -> 1.730 / 2.090 -> 2.742 |
| control_fon_N3000 | fon | unchanged | 4.00 -> 4.58 | 2.124 -> 2.431 |

- Close but not identical, as expected, but **not close**: the untrained mean-initialised rows raise the loss on the swapped script by 0.55 to 1.4 nats/char at N=3,000 (0.46 to 1.15 at N=1,000). Part of this is a leak of probability mass, not the tokenisation: the control (original tokenizer, same rows overwritten) already loses 0.38 (amh) / 0.65 (tir) / 0.31 (fon) nats/char because the new rows look like plausible Ethiopic/Fon tokens that the original tokenizer never uses. The embedding-retraining control of the proposal addendum is therefore necessary, and its first-step loss must be reported next to the swap arm's.
- Latin text: tokenisation identical on 300/300 sentences of hau, yor, swh, kin, ibo (and fon under the Ge'ez swap; amh and tir under the Fon swap); loss over the **non-re-used ids** (restricted softmax) is exactly identical (max sentence difference 0); the **full-softmax** loss differs slightly: nll/char +0.0001 to +0.0033 for hau, swh, kin, ibo and fon under geez_N3000, +0.0015 for yor; under fon_N3000 yor +0.0117 (1.1559 -> 1.1676; one sentence +22.6 nats) and ibo +0.0020. So "identical on Latin" holds for tokenisation, restricted loss and greedy outputs, **not** for the full-softmax loss.
- Greedy translation of the first 50 FLORES+ dev English sources (25 for N=1,000 and controls), prompt of `cpu_translate/translate_cpu.py`:

| checkpoint | hau | yor | swh | amh | tir / fon |
|---|---|---|---|---|---|
| geez_N3000 | 50/50 token-identical | 50/50 | 50/50 | 40/50 (chrF++ 16.05 -> 15.88) | tir 41/50 (11.16 -> 11.23) |
| fon_N3000 | 50/50 | 50/50 | 50/50 | 50/50 | fon 49/50 (20.88 -> 20.87) |
| geez_N1000 (25) | 25/25 | 25/25 | 25/25 | 21/25 | tir 21/25 |
| control_geez_N3000 (25) | 25/25 | - | - | 18/25 | - |

  Latin-script outputs are token-identical in every case. In the swapped Ge'ez model 10 of 50 amh outputs contain a new token (0.27% of generated tokens); in the control, 7 of 25 amh outputs contain a re-initialised id that the original tokenizer then decodes as `<unusedN>`. chrF++ on n=50 is a sanity figure, not an evaluation.

## 6. Fixed factors, commands, hashes, decision rule (for the later ablation)

Fixed across arms: base checkpoint, SFT examples and order, number of steps, optimizer, LR schedule, batch size, seed(s), precision, evaluation sets and decoding settings, chat template and prompt (two spaces), non-embedding weights. Arms, metrics and the decision rule with `[OWNER TO CONFIRM]` thresholds: `reports/data/tokswap_2026-10-25/DECISION_RULE_DRAFT.txt` (tokenizer as a separate factor; embedding-re-init control mandatory; per-character loss, not per-token, when comparing tokenizers). Commands: `gemma_mt/scripts/tokswap_run_all.sh` (stages A, A2, B, C and the laptop steps). sha256: `reports/data/tokswap_2026-10-25/SHA256SUMS.txt` (every small artefact) and `SHA256SUMS_artefacts.txt` (uncompressed tokenizer.json files, checkpoints, original tokenizer). Outputs: `reports/data/tokswap_2026-10-25/{free_list,merges,tokenizers,tests,logs}`; checkpoints `gemma_mt/artifacts/tokswap_2026-10-25/` (laptop had 256 GB free).

## 7. Open points (not done)

- Retraining of the new rows (planned, not run); the loss gap above is the measured starting point. The SFT-final is stored fp32; the swapped checkpoints are bf16 (use `--dtype fp32` for an ablation that starts from full precision).
- Ge'ez N=8,000 (needs removal of other tokens), RL-final (`rl2c`) surgery (the tool is generic; not run in the main plan), a combined Ge'ez+Fon tokenizer.
- Fon merges come from 8.9 M characters of mostly JW/web text; whether they transfer to the FLORES-like domain is shown only by fertility (51.6 tokens/sentence, S0 simulation with FLORES dev text 48.0).
- llama.cpp tokenisation, vLLM and CTranslate2 loads of the swapped models; speed measurements.
- The retraining mix for the Fon arms must include Kabiye, Nuer, Ewe, Bambara, Twi, Mossi, Kabyle, Dyula.

## 8. What happened on the box (facts)

- The box is not idle: a vLLM server on the A100 (75 GB, 100%), `cs_pipeline`, a CPU `llama-server` of gemma-4-26B using about 18 cores, `push_loop.py`. The container's CPU quota is **19.2 cores** (`cpu.max` 1,920,000/100,000), not 80; load average 20-110 during my work. All my jobs ran niced.
- **Two T1 sessions worked in `/workspace/tokswap` at the same time** (notes `NOTE_T1a.txt`, `NOTE_T1b.txt`). The other session followed the pre-change plan (re-using CJK leaf tokens, which removes vocabulary entries; Ge'ez N up to 8,000; Fon up to 5,000 with merge removal) and ran three sanity processes at 250-550% CPU. Its builds are kept unchanged but separate (`reports/data/tokswap_2026-10-25/oldplan_removal_variants_T1a/`, `gemma_mt/artifacts/tokswap_2026-10-25/oldplan_removal_variants_T1a/`) and are **not** the main result; its partial results agree with this report (Latin text identical, amh loss 1.32 -> 2.08 nats/char at N=1,000 with 49.5 tokens/sentence). Its three sanity processes were killed on the coordinator's instruction. One mistake of mine: I deleted four of its text files at 06:41 UTC (my clean-up of what I took for my own duplicates), restored identical copies at about 06:47.
- Disk: the box reached 91% (6.4 GB free) with 4.5 GB in `/workspace/tokswap`. Afterwards the footprint is 260 MB: kept are tokenizers, merges, small logs, scripts, venv (27 MB). Deleted: fp32 and bf16 model copies, all checkpoints, GGUF files, training text, FLORES copy. Nothing outside `/workspace/tokswap` was touched; no process of the other pipeline was touched.
- Duration: about 05:30 to 09:00 UTC on 2026-10-26 wall-clock (most of it waiting for the shared CPU).
