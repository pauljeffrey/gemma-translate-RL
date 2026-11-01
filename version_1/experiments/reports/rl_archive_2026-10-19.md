# The best-translation archive of the Gemma RL (D-41, 2026-10-19)

Owner request: "when rewards start getting good, close to +1, it might make sense to start saving the translations for later review."
Everything lives in `gemma_mt/`; the archive only READS the output of the reward, it never changes a reward, an advantage or an update.
(Companion of `rl_reward_explained_2026-10-19_v3.md`, whose section 31 points here.)

## 1. What is stored

One SQLite file (WAL mode; one writer thread; crash-safe, reopened and continued by every restart / OOM restart of the trainer). One row per SOURCE:

| column | meaning |
|---|---|
| `key` | first 16 hex digits of `sha1(iso + "\n" + NFC(source))` (NFC and NFD spellings of a source are the same key; the same English text for two languages is two keys) |
| `iso`, `mode` | target language and its NOMINAL content mode (`qe` / `blend` / `consensus`, D-38) |
| `source`, `translation` | the NFC English source, and exactly the text the reward scored: cut at the first stop id, special strings scrubbed, NFC, language tag stripped (`RewardOutput.texts`) |
| `reward` | the reward the sample got (after the length / lexical / diacritic penalties) |
| `qe` | mean raw AfriCOMET-QE of its scored chunks (NULL in consensus mode) |
| `consensus_chrf` | mean chrF++ (0-100) against the valid siblings of its GRPO group (NULL for `qe` languages and when the group had < 3 valid samples) |
| `diacritic_deficit` | source-weighted mean diacritic deficit (NULL = the language does not use diacritics in the calibration; stored entries always have 0) |
| `lid_ok`, `lid_label` | 1 and GlotLID's top-1 label when the text was long enough to be checked; NULL when it was not (below the language's `min_chars`, or the guard is off for the language) |
| `n_chunks` | chunks of the sample's plan (all scored: a stored entry has no failed chunk) |
| `step_first`, `step_best` | logged step number (1-based, as in `metrics.jsonl`) when the row was created, and when the stored translation was found |
| `length_bin`, `stage` | the prompt's length bin (b64 .. b1024) and the curriculum stage of the step that found the stored best (`b64` .. `b1024`, `mixed`) |
| `times_seen` | sampled completions of this source (eligible or not) since the row exists |
| `n_improved` | how often the stored translation was replaced by a better one |

Two more tables: `counters` (samples looked at, eligible, per-language seen / eligible, rejection reasons: the data to tune the thresholds) and `meta` (schema version, thresholds of the last start).

## 2. Which samples may enter or replace an entry

All of: status `ok` (none of the ten failure classes) and a non-empty translation; **no failed chunk** (omitted / extra / copied sentence, D-30); GlotLID passed wherever the guard is on (status `ok` already implies it; the text is simply not checked below the language's minimum length, `lid_ok` is then NULL); **diacritic_deficit == 0** (languages with diacritics in the calibration: bam ewe fon hau ibo kab kbp kik kin mlg mos nso nus tum twi umb wol yor); reward at least the threshold of the language's nominal mode:

| mode | threshold | why |
|---|---|---|
| `qe` | **1.0** | +1.0 is the language's median-human QE anchor `t_mid` (section 6 of the v3 report): the translation scores at least like the median human reference, and it survived the length / lexical / diacritic penalties |
| `blend` | **1.0** and consensus chrF++ >= **50** | reward is half QE, half consensus; blend languages are the ones whose QE is weak, so a reward of 1.0 that rests on the QE half alone must not qualify. 50 = between "heavy noise" (46-51 in the D-38 discrimination test) and "light noise" (66-71) between siblings of a good group |
| `consensus` | **0.5** | reward = -2 + 4.25 * chrF/100, so 0.5 means chrF++ >= 58.8 against the siblings; no QE, no calibrated anchor to refer to, hence the lower bar |

A blend or consensus language whose group had no consensus (fewer than 3 valid samples) is never eligible, even if the reward fell back to QE alone (`no_consensus`).
It REPLACES the stored translation only if its reward is higher by more than `epsilon` = 0.001 AND its text differs (the same text scored again in another fp16 batch is noise, not an improvement). `times_seen` is increased by every sample.
The 8 samples of a group share a source, so one step touches each key once (best eligible sample of the group wins).

Honest limits: (a) a `qe` reward is a function of the text alone, so rewards are comparable across steps; a `consensus` / `blend` reward also depends on the siblings of the group it was drawn in, so for those languages "best" means "best reward seen", a noisier ranking, and a consensus-heavy policy drifting towards its own mode raises the reward without the text being better. (b) QE is the same metric the policy is trained on; the archive is a candidate list for human review, not a certificate (hence `review`). (c) Rows exist only for sources that were eligible at least once; `times_seen` does not count the visits before the first eligible one. (d) A restart from `last/` replays steps with the same seed, so `times_seen` of those sources can be counted twice. (e) A hard kill loses at most `archive_flush_seconds` (20 s) of rows; normal exits, exceptions and OOM exits flush (`atexit` + `close()`). (f) Evaluation rollouts (greedy, held-out prompts) are not archived.

## 3. Where it hooks in, and what it costs

`train_rl.py`, in the step, right after the reward of the batch was computed and asserted (before the advantages): `archive_step(archive, reward, samples, out, step + 1, stage, prompt_bins, group_size)` builds one tuple per sample from fields that already exist (`RewardOutput.texts/rewards/status/qe/diacritic_deficit/lid_label/consensus_chrf` and two new per-sample lists, `n_chunks` and `failed_chunks`, which `TranslationReward` already knew) and does ONE `queue.put_nowait`. Hashing, eligibility, the UPSERT rule and the SQL run in the writer thread. A full queue (`archive_queue_size` = 64 steps) drops the step's rows and counts them; the training step never waits.
Logged every step: `archive/size`, `archive/inserted`, `archive/improved`, `archive/dropped_queue` (plus `archive/errors`, `archive/eligible_rate`, `archive/mean_best_reward`, `archive/submit_ms`, `archive/hook_ms`), and appended to the console line.
A failing write (disk full ...) is rolled back per step, counted in `archive/errors`, printed once, and never stops the run; a bad `archive_path` stops it at start-up.
Measured (1,000 samples = 125 groups of 8, 7 languages, laptop CPU): hook on the training thread median 0.26 ms (p95 0.37 ms); writer thread ~1 ms per batch of 1,000 samples; the reward call itself (fake scorer) is ~94 ms. The limit was 50 ms. On the rental CPU (shared with the running trainer, `nice 19`): hook median 0.95 ms (p95 1.16 ms), writer 4 ms per batch, reward call 194 ms.

## 4. Config (RLConfig, `rl.yaml` OFF, `rl_rtxpro_phase2.yaml` ON)

| key | default | meaning |
|---|---|---|
| `archive` | false | on / off |
| `archive_path` | null = `<out_dir>/archive/archive.sqlite` | phase 2: `/workspace/sft_build/runs/rl2/archive/archive.sqlite` |
| `archive_min_reward_by_mode` | `{qe: 1.0, blend: 1.0, consensus: 0.5}` | merged key by key over these defaults (also for `key=value` overrides) |
| `archive_min_consensus_chrf` | 50.0 | blend languages only |
| `archive_flush_seconds` | 20 | commit interval of the writer |
| `archive_queue_size` | 64 | steps queued for the writer |

## 5. How to review it (stdlib only, runs while training writes)

```
python -m gemma_mt.rl.archive stats  /workspace/sft_build/runs/rl2/archive/archive.sqlite            # rows per language / mode / stage / bin, mean best reward, improvements, rejection reasons
python -m gemma_mt.rl.archive export /workspace/sft_build/runs/rl2/archive/archive.sqlite            # <dir>/export/archive.parquet + archive.jsonl, all columns (parquet needs pyarrow: on the rental yes)
python -m gemma_mt.rl.archive review /workspace/sft_build/runs/rl2/archive/archive.sqlite --out review.md   # 3 random entries per language as a markdown table (source, translation, reward); --per-language N, --seed S, --languages yor,hau, --min-reward R
```
`stats` also prints the eligible rate per language: a language with rate 0.00 after the first 100 steps is one whose rewards do not reach its threshold yet (or whose guard fires), not a bug.

## 6. Tests (`tests/test_archive.py`, CPU, no model except the existing tiny-Gemma smoke)

Key (sha1 definition, NFC/NFD stability, iso separation); 17 eligibility cases; insert / improve / ineligible / dedupe in one run (epsilon, same text, failed chunk, diacritic deficit, wrong language never replace); stored source is NFC; counters table; flush on `close()` without a flush call; periodic commit; queue overflow with a stalled writer (`submit` returns at once, drops and counts exactly); a failing batch rolled back and the writer continues; crash-restart (a child process writes, flushes, queues one more batch and `os._exit`s; the parent reopens, `integrity_check` ok, improves one row, adds one); the hook on a real `TranslationReward` (qe / blend / consensus languages, tag stripped, dedupe of 8 samples, blend language without consensus rejected); overhead < 50 ms for 1,000 samples; CLI stats / export (jsonl, parquet when pyarrow is present) / review (<= N per language, deterministic, pipes and newlines escaped), `python -m gemma_mt.rl.archive` without the training stack; config defaults, validation, override merge, yaml files; two `run_smoke` runs with the archive on (metrics carry `archive/*`, second run reopens the file) and one with the default config (no archive created).
