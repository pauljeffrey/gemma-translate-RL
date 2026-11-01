# Weak-language selection from DEV data, final-run translator (agent W1, 2026-10-26)

Owner instruction 2026-10-26: define "weak" with dev-set data only. No FLORES+ devtest, NTREX test, MAFAND test or other test split was used for the selection; the devtest numbers in column "devtest chrF++ (post hoc)" and in section 6 were read AFTER the selection, to describe it.

## 0. Result

**Weak list (25 of 77), by the pre-registered rule (`reports/data/weak_selection_2026-10-26/DECISION_RULE.txt`, sha256 `4472d7115aaf21497dbd32bf7531b3bbd8b67e8b4243ffbccc7850d6c54c6ab7`, written before the selection generation was run):**

* **S1, 20 lowest dev chrF++ among the 45 languages that have a FLORES+ dev file** (lowest first): tir 14.4, dyu 18.4, amh 18.9, kbp 20.3, fon 20.4, nus 20.4, mos 23.4, kam 23.5, bam 24.4, umb 24.9, kea 25.3, kmb 25.6, vmw 27.2, kab 27.5, seh 29.1, yor 29.2, wol 29.9, ewe 30.2, luo 32.2, tum 32.6.
* **S2, below the cut value c20 = 32.59 on a different dev set (SALT dev, 2 languages with no FLORES+ dev):** nyn 29.3, ach 29.7.
* **S3, guard trips (computed for all 77, including the 30 with no reference):** kon (wrong-language 100%), pcm (21%), shi (13%). Read the caveat in section 5: for kon the trip is an artefact of the GlotLID label set.

**Not measurable: 27 of the 77 languages have no dev reference at all** (efi ibb urh ven nbl tiv bin dag oke bbj swc kqn lun iso loz wal tsc kua her yao sid ndc kri ish dua bas wes) and passed the guards, so they are "unmeasured", not "strong". Several are very likely among the weakest (19 of the 27 have an SFT token share below 0.3%), but the dev data cannot show it. They need a reference-free signal (QE / consensus reward) if the owner wants them in or out.

The model translates every dev sentence: finished 100% in 76 of 77 languages (ish 99%), loops 28 of 7,700 sentences in total, correct tag 7,699 of 7,700. The weakness is quality, not degeneration.

## 1. Rule (frozen before generation)

Rule file sha256 `4472d7115aaf21497dbd32bf7531b3bbd8b67e8b4243ffbccc7850d6c54c6ab7`. Deviations from the proposal in the task, all made BEFORE generating and written into the rule: the dev data cover only 45 of the 77 languages with FLORES+ (section 2), so (i) the "20 lowest" is taken among those 45 (44%, not 26%, of the scored languages), (ii) Tier B and Tier C were added, (iii) loop and wrong-language definitions were made explicit, (iv) the near-cut flag is defined on the midpoint of the 20th and 21st score. Thresholds as proposed: 20 lowest, finished < 95%, loop > 5%, wrong-language > 10%, 1,000 bootstrap resamples, seed 2026. Disclosure: before the rule was written I ran a 5-sentence smoke test of the wrapper (first 5 yor dev rows) to find the double-bos bug; nothing was selected on it.

## 2. Dev data

| tier | languages | dev set | rows |
|---|---|---|---|
| A | 45: afr amh bam bem dyu ewe fon hau ibo kab kam kbp kea kik kin kmb lin lua lug luo mlg mos nso nus nya orm run sag seh sna som sot ssw swh tir tsn tso tum twi umb vmw wol xho yor zul (Sena = `seh_Latn_MZ`; Twi = the `twi_Latn_akua1239` file) | `openlanguagedata/flores_plus` dev (997 sentences each), the SAME 100 ids for every language (random.Random(2026) from the 997) | 100 |
| B | 2: ach, nyn | `Sunbird/salt` text-all dev (496 rows), 100 drawn with the same seed. Different domain (agriculture / health, short), so only compared with the Tier-A cut value | 100 |
| C | 30 (27 unmeasured + kon pcm shi) | none exists on the Hub or on disk. Run on the same 100 FLORES+ dev English sentences; only the three guards can be computed | 100, no reference |

`pcm` (and twi) had MAFAND dev rows in `bench_dev.jsonl` on the wiped box, which is not on the laptop; `masakhane/mafand` on the Hub is only a loading script that downloads from GitHub and was not fetched, so pcm is Tier C here. Sentence ids: `sample_ids.json`; full sample `dev_sample.jsonl`.

## 3. Generation and wrapper notes

Model `cpu_translate/gemma-science-translate-final2-step401-Q8_0.gguf` (sha256 `6b4e2a2d...342a`), llama.cpp `llama-server` 0.7.0 (build 11669, brew), 5 threads of 10 cores, 8 parallel slots, greedy (temperature 0, top_k 1), max 512 new tokens, one run, 7,700 sentences in 934 s (**8.2 sentences/s**, not the 1-2/s of the docstring; dev sentences are short and the model is 270M). No tmux on the laptop: `nohup`, results appended per sentence to `gen_raw.jsonl` (sha256 `a44963e1...d0b4`), resumable.

Bugs found in `cpu_translate/translate_cpu.py` (written, never run end to end), all fixed in place except where noted:

1. **Double bos.** The prompt started with the literal text `<bos>`; llama-server parses it as the special token AND adds its own, so the model saw `[2, 2, 105, ...]` (checked with `/tokenize`). Training has one bos. The five test sentences changed output (5 of 5 different). Fixed: the literal is removed (the server adds the single bos). This changes the outputs of everybody who used the old wrapper; the docstring now says so.
2. A newline inside a translation would shift every later line of the output file: whitespace is now collapsed to one line.
3. "Finished" was inferred from `tokens_predicted < max_new`; now `stop_type == "eos"`.
4. The health-wait loop fell through silently after 120 s (then every request failed with a connection error); now exits with a message after 180 s.
5. One server start (about 3 s plus model load) per call; `weak_select_generate.py` starts one server for all 77 languages (not applied to the wrapper, which is meant for single-language use).
6. Environment: `sacrebleu` spBLEU (`flores200`) needs `sentencepiece`; the `fasttext` python `predict` breaks with numpy 2 (the scoring script calls the C++ binding). Venv at `/tmp/scratch`, not in the repo.

Scoring: chrF++ = sacrebleu 2.6.0 `CHRF(word_order=2)` corpus score on NFC text, tag stripped; spBLEU = `BLEU(tokenize="flores200")`; LID = GlotLID v3 (`cis-lmu/glotlid` model_v3.bin, already in the local HF cache) top-1 on the translation; loop and wrong-language definitions are in the rule file. Scripts: `gemma_mt/scripts/weak_select_{data,generate,pairflags,score,report}.py`.

## 4. Ranking of all 77 languages

Columns: rank 1 = lowest chrF++ among the 45 Tier-A languages; B and C have no rank (see section 2). P(in 20 lowest) = share of the 1,000 paired bootstrap resamples (same resampled sentences for all Tier-A languages) in which the language is among the 20 lowest. "near cut" = its 95% CI contains the midpoint 32.64 of c20 = 32.59 and c21 = 32.70. No-pair chrF++ = score on the dev rows whose reference translation was not in the SFT text (only available for 17 languages, section 6); the bracket is the share of the 100 rows whose pair WAS seen. Devtest column = final2 step-401 on FLORES+ devtest, all 1,012 rows, from `reports/data/final2_eval_2026-10-23/per_language_bench.csv` (not used for selection; seh has no FLORES+ devtest row).

| rank | iso | language | tier | chrF++ [95% CI] | P(in 20 lowest) | near cut | spBLEU | finished % | loop % | wrong-lang % | no-pair chrF++ (pair-seen %) | SFT token share % | devtest chrF++ (post hoc) | weak | why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | tir | Tigrinya | A | 14.4 [13.3, 15.6] | 1.00 |  | 4.2 | 100 | 0 | 0 | n/a | 0.42 | 13.9 | **weak** | S1 bottom-20 chrF++ |
| 2 | dyu | Dyula | A | 18.4 [17.2, 19.8] | 1.00 |  | 2.7 | 100 | 0 | 5 | n/a | 0.04 | 18.1 | **weak** | S1 bottom-20 chrF++ |
| 3 | amh | Amharic | A | 18.9 [17.5, 20.2] | 1.00 |  | 6.7 | 100 | 1 | 0 | n/a | 0.63 | 20.0 | **weak** | S1 bottom-20 chrF++ |
| 4 | kbp | Kabiye | A | 20.3 [19.4, 21.4] | 1.00 |  | 2.4 | 100 | 0 | 0 | n/a | 0.03 | 22.2 | **weak** | S1 bottom-20 chrF++ |
| 5 | fon | Fon | A | 20.4 [19.4, 21.5] | 1.00 |  | 4.2 | 100 | 0 | 0 | 20.4 (0) | 2.58 | 21.6 | **weak** | S1 bottom-20 chrF++ |
| 6 | nus | Nuer | A | 20.4 [19.5, 21.4] | 1.00 |  | 6.2 | 100 | 0 | 0 | n/a | 0.71 | 20.9 | **weak** | S1 bottom-20 chrF++ |
| 7 | mos | Mossi | A | 23.4 [22.0, 24.9] | 1.00 |  | 4.4 | 100 | 0 | 0 | n/a | 0.36 | 22.9 | **weak** | S1 bottom-20 chrF++ |
| 8 | kam | Kamba | A | 23.5 [22.0, 25.0] | 1.00 |  | 4.9 | 100 | 0 | 0 | n/a | 0.14 | 25.0 | **weak** | S1 bottom-20 chrF++ |
| 9 | bam | Bambara | A | 24.4 [23.1, 25.7] | 1.00 |  | 4.5 | 100 | 0 | 5 | n/a | 0.03 | 24.7 | **weak** | S1 bottom-20 chrF++ |
| 10 | umb | Umbundu | A | 24.9 [23.6, 26.4] | 1.00 |  | 2.6 | 100 | 0 | 0 | n/a | 0.28 | 25.4 | **weak** | S1 bottom-20 chrF++ |
| 11 | kea | Kabuverdianu | A | 25.3 [24.1, 26.7] | 1.00 |  | 5.2 | 100 | 2 | 2 | n/a | 0.01 | 26.3 | **weak** | S1 bottom-20 chrF++ |
| 12 | kmb | Kimbundu | A | 25.6 [24.7, 26.5] | 1.00 |  | 3.3 | 100 | 0 | 0 | n/a | 0.19 | 25.8 | **weak** | S1 bottom-20 chrF++ |
| 13 | vmw | Makhuwa | A | 27.2 [26.1, 28.3] | 1.00 |  | 5.4 | 100 | 0 | 0 | n/a | 0.16 | 26.9 | **weak** | S1 bottom-20 chrF++ |
| 14 | kab | Kabyle | A | 27.5 [26.2, 28.9] | 1.00 |  | 8.4 | 100 | 0 | 0 | n/a | 0.29 | 26.7 | **weak** | S1 bottom-20 chrF++ |
| 15 | seh | Sena | A | 29.1 [27.9, 30.4] | 1.00 |  | 5.2 | 100 | 0 | 1 | n/a | 0.11 | - | **weak** | S1 bottom-20 chrF++ |
| 16 | yor | Yoruba | A | 29.2 [26.8, 31.7] | 1.00 |  | 13.0 | 100 | 0 | 0 | 26.7 (28) | 11.01 | 30.4 | **weak** | S1 bottom-20 chrF++ |
| 17 | wol | Wolof | A | 29.9 [28.1, 31.7] | 1.00 |  | 10.7 | 100 | 1 | 0 | 29.9 (0) | 0.38 | 29.9 | **weak** | S1 bottom-20 chrF++ |
| 18 | ewe | Ewe | A | 30.2 [28.6, 32.1] | 1.00 |  | 9.2 | 100 | 0 | 0 | 30.2 (0) | 0.23 | 31.5 | **weak** | S1 bottom-20 chrF++ |
| 19 | luo | Luo | A | 32.2 [30.4, 34.1] | 0.69 | yes | 9.2 | 100 | 0 | 0 | n/a | 1.30 | 32.6 | **weak** | S1 bottom-20 chrF++ |
| 20 | tum | Tumbuka | A | 32.6 [30.7, 34.3] | 0.46 | yes | 8.6 | 100 | 0 | 0 | n/a | 0.29 | 33.8 | **weak** | S1 bottom-20 chrF++ |
| 21 | orm | Oromo | A | 32.7 [31.4, 34.1] | 0.38 | yes | 7.7 | 100 | 0 | 0 | 32.7 (0) | 1.10 | 31.7 |  |  |
| 22 | kik | Kikuyu | A | 33.1 [31.5, 34.9] | 0.15 | yes | 9.9 | 100 | 1 | 0 | n/a | 0.26 | 32.1 |  |  |
| 23 | lua | Luba-Kasai | A | 33.2 [31.8, 34.7] | 0.16 | yes | 7.6 | 100 | 1 | 0 | n/a | 0.66 | 33.4 |  |  |
| 24 | run | Kirundi | A | 33.3 [31.7, 35.0] | 0.11 | yes | 8.9 | 100 | 0 | 5 | n/a | 0.31 | 34.2 |  |  |
| 25 | bem | Bemba | A | 33.6 [32.0, 35.1] | 0.04 | yes | 8.5 | 100 | 0 | 0 | n/a | 0.38 | 34.3 |  |  |
| 26 | sag | Sango | A | 33.9 [32.7, 35.2] | 0.01 |  | 8.3 | 100 | 0 | 0 | n/a | 0.30 | 34.9 |  |  |
| 27 | lug | Luganda | A | 34.5 [32.9, 36.3] | 0.00 |  | 9.4 | 100 | 0 | 0 | n/a | 0.97 | 35.8 |  |  |
| 28 | twi | Twi | A | 38.1 [35.9, 40.3] | 0.00 |  | 15.9 | 100 | 0 | 0 | n/a | 12.58 | 37.4 |  |  |
| 29 | ssw | Swati | A | 38.9 [37.3, 40.6] | 0.00 |  | 13.4 | 100 | 0 | 0 | n/a | 0.27 | 38.0 |  |  |
| 30 | sna | Shona | A | 39.8 [37.7, 42.0] | 0.00 |  | 14.8 | 100 | 0 | 0 | 39.8 (0) | 1.20 | 39.6 |  |  |
| 31 | kin | Kinyarwanda | A | 40.2 [38.2, 42.3] | 0.00 |  | 15.5 | 100 | 1 | 7 | 40.2 (0) | 2.19 | 42.1 |  |  |
| 32 | ibo | Igbo | A | 40.5 [38.3, 42.8] | 0.00 |  | 18.8 | 100 | 0 | 0 | 34.0 (81) | 19.75 | 41.2 |  |  |
| 33 | som | Somali | A | 41.9 [40.2, 43.7] | 0.00 |  | 16.9 | 100 | 0 | 0 | 41.9 (0) | 2.87 | 40.7 |  |  |
| 34 | lin | Lingala | A | 42.9 [41.0, 44.7] | 0.00 |  | 15.2 | 100 | 0 | 0 | n/a | 0.29 | 42.6 |  |  |
| 35 | nya | Chichewa | A | 43.0 [41.3, 44.9] | 0.00 |  | 14.7 | 100 | 0 | 0 | 43.0 (0) | 1.59 | 43.4 |  |  |
| 36 | mlg | Malagasy | A | 44.2 [42.5, 45.8] | 0.00 |  | 17.1 | 100 | 0 | 0 | n/a | 0.77 | 43.5 |  |  |
| 37 | xho | Xhosa | A | 44.4 [42.4, 46.7] | 0.00 |  | 19.1 | 100 | 0 | 0 | 44.4 (0) | 3.54 | 45.2 |  |  |
| 38 | tso | Tsonga | A | 44.6 [42.5, 46.7] | 0.00 |  | 19.8 | 100 | 0 | 0 | n/a | 0.68 | 46.0 |  |  |
| 39 | sot | Sesotho | A | 44.7 [42.8, 46.5] | 0.00 |  | 18.9 | 100 | 0 | 0 | 44.7 (0) | 1.88 | 45.5 |  |  |
| 40 | tsn | Setswana | A | 45.6 [43.2, 47.9] | 0.00 |  | 22.1 | 100 | 0 | 0 | 46.1 (6) | 2.60 | 46.2 |  |  |
| 41 | hau | Hausa | A | 49.2 [46.8, 51.4] | 0.00 |  | 24.0 | 100 | 0 | 0 | 49.4 (37) | 10.24 | 50.1 |  |  |
| 42 | zul | Zulu | A | 49.7 [47.5, 51.8] | 0.00 |  | 26.3 | 100 | 0 | 1 | 49.6 (4) | 2.55 | 49.4 |  |  |
| 43 | nso | Sepedi | A | 51.7 [49.5, 53.9] | 0.00 |  | 27.4 | 100 | 0 | 0 | n/a | 0.79 | 49.9 |  |  |
| 44 | swh | Swahili | A | 59.0 [56.8, 61.0] | 0.00 |  | 36.5 | 100 | 0 | 6 | 59.0 (0) | 2.68 | 58.6 |  |  |
| 45 | afr | Afrikaans | A | 64.9 [62.8, 67.0] | 0.00 |  | 44.0 | 100 | 0 | 0 | 64.9 (0) | 2.52 | 63.6 |  |  |
| B | nyn | Nyankore | B | 29.3 [27.2, 31.4] | - |  | 5.8 | 100 | 0 | 5 | n/a | 0.11 | - | **weak** | S2 below c20 |
| B | ach | Acholi | B | 29.7 [27.4, 31.9] | - |  | 9.0 | 100 | 0 | 4 | n/a | 0.14 | - | **weak** | S2 below c20 |
| C | kon | Kongo | C | n/a (no reference) | - |  | - | 100 | 4 | 100 | - | 0.01 | - | **weak** | S3 wrong-lang>10% |
| C | pcm | Pidgin | C | n/a (no reference) | - |  | - | 100 | 0 | 21 | - | 0.81 | - | **weak** | S3 wrong-lang>10% |
| C | shi | Tachelhit | C | n/a (no reference) | - |  | - | 100 | 0 | 13 | - | 0.07 | - | **weak** | S3 wrong-lang>10% |
| C | wes | Cameroon Pidgin | C | n/a (no reference) | - |  | - | 100 | 0 | 7 | - | 0.03 | - |  |  |
| C | nbl | South Ndebele | C | n/a (no reference) | - |  | - | 100 | 0 | 6 | - | 0.31 | - |  |  |
| C | dag | Dagbani | C | n/a (no reference) | - |  | - | 100 | 0 | 1 | - | 0.15 | - |  |  |
| C | efi | Efik | C | n/a (no reference) | - |  | - | 100 | 0 | 0 | - | 1.19 | - |  |  |
| C | ibb | Ibibio | C | n/a (no reference) | - |  | - | 100 | 1 | 0 | - | 0.01 | - |  |  |
| C | urh | Urhobo | C | n/a (no reference) | - |  | - | 100 | 0 | 0 | - | 0.09 | - |  |  |
| C | ven | Venda | C | n/a (no reference) | - |  | - | 100 | 0 | 0 | - | 0.37 | - |  |  |
| C | tiv | Tiv | C | n/a (no reference) | - |  | - | 100 | 0 | 0 | - | 0.32 | - |  |  |
| C | bin | Edo | C | n/a (no reference) | - |  | - | 100 | 0 | 0 | - | 0.13 | - |  |  |
| C | oke | Okpe | C | n/a (no reference) | - |  | - | 100 | 0 | 0 | - | 0.06 | - |  |  |
| C | bbj | Ghomala | C | n/a (no reference) | - |  | - | 100 | 2 | 0 | - | 0.03 | - |  |  |
| C | swc | Congo Swahili | C | n/a (no reference) | - |  | - | 100 | 0 | 0 | - | 0.51 | - |  |  |
| C | kqn | Kaonde | C | n/a (no reference) | - |  | - | 100 | 0 | 0 | - | 0.48 | - |  |  |
| C | lun | Lunda | C | n/a (no reference) | - |  | - | 100 | 1 | 0 | - | 0.29 | - |  |  |
| C | iso | Isoko | C | n/a (no reference) | - |  | - | 100 | 2 | 0 | - | 0.56 | - |  |  |
| C | loz | Lozi | C | n/a (no reference) | - |  | - | 100 | 0 | 0 | - | 0.95 | - |  |  |
| C | wal | Wolaytta | C | n/a (no reference) | - |  | - | 100 | 1 | 0 | - | 0.29 | - |  |  |
| C | tsc | Tswa | C | n/a (no reference) | - |  | - | 100 | 0 | 0 | - | 0.19 | - |  |  |
| C | kua | Kuanyama | C | n/a (no reference) | - |  | - | 100 | 0 | 0 | - | 0.23 | - |  |  |
| C | her | Herero | C | n/a (no reference) | - |  | - | 100 | 0 | 0 | - | 0.02 | - |  |  |
| C | yao | Yao | C | n/a (no reference) | - |  | - | 100 | 0 | 0 | - | 0.10 | - |  |  |
| C | sid | Sidamo | C | n/a (no reference) | - |  | - | 100 | 2 | 0 | - | 0.10 | - |  |  |
| C | ndc | Ndau | C | n/a (no reference) | - |  | - | 100 | 0 | 0 | - | 0.06 | - |  |  |
| C | kri | Krio | C | n/a (no reference) | - |  | - | 100 | 1 | 0 | - | 0.10 | - |  |  |
| C | ish | Esan | C | n/a (no reference) | - |  | - | 99 | 1 | 0 | - | 0.01 | - |  |  |
| C | dua | Duala | C | n/a (no reference) | - |  | - | 100 | 5 | 0 | - | 0.04 | - |  |  |
| C | bas | Basaa | C | n/a (no reference) | - |  | - | 100 | 1 | 0 | - | 0.06 | - |  |  |

## 5. Noise at the cut

* c20 = 32.59 (tum, rank 20), c21 = 32.70 (orm): the cut falls between two languages 0.1 chrF++ apart. With 100 sentences the 95% CI of a language is about +-1.3 to +-2.4 chrF++ wide, so **seven languages are inside the noise of the cut**: luo (rank 19, P(in 20 lowest) 0.68, in the list), tum (rank 20, 0.45, in), orm (21, 0.38, out), kik (22, 0.15, out), lua (23, 0.15, out), run (24, 0.10, out), bem (25, 0.04, out). The 18 languages ranked 1-18 are in the 20 lowest in at least 99% of the resamples; the languages from rank 26 up are in at most 1% of them. If the owner wants only robust members, use the first 18 (P >= 0.99) and treat luo, tum, orm as undecided.
* **Guard trips need reading.** kon: GlotLID labels our Kongo output as `kwy_Latn` 74, `kng_Latn` 19, `ktu_Latn` 4 of 100, and `kon_Latn` 0 times; these are Kongo-cluster labels and `kon_Latn` is evidently not what GlotLID emits for this text, so the 100% is a label-set artefact (sibling-tolerant rate 97%); the rule as written (frozen) still lists kon. pcm: 19 of 100 outputs are labelled English and 1 `kri`; Pidgin and English are close for GlotLID, so this may be partly a LID limit, but a model that answers in English is also plausible; unresolved without a reference. shi: 12 of 100 are labelled Kabyle (Berber sibling), 1 Maltese. nyn and ach (S2) rest on one different dev set: ranking them against FLORES-domain cut values is a domain comparison, not a paired one.
* Domain: FLORES+ is Wikinews/Wikijunior/Wikivoyage; the final model was trained with an English side that includes these sentences (section 6).

## 6. Contamination of dev, and what it does to this list

* **English side.** `research_archive_2026-10-23/box/other/sft_build/bench/contamination_dev.json` (scan of 2026-10-18 over `gemma/windows_sft`, 11,442,172 windows) says that **984 of the 997 FLORES+ dev English sentences (98.7%) occur in the SFT text; 99 of my 100 sampled sentences do.** The flag is identical for every language (checked across the 17 scanned languages). A "clean" dev subset therefore does not exist (1 sentence). Dev has the same problem as devtest (95.3% there). The model has seen the English sentences paired with other languages, not with the dev reference.
* **Reference pair.** The same scan has a per-row "pair seen" flag (the reference translation is in the target side of a window of the same language), but only for 17 FLORES+ dev languages: afr ewe fon hau ibo kin nya orm sna som sot swh tsn wol xho yor zul. Pair seen in my 100 rows: ibo 81, hau 37, yor 28, tsn 6, zul 4, all others 0. Dev scores for ibo, hau, yor are therefore inflated by direct pair exposure; recomputed on the unseen-pair rows: ibo 40.5 -> 34.0 (19 rows), hau 49.2 -> 49.4 (63 rows), yor 29.2 -> 26.7 (72 rows). Effect on the list: yor stays in (lower without the seen rows); ibo would still be outside the 20 lowest (34.0 is above c21 32.7); nothing else changes.
* **For the other 28 Tier-A languages the pair flag is unknown.** The windows are not on disk and not on the Hub (`<HF_ORG>/sft-new` and `<HF_ORG_LEGACY>/sft-all` are the older SabiYarn SFT builds, not the Gemma windows; the `interlaanguage-translation` pairs were not scanned), and the old scan did not cover amh, tir, lin, lug, mlg, bem, ... So **I could not build a no_pair / clean flag for dev for them**. The pair flags that exist are 0 for 12 of the 17 scanned languages, which suggests small pair exposure, but that is an inference, not a check.
* **Consequence.** Every dev score is optimistic, most for languages with heavy SFT coverage of FLORES sentences (the English-only exposure helps through translation of the same sentence into related languages and through the memorised English). This biases the ranking toward calling data-poor languages weak: SFT token share and dev chrF++ are correlated (Spearman 0.60 over the 44 languages with a share and a devtest score; the same 0.59 on devtest), and the median SFT token share is 0.28% in the weak Tier-A set against 1.20% in the rest. Part of that correlation is genuine (less data, worse model) and part may be contamination; the data cannot separate them. The share is given as an information column and was not used.
* Post hoc (devtest, not used for selection): dev and devtest chrF++ ranks of the 44 Tier-A languages that have both agree with Spearman 0.996, mean |dev - devtest| 0.8 chrF++, mean dev - devtest -0.3; 19 of the 20 lowest on dev are among the 20 lowest on devtest (dev has tum where devtest has kik). Dev is as contaminated as devtest on the English side (98.7% vs 95.3%) and selects almost the same set; that is a statement about consistency, not about freedom from contamination.

## 7. Limits

* Only 45 of 77 languages can be ranked, 2 more can be compared with the cut; 27 are unmeasured. The selection is blind to the languages that are probably worst.
* 100 sentences per language: CI +-1.3 to 2.4 chrF++; seven languages sit inside the noise at the cut.
* Greedy decoding, Q8_0 GGUF (not the bf16 checkpoint; the Q8_0 versus bf16 difference was not measured here), one run, double-bos bug fixed (a user of the old wrapper sees different outputs).
* chrF++ at 100 sentences is a coarse measure for Ge'ez-script and no-space languages; amh and tir are low partly for script/segmentation reasons already known from the devtest reports.
* S2 and S3 membership rests on a different domain (SALT) and on GlotLID label sets (kon, pcm, shi); treat these 5 as lower-confidence members than the 18 robust S1 languages.
* No Hub writes, no paid API, no TranslatePsy/NLLB data. Hub reads: FLORES+ dev files, SALT dev (Sunbird/salt), repo file listings.

## 8. Files

`reports/data/weak_selection_2026-10-26/`: `DECISION_RULE.txt`, `dev_sample.jsonl`, `sample_ids.json`, `gen_raw.jsonl` (raw model output), `per_sentence.jsonl`, `per_language.csv`, `weak_list.json`, `ranking_table.md`, `pair_flags.json`, `contamination_dev_sample.json`, `generate.log`. Scripts `gemma_mt/scripts/weak_select_{data,generate,pairflags,score,report}.py`; fixed wrapper `cpu_translate/translate_cpu.py`.
