# Language-coverage claim check (2026-10-22, agent E2-lit)

Claim under test: "our model is one of the first to support close to 77 languages other than NLLB".
Full machine-readable table: `language_coverage_claim_check.csv` (35 rows, with URLs, notes and a `verified` flag). Everything below was checked in this session unless flagged "unverified".

## Our own numbers (from `gemma_mt/gemma_mt/languages.py`)

* 85 table rows; **77 RL/SFT target languages** (`SUPPORTED_RL`). 8 rows (nde fat knc gaa men luy tzm ful) have no SFT data and are not counted.
* **44 of the 77 are in FLORES-200** (official list of `facebookresearch/flores`, flores200/README.md): yor hau ibo twi ewe fon wol swh kin som orm amh tir lug lin zul xho afr sna nya sot tsn mlg bem tum run tso nso ssw sag umb kmb mos kik luo kam dyu bam kbp kea kon kab nus lua. **33 are not**: pcm efi ibb urh ven nbl tiv bin dag oke shi bbj swc kqn lun iso loz wal ach tsc vmw kua seh nyn her yao sid ndc kri ish dua bas wes. (44 also matches the NLLB-3.3B rows of the baselines csv.)
* 44 have a FLORES+ config in `languages.py` (vmw is FLORES+-only, kon has no FLORES+ file); 64 of 77 have some public en->xx test data (FLORES+, NTREX, MAFAND, SMOL, IbomMT, FLORES-MOZ, GlotStoryBook, SALT).
* 270M parameters (Gemma-3-270m base), 2,048-token documents.

## Evidence table (African = sub-Saharan, Malagasy, Berber, plus Arabic dialects where the source counts them; counts are as printed by the source unless marked "my tally")

| system | params | open weights | African langs supported (training/claimed) | African langs evaluated en->X | context | key evidence |
|---|---|---|---|---|---|---|
| **NLLB-200** | 0.6B-3.3B dense, 54.5B MoE | yes | **55** (paper: "55 are spoken in the African continent"); FLORES-200 list = 51 non-Arabic + aeb ary arz (+ MSA) ; 44 are ours | all 55 (FLORES-200 devtest, chrF++/spBLEU) | trained max 512 tokens | arXiv 2207.04672; flores200/README.md |
| **MADLAD-400 MT** | 3B / 7.2B / 10.7B | yes | **about 85** African entries among 419 (my tally of the Appendix table, +-5, includes variants); 45 of ours match exactly | aggregate FLORES-200/WMT only | not extracted | arXiv 2309.04662 (Sept 2023) |
| **TranslateGemma** | 4B / 12B / 27B | yes | SFT-data tables (Appendix Tables 5-6) name **about 60** distinct African languages (my tally); 46 of ours appear; Zulu absent from the tables yet evaluated | **2** (sw_KE/sw_TZ, zu_ZA; MetricX/COMET-22 on WMT24++) | 2K (card) | arXiv 2601.09012; HF card |
| **Omnilingual MT (Meta, Mar 2026)** | 1B-8B | unverified | **1,600+** languages total (source), ~1,200 as target; African count not extracted | Bible partition 1,560 languages (xCOMET/chrF++); African examples bam mos wol yor lug | not extracted | arXiv 2603.16309 |
| **Apertus** | 8B / 70B | yes | pretraining 1,811 languages; post-training 149; 94 evaluated "incl. many African" (non-MT) | none for MT | 65,536 | arXiv 2509.14233 |
| **Google Translate** | closed | no | 2022: +10 African; 2024: "about a quarter" of 110 new (8 named); total unknown | none public | - | blog.google (2024), droid-life (2022) |
| **Gemini 2.5 Flash** | closed | no | 100+ languages claimed, list unpublished | all FLORES-200 African variants, n=10, chrF | - | arXiv 2507.16809 |
| **GPT-4o** | closed | no | no claim | AfroBench pooled | - | arXiv 2311.07978 |
| **AfriqueLLM** (Llama/Gemma/Qwen) | 4B-14B | yes | **20** (3 are Arabic dialects) + 4 high-resource; 50-language variant | **24** (21 ours), SSA-COMET and chrF++ | CPT 4k-32k | arXiv 2601.06395 |
| **TranslatePsy-AfriSLM** | 0.8B / 2B / 4B | yes | **19** (+8 held-out OOD languages) | 19, both directions macro only | <= 2,048 train filter | arXiv 2608.18655; HF card |
| **Tiny Aya** | 3.35B | yes | **10** of 70 | 10 (FLORES chrF, en->xx) | 8,192 | arXiv 2603.11510 |
| **Aya-101** | 13B | yes | **13** (mT5 list) | AfroBench pooled | 1,024 | arXiv 2402.07827 |
| **Aya Expanse / Aya-23** | 8B-35B | yes | **0** sub-Saharan (23 languages) | - | 8,192 | HF card |
| **Hunyuan-MT-7B** | 7B | yes | **0** (33 + 5 Chinese minority) | 0 | not extracted | arXiv 2509.05209 |
| **AfriNLLB** | 548M-600M | yes | **10** | 9 en<->xx pairs (+ fr<->lin, wol), FLORES-200 devtest | sentence | arXiv 2602.09373 |
| **InkubaLM** | 0.4B | yes | **5** (sw yo xh ha zu) | AfroBench-type tasks | not extracted | arXiv 2408.17024 |
| **Lugha-Llama** | 8B | yes | **16** (WURA) | via AfriqueLLM tables | 128k | arXiv 2504.06536 |
| **Qwen3** | 0.6B-235B | yes | **2** sub-Saharan (Swahili, Kabuverdianu) of 119 (+8 Arabic varieties) | Belebele only | 32k | Qwen blog; arXiv 2505.09388 |
| **Qwen3.5** | 0.8B-122B | yes | "201 languages", no list | via TranslatePsy | 262k | HF card |
| **Llama 3.1** | 8B-405B | yes | **0** official (8 languages) | non-official | 128k | Meta model card |
| **Gemma 3** | 1B-27B | yes | "over 140", no list | via AfriqueLLM | 32k-128k | HF card |
| **SeamlessM4T v2** | 2.3B | yes (NC) | 15 text (2 are Arabic dialects) of 96 | not extracted | not extracted | HF card (unverified summary) |
| **mBART-50** | 0.6B | yes | **3** (af sw xh) | - | - | HF card |
| **mT5/ByT5/mT0** | <=13B | yes | **13** | - | - | Aya paper appendix |
| **BLOOM/BLOOMZ** | <=176B | yes | **21** Niger-Congo (<=0.02% of data each) | - | 2,048 | HF card |
| **Toucan / AfroLingu-MT** | 1.2B / 3.7B | yes | 43 African (secondary source; unverified) | 156 pairs | not extracted | arXiv 2407.04796 |
| **Sunflower-14B** | 14B | yes (unverified) | 31 Ugandan | SALT-31 | not extracted | arXiv 2510.07203 |
| **Lelapa Vulavula** | closed API | no | 11-13 (docs disagree) | none | 512 tok / 256 words | docs.lelapa.ai |
| SERENGETI / AfroXLM-R / AfroLM / AfriBERTa | encoders | yes | 517 / 17-76-114 / 23 / 11 | NLU only, **not translators** | - | arXiv 2212.10785 etc. |
| SMOL + GATITOS | data | yes | 115 languages (66 Africa-tagged rows) | - | - | arXiv 2502.12301 |
| Omnilingual ASR / MMS | speech | yes | 1,600+ (ASR) | **not MT** | - | arXiv 2511.09690 |

Not found after searching: a WMT25 African translation track (WMT25 had a metrics task on the SSA-MTE African challenge set and a Creole task, no African-language MT track), "UbuntuTranslate" (no hit), Uhura/AfriMed (benchmarks, not translators, not opened), a Gemma 3n African-language list.

## Verdict

1. **"Support ~77 African languages" is accurate** for us (exactly 77 target languages; 44 of them verified against a public FLORES test set, 64 against some public en->xx data).
2. **"Other than NLLB" is misleading, because NLLB supports only 55.** We exceed NLLB's African coverage by 22, and 33 of our 77 are not in FLORES-200 at all. But NLLB is far from the only system near or above 77:
   * **MADLAD-400** (Sept 2023, open MT checkpoints) trains on about 85 African entries (my tally; check before quoting);
   * **TranslateGemma** (Jan 2026) lists about 60 African languages in its SFT data (46 of ours), evaluated on 2;
   * **Omnilingual MT** (Mar 2026) claims 1,600+ source languages (African count not extracted; weights unverified);
   * Google Translate, Apertus (1,811 pretraining languages) and Gemini claim breadth but publish no African per-language results.
3. Therefore **"one of the first to support close to 77 African languages other than NLLB" is NOT supported by the literature** and should not be used. Evidence against "first": MADLAD-400 (2023) and OMT (2026); against "close to": we have exactly 77, so say 77.
4. What *is* true on current evidence (as of the sources above): among open models **below 1B parameters that we could find**, the largest African target-language set belongs to NLLB-200-600M/distilled-600M (55, of which 44 are ours); AfriNLLB-548M (10), InkubaLM-0.4B (5), mBART-50-0.6B (3), Qwen3.5-0.8B (no list) are smaller or unspecified. No sub-1B system we found combines 77 African targets with RL fine-tuning and 2,048-token documents: TranslateGemma and OMT use RL but are >=1B/4B and cover far fewer evaluated African languages; TranslatePsy covers 19 languages and uses no RL.

## Defensible wordings (each states its condition)

* **W1 (breadth at small scale, safest):** "To our knowledge, the open translation model with the widest African target-language coverage below 1B parameters: 77 languages, versus 55 for NLLB-200-600M." Conditions: wording must say "below 1B" and "open"; survey above is the evidence; cannot exclude unlisted community fine-tunes or Gemma-3-270M/1B base-model claims ("140 languages", no African list, 270M card not opened).
* **W2 (combination claim):** "To our knowledge, the first open sub-1B (270M) model fine-tuned for 77 African target languages with reinforcement learning against an African-language quality estimator on documents up to 2,048 tokens." Conditions: every clause must be true of our released model (RL stage, QE reward, 2,048-token documents, open release are owner facts, not checked here); the closest published systems (TranslateGemma: RL, 2K context, but 4-27B and 2 evaluated African languages; TranslatePsy: <=2,048 tokens, 19 languages, no RL; AfriNLLB: 10 languages, no RL) each miss at least one clause.
* **W3 (beyond-NLLB coverage, factual):** "Covers 33 African languages that are absent from FLORES-200 and thus from NLLB-200's evaluated set, 17 of which (urh bin dag oke shi bbj kqn lun loz vmw nyn her yao sid ish dua wes) we could not find in NLLB-200, MADLAD-400 or the TranslateGemma training-language tables." Conditions: exact-language matches only (shi/Tachelhit, nyn/Runyankore and Kiga, vmw/Makhuwa varieties are related to listed Tamazight, Kiga, Makhuwa-Meetto entries; Google Translate, Gemini, OMT and Apertus were not checked language by language).
* Do **not** use: "first", "only", "one of the first ... other than NLLB", or any wording implying that MADLAD-400/OMT/TranslateGemma are narrower than ours.

## Uncertainties and things not verified

* MADLAD-400 African count (about 85) and TranslateGemma African count (about 60) are my manual tallies from the papers' language tables; the TranslateGemma tables were machine-extracted from the PDF text and lack Zulu even though Zulu is evaluated, so the extraction may have dropped entries.
* Omnilingual MT: open-weights status, African counts and any per-language en->xx scores for our 77 were not extracted (the paper is 6,000+ text lines; only keyword searches were run).
* Google Translate/Gemini total African counts are not on a primary page we could open; the 2024 post says "about a quarter" of 110 without a number.
* SeamlessM4T, Toucan, Sunflower, Lelapa, Aya Expanse lists come from model cards, search snippets or secondary reviews and are flagged `partial` in the csv.
* "Evaluated en->X" for papers that publish only pooled or macro numbers (AfroBench, TranslatePsy, MADLAD aggregates) is recorded as pooled; per-language numbers may exist in figures I did not read.
* Today's date is 2026-10-22; claims about 2026 papers (arXiv 26xx) were checked directly on arXiv pages or HTML, not from memory.
