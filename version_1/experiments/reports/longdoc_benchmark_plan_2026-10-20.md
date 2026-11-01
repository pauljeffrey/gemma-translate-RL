# A public long-context translation benchmark, English -> 77 African languages: plan, tooling and what was measured (2026-10-20)

> **Superseded in part (2026-10-20 evening):** six length bins, relaxed licences, hard split, the private Hub repo and the runner are in `longdoc_benchmark_plan_2026-10-20_addendum2_six_bins.md`.

> **Superseded in part (2026-10-20, later):** the owner decided 500 documents (no core/extended split), no medicine / law in the core, trusted-source rebalancing, a release schema with the total RL reward, and two translators (NLLB-200 + MADLAD-400). See `longdoc_benchmark_plan_2026-10-20_addendum_500docs.md`; its numbers win where the two disagree.

Owner request 2026-10-20: new evaluation documents (about 100-1000 per language, chosen for a public release) that test LONG-CONTEXT translation; the best open translator translates each source SENTENCE by sentence, the concatenation is the pseudo ground truth of the whole document, and our model's full-document translation is compared with it (BLEU, chrF++, AfriCOMET). Everything in this report was prepared on the CPU side while the GRPO run (phase 2c) kept training; **nothing was run on the GPU, nothing was uploaded, no tokens were stored**. Numbers are marked **measured** (we ran it, file named), **published** (somebody else's number, not ours) or **estimate** (arithmetic from stated assumptions).

Companion parts (same directory `reports/longdoc_benchmark_plan_2026-10-20_parts/`): `sources.md` (source and licence table, rejected sources), `translators.md` (29 systems, the 77-row coverage matrix, licences, published scores), `contamination.md` (scanner, validation, per-source result), `system_cards.json` (licence/language-list evidence), `data/*.json` (the measured numbers below). Code: `gemma_mt/gemma_mt/eval/longdoc*.py`, `gemma_mt/scripts/longdoc_*.py`, tests `gemma_mt/tests/test_longdoc*.py`. Rental: `/workspace/sft_build/longdoc_bench/` (pool, contamination flags, source set, bake-off dry runs). Decision record: `gemma_mt/DECISIONS.md` D-44.

## 0. Summary

* **What the benchmark measures.** An English document of about 512 / 1024 / 2048 / 4096 Gemma tokens goes in, the model writes the whole translation in one go. Per language we report document-level chrF++ and BLEU (every document is one segment), chunked AfriCOMET (reference-based and QE), and the failure metrics that BLEU hides: **sentence coverage / omission**, **repetition-loop rate**, **finished rate**, **length ratio**. The references come in three tracks: (P) pseudo references = sentence-by-sentence output of the best open translator per language, with a confidence tier; (H) human references: SmolDoc documents, FLORES+ passages, and SmolDoc documents **stitched** into 512 / 1024 / 2048-token documents (a real long-context test with a perfect reference); the synthetic part only fills the gaps.
* **English source set (measured).** The pool holds 21,646 licence-checked documents from 17 files (section 2); after the cleaning guards, the contamination filter (1517 contaminated and 1755 weak documents dropped), duplicate removal and the main-set licence rule, **9,462 candidate documents** remain, of which 9,216 / 3,949 / 1,899 / 882 can serve the 512 / 1024 / 2048 / 4096 bin. The selection keeps documents whose licence is public domain or permissive (CC-BY, OGL, US federal works), whose text passes cleaning guards and the contamination filter, one excerpt per document, stratified by length bin and domain. **Core: 200 documents, extended: 1000** (core is a stratified subset of extended), 0.28 M / 1.42 M English tokens. Share-alike sources (Wikipedia, Wikivoyage, FLORES+) are kept out of the main set; whether to add them is decision 2.
* **Human references (measured).** SmolDoc covers **45** of the 77 languages (12,499 documents, 5,182 whose English was not in the SFT windows), FLORES+ passages 44, together **54 languages**; stitched documents: median 95 / 52 / 26 per language at 512 / 1024 / 2048 tokens. **23 languages have no human document at all.**
* **Translators (registry measured from the cards, quality published only).** 29 systems checked. **68 of the 77 languages are named by at least one open-weight system, 67 by one with a permissive licence** (Apache-2.0 / CC-BY-4.0), 56 by a permissive system that is not just a Bible-trained OPUS-MT model; **no open system names urh, bin, oke, bbj, yao, sid, ish, dua (and wes only through a proxy).** NLLB-200 (44 languages) is CC-BY-NC; MADLAD-400 (44, Apache-2.0) is far behind it on many of our languages in its own paper; Sunflower-Qwen (31, Apache-2.0) and AfriqueQwen3.5 (41, CC-BY-4.0) are new, strong on paper, and untested by us. Which one is best per language is decided by the **bake-off** on FLORES+ dev, whose script is written and was dry-run on CPU.
* **Sizes (measured spread + estimate).** From our SFT model's real output: the 95% CI half-width of corpus chrF++ with 200 documents of 430-1700 tokens is 0.3-0.5 points (failure-free); with 20-40% failed documents it is 0.9-1.1. **Recommendation: core 200 per language (all languages with a validated reference), extended 1000 for tier A/B languages.**
* **GPU cost (estimate, assumptions stated in section 5).** Bake-off 0.5-1 GPU-h of decoding plus about 150 GB of weight downloads; **core set end to end (pseudo references from four systems, three checkpoints of our model, AfriCOMET) about 4.4-10.6 GPU-h, extended about 22-53 GPU-h** (translators 2.5-4.7 / 12.7-23.3; our model 0.9-3.6 / 4.5-18; AfriCOMET 1.0-2.4 / 4.9-12). Everything fits in one or two working days on the 96 GB card once the trainer is done; the real work is the downloads and the human review.
* **Harness (measured).** `longdoc.py` + 5 helper modules, **44 tests of mine (and 191 in the whole longdoc family) pass on the rental CPU with real pytest**. The failure metrics were validated on 400 real human documents with known damage (section 7): intact documents are flagged 1.5% of the time, truncation and loops are found, but **one dropped sentence is flagged in only 55% of the documents and the omitted share is underestimated by about 40%**: the omission metric is a lower bound.
* **What I did not do.** No bake-off result exists (no GPU), so **no tier is assigned yet and the tier thresholds are provisional**; no pseudo reference has been produced; no native-speaker check; the CTranslate2 and vLLM backends are code-complete but only exercised with fake libraries (neither is installed / usable here).

## 1. Design

### 1.1 Three tracks

| track | documents | reference | languages | role |
|---|---|---|---|---|
| **P** pseudo | the English source set (core 200 / extended 1000), 512-4096 tokens | the language's best open translator(s), every English sentence translated alone, joined | tiers A and B ranked; C reported, never ranked; N none | the long-document test at all lengths and domains |
| **H1** SmolDoc | 130-668 documents per language (22-443 clean), median 230 English tokens | professional translation, sentence aligned by the publisher | 45 | authoritative short-document quality; "clean" subset (English not in the SFT windows) is the headline |
| **H2** stitched human | 54-286 / 26-143 / 14-75 documents of 512 / 1024 / 2048 tokens (median 95 / 52 / 26) | the concatenated SmolDoc references | 45 | long-context behaviour (omission, loops, drift) with a perfect reference; not one coherent text |
| **H3** FLORES+ passages | 281 articles of 2-6 sentences (about 110 tokens) | professional translation (CC-BY-SA-4.0) | 44 | short anchor; FLORES+ articles are too short to make long passages, so FLORES+ only feeds H2-style stitching and the bake-off |

The owner's idea of concatenating FLORES+ sentences "of the same article" gives only 2-6 sentences (measured: 281 articles in devtest, mean 3.6 sentences, max 6), i.e. about 110 tokens. SmolDoc stitching is the long human track.

### 1.2 Protocol (what runs when)

1. `longdoc_fetch_sources.py` -> pool (licence-clean English documents) -> `longdoc_contam_scan.py` (training / RL / benchmark / duplicate overlap) -> `longdoc_build_source_set.py` (cleaning guards, excerpts, quotas) -> `docs.jsonl` of core and extended. **Done on the rental (section 3).**
2. `longdoc_refs.py human` -> SmolDoc, FLORES+ and stitched human documents with `contaminated` flags. **Done (section 4).**
3. `longdoc_bakeoff.py run` for each candidate system on FLORES+ dev (997 human sentences per language; SmolSent / GlotStory / IbomMT / FLORES-MOZ for the 19 languages FLORES+ does not have; 14 languages have no human sentence set) -> `compare` -> `tiers.json` (best system per language, tier, agreement of the runner-up). **GPU, not run.**
4. `longdoc_translate.py --tiers tiers.json` -> `refs_pseudo_<system>_<iso>.jsonl` (sentence-aligned, with per-sentence flags: empty / copy / loop / absurd length; a document with more than 5% flagged sentences is `valid=false` and dropped). Two systems for tier A/B languages -> agreement. **GPU, not run.**
5. The system under test translates every English document in one call (`Translate the text to {language}:  {english}`); `longdoc.py score` computes everything. **GPU for generation and AfriCOMET, not run.**
6. Release (decisions 7-9): English documents, references with their `ref_system`, tier, per-document licence and attribution, scores of the baseline models, the evaluation code.

### 1.3 Metric definitions (all in `gemma_mt/eval/longdoc.py`)

* **d-chrF++ / d-BLEU**: corpus chrF++ (char 6, word 2) and BLEU (13a) with each document one segment; hypotheses and references NFC and whitespace-collapsed. Bootstrap over DOCUMENTS (1000 resamples, optional stratification by length bin); the per-document sufficient statistics are summed, so a resample costs microseconds. `paired_bootstrap` compares two systems on the same resamples.
* **Chunked AfriCOMET**: the hypothesis is split into sentences (`split_sentences`, the splitter of the RL reward; the repository's linear-time copy `longdoc.split_sentences` is tested identical on 1,500 fuzz texts), aligned to the source sentences with the RL reward's Gale-Church beads using a language length model fitted on the references (D-30 settings: omission / extra bead cost 4, shortfall check 0.7); each bead (source, hypothesis, reference) is scored by `masakhane/africomet-stl-1.1` (reference based; both models are in the rental cache) and the document score is the source-length weighted mean in which omitted, copied and missing parts count 0.0 (`africomet_ref`; `africomet_ref_covered` leaves them out). `africomet_qe` is the reference-free variant on the same beads. The encoder truncates at 510 tokens per side; beads above that are counted (`africomet_truncated_chunks`). The chunk rules are the ones of `rl/chunk_plan.plan_chunks`; `plan_document` works on explicit sentence lists so the reference of every bead is known, and a test asserts it makes the same decisions as `plan_chunks` on 40 random documents.
* **Coverage / omission**: `coverage_chars` = 1 - (omitted + copied + missing-part source characters) / source characters; also omitted sentences, copied (untranslated) sentences, shortfall beads, `extra_share` (appended or unmatched output).
* **Repetition loop**: a unit of 1-30 words repeated at least 4 times in a row and at least 8 words long, counted only when the reference does not contain one (`loop_rate`).
* **Finished rate** (generator's own stop inside the budget), **length ratio** (hypothesis / reference characters: mean, median, p05, p95, share outside [0.6, 1.6]), `ends_cleanly_rate`.
* **Tier** (per language, from the bake-off of its best system against human references): A = chrF++ >= 45 and AfriCOMET-ref >= 0.65 (or no AfriCOMET), B = >= 30 and >= 0.50, C below (reported, never ranked), N = no open system. **Provisional** thresholds (`DEFAULT_TIER_THRESHOLDS`); the only calibration points we have are 6 tiny measurements: opus-mt tc-bible on 30 FLORES dev sentences gave (chrF++, AfriCOMET-ref) = (36.8, 0.493) zul, (36.3, 0.529) sna, (33.6, 0.502) kin, (19.2, 0.383) yor; opus-mt-en-ig gave (30.2, 0.526) on 40 sentences and (43.3, 0.607) on 10 stitched documents, which puts AfriCOMET 0.65 near chrF++ 48 and 0.50 near chrF++ 30. They must be re-set from the bake-off distribution (decision 5).

## 2. Sources and licences

Full tables (licence statement URL, attribution rule, access method, caveats per source; rejected sources with the reason; commands that reproduce the pool) are in `sources.md`; the fetchers are `scripts/longdoc_fetch_sources.py` (38 offline tests). Every source was test-fetched on the rental with a polite client (identifying User-Agent, robots.txt and Crawl-delay honoured, at most 1 request/s per host, no authentication, nothing uploaded), cleaned to prose paragraphs, and pooled with its precise licence. **Pool: 17 files, 21,646 documents (>= 400 Gemma tokens), 162 MB**: 16,593 public domain or permissive (this includes the 1,889-document OpenStax snapshot discussed below) and 5,053 share-alike (separate files).

| source (pool file) | domain | licence (as read on the licensor's page, 2026-10-20) | class | documents | median tokens | main caveat |
|---|---|---|---|---|---|---|
| Wikinews (`wikinews__permissive`, `__pd`) | news | CC-BY-2.5 (2005-09-25..2024-12-15), CC-BY-4.0 (from 2024-12-16), public domain before 2005-09-25; chosen per article | permissive / pd | 5,988 / 573 | 546 | fair-use quotes inside articles; short pieces; FLORES+ source articles are in here (flagged) |
| Global Voices (`globalvoices__permissive`) | news | CC BY 3.0 "unless otherwise stated" | permissive | 1,299 | 970 | author names missing in the pool (must be filled for the credit file); other creators' media removed |
| VOA (`voa__pd`) | news | text produced exclusively by VOA is public domain | pd | 274 | 670 | wire-service content is not PD: every article mentioning one was dropped |
| GOV.UK (`govuk__permissive`) | travel, news, law_government, health, education, science_explainer | Open Government Licence v3.0 | permissive | 1,223 | 1,132 | "except where otherwise stated"; UK-centred; the travel domain is 210 FCDO country pages |
| Federal Register (`federalregister__pd`) | law_government | US federal government work | pd | 898 | 4,363 | bureaucratic register; many near-duplicate notices |
| NASA (`nasa__pd`) | science_explainer | US government work | pd | 1,598 | 783 | partner-agency credits (ESA, JAXA, universities) may be copyrighted; heavy internal duplication |
| NIDDK / NIMH / NIDA (`fedhealth__pd`) | health, science_explainer | US federal text | pd | 473 | 769 | co-sponsored material may be third-party |
| National Park Service (`nps__pd`) | encyclopedia | "generally considered in the public domain" | pd | 240 | 1,051 | US-centred; third-party photos / quotes |
| Project Gutenberg (`gutenberg__pd`) | stories, travel, science_explainer, education, law_government, health, encyclopedia | public domain in the US, author died <= 1954 | pd | 800 | 3,136 | archaic 18th-20th century English; trademark text stripped |
| Encyclopaedia Britannica 1911 (`eb1911__pd`) | encyclopedia | public domain (US) | pd | 421 | 2,312 | Edwardian style, outdated facts |
| OpenStax current books (`openstax_ccby_current`) | education, science_explainer, law_government | CC BY 4.0 (LICENSE of the repository head) | permissive | 258 | 1,094 | images excluded |
| OpenStax **snapshot** (`openstax_ccby_snapshot__permissive`) | education, science, health, law | CC BY 4.0 at the pinned commit; **relicensed by OpenStax to CC BY-NC-SA 4.0 in 2025-26** | permissive, owner decision | 1,889 | 1,936 | irrevocable for text already distributed, but the licensor's current offer is NC: **excluded from the main set by default** |
| Our World in Data (`owid__permissive`) | science_explainer, health | CC BY 4.0 (text; third-party data keeps its own licence) | permissive | 484 | 1,660 | charts / data not included |
| PubMed Central OA (`pmc__permissive`) | health | CC BY (version per article) | permissive | 175 | 1,718 | intro / discussion prose only; stopped early (175 of 1,100 planned) |
| Wikipedia, 2% sample (`wikipedia__share_alike`) | encyclopedia | CC BY-SA 4.0 / 3.0 + GFDL | **share-alike** | 2,568 | 1,070 | heavily contaminated (it is in the SFT/RL English) |
| Wikivoyage, 30% sample (`wikivoyage__share_alike`) | travel | CC BY-SA 4.0 | **share-alike** | 2,485 | 1,156 | listing entries removed |

**Rejected** (reasons in `sources.md`): OpenStax live books and WHO (CC BY-NC-SA), The Conversation (CC BY-ND), World Bank (non-commercial terms), UN and FAO prose (no CC), African Storybook (CC-BY but stories of 100-400 words, and probably in the training data), StoryWeaver (API 404), MedlinePlus (mixed with copyrighted A.D.A.M. text), CIA Factbook (600 tokens of prose per country), FineWeb and pile-of-law (mixed copyright / NC-SA), and a list of sites that blocked the rental's IP (congress.gov and CRS, state.gov, loc.gov, eur-lex, noaa.gov, usda.gov, peacecorps.gov). **UK legislation (OGL) and EUR-Lex are not in the pool** (the first legislation run had a cleaner bug, the fixed fetcher needs about 2 h because of a 5 s Crawl-delay; EUR-Lex is behind a WAF), so `law_government` is GOV.UK + Federal Register + Gutenberg + OpenStax.

**Pool capacity** (documents that can serve a bin, because long documents are cut at paragraph boundaries; pd + permissive, measured from the pool files, **including** the OpenStax snapshot; without it education has 47 documents for the 4096 bin, health 101, science_explainer 177):

| domain | usable for 512 (>= 400 tokens) | for 1024 (>= 800) | for 2048 (>= 1600) | for 4096 (>= 3200) |
|---|---|---|---|---|
| education | 1,422 | 1,184 | 827 | 299 |
| encyclopedia | 687 | 588 | 385 | 188 |
| health | 1,084 | 829 | 515 | 195 |
| law_government | 1,500 | 1,365 | 1,141 | 740 |
| news | 8,426 | 2,280 | 353 | **53** |
| science_explainer | 2,847 | 1,652 | 627 | 211 |
| stories | 289 | 281 | 250 | 152 |
| travel | 338 | 323 | 294 | 137 |
| share-alike only: encyclopedia (Wikipedia) | 2,568 | 1,864 | 676 | 200 |
| share-alike only: travel (Wikivoyage) | 2,485 | 1,693 | 856 | 314 |

Honest weak spots: news is almost entirely short pieces (53 documents above 3,200 tokens); stories is Gutenberg only (archaic English, no modern CC-BY fiction >= 400 tokens was found); encyclopedia without share-alike rests on EB1911 and NPS; health on 175 PMC articles, 473 NIH pages and OpenStax clinical books. Licence risks of the whole pool (third-party material inside permissive text, per-document licence variation, US-centred public-domain claims, trademarks, personal data) and the recommended `ATTRIBUTION` format (credit string per licensor + "changes: extracted running text, cut at paragraph boundary, translated by X", which CC BY requires) are in `sources.md` (d).


## 3. Strata, selection and the English source set

Length bins (Gemma tokens of the English source; one excerpt per document = the longest prefix of whole paragraphs inside the bin's range, closest to the nominal length): 512 = 410-640, 1024 = 820-1280, 2048 = 1640-2560, 4096 = 3280-5120. Domains: education, encyclopedia, health, law_government, news, science_explainer, stories, travel. Quotas: equal over domains, 35 / 30 / 22 / 13 % over the bins (so most documents are short, the 4096 bin still has 24 core documents); round-robin over sources inside a cell (diversity first), at most 35% of a cell from one source, shortfalls filled from the other domains of the same bin, core = first documents of every cell's ordering (nested). Cleaning guards (`longdoc_select.paragraph_ok`, `text_quality`): template remnants like `the 's`, markup, learning-objective / key-term blocks, abbreviation-heavy tables, unterminated fragments, shouting, non-English or non-Latin text, URLs, numeric tables, too-long sentences. Deterministic by seed.

Funnel on the final pool (`SELECTION_REPORT.json`, measured; the drop reasons are exclusive and applied in the order contamination, cleaning, length, quality):

| step | count |
|---|---|
| documents in the pool files used (public domain + permissive, without the OpenStax snapshot) | 14,704 |
| dropped: contaminated (training data / benchmarks / duplicate of another pool document) | 1,517 |
| dropped: weak contamination flag (2-10% of the windows seen) | 1,755 |
| dropped: no paragraph left after the cleaning guards | 8 |
| dropped: more than 20% of the paragraphs removed by the guards | 1,236 |
| dropped: under 400 tokens after cleaning | 282 |
| dropped: no whole-paragraph prefix lands inside any bin range | 330 |
| dropped: more than 3% of the sentences over 100 words | 69 |
| dropped: mean sentence length outside 6-45 words | 30 |
| dropped: more than 25% fragments (< 3 words) | 11 |
| dropped: less than 70% letters | 2 |
| dropped: non-Latin text | 2 |
| **candidate documents** | 9,462 |
|   of which can serve the 512 bin | 9,216 |
|   ... 1024 | 3,949 |
|   ... 2048 | 1,899 |
|   ... 4096 | 882 |
| (paragraphs removed inside the surviving or dropped documents by the guards) | 24,325 |

Candidates per cell (documents that can fill the bin):

| domain | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|
| education | 253 | 199 | 121 | 39 |
| encyclopedia | 409 | 457 | 321 | 149 |
| health | 338 | 271 | 177 | 71 |
| law_government | 508 | 481 | 386 | 277 |
| news | 6002 | 1364 | 229 | 39 |
| science_explainer | 1345 | 818 | 343 | 130 |
| stories | 258 | 256 | 227 | 125 |
| travel | 103 | 103 | 95 | 52 |

**Core (200 documents)**:

| domain | 512 | 1024 | 2048 | 4096 | total |
|---|---|---|---|---|---|
| education | 9 | 8 | 6 | 3 | 26 |
| encyclopedia | 9 | 8 | 6 | 3 | 26 |
| health | 9 | 8 | 6 | 3 | 26 |
| law_government | 9 | 8 | 6 | 3 | 26 |
| news | 9 | 7 | 5 | 3 | 24 |
| science_explainer | 9 | 7 | 5 | 3 | 24 |
| stories | 9 | 7 | 5 | 3 | 24 |
| travel | 9 | 7 | 5 | 3 | 24 |
| **total** | 72 | 60 | 44 | 24 | 200 |

**Extended (1000 documents)**:

| domain | 512 | 1024 | 2048 | 4096 | total |
|---|---|---|---|---|---|
| education | 49 | 43 | 29 | 16 | 137 |
| encyclopedia | 39 | 37 | 31 | 21 | 128 |
| health | 51 | 43 | 31 | 20 | 145 |
| law_government | 49 | 48 | 34 | 21 | 152 |
| news | 66 | 50 | 33 | 9 | 158 |
| science_explainer | 65 | 53 | 35 | 23 | 176 |
| stories | 18 | 14 | 10 | 11 | 53 |
| travel | 15 | 12 | 17 | 7 | 51 |
| **total** | 352 | 300 | 220 | 128 | 1000 |

Shortfall against the quotas: none. Composition of the core set (measured, `SELECTION_REPORT.json`): 119 public-domain and 81 permissive documents; by source Project Gutenberg 78 (39%: the stories, travel and a third of the long documents of every other domain are Gutenberg, i.e. archaic English), GOV.UK 33, OpenStax (current books) 18, NIH institutes 17, OWID 12, EB1911 8, Wikinews 8, Global Voices 6, Federal Register 6, PMC 5, VOA 5, NASA 4. The cleaning guards removed 24,325 paragraphs and dropped 1,236 documents with more than 20% bad paragraphs; contamination dropped 1,517 contaminated and 1,755 weak documents. The thin cells are the 4096 bin of news (39 candidates, 9 in the extended set), education (39) and travel (52), and the whole stories / travel domains (53 / 51 documents in the extended set because the per-source cap and the small pool bind). Nothing here was read by a human beyond the examples below: **a review pass over the 200 core documents (cleaning leftovers such as `(ppt, 541kb)` link lists in GOV.UK pages, epigraph fragments at the start of Gutenberg chapters) is advised before the references are produced.**

How the choice of licence classes changes the pool (candidate documents per domain for the 1024 and 4096 bins, after all filters):

| domain | main set: 1024 / 4096 | + OpenStax snapshot: 1024 / 4096 | + share-alike (Wikipedia, Wikivoyage): 1024 / 4096 |
|---|---|---|---|
| education | 199 / 39 | 972 / 231 | 199 / 39 |
| encyclopedia | 457 / 149 | 457 / 149 | 1512 / 269 |
| health | 271 / 71 | 512 / 157 | 271 / 71 |
| law_government | 481 / 277 | 627 / 301 | 481 / 277 |
| news | 1364 / 39 | 1364 / 39 | 1364 / 39 |
| science_explainer | 818 / 130 | 967 / 151 | 818 / 130 |
| stories | 256 / 125 | 256 / 125 | 256 / 125 |
| travel | 103 / 52 | 103 / 52 | 1351 / 281 |
| **all candidates** | 9,462 | 10,935 | 12,830 |

The OpenStax snapshot would fix education, health and law at the 4096 bin (231 / 157 / 301 instead of 39 / 71 / 277); share-alike would fix encyclopedia and travel (269 / 281 instead of 149 / 52). Neither fixes news at 4096 (39 documents).

Files on the rental: `/workspace/sft_build/longdoc_bench/source_set/{core,extended}/docs.jsonl` (+ `ATTRIBUTION.jsonl`), `SELECTION_REPORT.json`, examples file; sizes core docs.jsonl 2.9 MB, extended docs.jsonl 15 MB, human-reference documents 131 MB (`source_set/human`), pool 162 MB, contamination flags 14 MB.

### 3.1 Three examples per domain (core set)

#### education

* `gutenberg:pg67987-003` | bin 512 (520 Gemma tokens, 15 sentences, excerpt) | Public domain (US; author died 1933) | Sneyd-Kynnersley, E. M. (Edmund McKenzie), 1841-1933; via Project Gutenberg (ebook #67987) | https://www.gutenberg.org/ebooks/67987
  > “O good old man; how well in thee appears The constant service of the antique world.” On a dreary day in October I had to visit one of the remote villages high up in the moorland district. The wind howled, and the rain beat down pitilessly, as my Welsh-car jolted me, and swayed me to Llangastanau Rectory, where I had been invited to spend the night. I was by no ...

* `govuk:edu-speech|government~speeches~childcare-minister-speaks-about-more-pre-schools-in-schools` | bin 2048 (2063 Gemma tokens, 94 sentences, excerpt) | OGL-3.0 | Crown copyright, Department for Education, GOV.UK, licensed under the Open Government Licence v3.0 | https://www.gov.uk/government/speeches/childcare-minister-speaks-about-more-pre-schools-in-schools
  > It is a pleasure to be here at Policy Exchange to speak to you as the first male childcare minister, something I only became aware of when an official drafted some copy in my name, and in it was a line that read, ‘I may be a man but I understand childcare’. I smiled when I read the line because I genuinely believe that childcare is an issue for ‘parents’ ...

* `openstax:life-liberty-and-pursuit-happiness:m76334` | bin 1024 (946 Gemma tokens, 30 sentences, excerpt) | CC-BY-4.0 | OpenStax, Life, Liberty, and the Pursuit of Happiness, Rice University, CC BY 4.0 (https://openstax.org) | https://github.com/openstax/osbooks-life-liberty-and-pursuit-happiness/blob/acb6fbe2744dd112187b6aff5946cfe969410fca/modules/m76334/index.cnxml
  > The French and Indian War was the climactic struggle between Great Britain and France for imperial control of North America. The war began in 1754, when a young Virginia militia officer named George Washington engaged in a skirmish with a party of French soldiers, and it ended six years later when the governor-general of New France surrendered to a British army at Montreal. The conflict was part of a much ...

#### encyclopedia

* `nps-articles:articles~000~cave-mapping-for-conservation.htm` | bin 512 (412 Gemma tokens, 18 sentences) | Public domain (US federal government work) | National Park Service (NPS), U.S. Department of the Interior, https://www.nps.gov/articles/000/cave-mapping-for-conservation.htm | https://www.nps.gov/articles/000/cave-mapping-for-conservation.htm
  > Buffalo National River has more than 500 known caves, and new caves are still being discovered. These underground passageways are formed by the carving and dissolving of limestone by moving water. This continuous flow of water is what has created the karst landscape we see here today. The National Park Service works closely with partnering organizations like the Cave Research Foundation (CRF) to map caves, conduct biological surveys, and monitor ...

* `gutenberg:pg60985-003` | bin 2048 (1979 Gemma tokens, 63 sentences, excerpt) | Public domain (US; author died 1939) | Wright, Willard Huntington, 1888-1939; via Project Gutenberg (ebook #60985) | https://www.gutenberg.org/ebooks/60985
  > In the field of poetry the Encyclopædia Britannica comes nearer being a competent reference library than in the field of painting, fiction, or drama. This fact, however, is not due to a spirit of fairness on the part of the Encyclopædia’s editors so much as to the actual superiority of English poetry. In this field England has led the world. It is the one branch of culture in which modern ...

* `eb1911:pg33614-boole-george` | bin 1024 (966 Gemma tokens, 26 sentences, excerpt) | Public domain (published 1910-1911) | Encyclopaedia Britannica, 11th ed. (1910-11), article Boole, George; via Project Gutenberg ebook #33614 | https://www.gutenberg.org/ebooks/33614
  > BOOLE, GEORGE (1815-1864), English logician and mathematician, was born in Lincoln on the 2nd of November 1815. His father was a tradesman of limited means, but of studious character and active mind. Being especially interested in mathematical science, the father gave his son his first lessons; but the extraordinary mathematical powers of George Boole did not manifest themselves in early life. At first his favourite subject was classics. Not until ...

#### health

* `pmc:PMC12616657` | bin 1024 (1103 Gemma tokens, 35 sentences, excerpt) | CC-BY (version not stated in the article) | Saeedzai SA, Knight J, Filipe L, Moore S, Loevinsohn B. (2025). Socio-economic determinants for reduced uptake of routine childhood vaccination in Afghanistan. . doi:10.1080/16549716.2025.2582262. CC-BY (version not stated in the article) | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12616657/
  > Coverage of many routine childhood vaccinations in Afghanistan is low and declining. For instance, only 60% of eligible children received doses of the pentavalent vaccine in 2013–18, and this fell to 51% in 2022/23. To help explain the underlying causes behind this trend, this study aims to identify socio-demographic factors relating to caregivers whose children are either entirely unvaccinated or only partially vaccinated. A secondary analysis of Multi Indictor Cluster ...

* `govuk:health-news|government~news~ukhsa-expands-support-to-help-countries-identify-emerging-variants` | bin 512 (502 Gemma tokens, 15 sentences, excerpt) | OGL-3.0 | Crown copyright, UK Health Security Agency, GOV.UK, licensed under the Open Government Licence v3.0 | https://www.gov.uk/government/news/ukhsa-expands-support-to-help-countries-identify-emerging-variants
  > One year on from its launch, the New Variant Assessment Platform (NVAP) will continue expanding its work to other countries and regions worldwide to help global health security efforts. Nine countries and territories have now received direct support from NVAP to improve early detection of variants, putting the world in a stronger position to respond to newly emerging variants of SARS-CoV-2. Set up a year ago and led by the ...

* `nida:publications~drugfacts~prescription-opioids` | bin 2048 (1959 Gemma tokens, 82 sentences) | Public domain (US federal government work) | National Institute on Drug Abuse (NIDA), National Institutes of Health, https://nida.nih.gov/publications/drugfacts/prescription-opioids | https://nida.nih.gov/publications/drugfacts/prescription-opioids
  > Opioids are a class of drugs naturally found in the opium poppy plant. Some prescription opioids are made from the plant directly, and others are made by scientists in labs using the same chemical structure. Opioids are often used as medicines because they contain chemicals that relax the body and can relieve pain. Prescription opioids are used mostly to treat moderate to severe pain, though some opioids can be used ...

#### law_government

* `govuk:law-speech|government~speeches~the-legacy-of-the-icty-should-be-a-clear-and-resounding-message-to-those-who-commit-these-crimes-you-cannot-escape-justice` | bin 1024 (871 Gemma tokens, 32 sentences) | OGL-3.0 | Crown copyright, Foreign & Commonwealth Office, GOV.UK, licensed under the Open Government Licence v3.0 | https://www.gov.uk/government/speeches/the-legacy-of-the-icty-should-be-a-clear-and-resounding-message-to-those-who-commit-these-crimes-you-cannot-escape-justice
  > I would like to thank the Presidents of the Tribunal and the Mechanism and the Prosecutor for their reports and their presentations. And, at the outset, let me reiterate the UK’s continued support for the ICTY and the Mechanism, and for all that they do to end impunity for the most serious international crimes. As the ICTY progresses through its final year, the international community as a whole must continue ...

* `gutenberg:pg59961-005` | bin 512 (455 Gemma tokens, 13 sentences, excerpt) | Public domain (US; author died 1928) | Hall, John, 1865-1928; via Project Gutenberg (ebook #59961) | https://www.gutenberg.org/ebooks/59961
  > Wellington had been chiefly instrumental in effecting the restoration of the Bourbons after Waterloo. The news of their expulsion could not, under these circumstances, fail to cause him some personal regret. But, in addition, he was too well acquainted with French affairs not to be aware that the triumph of the democratic party was a grave menace to the peace of Europe. On the other hand, however, far from being, ...

* `federalregister:2012-23078` | bin 2048 (2025 Gemma tokens, 46 sentences, excerpt) | Public domain (US federal government work) | Agriculture Department; Food Safety and Inspection Service, Federal Register 2012-09-19 (2012-23078) | https://www.federalregister.gov/documents/2012/09/19/2012-23078/risk-based-sampling-of-beef-manufacturing-trimmings-for-escherichia-coli-e-coli-o157h7-and-plans-for
  > Food Safety and Inspection Service (FSIS), U.S. Department of Agriculture (USDA). FSIS is announcing its intention to redesign its E. coli O157:H7 verification testing program for beef manufacturing trimmings to make the program more risk-based and to enable the Agency to calculate on-going statistical prevalence estimates for E. coli O157:H7 in raw beef manufacturing trimmings. This notice also discusses FSIS's plans to perform a beef carcass baseline. FSIS seeks public ...

#### news

* `globalvoices:46225` | bin 512 (572 Gemma tokens, 23 sentences, excerpt) | CC-BY-3.0 | Global Voices contributor, Global Voices, https://globalvoices.org/2008/07/07/china-is-sarkozy-the-problem-with-sino-french-relations/, CC BY 3.0 | https://globalvoices.org/2008/07/07/china-is-sarkozy-the-problem-with-sino-french-relations/
  > Okay, this post comes inspired by GVO Lingua. Bush is coming to the Olympics, but French President Nicolas Sarkozy has pegged his decision on the outcome of talks between Beijing and the Dalai Lama, currently underway. Pending that, many Chinese netizens are now telling Sarkozy not to bother showing up. With his bling-bling-whatever politics, Sarkozy is going to enrage the world. Liberty, Equality, Fraternity, do you believe in all that? ...

* `wikinews:18597` | bin 1024 (1008 Gemma tokens, 40 sentences, excerpt) | Public domain (published before 2005-09-25) | Wikinews contributors, "Cypriot plane with 121 on board crashes in Greece", en.wikinews.org, Public domain (published before 2005-09-25) | https://en.wikinews.org/wiki/Cypriot_plane_with_121_on_board_crashes_in_Greece
  > A Helios airliner (Boeing 737, flight HCY 522) with 115 passengers and 6 crew onboard has crashed into a mountain at 09:04 UTC (12:04 p.m. EEST) near Grammatiko, 40 kilometres (25 miles) north of Athens, while approaching the capital's airport. All on board perished. The plane was travelling from Larnaca, in the southeast of Cyprus (Greek:Kypros, Κύπρος) and scheduled to arrive in Prague after a stop in Athens. Immediately after ...

* `govuk:news-story|government~news~the-crown-and-suppliers-a-new-way-of-working` | bin 2048 (2044 Gemma tokens, 96 sentences, excerpt) | OGL-3.0 | Crown copyright, Cabinet Office, GOV.UK, licensed under the Open Government Licence v3.0 | https://www.gov.uk/government/news/the-crown-and-suppliers-a-new-way-of-working
  > The Crown and suppliers: a new way of working’ event was held on 21 November 2011 in London. Read the full press release here. Read the text version of the speech below. Common Infrastructure: network, desktop and cloud (ppt, 541kb) Digital: ICT strategy supporting the new approach to digital (ppt, 558kb) For a long time UK governments have held UK businesses at arm’s length. Too often buying goods and services ...

#### science_explainer

* `nasa:www-264694` | bin 2048 (1733 Gemma tokens, 64 sentences) | Public domain (US government work) | NASA (https://www.nasa.gov/missions/gpm/nasa-probes-environment-covid-19-impacts-possible-links/) | https://www.nasa.gov/missions/gpm/nasa-probes-environment-covid-19-impacts-possible-links/
  > Scientists are using information from NASA’s Earth-observing satellites, on-the-ground sensors and computer-based datasets to study the environmental, economic and societal impacts of the COVID-19 pandemic. In addition, the agency’s Earth Science Division recently sponsored new projects to examine how the shutdowns in response to the pandemic are changing the environment, especially the atmosphere, and determine what, if any, natural environmental phenomena might impact the spread of the pandemic. “NASA has ...

* `owid:work-employment` | bin 4096 (4066 Gemma tokens, 168 sentences, excerpt) | CC-BY-4.0 | Our World in Data (Max Roser, Hannah Ritchie and colleagues; see the page for the authors), CC BY 4.0, https://ourworldindata.org/work-employment | https://ourworldindata.org/work-employment
  > Work is at the center of most people’s lives. For billions of adults around the world, it is the main way to earn a living, support their families, and contribute to society. What share of the population in different countries is part of the labor force? How is employment split between people who work for an employer and those who run their own business or work independently? And how many ...

* `nimh-news:news~science-updates~2014~schizophrenias-genetic-skyline-rising` | bin 1024 (1029 Gemma tokens, 31 sentences) | Public domain (US federal government work) | National Institute of Mental Health (NIMH), National Institutes of Health, https://www.nimh.nih.gov/news/science-updates/2014/schizophrenias-genetic-skyline-rising | https://www.nimh.nih.gov/news/science-updates/2014/schizophrenias-genetic-skyline-rising
  > The largest genomic dragnet of any psychiatric disorder to date has unmasked 108 chromosomal sites harboring inherited variations in the genetic code linked to schizophrenia, 83 of which had not been previously reported. By contrast, the “skyline” of such suspect variants associated with the disorder contained only 5 significant peaks in 2011. By combining data from all available schizophrenia genetic samples, researchers supported by the National Institutes of Health powered ...

#### stories

* `gutenberg:pg66881-001` | bin 512 (525 Gemma tokens, 28 sentences, excerpt) | Public domain (US; author died 1936) | James, M. R. (Montague Rhodes), 1862-1936; via Project Gutenberg (ebook #66881) | https://www.gutenberg.org/ebooks/66881
  > The first day after his arrival at the Swan Hotel at Longbridge was so stormy that he got no farther than the tobacconist’s. The next, comparatively bright, he used for his visit to Gaulsford, which interested him more than a little, but had no ulterior consequences. The third, which was really a pearl of a day for early January, was too fine to be spent indoors. He gathered from the ...

* `gutenberg:pg20486-004` | bin 4096 (4158 Gemma tokens, 209 sentences, excerpt) | Public domain (US; author died 1948) | Brown, Alice, 1857-1948; via Project Gutenberg (ebook #20486) | https://www.gutenberg.org/ebooks/20486
  > This happened in what Dilly Joyce, in deference to a form of speech, was accustomed to call her young days; though really her spirit seemed to renew itself with every step, and her body was to the last a willing instrument. She lived in a happy completeness which allowed her to carry on the joys of youth into the maturity of years. But things did happen to her from twenty ...

* `gutenberg:pg51076-003` | bin 512 (514 Gemma tokens, 30 sentences, excerpt) | Public domain (US; author died 1946) | Oppenheim, E. Phillips (Edward Phillips), 1866-1946; via Project Gutenberg (ebook #51076) | https://www.gutenberg.org/ebooks/51076
  > On the following morning, Aaron Rodd, somewhat to his surprise, received a visit from his only client. Mr. Jacob Potts, who was a publican and retired pugilist, and whose appearance entirely coincided with his dual profession, looked around the apartment with a little sniff. "Ho!" he exclaimed. "Better times arrived, eh? 'Ad a spring-cleaning, 'aven't you? Telephone, too, and new chairs! Golly! Does it run to cigars?" "Know anything about ...

#### travel

* `gutenberg:pg46034-001` | bin 1024 (1110 Gemma tokens, 29 sentences, excerpt) | Public domain (US; author died 1907) | Field, Henry M. (Henry Martyn), 1822-1907; via Project Gutenberg (ebook #46034) | https://www.gutenberg.org/ebooks/46034
  > It was a bright New-Year's morning, that first day of 1887, and how could we begin the year better than by climbing to the top of the Rock to get the outlook over land and sea? The ascent is not difficult, for though the Rock is steep as well as high, a zigzag path winds up its side, which to a good pedestrian is only a bracing walk, while a ...

* `govuk:travel|foreign-travel-advice~montserrat` | bin 2048 (1941 Gemma tokens, 110 sentences) | OGL-3.0 | Crown copyright, Foreign, Commonwealth & Development Office, GOV.UK, licensed under the Open Government Licence v3.0 | https://www.gov.uk/foreign-travel-advice/montserrat
  > If you choose to travel, research your destinations and get appropriate travel insurance. Insurance should cover your itinerary, planned activities and expenses in an emergency. The Foreign, Commonwealth & Development Office (FCDO) provides advice about risks of travel to help you make informed decisions. Find out more about FCDO travel advice. This information is for people travelling on a full ‘British citizen’ passport from the UK. It is based on ...

* `gutenberg:pg1865-008` | bin 4096 (3298 Gemma tokens, 108 sentences) | Public domain (US; author died 1882) | Trollope, Anthony, 1815-1882; via Project Gutenberg (ebook #1865) | https://www.gutenberg.org/ebooks/1865
  > To these must be added to make up the population of the United States, as it stood in 1860. Each of the three interests would consider itself wronged by the division above made, but the South would probably be the loudest in asserting its grievance. The South claims all the slave States, and would point to secession in Virginia to justify such claim,--and would point also to Maryland and Baltimore, ...


## 4. Contamination and de-duplication (measured)

Scanner (`longdoc_contam.py`, details, thresholds and evidence in `contamination.md`): one primitive, a uint64 polynomial rolling hash of every window of n consecutive units, used four times. (a) **SFT training data**: the text windows were deleted from the rental, so the English side of all 11,356,647 packed samples (train + val, 687 M tokens; layout read from the packing code and checked on every sample, 0 invalid) is scanned in **Gemma token space, n = 12** (a decoded-word scan, word 8-grams, runs next to it); (b) **RL prompts**: `rl_prompts_v2`, 77 files, 5.94 M rows = 1.41 M distinct windows, word 8-grams (the D-42 normalisation); (c) **existing benchmarks' English**: FLORES+ devtest and dev, FLORES-MOZ, NTREX, MAFAND, SmolDoc, SmolSent, GlotStory, SALT, IbomMT, `bench_dev`; (d) **the other pool documents** (containment, `dup_of`). Per document: the share of its windows found in each source and the longest run; **clean** = every share < 2%, **weak** = largest share < 10%, **contaminated** = more, or a verbatim run >= 50 words / 64 tokens, or >= 50% shared with another pool document. n = 12 was chosen on measurements: excerpts of training windows are recovered at 99.2% for every n from 8 to 20 (the only loss is one window at the start), the false-positive rate of a clean text falls steeply up to n = 12 and flattens after.

Validation (controls built from the training data itself and from text that cannot be in it; `contamination.md` section 3):

| control | n | result |
|---|---|---|
| random RL windows | 300 | 300 / 300 contaminated |
| random SFT windows decoded from the packed stream | 300 | 271 / 300; all 248 windows of >= 12 tokens found, all 29 misses are fragments shorter than any n-gram |
| benchmark passages (FLORES+, NTREX, SmolDoc) | 100 | 100 / 100 contaminated, the right benchmark named every time |
| Wikinews written after the data was collected (2025-08 .. 2026-05) | 112 | 109 clean, 3 weak (stock phrases of 8-13 words in short documents), 0 contaminated |
| text with 30% / 60% copied from RL windows (and from SFT windows) | 40 each | measured share 0.292 / 0.590 (token path 0.256 / 0.554) against the exact expectation 0.291 / 0.590 (0.255 / 0.554) |
| exact copies, 60% and 30% prefixes of 40 Wikinews articles | 120 | 120 / 120 flagged contaminated, `dup_of` = the original |
| Project Gutenberg slices | 80 | 67 clean, 13 flagged: **not a valid negative**, the RL and SFT data really contain passages of classic novels |

Result on the final pool (**measured**, `contam/SUMMARY.json`; the percentages in `contamination.md` section 4 are those of the 07:06 snapshot):

| pool file | documents | clean | weak | contaminated |
|---|---|---|---|---|
| eb1911__pd | 421 | 401 (95%) | 7 | 13 |
| federalregister__pd | 898 | 408 (45%) | 236 | 254 |
| fedhealth__pd | 473 | 212 (45%) | 187 | 74 |
| globalvoices__permissive | 1299 | 814 (63%) | 181 | 304 |
| govuk__permissive | 1223 | 900 (74%) | 197 | 126 |
| gutenberg__pd | 800 | 781 (98%) | 10 | 9 |
| nasa__pd | 1598 | 893 (56%) | 323 | 382 |
| nps__pd | 240 | 210 (88%) | 22 | 8 |
| openstax_ccby_current__permissive | 258 | 220 (85%) | 30 | 8 |
| openstax_ccby_snapshot__permissive | 1889 | 1600 (85%) | 158 | 131 |
| owid__permissive | 484 | 339 (70%) | 93 | 52 |
| pmc__permissive | 175 | 145 (83%) | 23 | 7 |
| voa__pd | 274 | 266 (97%) | 7 | 1 |
| wikinews__pd | 573 | 526 (92%) | 26 | 21 |
| wikinews__permissive | 5988 | 5317 (89%) | 413 | 258 |
| wikipedia__share_alike | 2568 | 1866 (73%) | 252 | 450 |
| wikivoyage__share_alike | 2485 | 2185 (88%) | 200 | 100 |

What the training data clearly contained: **Wikipedia** (about a quarter of the sampled articles have >= 2% of their windows in the RL windows: the RL set is built from Wikipedia 20231101 and FineWeb-Edu), **NASA pages** (31% >= 2% in RL; FineWeb-Edu has nasa.gov), and **Wikinews beyond FLORES** (the FLORES+ source articles themselves are in the pool and are flagged: 38 documents hit FLORES+/IbomMT/FLORES-MOZ, 34 the dev sets; 122 documents have >= 2% of their tokens in the SFT stream, 7 more than 20%). GOV.UK, OWID, Federal Register and Gutenberg are almost clean against the training data; GOV.UK / OWID / Federal Register / NASA lose documents to the **duplicate rule** (shared templates, repeated news families), which is the pool-internal dedup the selection needs.


### 4.1 Human-reference documents and the same filters

The D-42 flag files give, for every SmolDoc document, whether its English was in the SFT windows (exact or >= 50% of the 8-grams): **5,182 of 12,499 documents are clean** (median 105 per language; **18 languages have exactly 130 SmolDoc documents and only 22 of them are clean**); the pair (English and its translation) was seen only for the 8 languages whose SMOL translations were trained on. FLORES+ English is in the SFT windows for 95% of the sentences (43,380 of 45,540 devtest rows exact), so FLORES+ passages are reported "all" and "clean" and are not a headline. The stitched documents inherit `contaminated = any component` (`--clean-only-stitch` builds them only from clean documents; with 22 clean documents per rare language that gives only 3-5 stitched documents of 1024 tokens, so the default stitches everything and reports clean-only separately).

## 5. Sizes, standard errors and GPU cost

### 5.1 How many documents per language

Measured on real model output: sentence-level predictions of our SFT model (`eval/sft_best/predictions.jsonl`, FLORES+ devtest, 45 languages) were concatenated into pseudo-documents of k consecutive sentences, document-level chrF++ computed per document, the 95% bootstrap half-width of the corpus chrF++ estimated by resampling documents to n = 100 / 200 / 400 (`longdoc_costplan.py se`, 200 repetitions, `data/se_analysis.json`):

| pseudo-document | English tokens | mean corpus chrF++ | document chrF++ SD, median (p90) over 45 languages | 95% half-width, n=100 | n=200 | n=400 |
|---|---|---|---|---|---|---|
| 4 consecutive sentences | 107 | 24.1 | 5.52 (6.46) | 1.15 (1.47) | 0.83 (1.05) | 0.59 (0.75) |
| 8 consecutive sentences | 214 | 26.2 | 4.39 (5.21) | 0.87 (1.10) | 0.63 (0.80) | 0.44 (0.56) |
| 16 consecutive sentences | 428 | 28.3 | 3.43 (4.12) | 0.65 (0.85) | 0.46 (0.63) | 0.35 (0.41) |
| 32 consecutive sentences | 856 | 30.4 | 2.66 (3.20) | 0.51 (0.63) | 0.36 (0.43) | 0.26 (0.33) |
| 64 consecutive sentences | 1715 | 32.7 | 2.05 (2.57) | 0.39 (0.51) | 0.27 (0.36) | 0.19 (0.25) |

The spread falls roughly as length^-0.36 (fit over the five lengths; predicted document SD 3.2 / 2.5 / 1.9 / 1.5 chrF++ at 512 / 1024 / 2048 / 4096 tokens, the last two are extrapolations). **This is a lower bound**: it has no loops, omissions or unfinished generations, and the pseudo-documents are stitched articles. A document that fails (the earlier SmolDoc run finished only 53-63% of the documents of a rare language, `reports/gemma_rare_language_benchmarks_2026-10-19.md`) loses ten or more chrF++, which is a mixture and raises the SD a lot. Two-component scenario (ASSUMED failure rate and loss; failure-free spread from the table above):

| scenario (document SD) | n=26 | n=50 | n=100 | n=200 | n=400 | n=1000 |
|---|---|---|---|---|---|---|
| failure-free, 512-token documents (measured fit, SD 3.2): 3.2 | 1.23 | 0.88 | 0.63 | 0.44 | 0.31 | 0.20 |
| failure-free, 2048-token documents (extrapolated SD 1.9): 1.9 | 0.75 | 0.54 | 0.38 | 0.27 | 0.19 | 0.12 |
| 20% of the documents fail by 15 chrF++ (SD_q 2.5): 6.5 | 2.50 | 1.80 | 1.27 | 0.90 | 0.64 | 0.40 |
| 40% fail by 15 chrF++ (SD_q 2.5): 7.8 | 2.98 | 2.15 | 1.52 | 1.08 | 0.76 | 0.48 |

And the smallest true chrF++ difference between two systems that paired resampling detects with 80% power at 5% (sd_diff = SD x sqrt(2 (1 - rho)), rho = correlation of the two systems' document scores, **assumed**: we have one system's output only):

| paired comparison of two systems | n=100 | n=200 | n=400 | n=1000 |
|---|---|---|---|---|
| failure-free, SD 3.2, correlation 0.7 | 0.69 | 0.49 | 0.35 | 0.22 |
| failure-free, SD 3.2, correlation 0.9 | 0.40 | 0.28 | 0.20 | 0.13 |
| 40% failures, SD 7.8, correlation 0.7 | 1.69 | 1.20 | 0.85 | 0.53 |
| 40% failures, SD 7.8, correlation 0.9 | 0.98 | 0.69 | 0.49 | 0.31 |

Reading: with the core size of 200 documents a language's chrF++ is known to about +-0.3-0.5 (failure-free) to +-1.1 (heavy failures) points and two checkpoints 0.3-1.2 points apart are separable; with 1000 documents +-0.12-0.5 and 0.13-0.53. A per-bin number rests on 26-70 documents (core) and is good to +-1-3 points. Across the 77 languages the macro average is far more precise than any language. That is why the recommendation is **core = 200 for every language that has a validated reference, extended = 1000 for tier A/B languages** (more documents do not repair a weak reference; they only shrink the sampling error, and the pseudo-reference's own error is a bias, not noise).

### 5.2 Volumes

**Measured** on the human references of 54 languages (17,989 SmolDoc / FLORES+ documents, Gemma tokenizer): the target text has 1.88 Gemma tokens per English token (median over languages; mean 1.92, p90 2.24), from 1.12 (Pidgin) and 1.45 (Afrikaans) to 2.66 (Yoruba), 2.93 (Nuer) and 3.19 (Kabiye). A 4,096-token English document therefore needs about 7,900 tokens of answer on average and up to 13,000 for the worst language, which is the length our model has to generate. Plan input: mean 1455 English tokens per document, 27.3 per sentence.

| item | core | extended |
|---|---|---|
| documents x languages | 200 x 77 | 1000 x 77 |
| English tokens per language | 291 k | 1455 k |
| English sentences per language | 10648 | 53239 |
| target Gemma tokens, all languages | 43.1 M | 215.5 M |
| translator nllb200-3.3b (44 languages): output tokens / GPU hours | 18.5 M / 0.1 - 0.2 | 92.4 M / 0.5 - 1.0 |
| translator madlad400-7b-mt (44 languages): output tokens / GPU hours | 18.5 M / 0.4 - 0.6 | 92.4 M / 1.8 - 3.2 |
| translator sunflower-qwen3.8-27b (31 languages): output tokens / GPU hours | 13.0 M / 1.9 - 3.5 | 65.1 M / 9.6 - 17.3 |
| translator opus-mt-tc-bible-big-eng-mul (59 languages): output tokens / GPU hours | 24.8 M / 0.2 - 0.4 | 123.8 M / 0.8 - 1.8 |
| our model (3 checkpoints): output tokens / GPU hours | 129.3 M / 0.9 - 3.6 | 646.5 M / 4.5 - 18.0 |
| AfriCOMET (ref-based) triples / GPU hours | 2.46 M / 1.0 - 2.4 | 12.30 M / 4.9 - 11.8 |


Recommended bundle (NLLB-3.3B as internal referee + MADLAD-7B + Sunflower-27B + OPUS tc-bible, each on the languages it names; three checkpoints of our model; AfriCOMET-ref on every scored triple): **core 4.4 - 10.6 GPU-h** (translators 2.5 - 4.7, our model 0.9 - 3.6, AfriCOMET 1.0 - 2.4), **extended 22 - 53 GPU-h** (translators 12.7 - 23.3, our model 4.5 - 18.0, AfriCOMET 4.9 - 11.8). The 27B Sunflower model is 74% of the translator time; with MADLAD + NLLB + OPUS-MT only the translators cost 0.6 - 1.2 GPU-h (core).

Assumptions (all **estimates**, none measured on the target card): RTX PRO 6000 Blackwell 96 GB, memory bandwidth 1,792 GB/s (datasheet); a decoder's throughput is bounded by `batch x bandwidth / weight bytes` and a real engine reaches 25-45% of that bound with CTranslate2 or vLLM at the batches shown, 8-18% with HF `generate` (so multiply the HF numbers by 2-4); int8 weights for CTranslate2 (NLLB, MADLAD), bf16 for vLLM (Sunflower, TranslateGemma). Translator tokens = 0.75 x the Gemma tokens of the same target text (their vocabularies are denser on African languages; assumption). Anchors we do have: vLLM on our 270M model measured 5.3k tok/s with 4 sequences on a card at 100% load from another job (`gemma_rl_vllm_2026-10-17.md`), assumed 10-40k tok/s on a dedicated card with 256+ sequences; AfriCOMET-QE measured 290-350 pairs/s on the RTX 5070, assumed x1.5-3 on this card and x1.5 tokens for the reference segment; published for this very GPU: gemma-3-12b-it on vLLM 1,560 output tok/s at 50 concurrent requests (hosting-blog measurement quoted in `translators.md` section e), about 40-60% of the bandwidth bound, consistent with the factor used. The enc-dec systems (NLLB, MADLAD, OPUS-MT, Aya-101, mT0, Seamless) cannot be served by vLLM 0.31 (checked: architectures absent from its registry); CTranslate2 is not installed and its Blackwell build is unconfirmed, so the first thing to do on the GPU is a 10-minute pilot of NLLB-3.3B and MADLAD-7B-bt with CT2 and with HF to replace these assumptions.

**Bake-off cost** (FLORES+ dev, 997 sentences per language, only the languages each system names; estimate):

| bake-off system | languages | output tokens | GPU hours (decode only) | weights bf16 |
|---|---|---|---|---|
| nllb200-3.3b | 44 | 1.77 M | 0.01 - 0.02 | 6.6 GB |
| madlad400-7b-mt | 44 | 1.77 M | 0.03 - 0.06 | 14.4 GB |
| sunflower-qwen3.8-27b | 31 | 1.25 M | 0.18 - 0.33 | 54.8 GB |
| opus-mt-tc-bible-big-eng-mul | 59 | 2.38 M | 0.02 - 0.03 | 0.5 GB |
| translategemma-12b-it | 47 | 1.89 M | 0.13 - 0.24 | 26.4 GB |
| afriqueqwen3.5-9b-50langs-instruct | 41 | 1.65 M | 0.05 - 0.10 | 18.0 GB |
| AfriCOMET (ref-based) on the covered languages | | 263,208 triples | 0.10 - 0.25 | |
| **total** | | | **0.5 - 1.0** | |

The decoding is minutes; what costs time is fetching about 150 GB of weights (Sunflower-27B 55 GB, TranslateGemma-12B 26 GB, MADLAD-7B 17-30 GB, NLLB-3.3B 17 GB, AfriqueQwen-9B 18 GB) onto a box with 40 GB free: download, run, delete one system at a time.

## 6. Candidate translators, coverage and the bake-off

### 6.1 Systems (from the model cards, retrieved 2026-10-20; full table, licence paraphrases and the 77-row matrix in `translators.md`)

| system | params (B) | licence | class | languages named (proxy) | backends |
|---|---|---|---|---|---|
| `nllb-200-distilled-600m` | 0.615 | CC-BY-NC-4.0 | noncommercial | 44 (6) | hf, ctranslate2 |
| `nllb-200-distilled-1.3b` | 1.37 | CC-BY-NC-4.0 | noncommercial | 44 (6) | hf, ctranslate2 |
| `nllb-200-1.3b` | 1.37 | CC-BY-NC-4.0 | noncommercial | 44 (6) | hf, ctranslate2 |
| `nllb-200-3.3b` | 3.34 | CC-BY-NC-4.0 | noncommercial | 44 (6) | hf, ctranslate2 |
| `madlad400-3b-mt` | 2.94 | Apache-2.0 | permissive | 44 (7) | hf, ctranslate2 |
| `madlad400-7b-mt` | 8.3 | Apache-2.0 | permissive | 44 (7) | hf, ctranslate2 |
| `madlad400-7b-mt-bt` | 8.3 | Apache-2.0 | permissive | 44 (7) | hf, ctranslate2 |
| `madlad400-10b-mt` | 10.71 | Apache-2.0 | permissive | 44 (7) | hf, ctranslate2 |
| `translategemma-4b-it` | 4.97 | gemma | gemma_terms | 47 (8) | hf, vllm |
| `translategemma-12b-it` | 13.19 | gemma | gemma_terms | 47 (8) | hf, vllm |
| `translategemma-27b-it` | 28.84 | gemma | gemma_terms | 47 (8) | hf, vllm |
| `gemma-3-27b-it` | 27.43 | gemma | gemma_terms | 2 (0) | hf, vllm |
| `gemma-4-31b-it` | 31.27 | Apache-2.0 | permissive | 0 (0) | hf, vllm |
| `qwen3-32b` | 32.76 | Apache-2.0 | permissive | 3 (0) | hf, vllm |
| `aya-101` | 12.92 | Apache-2.0 | permissive | 15 (4) | hf |
| `tiny-aya-earth` | 3.35 | CC-BY-NC-4.0 + Cohere Labs Acceptable Us | noncommercial | 10 (3) | hf |
| `tiny-aya-global` | 3.35 | CC-BY-NC-4.0 + Cohere Labs Acceptable Us | noncommercial | 10 (3) | hf |
| `seamless-m4t-v2-large` | 2.31 | CC-BY-NC-4.0 | noncommercial | 12 (3) | hf |
| `mt0-xxl-mt` | 13.95 | Apache-2.0 | permissive | 13 (4) | hf |
| `bloomz-7b1-mt` | 7.07 | bigscience-bloom-rail-1.0 | other | 21 (5) | hf, vllm |
| `opus-mt-en-x` | 0.077 | Apache-2.0 | permissive | 35 (3) | hf, ctranslate2 |
| `opus-mt-tc-bible-big-eng-mul` | 0.241 | Apache-2.0 | permissive | 59 (5) | hf, ctranslate2 |
| `opus-mt-tc-bible-big-eng-bnt` | 0.24 | Apache-2.0 | permissive | 31 (3) | hf, ctranslate2 |
| `opus-mt-tc-bible-big-eng-afa` | 0.24 | Apache-2.0 | permissive | 8 (0) | hf, ctranslate2 |
| `sunflower-qwen3.8-27b` | 27.36 | Apache-2.0 | permissive | 31 (6) | hf, vllm |
| `sunflower-qwen3.5-9b` | 9.41 | Apache-2.0 | permissive | 31 (6) | hf, vllm |
| `afriqueqwen3.5-9b-50langs-instruct` | 8.95 | CC-BY-4.0 | permissive | 41 (6) | hf, vllm |
| `llamax3-8b-alpaca` | 8.03 | Llama-3 Community License (Hub tag: MIT) | llama | 20 (3) | hf, vllm |
| `n-atlas-llm` | 8.03 | N-ATLaS Open-Source Research and Innovat | other | 3 (0) | hf, vllm |

Coverage = "named in the system's own language list", not a quality claim. NLLB, MADLAD, OPUS-MT and the Qwen-based systems are the realistic pseudo-reference producers; the others are listed for completeness.

* **Licence of the outputs.** NLLB-200 / SeamlessM4T / Tiny Aya are CC-BY-NC-4.0 (the card does not say whether the NC clause reaches generated text, and calls NLLB a research model "not intended for document translation"); Gemma 1-3 / TranslateGemma: Google claims no rights in outputs, but a model trained on them is a "Model Derivative" carrying the Gemma terms; Llama-3-licensed systems forbid using outputs to improve another LLM; Apache-2.0 systems (MADLAD, OPUS-MT, Sunflower, Gemma 4, mT0, Aya-101) and CC-BY-4.0 AfriqueQwen are the clean path (attribution only). None of this is legal advice.
* **Published quality (not ours)** (MADLAD paper Table 17, chrF on FLORES-200 en->xx, MADLAD-7.2B vs NLLB-54B-MoE): Kinyarwanda 9.6 vs 53.8, Lingala 18.8 vs 50.8, Wolof 12.6 vs 31.8, Yoruba 20.3 vs 27.4, Luganda 30.3 vs 44.7, Oromo 29.3 vs 43.3, Zulu 48.6 vs 58.8, Bemba 17.7 vs 42.3; the 8 languages MADLAD has no `<2xx>` token for score about 2 in its 3B/10B variants. The 240M Apache-2.0 OPUS tc-bible Bantu model reports FLORES devtest chrF above MADLAD-7.2B on zu, tso, nso, kin, lin, tsn, ssw, bem (card numbers, implementation not guaranteed identical). Sunflower reports the best numbers on the languages it names, but on its own test set. **Expect the best permissive system to differ by language and to lose to NLLB on many: the bake-off decides, and a permissive system must be within `--licence-margin` (1.5 chrF++) of the best non-permissive one to win.**
* **Bake-off systems recommended** (permissive first, one ceiling): `sunflower-qwen3.8-27b` (31 languages), `madlad400-7b-mt-bt` (44), the three `opus-mt-tc-bible-big-eng-{bnt,afa,mul}` models (59 together, 240M each, trivially cheap) and `nllb-200-3.3b` (44, CC-BY-NC, as an internal referee and quality ceiling); optional: `translategemma-12b-it` (47 named, only sw and zu evaluated by Google), `afriqueqwen3.5-9b-50langs-instruct` (41), `gemma-4-31b-it` (Apache-2.0, no language list).

### 6.2 Where the human data and the systems leave gaps (measured / registry)

* Human sentence set for the bake-off: FLORES+ dev for 44 languages (downloaded, 20 MB, `raw/flores_dev`), SmolSent for 10 more (pcm efi ven nbl tiv kon ach nyn ndc kri), GlotStoryBook for 7 (dag kqn lun loz tsc kua her; story-aligned, noisier), IbomMT for ibb (CC-BY-NC), FLORES-MOZ for seh: **63 languages can be bake-off'ed; 14 cannot** (urh bin oke shi bbj swc iso wal yao sid ish dua bas wes): no pseudo-reference tier is possible for them, even where a system names them (iso, wal, bas, shi, swc).
* **No open system names** urh, bin, oke, bbj, yao, sid, ish, dua; wes has only a Nigerian-Pidgin proxy; **ndc** is named only by TranslateGemma (Gemma terms). No human reference exists for any of these either (they are among the 23 languages without human documents), so they cannot be in the benchmark unless somebody writes references.
* **Permissive coverage rests only on Bible-trained OPUS-MT** for efi, her, kqn, loz, lun, nus, shi, swc, tum, umb, vmw; expect tier C or N (domain mismatch: Bible vs news / science / law).
* NLLB-only gains over MADLAD: bem, kab, kam, kea, kik, lua, luo, mos, nus, tum, umb (MADLAD has no `<2xx>` token there); MADLAD-only: bas, ibb, iso, kri, kua, nbl, seh, tiv, tsc, ven, wal.

### 6.3 Coverage matrix of the key systems (all 77 languages; the other 21 systems are columns of `translators.md`)

Cell = the system's own code for the language; `code(p)` = a close variety only (not support); `-` = not named. NLLB codes are FLORES-200 codes (44 of the 77 languages are in FLORES-200).

| iso | language | nllb-200-3.3b | madlad400-7b-mt-bt | translategemma-27b-it | sunflower-qwen3.8-27b | afriqueqwen3.5-9b-50langs-instruct | opus-mt-tc-bible-big-eng-mul | opus-mt-en-x | aya-101 |
|---|---|---|---|---|---|---|---|---|---|
| yor | Yoruba | yor_Latn | yo | yo | yor | yor_Latn | yor | - | yor |
| hau | Hausa | hau_Latn | ha | ha | hau | hau_Latn | hau | ha | hau |
| ibo | Igbo | ibo_Latn | ig | ig | ibo | ibo_Latn | ibo | ig | ibo |
| pcm | Pidgin | - | - | pcm | pcm | - | pcm | - | - |
| efi | Efik | - | ibb(p) | efi | - | - | efi | efi | - |
| twi | Twi | twi_Latn | ak | ak | aka(p) | twi_Latn | twi | tw | twi |
| ewe | Ewe | ewe_Latn | ee | ee | ewe | ewe_Latn | ewe | ee | - |
| fon | Fon | fon_Latn | fon | fon | - | fon_Latn | fon | - | - |
| wol | Wolof | wol_Latn | wo | wo | wol | wol_Latn | wol | - | - |
| swh | Swahili | swh_Latn | sw | sw | swa | swh_Latn | swh | sw | swa |
| kin | Kinyarwanda | kin_Latn | rw | rw | kin | kin_Latn | kin | rw | - |
| som | Somali | som_Latn | so | so | som | som_Latn | som | - | som |
| orm | Oromo | gaz_Latn | om | om | orm | gaz_Latn | orm | om | - |
| amh | Amharic | amh_Ethi | am | am | amh | amh_Ethi | amh | - | amh |
| tir | Tigrinya | tir_Ethi | ti | ti | - | tir_Ethi | tir | ti | - |
| lug | Luganda | lug_Latn | lg | lg | lug | lug_Latn | lug | lg | - |
| lin | Lingala | lin_Latn | ln | ln | lin | lin_Latn | lin | ln | - |
| zul | Zulu | zul_Latn | zu | zu | zul | zul_Latn | zul | - | zul |
| xho | Xhosa | xho_Latn | xh | xh | xho | xho_Latn | xho | xh | xho |
| afr | Afrikaans | afr_Latn | af | af | afr | afr_Latn | afr | af | afr |
| sna | Shona | sna_Latn | sn | sn | sna | sna_Latn | sna | sn | sna |
| nya | Chichewa | nya_Latn | ny | ny | nya | nya_Latn | nya | ny | nya |
| sot | Sesotho | sot_Latn | st | st | sot | sot_Latn | sot | st | sot |
| tsn | Setswana | tsn_Latn | tn | tn | tsn | tsn_Latn | tsn | tn | - |
| mlg | Malagasy | plt_Latn | mg | mg | mlg | plt_Latn | mlg | mg | mlg |
| ibb | Ibibio | - | ibb | efi(p) | - | - | efi(p) | efi(p) | - |
| urh | Urhobo | - | - | - | - | - | - | - | - |
| bem | Bemba | bem_Latn | - | bem | bem | bem_Latn | bem | bem | - |
| tum | Tumbuka | tum_Latn | - | tum | - | - | tum | - | - |
| run | Kirundi | run_Latn | rn | rn | run | run_Latn | run | rn | - |
| ven | Venda | - | ve | ve | - | - | ven | - | - |
| tso | Tsonga | tso_Latn | ts | ts | - | tso_Latn | tso | ts | - |
| nso | Sepedi | nso_Latn | nso | nso | - | nso_Latn | nso | nso | nso |
| ssw | Swati | ssw_Latn | ss | ss | - | ssw_Latn | ssw | ss | - |
| nbl | South Ndebele | zul_Latn(p) | nr | nr | nbl | zul_Latn(p) | nbl | - | zul(p) |
| sag | Sango | sag_Latn | sg | sg | - | sag_Latn | sag | sg | - |
| tiv | Tiv | - | tiv | tiv | - | - | - | tiv | - |
| umb | Umbundu | umb_Latn | - | - | - | - | umb | umb | - |
| kmb | Kimbundu | kmb_Latn | kmb | - | - | kmb_Latn | kmb | - | - |
| mos | Mossi | mos_Latn | - | mos | - | mos_Latn | mos | mos | - |
| kik | Kikuyu | kik_Latn | - | ki | kik | kik_Latn | kik | - | - |
| luo | Luo | luo_Latn | - | luo | luo | luo_Latn | luo | luo | - |
| kam | Kamba | kam_Latn | - | - | - | kam_Latn | kam | - | - |
| bin | Edo | - | - | - | - | - | - | - | - |
| dag | Dagbani | - | - | - | dag | - | - | - | - |
| oke | Okpe | - | - | - | - | - | - | - | - |
| shi | Tachelhit | tzm_Tfng(p) | ber(p) | ber(p) | ber(p) | tzm_Tfng(p) | shi | - | - |
| dyu | Dyula | dyu_Latn | dyu | dyu | - | dyu_Latn | dyu | - | - |
| bam | Bambara | bam_Latn | bm | bm | bam | bam_Latn | bam | - | - |
| kbp | Kabiye | kbp_Latn | kbp | - | - | kbp_Latn | kbp | - | - |
| bbj | Ghomala | - | - | - | - | - | - | - | - |
| kon | Kongo | kon_Latn | kg | kg | - | kon_Latn | kon | kg | - |
| kea | Kabuverdianu | kea_Latn | - | - | - | kea_Latn | kea | - | - |
| kab | Kabyle | kab_Latn | ber_Latn(p) | ber-Latn(p) | kab | kab_Latn | kab | - | - |
| nus | Nuer | nus_Latn | - | nus | - | - | nus | - | - |
| swc | Congo Swahili | swh_Latn(p) | sw(p) | sw(p) | swa(p) | swh_Latn(p) | swc | swc | swa(p) |
| lua | Luba-Kasai | lua_Latn | lu(p) | lua | - | lua_Latn | lua | lua | - |
| kqn | Kaonde | - | - | - | - | - | - | kqn | - |
| lun | Lunda | - | - | - | - | - | - | lun | - |
| iso | Isoko | - | iso | iso | - | - | - | iso | - |
| loz | Lozi | sot_Latn(p) | st(p) | st(p) | sot(p) | sot_Latn(p) | sot(p) | loz | sot(p) |
| wal | Wolaytta | - | wal | - | - | - | wal | - | - |
| ach | Acholi | - | - | ach | ach | - | ach | - | - |
| tsc | Tswa | tso_Latn(p) | tsc | ts(p) | - | tso_Latn(p) | tso(p) | ts(p) | - |
| vmw | Makhuwa | - | - | - | - | - | vmw | - | - |
| kua | Kuanyama | - | kj | - | - | - | kua | - | - |
| seh | Sena | - | seh | - | - | - | seh | - | - |
| nyn | Nyankore | - | - | cgg(p) | nyn | - | nyn | - | - |
| her | Herero | - | - | - | - | - | her | - | - |
| yao | Yao | - | - | - | - | - | - | - | - |
| sid | Sidamo | - | - | - | - | - | - | - | - |
| ndc | Ndau | sna_Latn(p) | sn(p) | ndc-ZW | sna(p) | sna_Latn(p) | sna(p) | sn(p) | sna(p) |
| kri | Krio | - | kri | kri | - | - | kri | - | - |
| ish | Esan | - | - | - | - | - | - | - | - |
| dua | Duala | - | - | - | - | - | - | - | - |
| bas | Basaa | - | bas | - | - | - | bas | - | - |
| wes | Cameroon Pidgin | - | - | pcm(p) | pcm(p) | - | pcm(p) | - | - |

### 6.4 Language readiness (77 languages)

Columns: AfriCOMET status in `languages.py`; the human sentence set the bake-off would use and its size; SmolDoc documents (clean ones); FLORES+ passages; stitched 1024-token documents; number of open systems naming the language (permissive ones).

| iso | language | AfriCOMET | bake-off human sentences | human documents (SmolDoc, clean) | FLORES+ passages | stitched 1024 | open systems naming it (permissive) |
|---|---|---|---|---|---|---|---|
| `yor` | Yoruba | verified | flores_dev 997 | 668 (443) | 281 | 142 | 23 (10) |
| `hau` | Hausa | verified | flores_dev 997 | 667 (442) | 281 | 143 | 23 (12) |
| `ibo` | Igbo | verified | flores_dev 997 | 478 (253) | 281 | 103 | 24 (11) |
| `pcm` | Pidgin | verified | smol_sent 863 | 213 (105) | - | 53 | 6 (3) |
| `efi` | Efik | verified | smol_sent 863 | 130 (22) | - | 26 | 5 (2) |
| `twi` | Twi | verified | flores_dev 997 | 347 (122) | 281 | 78 | 16 (8) |
| `ewe` | Ewe | verified | flores_dev 997 | 212 (104) | 281 | 52 | 16 (9) |
| `fon` | Fon | verified | flores_dev 997 | 130 (22) | 281 | 26 | 14 (6) |
| `wol` | Wolof | verified | flores_dev 997 | 345 (120) | 281 | 76 | 19 (8) |
| `swh` | Swahili | verified | flores_dev 997 | 664 (439) | 281 | 143 | 26 (13) |
| `kin` | Kinyarwanda | verified | flores_dev 997 | 345 (120) | 281 | 78 | 18 (10) |
| `som` | Somali | verified | flores_dev 997 | 339 (114) | 281 | 77 | 20 (11) |
| `orm` | Oromo | verified | flores_dev 997 | 477 (252) | 281 | 105 | 19 (10) |
| `amh` | Amharic | verified | flores_dev 997 | 663 (438) | 281 | 142 | 22 (11) |
| `tir` | Tigrinya | verified | flores_dev 997 | 339 (114) | 281 | 79 | 15 (8) |
| `lug` | Luganda | verified | flores_dev 997 | 343 (118) | 281 | 78 | 20 (10) |
| `lin` | Lingala | verified | flores_dev 997 | 343 (118) | 281 | 78 | 19 (10) |
| `zul` | Zulu | verified | flores_dev 997 | 339 (114) | 281 | 78 | 24 (11) |
| `xho` | Xhosa | verified | flores_dev 997 | 344 (119) | 281 | 78 | 23 (12) |
| `afr` | Afrikaans | verified | flores_dev 997 | 209 (101) | 281 | 50 | 21 (12) |
| `sna` | Shona | verified | flores_dev 997 | 345 (120) | 281 | 77 | 24 (12) |
| `nya` | Chichewa | verified | flores_dev 997 | 354 (129) | 281 | 77 | 22 (12) |
| `sot` | Sesotho | verified | flores_dev 997 | 341 (116) | 281 | 77 | 20 (12) |
| `tsn` | Setswana | verified | flores_dev 997 | 130 (22) | 281 | 26 | 18 (10) |
| `mlg` | Malagasy | verified | flores_dev 997 | 474 (249) | 281 | 104 | 20 (11) |
| `ibb` | Ibibio | unsupported | ibom_mt 1012 | - | - | - | 4 (4) |
| `urh` | Urhobo | unsupported | - | - | - | - | 0 (0) |
| `bem` | Bemba | verified | flores_dev 997 | 130 (22) | 281 | 26 | 13 (6) |
| `tum` | Tumbuka | verified | flores_dev 997 | 130 (22) | 281 | 26 | 10 (2) |
| `run` | Kirundi | verified | flores_dev 997 | 130 (22) | 281 | 26 | 18 (10) |
| `ven` | Venda | listed | smol_sent 863 | 130 (22) | - | 26 | 9 (6) |
| `tso` | Tsonga | verified | flores_dev 997 | 130 (22) | 281 | 26 | 16 (8) |
| `nso` | Sepedi | verified | flores_dev 997 | 213 (105) | 281 | 51 | 18 (9) |
| `ssw` | Swati | verified | flores_dev 997 | 130 (22) | 281 | 26 | 15 (8) |
| `nbl` | South Ndebele | unverified | smol_sent 863 | 130 (22) | - | 26 | 11 (8) |
| `sag` | Sango | listed | flores_dev 997 | - | 281 | - | 14 (7) |
| `tiv` | Tiv | listed | smol_sent 863 | 130 (22) | - | 26 | 8 (5) |
| `umb` | Umbundu | verified | flores_dev 997 | - | 281 | - | 8 (3) |
| `kmb` | Kimbundu | verified | flores_dev 997 | - | 281 | - | 11 (7) |
| `mos` | Mossi | verified | flores_dev 997 | 130 (22) | 281 | 26 | 10 (3) |
| `kik` | Kikuyu | verified | flores_dev 997 | 130 (22) | 281 | 26 | 13 (5) |
| `luo` | Luo | verified | flores_dev 997 | 342 (117) | 281 | 78 | 14 (5) |
| `kam` | Kamba | verified | flores_dev 997 | - | 281 | - | 8 (3) |
| `bin` | Edo | unverified | - | - | - | - | 0 (0) |
| `dag` | Dagbani | unverified | glot_story 135 | - | - | - | 2 (2) |
| `oke` | Okpe | unverified | - | - | - | - | 0 (0) |
| `shi` | Tachelhit | unverified | - | - | - | - | 2 (2) |
| `dyu` | Dyula | listed | flores_dev 997 | 130 (22) | 281 | 26 | 13 (6) |
| `bam` | Bambara | verified | flores_dev 997 | 340 (115) | 281 | 77 | 16 (8) |
| `kbp` | Kabiye | verified | flores_dev 997 | - | 281 | - | 10 (6) |
| `bbj` | Ghomala | unverified | - | - | - | - | 0 (0) |
| `kon` | Kongo | listed | smol_sent 863 | 130 (22) | - | 26 | 15 (8) |
| `kea` | Kabuverdianu | unverified | flores_dev 997 | - | 281 | - | 8 (3) |
| `kab` | Kabyle | verified | flores_dev 997 | 208 (100) | 281 | 51 | 9 (5) |
| `nus` | Nuer | verified | flores_dev 997 | - | 281 | - | 8 (1) |
| `swc` | Congo Swahili | listed | - | - | - | - | 3 (3) |
| `lua` | Luba-Kasai | verified | flores_dev 997 | - | 281 | - | 11 (4) |
| `kqn` | Kaonde | listed | glot_story 420 | - | - | - | 1 (1) |
| `lun` | Lunda | listed | glot_story 420 | - | - | - | 1 (1) |
| `iso` | Isoko | listed | - | - | - | - | 8 (5) |
| `loz` | Lozi | listed | glot_story 960 | - | - | - | 1 (1) |
| `wal` | Wolaytta | listed | - | - | - | - | 6 (6) |
| `ach` | Acholi | unverified | smol_sent 863 | 130 (22) | - | 26 | 6 (3) |
| `tsc` | Tswa | unverified | glot_story 101 | - | - | - | 4 (4) |
| `vmw` | Makhuwa | unverified | flores_dev 997 | - | 281 | - | 2 (2) |
| `kua` | Kuanyama | listed | glot_story 520 | - | - | - | 6 (6) |
| `seh` | Sena | unverified | flores_moz 1012 | - | - | - | 6 (6) |
| `nyn` | Nyankore | unverified | smol_sent 863 | 130 (22) | - | 26 | 4 (4) |
| `her` | Herero | unverified | glot_story 549 | - | - | - | 2 (2) |
| `yao` | Yao | unverified | - | - | - | - | 0 (0) |
| `sid` | Sidamo | unverified | - | - | - | - | 0 (0) |
| `ndc` | Ndau | unverified | smol_sent 863 | 130 (22) | - | 26 | 3 (0) |
| `kri` | Krio | unverified | smol_sent 863 | 207 (99) | - | 51 | 8 (5) |
| `ish` | Esan | unverified | - | - | - | - | 0 (0) |
| `dua` | Duala | unverified | - | - | - | - | 0 (0) |
| `bas` | Basaa | unverified | - | - | - | - | 6 (6) |
| `wes` | Cameroon Pidgin | unverified | - | - | - | - | 0 (0) |

### 6.5 What was run on the CPU (measured, not bake-off results)

* `longdoc_bakeoff.py run`, Helsinki-NLP/opus-mt-en-ig, FLORES+ dev, 40 sentences, real AfriCOMET-STL and AfriCOMET-QE on CPU: chrF++ 30.2, BLEU 6.1, AfriCOMET-ref 0.526 (13 s of decoding at 3 sentences/s). `data/dryrun_opus_ig_results.json`.
* Registry system `opus-mt-tc-bible-big-eng-mul` (multi-target `>>xxx<<` prompt), 30 FLORES dev sentences per language: zul 36.8, sna 36.3, kin 33.6, yor 19.2 chrF++ (AfriCOMET 0.49 / 0.53 / 0.50 / 0.38); n = 30, CIs about +-3, **plumbing only**. `data/dryrun_tcbible_mul_results.json`.
* End-to-end smoke test (`longdoc_smoke_cpu.py`): ten stitched FLORES+ dev documents (mean 540 English tokens, ~21 sentences) translated by the 75M Marian model (a) sentence by sentence and (b) as one document, scored against the human reference and (c) against (a):

| translation (Helsinki-NLP/opus-mt-en-ig, 10 stitched FLORES+ dev documents, mean 540 English tokens) | chrF++ | BLEU | AfriCOMET-ref | coverage | docs with an omission | length ratio |
|---|---|---|---|---|---|---|
| A: sentence by sentence (the pseudo-reference protocol) against the HUMAN reference | 43.3 | 8.4 | 0.607 | 1.000 | 0.00 | 1.06 |
| B: the whole document in one call against the HUMAN reference | 8.5 | 0.1 | 0.072 | 0.155 | 1.00 | 0.14 |
| C: B against the pseudo reference A | 10.4 | 0.0 | 0.073 | 0.146 | 1.00 | 0.14 |

The same model that is a usable sentence translator (43.3 chrF++, 0.607) writes 14% of the expected length when given the whole document: the harness reports coverage 0.155 and an omission in 10 of 10 documents, which is exactly the long-context failure the benchmark is meant to expose. The small rise from B to C (8.5 -> 10.4 chrF++) is the self-agreement bias of a pseudo reference made by the evaluated model itself (section 8).

## 7. Does the failure metric see what it should? (measured)

`longdoc_validate.py`: 400 real human documents (stitched SmolDoc and SmolDoc documents of >= 25 sentences, mean 30 sentences, 10 languages: amh hau ibo kin sna som swh xho yor zul), the human translation damaged in known ways, 3 trials per document and damage; the intact translation must score perfectly (`data/metric_validation.json`):

| damage applied to a human translation | documents | document flagged | measured coverage (chars) | true share kept / omitted | note |
|---|---|---|---|---|---|
| none (false alarms) | 400 | 0.015 | 0.999 |  | loop flagged 0.0 |
| 1 sentence removed | 1200 | 0.554 | 0.979 | omitted 0.034, estimated 0.021 |  |
| 2 sentences removed | 1200 | 0.713 | 0.956 | omitted 0.068, estimated 0.044 |  |
| 4 sentences removed | 348 | 0.773 | 0.928 | omitted 0.113, estimated 0.072 |  |
| cut after 85% of the sentences | 1200 | 0.873 | 0.867 | kept 0.828 |  |
| cut after 60% of the sentences | 1200 | 0.988 | 0.629 | kept 0.580 |  |
| cut at 70% + a 3-word unit repeated 25 times | 1200 | 0.902 | 0.856 | kept 0.676 | loop detected 1.00 |
| 3 sentences replaced by the English | 1200 | 0.988 | 0.904 |  | copied sentences found 0.93 |
| two neighbouring sentences swapped | 1200 | 0.022 | 0.999 |  | not an omission: only chrF++ sees it |

* **No false alarms to speak of**: intact documents are flagged 1.5% of the time, swaps 2.2% (a swap is invisible to coverage by design; chrF++ sees it).
* **Truncation, loops, untranslated English are found** (flagged 87-99%; loop 100%; copied sentences 93%), and coverage tracks the true share (0.629 vs 0.580 kept; 0.867 vs 0.828), a slight over-estimate.
* **Single omissions are under-detected**: a length-only alignment cannot tell which of two neighbouring sentences of similar length is missing; one dropped sentence is flagged in 55% of the documents and the estimated omitted share is 62% of the true one (0.021 vs 0.034), four dropped sentences 64% (0.072 vs 0.113). Treat `coverage` and `omitted_sents` as lower bounds on what is missing; the sentence-index recall is only 25% (index precision 52%), so never use the indices.
* Cost: 10 documents of 200 sentences (2,500 words) are evaluated in 1.7 s on a laptop core including 200 bootstrap resamples; `rl/segments.split_sentences` itself is quadratic in the text length (0.2 s for 40,000 characters, over a second for a long generated document), hence the linear-time copy in `longdoc.py`.

## 8. Risks

1. **Pseudo-reference bias.** A reference-based metric against machine translation rewards outputs that resemble that translator: (a) the evaluated model may have been trained on, or distilled from, outputs of the same translators (NLLB / MADLAD translations circulate in many public African datasets; the lineage of our SFT and RL data was not audited here) -> circularity; (b) the ceiling of every score is the reference's own accuracy (tier C languages have none worth ranking); (c) sentence-level references have no discourse (terminology, pronouns, paragraph structure). Mitigations built in: tiers, two systems with agreement, human tracks that are authoritative, AfriCOMET-QE as a reference-free cross-check, `valid` filtering of flagged reference sentences.
2. **Translator training-data overlap with the sources and with human test sets**: MADLAD/NLLB/OPUS saw web text and Bible-heavy parallel text; Sunflower's data is undocumented; Sunbird itself reports a frontier model reproducing FLORES references verbatim. FLORES+ dev numbers (all published ones, and the bake-off) are an optimistic upper bound for LLM-type translators.
3. **Domain mismatch of Bible-trained systems** for the rare languages (bem tum umb kmb kam kik kon lua seh vmw tiv iso loz lun kqn her kua ...): fluent-sounding wrong text for news / law / science. Needs a native-speaker spot check of a handful of documents per rare language before a tier A/B label.
4. **Licence of the outputs and of the sources** (section 6.1; share-alike sources; per-document third-party content inside news pages; Gutenberg trademark text; OpenStax and OGL attribution strings; GFDL dual licences). The release needs an attribution file (`ATTRIBUTION.jsonl` is produced) and a per-document licence field.
5. **Excerpts are prefixes** of longer documents (flagged `excerpt`), so ends are abrupt; whole-document coherence is only as good as the prefix.
6. **Documents longer than the training length**: our SFT/RL sequences are 4,096 tokens in total (prompt + answer), the 2048 and 4096 bins need 4-12 k tokens of output; they test length extrapolation by design, and a model that cannot finish them is reported through `finished_rate`, not hidden.
7. **Contamination drift**: the scanner covers English text only; the target side, paraphrases and same-event articles are not seen; the flags are a snapshot of the pool and of the SFT/RL data of 2026-10-20. A public benchmark leaks into future training data: publish with a canary string and keep a private half (decision 9).
8. **Metric limits**: single omissions are under-detected (section 7); AfriCOMET covers 42 validated + 12 listed languages and its encoder truncates at 510 tokens, so it sees beads, not documents; chrF++ against a pseudo reference of 45-50 caps what can be learned about a 25-point model.
9. **Unverified by us**: published translator scores; Google/Microsoft language lists (read through a page extractor; wol, fon, dyu absent); TranslateGemma prompts for 17 of its 47 codes (not in the template's built-in table); CT2 on Blackwell; Sunflower / Tiny Aya config files (gated).

## 9. Decisions the owner must make

1. **Core and extended size.** Recommended core 200 for every language with a validated reference (tier A/B/C reported), extended 1000 for tier A/B only. Alternatives: 100 (+-0.4-1.5 chrF++ at n=100), 400 (+-0.2-0.8). Cost is not the constraint (section 5.2), reviewing is.
2. **Share-alike sources.** The main set excludes CC-BY-SA (Wikipedia, Wikivoyage, FLORES+ passages) because you excluded share-alike from training; a benchmark can be released under a share-alike licence, but then the whole composite should be (derivative English texts, their translations). Options: (a) exclude (current; the encyclopedia and travel domains stay thin: see the cells), (b) add them as a separate, clearly licensed SA subset, (c) release everything under CC-BY-SA. Note Wikipedia and Wikinews are the most contaminated sources (section 4).
3. **Which translators may produce released references.** Recommended: only permissive-licence systems (Apache-2.0 / CC-BY) for the released pseudo references; NLLB-200 (CC-BY-NC) only as an internal referee (bake-off ceiling, agreement), never released. If a permissive system is much weaker than NLLB for a language the language falls to tier C instead of using NLLB. Also decide whether Gemma-terms systems (TranslateGemma) may be used (the Model-Derivative clause).
4. **One system or two per language.** Recommended: two for tier A/B languages (agreement reported, the second system's references released as `ref_alt`), one for the rest.
5. **Tier thresholds** after the bake-off (provisional: A >= 45 chrF++ and 0.65, B >= 30 and 0.50); whether C languages are shown in the main table (recommended: shown, greyed, excluded from the macro average).
6. **Human-reference policy.** Recommended: H1/H2 are the authoritative numbers for the 45 SmolDoc languages, headline on the clean subset (5,182 documents, 22 per rare language) with "all" next to it; the pseudo track fills lengths and the 23 languages without human documents; for languages with neither, no ranking.
7. **Languages.** 14 languages have no human sentence set (no tier possible) and 9 no open system; recommended: drop urh bin oke bbj yao sid ish dua wes from the benchmark proper, keep iso wal bas shi swc as "unvalidated, unranked" if a system names them.
8. **Name and licence.** Working name `AfriLongDoc` (check for clashes). Recommended: code Apache-2.0 (the repository is AGPL-3.0; decide), data CC-BY-4.0 with a per-document `licence` and `attribution` field, OGL/PD documents carried as such; SA subset (if any) separate and labelled; references of Apache-2.0 translators released as CC-BY-4.0 derivative of the source licence.
9. **Leakage control.** Canary string in every file, a private half of the core set (scores via an evaluation script / server), or fully public; hashing of the references.
10. **Bins.** Keep 2048 and 4096 although they exceed the 4,096-token training length (they test extrapolation), or cap at 2048 for the first release.
11. **GPU time and downloads** for the bake-off (about 150 GB of weights, one system at a time on a box with 40 GB free) and for the core run; after the trainer is done.

## 10. Files, tests, commands

* Code (`gemma_mt/`, synced to the rental `longdoc_bench/code/`): `eval/longdoc.py` (harness), `longdoc_backends.py` (HF seq2seq / HF causal / vLLM / CTranslate2 / stub translators, prompt builders for the six prompt styles), `longdoc_bakeoff.py` + `scripts/longdoc_bakeoff.py`, `longdoc_pseudo.py` + `scripts/longdoc_translate.py`, `longdoc_refs.py` (human references, bake-off sentence sets), `longdoc_select.py` + `scripts/longdoc_build_source_set.py`, `longdoc_validate.py`, `longdoc_costplan.py`, `longdoc_systems.py` (registry, by the translator study), `longdoc_contam.py` + `scripts/longdoc_contam_*.py`, `scripts/longdoc_fetch_sources.py`, `scripts/longdoc_smoke_cpu.py`.
* Tests: `tests/test_longdoc.py`, `test_longdoc_pipeline.py` (mine), `test_longdoc_systems.py`, `test_longdoc_contam.py`, `test_longdoc_sources.py`. On the rental with real pytest (`python -m pytest tests/test_longdoc*.py`, `venv_gemma`): test_longdoc 22 + test_longdoc_pipeline 22 (mine, 44) + systems 95 + contam 14 + sources 38 = **191 tests, 191 passed in 13.5 s**; `tests/run_tests.py` (no pytest, laptop `.venv-comet`) passes my two files too.
* Commands (rental, `PYTHONPATH=/workspace/sft_build/longdoc_bench/code:/workspace/sft_build/pylibs`, `. /workspace/sft_build/env.sh` for the read-only Hub access, `nice -n 10`, `CUDA_VISIBLE_DEVICES=""` for anything CPU):

```
python -m gemma_mt.eval.longdoc_refs fetch-flores-dev --dest longdoc_bench/raw/flores_dev
python scripts/longdoc_build_source_set.py --pool-dir longdoc_bench/pool --contam-dir longdoc_bench/contam --out longdoc_bench/source_set --core 200 --extended 1000 --examples-md longdoc_bench/source_set/core_examples.md
python -m gemma_mt.eval.longdoc_refs human --out longdoc_bench/source_set/human
# GPU, later (one system at a time):
python scripts/longdoc_bakeoff.py run --system madlad400-7b-mt-bt --backend ctranslate2 --model-dir <ct2 dir> --device cuda --refset flores_dev --langs <iso ...> --comet worker --out longdoc_bench/bakeoff/madlad7b
python scripts/longdoc_bakeoff.py compare --dirs longdoc_bench/bakeoff/* --out longdoc_bench/bakeoff/tiers.json --report longdoc_bench/bakeoff/compare.md
python scripts/longdoc_translate.py --docs longdoc_bench/source_set/core/docs.jsonl --tiers longdoc_bench/bakeoff/tiers.json --backend hf --device cuda --out longdoc_bench/source_set/core/refs
python -m gemma_mt.eval.longdoc score --docs .../docs.jsonl --refs .../refs_pseudo_*.jsonl --preds preds.jsonl --tiers tiers.json --comet-ref worker --comet-qe worker --out results
python -m gemma_mt.eval.longdoc_validate --human-dir longdoc_bench/source_set/human --out metric_validation.json
```
