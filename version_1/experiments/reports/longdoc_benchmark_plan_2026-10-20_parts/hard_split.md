# The HARD / DOMAIN split of the long-document benchmark: medicine, law, maths-science (2026-10-20)

> **Medical and legal translations produced or scored here must not be used for clinical, diagnostic, treatment, legal or compliance purposes; no output has been verified by a domain expert.**

Status: English documents and metric configuration BUILT; nothing translated, nothing scored with a system, nothing uploaded (the coordinator does the single authorised push). Built on the rental from the pool of the long-document campaign (design: addendum section 7). Owner decision of 2026-10-20: any data licence is acceptable for this pure public research / education evaluation set (nothing is trained on it), but every record keeps provenance and the exact licence name, class (`pd | permissive | share_alike | noncommercial | unclear`) and evidence URL; the contamination filter (more than 2% of a document's windows in the SFT stream, the RL prompts v3 incl. `zz_b2048`, the existing benchmarks, or another pool document => not in the split) and the canary / private 10% apply as for the core.

## 1. What was built

| item | where |
|---|---|
| 150 English documents (50 medicine, 50 law, 50 maths-science), 135 public + 15 private (10%), 163,924 Gemma tokens, 5,336 sentences | rental `/workspace/sft_build/longdoc_bench/hard/` : `docs.jsonl` (PUBLIC rows only), `docs_private.jsonl` (private rows only), `docs_master_do_not_publish.jsonl` (all, `split` column), `ATTRIBUTION.jsonl`, `ATTRIBUTION_private.jsonl`, `SELECTION_REPORT.json`, `examples.md`, `metric_config.json` |
| fetchers for the new sources | `scripts/longdoc_fetch_hard.py` (arxiv, caselaw, eurlex, cuad, wikipedia, europepmc, who) |
| contamination scan of the new files (private copy of the coordinator's cache; the coordinator's `contam/` is untouched) | `scripts/longdoc_contam_hard.sh` |
| build script | `scripts/longdoc_build_hard.py` |
| library: formulas, domain cleaning, metrics, `evaluate_hard`, `METRIC_CONFIG` | `gemma_mt/eval/longdoc_hard.py` |
| tests (12, offline; pass on the laptop with `tests/run_tests.py` and on the rental with `python -m pytest tests/test_longdoc_hard.py`) | `tests/test_longdoc_hard.py` |

Row schema = the core `docs.jsonl` (`doc_id, domain, source, licence, licence_class, url, title, attribution, text, sents, para_starts, bin, n_tokens, n_tokens_full, excerpt, contam, canary, subset: "hard", split public|private`) plus `domain in {medicine, law, math_science}`, `subdomain`, `licence_evidence`, `licence_note`, `date`, `n_paragraphs, n_sentences` and the domain features of the excerpt (`n_formulas, formula_char_share, n_latex_cmds, n_numerals, n_protected, n_legal_refs, n_modality_cues, hard_features`). The canary string of `longdoc_release.CANARY` is in every row. The private split is chosen with the core's `longdoc_select.assign_private` (same strata). **Integration note for the coordinator:** the pool directory now also needs the new `hard_*__<class>.jsonl` files only after the single batch copy; `longdoc_select.DEFAULT_EXCLUDE_FILES` must contain `hard*` (otherwise the core selection would read the three new domains); the core must not use `docs_master_do_not_publish.jsonl`.

## 2. Sources

Pool files that were already there (excluded from the core, taken as they are; their licence evidence is in `sources.md` / `sources_v2.md`) and the new files fetched for this split (all fetched 2026-10-20 from the rental; User-Agent `SabiYarn-longdoc-benchmark/0.1 (research; contact ...)`; web hosts at 1 request per second with robots.txt honoured; Hub data streamed, nothing stored).

| domain | source key | what / provenance | licence | evidence (URL / quote) | citation / credit | pool docs | clean candidates | used |
|---|---|---|---|---|---|---|---|---|
| medicine | `pmc` | PubMed Central OA articles, CC BY (existing pool `pmc__permissive` + `pmc_ext__permissive`; Europe PMC REST) | CC BY (version per article XML) | licence URL inside each article's JATS (`licence_evidence`) | NLM PubMed Central open-access subset; authors, journal, DOI in `attribution` | 263 | 220 | 7 |
| medicine | `europepmc` | **new** PubMed Central OA articles that are NOT CC BY (CC BY-NC / -NC-ND / -NC-SA) via Europe PMC REST (`fetch hard europepmc`); abstract, introduction, results, discussion | CC BY-NC family (noncommercial) | article XML licence link, else the Europe PMC search field `license` | same, authors / journal / DOI in `attribution` | 70 | 68 | 8 |
| medicine | `fedhealth` | NIDDK / NIMH / NIDA health-information pages (existing pool `fedhealth__pd`) | US federal government work (public domain) | https://www.niddk.nih.gov/copyright (NIDDK: majority of information copyright free); NIMH / NIDA pages per document | institute credited; no endorsement implied | 243 | 70 | 5 |
| medicine | `who` | **new** WHO fact sheets (`fetch hard who`, 243 listed, HTML) | CC BY-NC-SA 3.0 IGO | https://www.who.int/about/policies/publishing/copyright | World Health Organization, fact-sheet title and URL | 120 | 73 | 6 |
| medicine | `govuk` | GOV.UK health guidance (existing pool `govuk__permissive`, rows labelled health) | Open Government Licence v3.0 | https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/ | Contains public sector information licensed under the OGL v3.0 | 154 | 71 | 7 |
| medicine | `openstax` | OpenStax health textbooks at the pinned CC-BY commit (existing pool `openstax_ccby_snapshot`; relicensed NC-SA upstream, allowed by the owner) | CC BY 4.0 at the pinned commit | LICENSE at the pinned commit (`extra.commit`) | OpenStax, book, Rice University | 320 | 264 | 6 |
| medicine | `cp_pressbooks` | Pressbooks health textbooks (Common Pile, existing pool) | CC BY 4.0 / CC0 | per chapter (Common Pile field) | Kandpal et al. 2025, arXiv:2506.05209 | 202 | 93 | 5 |
| medicine | `cp_libretexts` | LibreTexts health pages (Common Pile, existing pool) | CC BY 4.0 / 3.0 | per page (Common Pile field) | Kandpal et al. 2025 | 103 | 46 | 6 |
| law | `legislation` | UK Explanatory Notes windows (existing pool `legislation__permissive`, legislation.gov.uk) | Open Government Licence v3.0 | https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/ | Contains public sector information licensed under the OGL v3.0 | 224 | 150 | 8 |
| law | `federalregister` | Federal Register rules and notices (existing pool `federalregister__pd`) | US federal government work (public domain) | https://www.govinfo.gov/about/policies | agency and FR citation | 898 | 400 | 7 |
| law | `billsum` | US bills (BillSum, existing pool `billsum__pd`) | public domain (US Congress); dataset card CC0-1.0 | HF `FiscalNote/billsum` card; Kornilova & Eidelman 2019 | United States Congress, BillSum | 330 | 62 | 5 |
| law | `cp_usgpo` | US committee reports and hearings (Common Pile `usgpo`, existing pools `cp_usgpo_a/b__pd`) | public domain (GPO) | https://www.govinfo.gov/about/policies | GovInfo / GPO | 563 | 153 | 7 |
| law | `caselaw` | **new** US court opinions (Caselaw Access Project via Common Pile `caselaw_access_project`; `fetch hard caselaw`) | Public Domain / CC0 | https://case.law/about/ ; Common Pile card: only public-domain documents | Caselaw Access Project (Harvard LIL) | 220 | 126 | 8 |
| law | `eurlex` | **new** EU regulations, directives, decisions, recommendations (Hub mirror `joelniklaus/eurlex_resources`, English; `fetch hard eurlex`) | CC BY 4.0 (EU reuse policy 2011/833/EU) | Hub card `license: cc-by-4.0` | EUR-Lex, Publications Office of the EU, CELEX id | 130 | 64 | 9 |
| law | `cuad` | **new** commercial contracts (CUAD v1 `full_contract_txt`, SEC EDGAR exhibits; `fetch hard cuad`) | CC BY 4.0 (dataset); contracts are public filings | Hub card `license: cc-by-4.0`; https://www.atticusprojectai.org/cuad | Hendrycks et al. 2021, CUAD, NeurIPS D&B | 98 | 90 | 6 |
| math_science | `arxiv` | **new** arXiv papers, abstract + introduction with LaTeX kept (Common Pile `arxiv_papers`; `fetch hard arxiv`) | CC BY 4.0 / 3.0 / CC0 per paper | per-paper licence from the arXiv metadata (Common Pile) | authors, title, arXiv id | 220 | 203 | 17 |
| math_science | `wikipedia` | **new** Wikipedia mathematics / physics articles, MediaWiki API, formulas as LaTeX (`fetch hard wikipedia`, revision id stored) | CC BY-SA 4.0 (share-alike) | https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use | Wikipedia contributors, title, URL, revision | 61 | 59 | 16 |
| math_science | `openstax` | OpenStax science textbooks (formula-free; existing pool snapshot) | CC BY 4.0 at the pinned commit | LICENSE at the pinned commit | OpenStax, book, Rice University | 320 | 219 | 6 |
| math_science | `cp_libretexts` | LibreTexts science pages (formula-free; Common Pile, existing pool) | CC BY 4.0 / 3.0 | per page | Kandpal et al. 2025 | 187 | 97 | 6 |
| math_science | `cp_pressbooks` | Pressbooks science textbooks (formula-free; Common Pile, existing pool) | CC BY 4.0 / CC0 | per chapter | Kandpal et al. 2025 | 200 | 107 | 5 |

Licence evidence quoted (read 2026-10-20): **WHO** (https://www.who.int/about/policies/publishing/copyright): "WHO's open access applies to: all publications published by WHO CC BY-NC-SA 3.0 IGO". **Wikipedia** (https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use): contributed text is licensed under "Creative Commons Attribution-ShareAlike 4.0 International License" and the GFDL. **CAP** (Common Pile `caselaw_access_project` card): "we only included documents that were in the public domain". **EUR-Lex** (Hub card of `joelniklaus/eurlex_resources`): `license: cc-by-4.0` (the EUR-Lex legal notice itself is behind a WAF from the rental; the Commission reuse decision 2011/833/EU is the legal basis). **CUAD** (Hub card `theatticusproject/cuad`): `license: cc-by-4.0`; the contracts are public SEC EDGAR exhibits. **arXiv** papers: per-paper licence read by the Common Pile from the arXiv metadata (CC BY and CC0 only). **PubMed Central** CC BY: licence URL inside each article's JATS XML (existing pool); non-CC-BY (CC BY-NC, -NC-ND, -NC-SA): same, fetched now through Europe PMC.

Not used (reason): **ProofWiki** (CC BY-SA; `proofwiki.org/w/api.php` answers 403 from the rental and no NaturalProofs mirror was found on the Hub under the expected name: not fetched, nothing else substituted), **OpenStax / LibreTexts with MathML** (a CNXML extractor that keeps MathML would be new work; the existing pool files strip equations, so their science rows are used as formula-free prose and are only a filler in maths-science), **CDC / NCI** (cdc.gov and cancer.gov answered HTTP 200 from the rental in a probe, but were not fetched for lack of time; the NIH institutes and WHO fact sheets fill that role), **MedlinePlus** (the topics file URL I guessed returned 404; not pursued), **congress.gov, state.gov, loc.gov, eur-lex.europa.eu** (reported as blocked from the rental in the earlier campaign, not retried; EUR-Lex through the Hub mirror), `joelniklaus/eurlex_resources` sanctions lists (dropped because they name individuals with date and place of birth: personal data).

## 3. Selection funnel

Candidates: pool documents mapped to a hard domain -> contamination flag `clean` (flags from the scan of ALL pool files incl. the new ones: SFT stream, RL prompts v3 incl. b2048, existing benchmarks, duplicates between pool documents; weak = a share between 2% and 10%, contaminated above, are both dropped) -> not in the 500-document core -> not a b2048 RL-prompt URL -> domain cleaning (`longdoc_hard.paragraph_ok_hard`: running prose only, formulas / numbers / citations kept, URLs / e-mails / phones removed; the first run of acceptable paragraphs is the document) -> exact duplicate -> length and quality (`text_quality_hard`) -> a bin fits.

| step (documents) | medicine | law | math_science |
|---|---|---|---|
| pool | 1475 | 2463 | 988 |
| drop_in_core_500 | 0 | 0 | 15 |
| drop_url_in_rl_b2048_prompts | 0 | 0 | 0 |
| drop_no_contamination_flag | 0 | 0 | 0 |
| drop_contam_contaminated | 220 | 653 | 129 |
| drop_contam_weak | 258 | 593 | 80 |
| drop_dup_of_other_pool_doc | 0 | 0 | 0 |
| drop_no_acceptable_paragraph | 0 | 1 | 0 |
| drop_too_short | 74 | 103 | 62 |
| drop_exact_duplicate | 0 | 2 | 0 |
| drop_no_bin_fits | 15 | 34 | 15 |
| drop_few_sentences | 0 | 9 | 0 |
| drop_fragments | 1 | 6 | 0 |
| drop_long_sentences | 2 | 5 | 0 |
| drop_ocr_single_letters | 0 | 1 | 1 |
| drop_sentence_length | 0 | 11 | 0 |
| drop_url_or_email | 0 | 0 | 1 |
| **candidates** | 905 | 1045 | 685 |
|   of which can serve bin 512 | 887 | 900 | 659 |
|   of which can serve bin 1024 | 606 | 761 | 363 |
|   of which can serve bin 2048 | 306 | 510 | 134 |
|   of which can serve bin 4096 | 83 | 240 | 27 |

Clean candidates per domain and source (a document can serve every bin it is long enough for):

| domain | source | clean candidates |
|---|---|---|
| law | billsum | 62 |
| law | caselaw | 126 |
| law | cp_usgpo | 153 |
| law | cuad | 90 |
| law | eurlex | 64 |
| law | federalregister | 400 |
| law | legislation | 150 |
| math_science | arxiv | 203 |
| math_science | cp_libretexts | 97 |
| math_science | cp_pressbooks | 107 |
| math_science | openstax | 219 |
| math_science | wikipedia | 59 |
| medicine | cp_libretexts | 46 |
| medicine | cp_pressbooks | 93 |
| medicine | europepmc | 68 |
| medicine | fedhealth | 70 |
| medicine | govuk | 71 |
| medicine | openstax | 264 |
| medicine | pmc | 220 |
| medicine | who | 73 |

Selection (`longdoc_build_hard.select`): 50 documents per domain, bins 512 / 1024 / 2048 / 4096 = 40 / 35 / 20 / 5% (largest remainder), bins from the longest to the shortest, inside a (domain, bin) cell round-robin over sources with at most 35% of a DOMAIN per source (17 of 50), documents with the domain features first (medicine: at least 4 numerals or 3 acronym / chemical / drug tokens; law: a legal reference or 3 modality cues; maths-science: at least 3 formulas) and the rest only when a feature-rich document is not available; a shortfall of a cell moves to the next smaller bin of the same domain. Shortfalls: {} / unfilled after the last bin: {}.

## 4. Composition

All 150 documents (public + private) by domain and bin (Gemma tokens of the English excerpt, whole paragraphs, one excerpt per document):

| domain | 512 | 1024 | 2048 | 4096 | total |
|---|---|---|---|---|---|
| medicine | 20 | 18 | 10 | 2 | 50 |
| law | 20 | 18 | 10 | 2 | 50 |
| math_science | 20 | 18 | 10 | 2 | 50 |
| total | 60 | 54 | 30 | 6 | 150 |

Public file (`docs.jsonl`):

| domain | 512 | 1024 | 2048 | 4096 | total |
|---|---|---|---|---|---|
| medicine | 18 | 16 | 9 | 2 | 45 |
| law | 18 | 16 | 9 | 2 | 45 |
| math_science | 18 | 16 | 9 | 2 | 45 |
| total | 54 | 48 | 27 | 6 | 135 |

By source (diversity cap 35% of a domain):

| domain | source | documents | share of domain |
|---|---|---|---|
| medicine | europepmc | 8 | 16% |
| medicine | pmc | 7 | 14% |
| medicine | govuk | 7 | 14% |
| medicine | cp_libretexts | 6 | 12% |
| medicine | who | 6 | 12% |
| medicine | openstax | 6 | 12% |
| medicine | fedhealth | 5 | 10% |
| medicine | cp_pressbooks | 5 | 10% |
| law | eurlex | 9 | 18% |
| law | legislation | 8 | 16% |
| law | caselaw | 8 | 16% |
| law | cp_usgpo | 7 | 14% |
| law | federalregister | 7 | 14% |
| law | cuad | 6 | 12% |
| law | billsum | 5 | 10% |
| math_science | arxiv | 17 | 34% |
| math_science | wikipedia | 16 | 32% |
| math_science | cp_libretexts | 6 | 12% |
| math_science | openstax | 6 | 12% |
| math_science | cp_pressbooks | 5 | 10% |

By licence class:

| domain | licence classes (documents) |
|---|---|
| medicine | noncommercial 14, pd 5, permissive 31 |
| law | pd 27, permissive 23 |
| math_science | pd 2, permissive 32, share_alike 16 |

By exact licence name:

| licence | documents |
|---|---|
| CC-BY-4.0 | 51 |
| CC-BY-SA-4.0 | 16 |
| OGL-3.0 | 15 |
| Public domain (US federal government work) | 12 |
| CC-BY-4.0 (EU reuse) | 9 |
| Public Domain / CC0 (Caselaw Access Project) | 8 |
| Public domain (US government work) | 7 |
| CC-BY-4.0 (CUAD v1) | 6 |
| CC-BY-NC-SA-3.0-IGO | 6 |
| Public domain (US Congressional bill; dataset CC0-1.0) | 5 |
| CC-BY (version not stated in the article) | 5 |
| CC-BY-NC-ND-4.0 | 3 |
| CC-BY-NC-SA (version not stated; Europe PMC licence field) | 2 |
| CC0-1.0 | 1 |
| Public domain | 1 |
| CC-BY-NC-ND (version not stated; Europe PMC licence field) | 1 |
| CC-BY-NC (version not stated; Europe PMC licence field) | 1 |
| CC-BY-NC-SA-4.0 | 1 |

Domain features of the selected documents (share with the features of section 3): {"law": 1.0, "math_science": 0.58, "medicine": 1.0}; documents with at least one formula span: {"law": 0, "math_science": 30, "medicine": 0}. Maths-science is therefore only partly equation-heavy: arXiv and Wikipedia carry the formulas, the formula-free OpenStax / LibreTexts / Pressbooks science prose is the filler for the diversity cap.

## 5. Contamination

Scan of the new `hard_*` files together with ALL pool files (the coordinator's scan cache was copied, only new files were scanned; `scripts/longdoc_contam_hard.sh`, CPU, 10 processes, nice 10; the coordinator's `contam/` directory was not touched). Sources compared against: the SFT stream (`gemma/packed_fwd` train + val, English side of the en2x samples, token n-grams and decoded word n-grams), the RL prompts v3 (incl. `zz_b2048`), the existing benchmarks (`bench/test_sets`, `bench_dev`), and duplicates between pool documents (more words wins). Flag: clean = every share < 2%; weak = largest share 2-10%; contaminated = above 10%, a verbatim run of >= 50 words (RL / benchmark) or >= 64 tokens (SFT), or >= 50% duplicate share.

| pool file (new) | documents | clean | weak | contaminated |
|---|---|---|---|---|
| `hard_arxiv__permissive` | 220 | 204 | 14 | 2 |
| `hard_caselaw__pd` | 220 | 127 | 75 | 18 |
| `hard_cuad__permissive` | 98 | 90 | 7 | 1 |
| `hard_eurlex__permissive` | 130 | 64 | 26 | 40 |
| `hard_europepmc__noncommercial` | 70 | 70 | 0 | 0 |
| `hard_who__noncommercial` | 120 | 73 | 43 | 4 |
| `hard_wikipedia__share_alike` | 61 | 59 | 1 | 1 |

Whole pool in this scan: {"clean": 30192, "weak": 4411, "contaminated": 23874}.

Selected documents (150): flags {"clean": 150}; largest SFT share 0.0150, largest RL-prompt share 0.0199, largest benchmark share 0.0020, largest duplicate share 0.0187 (threshold for `clean`: < 0.02 for each of the first three; the duplicate rule drops a document that is a copy of another pool document). Documents dropped by the filter, per domain, are in the funnel table above (`drop_contam_*`, `drop_dup_of_other_pool_doc`, `drop_in_core_500`, `drop_url_in_rl_b2048_prompts`).

## 6. The metrics (`gemma_mt/eval/longdoc_hard.py`, configuration `hard/metric_config.json`)

`evaluate_hard(refdocs, hyps, iso=..., with_core=True)` returns for ONE language and ONE system the per-document rows (`per_doc`), the per-language aggregates with 95% bootstrap CIs over documents (`metrics`, `rates`, `by_domain`), the modality audit list, the learned cue lexicon and, with `with_core`, the result of `longdoc.evaluate_language` unchanged (chrF++, BLEU, AfriCOMET, coverage, loops, length ratio, by bin and by domain). `evaluate_hard_run` loops over languages. All functions are pure Python (formula masking, regexes, multisets, edit distance); the only dependency is `longdoc` for the sentence beads. Every metric is a string-level check that can prove an error but not correctness.

1. **Numbers, units, dates** (`number_metrics`). Digits of every script are mapped to ASCII (Arabic-Indic, Devanagari, N'Ko, ...). Numerals are read from the English source with the English convention (`,` thousands, `.` decimal) and from the translation under both the English and the European reading (`1.234,5`, `3,5`; no-break / thin / narrow no-break space as thousands separator); the multiset of source numerals is matched against the multiset of translation numerals (any reading may match; formulas are masked first so that `$x_1$` does not count). Reported: `num_precision / recall / f1`; the same without small integers (`num_ge11_*`, integers 0..10 may be spelled out in the target language, so they are not penalised blindly); `missing_small_int_share` (share of the missing numerals that are small integers: a high value says "probably word numerals", not "error"); `copy_rate` (= numeral recall); `year_recall` (4-digit numerals 1000-2100); number+unit pairs with symbol units only (`%`, `°C`, `mg`, `mmHg`, `kg`, `km`, `$`, `EUR`, ... ; the unit words "years", "days", "percent" are translated and are not checked) as `pair_precision / recall / f1`, accepting `5 mg` and `mg 5` orders and `$5` / `5 USD` equivalence. `any_error_strict` = any missing or extra numeral; `any_error_lenient` = the same ignoring small integers; the language-level `rates.any_error_*` are the shares of documents with an error (Wilson CIs). Numerals the target language writes as words are therefore visible (strict minus lenient) but not charged in the lenient rate.
2. **Protected tokens** (`protected_metrics`). Classes (regexes in the config): acronyms and gene / chemical names (`WHO`, `BRCA1`, `CO2`, `COVID-19`), INN-style drug names by stem (`-mab`, `-nib`, `-vir`, `-cillin`, `-statin`, ... : a heuristic), Latin binomials (heuristic: capitalised genus + Latin-looking epithet; sentence-initial false positives are possible), statute / section / article locators (`Section 5(1)(a)`, `Article 12`: the locator after the keyword is checked, the keyword is translated), EU numbers (`2016/635`, `C-123/45`), US citations (`410 U.S. 113`, `5 U.S.C. 552`), numeric citation markers `[12]`, author-year citations, URLs (the documents contain none: removed). Recall = sum over distinct tokens of min(source count, translation count) / source count, overall and per class. Classes marked `latin_only` (drug, Latin names, author-year) are scored only when at least 80% of the translation's letters are Latin / Greek script, because names are legitimately transliterated in other scripts; `protected_latin_target` says which case applied.
3. **Formulas** (`formula_metrics`). Spans `$..$`, `$$..$$`, `\(..\)`, `\[..\]`, LaTeX math environments and `<math>` are found in the source and in the translation and compared on their content without delimiters and white space. `formula_exact_share` = multiset overlap / number of source spans; `formula_ordered_share` = longest common subsequence of the span sequence / number of source spans; `formula_edit_distance` and `formula_edit_norm` = Levenshtein distance of the span sequences (normalised by the longer sequence); `formula_extra` = spans in the translation that are not in the source. Documents without a formula are excluded from these averages. `mask_formulas_with_ids` / `unmask_formulas` replace the spans by `[[F1]]` ... for a system that should not touch them.
4. **Term consistency** (`term_consistency`, alignment-free, a heuristic). English terms = content unigrams / bigrams (light stemming, stop words removed) that occur in at least 3 sentences of the document (up to 20 terms). The beads are those of `longdoc.plan_document` (Gale-Church sentence beads between the source and the translation). For a term the candidate renderings are the 5-character prefixes of the target words of its beads; a prefix qualifies if (share of the term's beads containing it) - (share of the other beads containing it) is at least 0.30 (this removes function words and the 40 most frequent words that occur in more than 60% of the beads); the qualifying prefix covering most of the term's beads is the "modal rendering" and the consistency of the term is that coverage; terms without a qualifying prefix are "undetermined" and not counted. The document value is the mean over determined terms. The same statistic is computed on the (sentence-aligned) pseudo-reference as a baseline (`term_consistency_ref`): read the system value relative to the baseline of the same language, not as an absolute. Limits: it fails when the bead alignment is wrong, when a term is rendered by several words or by an inflected form whose first five characters change, and it cannot tell a consistent wrong rendering from a right one. Protected terms that repeat (>= 3 times) must keep one surface form: `protected_term_consistency` = share of their occurrences found verbatim in the translation (Latin-script targets).
5. **Modality and negation audit** (`modality_flags`, `learn_cue_lexicon`) - a flag list for human audit, NOT a score. Source cues: not, no, never, cannot, without, neither, nor, n't, must, shall, required, prohibited, may. Without a target-language lexicon (none is invented here) the equivalents are learned from the sentence-aligned reference of the language: for each cue the target 5-character prefixes with the highest contrast between cue and non-cue sentences (at least 12 cue sentences, contrast >= 0.35). A bead is flagged `cue_missing` when the learned equivalent is in the reference bead but not in the translation bead; `short_vs_ref` when no equivalent was learned and the translation bead is shorter than 0.6x the reference bead; `copied` / `empty`. The list (`modality_flags`, at most 200 per language and system) is meant for a human who reads the flagged sentence pairs; `rates.docs_with_modality_flag` is the share of documents with a cue that have at least one flag and is only a pointer for where to look.
6. **Core metrics** (`with_core=True`): `longdoc.evaluate_language` unchanged.

Bootstrap: 1000 resamples of the documents (percentile 95% CI), `seed 0`, as in the core. The thresholds and regexes above are saved in `hard/metric_config.json` (`longdoc_hard.save_config`); a score table of this split should quote its `version`.

**Self-check of the implementation on the real documents (synthetic, not a translation).** The English source of all 150 documents is used as its own "translation" (identity) and then damaged in a known way; the metrics must be 1.0 for the identity and drop for the damaged aspect only (`hard/selfcheck.json`; the numeral, formula and acronym damages are crude regex edits that also touch digits inside formulas and numerals inside acronyms such as COVID-19, so the neighbouring columns move a little; 30 documents have formulas):

| perturbation of the English source used as the 'translation' | num_f1 | num_ge11_f1 | pair_f1 | protected_recall | formula_exact_share | formula_ordered_share | any_error_strict | any_error_lenient |
|---|---|---|---|---|---|---|---|---|
| identity | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 |
| every 5th numeral +1 | 0.909 | 0.909 | 0.924 | 0.961 | 0.941 | 0.94 | 0.748 | 0.64 |
| 1/3 of formulas altered (x->y) | 1.0 | 1.0 | 1.0 | 1.0 | 0.929 | 0.929 | 0.0 | 0.0 |
| acronyms deleted | 0.98 | 0.986 | 0.999 | 0.309 | 0.982 | 0.982 | 0.086 | 0.05 |

Density of what the metrics look at, mean per document: numerals 22.6 (medicine) / 37.1 (law) / 18.9 (maths-science); protected tokens 19.9 / 14.8 / 14.4; number+unit pairs 4.1 / 2.9 / 2.2; formula spans 20.3 per formula-bearing maths-science document (30 of 50 have any); in the whole split the protected tokens are 1,604 acronyms, 252 numeric citation markers, 206 statute locators, 151 chemical / gene names, 102 EU numbers, 47 Latin names, 41 author-year citations, 29 US citations, 20 drug names (stem heuristic); 1,357 modality / negation cue words. The term-consistency baseline of English against itself is 0.986 (149 documents), i.e. the heuristic reads a perfectly consistent text as consistent; on a real reference it is the baseline of the language.

## 7. Examples (60 words each, public file)

#### medicine

* `cp_libretexts:42e8988a077b5b` | bin 512 (526 Gemma tokens) | CC-BY-4.0 | Erin O'Hara-Leslie, Amdra C. Wade, Kimberly B. McLain, SUNY Broome, 4.5: Problems Affecting the Family and How Children React to Stress, htt | https://med.libretexts.org/Bookshelves/Allied_Health/Foundations_for_Assisting_in_Home_Care_(McLain_O'Hara-Leslie_and_Wade)/04%3A_Working_with_Children/4.05%3A_Problems_Affecting_the_Family_and_How_Children_React_to_Stress
  > Explain that the separation is not the child’s fault and that they are loved by both parents and the entire family. Ensure the child understands that they will still be taken care of by their parents and others who love them. Give the child room to express their thoughts and feelings. Some children may prefer to talk about and others ...

* `pmc:PMC12616657` | bin 1024 (1008 Gemma tokens) | CC-BY (version not stated in the article) | Saeedzai SA, Knight J, Filipe L, Moore S, Loevinsohn B. (2025). Socio-economic determinants for reduced uptake of routine childhood vaccinat | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12616657/
  > Coverage of many routine childhood vaccinations in Afghanistan is low and declining. For instance, only 60% of eligible children received doses of the pentavalent vaccine in 2013–18, and this fell to 51% in 2022/23. To help explain the underlying causes behind this trend, this study aims to identify socio-demographic factors relating to caregivers whose children are either entirely unvaccinated or ...

#### law

* `hard_eurlex:32014H0729(09)` | bin 4096 (4112 Gemma tokens) | CC-BY-4.0 (EU reuse) | Council Recommendation of 8 July 2014 on the National Reform Programme 2014 of France and delivering a Council opinion on the Stability Prog | https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32014H0729(09)
  > On 26 March 2010, the European Council agreed to the Commission's proposal to launch a new strategy for growth and jobs, Europe 2020, based on enhanced coordination of economic policies, which focuses on the key areas where action is needed to boost Europe's potential for sustainable growth and competitiveness. On 13 July 2010, the Council, on the basis of the ...

* `hard_caselaw:adb392777d8831` | bin 4096 (3782 Gemma tokens) | Public Domain / CC0 (Caselaw Access Project) | IN RE KIARA R., Caselaw Access Project (Harvard Law School Library Innovation Lab), record conn-app_129/html/0604-01.html | https://static.case.law/
  > The respondent mother appeals from the decision of the trial court not to rule on a portion of an emergency motion she had filed to restore, inter alia, her unsupervised visitation with her minor daughter but, rather, to refer the matter for an administrative hearing. The respondent claims that the court erred in (1) failing to hold an evidentiary hearing ...

#### math_science

* `hard_wikipedia:1607648` | bin 512 (423 Gemma tokens) | CC-BY-SA-4.0 | Wikipedia contributors, "Postulates of special relativity", Wikipedia, The Free Encyclopedia, https://en.wikipedia.org/wiki/Postulates_of_sp | https://en.wikipedia.org/wiki/Postulates_of_special_relativity
  > ... class of inertial frames of reference. Each inertial frame of reference provides a coordinate system $(x_{1},x_{2},x_{3},t)$ for events in the spacetime M. Furthermore, this frame of reference also gives coordinates to all other physical characteristics of objects in the spacetime; for instance, it will provide coordinates $(p_{1},p_{2},p_{3},E)$ for the momentum and energy of an object, coordinates $(E_{1},E_{2},E_{3},B_{1},B_{2},B_{3})$ for an electromagnetic ...

* `hard_arxiv:2401.12805` | bin 2048 (1769 Gemma tokens) | CC-BY-4.0 | N. Lyutorovich, Description of the odd 249-253No nuclei using Skyrme functionals with, arXiv:2401.12805, CC-BY-4.0 | https://arxiv.org/abs/2401.12805
  > ... [5]. The EDF had the spin-spin terms constructed from the Landau-Migdal interaction with the parameters $g$ and $g^{\prime}$. The $g,g^{\prime}$ and new spinorbit parameters were fitted to reproduce the basic experimental characteristics of the M1 excitations in 208Pb within the renormalized timeblocking approximation and, at the same time, to describe the nuclear groundstate properties with approximately the same accuracy as ...


## 8. Caveats

* **No domain-expert review.** The text was selected and cleaned by rules; nobody checked the medical or legal content, and no translation or score has been checked by a clinician, lawyer or native speaker. Medical and legal translations produced or scored here must not be used for clinical, diagnostic, treatment, legal or compliance purposes.
* **Excerpts are not complete documents.** A court opinion, a regulation or an article is cut at a paragraph boundary at the bin length; legal cross-references to articles outside the excerpt remain. arXiv excerpts are abstract + introduction (no results or proofs); PubMed Central excerpts are abstract / introduction / discussion prose (the older pool files drop methods and results); Wikipedia excerpts are the plain-text extract with `{\displaystyle ..}` converted to `$..$`, which sometimes leaves a display equation inside a sentence.
* **Formula-free science filler.** Maths-science has fewer equation-bearing documents than the name suggests (section 4) because the diversity cap (17 of 50 per source) needs more than two sources and ProofWiki was unreachable.
* **Licence mix.** The split contains share-alike (Wikipedia CC BY-SA 4.0, WHO CC BY-NC-SA 3.0 IGO) and non-commercial (PubMed Central CC BY-NC*, WHO) material; derivative translations and score tables of those documents carry the same terms. The owner decision allows it for this public research / education set; a release must reproduce `ATTRIBUTION.jsonl` and state the per-document licences. Third-party material inside a document (figures, quoted text, party names in court opinions as published) may carry other terms or be personal data in the original; URLs, e-mails and telephone numbers were removed by rule, sanctions lists were dropped, court-opinion party names are those of the public record.
* **Contamination is measured, not excluded absolutely.** The scanner compares n-gram windows with our SFT stream, RL prompts and the benchmarks it knows; a public document (arXiv, Wikipedia, court opinion) may well be in other models' pre-training data. The 10% private split exists for leakage checks.
* **Pseudo-references of rare languages are machine output** of unknown accuracy on domain text (the FLORES+ bake-off says nothing about legal Zulu); the metrics above compare a system with the reference only through sentence-aligned beads and string checks.
* **Metrics are checks, not proofs.** A translation can keep every number, symbol and acronym and still reverse the meaning; the modality list exists for exactly that and needs a human.
* OCR-origin text (court opinions, older EUR-Lex acts) contains residual OCR errors (`laparos-copic`, `8,1999`) that the guards do not catch.

## 9. For the coordinator

* `hard_*` was added to `longdoc_select.DEFAULT_EXCLUDE_FILES` in the LAPTOP copy (one line; `tests/test_longdoc_pipeline.py` still passes) and NOT synced to the rental, because the laptop copy of that file is newer than the rental's (it has the stricter `effective_flag`); sync it before copying the `hard_*` files into the pool, otherwise the core selection would read the three new domains. Nothing was copied into `pool/` or `contam/`: the core build is unaffected until the coordinator decides. The staged files are in `longdoc_bench/hard_stage/` (not in `pool/`); copy them in ONE batch with `cp -p` (the scan cache keys on size and mtime) together with `longdoc_bench/contam_hard/cache/hard_*.scan*` (the scan cache of the new files, so the incremental scanner does not rescan them) and `contam_hard/hard_*.flags.jsonl` into `contam/`.
* To rebuild: `scripts/longdoc_contam_hard.sh` then `python scripts/longdoc_build_hard.py --pool-dir pool --stage-dir hard_stage --contam-dir contam_hard --out hard --n 150`.
* Needs from the owner: a domain-expert spot check (a few documents per domain, the flagged modality beads) before any score table of this split is published; a decision on the share-alike / non-commercial documents for the release licence (Wikipedia, WHO, PMC NC).
