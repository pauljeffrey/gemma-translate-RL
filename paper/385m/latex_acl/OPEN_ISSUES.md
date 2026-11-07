# OPEN ISSUES for the ACL drafts (latex_acl/ full, latex_acl_trim/ trimmed), updated 2026-11-05 after the VERIFY pass

Answers and sources of everything that was closed are in `paper_385m/VERIFY_RESOLVED.md`. Closed and removed from this list: all `[VERIFY: ...]` marks (sacreBLEU, penalty list, licences, compute, instruction sets), the three missing citations (IbomMT, FLORES-MOZ, Glottolog; GlotStoryBook via the GlotLID paper its card asks to cite), the truncated author lists, the E-14 pending mark, B1 final, eight/nine languages, fp32 statements, 3.0-4.4%, the +2.63 source (confirmed with its interval).

## 1. Visible marks that remain in the PDFs

| mark | where | what is needed |
|---|---|---|
| `[OWNER: confirm release licence]` | Ethical considerations, Responsible NLP and reproducibility appendix | the Hub repos currently declare no licence field and have no model card; the paper says exactly that; no licence was chosen |
| `[OWNER: which parts of the data are released]` | Ethical considerations, appendix | release decision for data; the data repositories named in the reports were private at the last check |

No `[VERIFY ...]` and no `[PENDING ...]` mark is left in either build.

## 2. Builds
* `main.pdf` is the review build (anonymous author, line numbers, no author-identifying links; repository names print as "anonymised for review").
* `main_preprint.pdf` is the preprint build (`main_preprint.tex` defines `\preprintbuild` and inputs `main.tex`): author "Jeffrey Otoibhi, Aletheia AI", live links to the three Hugging Face repositories and the branch lists. Camera-ready: use `[final]` instead of `[review]` and `\anonfalse`.

## 3. Still open for the owner
1. **Release licence and model card** (see above); legal decisions on `*_mt560`, `multi-open`, AfroLingu-MT, the Efik sheet and other unspecified-licence sources.
2. **Statement on AI assistants** (ACL checklist) and **personal information / offensive content** in the data (BBC/Wikinews prompts, religious text) and how it was handled.
3. **Total compute and unsized components**: A0, SFT-1, E-01, E-09/E-09b, E-13, E-20, E-05 scoring, per-stage evaluation passes, data-preparation boxes, aborted runs, earlier 270M models have no logged duration; the paper reports sized components (about 94 A100-hours, lower bound) only.
4. **Instruction sets**: translation systems of `african_ultrachat`, `alpaca_*`, `amharic_*`, `kiya_sft` are not stated anywhere; per-source tokens drawn in stage B are not recorded.
5. **sacreBLEU signature strings** were never saved (version 2.6.0 and settings are stated).
6. **Matched-compute SFT-vs-RL literature**: none verified; the sentence is hedged ("searched, found none").
7. **Dual-use sentence** in Ethical considerations and "not for health, legal, safety" are added text, not in paper.md Section 14.6.
8. **Benchmark-exposure framing** (owner rule): no "benchmark exposure" limitation in the ACL versions; residual counts listed once in Ethical considerations; dev-only checkpoint selection and other analysis decisions under "Analysis decisions and process"; the appendix copy of the full limitations list omits items 6 and 7 of paper.md. Please confirm this reading.
9. **Pipeline attribution**: the 95%-of-FLORES+-English exposure statement is about the earlier 270M pipeline, not the clean 385M rerun; the text says so.
10. **Human-evaluation wording**: "no human judgement of system outputs was collected; all scoring is automatic".
11. **NLLB size ratio**: "about 8.6 times the parameters" is nominal (3.3B variant of its model card; the Hub gives no exact count).
12. **Title**: the working title asserts "Half of the chrF++ gain is finishing the sentence"; E-01 shows it is also a property of unguarded greedy decoding. Alternatives in PAPER_PLAN.md section 2.
13. **E-14 caveats** stay visible: one seed, one out-of-memory restart, final checkpoint only, the comparison with the science RL final mixes the science SFT with its alpha 0.5 interpolation; SSA-COMET-QE is reward-family.
14. **Citation details**: GlotStoryBook's own sources (global-asp, nalibali) are not in the bib; "OMahony" in `singh2024aya` is as printed by arXiv.
15. **Figures** are the PDFs of `paper_385m/figures` (another agent's files, read only); no E-14 figure exists. `fig15` is cropped with `trim` in the full appendix.


## 5. Core rewrite of 2026-11-06 (plain-language version)
* Title: "What Does Reinforcement Learning Add to a Small Translator for African Languages?" (13 words). Alternatives in `paper_385m/TITLE_OPTIONS.md`.
* Core structure: Introduction (motivation, RQ1-RQ4, findings at a glance, contributions), Related Work, Approach (overview figure drawn in TikZ), Experimental Setup, Results by research question, Discussion and Conclusion; then Limitations and Ethical considerations. The core ends on page 7 (limit 8) in both versions and both builds; full version 42 pages, trimmed 28 pages.
* Core terms: "SFT baseline", "RL model" (the final general RL model), "SFT-only control", "science variant", "long-document variant". Internal names (stage A, B1, mixed best, E-xx, cells, clean rows) appear only in the appendix and are explained in the new Appendix A (glossary) and in LaTeX comments.
* Moved out of the core (they are in the appendix): the full metric-independence table; the per-benchmark SFT-versus-A0 numbers and the earlier-270M table; the loop-flag audit table and the English-output table (the core keeps one paragraph and one pointer); the NLLB per-language table and tertile details; per-stage chain and B1-final tables; the engine-precision details (core keeps one limitation sentence); full hyper-parameters; the reward's soft-penalty list; the stage-A log analysis (full version only); the extended limitations list.
* Cut from the core text without a replacement (still in `paper.md` and the full appendix where listed above): the three-estimates decomposition table (cut-at-loop is stated in one sentence), the AfriCOMET interval in prose, the seven-setting guard grid values (Figure 2c shows them), the Spearman family analysis, the post-editor negative result, the weak-group RL negative result (one sentence only in the full appendix text), the engine-parity numbers.
* Not made simple: (i) the registered decision rule of the matched-compute control is explained in words (four conditions, three met) but the exact margin and endpoint are in Figure 4 and the full appendix; (ii) the reward-family caveat for SSA-COMET-QE is one sentence in Section 5.5 and one in Section 4; (iii) the beam-search result for the RL model depends on the stopping rule and is stated as "no conclusion"; (iv) the four extra training features are described in one paragraph and a figure; their definitions are in the full appendix text.
* Preprint build only: author block "Jeffrey Otoibhi, Aletheia AI" plus "[Co-authors to be added]", and an "Author contributions" placeholder paragraph (to be filled by the lead author, one sentence per person; authorship by contribution). The review build contains neither.
* New file: `paper_385m/PLAIN_SUMMARY.md` (one-page summary for the owner).

## 6. Full-results comparison integrated (2026-11-06)
* Source: `reports/full_results_comparison_2026-11-06.md` and `paper_385m/latex_snippets_full_results/`. New core subsection 5.1 "How does the model compare with other systems?" with one table (FLORES+ 44 languages, the 19 TranslatePsy languages, NTREX and MAFAND-MT; chrF++ and AfriCOMET; RL minus each system with intervals; languages RL is ahead in) and one figure (`fig_full_results`, caption states that the vertical axes do not start at zero). The old core NLLB table and figure (43 published-table languages) moved to the appendix.
* Appendix (full version): row-sets paragraph, disclosures paragraph, all systems (tabC), paired differences (tabD), TranslatePsy-19 with four metrics (tabA2), earlier 270M models on clean rows (tabA3), SSA-COMET reward-family table (labelled not independent), published-NLLB block, and per-language tables B1-B10 (FLORES+ chrF++, AfriCOMET, chrF, spBLEU; NTREX and MAFAND-MT chrF++ and AfriCOMET; SmolSent; SALT), converted from `table*` to `longtable` so they break across pages (numbers unchanged).
* Appendix (trimmed version): tabC, tabD, SSA table, published-NLLB block, B1-B3 per language, one merged chrF++/AfriCOMET table each for NTREX and MAFAND-MT, one merged SmolSent+SALT table. **Not used in the trimmed version:** tabA2 (its four-metric TranslatePsy block; the core table has chrF++ and AfriCOMET), tabA3 (the 270M rows are in tabC's FLORES+ clean columns), B4 (spBLEU per language; the language averages are quoted in one sentence). The trimmed version also dropped, to stay near 18 appendix pages: the older per-language RL-minus-SFT tables with per-language intervals, the SFT-versus-A0 and earlier-270M detail tables, the stage-chain and B1-final tables, the finishing/decomposition tables, the science-alpha and document tables, the E-14 score table (the E-14 difference table stays), the English-output table, the 256-token decoding table and the loop figure; each is in the full version.
* Sizes (after the 2026-11-06 repair of the per-language tables): core ends on page 8 in both versions; full version 58 pages (appendix 46); trimmed version 32 pages (appendix 20, above the 15-16 target because of the full-results block, about 8 pages).
* Wording changes forced by the report: the 43-language published-table comparison (92%) and the 44-language identical-sentence comparison (91.6%) are both quoted; the report corrects the earlier claim that NLLB hypotheses were not archived (ledger N2/N2b), so the sentence-paired comparison is now in the core. paper.md Section 7.6 (updated by the owner) has the same tables.
* Snippet fixes (layout only): per-language `table*` to `longtable`; `[t]` placement to `[htbp]`; no number was changed. Core Table 1 was written by hand from the report (the snippet tabA has no intervals); every number in it is in tabA, tabA2, tabD or the report section 3.
* Core limitation text now says NTREX/MAFAND-MT NLLB values are language-level averages (no sentence pairing, no AfriCOMET) and that decoding differs (greedy against beam 4).
* Open: SmolSent, SALT, GlotStoryBook, IbomMT and FLORES-MOZ still have no RL row (missing item 8 of the report); NLLB-600M on NTREX/MAFAND-MT/SmolSent/SALT and AfriCOMET for NLLB outside FLORES+ would need about 3 A100-hours (report section 5).

## 4. Layout and build notes
* Core ends on page 8 in both versions and both builds. Limitations and Ethical considerations pages 9-10; references to page 12; appendix from page 13, one-column.
* Full version: 58 pages (both builds). Trimmed version: 32 pages (both builds), see `latex_acl_trim/OPEN_ISSUES_TRIM.md`.
* Compiled with tectonic (XeTeX) only; pdfLaTeX is not installed here (untested pdfLaTeX branch in the preamble). No overfull boxes, no undefined references or citations in any of the four builds.
* The generic-article version in `paper_385m/latex/` is superseded and was not updated.

## 7. Repair note (2026-11-06)
The per-language tables tabB1..B10 were 0 bytes in both trees (a conversion script truncated each file before reading it). They were regenerated from `latex_snippets_full_results/` (table* to longtable, numbers unchanged), the unused B4-B10 stubs were removed from the trimmed tree, and every table row was checked in the PDF text. README has a build check that fails on empty section files or missing inputs.
