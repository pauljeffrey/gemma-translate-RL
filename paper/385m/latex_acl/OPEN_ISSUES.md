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

## 4. Layout and build notes
* Core ends on page 8 in both versions and both builds. Limitations and Ethical considerations pages 9-10; references to page 12; appendix from page 13, one-column.
* Full version: 41 pages (preprint build 42). Trimmed version: 28 pages (both builds), see `latex_acl_trim/OPEN_ISSUES_TRIM.md`.
* Compiled with tectonic (XeTeX) only; pdfLaTeX is not installed here (untested pdfLaTeX branch in the preamble). No overfull boxes, no undefined references or citations in any of the four builds.
* The generic-article version in `paper_385m/latex/` is superseded and was not updated.
