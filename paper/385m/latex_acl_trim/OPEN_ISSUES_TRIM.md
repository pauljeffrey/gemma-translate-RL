# Trimmed ACL version: what was removed or condensed (2026-11-05)

Folder `paper_385m/latex_acl_trim/`; the full 41-page version is in `paper_385m/latex_acl/` (updated for E-14 in the same way). Core (pages 1-8), Limitations, Ethical considerations and references are identical in both (text of pages 1-12 differs only in appendix letters, e.g. "Appendix K" instead of "N"). Core rewritten 2026-11-06 (identical to the full version, ends on page 7). Trimmed total: 28 pages (appendix from page 12; grown for the E-14 tables, the compute table and the branch lists). Build: tectonic, no undefined references, no overfull boxes.

Every core reference points to a kept block: Appendix `app:hyper`, `app:sft`, `app:rlsft`, `app:degenapp`, `app:decoding`, `app:indep`. To put a block back, copy the `\section` of the same label (or the listed paper.md lines) from `../latex_acl/sections/appendix.tex` into `sections/appendix.tex` here.

## Kept (same content as the full version)
E-14 (science RL without the science SFT): the rule paragraph, both result tables (stage scores, differences with intervals) and the registered-rule paragraph (paper.md 397-414) are kept in full.
Hyper-parameters (SFT recipe line 59, RL stage table, stages, reward, features, KL guard, monitors, cost), SFT-vs-A0 and E-13 tables, stage chain table and B1-final table (E-20), finishing and decomposition tables, E-03 details and decision-rule table, loop-flag table and English-output table, E-04/E-11 text and the E-11 table, earlier 256-token table and the full E-02 table, engine-parity text, same-sentence NLLB and AfriqueLLM paragraphs, science alpha table and chunked-document table, full independence table, FLORES+ and NLLB-43 per-language tables, disclosures (full), reproducibility, prompt format and data statement, Responsible NLP checklist.

## Condensed
| block (full version label) | what remains | paper.md lines dropped |
|---|---|---|
| per-language NTREX and MAFAND-MT (`app:perlang`) | one merged table (SFT, RL, difference with CI per benchmark); AfriCOMET columns dropped | Tables 7a, 7b of `tables.md` (AfriCOMET and paired-row columns) |
| per-language FLORES+ | columns iso, SFT, RL, difference [CI], SFT and RL finished/loop | family, weak flag, chrF and AfriCOMET differences, rows kept, paired and loop-cut differences |
| RL stage table | learning rate, KL coefficient, group, guide, steps (optimiser and KL-reference columns dropped; stated in the caption) | columns of lines 71-75 |
| stage chain table | stage, steps, cost, result | columns "start", "prompts x group", "bin" of lines 453-463 |
| stage-A log analysis, family/script/resource, NLLB regression, negative results, contamination follow-ups, extended limitations | one paragraph `app:supp` with a pointer to supplementary material released on acceptance (no numbers) | see below |

## Removed
* Figures: SFT validation curve (fig13), old-270M same-harness figure (fig10), weak-group figure (fig6), RL training curves (fig2), per-language FLORES+ figure (fig1), E-02 appendix copy (fig15), engine parity (fig8, fig11), science and document figures (fig5a, fig5b, fig12, fig9). The core keeps fig3, fig7, fig14, fig16, fig4.
* Prose: RL "GRPO as used" and "Prompts" paragraphs (paper.md 67, 81; covered by the core and the format section); SFT-vs-A0 discussion (149, 151, 153); weak and added languages (291); stage accrual text (275); finishing/decomposition/E-01 prose (187, 199); degeneration bullets (217-220, 231; the tables and the "what the audit does not establish" paragraph 232 stay); stage-A log analysis (423, 425, 429) and E-04/E-11 setup paragraph (433); decoding prose (299, 307, 322); NLLB per-tertile paragraph (346); family/script/resource (358-379); science and document prose (383, 393, 395, 399, 405, 413, 417); contamination (120, 122-130; residual counts remain in Ethical considerations); NLLB regression (470); negative results (474-475); full limitations list (514-529).
* Consequence: the stage-A KL-slope analysis (factor 2.0, interrupted time series), the family/script/resource regressions, the NLLB-share Spearman values and the post-editor numbers are not in the PDF of this version, and the core does not cite them.

## Other changes in this version (also applied to the full version)
Fixed: "at most 3 to 4%" now 3.0% to 4.4% everywhere; the science-RL yaml comment follows Section 15; the E-11 table added to the appendix; a disclosure bullet pointing to a removed limitations item reworded. The remaining open items are in `OPEN_ISSUES.md`.
