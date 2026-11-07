# paper/385m/

The paper on the 385M English-to-African translator. Title of the LaTeX sources (rewritten 2026-11-06 in plain language): "What Does Reinforcement Learning Add to a Small Translator for African Languages?". `paper.md` and the claims ledger still carry the earlier working title ("Half of the chrF++ Gain Is Finishing the Sentence ..."); the alternatives are in `TITLE_OPTIONS.md`. The core ends on page 7 (limit 8) in both versions and both builds.

## Released models

* RL model (branch `main` is the mixed best, the final general RL model): https://huggingface.co/AletheiaAING/gemma-0.4B-translate-rl
* SFT model the RL stages start from: https://huggingface.co/AletheiaAING/gemma-0.4B-translate-sft
* Weights of the seed, ablation and science-RL experiments (one branch per run): https://huggingface.co/AletheiaAING/gemma-0.4B-translate-rl-experiments

These links (here and in the LaTeX sources and PDFs of the preprint build) are the only places where the Hugging Face organisation id is written out in the repository; everywhere else it is the placeholder `<HF_ORG>`. The review build of the LaTeX sources prints "anonymised for review" instead of the links.

## Contents

| Path | What |
|---|---|
| `PLAIN_SUMMARY.md` | plain-language summary of what was done, found and not shown |
| `TITLE_OPTIONS.md` | five title options with the risk of each; option 1 is the default of the LaTeX sources |
| `paper.md` | the long text; every number carries an HTML comment with the report that supports it |
| `CLAIMS_LEDGER.md` | maps every quantitative claim to its source report or data file, with a status (`supported`, `weak`, `pending`) and the caveat that limits the claim |
| `DISCLOSURES.md` | what the paper must disclose: data used, licences, compute, deviations from plan, execution facts of each experiment |
| `PAPER_PLAN.md`, `EXPERIMENTS_LEFT.md` | the plan of the paper and the list of controls (E-01 to E-23) with their status |
| `VERIFY_RESOLVED.md` | answers and sources for the facts that were marked for verification |
| `E14_LATEX_DELTA.md` | the text changes that the E-14 result (science RL without science SFT) needed in the LaTeX sources |
| `make_figures.py`, `figures/` | the figure script (reads the tables in `../../experiments/data/`) and its output (PDF and PNG) |
| `refs_new.bib` | references added for this paper (the cited ones are in `latex_acl/refs.bib`) |
| `latex_acl/` | ACL-format sources of the rewritten paper, full appendix (42 pages) |
| `latex_acl_trim/` | the same core with a condensed appendix (28 pages) |

Paths such as `reports/...` and `reports/data/...` inside these files refer to `../../experiments/reports/` and `../../experiments/data/`; `paper_385m/` refers to this folder.

## Which PDF is which

| File | Build | Pages | Content |
|---|---|---|---|
| `latex_acl/main.pdf` | review (double-blind) | 42 | "Anonymous ACL submission", line numbers, no author, no organisation, no repository links (prints "anonymised for review") |
| `latex_acl/main_preprint.pdf` | preprint | 42 | author block, the three Hugging Face links above |
| `latex_acl_trim/main.pdf` | review (double-blind) | 28 | as above, condensed appendix |
| `latex_acl_trim/main_preprint.pdf` | preprint | 28 | as above, condensed appendix |

Both builds come from the same sources: `main_preprint.tex` only defines `\preprintbuild` and inputs `main.tex`; the switch is `\ifanon` in `main.tex` (`\anonreal{review text}{preprint text}` in `macros.tex`).

## Building

```
cd latex_acl            # or latex_acl_trim
tectonic main.tex            # review build  -> main.pdf
tectonic main_preprint.tex   # preprint build -> main_preprint.pdf
```

pdfLaTeX, XeLaTeX and LuaLaTeX also work (on Overleaf set `main.tex` or `main_preprint.tex` as the main document). `acl.sty` and `acl_natbib.bst` are the official ACL style files. Open points of the drafts (release licence, data release, statements required by the venue, co-authors) are in `OPEN_ISSUES.md` of each LaTeX folder; what the trimmed appendix dropped is in `latex_acl_trim/OPEN_ISSUES_TRIM.md`.
