# paper/385m/

The paper on the 385M English-to-African translator: "Half of the chrF++ Gain Is Finishing the Sentence: What Reinforcement Learning Adds to a 385M English-to-African Translator Beyond a Decoding Guard". Draft of 2026-11-05.

## Released models

* RL model (branch `main` is the mixed best, the final general RL model): https://huggingface.co/AletheiaAING/gemma-0.4B-translate-rl
* SFT model the RL stages start from: https://huggingface.co/AletheiaAING/gemma-0.4B-translate-sft
* Weights of the seed, ablation and science-RL experiments (one branch per run): https://huggingface.co/AletheiaAING/gemma-0.4B-translate-rl-experiments

These links (here and in the LaTeX sources and PDFs of the preprint build) are the only places where the Hugging Face organisation id is written out in the repository; everywhere else it is the placeholder `<HF_ORG>`. The review build of the LaTeX sources prints "anonymised for review" instead of the links.

## Contents

| Path | What |
|---|---|
| `paper.md` | the paper text; every number carries an HTML comment with the report that supports it |
| `CLAIMS_LEDGER.md` | maps every quantitative claim to its source report or data file, with a status (`supported`, `weak`, `pending`) and the caveat that limits the claim |
| `DISCLOSURES.md` | what the paper must disclose: data used, licences, compute, deviations from plan, execution facts of each experiment |
| `PAPER_PLAN.md`, `EXPERIMENTS_LEFT.md` | the plan of the paper and the list of controls (E-01 to E-23) with their status |
| `VERIFY_RESOLVED.md` | answers and sources for the facts that were marked for verification |
| `E14_LATEX_DELTA.md` | the text changes that the E-14 result (science RL without science SFT) needed in the LaTeX sources |
| `make_figures.py`, `figures/` | the figure script (reads the tables in `../../experiments/data/`) and its output (PDF and PNG) |
| `refs_new.bib` | references added for this paper (the cited ones are in `latex_acl/refs.bib`) |
| `latex_acl/` | ACL-format sources, full appendix; `main.pdf` (review build) and `main_preprint.pdf` (preprint build) |
| `latex_acl_trim/` | the same sources with a condensed appendix |

Paths such as `reports/...` and `reports/data/...` inside these files refer to `../../experiments/reports/` and `../../experiments/data/`; `paper_385m/` refers to this folder.

## Building

```
cd latex_acl            # or latex_acl_trim
tectonic main.tex            # review build (anonymous, line numbers)       -> main.pdf
tectonic main_preprint.tex   # preprint build (author block, live links)    -> main_preprint.pdf
```

pdfLaTeX, XeLaTeX and LuaLaTeX also work (on Overleaf set `main.tex` or `main_preprint.tex` as the main document). `acl.sty` and `acl_natbib.bst` are the official ACL style files. Open points of the drafts (release licence, data release, statements required by the venue) are in `OPEN_ISSUES.md` of each LaTeX folder.
