# Community corrections

The translations in this repository are **machine output of a 270M-parameter model** (`<HF_ORG_LEGACY>/gemma-science-translate`, revision `final2-step401`). Nobody has verified them
yet. This folder is where native and fluent speakers can improve them, so that the long-document set can grow into a human-checked benchmark.

## What can be corrected

* `long_docs/` rows (one document in one language). The `translation_sentences` column holds the same translation split into the source sentences (aligned one-to-one with the
  English sentences of the document), which is the easiest unit to correct.
* `eval_benchmarks/` rows are the model's outputs on public test sets whose human references already exist; they do not need corrections, but an error in a benchmark reference can
  be reported in the same way (say which benchmark, split and `row_id`).

## How to submit

Use **one** of the two routes:

1. **Hugging Face Discussion** on this dataset repository (title `Corrections: <language> <doc_id or batch name>`): attach or paste a CSV (format below).
2. **Pull request** that adds one CSV file to `corrections/submissions/` (create the folder). File name: `<iso>_<yourhandle>_<YYYYMMDD>.csv`. One language per file.

CSV columns (UTF-8, comma separated, header required):

| column | meaning |
|---|---|
| `row_id` | the `row_id` of the row you correct (`<doc_id>\|<iso>\|final2-step401`) |
| `sentence_index` | 0-based index into `translation_sentences`; leave empty if you give a corrected whole document |
| `correction` | your corrected text (the sentence, or the whole document when `sentence_index` is empty) |
| `corrector` | a stable handle or ORCID (a pseudonym is fine; do not use an e-mail address) |
| `correction_notes` | optional: what was wrong (omission, mistranslation, wrong dialect or orthography, hallucinated content, untranslated English, ...) |
| `native_or_fluent` | `native`, `fluent` or `learner` (self-declared, used only for review statistics) |

Do **not** paste text you did not write yourself (for example output of another machine-translation system, of NLLB or of any service whose terms forbid reuse) and do not copy from
copyrighted translations.

## Review rule

A correction becomes `accepted` only when **two independent contributors agree** on it: two different `corrector` handles submit the same corrected sentence, or one submits it and a
second reviewer confirms it in the Discussion (a confirmation counts only if the reviewer did not see the first correction before writing their own, or after they re-translated
the sentence independently). A disagreement stays `disputed` until a third contributor resolves it. Maintainers merge accepted corrections into the `correction`, `corrector`,
`correction_notes` and `correction_status` columns of a new dataset revision (values of `correction_status`: empty = not reviewed, `proposed`, `accepted`, `disputed`, `rejected`).
The original model output is never overwritten: it stays in `translation` / `translation_sentences`.

## Licence of corrections

By submitting a correction you agree that it is released under **the licence of this dataset** (see the licence section of the main `README.md`; as long as the owner has not
chosen one, the proposal is CC BY-SA 4.0 for the open part), and that you have the right to license it. Corrections to the one non-commercial document are licensed CC BY-NC-SA 4.0.
Your handle is credited in the dataset card; tell us in `correction_notes` if you want to stay anonymous.

## Why two contributors

The metrics used to rank systems (AfriCOMET, SSA-COMET) are learned and correlate with human judgement only on the languages where they were validated; for many languages in this
dataset no human check of the model output exists. Agreement of two independent speakers is the cheapest reliable signal we can collect.
