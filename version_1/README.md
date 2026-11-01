# version_1

The first set of experiments and the paper written about them: **"Small but Sabi: How Far Can Reinforcement Learning Take a 270M Translator into 77 African Languages?"**
The model that came out of it is called `rl2c` (SFT, then three RL phases, step 1,433).

```
paper/                 the paper: paper.md and sections/, tables/, figures/, latex/ (typeset sources), refs.bib, lit/, claims ledger, scripts that build tables and figures,
                       data/ (run metrics and evaluation tables the paper cites)
experiments/
  reports/             written reports of the experiments (reward design, decoding study, ablations, benchmark comparison, long-document benchmark, weak-language data ...)
  module_reports/      reports written while building the training module (anchors, inter-African data, QE ensemble, final-run integration ...)
  data/                result tables and small data artefacts behind the reports
  figures/             figures of the reports
```

Known limitation of this generation, stated in the paper (Section 6.9a): the fine-tuning text contained the English side of most FLORES+ devtest sentences and the reward calibration
used test splits in places. The second generation (`../experiments`, `../paper`) rebuilds the pipeline without any benchmark text in training.
