| stage | data used | $\Delta$ macro chrF++ | per billion tokens | per million completions |
|:--|:--|--:|--:|--:|
| SFT, base model to step 20,500 | 2,687 M training tokens | +16.58 | 6.17 | n/a |
| SFT, step 20,500 to 27,638 | 936 M training tokens | +0.44 | 0.47 | n/a |
| RL, all phases to step 1,433 | 864 M generated tokens; 1.76 M completions | +3.72 | 4.31 | 2.11 |
| RL, all phases, clean rows | same | +4.66 | 5.39 | 2.65 |
| RL, phases 1, 2, 2b (dev set, 760 sent.) | 198 M generated tokens; 0.86 M completions | +2.24 | 11.28 | 2.60 |
| RL, phase 2c (dev set, 760 sent.) | 665 M generated tokens; 0.90 M completions | -0.01 | -0.01 | -0.01 |
