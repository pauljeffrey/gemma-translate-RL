| step | data / compute used | $\Delta$ chrF++ (all rows) | $\Delta$ chrF++ (clean rows) | per billion tokens (all) | per billion tokens (clean) |
|:--|:--|--:|--:|--:|--:|
| SFT, base model to step 20,500 | 2.69 B training tokens (teacher forcing) | 16.58 | 14.05 | 6.17 | 5.23 |
| SFT, step 20,500 to 27,638 (last 26%) | 0.94 B training tokens | 0.44 | 0.05 | 0.47 | 0.05 |
| RL, whole lineage to step 1,433 | 1.76 M sampled completions, 0.86 B generated tokens | 3.72 | 4.66 | 4.31 | 5.39 |
