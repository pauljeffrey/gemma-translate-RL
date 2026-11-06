
## Ge'ez script: Amharic and Tigrinya (FLORES+ devtest, 1,012 sentences each)

| language | system | chrF++ | spBLEU | AfriCOMET (held-out) | finished |
|---|---|---:|---:|---:|---:|
| amh | 385M SFT | 16.6 | 4.7 | 0.504 | 0.87 |
| amh | stage A | 25.9 | 14.6 | 0.588 | 0.99 |
| amh | B1 best | 26.7 | 15.4 | 0.592 | 1.00 |
| amh | old 270M SFT (stored, contaminated for FLORES) | 14.5 | 3.3 | - | 0.91 |
| amh | old 270M RL (stored) | 18.9 | 7.3 | - | 1.00 |
| amh | B1 best minus SFT | +10.1 [+9.1, +11.0] | +10.7 | | |
| tir | 385M SFT | 10.5 | 2.3 | 0.507 | 0.77 |
| tir | stage A | 18.3 | 8.4 | 0.641 | 1.00 |
| tir | B1 best | 18.3 | 8.2 | 0.648 | 1.00 |
| tir | old 270M SFT (stored, contaminated for FLORES) | 9.4 | 1.5 | - | 0.82 |
| tir | old 270M RL (stored) | 13.8 | 4.0 | - | 1.00 |
| tir | B1 best minus SFT | +7.8 [+7.2, +8.5] | +5.9 | | |

Old-model AfriCOMET per language was not stored (only an aggregate over 42 languages, see `own_270m_vs_systems_flores_devtest.csv`). NTREX / MAFAND rows: amh mafand b1best_x=18.1, amh ntrex b1best_x=17.8, tir ntrex b1best_x=16.9, amh mafand sft=11.6, amh ntrex sft=10.6, tir ntrex sft=9.1, amh mafand stageA_x=17.9, amh ntrex stageA_x=17.5, tir ntrex stageA_x=16.5
