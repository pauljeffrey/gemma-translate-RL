| group | languages | definition | content reward |
|:--|--:|:--|:--|
| QE-validated | 42 | AfriCOMET-QE validated against human references (AUC human vs mismatched >= 0.70) | QE (ensemble) + consensus where weak |
| QE-blend | 12 | named on the AfriCOMET card or a variety of a validated language, no reference-based validation | 0.5 x QE + 0.5 x consensus |
| consensus-only | 23 | not covered by AfriCOMET | consensus chrF++ in the group + constraint penalties |
| SFT-only (no RL) | 8 | SFT targets that were not RL-trained (some without any SFT data) | none |
| total | 85 | 85 language-table rows, of which 77 are RL-trained (first three groups) |  |
