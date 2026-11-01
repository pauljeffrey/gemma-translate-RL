| sentences | metric | n | macro delta | RL wins / ties / loses % | RL preferred % |
|:--|:--|--:|--:|--:|--:|
| all rows | chrF++ (all rows) | 153220 | +3.32 | 50.7 / 20.4 / 28.9 | 63.7 |
| all rows | AfriCOMET, reference-based | 76804 | +0.0591 | 60.4 / 13.8 / 25.8 | 70.0 |
| all rows | SSA-COMET-MTL, reference-based | 76804 | +0.0591 | 60.6 / 14.0 / 25.5 | 70.4 |
| all rows | AfriCOMET-QE (the RL reward model) | 76804 | +0.0692 | 63.5 / 13.0 / 23.6 | 72.9 |
| all rows | SSA-COMET-MTL, reference-free | 76804 | +0.0687 | 61.7 / 12.8 / 25.5 | 70.8 |
| neither model loops / unfinished | chrF++ (all rows) | 140204 | +1.67 | 47.1 / 21.4 / 31.5 | 60.0 |
| neither model loops / unfinished | AfriCOMET, reference-based | 71794 | +0.0387 | 57.7 / 14.7 / 27.6 | 67.7 |
| neither model loops / unfinished | SSA-COMET-MTL, reference-based | 71794 | +0.0478 | 58.4 / 14.8 / 26.8 | 68.6 |
| neither model loops / unfinished | AfriCOMET-QE (the RL reward model) | 71794 | +0.048 | 61.0 / 13.8 / 25.1 | 70.8 |
| neither model loops / unfinished | SSA-COMET-MTL, reference-free | 71794 | +0.0527 | 59.3 / 13.6 / 27.1 | 68.7 |
| neither loops, both length ratio in [0.5, 2] | chrF++ (all rows) | 136993 | +1.54 | 46.9 / 21.4 / 31.7 | 59.6 |
| neither loops, both length ratio in [0.5, 2] | AfriCOMET, reference-based | 70207 | +0.0377 | 57.5 / 14.8 / 27.7 | 67.5 |
| neither loops, both length ratio in [0.5, 2] | SSA-COMET-MTL, reference-based | 70207 | +0.0467 | 58.1 / 14.9 / 27.0 | 68.3 |
| neither loops, both length ratio in [0.5, 2] | AfriCOMET-QE (the RL reward model) | 70207 | +0.0472 | 60.9 / 13.9 / 25.2 | 70.7 |
| neither loops, both length ratio in [0.5, 2] | SSA-COMET-MTL, reference-free | 70207 | +0.0515 | 59.1 / 13.7 / 27.2 | 68.4 |
