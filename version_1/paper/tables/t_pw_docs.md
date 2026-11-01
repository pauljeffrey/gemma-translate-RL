| documents (clean, sentence-by-sentence decoding) | languages | documents | chrF++ delta | RL preferred % (chrF++) | finished %: SFT / RL | with a loop %: SFT / RL |
|:--|--:|--:|--:|--:|--:|--:|
| NLLB-ref b64 | 8 | 584 | +3.41 | 63 | 93 / 99 | 8 / 1 |
| NLLB-ref b256 | 8 | 720 | +3.23 | 71 | 72 / 98 | 29 / 3 |
| NLLB-ref b512 | 8 | 800 | +2.26 | 75 | 48 / 94 | 55 / 21 |
| NLLB-ref b1024 | 8 | 712 | +2.31 | 84 | 39 / 92 | 62 / 32 |
| NLLB-ref b2048 | 8 | 504 | +2.07 | 80 | 24 / 86 | 78 / 43 |
| NLLB-ref b4096 | 8 | 280 | +2.33 | 79 | 13 / 82 | 88 / 60 |
| human, short | 6 | 360 | +2.64 | 80 | 84 / 98 | 16 / 2 |
| human, stitched ~512 | 6 | 224 | +2.05 | 79 | 76 / 97 | 24 / 4 |
| human, stitched ~1024 | 4 | 80 | +3.70 | 100 | 54 / 96 | 46 / 4 |
| human, stitched ~2048 | 1 | 5 | +1.24 | 100 | 40 / 100 | 60 / 0 |
