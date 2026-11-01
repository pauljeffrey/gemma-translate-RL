| configuration (chrF++ vs NLLB-3.3B pseudo-reference, macro of 8 languages) | b64 | b256 | b512 | b1024 | b2048 | b4096 |
|:--|--:|--:|--:|--:|--:|--:|
| final, whole document, greedy | 47.6 | 47.7 | 47.7 | 29.7 | 18.2 | 9.8 |
| final, whole doc., freq. penalty 0.05 | 47.5 | 47.6 | 48.7 | 33.4 | 25.3 | 17.7 |
| final, whole doc., T 0.7 | 46.4 | 46.7 | 47.8 | 36.0 | 26.2 | 14.5 |
| final, whole doc., best-of-4 (T 0.7) | 47.1 | 47.2 | 48.4 | 36.9 | 26.7 | 14.5 |
| final, 256-token windows | 47.4 | 47.6 | 51.8 | 54.0 | 57.0 | 58.6 |
| final, sentence by sentence | 47.6 | 49.0 | 53.5 | 55.6 | 58.4 | 60.1 |
| final, sentence by sentence + repair (sentfix) | 47.6 | 49.0 | 53.5 | 55.6 | 58.4 | 60.1 |
| SFT, whole document, greedy | 44.3 | 39.9 | 32.5 | 17.0 | 11.1 | 6.9 |
| SFT, sentence by sentence | 44.3 | 45.7 | 51.1 | 53.3 | 56.3 | 57.8 |
