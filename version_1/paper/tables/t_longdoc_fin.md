| configuration (finished / loop, % of documents) | b64 | b256 | b512 | b1024 | b2048 | b4096 |
|:--|--:|--:|--:|--:|--:|--:|
| final, whole document, greedy | 98% / 0% | 97% / 1% | 92% / 13% | 65% / 53% | 42% / 73% | 31% / 73% |
| final, whole doc., freq. penalty 0.05 | 99% / 0% | 97% / 2% | 99% / 6% | 96% / 39% | 91% / 60% | 62% / 66% |
| final, whole doc., T 0.7 | 97% / 0% | 99% / 0% | 100% / 0% | 99% / 0% | 95% / 6% | 78% / 15% |
| final, whole doc., best-of-4 (T 0.7) | 98% / 0% | 99% / 0% | 100% / 0% | 100% / 1% | 99% / 5% | 87% / 14% |
| final, 256-token windows | 99% / 0% | 96% / 2% | 93% / 6% | 91% / 9% | 87% / 11% | 86% / 13% |
| final, sentence by sentence | 99% / 0% | 98% / 0% | 94% / 1% | 92% / 2% | 86% / 4% | 82% / 5% |
| final, sentence by sentence + repair (sentfix) | 100% / 0% | 100% / 0% | 99% / 1% | 98% / 2% | 97% / 3% | 94% / 4% |
| SFT, whole document, greedy | 93% / 5% | 74% / 24% | 41% / 50% | 8% / 76% | 2% / 78% | 1% / 72% |
| SFT, sentence by sentence | 93% / 6% | 72% / 23% | 48% / 42% | 39% / 48% | 24% / 55% | 13% / 59% |
