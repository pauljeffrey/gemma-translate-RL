| system (same 4 cores, batch 32) | macro chrF++ (6 languages) | swh hau yor kin mean | Amharic | Tigrinya | sentences/s | hours per 1 M | peak RAM (GB) | output tokens per sentence |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| NLLB-200-distilled-600M, greedy (reference) | 37.35 | 41.8 | 34.7 | 22.1 | 4.17 | 67 | 2.3 | 43.7 |
| NLLB-200-distilled-600M, beam 4 (reference) | 38.72 | 43.1 | 37.4 | 22.6 | 1.49 | 186 | 3.2 | 44.4 |
| NLLB-200 3.3B, greedy (reference) | 39.72 | 45.0 | 34.1 | 24.3 | 1.19 | 233 | 8.2 | 43.0 |
| NLLB-200 3.3B, beam 4 (reference) | 41.03 | 45.9 | 37.6 | 25.1 | 0.48 | 574 | 10.9 | 44.0 |
| ours, GGUF Q8\_0 greedy | 34.96 | 44.2 | 19.6 | 13.2 | 2.00 | 139 | 1.2 | 60.4 |
| ours, CTranslate2 int8 greedy | 33.27 | 42.2 | 17.8 | 12.9 | 2.87 | 97 | 2.3 | 59.1 |
