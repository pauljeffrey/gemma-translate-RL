| variant | timing | macro chrF++ | sentences/s, batch 1 (min-max of 3) | batch 8 | batch 32 | hours per 1 M sentences (batch 32) | decode tokens/s (batch 32) | first token, batch 1 (ms) | peak RAM batch 1 / 32 (GB, incl. Python for CT2) |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| Q8\_0 (llama.cpp) | solo | 34.96 | 0.97 (0.96-0.98) | 1.85 (1.83-1.86) | 2.00 (1.94-2.02) | 139 | 135 | 80 | 0.4 / 1.2 |
| CTranslate2 int8 | solo | 33.27 | 0.77 (0.75-0.79) | 1.95 (1.90-2.03) | 2.87 (2.83-2.90) | 97 | 208 | 68 | 1.8 / 2.3 |
| Q4\_K\_M (llama.cpp) | shared box | 33.56 | 0.83 (0.78-0.93) | 1.60 (1.47-1.82) | 1.72 (1.27-2.01) | 162 | 117 | 90 | 0.4 / 1.2 |
| CTranslate2 hybrid | shared box | 34.73 | 0.30 (0.28-0.33) | 0.90 (0.83-1.01) | 1.69 (1.53-1.89) | 165 | 120 | 144 | 3.5 / 4.1 |
| CTranslate2 fp32 | shared box | 35.03 | 0.27 (0.27-0.27) | 0.80 (0.80-0.80) | 1.57 (1.57-1.57) | 177 | 108 | 158 | 3.4 / 4.0 |
