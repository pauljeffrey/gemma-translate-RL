| variant | macro chrF++ | $\Delta$ vs GPU [95% CI] | Yoruba | Amharic | outputs identical to the GPU output (%) |
|:--|--:|--:|--:|--:|--:|
| GPU (vLLM, bf16), reference | 34.98 |  | 29.4 | 19.3 |  |
| CTranslate2 fp32 | 35.03 | +0.03 [-0.12, +0.18] | 29.4 | 19.7 | 34.3 |
| GGUF Q8\_0 (llama.cpp) | 34.96 | -0.02 [-0.22, +0.16] | 29.3 | 19.6 | 18.0 |
| GGUF Q6\_K | 34.76 | -0.20 [-0.42, -0.01] | 29.3 | 19.4 | 9.7 |
| CTranslate2 hybrid (attention fp32, rest int8) | 34.73 | -0.25 [-0.48, -0.06] | 29.6 | 18.6 | 14.4 |
| GGUF Q5\_K\_M | 34.29 | -0.68 [-0.92, -0.48] | 28.7 | 19.1 | 3.3 |
| GGUF Q4\_K\_M | 33.56 | -1.40 [-1.83, -1.07] | 28.2 | 18.0 | 1.3 |
| CTranslate2 int8 | 33.27 | -1.71 [-1.99, -1.38] | 24.6 | 17.8 | 2.4 |
