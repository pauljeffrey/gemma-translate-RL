| system (20 stitched SmolDoc paragraphs, 475-516 prompt tokens) | chrF++ vs human reference | hit the 1,024-token cap | output tokens per paragraph |
|:--|--:|--:|--:|
| CTranslate2 fp32 | 47.6 | 4 / 20 | 830 |
| GGUF Q8\_0 | 47.4 | 5 / 20 | 844 |
| CTranslate2 hybrid | 47.5 | 5 / 20 | 842 |
| GGUF Q4\_K\_M | 46.5 | 4 / 20 | 824 |
| CTranslate2 int8 | 46.1 | 4 / 20 | 842 |
| NLLB-200-distilled-600M, beam 4 (reference) | 23.0 | 0 / 20 | 244 |
| NLLB-200 3.3B, beam 4 (reference) | 24.4 | 0 / 20 | 239 |
