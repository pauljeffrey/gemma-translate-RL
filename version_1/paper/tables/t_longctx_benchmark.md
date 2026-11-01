### Table LC-B. LongCtx-MT benchmark definition (English -> African languages)

| bin | documents | median Gemma tokens (min-max) | languages | reference |
|:---|---:|---:|---:|---:|
| b256 (control) | 20 | 120 (60-224) | 12 | NLLB-3.3B sentence-by-sentence (pseudo) |
| b512 | 20 | 518 (427-573) | 12 | NLLB-3.3B sentence-by-sentence (pseudo) |
| b1024 | 20 | 1034 (868-1098) | 12 | NLLB-3.3B sentence-by-sentence (pseudo) |
| b2048 | 20 | 2057 (1673-2108) | 12 | NLLB-3.3B sentence-by-sentence (pseudo) |
| b4096 | 10 | 4108 (3706-4197) | 12 | NLLB-3.3B sentence-by-sentence (pseudo) |
| SmolDoc short | 60 | 231 (121-527) | 12 | SmolDoc professional translation (human) |
| stitched ~512 | 38 | 504 (423-679) | 12 | SmolDoc professional translation (human) |
| stitched ~1,024 | 33 | 1066 (879-1127) | 12 | SmolDoc professional translation (human) |
| stitched ~2,048 | 16 | 2001 (1931-2151) | 12 | SmolDoc professional translation (human) |
| AfriScience dev | 25 | 876 (531-1165) | 6 | AfriScience-MT professional translation (human) |
| AfriScience test | 28 | 928 (577-1370) | 6 | AfriScience-MT professional translation (human) |
