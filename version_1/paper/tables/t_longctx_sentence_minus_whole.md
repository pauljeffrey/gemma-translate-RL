### Table LC-D. Sentence-by-sentence wrapper minus whole-document input, same model, same documents (paired document bootstrap)

| model | set | bin | ref | delta chrF++ (sentence - whole) [95 % CI] |
|:---|---:|---:|---:|---:|
| SabiYarn-270M SFT-final | core | b1024 | nllb | +40.3 [+38.1, +42.5] |
| SabiYarn-270M SFT-final | core | b2048 | nllb | +49.4 [+47.7, +50.8] |
| SabiYarn-270M SFT-final | core | b256 (control) | nllb | +2.7 [+0.7, +4.5] |
| SabiYarn-270M SFT-final | core | b4096 | nllb | +55.0 [+53.1, +56.7] |
| SabiYarn-270M SFT-final | core | b512 | nllb | +14.7 [+12.3, +17.1] |
| SabiYarn-270M SFT-final | human | SmolDoc short | human | +3.0 [+1.1, +5.0] |
| SabiYarn-270M SFT-final | human | stitched ~1,024 | human | +34.8 [+33.1, +36.7] |
| SabiYarn-270M SFT-final | human | stitched ~2,048 | human | +40.0 [+38.9, +41.2] |
| SabiYarn-270M SFT-final | human | stitched ~512 | human | +6.5 [+4.2, +8.8] |
| SabiYarn-270M SFT-final | sci | AfriScience dev | human | +33.6 [+31.0, +35.6] |
| SabiYarn-270M SFT-final | sci | AfriScience test | human | +37.0 [+34.3, +39.3] |
| SabiYarn-270M RL-final (2c) | core | b1024 | nllb | +25.2 [+22.5, +28.0] |
| SabiYarn-270M RL-final (2c) | core | b2048 | nllb | +43.5 [+42.1, +44.7] |
| SabiYarn-270M RL-final (2c) | core | b256 (control) | nllb | +0.7 [+0.1, +1.2] |
| SabiYarn-270M RL-final (2c) | core | b4096 | nllb | +52.8 [+51.4, +54.1] |
| SabiYarn-270M RL-final (2c) | core | b512 | nllb | +4.1 [+3.2, +4.9] |
| SabiYarn-270M RL-final (2c) | human | SmolDoc short | human | +0.7 [+0.1, +1.3] |
| SabiYarn-270M RL-final (2c) | human | stitched ~1,024 | human | +26.5 [+24.9, +28.2] |
| SabiYarn-270M RL-final (2c) | human | stitched ~2,048 | human | +37.2 [+36.0, +38.6] |
| SabiYarn-270M RL-final (2c) | human | stitched ~512 | human | +2.7 [+1.8, +4.0] |
| SabiYarn-270M RL-final (2c) | sci | AfriScience dev | human | +19.5 [+16.3, +22.3] |
| SabiYarn-270M RL-final (2c) | sci | AfriScience test | human | +21.5 [+18.6, +24.1] |
| SabiYarn-270M science start (A2W a0.7) | core | b1024 | nllb | +14.4 [+12.9, +16.1] |
| SabiYarn-270M science start (A2W a0.7) | core | b2048 | nllb | +30.0 [+27.9, +32.0] |
| SabiYarn-270M science start (A2W a0.7) | core | b256 (control) | nllb | +0.6 [-0.2, +1.4] |
| SabiYarn-270M science start (A2W a0.7) | core | b4096 | nllb | +46.1 [+44.2, +47.6] |
| SabiYarn-270M science start (A2W a0.7) | core | b512 | nllb | +4.0 [+2.3, +5.4] |
| SabiYarn-270M science start (A2W a0.7) | human | SmolDoc short | human | +0.3 [-0.5, +1.0] |
| SabiYarn-270M science start (A2W a0.7) | human | stitched ~1,024 | human | +13.1 [+12.0, +14.3] |
| SabiYarn-270M science start (A2W a0.7) | human | stitched ~2,048 | human | +26.6 [+24.7, +28.4] |
| SabiYarn-270M science start (A2W a0.7) | human | stitched ~512 | human | +2.1 [+1.2, +3.1] |
| SabiYarn-270M science start (A2W a0.7) | sci | AfriScience dev | human | +12.0 [+10.3, +13.6] |
| SabiYarn-270M science start (A2W a0.7) | sci | AfriScience test | human | +13.8 [+11.9, +15.8] |
| SabiYarn-270M live run (step 250) | core | b1024 | nllb | +14.0 [+12.6, +15.4] |
| SabiYarn-270M live run (step 250) | core | b2048 | nllb | +30.9 [+29.7, +32.3] |
| SabiYarn-270M live run (step 250) | core | b256 (control) | nllb | +0.4 [-0.3, +1.0] |
| SabiYarn-270M live run (step 250) | core | b4096 | nllb | +49.1 [+47.4, +50.9] |
| SabiYarn-270M live run (step 250) | core | b512 | nllb | +3.1 [+2.3, +4.0] |
| SabiYarn-270M live run (step 250) | human | SmolDoc short | human | +0.3 [-0.3, +0.9] |
| SabiYarn-270M live run (step 250) | human | stitched ~1,024 | human | +13.0 [+11.9, +14.3] |
| SabiYarn-270M live run (step 250) | human | stitched ~2,048 | human | +27.4 [+25.9, +28.8] |
| SabiYarn-270M live run (step 250) | human | stitched ~512 | human | +2.1 [+1.5, +2.7] |
| SabiYarn-270M live run (step 250) | sci | AfriScience dev | human | +11.8 [+10.1, +13.7] |
| SabiYarn-270M live run (step 250) | sci | AfriScience test | human | +13.9 [+11.9, +15.8] |
| SabiYarn-270M final run (final2, step 401) | core | b1024 | nllb | +14.5 [+13.2, +15.6] |
| SabiYarn-270M final run (final2, step 401) | core | b2048 | nllb | +30.7 [+29.0, +32.5] |
| SabiYarn-270M final run (final2, step 401) | core | b256 (control) | nllb | +0.4 [-0.4, +1.5] |
| SabiYarn-270M final run (final2, step 401) | core | b4096 | nllb | +49.0 [+47.8, +50.1] |
| SabiYarn-270M final run (final2, step 401) | core | b512 | nllb | +3.2 [+2.3, +4.1] |
| SabiYarn-270M final run (final2, step 401) | human | SmolDoc short | human | +0.2 [-0.3, +0.7] |
| SabiYarn-270M final run (final2, step 401) | human | stitched ~1,024 | human | +13.1 [+11.9, +14.4] |
| SabiYarn-270M final run (final2, step 401) | human | stitched ~2,048 | human | +27.8 [+26.4, +29.1] |
| SabiYarn-270M final run (final2, step 401) | human | stitched ~512 | human | +2.2 [+1.5, +2.9] |
| SabiYarn-270M final run (final2, step 401) | sci | AfriScience dev | human | +12.4 [+10.3, +14.2] |
| SabiYarn-270M final run (final2, step 401) | sci | AfriScience test | human | +14.1 [+12.3, +15.8] |
| NLLB-200-distilled-600M | core | b1024 | nllb | +55.8 [+55.0, +56.6] |
| NLLB-200-distilled-600M | core | b2048 | nllb | +63.5 [+62.5, +64.7] |
| NLLB-200-distilled-600M | core | b256 (control) | nllb | +10.7 [+7.3, +14.6] |
| NLLB-200-distilled-600M | core | b512 | nllb | +44.7 [+42.8, +46.6] |
| NLLB-200-distilled-600M | human | SmolDoc short | human | +15.6 [+12.4, +18.8] |
| NLLB-200-distilled-600M | human | stitched ~1,024 | human | +41.5 [+40.6, +42.4] |
| NLLB-200-distilled-600M | human | stitched ~2,048 | human | +48.1 [+47.4, +48.8] |
| NLLB-200-distilled-600M | human | stitched ~512 | human | +28.6 [+27.1, +30.1] |
| NLLB-200-distilled-600M | sci | AfriScience dev | human | +43.6 [+42.1, +45.0] |
| NLLB-200-distilled-600M | sci | AfriScience test | human | +43.8 [+42.2, +45.1] |
| NLLB-200-3.3B (CT2 int8) | human | SmolDoc short | human | +12.6 [+8.9, +16.6] |
| NLLB-200-3.3B (CT2 int8) | human | stitched ~1,024 | human | +45.4 [+44.5, +46.1] |
| NLLB-200-3.3B (CT2 int8) | human | stitched ~2,048 | human | +51.8 [+51.1, +52.4] |
| NLLB-200-3.3B (CT2 int8) | human | stitched ~512 | human | +29.6 [+27.5, +31.6] |
| NLLB-200-3.3B (CT2 int8) | sci | AfriScience dev | human | +47.3 [+45.9, +48.6] |
| NLLB-200-3.3B (CT2 int8) | sci | AfriScience test | human | +48.5 [+46.8, +50.0] |
| TranslatePsy-AfriSLM-0.8B | core | b1024 | nllb | +58.4 [+56.4, +61.0] |
| TranslatePsy-AfriSLM-0.8B | core | b2048 | nllb | +66.4 [+64.3, +68.1] |
| TranslatePsy-AfriSLM-0.8B | core | b256 (control) | nllb | +15.7 [+9.3, +21.3] |
| TranslatePsy-AfriSLM-0.8B | core | b4096 | nllb | +70.5 [+67.9, +73.4] |
| TranslatePsy-AfriSLM-0.8B | core | b512 | nllb | +48.6 [+45.4, +52.4] |
| TranslatePsy-AfriSLM-0.8B | human | SmolDoc short | human | +17.8 [+13.6, +21.5] |
| TranslatePsy-AfriSLM-0.8B | human | stitched ~1,024 | human | +44.3 [+42.6, +46.0] |
| TranslatePsy-AfriSLM-0.8B | human | stitched ~2,048 | human | +50.6 [+49.0, +52.0] |
| TranslatePsy-AfriSLM-0.8B | human | stitched ~512 | human | +31.3 [+28.1, +34.6] |
| TranslatePsy-AfriSLM-0.8B | sci | AfriScience dev | human | +39.6 [+37.9, +41.5] |
| TranslatePsy-AfriSLM-0.8B | sci | AfriScience test | human | +39.7 [+38.5, +40.9] |
| TranslatePsy-AfriSLM-2B | core | b2048 | nllb | +67.2 [+64.5, +69.8] |
| TranslatePsy-AfriSLM-2B | human | stitched ~1,024 | human | +40.9 [+38.0, +43.7] |
| TranslatePsy-AfriSLM-2B | human | stitched ~2,048 | human | +50.8 [+48.3, +53.4] |
| TranslatePsy-AfriSLM-2B | sci | AfriScience dev | human | +35.6 [+32.6, +37.9] |
