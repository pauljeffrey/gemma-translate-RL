### C3 alpha 0.50 vs A2W alpha 0.7
| language | weak-dev chrF++ A2W0.7 | C3_0.50 | delta | general-dev delta | afs sentence delta | afs paper chrF++ delta |
|---|---|---|---|---|---|---|
| amh | 17.94 | 17.96 | +0.02 | - | -0.01 | +1.78 |
| tir | 13.73 | 14.29 | +0.55 | - | - | - |
| kea | 26.42 | 27.67 | +1.25 | - | - | - |
| kab | 27.87 | 28.21 | +0.33 | - | - | - |
| nus | 19.83 | 20.33 | +0.50 | - | - | - |
| bam | 26.04 | 25.62 | -0.42 | - | - | - |
| kbp | 20.03 | 20.26 | +0.24 | - | - | - |

| group | n languages | mean delta chrF++ [95% CI over languages] |
|---|---|---|
| NLLB-covered teacher languages (weak) | 26 | +0.28 [+0.07, +0.47] |
| NLLB-covered teacher languages (general) | 7 | +0.01 [-0.08, +0.11] |
| NOT NLLB-covered (weak) | 37 | -0.08 [-0.24, +0.06] |
| Ge'ez (amh, tir) (weak) | 2 | +0.29 [+nan, +nan] |
| consensus-only/blend focus (kea kab nus bam kbp) (weak) | 5 | +0.38 [-0.09, +0.90] |
### C3 alpha 0.70 vs A2W alpha 0.7
| language | weak-dev chrF++ A2W0.7 | C3_0.70 | delta | general-dev delta | afs sentence delta | afs paper chrF++ delta |
|---|---|---|---|---|---|---|
| amh | 17.94 | 17.53 | -0.41 | - | +0.01 | +0.30 |
| tir | 13.73 | 14.36 | +0.63 | - | - | - |
| kea | 26.42 | 27.26 | +0.84 | - | - | - |
| kab | 27.87 | 27.25 | -0.63 | - | - | - |
| nus | 19.83 | 18.69 | -1.14 | - | - | - |
| bam | 26.04 | 25.86 | -0.18 | - | - | - |
| kbp | 20.03 | 18.18 | -1.85 | - | - | - |

| group | n languages | mean delta chrF++ [95% CI over languages] |
|---|---|---|
| NLLB-covered teacher languages (weak) | 26 | -0.05 [-0.33, +0.21] |
| NLLB-covered teacher languages (general) | 7 | -0.11 [-0.28, +0.06] |
| NOT NLLB-covered (weak) | 37 | -0.14 [-0.30, -0.00] |
| Ge'ez (amh, tir) (weak) | 2 | +0.11 [+nan, +nan] |
| consensus-only/blend focus (kea kab nus bam kbp) (weak) | 5 | -0.59 [-1.36, +0.24] |
