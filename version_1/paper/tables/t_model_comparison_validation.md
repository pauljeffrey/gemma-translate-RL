### Table M4. Metric-stack check: our chrF++ on the whole devtest minus the published chrF++ of the same checkpoint family (per language)

| re-run | published row | languages | mean diff | mean |diff| | largest diff |
| :--- | :--- | ---: | ---: | ---: | ---: |
| NLLB-200-3.3B (re-run: CT2 int8, beam 4) | NLLB-200-3.3B | 43 | -0.17 | 0.40 | +3.02 |

int8 CTranslate2 inference and sentence-level beam 4 explain differences below 1 chrF++; the Yoruba difference comes from NFC normalisation (28 % of Yoruba references are decomposed).
