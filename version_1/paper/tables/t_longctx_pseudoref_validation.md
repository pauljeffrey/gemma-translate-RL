### Table LC-V. Is the NLLB-3.3B pseudo-reference trustworthy? chrF++ of NLLB-3.3B (sentence-by-sentence, beam 4) against the HUMAN references, per language, next to the systems it is used to judge

| language | NLLB-3.3B vs human (mean over SmolDoc bins) | NLLB-3.3B vs human (AfriScience dev+test) | best SabiYarn (live / RL-final) | TranslatePsy-0.8B | NLLB-600M | NLLB-3.3B minus best SabiYarn | pseudo-reference verdict |
|:---|---:|---:|---:|---:|---:|---:|---:|
| amh | 37.7 | 46.0 | 27.9 | 42.4 | 37.9 | +9.8 | usable (>= 2 above SabiYarn) |
| hau | 57.9 | 68.9 | 57.5 | 58.9 | 56.9 | +0.5 | marginal (0-2 above) |
| ibo | 59.2 | - | 57.3 | 58.8 | 57.1 | +1.8 | marginal (0-2 above) |
| kin | 61.5 | - | 52.8 | - | 56.1 | +8.7 | usable (>= 2 above SabiYarn) |
| lug | - | 55.2 | 57.8 | 56.1 | 52.0 | -2.6 | NOT above SabiYarn: do not use as a quality yardstick |
| mlg | 68.4 | - | 61.5 | - | 66.7 | +6.9 | usable (>= 2 above SabiYarn) |
| nso | - | 63.1 | 68.3 | - | 62.2 | -5.2 | NOT above SabiYarn: do not use as a quality yardstick |
| som | 58.2 | - | 57.3 | - | 57.0 | +0.9 | marginal (0-2 above) |
| swh | 66.9 | - | 66.3 | 67.9 | 65.8 | +0.7 | marginal (0-2 above) |
| twi | 45.0 | - | 44.2 | - | 42.0 | +0.8 | marginal (0-2 above) |
| wol | 40.7 | - | 43.9 | - | 34.5 | -3.2 | NOT above SabiYarn: do not use as a quality yardstick |
| xho | 57.8 | - | 57.4 | - | 57.1 | +0.4 | marginal (0-2 above) |
| yor | 45.5 | 47.0 | 46.3 | 46.1 | 44.4 | -0.8 | NOT above SabiYarn: do not use as a quality yardstick |
| zul | 59.0 | 67.9 | 56.3 | 59.0 | 56.7 | +2.7 | usable (>= 2 above SabiYarn) |

All systems in sentence-by-sentence mode; chrF++ against ONE human reference rarely exceeds 60-65, so 'usable' is a relative statement: a pseudo-reference can rank systems that are clearly below it. Rule used in the report: NLLB-3.3B counts as a better-than-ours yardstick for a language only when it beats the best SabiYarn model by >= 2 chrF++ on the human references.
