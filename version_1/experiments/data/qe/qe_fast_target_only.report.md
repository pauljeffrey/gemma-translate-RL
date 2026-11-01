# QE model report

label `label`, features `target_only` (18), kind `gbm`, rows 95139, languages 53

train 68278 | test in-domain 16485 | test out-of-domain 10376

## in_domain: within-language Spearman rho 0.287

| keep top X% of each language | precision (chrF++>=40) | recall of good | truly bad (<=15) in kept set | share of all bad leaking | base good | base bad |
|---|---|---|---|---|---|---|
| 10% | 50% | 15% | 6.9% | 7% | 34% | 10.6% |
| 20% | 47% | 28% | 7.4% | 14% | 34% | 10.6% |
| 30% | 45% | 40% | 7.9% | 22% | 34% | 10.6% |
| 40% | 43% | 50% | 8.0% | 30% | 34% | 10.6% |
| 50% | 41% | 60% | 8.3% | 39% | 34% | 10.6% |
| 60% | 40% | 69% | 8.7% | 49% | 34% | 10.6% |
| 70% | 38% | 78% | 8.8% | 58% | 34% | 10.6% |

## out_of_domain: within-language Spearman rho 0.209

| keep top X% of each language | precision (chrF++>=40) | recall of good | truly bad (<=15) in kept set | share of all bad leaking | base good | base bad |
|---|---|---|---|---|---|---|
| 10% | 52% | 11% | 2.6% | 5% | 47% | 5.8% |
| 20% | 50% | 21% | 2.4% | 8% | 47% | 5.8% |
| 30% | 50% | 32% | 2.4% | 12% | 47% | 5.8% |
| 40% | 50% | 42% | 2.6% | 18% | 47% | 5.8% |
| 50% | 49% | 52% | 3.1% | 27% | 47% | 5.8% |
| 60% | 49% | 62% | 3.2% | 33% | 47% | 5.8% |
| 70% | 48% | 72% | 3.6% | 43% | 47% | 5.8% |

