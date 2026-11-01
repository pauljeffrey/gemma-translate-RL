# QE model report

label `label`, features `cheap` (43), kind `gbm`, rows 95139, languages 53

train 68278 | test in-domain 16485 | test out-of-domain 10376

## in_domain: within-language Spearman rho 0.338

| keep top X% of each language | precision (chrF++>=40) | recall of good | truly bad (<=15) in kept set | share of all bad leaking | base good | base bad |
|---|---|---|---|---|---|---|
| 10% | 54% | 16% | 6.4% | 6% | 34% | 10.6% |
| 20% | 49% | 29% | 6.9% | 13% | 34% | 10.6% |
| 30% | 46% | 41% | 7.4% | 21% | 34% | 10.6% |
| 40% | 45% | 52% | 7.7% | 29% | 34% | 10.6% |
| 50% | 43% | 62% | 8.0% | 38% | 34% | 10.6% |
| 60% | 41% | 71% | 8.3% | 47% | 34% | 10.6% |
| 70% | 39% | 79% | 8.6% | 57% | 34% | 10.6% |

## out_of_domain: within-language Spearman rho 0.288

| keep top X% of each language | precision (chrF++>=40) | recall of good | truly bad (<=15) in kept set | share of all bad leaking | base good | base bad |
|---|---|---|---|---|---|---|
| 10% | 58% | 12% | 1.4% | 3% | 47% | 5.8% |
| 20% | 54% | 23% | 1.7% | 6% | 47% | 5.8% |
| 30% | 53% | 34% | 1.8% | 9% | 47% | 5.8% |
| 40% | 52% | 44% | 2.0% | 14% | 47% | 5.8% |
| 50% | 51% | 54% | 2.4% | 20% | 47% | 5.8% |
| 60% | 50% | 64% | 2.9% | 31% | 47% | 5.8% |
| 70% | 49% | 73% | 3.4% | 41% | 47% | 5.8% |

