# QE model report

label `label`, features `all` (51), kind `gbm`, rows 30000, languages 53

train 21479 | test in-domain 5279 | test out-of-domain 3242

## in_domain: within-language Spearman rho 0.487

| keep top X% of each language | precision (chrF++>=40) | recall of good | truly bad (<=15) in kept set | share of all bad leaking | base good | base bad |
|---|---|---|---|---|---|---|
| 10% | 56% | 17% | 6.4% | 6% | 34% | 10.9% |
| 20% | 54% | 31% | 6.7% | 12% | 34% | 10.9% |
| 30% | 51% | 44% | 6.6% | 18% | 34% | 10.9% |
| 40% | 47% | 55% | 6.9% | 25% | 34% | 10.9% |
| 50% | 45% | 65% | 7.3% | 34% | 34% | 10.9% |
| 60% | 43% | 74% | 7.7% | 42% | 34% | 10.9% |
| 70% | 41% | 83% | 8.1% | 52% | 34% | 10.9% |

## out_of_domain: within-language Spearman rho 0.503

| keep top X% of each language | precision (chrF++>=40) | recall of good | truly bad (<=15) in kept set | share of all bad leaking | base good | base bad |
|---|---|---|---|---|---|---|
| 10% | 68% | 15% | 1.8% | 3% | 46% | 5.9% |
| 20% | 64% | 27% | 1.2% | 4% | 46% | 5.9% |
| 30% | 61% | 39% | 1.2% | 6% | 46% | 5.9% |
| 40% | 57% | 49% | 1.9% | 13% | 46% | 5.9% |
| 50% | 55% | 59% | 2.0% | 17% | 46% | 5.9% |
| 60% | 53% | 69% | 2.4% | 24% | 46% | 5.9% |
| 70% | 51% | 77% | 2.6% | 31% | 46% | 5.9% |

