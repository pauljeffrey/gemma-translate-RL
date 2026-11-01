# QE model report

label `label`, features `gen` (50), kind `gbm`, rows 30000, languages 53

train 21479 | test in-domain 5279 | test out-of-domain 3242

## in_domain: within-language Spearman rho 0.482

| keep top X% of each language | precision (chrF++>=40) | recall of good | truly bad (<=15) in kept set | share of all bad leaking | base good | base bad |
|---|---|---|---|---|---|---|
| 10% | 56% | 17% | 6.2% | 6% | 34% | 10.9% |
| 20% | 54% | 31% | 7.0% | 13% | 34% | 10.9% |
| 30% | 50% | 44% | 7.1% | 19% | 34% | 10.9% |
| 40% | 47% | 55% | 7.1% | 26% | 34% | 10.9% |
| 50% | 45% | 65% | 7.3% | 33% | 34% | 10.9% |
| 60% | 43% | 75% | 7.7% | 42% | 34% | 10.9% |
| 70% | 41% | 83% | 8.3% | 53% | 34% | 10.9% |

## out_of_domain: within-language Spearman rho 0.491

| keep top X% of each language | precision (chrF++>=40) | recall of good | truly bad (<=15) in kept set | share of all bad leaking | base good | base bad |
|---|---|---|---|---|---|---|
| 10% | 67% | 15% | 2.1% | 4% | 46% | 5.9% |
| 20% | 64% | 27% | 1.4% | 5% | 46% | 5.9% |
| 30% | 60% | 39% | 1.8% | 9% | 46% | 5.9% |
| 40% | 58% | 50% | 2.0% | 14% | 46% | 5.9% |
| 50% | 55% | 60% | 2.1% | 18% | 46% | 5.9% |
| 60% | 53% | 69% | 2.4% | 25% | 46% | 5.9% |
| 70% | 52% | 78% | 2.8% | 34% | 46% | 5.9% |

