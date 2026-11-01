# QE model report

label `label`, features `lite` (10), kind `gbm`, rows 30000, languages 53

train 21479 | test in-domain 5279 | test out-of-domain 3242

## in_domain: within-language Spearman rho 0.459

| keep top X% of each language | precision (chrF++>=40) | recall of good | truly bad (<=15) in kept set | share of all bad leaking | base good | base bad |
|---|---|---|---|---|---|---|
| 10% | 56% | 16% | 7.2% | 7% | 34% | 10.9% |
| 20% | 52% | 30% | 6.7% | 12% | 34% | 10.9% |
| 30% | 50% | 43% | 7.0% | 19% | 34% | 10.9% |
| 40% | 46% | 54% | 7.3% | 27% | 34% | 10.9% |
| 50% | 45% | 65% | 7.5% | 34% | 34% | 10.9% |
| 60% | 42% | 74% | 7.8% | 43% | 34% | 10.9% |
| 70% | 41% | 83% | 8.3% | 54% | 34% | 10.9% |

## out_of_domain: within-language Spearman rho 0.490

| keep top X% of each language | precision (chrF++>=40) | recall of good | truly bad (<=15) in kept set | share of all bad leaking | base good | base bad |
|---|---|---|---|---|---|---|
| 10% | 68% | 15% | 2.1% | 4% | 46% | 5.9% |
| 20% | 64% | 27% | 1.5% | 5% | 46% | 5.9% |
| 30% | 61% | 40% | 1.7% | 9% | 46% | 5.9% |
| 40% | 58% | 50% | 1.9% | 13% | 46% | 5.9% |
| 50% | 55% | 60% | 2.2% | 18% | 46% | 5.9% |
| 60% | 53% | 69% | 2.3% | 24% | 46% | 5.9% |
| 70% | 52% | 78% | 2.9% | 35% | 46% | 5.9% |

