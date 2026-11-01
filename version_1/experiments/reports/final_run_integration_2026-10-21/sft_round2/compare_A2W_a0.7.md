# Science-SFT stage: before / after (final_1433, alpha_0.70)

### AfriScience-MT dev, sentence level: chrf

| lang | group | final_1433 | alpha_0.70 | d alpha_0.70 |
|---|---|---|---|---|
| amh | science6 | 26.07 | 27.02 | +0.96 |
| hau | science6 | 61.45 | 63.96 | +2.50 |
| lug | science6 | 46.54 | 49.51 | +2.97 |
| nso | science6 | 58.10 | 61.40 | +3.30 |
| yor | science6 | 38.13 | 40.19 | +2.07 |
| zul | science6 | 61.23 | 62.61 | +1.38 |

### AfriScience-MT dev, sentence level: chrf_pp

| lang | group | final_1433 | alpha_0.70 | d alpha_0.70 |
|---|---|---|---|---|
| amh | science6 | 23.63 | 24.68 | +1.05 |
| hau | science6 | 59.26 | 62.02 | +2.76 |
| lug | science6 | 42.55 | 45.36 | +2.80 |
| nso | science6 | 55.98 | 59.53 | +3.55 |
| yor | science6 | 37.03 | 39.15 | +2.12 |
| zul | science6 | 55.45 | 57.06 | +1.61 |

### AfriScience-MT dev, sentence level: bleu

| lang | group | final_1433 | alpha_0.70 | d alpha_0.70 |
|---|---|---|---|---|
| amh | science6 | 5.90 | 6.40 | +0.49 |
| hau | science6 | 35.61 | 39.75 | +4.14 |
| lug | science6 | 16.96 | 18.67 | +1.71 |
| nso | science6 | 32.50 | 37.33 | +4.83 |
| yor | science6 | 15.19 | 17.48 | +2.29 |
| zul | science6 | 22.97 | 25.28 | +2.30 |

### AfriScience-MT dev, sentence level: spbleu

| lang | group | final_1433 | alpha_0.70 | d alpha_0.70 |
|---|---|---|---|---|
| amh | science6 | 13.09 | 14.47 | +1.38 |
| hau | science6 | 37.46 | 41.81 | +4.35 |
| lug | science6 | 20.39 | 24.47 | +4.09 |
| nso | science6 | 35.93 | 40.78 | +4.85 |
| yor | science6 | 20.98 | 23.53 | +2.55 |
| zul | science6 | 33.92 | 36.38 | +2.46 |

### AfriScience-MT dev, document level: chrf

| lang | group | final_1433 | alpha_0.70 | d alpha_0.70 |
|---|---|---|---|---|
| amh | science6 | 23.47 | 25.66 | +2.19 |
| hau | science6 | 49.42 | 63.86 | +14.45 |
| lug | science6 | 33.46 | 43.84 | +10.38 |
| nso | science6 | 39.81 | 56.97 | +17.17 |
| yor | science6 | 22.54 | 35.03 | +12.49 |
| zul | science6 | 47.28 | 61.49 | +14.20 |

### AfriScience-MT dev, document level: chrf_pp

| lang | group | final_1433 | alpha_0.70 | d alpha_0.70 |
|---|---|---|---|---|
| amh | science6 | 20.51 | 22.75 | +2.24 |
| hau | science6 | 46.16 | 61.21 | +15.05 |
| lug | science6 | 27.77 | 38.25 | +10.48 |
| nso | science6 | 36.86 | 54.56 | +17.71 |
| yor | science6 | 20.92 | 33.71 | +12.79 |
| zul | science6 | 40.32 | 54.31 | +13.99 |

### AfriScience-MT dev, document level: bleu

| lang | group | final_1433 | alpha_0.70 | d alpha_0.70 |
|---|---|---|---|---|
| amh | science6 | 2.50 | 3.20 | +0.70 |
| hau | science6 | 13.13 | 33.33 | +20.20 |
| lug | science6 | 1.63 | 6.29 | +4.67 |
| nso | science6 | 11.12 | 26.48 | +15.36 |
| yor | science6 | 2.63 | 9.00 | +6.36 |
| zul | science6 | 6.10 | 18.43 | +12.33 |

### AfriScience-MT dev, document level: finished_rate (%)

| lang | group | final_1433 | alpha_0.70 | d alpha_0.70 |
|---|---|---|---|---|
| amh | science6 | 92.00 | 92.00 | +0.00 |
| hau | science6 | 76.00 | 100.00 | +24.00 |
| lug | science6 | 48.00 | 84.00 | +36.00 |
| nso | science6 | 88.00 | 100.00 | +12.00 |
| yor | science6 | 60.00 | 76.00 | +16.00 |
| zul | science6 | 84.00 | 100.00 | +16.00 |

### General dev (bench_dev_clean, 19 languages): chrF++ (macro over languages; delta CI = bootstrap over languages)

| group | n | final_1433 | alpha_0.70 | delta alpha_0.70 [95% CI] |
|---|---|---|---|---|
| science6 | 3 | 36.06 | 36.32 | +0.26 [+0.02, +0.47] |
| weak-blend | 1 | 21.75 | 21.83 | +0.08 [+nan, +nan] |
| control | 7 | 42.12 | 42.34 | +0.21 [+0.09, +0.37] |
| other | 8 | 44.17 | 44.31 | +0.13 [-0.16, +0.42] |
| ALL | 19 | 40.96 | 41.14 | +0.18 [+0.04, +0.32] |

### General dev (bench_dev_clean, 19 languages): chrF++, per language

| lang | group | final_1433 | alpha_0.70 | d alpha_0.70 |
|---|---|---|---|---|
| ibo | control | 41.68 | 42.29 | +0.60 |
| kin | control | 45.94 | 45.98 | +0.04 |
| sna | control | 36.39 | 36.71 | +0.32 |
| som | control | 39.39 | 39.53 | +0.14 |
| swh | control | 52.13 | 52.32 | +0.19 |
| wol | control | 29.88 | 29.88 | +0.00 |
| xho | control | 49.46 | 49.64 | +0.19 |
| afr | other | 62.50 | 62.23 | -0.27 |
| ewe | other | 32.30 | 33.14 | +0.84 |
| nya | other | 42.31 | 42.24 | -0.07 |
| orm | other | 30.67 | 30.59 | -0.08 |
| pcm | other | 56.57 | 56.04 | -0.53 |
| sot | other | 43.54 | 43.73 | +0.20 |
| tsn | other | 52.26 | 52.81 | +0.55 |
| twi | other | 33.24 | 33.65 | +0.41 |
| hau | science6 | 33.87 | 33.90 | +0.02 |
| yor | science6 | 32.32 | 32.62 | +0.30 |
| zul | science6 | 41.99 | 42.46 | +0.47 |
| fon | weak-blend | 21.75 | 21.83 | +0.08 |

### General dev (bench_dev_clean, 19 languages): finished rate (%) (macro over languages; delta CI = bootstrap over languages)

| group | n | final_1433 | alpha_0.70 | delta alpha_0.70 [95% CI] |
|---|---|---|---|---|
| science6 | 3 | 100.00 | 100.00 | +0.00 [+0.00, +0.00] |
| weak-blend | 1 | 100.00 | 100.00 | +0.00 [+nan, +nan] |
| control | 7 | 99.95 | 99.95 | +0.00 [+0.00, +0.00] |
| other | 8 | 99.95 | 99.98 | +0.02 [+0.00, +0.07] |
| ALL | 19 | 99.96 | 99.97 | +0.01 [+0.00, +0.03] |

### Weak-language dev (FLORES+ dev 60 sentences / pool 60): chrF++ (macro over languages; delta CI = bootstrap over languages)

| group | n | final_1433 | alpha_0.70 | delta alpha_0.70 [95% CI] |
|---|---|---|---|---|
| science6 | 6 | 37.98 | 38.83 | +0.85 [+0.58, +1.10] |
| weak-consensus | 13 | 23.21 | 23.59 | +0.38 [-0.15, +1.09] |
| weak-blend | 19 | 27.20 | 26.83 | -0.36 [-0.58, -0.16] |
| control | 7 | 42.87 | 43.28 | +0.42 [-0.02, +0.92] |
| other | 18 | 38.27 | 38.21 | -0.06 [-0.30, +0.19] |
| ALL | 63 | 32.31 | 32.39 | +0.08 [-0.11, +0.29] |

### Weak-language dev (FLORES+ dev 60 sentences / pool 60): chrF++, per language

| lang | group | final_1433 | alpha_0.70 | d alpha_0.70 |
|---|---|---|---|---|
| ibo | control | 40.55 | 40.97 | +0.42 |
| kin | control | 43.25 | 43.51 | +0.26 |
| sna | control | 41.63 | 41.93 | +0.30 |
| som | control | 41.63 | 42.69 | +1.06 |
| swh | control | 58.42 | 58.00 | -0.42 |
| wol | control | 30.77 | 30.56 | -0.20 |
| xho | control | 43.83 | 45.33 | +1.51 |
| afr | other | 64.19 | 64.08 | -0.11 |
| bem | other | 35.28 | 35.37 | +0.09 |
| ewe | other | 31.75 | 32.67 | +0.92 |
| kik | other | 32.66 | 32.05 | -0.61 |
| lin | other | 43.26 | 42.95 | -0.32 |
| lua | other | 34.44 | 34.75 | +0.30 |
| luo | other | 34.00 | 33.31 | -0.69 |
| mlg | other | 42.55 | 43.17 | +0.62 |
| nya | other | 42.82 | 42.59 | -0.22 |
| orm | other | 31.98 | 31.60 | -0.38 |
| run | other | 32.80 | 33.48 | +0.68 |
| sot | other | 46.18 | 46.24 | +0.05 |
| ssw | other | 39.33 | 39.57 | +0.24 |
| tir | other | 13.60 | 13.73 | +0.14 |
| tsn | other | 44.43 | 44.74 | +0.31 |
| tso | other | 46.21 | 45.43 | -0.78 |
| tum | other | 34.48 | 33.36 | -1.12 |
| twi | other | 38.90 | 38.72 | -0.18 |
| amh | science6 | 17.21 | 17.94 | +0.73 |
| hau | science6 | 48.38 | 49.59 | +1.22 |
| lug | science6 | 35.38 | 35.90 | +0.52 |
| nso | science6 | 48.72 | 49.81 | +1.09 |
| yor | science6 | 29.55 | 30.73 | +1.18 |
| zul | science6 | 48.65 | 49.02 | +0.37 |
| bam | weak-blend | 26.43 | 26.04 | -0.39 |
| dyu | weak-blend | 18.86 | 18.16 | -0.70 |
| efi | weak-blend | 28.99 | 28.63 | -0.36 |
| fon | weak-blend | 22.62 | 22.83 | +0.21 |
| kab | weak-blend | 27.92 | 27.87 | -0.04 |
| kam | weak-blend | 26.01 | 25.11 | -0.89 |
| kbp | weak-blend | 20.90 | 20.03 | -0.87 |
| kmb | weak-blend | 25.22 | 25.15 | -0.07 |
| kon | weak-blend | 16.69 | 16.57 | -0.11 |
| kqn | weak-blend | 34.64 | 33.43 | -1.22 |
| kua | weak-blend | 31.34 | 30.73 | -0.61 |
| loz | weak-blend | 36.92 | 37.05 | +0.13 |
| lun | weak-blend | 28.69 | 27.11 | -1.58 |
| mos | weak-blend | 24.26 | 23.97 | -0.29 |
| nus | weak-blend | 20.05 | 19.83 | -0.21 |
| sag | weak-blend | 36.10 | 36.18 | +0.07 |
| tiv | weak-blend | 28.48 | 28.26 | -0.22 |
| umb | weak-blend | 25.31 | 25.38 | +0.06 |
| ven | weak-blend | 37.31 | 37.47 | +0.16 |
| ach | weak-consensus | 26.93 | 26.33 | -0.60 |
| dag | weak-consensus | 25.83 | 26.19 | +0.36 |
| her | weak-consensus | 24.61 | 26.82 | +2.21 |
| ibb | weak-consensus | 21.57 | 21.43 | -0.14 |
| kea | weak-consensus | 26.15 | 26.42 | +0.27 |
| kri | weak-consensus | 24.63 | 24.97 | +0.34 |
| nbl | weak-consensus | 28.57 | 28.71 | +0.14 |
| ndc | weak-consensus | 21.65 | 21.87 | +0.22 |
| nyn | weak-consensus | 25.81 | 24.98 | -0.83 |
| seh | weak-consensus | 29.12 | 28.92 | -0.20 |
| shi | weak-consensus | 1.05 | 1.07 | +0.01 |
| tsc | weak-consensus | 18.61 | 22.15 | +3.54 |
| vmw | weak-consensus | 27.17 | 26.86 | -0.32 |

### Weak-language dev (FLORES+ dev 60 sentences / pool 60): finished rate (%) (macro over languages; delta CI = bootstrap over languages)

| group | n | final_1433 | alpha_0.70 | delta alpha_0.70 [95% CI] |
|---|---|---|---|---|
| science6 | 6 | 100.00 | 100.00 | +0.00 [+0.00, +0.00] |
| weak-consensus | 13 | 98.08 | 99.87 | +1.79 [-0.13, +4.62] |
| weak-blend | 19 | 99.82 | 100.00 | +0.18 [+0.00, +0.44] |
| control | 7 | 100.00 | 100.00 | +0.00 [+0.00, +0.00] |
| other | 18 | 100.00 | 100.00 | +0.00 [+0.00, +0.00] |
| ALL | 63 | 99.55 | 99.97 | +0.42 [+0.00, +1.03] |
