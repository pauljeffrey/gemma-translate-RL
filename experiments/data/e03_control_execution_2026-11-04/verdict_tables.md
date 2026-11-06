## Headline (RL minus control, clean rows; chrF++ and AfriCOMET macro over cells; 95% paired bootstrap)

| RL model | cells | control chrF++ | RL chrF++ | RL - control chrF++ | RL - control chrF (plain) | AfriCOMET cells | control AfriCOMET | RL AfriCOMET | RL - control AfriCOMET |
|---|---|---|---|---|---|---|---|---|---|
| mixed best | 93 | 29.97 | 34.55 | +4.58 [+4.45, +4.70] | +4.91 [+4.78, +5.04] | 76 | 0.6101 | 0.6284 | +0.0183 [+0.0172, +0.0192] |
| stage A | 93 | 29.97 | 33.94 | +3.97 [+3.86, +4.08] | +4.22 [+4.10, +4.34] | 76 | 0.6101 | 0.6265 | +0.0164 [+0.0154, +0.0173] |
| B1 best | 93 | 29.97 | 34.33 | +4.36 [+4.24, +4.47] | +4.66 [+4.54, +4.78] | 76 | 0.6101 | 0.6274 | +0.0173 [+0.0163, +0.0182] |

## By benchmark (mixed best minus control)

| benchmark | cells | control | RL | RL - control chrF++ | AfriCOMET control | RL | diff |
|---|---|---|---|---|---|---|---|
| flores_plus | 54 | 25.38 | 31.66 | +6.27 [+6.07, +6.45] | 0.5894 | 0.6267 | +0.0373 [+0.0355, +0.0389] |
| ntrex | 24 | 33.74 | 37.17 | +3.42 [+3.26, +3.59] | 0.6088 | 0.6139 | +0.0051 [+0.0037, +0.0064] |
| mafand | 15 | 40.44 | 40.78 | +0.34 [+0.17, +0.50] | 0.6672 | 0.6531 | -0.0141 [-0.0154, -0.0128] |

## Language panels (the 19 TranslatePsy languages vs the others)

| RL model | panel | cells | control chrF++ | RL chrF++ | RL - control | AfriCOMET diff |
|---|---|---|---|---|---|---|
| mixed best | TP19 | 47 | 41.37 | 41.34 | -0.03 [-0.13, +0.09] | -0.0249 [-0.0257, -0.0241] (n=47) |
| mixed best | OTHERS | 46 | 18.32 | 27.61 | +9.29 [+9.09, +9.49] | +0.0883 [+0.0861, +0.0903] (n=29) |
| stage A | TP19 | 47 | 41.37 | 41.17 | -0.20 [-0.31, -0.08] | -0.0244 [-0.0252, -0.0235] (n=47) |
| stage A | OTHERS | 46 | 18.32 | 26.55 | +8.23 [+8.04, +8.41] | +0.0824 [+0.0803, +0.0843] (n=29) |

## SFT to control and SFT to RL (language-macro chrF++ per benchmark)

| benchmark | SFT | control | control - SFT | mixed | mixed - SFT |
|---|---|---|---|---|---|
| flores_plus | 22.85 | 25.38 | +2.53 [+2.35, +2.71] | 31.66 | +8.81 [+8.59, +9.03] |
| ntrex | 29.25 | 33.74 | +4.49 [+4.29, +4.70] | 37.17 | +7.91 [+7.69, +8.12] |
| mafand | 35.77 | 40.44 | +4.67 [+4.39, +4.92] | 40.78 | +5.01 [+4.76, +5.23] |
| all 93 cells | 26.59 | 29.97 | +3.38 | 34.55 | +7.96 |

Share of the SFT-to-RL (mixed best) chrF++ gap closed by the control: 42.5% (arithmetic of the macros above; cell-count weighted).

## Finished and loop rates (reward-family loop flag; clean rows; language macro)

| system | FLORES+ finished | FLORES+ loop | NTREX finished | NTREX loop | MAFAND finished | MAFAND loop | FLORES+ length ratio |
|---|---|---|---|---|---|---|---|
| SFT | 0.829 | 0.209 | 0.890 | 0.122 | 0.963 | 0.043 | 3.03 |
| control C1 (6,000) | 0.854 | 0.178 | 0.937 | 0.072 | 0.990 | 0.014 | 2.68 |
| stage A | 0.983 | 0.068 | 0.996 | 0.016 | 0.999 | 0.003 | 1.33 |
| B1 best | 0.993 | 0.054 | 0.999 | 0.009 | 1.000 | 0.001 | 1.17 |
| mixed best | 0.995 | 0.048 | 0.999 | 0.006 | 1.000 | 0.001 | 1.14 |

## Pre-registered decision rule applied to mixed best vs C1 at 6,000 steps

| condition | value | met |
|---|---|---|
| lower 95% bound of d (chrF++, 93 cells) > +1.0 | d = +4.58 [+4.45, +4.70] | True |
| lower bound of the AfriCOMET difference > 0 | +0.0183 [+0.0172, +0.0192] | True |
| d > 0 on the 19 TranslatePsy languages | d = -0.03 [-0.13, +0.09] (AfriCOMET -0.0249 [-0.0257, -0.0241]) | False |
| d > 0 on the other languages | d = +9.29 [+9.09, +9.49] | True |

**Verdict by the rule as registered: inconclusive (registered rule: anything else).**
