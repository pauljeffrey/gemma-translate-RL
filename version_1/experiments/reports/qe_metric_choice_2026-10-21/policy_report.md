Policy samples: 28800 (valid for ranking: 28778 = 99.9%), 3600 prompt groups, 45 languages (afr, amh, bam, bem, dyu, ewe, fon, hau, ibo, kab, kam, kbp, kea, kik, kin, kmb, lin, lua, lug, luo, mlg, mos, nso, nus, nya, orm, pcm, run, sag, sna, som, sot, ssw, swh, tir, tsn, tso, tum, twi, umb, vmw, wol, xho, yor, zul).


#### Within-prompt Spearman of each metric with **chrF++ vs reference**, languages: all (45: afr amh bam bem dyu ewe fon hau ibo kab kam kbp kea kik kin kmb lin lua lug luo mlg mos nso nus nya orm pcm run sag sna som sot ssw swh tir tsn tso tum twi umb vmw wol xho yor zul)

| metric | mean over languages [90% CI] | diff vs africomet_qe_stl11 [90% CI] | languages better / worse than africomet_qe_stl11 | selection gain (share of oracle) [90% CI] |
|---|---|---|---|---|
| africomet_qe_stl11 | 0.145 [0.134, 0.156] | - | - | 0.134 [0.114, 0.152] |
| ssa_qe | 0.120 [0.110, 0.132] | -0.025 [-0.031, -0.019] | 10 / 35 | 0.113 [0.094, 0.134] |
| ssa_mtl_qe | 0.164 [0.152, 0.174] | 0.019 [0.009, 0.029] | 27 / 18 | 0.144 [0.124, 0.163] |
| ens_afri_ssaqe | 0.137 [0.126, 0.148] | -0.008 [-0.012, -0.005] | 14 / 31 | 0.140 [0.122, 0.158] |
| ens_afri_ssamtl | 0.170 [0.159, 0.181] | 0.025 [0.018, 0.031] | 36 / 9 | 0.165 [0.145, 0.183] |
| ens_min_afri_ssamtl | 0.164 [0.152, 0.174] | 0.019 [0.012, 0.026] | 36 / 9 | 0.150 [0.129, 0.169] |
| ens3 | 0.158 [0.147, 0.169] | 0.013 [0.007, 0.018] | 30 / 15 | 0.153 [0.133, 0.172] |
| ens_shaped | 0.172 [0.160, 0.182] | 0.026 [0.020, 0.033] | 36 / 9 | 0.164 [0.143, 0.181] |
| consensus | 0.370 [0.360, 0.379] | 0.224 [0.212, 0.236] | 45 / 0 | 0.380 [0.366, 0.392] |

#### Within-prompt Spearman of each metric with **chrF++ vs reference**, languages: mode=qe (32: afr amh bem ewe hau ibo kik kin lin lua lug luo mlg nso nya orm pcm run sna som sot ssw swh tir tsn tso tum twi wol xho yor zul)

| metric | mean over languages [90% CI] | diff vs africomet_qe_stl11 [90% CI] | languages better / worse than africomet_qe_stl11 | selection gain (share of oracle) [90% CI] |
|---|---|---|---|---|
| africomet_qe_stl11 | 0.160 [0.147, 0.173] | - | - | 0.151 [0.130, 0.171] |
| ssa_qe | 0.132 [0.119, 0.145] | -0.028 [-0.035, -0.020] | 6 / 26 | 0.130 [0.109, 0.152] |
| ssa_mtl_qe | 0.184 [0.171, 0.196] | 0.024 [0.012, 0.037] | 21 / 11 | 0.165 [0.144, 0.187] |
| ens_afri_ssaqe | 0.150 [0.138, 0.162] | -0.010 [-0.014, -0.006] | 8 / 24 | 0.148 [0.127, 0.169] |
| ens_afri_ssamtl | 0.187 [0.175, 0.200] | 0.027 [0.020, 0.035] | 26 / 6 | 0.185 [0.163, 0.206] |
| ens_min_afri_ssamtl | 0.182 [0.169, 0.194] | 0.022 [0.014, 0.032] | 26 / 6 | 0.173 [0.150, 0.197] |
| ens3 | 0.174 [0.162, 0.187] | 0.014 [0.007, 0.021] | 22 / 10 | 0.172 [0.152, 0.195] |
| ens_shaped | 0.187 [0.175, 0.199] | 0.027 [0.021, 0.035] | 25 / 7 | 0.183 [0.161, 0.206] |
| consensus | 0.375 [0.363, 0.385] | 0.215 [0.200, 0.230] | 32 / 0 | 0.388 [0.372, 0.403] |

#### Within-prompt Spearman of each metric with **chrF++ vs reference**, languages: mode=blend (11: bam dyu fon kab kam kbp kmb mos nus sag umb)

| metric | mean over languages [90% CI] | diff vs africomet_qe_stl11 [90% CI] | languages better / worse than africomet_qe_stl11 | selection gain (share of oracle) [90% CI] |
|---|---|---|---|---|
| africomet_qe_stl11 | 0.113 [0.092, 0.134] | - | - | 0.089 [0.047, 0.123] |
| ssa_qe | 0.088 [0.068, 0.109] | -0.025 [-0.039, -0.010] | 2 / 9 | 0.057 [0.021, 0.095] |
| ssa_mtl_qe | 0.116 [0.094, 0.135] | 0.003 [-0.020, 0.022] | 4 / 7 | 0.088 [0.046, 0.126] |
| ens_afri_ssaqe | 0.106 [0.085, 0.127] | -0.007 [-0.016, 0.002] | 4 / 7 | 0.114 [0.074, 0.149] |
| ens_afri_ssamtl | 0.130 [0.109, 0.147] | 0.016 [0.003, 0.029] | 8 / 3 | 0.111 [0.069, 0.149] |
| ens_min_afri_ssamtl | 0.124 [0.103, 0.146] | 0.011 [-0.005, 0.027] | 8 / 3 | 0.090 [0.052, 0.126] |
| ens3 | 0.117 [0.096, 0.137] | 0.004 [-0.009, 0.015] | 6 / 5 | 0.100 [0.064, 0.137] |
| ens_shaped | 0.135 [0.113, 0.154] | 0.021 [0.006, 0.036] | 9 / 2 | 0.112 [0.069, 0.147] |
| consensus | 0.356 [0.339, 0.376] | 0.243 [0.217, 0.268] | 11 / 0 | 0.365 [0.336, 0.396] |

#### Within-prompt Spearman of each metric with **chrF++ vs reference**, languages: mode=consensus (2: kea vmw)

| metric | mean over languages [90% CI] | diff vs africomet_qe_stl11 [90% CI] | languages better / worse than africomet_qe_stl11 | selection gain (share of oracle) [90% CI] |
|---|---|---|---|---|
| africomet_qe_stl11 | 0.088 [0.033, 0.141] | - | - | 0.112 [0.033, 0.204] |
| ssa_qe | 0.109 [0.056, 0.162] | 0.020 [-0.007, 0.048] | 2 / 0 | 0.158 [0.068, 0.248] |
| ssa_mtl_qe | 0.116 [0.066, 0.163] | 0.027 [-0.026, 0.073] | 2 / 0 | 0.113 [0.032, 0.205] |
| ens_afri_ssaqe | 0.102 [0.048, 0.153] | 0.014 [-0.004, 0.031] | 2 / 0 | 0.156 [0.066, 0.241] |
| ens_afri_ssamtl | 0.118 [0.064, 0.174] | 0.030 [-0.004, 0.057] | 2 / 0 | 0.136 [0.050, 0.227] |
| ens_min_afri_ssamtl | 0.094 [0.040, 0.146] | 0.006 [0.002, 0.011] | 2 / 0 | 0.108 [0.027, 0.202] |
| ens3 | 0.129 [0.076, 0.187] | 0.041 [0.017, 0.064] | 2 / 0 | 0.131 [0.043, 0.224] |
| ens_shaped | 0.123 [0.069, 0.176] | 0.035 [0.004, 0.062] | 2 / 0 | 0.149 [0.060, 0.242] |
| consensus | 0.366 [0.314, 0.418] | 0.278 [0.206, 0.343] | 2 / 0 | 0.325 [0.252, 0.390] |

#### Within-prompt Spearman of each metric with **AfriCOMET-STL-1.1 with reference**, languages: all (45: afr amh bam bem dyu ewe fon hau ibo kab kam kbp kea kik kin kmb lin lua lug luo mlg mos nso nus nya orm pcm run sag sna som sot ssw swh tir tsn tso tum twi umb vmw wol xho yor zul)

| metric | mean over languages [90% CI] | diff vs africomet_qe_stl11 [90% CI] | languages better / worse than africomet_qe_stl11 | selection gain (share of oracle) [90% CI] |
|---|---|---|---|---|
| africomet_qe_stl11 | 0.696 [0.690, 0.703] | - | - | 0.716 [0.704, 0.727] |
| ssa_qe | 0.650 [0.643, 0.657] | -0.046 [-0.052, -0.040] | 6 / 39 | 0.686 [0.673, 0.698] |
| ssa_mtl_qe | 0.492 [0.483, 0.501] | -0.204 [-0.213, -0.196] | 1 / 44 | 0.521 [0.506, 0.537] |
| ens_afri_ssaqe | 0.693 [0.686, 0.700] | -0.003 [-0.006, 0.000] | 20 / 25 | 0.724 [0.712, 0.735] |
| ens_afri_ssamtl | 0.654 [0.646, 0.660] | -0.043 [-0.048, -0.038] | 5 / 40 | 0.676 [0.664, 0.688] |
| ens_min_afri_ssamtl | 0.605 [0.597, 0.613] | -0.092 [-0.098, -0.086] | 3 / 42 | 0.650 [0.637, 0.663] |
| ens3 | 0.682 [0.674, 0.689] | -0.015 [-0.019, -0.011] | 13 / 32 | 0.709 [0.697, 0.721] |
| ens_shaped | 0.658 [0.650, 0.664] | -0.038 [-0.044, -0.033] | 6 / 39 | 0.681 [0.669, 0.694] |
| consensus | 0.212 [0.201, 0.223] | -0.484 [-0.497, -0.472] | 0 / 45 | 0.243 [0.223, 0.260] |

#### Within-prompt Spearman of each metric with **AfriCOMET-STL-1.1 with reference**, languages: mode=qe (32: afr amh bem ewe hau ibo kik kin lin lua lug luo mlg nso nya orm pcm run sna som sot ssw swh tir tsn tso tum twi wol xho yor zul)

| metric | mean over languages [90% CI] | diff vs africomet_qe_stl11 [90% CI] | languages better / worse than africomet_qe_stl11 | selection gain (share of oracle) [90% CI] |
|---|---|---|---|---|
| africomet_qe_stl11 | 0.713 [0.706, 0.721] | - | - | 0.737 [0.723, 0.751] |
| ssa_qe | 0.669 [0.660, 0.677] | -0.044 [-0.051, -0.038] | 4 / 28 | 0.705 [0.690, 0.719] |
| ssa_mtl_qe | 0.526 [0.515, 0.535] | -0.188 [-0.198, -0.179] | 1 / 31 | 0.553 [0.535, 0.572] |
| ens_afri_ssaqe | 0.710 [0.702, 0.718] | -0.003 [-0.007, 0.001] | 12 / 20 | 0.742 [0.728, 0.754] |
| ens_afri_ssamtl | 0.674 [0.666, 0.683] | -0.039 [-0.045, -0.034] | 5 / 27 | 0.702 [0.687, 0.715] |
| ens_min_afri_ssamtl | 0.626 [0.617, 0.636] | -0.088 [-0.094, -0.081] | 1 / 31 | 0.669 [0.655, 0.685] |
| ens3 | 0.700 [0.692, 0.709] | -0.013 [-0.019, -0.008] | 9 / 23 | 0.730 [0.717, 0.743] |
| ens_shaped | 0.687 [0.678, 0.695] | -0.027 [-0.032, -0.020] | 5 / 27 | 0.714 [0.699, 0.728] |
| consensus | 0.249 [0.236, 0.262] | -0.464 [-0.477, -0.451] | 0 / 32 | 0.296 [0.275, 0.314] |

#### Within-prompt Spearman of each metric with **AfriCOMET-STL-1.1 with reference**, languages: mode=blend (11: bam dyu fon kab kam kbp kmb mos nus sag umb)

| metric | mean over languages [90% CI] | diff vs africomet_qe_stl11 [90% CI] | languages better / worse than africomet_qe_stl11 | selection gain (share of oracle) [90% CI] |
|---|---|---|---|---|
| africomet_qe_stl11 | 0.667 [0.653, 0.682] | - | - | 0.674 [0.651, 0.699] |
| ssa_qe | 0.609 [0.595, 0.626] | -0.058 [-0.070, -0.045] | 2 / 9 | 0.646 [0.620, 0.671] |
| ssa_mtl_qe | 0.420 [0.401, 0.436] | -0.248 [-0.264, -0.232] | 0 / 11 | 0.461 [0.429, 0.493] |
| ens_afri_ssaqe | 0.662 [0.649, 0.677] | -0.005 [-0.013, 0.003] | 6 / 5 | 0.694 [0.670, 0.717] |
| ens_afri_ssamtl | 0.613 [0.598, 0.628] | -0.054 [-0.065, -0.044] | 0 / 11 | 0.633 [0.607, 0.658] |
| ens_min_afri_ssamtl | 0.548 [0.531, 0.563] | -0.120 [-0.133, -0.108] | 0 / 11 | 0.602 [0.578, 0.626] |
| ens3 | 0.646 [0.632, 0.660] | -0.021 [-0.031, -0.011] | 3 / 8 | 0.674 [0.648, 0.698] |
| ens_shaped | 0.598 [0.581, 0.613] | -0.070 [-0.083, -0.058] | 1 / 10 | 0.618 [0.592, 0.642] |
| consensus | 0.127 [0.105, 0.149] | -0.540 [-0.566, -0.516] | 0 / 11 | 0.117 [0.075, 0.156] |

#### Within-prompt Spearman of each metric with **AfriCOMET-STL-1.1 with reference**, languages: mode=consensus (2: kea vmw)

| metric | mean over languages [90% CI] | diff vs africomet_qe_stl11 [90% CI] | languages better / worse than africomet_qe_stl11 | selection gain (share of oracle) [90% CI] |
|---|---|---|---|---|
| africomet_qe_stl11 | 0.584 [0.544, 0.618] | - | - | 0.616 [0.548, 0.678] |
| ssa_qe | 0.575 [0.535, 0.615] | -0.008 [-0.034, 0.019] | 0 / 2 | 0.603 [0.532, 0.669] |
| ssa_mtl_qe | 0.358 [0.309, 0.397] | -0.225 [-0.280, -0.178] | 0 / 2 | 0.338 [0.248, 0.418] |
| ens_afri_ssaqe | 0.594 [0.553, 0.629] | 0.010 [-0.005, 0.027] | 2 / 0 | 0.604 [0.529, 0.669] |
| ens_afri_ssamtl | 0.540 [0.499, 0.574] | -0.044 [-0.073, -0.017] | 0 / 2 | 0.509 [0.429, 0.578] |
| ens_min_afri_ssamtl | 0.584 [0.544, 0.620] | 0.001 [-0.004, 0.005] | 2 / 0 | 0.605 [0.533, 0.672] |
| ens3 | 0.579 [0.539, 0.615] | -0.005 [-0.027, 0.016] | 1 / 1 | 0.560 [0.488, 0.627] |
| ens_shaped | 0.534 [0.496, 0.568] | -0.049 [-0.083, -0.018] | 0 / 2 | 0.515 [0.436, 0.584] |
| consensus | 0.088 [0.038, 0.137] | -0.496 [-0.553, -0.440] | 0 / 2 | 0.086 [0.018, 0.175] |

#### Within-prompt Spearman of each metric with **SSA-COMET-MTL with reference**, languages: all (45: afr amh bam bem dyu ewe fon hau ibo kab kam kbp kea kik kin kmb lin lua lug luo mlg mos nso nus nya orm pcm run sag sna som sot ssw swh tir tsn tso tum twi umb vmw wol xho yor zul)

| metric | mean over languages [90% CI] | diff vs africomet_qe_stl11 [90% CI] | languages better / worse than africomet_qe_stl11 | selection gain (share of oracle) [90% CI] |
|---|---|---|---|---|
| africomet_qe_stl11 | 0.537 [0.529, 0.545] | - | - | 0.567 [0.554, 0.582] |
| ssa_qe | 0.566 [0.557, 0.574] | 0.029 [0.023, 0.035] | 32 / 13 | 0.595 [0.580, 0.610] |
| ssa_mtl_qe | 0.873 [0.869, 0.876] | 0.335 [0.328, 0.344] | 45 / 0 | 0.899 [0.893, 0.905] |
| ens_afri_ssaqe | 0.568 [0.560, 0.576] | 0.031 [0.027, 0.035] | 39 / 6 | 0.599 [0.585, 0.613] |
| ens_afri_ssamtl | 0.794 [0.789, 0.799] | 0.257 [0.251, 0.263] | 45 / 0 | 0.843 [0.834, 0.851] |
| ens_min_afri_ssamtl | 0.739 [0.732, 0.745] | 0.202 [0.195, 0.208] | 45 / 0 | 0.773 [0.763, 0.784] |
| ens3 | 0.734 [0.728, 0.740] | 0.197 [0.192, 0.202] | 45 / 0 | 0.783 [0.773, 0.793] |
| ens_shaped | 0.774 [0.768, 0.779] | 0.237 [0.231, 0.243] | 45 / 0 | 0.824 [0.815, 0.832] |
| consensus | 0.246 [0.237, 0.257] | -0.291 [-0.303, -0.277] | 1 / 44 | 0.261 [0.243, 0.276] |

#### Within-prompt Spearman of each metric with **SSA-COMET-MTL with reference**, languages: mode=qe (32: afr amh bem ewe hau ibo kik kin lin lua lug luo mlg nso nya orm pcm run sna som sot ssw swh tir tsn tso tum twi wol xho yor zul)

| metric | mean over languages [90% CI] | diff vs africomet_qe_stl11 [90% CI] | languages better / worse than africomet_qe_stl11 | selection gain (share of oracle) [90% CI] |
|---|---|---|---|---|
| africomet_qe_stl11 | 0.562 [0.552, 0.571] | - | - | 0.584 [0.566, 0.601] |
| ssa_qe | 0.581 [0.571, 0.591] | 0.020 [0.014, 0.026] | 19 / 13 | 0.597 [0.579, 0.614] |
| ssa_mtl_qe | 0.874 [0.870, 0.878] | 0.312 [0.302, 0.321] | 32 / 0 | 0.900 [0.894, 0.906] |
| ens_afri_ssaqe | 0.586 [0.576, 0.595] | 0.024 [0.020, 0.028] | 26 / 6 | 0.601 [0.584, 0.619] |
| ens_afri_ssamtl | 0.804 [0.798, 0.810] | 0.242 [0.235, 0.249] | 32 / 0 | 0.846 [0.836, 0.856] |
| ens_min_afri_ssamtl | 0.761 [0.753, 0.767] | 0.199 [0.191, 0.207] | 32 / 0 | 0.787 [0.774, 0.799] |
| ens3 | 0.746 [0.739, 0.753] | 0.184 [0.178, 0.190] | 32 / 0 | 0.789 [0.778, 0.803] |
| ens_shaped | 0.776 [0.770, 0.783] | 0.214 [0.208, 0.221] | 32 / 0 | 0.822 [0.812, 0.833] |
| consensus | 0.279 [0.268, 0.292] | -0.282 [-0.296, -0.266] | 1 / 31 | 0.302 [0.281, 0.320] |

#### Within-prompt Spearman of each metric with **SSA-COMET-MTL with reference**, languages: mode=blend (11: bam dyu fon kab kam kbp kmb mos nus sag umb)

| metric | mean over languages [90% CI] | diff vs africomet_qe_stl11 [90% CI] | languages better / worse than africomet_qe_stl11 | selection gain (share of oracle) [90% CI] |
|---|---|---|---|---|
| africomet_qe_stl11 | 0.484 [0.467, 0.499] | - | - | 0.536 [0.507, 0.561] |
| ssa_qe | 0.538 [0.522, 0.552] | 0.054 [0.041, 0.067] | 11 / 0 | 0.603 [0.573, 0.631] |
| ssa_mtl_qe | 0.875 [0.869, 0.881] | 0.391 [0.375, 0.409] | 11 / 0 | 0.901 [0.888, 0.912] |
| ens_afri_ssaqe | 0.535 [0.519, 0.549] | 0.051 [0.042, 0.059] | 11 / 0 | 0.612 [0.584, 0.638] |
| ens_afri_ssamtl | 0.780 [0.770, 0.789] | 0.296 [0.284, 0.309] | 11 / 0 | 0.842 [0.824, 0.857] |
| ens_min_afri_ssamtl | 0.727 [0.713, 0.739] | 0.243 [0.229, 0.257] | 11 / 0 | 0.782 [0.762, 0.801] |
| ens3 | 0.715 [0.703, 0.726] | 0.231 [0.220, 0.243] | 11 / 0 | 0.771 [0.751, 0.791] |
| ens_shaped | 0.779 [0.769, 0.788] | 0.295 [0.282, 0.309] | 11 / 0 | 0.832 [0.814, 0.847] |
| consensus | 0.158 [0.136, 0.178] | -0.326 [-0.350, -0.298] | 0 / 11 | 0.146 [0.109, 0.180] |

#### Within-prompt Spearman of each metric with **SSA-COMET-MTL with reference**, languages: mode=consensus (2: kea vmw)

| metric | mean over languages [90% CI] | diff vs africomet_qe_stl11 [90% CI] | languages better / worse than africomet_qe_stl11 | selection gain (share of oracle) [90% CI] |
|---|---|---|---|---|
| africomet_qe_stl11 | 0.436 [0.388, 0.474] | - | - | 0.481 [0.399, 0.551] |
| ssa_qe | 0.470 [0.429, 0.507] | 0.034 [0.009, 0.059] | 2 / 0 | 0.522 [0.444, 0.588] |
| ssa_mtl_qe | 0.839 [0.818, 0.856] | 0.403 [0.365, 0.448] | 2 / 0 | 0.866 [0.832, 0.898] |
| ens_afri_ssaqe | 0.468 [0.425, 0.503] | 0.032 [0.014, 0.050] | 2 / 0 | 0.498 [0.413, 0.571] |
| ens_afri_ssamtl | 0.720 [0.693, 0.741] | 0.284 [0.254, 0.317] | 2 / 0 | 0.792 [0.750, 0.831] |
| ens_min_afri_ssamtl | 0.453 [0.408, 0.494] | 0.017 [0.009, 0.026] | 2 / 0 | 0.513 [0.433, 0.580] |
| ens3 | 0.650 [0.617, 0.676] | 0.214 [0.192, 0.240] | 2 / 0 | 0.743 [0.690, 0.790] |
| ens_shaped | 0.718 [0.688, 0.743] | 0.282 [0.253, 0.315] | 2 / 0 | 0.799 [0.756, 0.836] |
| consensus | 0.203 [0.159, 0.252] | -0.233 [-0.291, -0.156] | 0 / 2 | 0.247 [0.165, 0.329] |

#### Within-prompt Spearman of each metric with **mean z of the two reference-based COMETs**, languages: all (45: afr amh bam bem dyu ewe fon hau ibo kab kam kbp kea kik kin kmb lin lua lug luo mlg mos nso nus nya orm pcm run sag sna som sot ssw swh tir tsn tso tum twi umb vmw wol xho yor zul)

| metric | mean over languages [90% CI] | diff vs africomet_qe_stl11 [90% CI] | languages better / worse than africomet_qe_stl11 | selection gain (share of oracle) [90% CI] |
|---|---|---|---|---|
| africomet_qe_stl11 | 0.686 [0.679, 0.692] | - | - | 0.705 [0.694, 0.717] |
| ssa_qe | 0.676 [0.669, 0.682] | -0.010 [-0.015, -0.004] | 18 / 27 | 0.705 [0.692, 0.717] |
| ssa_mtl_qe | 0.753 [0.747, 0.758] | 0.067 [0.060, 0.075] | 37 / 8 | 0.780 [0.768, 0.790] |
| ens_afri_ssaqe | 0.701 [0.695, 0.707] | 0.016 [0.012, 0.019] | 33 / 12 | 0.728 [0.717, 0.740] |
| ens_afri_ssamtl | 0.819 [0.815, 0.824] | 0.134 [0.129, 0.138] | 45 / 0 | 0.842 [0.834, 0.851] |
| ens_min_afri_ssamtl | 0.750 [0.744, 0.756] | 0.064 [0.059, 0.069] | 44 / 1 | 0.783 [0.773, 0.795] |
| ens3 | 0.800 [0.794, 0.805] | 0.114 [0.110, 0.119] | 45 / 0 | 0.826 [0.817, 0.836] |
| ens_shaped | 0.807 [0.802, 0.812] | 0.122 [0.117, 0.127] | 45 / 0 | 0.834 [0.825, 0.843] |
| consensus | 0.254 [0.243, 0.264] | -0.432 [-0.444, -0.420] | 0 / 45 | 0.274 [0.257, 0.290] |

#### Within-prompt Spearman of each metric with **mean z of the two reference-based COMETs**, languages: mode=qe (32: afr amh bem ewe hau ibo kik kin lin lua lug luo mlg nso nya orm pcm run sna som sot ssw swh tir tsn tso tum twi wol xho yor zul)

| metric | mean over languages [90% CI] | diff vs africomet_qe_stl11 [90% CI] | languages better / worse than africomet_qe_stl11 | selection gain (share of oracle) [90% CI] |
|---|---|---|---|---|
| africomet_qe_stl11 | 0.698 [0.690, 0.705] | - | - | 0.717 [0.703, 0.731] |
| ssa_qe | 0.686 [0.678, 0.694] | -0.012 [-0.018, -0.006] | 11 / 21 | 0.707 [0.693, 0.721] |
| ssa_mtl_qe | 0.763 [0.756, 0.769] | 0.065 [0.056, 0.074] | 26 / 6 | 0.789 [0.777, 0.801] |
| ens_afri_ssaqe | 0.711 [0.703, 0.718] | 0.013 [0.009, 0.017] | 22 / 10 | 0.729 [0.715, 0.744] |
| ens_afri_ssamtl | 0.824 [0.819, 0.829] | 0.126 [0.121, 0.132] | 32 / 0 | 0.847 [0.837, 0.858] |
| ens_min_afri_ssamtl | 0.764 [0.757, 0.770] | 0.066 [0.060, 0.072] | 32 / 0 | 0.791 [0.779, 0.804] |
| ens3 | 0.805 [0.799, 0.811] | 0.108 [0.102, 0.113] | 32 / 0 | 0.831 [0.820, 0.843] |
| ens_shaped | 0.814 [0.808, 0.820] | 0.116 [0.111, 0.122] | 32 / 0 | 0.842 [0.832, 0.852] |
| consensus | 0.287 [0.275, 0.299] | -0.410 [-0.424, -0.397] | 0 / 32 | 0.322 [0.302, 0.339] |

#### Within-prompt Spearman of each metric with **mean z of the two reference-based COMETs**, languages: mode=blend (11: bam dyu fon kab kam kbp kmb mos nus sag umb)

| metric | mean over languages [90% CI] | diff vs africomet_qe_stl11 [90% CI] | languages better / worse than africomet_qe_stl11 | selection gain (share of oracle) [90% CI] |
|---|---|---|---|---|
| africomet_qe_stl11 | 0.664 [0.651, 0.676] | - | - | 0.682 [0.656, 0.706] |
| ssa_qe | 0.655 [0.640, 0.669] | -0.009 [-0.022, 0.002] | 5 / 6 | 0.703 [0.679, 0.727] |
| ssa_mtl_qe | 0.737 [0.726, 0.747] | 0.073 [0.058, 0.088] | 9 / 2 | 0.773 [0.751, 0.792] |
| ens_afri_ssaqe | 0.685 [0.673, 0.697] | 0.021 [0.013, 0.029] | 9 / 2 | 0.740 [0.717, 0.760] |
| ens_afri_ssamtl | 0.819 [0.809, 0.827] | 0.155 [0.144, 0.166] | 11 / 0 | 0.846 [0.830, 0.860] |
| ens_min_afri_ssamtl | 0.732 [0.720, 0.744] | 0.068 [0.057, 0.079] | 10 / 1 | 0.783 [0.766, 0.801] |
| ens3 | 0.796 [0.786, 0.805] | 0.132 [0.121, 0.142] | 11 / 0 | 0.824 [0.807, 0.841] |
| ens_shaped | 0.802 [0.792, 0.810] | 0.138 [0.127, 0.149] | 11 / 0 | 0.827 [0.811, 0.841] |
| consensus | 0.173 [0.151, 0.195] | -0.491 [-0.516, -0.465] | 0 / 11 | 0.154 [0.115, 0.189] |

#### Within-prompt Spearman of each metric with **mean z of the two reference-based COMETs**, languages: mode=consensus (2: kea vmw)

| metric | mean over languages [90% CI] | diff vs africomet_qe_stl11 [90% CI] | languages better / worse than africomet_qe_stl11 | selection gain (share of oracle) [90% CI] |
|---|---|---|---|---|
| africomet_qe_stl11 | 0.614 [0.577, 0.647] | - | - | 0.645 [0.577, 0.702] |
| ssa_qe | 0.628 [0.590, 0.661] | 0.014 [-0.008, 0.038] | 2 / 0 | 0.674 [0.608, 0.733] |
| ssa_mtl_qe | 0.680 [0.645, 0.706] | 0.066 [0.024, 0.108] | 2 / 0 | 0.669 [0.609, 0.720] |
| ens_afri_ssaqe | 0.639 [0.600, 0.671] | 0.026 [0.010, 0.041] | 2 / 0 | 0.660 [0.587, 0.722] |
| ens_afri_ssamtl | 0.749 [0.724, 0.772] | 0.135 [0.109, 0.162] | 2 / 0 | 0.751 [0.705, 0.794] |
| ens_min_afri_ssamtl | 0.624 [0.585, 0.657] | 0.010 [0.005, 0.015] | 2 / 0 | 0.654 [0.587, 0.711] |
| ens3 | 0.740 [0.711, 0.766] | 0.126 [0.106, 0.148] | 2 / 0 | 0.756 [0.697, 0.802] |
| ens_shaped | 0.737 [0.711, 0.761] | 0.123 [0.095, 0.150] | 2 / 0 | 0.753 [0.703, 0.800] |
| consensus | 0.163 [0.121, 0.212] | -0.450 [-0.507, -0.384] | 0 / 2 | 0.169 [0.093, 0.248] |

#### Absolute selection gain: target value of the sample each metric ranks first MINUS the group mean (all languages; chrF++ points / COMET points x 100), 90% CI over prompts

| metric | chrF++ vs reference | AfriCOMET-STL-1.1 with reference | SSA-COMET-MTL with reference | mean z of the two reference-based COMETs |
|---|---|---|---|---|
| africomet_qe_stl11 | 0.87 [0.78, 0.98] | 5.61 [5.50, 5.72] | 5.19 [5.05, 5.32] | 41.26 [40.47, 42.06] |
| ssa_qe | 0.70 [0.60, 0.81] | 5.31 [5.20, 5.43] | 5.34 [5.20, 5.48] | 40.66 [39.77, 41.49] |
| ssa_mtl_qe | 0.86 [0.75, 0.96] | 4.09 [3.96, 4.22] | 7.67 [7.55, 7.77] | 44.35 [43.53, 45.07] |
| ens_afri_ssaqe | 0.86 [0.76, 0.96] | 5.62 [5.51, 5.73] | 5.43 [5.30, 5.57] | 42.22 [41.43, 43.01] |
| ens_afri_ssamtl | 0.99 [0.88, 1.09] | 5.27 [5.15, 5.38] | 7.32 [7.20, 7.43] | 47.75 [47.01, 48.47] |
| ens_min_afri_ssamtl | 0.93 [0.83, 1.04] | 5.09 [4.97, 5.21] | 6.80 [6.67, 6.93] | 45.14 [44.35, 45.87] |
| ens3 | 0.93 [0.82, 1.03] | 5.50 [5.39, 5.61] | 6.88 [6.77, 7.00] | 47.05 [46.32, 47.80] |
| ens_shaped | 0.99 [0.88, 1.09] | 5.32 [5.20, 5.42] | 7.13 [7.02, 7.24] | 47.27 [46.53, 47.96] |
| consensus | 1.99 [1.91, 2.06] | 1.85 [1.71, 1.98] | 2.17 [2.00, 2.32] | 15.31 [14.26, 16.20] |
| oracle | 5.12 [5.05, 5.19] | 7.29 [7.20, 7.38] | 8.32 [8.22, 8.42] | 54.15 [53.51, 54.79] |

#### Per language: within-prompt Spearman with chrF++ (and with AfriCOMET-STL-1.1 + reference), mean over prompts

| lang | mode | prompts | africomet_qe_stl11 (chrF / refCOMET) | ssa_qe (chrF / refCOMET) | ssa_mtl_qe (chrF / refCOMET) | ens_afri_ssaqe (chrF / refCOMET) | ens_afri_ssamtl (chrF / refCOMET) | ens_min_afri_ssamtl (chrF / refCOMET) | ens3 (chrF / refCOMET) | ens_shaped (chrF / refCOMET) | consensus (chrF / refCOMET) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| afr | qe | 79 | 0.18 / 0.77 | 0.21 / 0.81 | 0.21 / 0.72 | 0.19 / 0.79 | 0.17 / 0.77 | 0.19 / 0.75 | 0.18 / 0.79 | 0.16 / 0.77 | 0.36 / 0.20 |
| amh | qe | 80 | 0.14 / 0.76 | 0.13 / 0.72 | 0.25 / 0.58 | 0.13 / 0.76 | 0.19 / 0.76 | 0.25 / 0.60 | 0.16 / 0.77 | 0.21 / 0.75 | 0.49 / 0.21 |
| bam | blend | 80 | 0.10 / 0.69 | 0.05 / 0.60 | 0.10 / 0.46 | 0.10 / 0.68 | 0.11 / 0.67 | 0.12 / 0.61 | 0.09 / 0.68 | 0.12 / 0.66 | 0.30 / 0.06 |
| bem | qe | 80 | 0.18 / 0.77 | 0.12 / 0.66 | 0.18 / 0.52 | 0.14 / 0.73 | 0.19 / 0.69 | 0.18 / 0.70 | 0.17 / 0.70 | 0.21 / 0.70 | 0.41 / 0.30 |
| dyu | blend | 80 | 0.15 / 0.70 | 0.12 / 0.67 | 0.17 / 0.42 | 0.13 / 0.71 | 0.17 / 0.67 | 0.17 / 0.63 | 0.15 / 0.70 | 0.16 / 0.65 | 0.47 / 0.09 |
| ewe | qe | 80 | 0.09 / 0.64 | 0.09 / 0.57 | 0.11 / 0.45 | 0.10 / 0.63 | 0.10 / 0.60 | 0.10 / 0.54 | 0.10 / 0.61 | 0.12 / 0.62 | 0.39 / 0.18 |
| fon | blend | 80 | 0.13 / 0.54 | 0.15 / 0.51 | 0.12 / 0.32 | 0.14 / 0.56 | 0.14 / 0.47 | 0.15 / 0.40 | 0.15 / 0.52 | 0.14 / 0.47 | 0.30 / 0.14 |
| hau | qe | 80 | 0.25 / 0.76 | 0.21 / 0.73 | 0.23 / 0.60 | 0.24 / 0.76 | 0.25 / 0.71 | 0.22 / 0.61 | 0.23 / 0.75 | 0.25 / 0.75 | 0.35 / 0.23 |
| ibo | qe | 80 | 0.15 / 0.61 | 0.11 / 0.62 | 0.19 / 0.37 | 0.16 / 0.65 | 0.17 / 0.55 | 0.20 / 0.48 | 0.17 / 0.62 | 0.19 / 0.57 | 0.31 / 0.22 |
| kab | blend | 80 | 0.09 / 0.64 | 0.07 / 0.64 | 0.06 / 0.46 | 0.09 / 0.67 | 0.08 / 0.63 | 0.11 / 0.63 | 0.07 / 0.66 | 0.11 / 0.65 | 0.48 / 0.15 |
| kam | blend | 80 | 0.14 / 0.72 | 0.17 / 0.71 | 0.13 / 0.52 | 0.17 / 0.73 | 0.14 / 0.67 | 0.13 / 0.69 | 0.16 / 0.71 | 0.15 / 0.68 | 0.32 / 0.16 |
| kbp | blend | 80 | 0.10 / 0.68 | 0.05 / 0.50 | 0.06 / 0.23 | 0.08 / 0.58 | 0.11 / 0.54 | 0.07 / 0.33 | 0.10 / 0.57 | 0.10 / 0.45 | 0.50 / 0.15 |
| kea | consensus | 80 | 0.10 / 0.57 | 0.10 / 0.55 | 0.10 / 0.37 | 0.10 / 0.58 | 0.12 / 0.54 | 0.10 / 0.57 | 0.12 / 0.57 | 0.12 / 0.53 | 0.41 / 0.12 |
| kik | qe | 80 | 0.19 / 0.71 | 0.14 / 0.64 | 0.19 / 0.54 | 0.15 / 0.68 | 0.22 / 0.66 | 0.20 / 0.68 | 0.20 / 0.68 | 0.20 / 0.66 | 0.29 / 0.21 |
| kin | qe | 80 | 0.20 / 0.77 | 0.15 / 0.69 | 0.20 / 0.55 | 0.19 / 0.76 | 0.24 / 0.72 | 0.21 / 0.69 | 0.20 / 0.74 | 0.24 / 0.74 | 0.40 / 0.28 |
| kmb | blend | 80 | 0.11 / 0.64 | 0.10 / 0.65 | 0.08 / 0.45 | 0.11 / 0.68 | 0.12 / 0.60 | 0.08 / 0.57 | 0.11 / 0.65 | 0.12 / 0.60 | 0.26 / 0.04 |
| lin | qe | 80 | 0.17 / 0.80 | 0.14 / 0.74 | 0.24 / 0.51 | 0.16 / 0.79 | 0.22 / 0.73 | 0.23 / 0.67 | 0.21 / 0.76 | 0.20 / 0.77 | 0.37 / 0.21 |
| lua | qe | 80 | 0.21 / 0.70 | 0.14 / 0.67 | 0.16 / 0.52 | 0.19 / 0.71 | 0.20 / 0.65 | 0.21 / 0.64 | 0.21 / 0.69 | 0.19 / 0.65 | 0.39 / 0.20 |
| lug | qe | 80 | 0.17 / 0.72 | 0.12 / 0.69 | 0.16 / 0.46 | 0.15 / 0.72 | 0.18 / 0.64 | 0.19 / 0.62 | 0.18 / 0.69 | 0.18 / 0.65 | 0.37 / 0.25 |
| luo | qe | 80 | 0.19 / 0.69 | 0.15 / 0.63 | 0.24 / 0.47 | 0.16 / 0.67 | 0.24 / 0.61 | 0.23 / 0.52 | 0.23 / 0.66 | 0.25 / 0.67 | 0.44 / 0.34 |
| mlg | qe | 80 | 0.28 / 0.82 | 0.25 / 0.78 | 0.26 / 0.72 | 0.27 / 0.82 | 0.31 / 0.84 | 0.29 / 0.78 | 0.30 / 0.84 | 0.31 / 0.84 | 0.38 / 0.39 |
| mos | blend | 80 | 0.20 / 0.66 | 0.16 / 0.65 | 0.19 / 0.43 | 0.18 / 0.68 | 0.23 / 0.62 | 0.20 / 0.46 | 0.20 / 0.67 | 0.24 / 0.62 | 0.32 / 0.09 |
| nso | qe | 80 | 0.12 / 0.77 | 0.11 / 0.73 | 0.15 / 0.60 | 0.11 / 0.76 | 0.14 / 0.74 | 0.15 / 0.71 | 0.12 / 0.77 | 0.12 / 0.76 | 0.42 / 0.31 |
| nus | blend | 80 | -0.00 / 0.65 | -0.06 / 0.48 | -0.01 / 0.30 | -0.04 / 0.59 | -0.01 / 0.54 | 0.02 / 0.51 | -0.03 / 0.55 | -0.02 / 0.50 | 0.33 / 0.13 |
| nya | qe | 80 | 0.10 / 0.73 | 0.07 / 0.68 | 0.14 / 0.54 | 0.09 / 0.72 | 0.15 / 0.70 | 0.14 / 0.70 | 0.13 / 0.71 | 0.15 / 0.71 | 0.32 / 0.21 |
| orm | qe | 80 | 0.16 / 0.68 | 0.09 / 0.53 | 0.13 / 0.41 | 0.14 / 0.63 | 0.15 / 0.58 | 0.15 / 0.52 | 0.13 / 0.58 | 0.15 / 0.59 | 0.46 / 0.33 |
| pcm | qe | 79 | -0.01 / 0.31 | 0.01 / 0.32 | 0.19 / 0.38 | -0.00 / 0.34 | 0.07 / 0.42 | -0.00 / 0.32 | 0.06 / 0.43 | 0.12 / 0.41 | 0.25 / 0.20 |
| run | qe | 80 | 0.19 / 0.82 | 0.15 / 0.76 | 0.22 / 0.58 | 0.19 / 0.82 | 0.23 / 0.75 | 0.21 / 0.75 | 0.21 / 0.79 | 0.24 / 0.77 | 0.39 / 0.21 |
| sag | blend | 80 | 0.11 / 0.71 | 0.07 / 0.62 | 0.16 / 0.48 | 0.09 / 0.68 | 0.16 / 0.66 | 0.14 / 0.55 | 0.13 / 0.67 | 0.17 / 0.63 | 0.32 / 0.20 |
| sna | qe | 80 | 0.12 / 0.74 | 0.14 / 0.72 | 0.15 / 0.53 | 0.14 / 0.75 | 0.16 / 0.71 | 0.11 / 0.66 | 0.14 / 0.73 | 0.14 / 0.73 | 0.26 / 0.18 |
| som | qe | 80 | 0.12 / 0.78 | 0.08 / 0.75 | 0.10 / 0.59 | 0.09 / 0.77 | 0.11 / 0.74 | 0.10 / 0.65 | 0.10 / 0.78 | 0.10 / 0.75 | 0.37 / 0.21 |
| sot | qe | 80 | 0.22 / 0.77 | 0.20 / 0.74 | 0.23 / 0.58 | 0.21 / 0.78 | 0.26 / 0.73 | 0.28 / 0.71 | 0.25 / 0.76 | 0.24 / 0.73 | 0.34 / 0.27 |
| ssw | qe | 80 | 0.25 / 0.82 | 0.21 / 0.74 | 0.27 / 0.56 | 0.24 / 0.81 | 0.30 / 0.74 | 0.28 / 0.65 | 0.28 / 0.78 | 0.30 / 0.77 | 0.46 / 0.29 |
| swh | qe | 80 | 0.18 / 0.73 | 0.16 / 0.74 | 0.22 / 0.65 | 0.18 / 0.74 | 0.20 / 0.74 | 0.20 / 0.70 | 0.19 / 0.77 | 0.19 / 0.74 | 0.41 / 0.30 |
| tir | qe | 80 | 0.19 / 0.76 | 0.15 / 0.73 | 0.29 / 0.57 | 0.17 / 0.75 | 0.28 / 0.76 | 0.29 / 0.61 | 0.23 / 0.77 | 0.26 / 0.75 | 0.47 / 0.31 |
| tsn | qe | 80 | 0.03 / 0.71 | 0.04 / 0.70 | 0.11 / 0.51 | 0.04 / 0.72 | 0.08 / 0.64 | 0.08 / 0.62 | 0.08 / 0.69 | 0.08 / 0.68 | 0.39 / 0.23 |
| tso | qe | 80 | 0.16 / 0.77 | 0.17 / 0.68 | 0.20 / 0.52 | 0.18 / 0.74 | 0.22 / 0.71 | 0.17 / 0.71 | 0.20 / 0.72 | 0.22 / 0.73 | 0.35 / 0.19 |
| tum | qe | 80 | 0.22 / 0.75 | 0.16 / 0.64 | 0.19 / 0.48 | 0.19 / 0.73 | 0.24 / 0.66 | 0.22 / 0.64 | 0.22 / 0.69 | 0.24 / 0.65 | 0.37 / 0.32 |
| twi | qe | 80 | 0.12 / 0.59 | 0.09 / 0.56 | 0.13 / 0.39 | 0.10 / 0.57 | 0.14 / 0.54 | 0.13 / 0.52 | 0.14 / 0.58 | 0.16 / 0.54 | 0.37 / 0.22 |
| umb | blend | 80 | 0.12 / 0.72 | 0.09 / 0.68 | 0.22 / 0.55 | 0.12 / 0.72 | 0.18 / 0.69 | 0.18 / 0.65 | 0.15 / 0.72 | 0.19 / 0.68 | 0.32 / 0.18 |
| vmw | consensus | 80 | 0.08 / 0.60 | 0.12 / 0.60 | 0.13 / 0.35 | 0.10 / 0.61 | 0.11 / 0.54 | 0.09 / 0.60 | 0.14 / 0.59 | 0.13 / 0.54 | 0.32 / 0.06 |
| wol | qe | 80 | 0.10 / 0.59 | 0.04 / 0.56 | 0.06 / 0.37 | 0.08 / 0.64 | 0.09 / 0.52 | 0.10 / 0.55 | 0.07 / 0.57 | 0.09 / 0.54 | 0.31 / 0.18 |
| xho | qe | 80 | 0.18 / 0.75 | 0.16 / 0.73 | 0.18 / 0.63 | 0.17 / 0.74 | 0.18 / 0.75 | 0.17 / 0.67 | 0.18 / 0.76 | 0.18 / 0.76 | 0.43 / 0.26 |
| yor | qe | 80 | 0.09 / 0.47 | 0.07 / 0.45 | 0.09 / 0.30 | 0.08 / 0.48 | 0.09 / 0.44 | 0.10 / 0.40 | 0.09 / 0.46 | 0.10 / 0.44 | 0.26 / 0.23 |
| zul | qe | 80 | 0.15 / 0.76 | 0.17 / 0.71 | 0.25 / 0.60 | 0.17 / 0.76 | 0.22 / 0.75 | 0.23 / 0.66 | 0.20 / 0.77 | 0.21 / 0.77 | 0.41 / 0.30 |

#### Pooled Spearman over all valid samples (not within prompt; confounded by prompt difficulty and language)

| metric | pooled over all languages: chrF++ | AfriCOMET-STL-1.1 ref | mean of per-language pooled: chrF++ | refCOMET |
|---|---|---|---|---|
| africomet_qe_stl11 | 0.429 | 0.738 | 0.271 | 0.696 |
| ssa_qe | 0.305 | 0.672 | 0.267 | 0.684 |
| ssa_mtl_qe | 0.479 | 0.631 | 0.292 | 0.549 |
| ens_afri_ssaqe | 0.380 | 0.733 | 0.276 | 0.709 |
| ens_afri_ssamtl | 0.495 | 0.751 | 0.306 | 0.683 |
| ens_min_afri_ssamtl | 0.485 | 0.712 | 0.306 | 0.647 |
| ens3 | 0.435 | 0.742 | 0.300 | 0.704 |
| ens_shaped | 0.380 | 0.658 | 0.302 | 0.684 |
| consensus | 0.802 | 0.526 | 0.582 | 0.357 |

#### Within-prompt Spearman BETWEEN metrics (mean over languages) - redundancy of the signals

| | africomet_qe_stl11 | ssa_qe | ssa_mtl_qe | ens_afri_ssaqe | ens_afri_ssamtl | ens_min_afri_ssamtl | ens3 | ens_shaped | consensus |
|---|---|---|---|---|---|---|---|---|---|
| africomet_qe_stl11 | 1.00 | 0.82 | 0.53 | 0.92 | 0.81 | 0.77 | 0.85 | 0.82 | 0.19 |
| ssa_qe | 0.82 | 1.00 | 0.57 | 0.94 | 0.77 | 0.71 | 0.88 | 0.77 | 0.13 |
| ssa_mtl_qe | 0.53 | 0.57 | 1.00 | 0.56 | 0.84 | 0.79 | 0.76 | 0.82 | 0.20 |
| ens_afri_ssaqe | 0.92 | 0.94 | 0.56 | 1.00 | 0.81 | 0.76 | 0.90 | 0.82 | 0.16 |
| ens_afri_ssamtl | 0.81 | 0.77 | 0.84 | 0.81 | 1.00 | 0.87 | 0.94 | 0.96 | 0.21 |
| ens_min_afri_ssamtl | 0.77 | 0.71 | 0.79 | 0.76 | 0.87 | 1.00 | 0.84 | 0.85 | 0.20 |
| ens3 | 0.85 | 0.88 | 0.76 | 0.90 | 0.94 | 0.84 | 1.00 | 0.92 | 0.19 |
| ens_shaped | 0.82 | 0.77 | 0.82 | 0.82 | 0.96 | 0.85 | 0.92 | 1.00 | 0.22 |
| consensus | 0.19 | 0.13 | 0.20 | 0.16 | 0.21 | 0.20 | 0.19 | 0.22 | 1.00 |