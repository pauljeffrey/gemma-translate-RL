| metric | median t_fail | median t_mid | median t_great | median (t_great - t_fail) | languages with t_fail > t_mid - 0.05 |
|---|---|---|---|---|---|
| africomet_qe_stl11 | 0.520 | 0.578 | 0.698 | 0.178 | 22 of 45 |
| ssa_qe | 0.616 | 0.629 | 0.702 | 0.083 | 39 of 45 |
| ssa_mtl_qe | 0.337 | 0.569 | 0.675 | 0.335 | 1 of 45 |
| v7 calibration (AfriCOMET, n=997, 43 languages) | 0.535 | 0.600 | 0.723 | 0.173 | - |

Sanity: my n=80 AfriCOMET anchors minus the v7 n=997 anchors, median |difference| (t_fail, t_mid, t_great) = [0.019, 0.016, 0.012].
ssa_mtl_qe anchors minus v7 AfriCOMET anchors: median shift (t_fail, t_mid, t_great) = [-0.198, -0.032, -0.036], median |shift| = [0.198, 0.049, 0.063], share of languages whose t_mid moves by more than 0.05: 0.49
ssa_qe anchors minus v7 AfriCOMET anchors: median shift (t_fail, t_mid, t_great) = [0.086, 0.031, -0.009], median |shift| = [0.086, 0.044, 0.051], share of languages whose t_mid moves by more than 0.05: 0.47

Pooled (mean, sd) of the human-reference scores, the constants of the z-average ensemble: {"africomet_qe_stl11": [0.5604, 0.141], "ssa_qe": [0.6187, 0.0766], "ssa_mtl_qe": [0.5572, 0.1151]}