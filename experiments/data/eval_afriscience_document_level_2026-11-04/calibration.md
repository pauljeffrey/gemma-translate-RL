| released run | released SSA-COMET | ssa-comet-mtl, reference-based (src+mt+ref, mean of 3 views) | ssa-comet-mtl, reference-free (src+mt) | ssa-comet-qe, reference-free |
|---|---:|---:|---:|---:|
| gpt54 semantic_10shot eng-amh | 0.666 | 0.608 | 0.604 | 0.667 |
| gpt54 semantic_10shot eng-hau | 0.646 | 0.585 | 0.560 | 0.647 |
| gpt54 semantic_10shot eng-zul | 0.663 | 0.598 | 0.576 | 0.646 |
| gemini_31_flash_lite semantic_10shot eng-amh | 0.676 | 0.621 | 0.615 | 0.674 |
| gemini_31_flash_lite semantic_10shot eng-hau | 0.653 | 0.593 | 0.567 | 0.650 |
| gemini_31_flash_lite semantic_10shot eng-zul | 0.661 | 0.596 | 0.575 | 0.647 |
| gemma2_9b_it template1_0shot eng-amh | 0.323 | 0.152 | 0.171 | 0.404 |
| gemma2_9b_it template1_0shot eng-hau | 0.485 | 0.375 | 0.387 | 0.566 |
| gemma2_9b_it template1_0shot eng-zul | 0.366 | 0.194 | 0.209 | 0.446 |

Mean absolute difference to the released value (all 9 runs / the 6 strongest runs): ssa-comet-mtl, reference-based (src+mt+ref, mean of 3 views): 0.091 / 0.061; ssa-comet-mtl, reference-free (src+mt): 0.097 / 0.078; ssa-comet-qe, reference-free: 0.031 / 0.006
