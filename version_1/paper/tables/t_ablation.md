| quantity (macro of yor, kin, orm, lin; FLORES+ devtest `no\_pair`; 95% interval) | arm F: RL from SFT-final | arm W: RL from weak SFT | W + continued SFT, equal GPU time (S1) | W + continued SFT, equal number of sentences (S1P) |
|:--|--:|--:|--:|--:|
| start model (F0 for arm F, W0 for the others), chrF++ | 28.77 [28.20, 29.38] | 15.16 [14.74, 15.65] | 15.16 [14.74, 15.65] | 15.16 [14.74, 15.65] |
| end model, chrF++ | 34.49 [34.09, 34.90] | 22.52 [22.07, 22.94] | 15.48 [15.04, 15.98] | 14.15 [13.74, 14.62] |
| gain (end minus start), chrF++ | +5.72 [+5.15, +6.30] | +7.36 [+6.80, +7.85] | +0.32 [-0.18, +0.82] | -1.01 [-1.49, -0.55] |
| difference of the gains, gain_W minus gain_F (DiD) | -- | +1.64 [+0.84, +2.36] | -- | -- |
| W_RL minus the control, chrF++ (RL minus continued SFT) | -- | -- | +7.04 [+6.49, +7.54] | +8.37 [+7.84, +8.83] |
| finished within 768 tokens, before / after (%) | 94.3 / 99.1 | 70.0 / 88.7 | 70.0 / 72.5 | 70.0 / 67.3 |
| gain of the finished rate (points) | +4.8 [+4.1, +5.6] | +18.7 [+17.2, +20.1] | +2.5 [+1.0, +3.9] | -2.6 [-4.3, -1.0] |
| gain, AfriCOMET with reference (x100) | +6.98 [+6.52, +7.41] | +10.31 [+9.84, +10.81] | +0.25 [-0.18, +0.65] | -0.72 [-1.14, -0.31] |
| gain, SSA-COMET with reference (x100) | +5.79 [+5.39, +6.15] | +7.79 [+7.35, +8.22] | +0.44 [+0.05, +0.83] | -0.87 [-1.27, -0.48] |
| sampled completions / training | 50176 | 50176 | none (SFT, 367 steps, about 48 M tokens) | none (SFT, 3,136 samples) |
| gain per million sampled completions (chrF++) | 114.0 | 146.68 | -- | -- |
