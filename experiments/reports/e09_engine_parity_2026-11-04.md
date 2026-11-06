# E-09: engine parity, vLLM fp32 against HF fp32 (2026-11-04)

Question: the headline numbers come from vLLM bf16, while the earlier parity gate (>= 95 % identical outputs) between vLLM bf16 and HF bf16 failed (9.6 % identical). Where does the difference come from, and does the engine matter for the RL-vs-SFT comparison? Set: the 240 sentences of `parity2.py` (10 languages yor hau swh amh zul ibo kin som xho wol x 24 FLORES+ devtest sentences), greedy, 256 new tokens, SFT final and RL (mixed best), four engines each: HF bf16, HF fp32, vLLM bf16, vLLM fp32 (`overnight/parity3.py`, `parity_compare.py`; data `reports/data/e09_engine_parity_2026-11-04/`; chrF++ language-macro with a paired Poisson bootstrap over sentences, `boot.py`).

## Result

| model | HF bf16 | HF fp32 | vLLM bf16 | vLLM fp32 |
|---|---:|---:|---:|---:|
| SFT, chrF++ (language macro) | 36.04 | 37.29 | 35.50 | 37.30 |
| RL, chrF++ | 39.28 | 39.61 | 39.51 | 39.59 |

Identical outputs (share of the 240 sentences):

| pair | SFT | RL |
|---|---:|---:|
| **vLLM fp32 vs HF fp32** | **0.988** | **0.971** |
| vLLM bf16 vs HF bf16 | 0.096 | 0.167 |
| HF bf16 vs HF fp32 | 0.079 | 0.154 |
| vLLM bf16 vs vLLM fp32 | 0.067 | 0.163 |

* **In fp32 the two engines agree** (98.8 % and 97.1 % identical outputs; chrF++ difference -0.01 [-0.08, +0.06] for the SFT and +0.02 [-0.02, +0.08] for the RL model, hf32 minus vllm32). The parity gate is met in fp32 and fails in bf16: **the disagreement between engines is bf16 rounding, not an engine bug.** Greedy outputs of a bf16 model are not engine-independent.
* **The precision effect on quality depends on the model.** RL: bf16 vs fp32 changes chrF++ by +0.09 [-0.41, +0.59] (vLLM) and +0.34 [-0.27, +0.96] (HF), i.e. nothing. SFT: fp32 is better by **+1.81 [+0.59, +3.12]** (vLLM) and +1.25 [+0.32, +2.19] (HF) on this 240-sentence set. The earlier 4,000-sentence study (20 languages x 200 sentences, vLLM, SFT, `dtype_float32.json`, in the ledger as D6) gives only +0.24 [-0.07, +0.55]. The two disagree; the larger set is the better estimate of the average effect and the small set contains languages with very low scores and runaway outputs where a few sentences move the macro (n = 24 per language). Direction and size for the SFT are therefore uncertain: between +0.2 and +1.8 chrF++ in favour of fp32.
* **Consequence for the RL-vs-SFT comparison:** the SFT is the more bf16-sensitive model, so a bf16 harness may understate the SFT by a point or less and thereby overstate the RL gain by the same amount (here: RL minus SFT 4.0 chrF++ in vLLM bf16, 2.3 in vLLM fp32, on this small set; the full-set gain is +8.8 on FLORES+ and the effect is a fraction of that). A full-set fp32 evaluation of the SFT (about 0.5 GPU-h) would settle it; it was not run (vLLM dtype is fixed to bf16 in the harness's engine config, `--dtype` only reaches the HF path).

## Recommendation for the paper

Report vLLM bf16 as the headline engine (all stages measured the same way), state that outputs of bf16 models differ between engines in over 80 % of the sentences and agree in fp32 (97-99 %), and give the fp32 sensitivity: negligible for RL, between +0.2 and +1.8 chrF++ for the SFT on the two small sets. Do not describe the engines as interchangeable in bf16.
