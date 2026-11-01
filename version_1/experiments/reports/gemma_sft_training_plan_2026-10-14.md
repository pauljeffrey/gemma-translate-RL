# Gemma-3-270m-it English -> African SFT: how the training works (2026-10-14)

Model `google/gemma-3-270m-it`, full fine-tuning, forward direction only (English -> African language). Run name `runs/sft_fwd` on the
vast box; Hub repo `<HF_ORG_LEGACY>/gemma3-270m-translate` (private). This file contains counts and settings only, no data.

## 1. What a training sample is
- A **window** = consecutive same-source, same-language-pair sentence pairs concatenated in order on both sides, no newline
  (pseudo-paragraph). Four length bins by the English side in AfriCOMET tokens: b64, b256, b512, b1024. Bins are token-balanced
  (each bin holds a similar number of tokens, so bins with long windows have far fewer windows).
- Prompt: `<bos><start_of_turn>user\nTranslate the text to {Language}:  {English window}<end_of_turn>\n<start_of_turn>model\n`
  Answer: `{iso} {translation}<end_of_turn>`. Loss ONLY on the answer tokens (tag + translation + `<end_of_turn>`).
- Samples longer than 4096 Gemma tokens are dropped (never truncated); none were dropped in this build.

## 2. Packing
- 4096-token packs, **one language per pack**, best-fit-decreasing inside each language, whole samples only (no sample is split).
  Packing efficiency 99.94% (538,480 train packs, 2,204,187,720 tokens, 1,318,270,369 of them target tokens; 21 windows per pack).
- Block-diagonal attention per sample, positions restart at 0 for each sample (no cross-sample attention). Tested numerically
  against separately computed logits, and audited on the real data (structure of every pack + token-level checks of 4,000 samples
  per split: one `<bos>`, exactly two turn markers, target ends in `<end_of_turn>`, labels cover exactly the target tokens): **AUDIT OK**.
- Because packs are per language but mix bins, every window bin is consumed: a pack holds whatever windows of that language fit.

## 3. Sampling (language-stratified)
- Each optimizer step draws 32 packs (micro-batch 2 packs x 16 accumulation steps = 32 x 4096 = 131,072 tokens).
- Language drawn with probability proportional to `packs ** 0.5`, floored at 0.2% so that every language appears in every
  250-step evaluation interval; inside a language packs go round in a seeded permutation (no repeats before the language is exhausted).
- A language is **retired after 4 passes** over its own data (small languages are not memorised; the freed share goes to the others).
- 16,827 steps = one natural pass over all packs (~100 h at 6,100 tokens/s on the RTX 5070). Per-language shares and effective epochs
  are in the table below. Large languages get <1 epoch, small ones up to 4.
- Dedup note: Pidgin windows reused by the x2 pass were mostly identical to pass-1 windows and were removed by exact deduplication
  (81,812 duplicates in total), so the x2 oversample only added ~15k new Pidgin windows; Pidgin gets its extra exposure through the
  sampler (1.6 effective epochs, 1.68% of draws).
- Validation: 26,938 windows (1,234 packs), split by the hash of the English text, Pidgin training windows that share an English
  sentence with a Pidgin validation window were removed (3,713).

## 4. Optimisation
- AdamW, lr 5e-5 (warmup 3% = 505 steps, cosine to 5e-6), betas (0.9, 0.95), weight decay 0.01, grad clip 1.0, fp32 master weights with
  bf16 autocast, gradient checkpointing, token-weighted loss over the 131k-token step.
- Evaluation every 250 steps (~90 min). **Monitored metric: macro validation loss** (mean of the per-language losses, the token-weighted
  loss is 25% Igbo). Per-language losses are logged and pushed.

## 5. Stopping and checkpoints
- Stop when the macro validation loss has not improved for **3 consecutive evaluations** (750 steps, ~4.5 h), no earlier than 25%
  of the natural pass. The best model is already on the Hub when that happens.
- Hub: only a new best is pushed (private repo, root = best checkpoint, tag `step-N`, `training_state.json` with the metrics and
  config). Locally one checkpoint only: `runs/sft_fwd/last/trainer_state.pt` (resumable, exact).

## 6. After training
- Benchmarks on the best checkpoint (FLORES+, NTREX-128, MAFAND, en -> X) with the base model as baseline, plus a contamination
  check of the benchmark English sentences against the training windows. Results are saved in `reports/`.
- RL (AfriCOMET-QE reward, staged length bins) only after you have seen the SFT results.

## Window bins (all languages, before the train/val split)
| bin | windows | English tokens (AfriCOMET) | target tokens (AfriCOMET) |
|---|---|---|---|
| b64 | 7,150,630 | 167,562,178 | 270,008,008 |
| b256 | 2,750,979 | 175,410,257 | 310,275,781 |
| b512 | 967,519 | 221,125,963 | 369,851,583 |
| b1024 | 573,044 | 226,517,412 | 395,887,844 |

## Per language
| tag | language | windows b64 / b256 / b512 / b1024 | train samples | target tokens (Gemma) | packs | share of draws | effective epochs | val windows |
|---|---|---|---|---|---|---|---|---|
| ibo | Igbo | 1,999,275 / 676,505 / 193,462 / 101,245 | 2,964,219 | 260,311,732 | 105,821 | 7.31% | 0.4 | 6,259 |
| yor | Yoruba | 780,854 / 615,674 / 82,919 / 43,591 | 1,519,727 | 145,155,296 | 56,817 | 5.38% | 0.5 | 3,289 |
| hau | Hausa | 767,785 / 268,697 / 81,600 / 40,241 | 1,155,980 | 135,024,789 | 58,519 | 5.43% | 0.5 | 2,338 |
| twi | Twi | 177,584 / 65,834 / 77,988 / 217,392 | 537,664 | 165,853,001 | 65,244 | 5.77% | 0.5 | 1,134 |
| fon | Fon | 253,599 / 85,506 / 24,817 / 12,309 | 375,361 | 34,028,294 | 13,172 | 2.55% | 1.0 | 870 |
| xho | Xhosa | 244,138 / 76,870 / 34,043 / 12,715 | 367,026 | 46,705,118 | 18,924 | 3.06% | 0.9 | 738 |
| tsn | Setswana | 217,371 / 81,150 / 23,296 / 11,538 | 332,597 | 34,209,543 | 13,790 | 2.63% | 1.0 | 758 |
| afr | Afrikaans | 216,878 / 52,689 / 19,637 / 9,676 | 298,234 | 33,190,969 | 15,005 | 2.74% | 1.0 | 646 |
| swh | Swahili | 202,288 / 49,008 / 19,104 / 9,192 | 279,040 | 35,364,489 | 15,347 | 2.79% | 1.0 | 552 |
| zul | Zulu | 141,465 / 43,624 / 38,435 / 6,984 | 229,980 | 33,585,226 | 13,422 | 2.62% | 1.0 | 515 |
| pcm | Pidgin | 173,350 / 23,824 / 8,992 / 4,456 | 124,861 | 10,612,775 | 5,596 | 1.68% | 1.6 | 308 |
| som | Somali | 102,656 / 31,966 / 54,789 / 4,706 | 193,714 | 37,763,040 | 14,873 | 2.75% | 1.0 | 403 |
| kin | Kinyarwanda | 99,165 / 39,666 / 40,114 / 4,967 | 183,557 | 28,839,951 | 11,935 | 2.49% | 1.1 | 354 |
| luo | Luo | 124,457 / 30,314 / 11,430 / 5,684 | 171,483 | 17,146,609 | 7,559 | 1.95% | 1.4 | 402 |
| efi | Efik | 105,842 / 45,725 / 11,894 / 5,908 | 168,862 | 15,626,749 | 6,456 | 1.77% | 1.5 | 507 |
| sot | Sesotho | 65,546 / 17,743 / 43,879 / 3,747 | 130,669 | 24,842,204 | 9,778 | 2.23% | 1.2 | 237 |
| nya | Chichewa | 75,029 / 19,974 / 31,209 / 3,639 | 129,583 | 20,969,998 | 8,465 | 2.08% | 1.3 | 268 |
| sna | Shona | 75,862 / 22,738 / 18,589 / 3,829 | 120,779 | 15,848,709 | 6,657 | 1.81% | 1.5 | 239 |
| lug | Luganda | 74,223 / 34,138 / 7,811 / 3,894 | 119,739 | 12,779,860 | 5,275 | 1.64% | 1.7 | 327 |
| loz | Lozi | 77,044 / 26,992 / 8,396 / 4,166 | 116,310 | 12,490,309 | 5,195 | 1.60% | 1.7 | 288 |
| orm | Oromo | 69,701 / 22,408 / 15,502 / 3,755 | 111,026 | 14,483,322 | 5,715 | 1.72% | 1.6 | 340 |
| nso | Sepedi | 63,350 / 27,198 / 6,902 / 3,423 | 100,637 | 10,464,494 | 4,194 | 1.49% | 1.9 | 226 |
| nus | Nuer | 56,829 / 24,588 / 6,432 / 3,196 | 90,891 | 9,371,086 | 3,747 | 1.36% | 2.0 | 154 |
| tso | Tsonga | 56,316 / 19,979 / 5,884 / 2,901 | 84,895 | 8,948,175 | 3,581 | 1.36% | 2.0 | 185 |
| mlg | Malagasy | 57,104 / 19,124 / 5,819 / 2,902 | 84,787 | 10,078,629 | 4,088 | 1.45% | 1.9 | 162 |
| lua | Luba-Kasai | 54,525 / 18,901 / 5,951 / 2,984 | 82,124 | 8,690,290 | 3,689 | 1.37% | 2.0 | 237 |
| iso | Isoko | 45,419 / 20,062 / 5,286 / 2,600 | 73,114 | 7,347,545 | 2,941 | 1.24% | 2.3 | 253 |
| amh | Amharic | 39,476 / 13,473 / 4,190 / 2,075 | 59,035 | 8,289,388 | 3,421 | 1.28% | 2.0 | 179 |
| kqn | Kaonde | 39,274 / 13,028 / 4,224 / 2,099 | 58,468 | 6,355,468 | 2,659 | 1.14% | 2.3 | 157 |
| swc | Congo Swahili | 37,706 / 10,084 / 3,813 / 1,875 | 53,316 | 6,757,824 | 2,887 | 1.20% | 2.2 | 162 |
| wol | Wolof | 34,725 / 10,827 / 3,256 / 1,768 | 50,459 | 5,031,731 | 2,214 | 1.05% | 2.5 | 117 |
| tir | Tigrinya | 31,722 / 11,838 / 3,338 / 1,655 | 48,410 | 5,600,654 | 2,180 | 1.06% | 2.6 | 143 |
| ven | Venda | 29,600 / 12,635 / 3,327 / 1,646 | 47,047 | 4,850,851 | 1,888 | 0.95% | 2.7 | 161 |
| bem | Bemba | 31,380 / 10,591 / 3,352 / 1,675 | 46,881 | 5,023,191 | 2,121 | 1.04% | 2.6 | 117 |
| mos | Mossi | 28,947 / 12,457 / 3,278 / 1,643 | 46,190 | 4,758,571 | 1,862 | 1.00% | 2.9 | 135 |
| tiv | Tiv | 26,374 / 10,041 / 2,897 / 1,449 | 40,656 | 4,270,512 | 1,721 | 0.93% | 2.9 | 105 |
| sag | Sango | 25,852 / 9,537 / 2,839 / 1,412 | 39,516 | 3,992,368 | 1,666 | 0.93% | 3.0 | 124 |
| kab | Kabyle | 26,916 / 8,396 / 2,599 / 1,294 | 39,105 | 3,885,266 | 1,594 | 0.87% | 2.9 | 99 |
| run | Kirundi | 26,002 / 8,888 / 2,755 / 1,373 | 38,923 | 4,108,968 | 1,694 | 0.92% | 2.9 | 95 |
| lin | Lingala | 24,918 / 8,439 / 2,668 / 1,319 | 37,254 | 3,793,007 | 1,594 | 0.91% | 3.1 | 90 |
| nbl | South Ndebele | 24,333 / 9,109 / 2,567 / 1,262 | 37,195 | 4,141,565 | 1,642 | 0.89% | 2.9 | 76 |
| wal | Wolaytta | 22,848 / 8,823 / 2,559 / 1,267 | 35,349 | 3,773,484 | 1,553 | 0.88% | 3.0 | 148 |
| tum | Tumbuka | 23,970 / 7,676 / 2,559 / 1,257 | 35,350 | 3,819,053 | 1,592 | 0.89% | 3.0 | 112 |
| lun | Lunda | 23,362 / 7,517 / 2,501 / 1,252 | 34,525 | 3,768,742 | 1,573 | 0.90% | 3.1 | 107 |
| umb | Umbundu | 23,413 / 7,337 / 2,482 / 1,222 | 34,376 | 3,675,942 | 1,541 | 0.89% | 3.1 | 78 |
| kik | Kikuyu | 21,117 / 9,310 / 2,426 / 1,201 | 33,879 | 3,477,346 | 1,346 | 0.83% | 3.3 | 175 |
| ssw | Swati | 21,097 / 7,108 / 2,267 / 1,117 | 31,467 | 3,565,671 | 1,461 | 0.84% | 3.1 | 122 |
| ewe | Ewe | 18,052 / 9,248 / 1,995 / 970 | 30,182 | 3,101,519 | 1,270 | 0.80% | 3.4 | 83 |
| kua | Kuanyama | 18,358 / 6,464 / 2,031 / 1,019 | 27,744 | 3,101,117 | 1,257 | 0.79% | 3.4 | 128 |
| kmb | Kimbundu | 16,087 / 5,222 / 1,727 / 866 | 23,819 | 2,522,745 | 1,036 | 0.72% | 3.7 | 83 |
| tsc | Tswa | 15,102 / 5,442 / 1,671 / 826 | 22,954 | 2,485,201 | 1,009 | 0.71% | 3.8 | 87 |
| vmw | Makhuwa | 13,020 / 4,556 / 1,401 / 708 | 19,613 | 2,132,204 | 874 | 0.65% | 4.0 | 72 |
| kam | Kamba | 11,264 / 5,016 / 1,291 / 643 | 18,073 | 1,859,930 | 716 | 0.53% | 4.0 | 141 |
| bin | Edo | 9,198 / 7,261 / 1,113 / 552 | 17,995 | 1,712,912 | 680 | 0.51% | 4.0 | 129 |
| ach | Acholi | 12,020 / 3,681 / 1,270 / 640 | 17,534 | 1,861,626 | 800 | 0.59% | 4.0 | 77 |
| dag | Dagbani | 7,115 / 6,790 / 1,068 / 410 | 15,310 | 1,927,495 | 795 | 0.59% | 4.0 | 73 |
| nyn | Nyankore | 8,887 / 3,033 / 964 / 483 | 13,297 | 1,439,508 | 594 | 0.44% | 4.0 | 70 |
| kri | Krio | 8,274 / 3,304 / 932 / 462 | 12,902 | 1,313,946 | 523 | 0.39% | 4.0 | 70 |
| seh | Sena | 8,629 / 2,859 / 939 / 464 | 12,821 | 1,426,424 | 582 | 0.43% | 4.0 | 70 |
| urh | Urhobo | 8,132 / 3,199 / 887 / 436 | 12,528 | 1,223,228 | 495 | 0.37% | 4.0 | 126 |
| sid | Sidamo | 8,402 / 2,796 / 912 / 457 | 12,467 | 1,363,870 | 557 | 0.41% | 4.0 | 100 |
| yao | Yao | 8,079 / 2,871 / 887 / 441 | 12,176 | 1,342,259 | 539 | 0.40% | 4.0 | 102 |
| shi | Tachelhit | 2,521 / 6,264 / 328 / 159 | 9,212 | 932,398 | 346 | 0.26% | 4.0 | 60 |
| oke | Okpe | 5,016 / 2,107 / 566 / 279 | 7,856 | 735,410 | 296 | 0.22% | 4.0 | 112 |
| bas | Basaa | 5,193 / 1,873 / 576 / 288 | 7,843 | 815,088 | 335 | 0.25% | 4.0 | 87 |
| ndc | Ndau | 5,148 / 1,605 / 544 / 275 | 7,511 | 826,303 | 342 | 0.25% | 4.0 | 61 |
| dua | Duala | 3,187 / 1,390 / 361 / 181 | 5,027 | 532,596 | 204 | 0.15% | 4.0 | 92 |
| dyu | Dyula | 2,983 / 1,094 / 330 / 167 | 4,504 | 481,760 | 192 | 0.14% | 4.0 | 70 |
| kbp | Kabiye | 2,045 / 1,963 / 244 / 121 | 4,311 | 430,136 | 162 | 0.12% | 4.0 | 62 |
| wes | Cameroon Pidgin | 3,032 / 762 / 294 / 146 | 4,170 | 428,204 | 205 | 0.15% | 4.0 | 64 |
| bam | Bambara | 2,600 / 812 / 246 / 138 | 3,736 | 377,272 | 154 | 0.11% | 4.0 | 60 |
| bbj | Ghomala | 1,888 / 1,364 / 235 / 113 | 3,540 | 375,594 | 147 | 0.11% | 4.0 | 60 |
| her | Herero | 1,479 / 518 / 162 / 76 | 2,170 | 234,739 | 95 | 0.07% | 4.0 | 65 |
| ibb | Ibibio | 1,339 / 544 / 163 / 74 | 2,060 | 202,632 | 84 | 0.06% | 4.0 | 60 |
| ish | Esan | 1,099 / 468 / 129 / 64 | 1,697 | 163,946 | 66 | 0.05% | 4.0 | 63 |
| kon | Kongo | 993 / 257 / 96 / 45 | 1,331 | 136,064 | 61 | 0.05% | 4.0 | 60 |
| kea | Kabuverdianu | 801 / 205 / 80 / 40 | 1,066 | 112,439 | 50 | 0.04% | 4.0 | 60 |
