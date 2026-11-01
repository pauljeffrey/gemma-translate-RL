# Wave 2: the next 24 languages, chosen by the performance of final2 (2026-10-25)

Rule (owner, 2026-10-25): use the translation model's measured performance to pick the next 24 languages after the 12 pilot languages.

## Method
* Source: `reports/data/final2_eval_2026-10-23/per_language_bench.csv`, system final2 (step 401), greedy, subset `no_pair` (rows whose English sentence AND pair were not seen in SFT training; the `clean` subset has only 48 rows per language).
* Ruler: FLORES+ devtest chrF++ (45 languages). Languages without a FLORES+ row (pcm, nbl, ven, tiv, efi, ach, nyn, kri, ndc) get a FLORES-equivalent estimate: linear fit of FLORES+ chrF++ on SmolSent (n=35 languages, residual sd 3.2), NTREX (n=21, sd 4.5) or MAFAND (n=14, sd 5.4), averaged over the benchmarks available. Their rank is therefore uncertain by about +-3 chrF++.
* Excluded: the 12 pilot languages; the 8 tags without SFT data (nde, fat, knc, gaa, men, luy, tzm, ful); 22 languages with no benchmark at all (ibb urh bin dag oke bbj swc kqn lun iso loz wal tsc kua seh her yao sid ish dua bas wes) stay for wave 3.
* All final2 outputs finished (finish rate 1.000) on every benchmark, so the ranking is by chrF++.

## The 24 (FLORES-equivalent chrF++)
| # | tag | language | chrF++ | source of the number |
|---|---|---|---|---|
| 1 | pcm | Pidgin | 55.3 | est. SmolSent + MAFAND |
| 2 | nso | Sepedi | 49.9 | FLORES+ |
| 3 | tsn | Setswana | 46.1 | FLORES+ |
| 4 | tso | Tsonga | 46.0 | FLORES+ |
| 5 | sot | Sesotho | 45.5 | FLORES+ |
| 6 | mlg | Malagasy | 43.5 | FLORES+ |
| 7 | lin | Lingala | 42.6 | FLORES+ |
| 8 | nbl | South Ndebele | 41.7 | est. SmolSent |
| 9 | ven | Venda | 39.6 | est. SmolSent + NTREX |
| 10 | ssw | Swati | 38.0 | FLORES+ |
| 11 | twi | Twi | 37.4 | FLORES+ |
| 12 | sag | Sango | 34.9 | FLORES+ (tier `listed`, no validated QE) |
| 13 | bem | Bemba | 34.3 | FLORES+ |
| 14 | run | Kirundi | 34.2 | FLORES+ |
| 15 | tum | Tumbuka | 33.8 | FLORES+ |
| 16 | lua | Luba-Kasai | 33.4 | FLORES+ |
| 17 | luo | Luo | 32.6 | FLORES+ |
| 18 | kik | Kikuyu | 32.1 | FLORES+ |
| 19 | orm | Oromo | 31.7 | FLORES+ |
| 20 | tiv | Tiv | 31.5 | est. SmolSent |
| 21 | ewe | Ewe | 31.5 | FLORES+ |
| 22 | efi | Efik | 30.8 | est. SmolSent |
| 23 | ach | Acholi | 30.5 | est. SmolSent |
| 24 | wol | Wolof | 29.9 | FLORES+ |

Next in line (not taken): nyn 29.5, kri 29.0, vmw 26.9, kab 26.7, kea 26.3, ndc 26.3, kmb 25.8, umb 25.4, kam 25.0, bam 24.7. Ranks 19-27 differ by less than the estimation error, so the cut at 24 is soft; the pilot languages yor (28.2) and lug (35.8) are in a similar range, and amh (20.0), tir (13.9), fon (21.6) are below the cut (poor; handle separately, see below).

## Pre-flight probe on real story sentences (`reports/data/wave2_probe_2026-10-25.jsonl`)
120 sentences (>= 40 characters) of Children-Stories / cosmopedia.stories / Wikipedia translated greedily with a huge budget on the CPU servers; output/source token ratio, GlotLID, loops, exact copies.
* No sentence was left unfinished, no loops, no exact copies in any of the 24.
* Output budget: the pilot default (3.5x) would have cut many sentences; the new default is 5.0x with overrides ven 7.0 (max ratio 6.0), orm 6.5 (max 5.75), kik 5.5 (max 4.7), sag 5.5 (max 4.9). All other languages peak below 4.5.
* GlotLID: 22 of 24 at >= 99% accepted labels. Exceptions: pcm 69% (28% labelled English: Pidgin is close to English; `eng_Latn` is now accepted for pcm and the copy check catches real copies), nbl 85% (labelled zul/xho/nde: sibling labels already accepted). luo/ach are confusable (sibling labels added).

## Not in wave 2, and why
amh (20.0 chrF++; the worst language in the 200-sample real-data check, `reports/real_data_band_check_200_samples_2026-10-24.md`), tir (13.9), fon (21.6), bam (24.7), kbp, nus, dyu: their FLORES+ chrF++ is below 25, so translated data would be mostly low-fidelity. They could be done later with extra repair samples and a quality estimator filter.
