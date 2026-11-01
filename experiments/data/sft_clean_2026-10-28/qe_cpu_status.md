# CPU AfriCOMET-QE job on box 2: status (written 2026-10-29)

**Job:** tmux session `qe_cpu` on $BOX_HOST (windows `s0`, `s1` = two scoring processes on cores 4-5 and 6-7, `push` = Hub push every 2 h). fp32, CUDA hidden (`CUDA_VISIBLE_DEVICES=""`, the RTX 3060 and `/workspace/tr`, `/workspace/chess` untouched), `taskset -c 4-7 nice -n 19 ionice -c3`, backs off above load average 20. Model `masakhane/africomet-qe-stl-1.1`, revision `a4bd2712feb066c2d3bb1f1e54e38bfabb17e743`.
**Resumable and GPU-compatible:** stable pair id = `key` (xxh64 of english.casefold, iso, target.casefold); each score row has key, qe, scorer, model_sha, device=cpu, dtype=fp32, ts. `qe_score.py --skip-scored` skips every key already scored by either device; `gate.py --scores dirA dirB` takes shards from both (first score per key wins). Details and the exact GPU resume command: `data_pipeline/sft_clean/chain/README_chain.md` (section "CPU QE job"). **GPU cross-check plan (500 pairs):** queue shard `q-000000` (500 mixed-language pairs, already scored on CPU fp32) is re-scored on the GPU fp16 (`qe_crosscheck.py prepare/compare`); pass = Pearson >= 0.999 and mean |diff| <= 0.003, else re-score the CPU part on the GPU and do not mix devices.
**H_cal switch:** rule CAL and `hcal.py` now use `heldout_split_cal_sel_v2.json` (article level, H_cal 261 / H_sel 296, Amendment 3), not my v1 (282/275). The H_cal references are NOT re-scored here: `gate.py fit --refs-json-dir /workspace/calib_r4/qe_work/africomet` reads the calibration agent's `<iso>.human.json` (10 languages so far: amh bam dyu fon kam kbp mos nus tir umb); the other languages use the pooled median offset until the GPU has scored their references. My own REF stage (v1 references) was removed from the queue before it ran.

## Live progress (read it any time)
```
ssh -i $SSH_KEY -p $BOX_PORT root@$BOX_HOST
cd /workspace/work; source /workspace/tr/env.sh
venv/bin/python chain/qe_cpu_status.py          # prints and writes qe_cpu/status.md
cat qe_cpu/progress_0.json qe_cpu/progress_1.json   # raw counters per process
tail -f qe_cpu/qe_cpu_0.log qe_cpu/qe_cpu_1.log     # pairs/s per chunk
tmux attach -t qe_cpu                                # windows s0 s1 push (detach: Ctrl-b d); do not Ctrl-C
ls qe_cpu/scores_cpu | wc -l                         # score shards (500 pairs each)
```
Hub copy (updated every 2 h): `<HF_ORG_LEGACY>/sabiyarn-clean-sft-2026-10-28` folder `qe_scores_cpu/` (`scores/`, `progress.json`, `status.md`, `queue_meta.json`, `queue_manifest.json`, `gate_fit_P1a_partial/`). Scores are keys and QE values only (no pair text, no benchmark text).

## Measured at the time of writing
# CPU QE status (updated 2026-10-29T01:54:04Z)

scored on CPU so far: 12,000 pairs; measured speed: 2.61 pairs/s recent, 2.56 run average; device cpu fp32; model_sha a4bd2712feb066c2d3bb1f1e54e38bfabb17e743
the GPU still has to score: 23,760,128 of 23,772,128 unique pairs

| stage | done | total | share | ETA at the recent speed |
|---|---:|---:|---:|---:|
| P1a | 12,000 | 38,188 | 31.4% | 2.8 h |
| P2a | 0 | 63,672 | 0.0% | 6.8 h |
| P1b | 0 | 105,000 | 0.0% | 11.2 h |
| P2b | 0 | 702,056 | 0.0% | 74.7 h |

stages (queue order): P1a = 500 random pairs per language (round-robin), REF = FLORES+ dev H_cal human references of the 20 specialist languages, P2a = all pairs (cap 50k) of nus kbp dyu bam kea, P1b = ranks 500-1999 of every language, P2b = the other priority languages.
 01:54:04 up 3 days,  4:32,  0 user,  load average: 10.91, 11.46, 11.56


Throughput: **about 2.3-2.9 pairs/s for both processes together** (4 cores shared with the farm; a single 4-thread process gave 2.55/s, two 2-core processes 2.6-2.9/s; load average on the box 10-13). That is about 220-250k pairs per day, 1% of the 23.77 M unique pairs per day. **The GPU still has to score all but the CPU-scored pairs: 23.7 M** (about 6 GPU-hours at the assumed 1,100 pairs/s; the CPU contribution changes this by well under 1%).

Queue (908,916 pairs): P1a 38,188 (500 random pairs per language) -> P2a 63,653 (all pairs of nus, kbp, dyu, bam, kea, cap 50k per language) -> P1b 105,119 (ranks 500-1999) -> P2b 701,956 (fon ewe wol seh kmb vmw kam, then ascending by size through the rest of the 20 specialist languages). I put P2a before P1b so that the thinnest languages are fully scored early; P1 up to 2,000 per language needs about 17 h of CPU in total.

## gate.py fit on what exists (rule OLD vs CAL, NOTHING DROPPED)
12,000 pairs scored (P1a is 31% done: about 155 random pairs per language, 77 languages; `gate_fit_P1a_partial/keep_rate_per_language.csv`, `thresholds.json`). Rule OLD = max(0.20, 15th percentile of the language's scored pairs): **keep 84.5-84.7% for every language by construction** (the 0.20 floor binds nowhere; the lowest 15th percentile is kea 0.219); extrapolated to all pairs the languages ending below 2,000 pairs are **ibb (about 845), men (1,370), nbl (158)** (unchanged from the build: these are source gaps). Rule CAL (threshold = max(0.20, P10 of H_cal reference QE), cap 40%, floor 5%, pooled offset -0.267 from 10 languages with references): keeps 83-95% on the specialist languages; for most the 5% floor binds (human references score lower than the data's low tail), so CAL drops about 5% where OLD drops 15%; amh bam tir nus have their own reference thresholds without binding. Samples of about 155 pairs give about +-3% on a threshold, so these are first looks, not the final numbers.

| iso | scored | median QE | thr OLD (P15) | keep OLD | own H_cal refs | thr CAL | CAL binding | keep CAL |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| amh | 157 | 0.7309 | 0.5629 | 0.8471 | 261 | 0.5565 | none | 0.8599 |
| bam | 157 | 0.4131 | 0.3034 | 0.8471 | 261 | 0.3134 | none | 0.828 |
| dyu | 157 | 0.6513 | 0.55 | 0.8471 | 261 | 0.4471 | floor5pct | 0.949 |
| ewe | 157 | 0.5841 | 0.4346 | 0.8471 | 0 | 0.3611 | floor5pct | 0.949 |
| fon | 157 | 0.6034 | 0.4871 | 0.8471 | 261 | 0.3936 | floor5pct | 0.949 |
| kab | 157 | 0.4633 | 0.362 | 0.8471 | 0 | 0.2778 | floor5pct | 0.949 |
| kam | 157 | 0.6737 | 0.5468 | 0.8471 | 261 | 0.4712 | floor5pct | 0.949 |
| kbp | 157 | 0.6237 | 0.508 | 0.8471 | 261 | 0.4477 | floor5pct | 0.949 |
| kea | 157 | 0.3805 | 0.2192 | 0.8471 | 0 | 0.2 | none | 0.8726 |
| kmb | 157 | 0.6652 | 0.5391 | 0.8471 | 0 | 0.4555 | floor5pct | 0.949 |
| umb | 156 | 0.6808 | 0.5513 | 0.8462 | 261 | 0.4666 | floor5pct | 0.9487 |
| vmw | 156 | 0.4539 | 0.336 | 0.8462 | 0 | 0.2789 | floor5pct | 0.9487 |
| wol | 156 | 0.5293 | 0.4333 | 0.8462 | 0 | 0.3343 | floor5pct | 0.9487 |
| yor | 156 | 0.6287 | 0.5087 | 0.8462 | 0 | 0.4389 | floor5pct | 0.9487 |
| luo | 155 | 0.5134 | 0.3457 | 0.8452 | 0 | 0.2484 | floor5pct | 0.9484 |
| mos | 155 | 0.6826 | 0.588 | 0.8452 | 261 | 0.5162 | floor5pct | 0.9484 |
| nus | 155 | 0.4361 | 0.3284 | 0.8452 | 261 | 0.297 | none | 0.9097 |
| seh | 155 | 0.6692 | 0.5638 | 0.8452 | 0 | 0.477 | floor5pct | 0.9484 |
| tir | 155 | 0.7211 | 0.5654 | 0.8452 | 261 | 0.4877 | none | 0.9226 |
| tum | 155 | 0.7059 | 0.6041 | 0.8452 | 0 | 0.529 | floor5pct | 0.9484 |

(Full 77-language table: `reports/data/sft_clean_2026-10-28/qe_cpu/keep_rate_per_language.csv`.)

## What the GPU must still score
All pairs except those scored here (23.77 M unique pairs in total minus the CPU shards), starting with `data/`, `data_devhalf/` and the 365,774 external `specialist/` rows (`--skip-keys-of data data_devhalf` for specialist). The CPU job keeps running until the GPU run takes over; pull `qe_scores_cpu/scores/` from the Hub (or `qe_cpu/scores_cpu/` on the box) before starting it.
