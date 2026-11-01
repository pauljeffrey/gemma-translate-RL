# Gemma SFT run: monitoring handoff (written 2026-10-15 06:40 UTC)

Everything a separate agent needs to monitor the run. It is READ-ONLY monitoring; the owner decides about stopping.
No token or password is in this file and none is needed for monitoring.

> **UPDATE 2026-10-15 15:25 UTC:** the run is no longer `sft_long.yaml`. Since 08:29 UTC the trainer is `python -m gemma_mt.train_sft --config gemma_mt/configs/sft_unimax.yaml` (pid 510687; started by someone else, not the monitoring agent): out_dir `runs/sft_unimax`, `init_from` the sft_fwd checkpoint, Muon optimizer lr 4e-5, UniMax language sampling (3B-token budget, 1-2 epochs/language, priority weight 4 for 20 languages). Logs: `logs/sft_unimax.log`, `logs/sft_unimax_watchdog.log`. Full macro val at step 4750: 1.639 (pushed `step-4750`). Sections 1-4 below describe the earlier segment; use the new paths/config for monitoring and relaunch.

## 1. What is running
- Gemma-3-270m-it English -> African translation SFT (forward only, en->X, 77 target languages), `python -m gemma_mt.train_sft --config gemma_mt/configs/sft_long.yaml` (pid 490702 at 06:38 UTC).
- 16,827 steps in total (about 1 natural pass). ~22 s/step, ~6,100 tok/s, ~4.3 days end to end. Started the current segment from step 3000 (planned restart 2026-10-14 21:30 UTC).
- Data: `/workspace/sft_build/gemma/packed_fwd` (538,480 train packs of 4,096 tokens, one language per pack, val 1,234 packs). Loss only on `{iso} translation<end_of_turn>`.
- Sampling: stratified across languages (p ~ packs^0.5, floor 0.002, cap 4 epochs/language). lr 5e-5 cosine, micro 2 x accum 16, gradient checkpointing.
- Validation: FULL per-language validation every 250 steps (about every 1.5 h); QUICK validation (1 pack per language, noisy) every 30 steps. Macro = mean over languages; macro full val drives best / push / stop.
- Hub: only NEW-BEST full-eval checkpoints are pushed by the trainer to `<HF_ORG_LEGACY>/gemma3-270m-translate` (tags `step-250`, `step-500`, ...). One local resumable checkpoint (`runs/sft_fwd/last`) + the previous one (`last_prev`).

## 2. How to reach the vast instance
Key-only ssh (the key is already on the owner's laptop at `$SSH_KEY`). The direct route times out; use the proxy:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=25 -o ServerAliveInterval=30 -o StrictHostKeyChecking=accept-new \
    -i "$SSH_KEY" -p $BOX_PORT root@$BOX_HOST 'bash -s' < monitor.sh
```
- Feed remote commands from a script file (`bash -s < script.sh`). Do not use `pkill -f` / `pgrep -f` inline. Retry 2-3 times 20 s apart if the connection fails (vast sometimes needs a few seconds).
- Working directory on the box: `/workspace/sft_build`. The box is SHARED: another agent runs pretraining prep on the same CPUs; never touch anything that is not this run.

## 3. The monitor script (`monitor.sh`)
```bash
cd /workspace/sft_build
date -u +%FT%TZ
PY=$(ls /workspace/sft_build/.venv*/bin/python /venv/*/bin/python 2>/dev/null | head -1); [ -z "$PY" ] && PY=$(command -v python3)
$PY val_report.py 2>&1 | tail -n 100 | cut -c1-220     # latest step + train loss, QUICK and FULL per-language tables, macro histories
echo "--- log";     tail -n 5 logs/sft_fwd_part2.log | cut -c1-200
echo "--- gpu";     nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "--- run";     ls runs/sft_fwd; grep -a pushed logs/sft_fwd_part2.log | tail -n 1
echo "--- trainer"; pgrep -af train_sft | head -2
echo "--- disk";    df -h /workspace | tail -1
echo "--- stop";    ls runs/sft_fwd/STOP_PROPOSED 2>&1
```
Files: `runs/sft_fwd/metrics.jsonl` (all evaluations: train_loss, lr, grad_norm per 10 steps; quick_val_loss*, full val_loss*), `logs/sft_fwd_part2.log` (trainer stdout), `logs/sft_watchdog.log` (relaunch log), `val_report.py` (prints the tables).

## 4. Current values (2026-10-15 06:38 UTC, step 4460 / 16,827)
- Train loss ~1.4-1.6 (1.498 at step 4460), grad norm 1.85-2.2, lr 4.38e-05. No NaN, no spikes. GPU 100% (11.1 GB), disk 69%.
- FULL macro val loss, every 250 steps: 250:4.191, 500:3.361, 750:2.864, 1000:2.586, 1250:2.395, 1500:2.252, 1750:2.151, 2000:2.054, 2250:1.992, 2500:1.928, 2750:1.871, 3000:1.832, 3250:1.792, 3500:1.756, 3750:1.725, 4000:1.697, **4250:1.675 (best, pushed as `step-4250`)**. The step-4500 full evaluation is due at about 06:55 UTC.
- QUICK macro val: 4230:1.762, 4260:1.760, 4290:1.758, 4320:1.755, 4350:1.751, 4380:1.750, 4410:1.749, **4440:1.747**.
- FULL per-language val loss at step 4250 (macro 1.675):

```
ach 1.83  afr 1.24  amh 2.16  bam 2.52  bas 1.91  bbj 2.20  bem 2.24  bin 2.12  dag 1.89  dua 1.64
dyu 1.98  efi 1.29  ewe 1.86  fon 1.19  hau 1.34  her 2.31  ibb 1.15  ibo 1.53  ish 1.96  iso 1.37
kab 2.11  kam 1.55  kbp 1.72  kea 3.43  kik 1.30  kin 1.57  kmb 1.54  kon 2.65  kqn 1.53  kri 1.29
kua 1.62  lin 1.50  loz 1.46  lua 1.53  lug 1.69  lun 1.69  luo 2.18  mlg 1.77  mos 1.67  nbl 1.74
ndc 1.75  nso 1.25  nus 0.23  nya 1.38  nyn 1.60  oke 1.51  orm 2.19  pcm 1.30  run 1.75  sag 1.40
seh 1.39  shi 1.65  sid 2.35  sna 1.64  som 2.09  sot 1.12  ssw 1.71  swc 1.63  swh 1.65  tir 1.83
tiv 1.88  tsc 1.36  tsn 1.22  tso 1.30  tum 1.60  twi 0.68  umb 1.57  urh 1.61  ven 1.05  vmw 1.95
wal 1.92  wes 1.52  wol 2.49  xho 1.52  yao 1.79  yor 1.36  zul 1.48
```
- Known outlier: `kea` (Cape Verdean Creole, ~4k pairs) rises on every full evaluation (2.82 -> 2.98 -> 3.18 -> 3.26 -> 3.43), probably memorisation. It does not hurt the macro trend. Capping its draws is only possible at the next restart. Mention it to the owner at the next restart/stop decision.
- Expect: full macro improving about 0.02 per 250 steps at this point; this will slow down.

## 5. Report format the owner wants
Simple: step, current train loss, macro val loss (last full evaluations + last quick), and the per-language val loss table ONLY when a new FULL evaluation has arrived (list the languages that rose). One status line: healthy / WARNING + reason.

## 6. When to raise a WARNING (describe it with numbers; do not act)
- trainer not running (no new step for > 20 min and GPU idle) -> see section 7.
- NaN/inf in the log, train loss > 2x its recent average, `push FAILED` in the log, disk > 90%.
- `runs/sft_fwd/STOP_PROPOSED` exists: the trainer writes it when 3 full evaluations in a row gave no new best. The trainer KEEPS training; the owner decides.
- full macro val rose on two consecutive full evaluations.

## 7. Rules (owner decisions)
- NEVER kill, stop, pause or signal the trainer without telling the owner first and waiting for an answer ("before you terminate the sft run for any reason, let me know first").
- Only a CRASHED trainer may be relaunched without asking (exact resume): fix the cause first (token file `/workspace/sft_build/.hf_write_token`, disk, code), then `cd /workspace/sft_build && python -m gemma_mt.train_sft --config gemma_mt/configs/sft_long.yaml` and restart `sft_watchdog.sh` (it relaunches a crashed trainer, max 3/h). A trainer that is merely slow is not crashed.
- Do not push anything to the Hub by hand (the trainer pushes its own new bests). Do not copy data to the laptop (metric reports are fine). Do not start RL, benchmarks or any other GPU job: the owner reviews SFT benchmark results BEFORE RL starts.
- After the owner agrees to stop: benchmark the best Hub checkpoint and the base `gemma-3-270m-it` en->X on FLORES+ (direct jsonl, orm = gaz_Latn, efi none), NTREX-128 and MAFAND with `gemma_mt.eval.eval_translation` (NFC-normalised chrF++ / BLEU / AfriCOMET), apply the English-sentence contamination filter (`bench/bench_dev_clean.jsonl` logic), report clean-subset numbers.
