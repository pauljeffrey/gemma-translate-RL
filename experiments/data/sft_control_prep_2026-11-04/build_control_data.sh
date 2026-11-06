#!/bin/bash
# E-03 control: build the full packed dataset (replay half + TranslatePsy half) and the schedule. CPU only, NO GPU. NOT RUN IN FULL (the laptop built and tested a 1/60-scale version of every
# step, see reports/sft_control_prep_2026-11-04.md section 6). Run it on a machine that has >= 8 GB of free disk (the A100 box had 3.2 GB free on 2026-11-04: use /dev/shm there) and
# keep it NICE: `nice -n 19 taskset -c <few cores>` so that it cannot slow a running RL job (a CPU-heavy job starved the RL once).
#
# Inputs (all read-only): HF read token in $HFT (qvac dataset is public; the index repo is private), bucket keys in /root/.bucket_env (AK=, SK=) for the replay packs.
# Sizes: TranslatePsy stream about 6 GB of Hub traffic -> tp_sample 1.5 GB jsonl -> tp_clean -> tp_packed 2.0 GB; replay_sel 1.6 GB; combined (in place) 3.7 GB.
set -euo pipefail
ROOT=/workspace/clean_sft; WORK=${WORK:-/dev/shm/control}; CODE=${CODE:-$ROOT/code}
PY=${PY:-$ROOT/venv_train/bin/python}                       # needs: numpy pyarrow huggingface_hub transformers boto3 xxhash
HERE=$(cd "$(dirname "$0")" && pwd)                         # the folder with this script (tp_*.py, cut_replay.py, make_control_schedule.py, tp_shard_map.json)
export PYTHONPATH=$CODE:$CODE/sft_clean/chain GEMMA_MT_PKG=$CODE GEMMA_MT_TAG_MODE=single_token HF_HOME=$ROOT/hf_home
source /root/.bucket_env; export AK SK                      # the file defines AK / SK (see make_replay.py)
mkdir -p $WORK; cd $WORK
STEPS=${STEPS:-6000}; TP_PACKS_PER_LANG=${TP_PACKS_PER_LANG:-6400}

# 0. model + tokenizer of the SFT final (public Hub repo; the tokenizer has the ORIGINAL tag layout, fingerprint 69f7f26e...)
$PY - <<PYEOF
from huggingface_hub import snapshot_download
snapshot_download("${HF_ORG}/gemma-0.4B-translate-sft", local_dir="$WORK/sft_model", allow_patterns=["*.json","*.model","*.jinja","model.safetensors"])
PYEOF

# 1. protected index (private Hub dataset ${HF_ORG_LEGACY}/sabiyarn-clean-sft-2026-10-28/protected_index; sha256 pinned in the manifest, checked at load)
$PY - <<PYEOF
from huggingface_hub import snapshot_download
snapshot_download("${HF_ORG_LEGACY}/sabiyarn-clean-sft-2026-10-28", repo_type="dataset", local_dir="$WORK/pidx", allow_patterns=["protected_index/protected_index.npz","protected_index/protected_manifest.json","protected_index/protected_exemption_devhalf_v3.npz"])
PYEOF

# 2. TranslatePsy en->xx: 2 shards per language (first and middle), 200 record batches (200k rows) each = 400k rows per language (need 220k to 373k per language after filters, tp_rows_needed.json)
nice -n 19 $PY $HERE/tp_stream_sample.py $WORK/tp_sample 200 2

# 3. cheap filters + protected-index scan of both sides (about 16 min single process); rows with ANY hit are dropped; report in tp_clean/prepare_report.json
nice -n 19 $PY $HERE/tp_prepare.py --in $WORK/tp_sample --out $WORK/tp_clean --index $WORK/pidx/protected_index/protected_index.npz --manifest $WORK/pidx/protected_index/protected_manifest.json \
   --decon-dir $CODE/sft_clean --plant $WORK/flores_eng_devtest.jsonl      # plant file = FLORES+ devtest English (positive control: every sentence must be flagged); drop the flag if the file is absent
# 3b. (optional CPU) GlotLID acceptance of the targets, as R10's tp_build.py: drop rows whose top GlotLID label is not accepted for the language (LANGUAGES[iso].glotlid); wol 73%, mlg 79% pass in R10's sample
# 3c. (optional GPU, NOT part of the budget) QE gate: AfriCOMET-QE and SSA-COMET >= p25 of the human FLORES+ references (R10 thresholds in reports/data/weak_language_data_2026-10-22/translatepsy_analysis.csv); the dataset is already QE-filtered by its authors (unified z >= 0.68)

# 4. pack: the exact SFT command (data_pipeline/sft_clean/chain/pack.py prepare), one language per pack, 4,096 tokens, tag mode single_token, then patch meta (tag_mode / tag_map_sha256)
nice -n 19 $PY -m gemma_mt.data.prepare_sft --input $WORK/tp_clean --out $WORK/tp_packed --tokenizer $WORK/sft_model --seq-len 4096 --max-sample-tokens 4096 --directions en2x \
   --pack-by-language --val-fraction 0.002 --min-val-per-language 60 --workers 4
$PY $HERE/verify_tag_mode.py $WORK/sft_model $WORK/tp_clean $WORK/tp_packed
$PY -m gemma_mt.data.audit_packed --data $WORK/tp_packed --split train --check-samples 4000 --expect-single-language --decode 3

# 5. replay half: 96,000 packs (= STEPS x 16) UniMax-selected from stage-B shard 6 + the tail-language replay (ranged reads from the bucket, byte-identical to what shard 6 trained on)
mkdir -p $WORK/s6_meta $WORK/g0_meta
$PY - <<PYEOF
import os, boto3
from botocore.config import Config
s3 = boto3.client("s3", endpoint_url=os.environ['S3_ENDPOINT_URL'], aws_access_key_id=os.environ["AK"], aws_secret_access_key=os.environ["SK"], region_name=os.environ.get("S3_REGION", "us-east-1"), config=Config(retries={"max_attempts": 5}))
B = os.environ['S3_BUCKET']
for pre, dst in (("clean_rerun_2026-11-01/shards/s6/", "$WORK/s6_meta/"), ("clean_rerun_2026-11-01/g0/packed_g0/", "$WORK/g0_meta/")):
    os.makedirs(dst + "train", exist_ok=True)
    for k in ("direction", "iso_id", "pack_start", "prompt_len", "sample_len", "sample_offset", "target_len"):
        s3.download_file(B, pre + f"train/{k}.npy", dst + f"train/{k}.npy")
    s3.download_file(B, pre + "meta.json", dst + "meta.json")
PYEOF
nice -n 19 $PY $HERE/cut_replay.py --s6-meta $WORK/s6_meta --g0-meta $WORK/g0_meta --n-packs $((STEPS * 16)) --out $WORK/replay_sel --threads 16

# 6. combine: replay_sel first (its val/ = shard-6 validation split is kept), TranslatePsy packs appended; the TranslatePsy validation split becomes val_tp/
$PY $CODE/sft_clean/chain/extend_packed.py --base $WORK/replay_sel --add $WORK/tp_packed --out $WORK/combined --inplace
cp -r $WORK/tp_packed/val $WORK/combined/val_tp
rm -rf $WORK/tp_sample                                        # 1.5 GB

# 7. schedule: 16 replay + 16 TranslatePsy packs per step, UniMax inside each half (max 1 pass: no pack repeats), sidecar with the pack-structure hash
N_REPLAY=$((STEPS * 16))
$PY $HERE/make_control_schedule.py --packed-dir $WORK/combined --n-replay $N_REPLAY --steps $STEPS --out $WORK/sched_ctl
# 8. verify with the trainer's own checker (must list the six checks: schedule_sha256 steps per_step n_packs pack_lang_sha256 packed_structure_sha256)
$PY - <<PYEOF
import numpy as np
from gemma_mt.data.packed_dataset import PackedDataset
from gemma_mt.train_sft import load_schedule
ds = PackedDataset("$WORK/combined", "train", pad=False)
arr, info = load_schedule("$WORK/sched_ctl/schedule.npy", ds, 32)
print(arr.shape, info["checks_passed"], "unique packs", len(np.unique(arr)), "max draws per pack", int(np.bincount(arr.reshape(-1)).max()))
PYEOF
echo BUILD_CONTROL_DATA_DONE
