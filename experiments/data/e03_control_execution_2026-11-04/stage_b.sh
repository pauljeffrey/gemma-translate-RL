#!/bin/bash
# E-03 control data build, stage B: replay half (read-only ranged reads from the bucket). CPU/network only. No GPU, no bucket writes.
set -euo pipefail
ROOT=/workspace/clean_sft; WORK=/dev/shm/control; CODE=$ROOT/code; HERE=/dev/shm/control_prep
PY=$ROOT/venv_train/bin/python
export PYTHONPATH=/dev/shm/pylibs_boto3:$CODE:$CODE/sft_clean/chain GEMMA_MT_PKG=$CODE GEMMA_MT_TAG_MODE=single_token HF_HOME=$WORK/hf_home TMPDIR=$WORK/tmp
source /root/.bucket_env; export AK SK
mkdir -p $WORK/s6_meta $WORK/g0_meta $WORK/logs; cd $WORK
log(){ echo "$(date -u +%H:%M:%S) $*"; }
[ -f $WORK/.b0 ] || { log meta; nice -n 19 $PY - <<PYEOF
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
touch $WORK/.b0; }
[ -f $WORK/.b1 ] || { log cut_replay; nice -n 19 $PY $HERE/cut_replay.py --s6-meta $WORK/s6_meta --g0-meta $WORK/g0_meta --n-packs ${REPLAY_PACKS:-96600} --out $WORK/replay_sel --threads 8; touch $WORK/.b1; }
log stage_B_done
