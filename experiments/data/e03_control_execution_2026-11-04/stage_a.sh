#!/bin/bash
# E-03 control data build, stage A (CPU, nice 19): model, index, TranslatePsy stream, clean+scan. Runs on the A100 box. No GPU.
set -euo pipefail
ROOT=/workspace/clean_sft; WORK=/dev/shm/control; CODE=$ROOT/code; HERE=/dev/shm/control_prep
PY=$ROOT/venv_train/bin/python
eval "$(grep '^export HF_TOKEN=' $ROOT/env.sh)"; export HFT=$HF_TOKEN HF_HOME=$WORK/hf_home TMPDIR=$WORK/tmp
export PYTHONPATH=/dev/shm/pylibs_boto3:$CODE:$CODE/sft_clean/chain GEMMA_MT_PKG=$CODE GEMMA_MT_TAG_MODE=single_token
mkdir -p $WORK/tmp $WORK/hf_home $WORK/logs; cd $WORK
log(){ echo "$(date -u +%H:%M:%S) $*"; }
[ -f $WORK/.a0 ] || { log model; nice -n 19 $PY - <<PYEOF
from huggingface_hub import snapshot_download
snapshot_download("${HF_ORG}/gemma-0.4B-translate-sft", local_dir="$WORK/sft_model", allow_patterns=["*.json","*.model","*.jinja","model.safetensors"])
snapshot_download("${HF_ORG_LEGACY}/sabiyarn-clean-sft-2026-10-28", repo_type="dataset", local_dir="$WORK/pidx", allow_patterns=["protected_index/protected_index.npz","protected_index/protected_manifest.json","protected_index/protected_exemption_devhalf_v3.npz"])
PYEOF
touch $WORK/.a0; }
[ -f $WORK/.a1 ] || { log stream; nice -n 19 $PY $HERE/tp_stream_sample.py $WORK/tp_sample 160 2; touch $WORK/.a1; }
[ -f $WORK/.a2 ] || { log clean; mkdir -p $WORK/tp_clean
  for G in "afr,amh,hau,ibo,kin,lin,lug" "mlg,nya,orm,sna,som,sot,swh" "tsn,wol,xho,yor,zul"; do
    n=$(echo $G | cut -d, -f1)
    nice -n 19 $PY $HERE/tp_prepare.py --in $WORK/tp_sample --out $WORK/tp_clean_$n --index $WORK/pidx/protected_index/protected_index.npz --manifest $WORK/pidx/protected_index/protected_manifest.json \
       --decon-dir $CODE/sft_clean --plant $HERE/flores_eng_devtest.jsonl --langs $G > $WORK/logs/tp_prepare_$n.log 2>&1 &
  done; wait; touch $WORK/.a2; }
log stage_A_done
