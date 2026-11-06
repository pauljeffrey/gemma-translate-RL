ROOT=/workspace/clean_sft; WORK=/dev/shm/control; CODE=$ROOT/code; HERE=/dev/shm/control_prep
PY=$ROOT/venv_train/bin/python
eval "$(grep '^export HF_TOKEN=' $ROOT/env.sh)"; export HFT=$HF_TOKEN HF_HOME=$WORK/hf_home TMPDIR=$WORK/tmp
export PYTHONPATH=/dev/shm/pylibs_boto3:$CODE:$CODE/sft_clean/chain GEMMA_MT_PKG=$CODE GEMMA_MT_TAG_MODE=single_token
log(){ echo "$(date -u +%H:%M:%S) $*"; }
