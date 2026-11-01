#!/bin/bash
# New RTX PRO 6000 box: rebuild the RL training environment + restore data/checkpoint from the object-storage bucket / Hub (the old rental went away at 23:03 UTC).
W=/workspace; S=$W/sft_build; mkdir -p $S/{logs,runs,gemma,bench,calibration,hf/hub,vllm_cache} $W/tmp
LOG=$S/logs/bootstrap.log
say() { echo "$(date -u +%T) $*" | tee -a $LOG; }
source $S/env.sh; source $S/.s3_env
export UV_HTTP_TIMEOUT=180 UV_LINK_MODE=copy
retry() { local n=$1; shift; local i=1; until "$@"; do [ $i -ge $n ] && return 1; say "  retry $i/$n: $*"; i=$((i+1)); sleep 10; done; }
say "START bootstrap"
retry 5 uv pip install -q --python /venv/main/bin/python huggingface_hub >> $LOG 2>&1
( retry 5 uv venv --python 3.12 $W/venv_vllm && retry 8 uv pip install -q --python $W/venv_vllm/bin/python vllm==0.31.0 --torch-backend cu130 && touch $S/.ok_venv_vllm && say "OK venv_vllm" ) > $S/logs/venv_vllm.log 2>&1 &
( retry 5 uv venv --python 3.12 $W/venv_gemma && retry 8 uv pip install -q --python $W/venv_gemma/bin/python torch "transformers==5.19.0" pyyaml pyarrow pandas numpy sentencepiece safetensors jinja2 huggingface_hub tokenizers tqdm requests accelerate psutil pytest && say "OK venv_gemma core" \
  && retry 5 uv pip install -q --python $W/venv_gemma/bin/python --target $S/pylibs --no-deps sacrebleu unbabel-comet==2.2.7 pytorch_lightning torchmetrics entmax jsonargparse lightning_utilities scipy portalocker tabulate colorama regex lxml structlog python-dotenv fasttext-wheel pybind11 pysbd iniconfig pluggy pygments certifi charset-normalizer idna urllib3 packaging protobuf sentencepiece \
  && retry 5 uv pip install -q --python $W/venv_gemma/bin/python --target $S/pylibs_tf4 "transformers==4.57.1" "huggingface_hub<1.0" "tokenizers>=0.22,<0.23" && touch $S/.ok_venv_gemma && say "OK venv_gemma + pylibs + pylibs_tf4" ) > $S/logs/venv_gemma.log 2>&1 &
# data from the bucket
( cd $W/tmp
  for f in ckpt/rl2c_resume_state data/gemma_rl_prompts_v2 small/rl_archives_snapshot_latest small/code_scripts_logs_bench small/run_metadata; do
    retry 8 rclone copyto remote:${S3_BUCKET}/gemma_mt/$f.tar.zst $W/tmp/$(basename $f).tar.zst --s3-upload-concurrency 4 --transfers 4 && say "downloaded $f"
  done
  ( zstd -dc $W/tmp/rl2c_resume_state.tar.zst | tar -xf - -C $S && say "extracted resume state" ) &
  ( zstd -dc $W/tmp/gemma_rl_prompts_v2.tar.zst | tar -xf - -C $S && say "extracted prompts v2" ) &
  ( mkdir -p $W/tmp/small && zstd -dc $W/tmp/code_scripts_logs_bench.tar.zst | tar -xf - -C $W/tmp/small bench calibration gemma_calibration 2>/dev/null; cp -r $W/tmp/small/bench/. $S/bench/; cp -rn $W/tmp/small/gemma_calibration $S/ 2>/dev/null; say "extracted bench" ) &
  ( mkdir -p $S/runs/rl2c/archive && zstd -dc $W/tmp/rl_archives_snapshot_latest.tar.zst | tar -xf - -C $W/tmp && cp $W/tmp/arch_snap/archive_rl2c.sqlite $S/runs/rl2c/archive/archive.sqlite && say "restored rl2c archive" ) &
  ( zstd -dc $W/tmp/run_metadata.tar.zst | tar -xf - -C $W/tmp 2>/dev/null; cp $W/tmp/runs/rl2c/metrics.jsonl $S/runs/rl2c/metrics.jsonl 2>/dev/null; say "restored metrics" ) &
  wait; touch $S/.ok_bucket ) > $S/logs/bucket.log 2>&1 &
# models from the Hub
( cat > $W/tmp/dl.py <<'PY'
import os
from huggingface_hub import snapshot_download
T=os.environ["HF_TOKEN"]; H="/workspace/sft_build/hf/hub"
snapshot_download("${HF_ORG}/gemma-3-270m-translate-rl2b", revision="stage-b64", local_dir="/workspace/sft_build/runs/rl_2b_stage_b64", token=T); print("ok stage-b64")
snapshot_download("cis-lmu/glotlid", revision="85cd6716494360367b75f642b5bc78667605d0b4", cache_dir=H, allow_patterns=["model_v3.bin","*.json"], token=T); print("ok glotlid")
snapshot_download("masakhane/africomet-qe-stl-1.1", cache_dir=H, token=T); print("ok africomet")
snapshot_download("Davlan/afro-xlmr-large-76L", cache_dir=H, allow_patterns=["*.json","*.model","*.txt","sentencepiece*"], token=T); print("ok afro-xlmr config")
PY
  retry 10 /venv/main/bin/python $W/tmp/dl.py ) > $S/logs/hub.log 2>&1 &
wait
say "ALL BACKGROUND JOBS FINISHED"
