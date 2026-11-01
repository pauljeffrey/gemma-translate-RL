#!/bin/bash
# RTX 3090 translation box (flaky network: every network step is retried): env + model downloads + CTranslate2 conversion for the long-document benchmark.
mkdir -p /workspace/translate/{models,src,out,logs,code}
LOG=/workspace/translate/logs/progress.log
say() { echo "$(date -u +%H:%M:%S) [setup] $*" | tee -a $LOG; }
source /venv/main/bin/activate
export HF_HOME=/workspace/translate/hf HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false UV_HTTP_TIMEOUT=120
retry() { local n=$1; shift; local i=1; until "$@"; do [ $i -ge $n ] && return 1; say "  retry $i/$n after failure: $*" ; i=$((i+1)); sleep 15; done; }
say "installing python packages (torch, transformers, ctranslate2, sacrebleu, ...)"
retry 40 uv pip install -q torch transformers sentencepiece ctranslate2 sacrebleu "huggingface_hub" pysbd nltk pyarrow pandas protobuf accelerate tqdm >> $LOG 2>&1 || { say "pip install FAILED permanently"; exit 1; }
python - <<'PY' 2>&1 | tee -a $LOG
import torch, ctranslate2, transformers
print("[setup] torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", "| ctranslate2", ctranslate2.__version__, "| ct2 cuda devices:", ctranslate2.get_cuda_device_count(), "| transformers", transformers.__version__)
PY
cat > /workspace/translate/dl.py <<'PY'
import sys
from huggingface_hub import snapshot_download
rid, out = sys.argv[1], sys.argv[2]
p = snapshot_download(rid, local_dir=out, allow_patterns=["*.json","*.model","*.txt","*.safetensors","*.bin","*.spm","*.py"], ignore_patterns=["*.msgpack","*.h5","*.ot","tf_model*","flax_model*","*.onnx","*.gguf"])
print("[setup] downloaded", rid, "->", p)
PY
get() {  # repo, hf-dir, ct2-dir, label
  say "downloading $1 ..."
  retry 60 python /workspace/translate/dl.py "$1" "$2" >> $LOG 2>&1 || { say "download of $1 FAILED permanently"; return 1; }
  say "converting $4 to CTranslate2 int8_float16 ..."
  retry 3 ct2-transformers-converter --model "$2" --output_dir "$3" --quantization int8_float16 --force >> $LOG 2>&1 || { say "conversion of $4 FAILED"; return 1; }
  say "$4 ready: $(du -sh "$3" | cut -f1)"
}
get facebook/nllb-200-3.3B /workspace/translate/models/nllb-200-3.3B-hf /workspace/translate/models/nllb-200-3.3B-ct2 NLLB-3.3B
get google/madlad400-3b-mt /workspace/translate/models/madlad400-3b-mt-hf /workspace/translate/models/madlad400-3b-mt-ct2 MADLAD-3B
say "SETUP DONE"
touch /workspace/translate/SETUP_DONE
