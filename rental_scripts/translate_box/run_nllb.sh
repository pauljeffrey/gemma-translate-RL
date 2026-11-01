#!/bin/bash
source /venv/main/bin/activate; export PYTHONPATH=/workspace/translate/bench/code HF_HOME=/workspace/translate/hf TOKENIZERS_PARALLELISM=false
cd /workspace/translate/bench/code
exec python scripts/longdoc_translate.py --docs ../source_set_v3/core/docs.jsonl --out /workspace/translate/out/nllb --system nllb200-3.3b --ct2-dir /workspace/translate/models/nllb-200-3.3B-ct2 --backend ct2 --device cuda --num-beams 4 --max-batch-tokens 3500 --max-batch-sents 96
