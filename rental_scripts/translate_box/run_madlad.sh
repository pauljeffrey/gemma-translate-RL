#!/bin/bash
source /venv/main/bin/activate; export PYTHONPATH=/workspace/translate/bench/code HF_HOME=/workspace/translate/hf TOKENIZERS_PARALLELISM=false
cd /workspace/translate/bench/code
exec python scripts/longdoc_translate.py --docs ../source_set_v3/core/docs.jsonl --out /workspace/translate/out/madlad --system madlad400-3b-mt --ct2-dir /workspace/translate/models/madlad400-3b-mt-ct2 --backend ct2 --device cuda --num-beams 4 --max-batch-tokens 3500 --max-batch-sents 96
