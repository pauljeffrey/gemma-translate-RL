#!/bin/bash
# wait for the list stage, then authors, then details (each resumable)
cd "$(dirname "$0")/.."
PY=.venv-comet/bin/python
while pgrep -f "crawl_hf.py list" >/dev/null; do sleep 20; done
$PY data-catalog/crawl_hf.py list            # completes any code the first run missed
$PY data-catalog/crawl_hf.py authors
$PY data-catalog/crawl_detail.py
echo CHAIN_DONE
