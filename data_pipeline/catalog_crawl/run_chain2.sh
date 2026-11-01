#!/bin/bash
cd "$(dirname "$0")/.."
PY=.venv-comet/bin/python
$PY data-catalog/crawl_hf.py authors
$PY data-catalog/crawl_detail.py
echo CHAIN_DONE
