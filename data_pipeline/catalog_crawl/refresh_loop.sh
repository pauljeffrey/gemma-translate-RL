#!/bin/bash
# regenerate the CSVs from whatever has been fetched so far, every 2 minutes, until the crawl chain is done
cd "$(dirname "$0")/.."
while true; do
  .venv-comet/bin/python data-catalog/analyze.py >> data-catalog/african_hf/_logs/analyze.log 2>&1
  grep -q CHAIN_DONE data-catalog/african_hf/_logs/chain.out && break
  sleep 120
done
