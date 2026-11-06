#!/usr/bin/env python
"""CPU check that a yaml loads into RLConfig and passes validate_config; prints the differences against a second yaml/config.json.   python e_cfgcheck.py A.yaml [B.yaml|B.json]"""
import json, sys
sys.path.insert(0, "/workspace/clean_rerun/tree/gemma_mt")
from gemma_mt.rl.train_rl import RLConfig, dataclass_from_dict, load_yaml, validate_config
import dataclasses
def load(p):
    return json.load(open(p))["config"] if p.endswith(".json") else load_yaml(p)
a = load(sys.argv[1]); cfg = dataclass_from_dict(RLConfig, a); validate_config(cfg); print("OK", sys.argv[1], "steps", cfg.steps, "seed", cfg.seed)
if len(sys.argv) > 2:
    b = dataclasses.asdict(dataclass_from_dict(RLConfig, load(sys.argv[2]))); aa = dataclasses.asdict(cfg)
    for k in sorted(set(aa) | set(b)):
        if aa.get(k) != b.get(k):
            print("DIFF", k, "|", json.dumps(aa.get(k))[:160], "|", json.dumps(b.get(k))[:160])
