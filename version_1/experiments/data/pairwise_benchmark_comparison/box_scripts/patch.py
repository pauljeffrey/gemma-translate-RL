p="/workspace/jobs/e1/code2c/gemma_mt/eval/runtime.py"; s=open(p).read()
s=s.replace("seed=seed, extra=dict(extra or {}))\n    if engine_factory","seed=seed, extra=dict(extra or {}), max_num_batched_tokens=int(os.environ.get(\"E1_MNBT\", \"0\")) or None)\n    if engine_factory",1)
open(p,"w").write(s); print("E1_MNBT" in s)
