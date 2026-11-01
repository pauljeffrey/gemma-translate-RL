import os, sys, torch, time
tot = torch.cuda.get_device_properties(0).total_memory
torch.cuda.set_per_process_memory_fraction(7.5 * 2**30 / tot)   # hard cap 7.5 GiB (the coordinator allows this agent 8-12 GB of the card)
from gemma_mt.train_sft import main
t = time.time()
cfg = sys.argv[1]
rc = main(["--config", cfg, "out_dir=/workspace/jobs/r11/runs/smoke", "max_steps=4", "micro_batch_size=2", "grad_accum_steps=1", "gradient_checkpointing=true", "eval_every=1000", "eval_at_start=false",
           "log_every=1", "save_every=1000", "save_optimizer=false", "resume=false", "run_name=smoke_r11"] + ([] if os.environ.get("SMOKE_KEEP_VAL") else ["extra_val_splits=[]"]) + sys.argv[2:])
print("PEAK_ALLOCATED_GiB", torch.cuda.max_memory_allocated() / 2**30, "seconds", time.time() - t, flush=True)
sys.exit(rc)
