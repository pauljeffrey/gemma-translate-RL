#!/bin/bash
# one heartbeat line of the active run (read from /workspace/e2405/ACTIVE = "RUNDIR LOGFILE"): step, restarts and last exit codes of the supervisor, root + /dev/shm free, metrics age, GPU
read RUN LOG < /workspace/e2405/ACTIVE 2>/dev/null
python3 - "$RUN" "$LOG" <<'PY'
import json, os, re, subprocess, sys, time, glob
run, log = sys.argv[1], sys.argv[2]
def sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
step, age, kl, rew = -1, -1, "", ""
m = os.path.join(run, "metrics.jsonl")
if os.path.exists(m):
    age = int(time.time() - os.path.getmtime(m))
    tail = open(m, "rb").read()[-300000:].decode("utf-8", "ignore").splitlines()
    for l in reversed(tail):
        try: r = json.loads(l)
        except Exception: continue
        if "reward" in r: step = r["step"]; kl = round(r.get("kl", 0), 4); rew = round(r["reward"], 3); break
tag = os.path.basename(os.path.dirname(run)) if os.path.basename(run) in ("A", "final") else os.path.basename(run)
launch = glob.glob(f"/workspace/clean_rerun/logs_{tag.replace('runs_', '')}/launch_clean_rerun.log") + glob.glob("/workspace/clean_rerun/logs/launch_clean_rerun.log")
ex = []
for p in launch[:1]:
    ex = [l for l in open(p) if "trainer exited" in l or "OOM ->" in l]
rcs = [re.search(r"rc=(\d+)", l).group(1) if "rc=" in l else "OOM" for l in ex]
free = sh("df --output=avail -k / | tail -1"); shm = sh("df --output=avail -k /dev/shm | tail -1")
gpu = sh("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader")
print(f"{time.strftime('%H:%MZ', time.gmtime())} run={tag} step={step} reward={rew} kl={kl} restarts={len(ex)} last_rcs={','.join(rcs[-3:])} root_free_MB={int(free)//1024} shm_free_MB={int(shm)//1024} metrics_age_s={age} gpu={gpu}")
PY
