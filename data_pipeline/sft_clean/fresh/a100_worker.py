#!/usr/bin/env python3
"""A100 hand-over worker (owner 2026-10-31). Runs on the A40 box. Moves A0-up6 (and the next arms) to the A100 when -- and only when -- the other agent that is using the A100 is done.

Rules (the A100 belongs to another agent right now; its files, processes and the GPU are NEVER touched):
  * everything of ours lives in ONE folder on the A100: /workspace/work (created by us; if it exists and has no .sabiyarn_owner marker the worker stops);
  * STAGING (copy code, schedules, packs, start checkpoint) only while  free_disk - item_size >= RESERVE_BUSY_GB  (so the other agent never runs out of room because of us);
  * GPU idle = utilisation <= 5 % and GPU memory <= 4 GB, continuously for 15 minutes;  storage emptied = free_disk + our_own_usage >= NEED_GB;
  * CUTOVER (only when both hold): install environments, 40-step speed probe on the A100 (must not be slower than the A40 s/step), copy the newest A0 checkpoint (model + optimizer),
    start A0 on the A100 from it, stop A0 on the A40, start the arm queue (G0, S1) on the A40 so that neither GPU idles;
  * every decision is logged to a100_worker/worker.log; important events are appended to ALERT.txt ([A100_*] tags).
    python3 a100_worker.py            # run forever (nohup setsid)      --dry-run : only probe and print what it would do
"""
import json, os, re, shlex, statistics, subprocess, sys, time

ROOT = "/workspace/work"; W = f"{ROOT}/a100_worker"; STATE_F = f"{W}/state.json"; LOG_F = f"{W}/worker.log"; ALERT_F = f"{ROOT}/ALERT.txt"
HOST, PORT, KEY = "root@" + os.environ["GPU_BOX_HOST"], os.environ["GPU_BOX_PORT"], os.path.expanduser(os.environ["SSH_KEY"])
AR = "/workspace/work"                                           # on the A100
SSH = ["ssh", "-i", KEY, "-o", "IdentitiesOnly=yes", "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=15", "-p", PORT, HOST]
RSYNC_E = f"ssh -i {KEY} -o IdentitiesOnly=yes -o LogLevel=ERROR -o ServerAliveInterval=15 -p {PORT}"
GB = 1024 ** 3
RESERVE_BUSY_GB, RESERVE_IDLE_GB, NEED_GB, IDLE_MIN, SPEEDUP_MIN = 3.0, 1.0, 30.0, 15, 1.0     # owner 2026-10-31: the move happens unless the A100 is SLOWER than the A40
DRY = "--dry-run" in sys.argv

# staging items: (name, local path on the A40, destination on the A100, approx GB) -- smallest first
ITEMS = [
    ("code", f"{ROOT}/code/", f"{AR}/code/", 0.05),
    ("sched_ext", f"{ROOT}/work/sched_ext/", f"{AR}/work/sched_ext/", 0.05),
    ("ckpt_up6", f"{ROOT}/ckpt/gemma3-270m-up6/", f"{AR}/ckpt/gemma3-270m-up6/", 1.3),
    ("packed_ext", f"{ROOT}/work/packed_ext/", f"{AR}/work/packed_ext/", 6.2),
]


def log(msg):
    line = f"{time.strftime('%F %T')} {msg}"
    print(line, flush=True)
    with open(LOG_F, "a") as f:
        f.write(line + "\n")


def alert(tag, msg):
    log(f"ALERT [{tag}] {msg}")
    with open(ALERT_F, "a") as f:
        f.write(f"{time.strftime('%FT%TZ', time.gmtime())} [{tag}] {msg}\n")


def sh(cmd, timeout=120):
    try:
        p = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def a100(cmd, timeout=120):
    return sh(SSH + [cmd], timeout)


def load_state():
    try:
        return json.load(open(STATE_F))
    except Exception:
        return {"done_items": [], "idle_since": None, "phase": "staging", "cutover": {}}


def save_state(s):
    json.dump(s, open(STATE_F, "w"), indent=1)


def probe():
    rc, out = a100(f"nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -n1; df -B1 --output=used,avail / | tail -n1; du -sb {AR} 2>/dev/null | cut -f1; test -d {AR} && echo DIR_EXISTS; test -f {AR}/.sabiyarn_owner && echo OWNER_OK; true")
    if rc != 0:
        return None
    L = [l.strip() for l in out.strip().splitlines() if l.strip() and "Welcome" not in l and "Have fun" not in l and "AI agents" not in l]
    try:
        util, mem = [int(x) for x in L[0].replace(" ", "").split(",")]
        used, avail = [int(x) for x in L[1].split()]
        mine = int(L[2]) if len(L) > 2 and re.fullmatch(r"\d+", L[2]) else 0
        return {"util": util, "mem_mib": mem, "used": used, "avail": avail, "mine": mine, "marker": "OWNER_OK" in out, "dir_exists": "DIR_EXISTS" in out}
    except Exception as e:
        log(f"probe parse failed: {e}: {out[:200]!r}")
        return None


def emptied(p):
    return (p["avail"] + p["mine"]) / GB >= NEED_GB


def a40_rate():
    """median s/step of the A40's A0 run over the last log intervals"""
    try:
        rows = [json.loads(l) for l in open(f"{ROOT}/run/runs/a0u_up6/metrics.jsonl") if '"train_loss"' in l][-12:]
        d = [(rows[i]["time"] - rows[i - 1]["time"]) / (rows[i]["step"] - rows[i - 1]["step"]) for i in range(1, len(rows)) if rows[i]["step"] > rows[i - 1]["step"]]
        return statistics.median(d) if d else None
    except Exception:
        return None


def stage_item(item, p, idle15):
    name, src, dst, gb = item
    reserve = RESERVE_IDLE_GB if idle15 else RESERVE_BUSY_GB
    if p["avail"] / GB - gb < reserve:
        return False
    log(f"staging {name} ({gb} GB) -> {dst}")
    if DRY:
        return False
    a100(f"mkdir -p {dst}")
    rc, out = sh(["rsync", "-a", "--partial", "--inplace", "--timeout=180", "-e", RSYNC_E, src, f"{HOST}:{dst}"], timeout=3 * 3600)
    if rc != 0:
        log(f"rsync {name} failed rc={rc}: {out[-300:]}")
        return False
    rc, out = sh(["rsync", "-a", "-n", "-i", "-e", RSYNC_E, src, f"{HOST}:{dst}"], timeout=1800)       # verify: nothing left to transfer
    if rc == 0 and not out.strip():
        log(f"staged {name} (verified)")
        return True
    log(f"verify {name} found differences: {out[:200]!r}")
    return False


# ------------------------------------------------------------------------------------------------ cutover
def put_text(path, text):
    import base64
    b = base64.b64encode(text.encode()).decode()
    return a100(f"echo '{b}' | base64 -d > {path}")


def envs_ready():
    rc, out = a100(f"test -f {AR}/state_env_train.done && echo yes")
    return "yes" in out


def install_envs():
    script = f"""#!/bin/bash
cd {AR}
nice -n 10 bash -c "uv venv --python 3.12 venv_train -q && uv pip install -q --python venv_train/bin/python torch --torch-backend=auto && uv pip install -q --python venv_train/bin/python 'transformers>=4.53,<5' 'numpy<2' pyyaml 'pyarrow<20' sacrebleu 'huggingface_hub<1' sentencepiece safetensors jinja2 xxhash && venv_train/bin/python -c 'import torch;assert torch.cuda.is_available();print(torch.__version__)' && touch state_env_train.done" > logs/env_train.log 2>&1
"""
    a100(f"mkdir -p {AR}/logs {AR}/a100_scripts && touch {AR}/.sabiyarn_owner")
    put_text(f"{AR}/a100_scripts/install_env.sh", script)
    a100(f"cd {AR} && nohup setsid bash a100_scripts/install_env.sh > /dev/null 2>&1 < /dev/null &")
    log("environment install started on the A100 (venv_train)")


def probe_speed():
    """40-step training speed on the A100 (micro-batch 4 and 8), compared with the A40's"""
    res = {}
    for mb, ga in ((4, 8), (8, 4)):
        body = f"""#!/bin/bash
cd {AR}/run; export HF_HOME={AR}/hf_home PYTHONPATH={AR}/code PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
rm -rf runs/_probe{mb}
timeout 900 {AR}/venv_train/bin/python -m gemma_mt.train_sft --config configs/clean_rerun_sft/a0u_up6.yaml out_dir=runs/_probe{mb} schedule_file=sched_ext/probe40/schedule.npy max_steps=0 resume=false eval_every=1000000 save_every=1000000 quick_eval_every=0 log_every=5 hub_push_every=1000000 eval_max_packs=8 micro_batch_size={mb} grad_accum_steps={ga} early_stopping_patience=0 run_name=probe{mb} > ../logs/probe{mb}.log 2>&1
"""
        put_text(f"{AR}/a100_scripts/probe{mb}.sh", body)
        a100(f"bash {AR}/a100_scripts/probe{mb}.sh", timeout=1000)
        rc, out = a100(f"cat {AR}/run/runs/_probe{mb}/metrics.jsonl 2>/dev/null")
        try:
            rows = [json.loads(l) for l in out.splitlines() if l.startswith("{") and '"train_loss"' in l]
            d = [(rows[i]["time"] - rows[i - 1]["time"]) / (rows[i]["step"] - rows[i - 1]["step"]) for i in range(2, len(rows))]
            res[mb] = statistics.median(d)
        except Exception as e:
            res[mb] = None
            log(f"probe mb={mb} failed: {e}; tail: {a100(f'tail -n 3 {AR}/logs/probe{mb}.log')[1][-300:]}")
        log(f"A100 probe micro-batch {mb}: {res[mb]} s/step")
    return res


def cutover(s, p):
    c = s["cutover"]
    # 1. environment
    if not c.get("env_started"):
        install_envs(); c["env_started"] = True; save_state(s)
    if not envs_ready():
        log("waiting for the A100 environment install"); return
    # 2. secrets + caches: only now (the other agent has left)
    if not c.get("env_sh"):
        envsh = open(f"{ROOT}/env.sh").read()
        envsh = re.sub(r"/workspace/work", AR, envsh)
        put_text(f"{AR}/env.sh", envsh); a100(f"chmod 600 {AR}/env.sh")
        a100(f"cd {AR} && source env.sh && HF_HOME={AR}/hf_home {AR}/venv_train/bin/python -c \"from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('google/gemma-3-270m-it'); print('tok ok')\"")
        a100(f"mkdir -p {AR}/run {AR}/logs; cd {AR}/run; ln -sfn ../work/packed_ext packed_ext; ln -sfn ../work/sched_ext sched_ext; ln -sfn ../ckpt ckpt; ln -sfn ../code/gemma_mt_root/configs configs")
        c["env_sh"] = True; save_state(s)
    # 3. speed probe (needs the probe schedule)
    if not c.get("probe"):
        sh(f"cd {ROOT}/run && source ../env.sh && PYTHONPATH={ROOT}/code:{ROOT}/code/sft_clean/chain {ROOT}/venv_data/bin/python {ROOT}/code/sft_clean/chain/sampler.py head --schedule {ROOT}/work/sched_ext/A0U --out {ROOT}/work/sched_ext/probe40 --steps 40 --name probe40", timeout=300)
        sh(["rsync", "-a", "-e", RSYNC_E, f"{ROOT}/work/sched_ext/probe40/", f"{HOST}:{AR}/work/sched_ext/probe40/"])
        res = probe_speed()
        ok = [v for v in res.values() if v]
        mine = a40_rate()
        c["probe"] = {"a100": res, "a40": mine}; save_state(s)
        if not ok or not mine:
            alert("A100_PROBE_FAILED", f"speed probe incomplete: {res}, A40 {mine}"); s["phase"] = "halted"; save_state(s); return
        best = min(ok); ratio = mine / best
        log(f"speed: A40 {mine:.2f} s/step vs A100 best {best:.2f} s/step = {ratio:.2f}x")
        c["best_mb"] = [mb for mb, v in res.items() if v == best][0]; c["ratio"] = ratio; save_state(s)
        if ratio < SPEEDUP_MIN:
            alert("A100_NOT_FASTER", f"A100 {best:.2f} s/step vs A40 {mine:.2f}: not {SPEEDUP_MIN}x faster; A0 stays on the A40"); s["phase"] = "halted"; save_state(s); return
    # 4. snapshot of the newest checkpoint -> A100 (the A40 keeps training meanwhile)
    if not c.get("snap_sent"):
        snap = f"{ROOT}/snap"; os.makedirs(snap, exist_ok=True)
        src = f"{ROOT}/run/runs/a0u_up6/last/trainer_state.pt"
        age = time.time() - os.path.getmtime(src)
        if age < 45:
            time.sleep(50)
        rc, out = sh(f"cp {src} {snap}/a0u_last.pt && cp {ROOT}/run/runs/a0u_up6/metrics.jsonl {snap}/metrics.jsonl && sha256sum {snap}/a0u_last.pt", timeout=900)
        c["snap_sha"] = out.split()[0] if rc == 0 else None
        rc3, out3 = sh(f"{ROOT}/venv_train/bin/python -c \"import torch;ck=torch.load('{snap}/a0u_last.pt',map_location='cpu',weights_only=False);print('STEP',ck['state']['step'],'optimizer' in ck)\"", timeout=900)
        m3 = re.search(r"STEP (\d+) True", out3)
        if rc3 != 0 or not m3:
            alert("A100_SNAPSHOT_UNREADABLE", f"the checkpoint copy does not load (rc {rc3}): {out3[-200:]}"); return
        c["snap_step"] = int(m3.group(1)) if m3 else None
        a100(f"mkdir -p {AR}/run/runs/a0u_up6/last")
        rc, out = sh(["rsync", "-a", "--partial", "--inplace", "--timeout=180", "-e", RSYNC_E, f"{snap}/a0u_last.pt", f"{HOST}:{AR}/run/runs/a0u_up6/last/trainer_state.pt"], timeout=3 * 3600)
        rc2, out2 = a100(f"sha256sum {AR}/run/runs/a0u_up6/last/trainer_state.pt", timeout=900)
        if rc != 0 or not c["snap_sha"] or c["snap_sha"] not in out2:
            alert("A100_SNAPSHOT_FAILED", f"checkpoint copy not verified ({rc}, {out2[:80]})"); return
        for src_fn, fn in (("config.json", "config.json"), ("schedule_manifest.json", "schedule_manifest.json")):
            sh(["rsync", "-a", "-e", RSYNC_E, f"{ROOT}/run/runs/a0u_up6/{src_fn}", f"{HOST}:{AR}/run/runs/a0u_up6/{fn}"])
        sh(["rsync", "-a", "-e", RSYNC_E, f"{snap}/metrics.jsonl", f"{HOST}:{AR}/run/runs/a0u_up6/metrics.jsonl"])      # the log as of the snapshot (rows past the checkpoint step are re-done on the A100)
        c["snap_sent"] = True; save_state(s)
        log("A0 checkpoint copied to the A100 and verified")
    # 5. last idle check, then switch
    q = probe()
    if not q or q["util"] > 5 or q["mem_mib"] > 4096:
        log("A100 busy again right before the switch; waiting"); s["idle_since"] = None; save_state(s); return
    mb = c.get("best_mb", 4); ga = 32 // mb
    launch = f"""#!/bin/bash
cd {AR}/run; source {AR}/env.sh
export HF_HOME={AR}/hf_home PYTHONPATH={AR}/code PY={AR}/venv_train/bin/python PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HUB_NO_MODEL_CARD=1
nohup setsid bash {AR}/code/sft_clean/fresh/sft_concurrent.sh "a0u_up6=configs/clean_rerun_sft/a0u_up6.yaml,micro_batch_size={mb},grad_accum_steps={ga}" > {AR}/logs/a0u_up6_launch.log 2>&1 &
"""
    put_text(f"{AR}/a100_scripts/launch_a0.sh", launch)
    rc, out = sh("pgrep -f '[g]emma_mt.train_sft --config configs/clean_rerun_sft/a0u_up6.yaml'")
    pids = out.split()
    for pid in pids:
        sh(f"kill {pid}")
    log(f"A0 stopped on the A40 (pids {pids}); starting it on the A100 from the verified checkpoint")
    time.sleep(10)
    running = a100("pgrep -f '[g]emma_mt.train_sft --config' | wc -l")[1].strip().splitlines()[-1:]
    if running and running[0].isdigit() and int(running[0]) > 0:
        log("a training process already runs on the A100 (a previous attempt): not launching a second one")
    else:
        a100(f"bash {AR}/a100_scripts/launch_a0.sh")
    ok = False
    for _ in range(40):
        time.sleep(15)
        rc, out = a100(f"tr '\\r' '\\n' < {AR}/run/logs/sft_a0u_up6.log 2>/dev/null | grep -a 'resumed from step\\|step [0-9]*/5625' | tail -n 2")
        if "resumed from step" in out and re.search(r"step \d+/5625", out):
            ok = True; break
    if not ok:
        alert("A100_START_FAILED", f"A0 did not start on the A100: {a100(f'tail -n 4 {AR}/run/logs/sft_a0u_up6.log')[1][-300:]}; restarting A0 on the A40 from its own checkpoint")
        sh(f"cd {ROOT}/run && source ../env.sh && export HF_HOME={ROOT}/hf_home PYTHONPATH={ROOT}/code PY={ROOT}/venv_train/bin/python HUB_NO_MODEL_CARD=1 && nohup setsid bash {ROOT}/code/sft_clean/fresh/sft_concurrent.sh 'a0u_up6=configs/clean_rerun_sft/a0u_up6.yaml' > {ROOT}/logs/a0u_up6_relaunch_a40.log 2>&1 < /dev/null &")
        s["phase"] = "halted"; save_state(s); return
    c["a0_on_a100"] = True; save_state(s)
    sh(f"mv {ROOT}/run/runs/a0u_up6 {ROOT}/run/runs/moved_to_a100_a0u_up6")                    # so that the A40 watchdog does not call the deliberate stop a crash
    # 6. A40 queue: G0, then S1 (up6-based) so that the A40 never idles
    q = f"""#!/bin/bash
cd {ROOT}/run; source ../env.sh
export HF_HOME={ROOT}/hf_home PYTHONPATH={ROOT}/code PY={ROOT}/venv_train/bin/python PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HUB_NO_MODEL_CARD=1
for arm in g0_up6 s1_up6; do
  echo "$(date +%T) start $arm" >> {ROOT}/logs/a40_arms_queue.log
  bash {ROOT}/code/sft_clean/fresh/sft_concurrent.sh "$arm=configs/clean_rerun_sft/$arm.yaml" > {ROOT}/logs/${{arm}}_launch.log 2>&1
  rc=$?; echo "$(date +%T) end $arm rc=$rc" >> {ROOT}/logs/a40_arms_queue.log
  [ $rc -eq 0 ] || {{ echo "$(date -u +%FT%TZ) [A40_ARM_FAILED] $arm" >> {ROOT}/ALERT.txt; exit 1; }}
done
echo "$(date -u +%FT%TZ) [A40_ARMS_DONE] g0_up6 and s1_up6 finished" >> {ROOT}/ALERT.txt
"""
    open(f"{ROOT}/a40_arms_queue.sh", "w").write(q)
    sh(f"cd {ROOT} && nohup setsid bash a40_arms_queue.sh > logs/a40_arms_queue.out 2>&1 < /dev/null &")
    # 7. a watchdog on the A100 (alerts are mirrored back below)
    rc, spd = a100(f"python3 - <<'EOF'\nimport json\nr=[json.loads(l) for l in open('{AR}/run/runs/a0u_up6/metrics.jsonl') if '\"train_loss\"' in l]\nprint(r[-1]['step'], round(r[-1]['train_loss'],3))\nEOF")
    s["phase"] = "done"; s["done_at"] = time.strftime("%F %T"); save_state(s)
    alert("A100_CUTOVER_DONE", f"A0 now trains on the A100 (speed ratio {float(c.get('ratio') or 0):.2f}x, micro-batch {mb}); A40 runs G0 then S1. A100 log tail: {spd.strip()[-80:]}")


def monitor(s):
    """after the cutover: alert (once per kind) when A0 on the A100 died or stalled, or the A100 GPU idles for 10 minutes"""
    sent = s.setdefault("monitor_sent", [])
    rc, out = a100(f"pgrep -f '[g]emma_mt.train_sft --config' | wc -l; stat -c %Y {AR}/run/runs/a0u_up6/metrics.jsonl 2>/dev/null; date +%s; nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n1")
    try:
        L = [l.strip() for l in out.strip().splitlines() if l.strip() and "Welcome" not in l and "Have fun" not in l and "AI agents" not in l]
        n, mt, now, util = int(L[0]), int(L[1]), int(L[2]), int(L[3])
    except Exception:
        return
    done = a100(f"test -f {AR}/run/runs/a0u_up6/summary.json && echo yes")[1].strip().endswith("yes")
    if done and "a0_finished" not in sent:
        sent.append("a0_finished"); alert("A100_A0_FINISHED", "A0-up6 finished training on the A100 (summary.json written)")
    if not done and n == 0 and "a0_dead" not in sent:
        sent.append("a0_dead"); alert("A100_A0_DEAD", "no training process on the A100 and no summary.json: A0 stopped; checkpoint every 20 steps is in run/runs/a0u_up6/last")
    if not done and n > 0 and now - mt > 900 and "a0_stall" not in sent:
        sent.append("a0_stall"); alert("A100_A0_STALLED", f"metrics not updated for {(now - mt) // 60} min")
    save_state(s)


def main():
    os.makedirs(W, exist_ok=True)
    s = load_state(); s["idle_since"] = None; log(f"worker started (dry={DRY}); phase {s['phase']}")   # an idle clock never survives a restart
    last_alert_mirror = 0
    while True:
        try:
            if s["phase"] in ("done", "halted"):
                if s["phase"] == "done":
                    monitor(s)
                time.sleep(120); s = load_state(); continue
            p = probe()
            if p is None:
                log("A100 unreachable"); s["idle_since"] = None; time.sleep(60); continue   # not seeing the GPU is not idle time
            if p["dir_exists"] and not p["marker"] and not s.get("claimed"):
                alert("A100_FOLDER_CONFLICT", f"{AR} exists on the A100 and is not ours (no .sabiyarn_owner marker): not touching it"); s["phase"] = "halted"; save_state(s); continue
            busy = p["util"] > 5 or p["mem_mib"] > 4096
            if busy:
                s["idle_since"] = None
            elif s["idle_since"] is None:
                s["idle_since"] = time.time()
            idle_min = 0 if s["idle_since"] is None else (time.time() - s["idle_since"]) / 60
            idle15 = idle_min >= IDLE_MIN
            em = emptied(p)
            log(f"A100 util {p['util']}% mem {p['mem_mib']} MiB | disk free {p['avail']/GB:.1f} GB, ours {p['mine']/GB:.1f} GB, free-for-us {(p['avail']+p['mine'])/GB:.1f} GB (need {NEED_GB}) | idle {idle_min:.0f} min | staged {s['done_items']}")
            # staging (small first), only with the disk reserve rule
            if not DRY:
                if not p["marker"]:
                    a100(f"mkdir -p {AR} && touch {AR}/.sabiyarn_owner"); s["claimed"] = True
            for item in ITEMS:
                if item[0] in s["done_items"]:
                    continue
                if stage_item(item, p, idle15):
                    s["done_items"].append(item[0]); save_state(s)
                break                                                          # one item per loop
            # the other agent has left the GPU (idle >= 15 min): the environment install (CPU + ~17 GB disk, no GPU) may start while it finishes clearing its disk, if 10 GB stay free for it
            if (not DRY) and idle15 and not s["cutover"].get("env_started") and p["avail"] / GB >= 27.0:
                install_envs(); s["cutover"]["env_started"] = True; save_state(s)
            all_staged = all(i[0] in s["done_items"] for i in ITEMS)
            if all_staged and idle15 and em and not DRY:
                log("conditions met: staged, idle >= 15 min, storage emptied -> cutover")
                s["phase"] = "cutover"; save_state(s)
                while s["phase"] == "cutover":
                    cutover(s, p)
                    if s["phase"] == "cutover":
                        time.sleep(60)
                        p = probe() or p
            elif DRY:
                log(f"[dry] all_staged={all_staged} idle15={idle15} emptied={em}")
                return
        except Exception as e:
            log(f"loop error: {type(e).__name__}: {e}")
        time.sleep(60 if not DRY else 1)


if __name__ == "__main__":
    main()
