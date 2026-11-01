#!/bin/bash
# Status board of everything running on the instance (read-only). Called by dash.sh over ssh every few seconds.
ROOT=${ROOT:-/workspace/work}; cd "$ROOT" || exit 1
line() { printf '%s\n' "------------------------------------------------------------------------------"; }
echo "SABIYARN CLEAN RERUN  |  $(date -u '+%F %T') UTC  |  $(hostname)"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader | sed 's/^/GPU: /'
echo "disk: $(df -h /workspace | tail -n1 | awk '{print $3" used, "$4" free"}')   load: $(cut -d' ' -f1-3 /proc/loadavg)"
line
echo "ALERTS (watchdog: crash / stall / idle GPU / failed arm or eval)"
{ [ -s ALERT.txt ] && tail -n 3 ALERT.txt | cut -c1-150; [ -s logs/ALERT_a100.txt ] && tail -n 3 logs/ALERT_a100.txt | cut -c1-150; } || echo "  none"
line
echo "TRAINING RUNS   (state: RUNNING / FROZEN / DONE / DEAD)"
python3 - <<'PY'
import glob, json, os, re, subprocess
def procs():
    out = subprocess.run(["ps", "-eo", "pid,stat,args"], capture_output=True, text=True).stdout
    d = {}
    for l in out.splitlines():
        m = re.search(r"^\s*(\d+)\s+(\S+)\s+.*gemma_mt\.train_sft --config \S*/(\w+)\.yaml", l)
        if m: d[m.group(3)] = m.group(2)
    return d
P = procs()
runs = sorted(glob.glob("run/runs/*/metrics.jsonl"))
if not runs: print("no training run has started yet")
for m in runs:
    name = m.split("/")[-2]
    if name.startswith("_"): continue
    rows = [json.loads(l) for l in open(m) if l.strip()]
    tr = [r for r in rows if "train_loss" in r and "step" in r]
    if not tr: print(f"{name}: no steps logged yet"); continue
    log = f"run/logs/sft_{name}.log"; total = None
    if os.path.exists(log):
        for l in open(log, errors="ignore"):
            g = re.search(r"(\d+) total steps", l)
            if g: total = int(g.group(1)); break
    st = P.get(name)
    state = "FROZEN" if st and st.startswith("T") else "RUNNING" if st else "DONE" if os.path.exists(f"run/runs/{name}/summary.json") else "DEAD"
    a, b = tr[-1], tr[-2] if len(tr) > 1 else tr[-1]
    sps = (a["time"] - b["time"]) / max(1, a["step"] - b["step"]) if a["step"] != b["step"] else float("nan")
    total = total or int(json.load(open(f"run/runs/{name}/schedule_manifest.json")).get("steps", 0) or 0) if os.path.exists(f"run/runs/{name}/schedule_manifest.json") else total
    if not total: print(f"{name}: step {a['step']}  loss {a['train_loss']:.3f}"); continue
    eta = (total - a["step"]) * sps / 3600 if total and sps == sps else float("nan")
    el = (a["time"] - tr[0]["time"]) / 3600
    eta_s = f"ETA {eta:.1f} h" if state == "RUNNING" else ""
    print(f"[{state}] {name}: step {a['step']}/{total} ({100*a['step']/total:.1f}%)  loss {a['train_loss']:.3f}  gnorm {a['grad_norm']:.1f}   {sps:.1f} s/step  {el:.1f} h elapsed  {eta_s}")
    ev = [r for r in rows if "val_loss" in r]
    if ev:
        e = ev[-1]; print(f"          last eval @ {e['step']}: val_loss {e['val_loss']:.4f}  macro {e.get('val_loss_macro', float('nan')):.4f}")
PY
line
echo "FLORES+ H_sel SCORES (chrF++, macro over languages, vLLM greedy)"
python3 - <<'PY'
import glob, json, os
rs = sorted(glob.glob("eval/*_hsel/results.json"))
for p in rs:
    try:
        r = json.load(open(p)); m = r["macro_all"]
        print(f"  {os.path.basename(os.path.dirname(p))[:-5]:12s} chrF++ {m['chrf']:.2f}  spBLEU {m['spbleu']:.2f}  ({m['chrf_n_languages']} languages)")
    except Exception as e:
        print("  ", p, "unreadable")
running = [l for l in os.popen("ps -eo args | grep '[e]val_translation' | head -n2").read().splitlines()]
if running: print("  evaluation running now:", running[0].split("--name")[1].split()[0] if "--name" in running[0] else "")
if not rs and not running: print("  none yet")
PY
line
echo "LIVE TRAINING LOGS (the runs that are running now)"
for pid in $(pgrep -f "[g]emma_mt.train_sft --config"); do
  st=$(ps -o stat= -p $pid | cut -c1); [ "$st" = "T" ] && continue
  arm=$(tr '\0' ' ' < /proc/$pid/cmdline | sed 's/.*clean_rerun_sft\/\([a-z0-9_]*\)\.yaml.*/\1/')
  echo "== $arm"; tr '\r' '\n' < run/logs/sft_$arm.log | grep -a "gemma_mt\]" | grep -av "Upload\|it/s" | tail -n 6 | cut -c1-140 | sed 's/^/   /'
  grep -a "pushed step\|push FAILED" run/logs/sft_$arm.log | tail -n 1 | cut -c1-120 | sed 's/^/   hub: /'
done
ev=$(ls -t logs/eval_*.log 2>/dev/null | head -n1); if [ -n "$ev" ] && ! grep -q "EVAL_EXIT" "$ev"; then echo "== evaluation ($(basename $ev))"; tr '\r' '\n' < "$ev" | grep -av "^$" | tail -n 3 | cut -c1-140 | sed 's/^/   /'; fi
line
echo "DATA CHAIN: last finished step $(ls -t state/*.done 2>/dev/null | head -n1 | xargs -r basename)  |  A0 schedule hash $(cut -c1-30 logs/a0_sha_check.txt 2>/dev/null | head -n1)"
