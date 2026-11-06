#!/bin/bash
[ -f .env ] && { set -a; . ./.env; set +a; }   # SSH_KEY, BOX_HOST, BOX_PORT, GPU_BOX_HOST, GPU_BOX_PORT (see .env.example)
# Local watch of the long-document RL (run on the laptop; stdout lines are the events). Heartbeat every POLL seconds with step, restarts, last exit codes, disk; ALERT when the launcher log shows
# a trainer restart whose exit code equals the previous one (2 restarts with the same rc = a crash loop, investigate at once), when the root disk has < 1.5 GB free, or when no new metric line for 25 min.
K=${SSH_KEY}; O="-i $K -o IdentitiesOnly=yes -o LogLevel=ERROR -o ConnectTimeout=25"; POLL=${POLL:-600}
R=/workspace/clean_rerun; prev_n=-1; prev_step=-1; stale=0
while true; do
  out=$(ssh $O -p ${GPU_BOX_PORT} root@${GPU_BOX_HOST} "python3 - <<PY
import json, re, os, subprocess, time
R = '$R'
L = [json.loads(l) for l in open(R + '/runs_385m_long/final/metrics.jsonl')]
tr = [r for r in L if 'reward' in r]
step = tr[-1]['step'] if tr else -1
age = int(time.time() - os.path.getmtime(R + '/runs_385m_long/final/metrics.jsonl'))
ex = [l for l in open(R + '/logs_long/launch.log') if 'trainer exited' in l]
rcs = [re.search(r'rc=(\\d+)', l).group(1) for l in ex]
free = subprocess.run(['df', '--output=avail', '-k', '/'], capture_output=True, text=True).stdout.split()[-1]
shm = subprocess.run(['df', '--output=avail', '-k', '/dev/shm'], capture_output=True, text=True).stdout.split()[-1]
print(step, len(ex), ','.join(rcs[-3:]), int(free) // 1024, int(shm) // 1024, age)
PY" 2>/dev/null | grep -v -E "Welcome|Have fun|AI agents")
  set -- $out
  if [ $# -lt 6 ]; then echo "$(date -u +%H:%M) ALERT watch could not read the run state (ssh/metrics): '$out'"; sleep $POLL; continue; fi
  step=$1; n=$2; rcs=$3; free=$4; shm=$5; age=$6
  echo "$(date -u +%H:%M) HEARTBEAT step=$step restarts=$n last_rcs=$rcs root_free_MB=$free shm_free_MB=$shm metrics_age_s=$age"
  [ "$prev_n" -lt 0 ] && prev_n=$n      # first poll: only restarts that happen from now on count
  if [ "$n" -ge 2 ] && [ "$n" -gt "$prev_n" ]; then
    a=$(echo $rcs | awk -F, '{print $NF}'); b=$(echo $rcs | awk -F, '{print $(NF-1)}')
    [ "$a" = "$b" ] && echo "$(date -u +%H:%M) ALERT crash loop: $n trainer restarts, the last two with the same exit code rc=$a - investigate now (launch.log, rl_final.log, df)"
  fi
  [ "$free" -lt 1500 ] && echo "$(date -u +%H:%M) ALERT root disk free $free MB (< 1500)"
  [ "$age" -gt 1500 ] && echo "$(date -u +%H:%M) ALERT no metrics line for $age s"
  prev_n=$n
  sleep $POLL
done
