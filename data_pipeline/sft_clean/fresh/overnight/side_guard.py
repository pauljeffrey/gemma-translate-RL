#!/usr/bin/env python3
"""Guard of a side GPU job that shares the A100 with the long-document RL (owner order 2026-11-04): kill the side job at once if the RL's step time rises by more than 25 % over its baseline
(median of the last 8 step intervals before the side job started) or the RL restarts / raises an alert (OOM).   side_guard.py NAME JOB_PID_FILE  -> exit 0 when the job ended by itself, 3 when it was killed.
The side job's process tree is the tree of the PID in JOB_PID_FILE (the `timeout` wrapper of the step)."""
import json, os, re, signal, statistics, subprocess, sys, time
name, pidfile = sys.argv[1:3]
R = "/workspace/clean_rerun"; M = f"{R}/runs_385m_long/final/metrics.jsonl"; LL = f"{R}/logs_long/launch.log"; LOG = "/workspace/science_sft/overnight/overnight.log"
def say(m): open(LOG, "a").write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + " " + m + "\n")
def steps():
    out = {}
    for l in open(M):
        try: r = json.loads(l)
        except Exception: continue
        if "reward" in r and "step" in r: out[r["step"]] = r["time"]
    return out
def intervals(s, after=None):
    ks = sorted(s); iv = []
    for a, b in zip(ks, ks[1:]):
        if b == a + 1 and (after is None or s[a] >= after): iv.append((b, s[b] - s[a]))
    return iv
def restarts(): return sum(1 for l in open(LL) if "trainer exited" in l)
def tree(p):
    out = [p]
    for c in subprocess.run(["pgrep", "-P", str(p)], capture_output=True, text=True).stdout.split(): out += tree(int(c))
    return out
t0 = time.time(); s0 = steps(); base_iv = [d for _, d in intervals(s0)][-8:]
if len(base_iv) < 3: say(f"side guard {name}: fewer than 3 baseline step intervals ({len(base_iv)}); using 145 s"); base = 145.0
else: base = statistics.median(base_iv)
n0 = restarts(); say(f"side guard {name}: baseline RL step time {base:.0f} s (median of last {len(base_iv)}), kill above {1.25 * base:.0f} s, restarts so far {n0}")
while True:
    time.sleep(30)
    try: pid = int(open(pidfile).read().strip())
    except Exception: pid = None
    if pid is None or not os.path.exists(f"/proc/{pid}"): sys.exit(0)
    why = None
    if restarts() > n0: why = "RL trainer restarted (possible OOM)"
    elif any(f.startswith("ALERT_final") for f in os.listdir(R)): why = "RL alert file"
    else:
        iv = [d for st, d in intervals(steps(), after=t0 + 60) if st % 25 not in (0, 1)]    # the interval around a 25-step evaluation is long by design
        if len(iv) >= 3 and statistics.median(iv[-3:]) > 1.25 * base: why = f"RL step time {statistics.median(iv[-3:]):.0f} s > 1.25 x {base:.0f} s"
    if why:
        say(f"side guard {name}: ABORT ({why}); killing the side job")
        for p in reversed(tree(pid)):
            try: os.kill(p, signal.SIGTERM)
            except Exception: pass
        open("/workspace/science_sft/overnight/SIDE_ABORTED", "w").write(why)
        sys.exit(3)
