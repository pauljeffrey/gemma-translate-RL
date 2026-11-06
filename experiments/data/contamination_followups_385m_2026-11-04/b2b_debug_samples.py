import re, sys, importlib.util
spec = importlib.util.spec_from_file_location("b2", "b2_replay_stageA_prompts.py"); b2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(b2)
P1 = b2.build_pool(False); sch1, _ = b2.cfg_schedule(P1, with_ref=False)
ARC = b2.ARC
cur = None; bad = []; n = 0; txt_ok = 0
for ln in open(f"{ARC}/clean_rerun_misc/logs/rl_A_prev_1793691697.log").read().splitlines():
    m = re.match(r"\[gemma_mt\] rl step (\d+)/400", ln)
    if m: cur = int(m.group(1)); continue
    m = re.match(r"\[gemma_mt\]   sample \[(\w+)\] '(.*?)' ->", ln)
    if m and cur and cur <= 40:
        n += 1; isos = [P1["iso"][p] for p in sch1.indices(cur - 1, 64)]
        if m.group(1) not in isos: bad.append((cur, m.group(1), sorted(set(isos))[:6]))
print("parsed", n, "iso not in replayed batch:", bad)
