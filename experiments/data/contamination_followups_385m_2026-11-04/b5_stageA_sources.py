"""E-07(b)/(e) side result: source_dataset composition of the stage-A prompt pool (steps 41-400 pool, 4,928,554 prompts) and of the prompts stage A actually drew
(steps 1-40 replayed, steps 41-400 from group_stats): settles 'is Wikivoyage in the stage-A news pool' and 'is LibreTexts in the general/news/ref pools'."""
import importlib.util, json, collections, csv
spec = importlib.util.spec_from_file_location("b2", "b2_replay_stageA_prompts.py"); b2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(b2)
P41 = b2.build_pool(True); P1 = b2.build_pool(False); sch1, _ = b2.cfg_schedule(P1, with_ref=False)
gs = [json.loads(l) for l in open(f"{b2.ARC}/runs_385m/A/group_stats.jsonl")]
pool = collections.Counter(zip(P41["dom"], P41["src"]))
drawn = collections.Counter()
for s in range(1, 41):
    for p in sch1.indices(s - 1, 64): drawn[(P1["dom"][p], P1["src"][p])] += 1
for x in gs: drawn[(P41["dom"][x["prompt"]], P41["src"][x["prompt"]])] += 1
with open("b_stageA_pool_and_drawn_by_source.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["domain", "source_dataset", "pool_rows_steps41_400", "prompts_drawn_steps1_400"])
    for k in sorted(set(pool) | set(drawn)): w.writerow([k[0], k[1], pool.get(k, 0), drawn.get(k, 0)])
for k in sorted(set(pool) | set(drawn)): print(k, pool.get(k, 0), drawn.get(k, 0))
