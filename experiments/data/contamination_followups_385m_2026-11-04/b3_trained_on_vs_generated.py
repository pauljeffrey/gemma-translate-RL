"""E-07(b) follow-up: of the flagged prompts that stage A drew (b_stageA_flagged_draw_list.csv), which ones fed the policy update?  dynamic_sampling keeps the first n_target (64)
informative groups in generation order and generates 71 (oversample 1.1); the order and per-group std of every step are in group_stats.jsonl (steps 41-400)."""
import json, csv, collections
ARC = "/path/to/repo/research_archive_2026-11-03"
gs = collections.defaultdict(list)
for l in open(f"{ARC}/runs_385m/A/group_stats.jsonl"):
    x = json.loads(l); gs[x["step"]].append(x)
MIN_STD = 0.01
out = []
for r in csv.DictReader(open("b_stageA_flagged_draw_list.csv")):
    step = int(r["step"]); gidx = int(r["global_index"]); L = gs[step]
    pos = [i for i, x in enumerate(L) if x["prompt"] == gidx][0]
    informative_before = sum(1 for x in L[:pos] if x["std"] > MIN_STD)
    me = L[pos]; kept = (me["std"] > MIN_STD) and informative_before < 64
    out.append(dict(iso=r["iso"], step=step, position_in_step=pos + 1, group_std=me["std"], informative_groups_before=informative_before, used_in_update=bool(kept)))
tot_kept = sum(o["used_in_update"] for o in out)
print(json.dumps(out, indent=1)); print("flagged drawn", len(out), "used in the update", tot_kept)
# overall: how many of the 25,560 drawn prompts of steps 41-400 were NOT used (surplus or flat), for context
nused = 0; nflat = 0
for step, L in gs.items():
    k = 0
    for x in L:
        if x["std"] > MIN_STD and k < 64: k += 1
    nused += k; nflat += sum(1 for x in L if x["std"] <= MIN_STD)
print("steps 41-400: groups drawn", sum(len(v) for v in gs.values()), "informative groups used (first 64 per step)", nused, "flat groups", nflat)
json.dump(dict(flagged=out, flagged_used_in_update=tot_kept, groups_used_41_400=nused, flat_groups_41_400=nflat), open("b_stageA_flagged_used_in_update.json", "w"), indent=1)
