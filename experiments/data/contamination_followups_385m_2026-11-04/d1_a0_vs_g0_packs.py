"""E-07(d): is the text of A0 (the 270M model the 385M was grown from) a subset of g0, the pack set CONT scanned?  Evidence available on the laptop:
 (1) build rule of g0:  data_pipeline/sft_clean/fresh/make_stage_schedules.py docstring: packed_g = [A0 old packs (retagged) | instruction packs | new translation packs], append-only (indices of packed_g0 unchanged);
 (2) per-language PACK COUNTS: A0's final run (a0u_up6, 138,873 packs, every pack drawn at least once = the full A0 pack set incl. the 173 Efik top-up packs) versus the per-language counts of the first
     138,873 entries of g0's pack_lang array (sched_g/A/pack_lang.npy, 445,221 packs).  Language ids are mapped to ISO by order (checked: the per-id totals equal the g1a schedule's per-ISO pack totals).
 Pack CONTENT identity (hash of the text) cannot be checked on the laptop: packed_g0 (object-storage bucket clean_rerun_2026-11-01/g0) is not here and the retagging changes the token ids."""
import json, collections, csv
ARC = "/path/to/repo/research_archive_2026-11-03"
import numpy as np
a0 = json.load(open(f"{ARC}/clean_sft_small/run/runs/a0u_up6/schedule_manifest.json")); A0 = {l: v["packs"] for l, v in a0["draws_per_language"].items()}
g1a = json.load(open(f"{ARC}/clean_sft_small/run/runs/g1a/schedule_manifest.json")); G = {l: v["packs"] for l, v in g1a["draws_per_language"].items()}; order = list(G)
pl = np.load(f"{ARC}/databox_backup/work/sched_g/A/pack_lang.npy"); ids = sorted(set(pl.tolist())); tot = collections.Counter(pl.tolist())
assert all(tot[i] == G[order[k]] for k, i in enumerate(ids)); idiso = {i: order[k] for k, i in enumerate(ids)}
rows = []
for n in (138700, 138873):
    c = collections.Counter(idiso[i] for i in pl[:n].tolist())
    diff = {l: (c.get(l, 0), A0.get(l, 0)) for l in set(c) | set(A0) if c.get(l, 0) != A0.get(l, 0)}
    rows.append(dict(prefix_packs=n, languages_in_prefix=len(c), languages_in_A0=len(A0), languages_with_different_count=len(diff), differences=json.dumps(diff)))
    print(n, len(diff), diff)
c = collections.Counter(idiso[i] for i in pl[:138873].tolist())
with open("d_a0_vs_g0_pack_counts_by_language.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["iso", "A0_packs", "g0_first_138873_packs", "g0_all_445221_packs"]); [w.writerow([l, A0.get(l, 0), c.get(l, 0), G.get(l, 0)]) for l in sorted(set(A0) | set(G))]
json.dump(dict(a0_n_packs=a0["n_packs"], a0_unique_packs_drawn=a0["unique_packs_drawn"], a0_schedule_sha256=a0["sha256"], g0_n_packs=int(len(pl)), g1a_schedule_sha256=g1a["sha256"], comparison=rows, a0_real_tokens_drawn=a0["real_tokens"]), open("d_a0_vs_g0_summary.json", "w"), indent=1)
