import json, sys, re
W = sys.argv[1]
cands = {}
for lr in ("3e-6", "1e-5", "3e-5"):
    recs = [json.loads(l) for l in open(f"{W}/probe_logs/metrics_{lr}.jsonl")]
    main = {r["step"]: r["val_loss_macro"] for r in recs if "val_loss_macro" in r}
    tp = {r["step"]: r["val_val_tp_loss_macro"] for r in recs if "val_val_tp_loss_macro" in r}
    assert 300 in main and 300 in tp, (lr, sorted(main), sorted(tp))
    cands[lr] = {"val_macro": main[300], "val_tp_macro": tp[300], "history_val_macro": sorted(main.items()), "history_val_tp_macro": sorted(tp.items())}
    cands[lr]["score"] = cands[lr]["val_macro"] + cands[lr]["val_tp_macro"]
best = min(c["score"] for c in cands.values())
tied = [lr for lr, c in cands.items() if c["score"] <= best + 0.002]
winner = min(tied, key=float)
json.dump({"rule": "score = val macro + val_tp macro at step 300; ties within 0.002 of the best go to the smallest LR", "candidates": cands, "tied": tied, "winner": winner}, open(f"{W}/lr_choice.json", "w"), indent=1)
for lr, c in cands.items(): print(lr, round(c["val_macro"], 4), round(c["val_tp_macro"], 4), "score", round(c["score"], 4))
print("tied", tied, "WINNER", winner)
